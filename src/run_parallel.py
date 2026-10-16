"""
src/run_parallel.py

Runs the whole training grid as many small, independent jobs, N at a time,
then merges their results into results/tables/raw_results.csv.

One job = one dataset x model x ablation variant x split x LOSO fold (all seeds),
i.e. one `python -m src.train ... --folds K` process. That's 264 jobs for the
full grid (1320 runs), which keeps N workers evenly busy until the end.

Each job writes only its own files, so parallel jobs never touch the same file:
    results/tables/parts/<dataset>/<job>.csv     its result rows
    results/logs/train/<dataset>/<job>.log       its full training log
(predictions and checkpoints already have one file per run.) When all jobs
finish, or on Ctrl-C, the parts are merged into raw_results.csv in a fixed
order (src/merge_results.py).

Resume: finished runs are skipped. Jobs whose runs are all in
raw_results.csv are not started at all, and each started job also skips
runs already in its part file or in raw_results.csv. A failed job is
retried once; if it still fails, the rest keep running and the exit code
is 1 at the end, so re-running finishes the leftovers.

--jobs auto picks N from the hardware: on NVIDIA GPU(s), up to 12, capped by
CPU cores, GPU memory (~1.5 GB per job) and RAM (~2.5 GB per job); on CPU
only, about 1 job per 4 cores. Each job's CPU threads are limited to
cores / N, so the jobs don't compete for cores.

    python -m src.run_parallel                                  # all datasets, --jobs auto
    python -m src.run_parallel --jobs 10 --datasets pamap2
    python -m src.run_parallel --dry_run                        # show the plan, run nothing
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.merge_results import merge, PARTS_DIR
from src.utils import PROJECT_ROOT, RAW_RESULTS_CSV, RESULTS_DIR, completed_run_keys, run_key, NO_VARIANT, PROTOCOL

ALL_MODELS = ["bigru_baseline", "bilstm_attention_baseline", "cnn_bigru_reference", "deepconvlstm",
              "dual_attention_bigru"]
DUAL_VARIANTS = ["full", "plain", "channel_only", "temporal_only"]
LOGS_DIR = RESULTS_DIR / "logs" / "train"

# rough relative cost of one run, used only to start the longest jobs first
# (from the M1 smoke timings: seconds per epoch by dataset/model)
DATASET_COST = {"ucihar": 1.0, "wisdm": 1.7, "pamap2": 2.2}
MODEL_COST = {"bilstm_attention_baseline": 1.3, "cnn_bigru_reference": 0.8, "deepconvlstm": 1.5}

MAX_AUTO_JOBS = 12
GPU_GB_PER_JOB = 1.5
RAM_GB_PER_GPU_JOB = 2.5
RAM_GB_PER_CPU_JOB = 1.5


# ---------------------------------------------------------------------------
# Job grid
# ---------------------------------------------------------------------------
def load_config(dataset):
    path = PROJECT_ROOT / "configs" / f"{dataset}_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ev = cfg.get("evaluation") or {}
    return path, ev.get("seeds", [0, 1, 2, 3, 4]), ev["loso_n_folds"]


def build_jobs(datasets, models):
    jobs = []
    for ds in datasets:
        cfg_path, seeds, n_folds = load_config(ds)
        for model in models:
            variants = DUAL_VARIANTS if model == "dual_attention_bigru" else [NO_VARIANT]
            for variant in variants:
                tag = model if variant == NO_VARIANT else f"{model}-{variant}"
                for split, fold in [("official", None)] + [("loso", k) for k in range(n_folds)]:
                    name = f"{tag}__{split}" + (f"_fold{fold}" if fold is not None else "")
                    keys = {run_key(dict(model=model, variant=variant, dataset=ds, split_mode=split,
                                         fold_id=fold, seed=s, protocol=PROTOCOL)) for s in seeds}
                    cmd = [sys.executable, "-m", "src.train", "--config", str(cfg_path),
                           "--model", model, "--split_mode", split,
                           "--results_csv", str(PARTS_DIR / ds / f"{name}.csv"),
                           "--resume_from", str(RAW_RESULTS_CSV)]
                    if variant != NO_VARIANT:
                        cmd += ["--dual_attn_variant", variant]
                    if fold is not None:
                        cmd += ["--folds", str(fold)]
                    jobs.append(dict(
                        name=f"{ds}/{name}", cmd=cmd, keys=keys, attempts=0,
                        log=LOGS_DIR / ds / f"{name}.log",
                        cost=len(seeds) * DATASET_COST.get(ds, 1.0) * MODEL_COST.get(model, 1.0),
                    ))
    return jobs


# ---------------------------------------------------------------------------
# Hardware -> number of jobs
# ---------------------------------------------------------------------------
def gpu_memory_gb():
    """Total memory of each NVIDIA GPU in GB ([] if none / no nvidia-smi)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return [float(x) / 1024 for x in out.split()]
    except Exception:
        return []


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024**3
    except Exception:
        return None


def auto_jobs(device, gpus, cpus, ram):
    """Returns (n_jobs, explanation)."""
    use_gpu = bool(gpus) and device in ("auto", "cuda")
    if use_gpu:
        caps = {"max": MAX_AUTO_JOBS, "cpu cores": max(1, cpus - 2),
                "GPU memory": max(1, int(sum(gpus) / GPU_GB_PER_JOB))}
        if ram:
            caps["RAM"] = max(1, int((ram - 4) / RAM_GB_PER_GPU_JOB))
    else:
        caps = {"cpu cores": max(1, cpus // 4)}
        if ram:
            caps["RAM"] = max(1, int((ram - 3) / RAM_GB_PER_CPU_JOB))
    n = min(caps.values())
    limit = min(caps, key=caps.get)
    return n, ", ".join(f"{k}->{v}" for k, v in caps.items()) + f" (limited by {limit})"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def fmt_time(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def start(job, args, threads, gpu_index):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    job["attempts"] += 1
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    log = open(job["log"], "a", encoding="utf-8")
    log.write(f"\n===== attempt {job['attempts']} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
              f"$ {' '.join(job['cmd'] + ['--device', args.device])}\n")
    log.flush()
    job["proc"] = subprocess.Popen(job["cmd"] + ["--device", args.device], cwd=PROJECT_ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    job["log_handle"], job["t0"] = log, time.time()


def run_pool(jobs, args, n_jobs, threads, n_gpus):
    queue = sorted(jobs, key=lambda j: -j["cost"])  # longest first
    total_cost = sum(j["cost"] for j in jobs)
    running, failed = [], []
    done_cost, n_done, t_start = 0.0, 0, time.time()
    slot = 0

    try:
        while queue or running:
            while queue and len(running) < n_jobs:
                job = queue.pop(0)
                start(job, args, threads, (slot % n_gpus) if n_gpus > 1 else None)
                slot += 1
                running.append(job)
                print(f"  start  {job['name']}" + (f" (retry {job['attempts'] - 1})" if job["attempts"] > 1 else ""))
            time.sleep(1)
            for job in [j for j in running if j["proc"].poll() is not None]:
                running.remove(job)
                job["log_handle"].close()
                rc, took = job["proc"].returncode, time.time() - job["t0"]
                if rc == 0:
                    n_done += 1
                    done_cost += job["cost"]
                    elapsed = time.time() - t_start
                    eta = elapsed / done_cost * (total_cost - done_cost) if done_cost else 0
                    print(f"  done   {job['name']} in {fmt_time(took)}  "
                          f"[{n_done}/{len(jobs)} jobs, {len(running)} running, elapsed {fmt_time(elapsed)}, "
                          f"eta ~{fmt_time(eta)}]")
                elif job["attempts"] <= args.max_retries:
                    print(f"  FAILED {job['name']} (exit {rc}) -> retrying; log: {job['log']}")
                    queue.insert(0, job)
                else:
                    failed.append(job)
                    print(f"  FAILED {job['name']} (exit {rc}), giving up; log: {job['log']}")
    except KeyboardInterrupt:
        print("\ninterrupted: stopping running jobs (finished runs are kept; re-run to resume)")
        for job in running:
            job["proc"].terminate()
        for job in running:
            try:
                job["proc"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                job["proc"].kill()
            job["log_handle"].close()
        raise
    return failed


def main():
    p = argparse.ArgumentParser(description="Train the full grid as parallel jobs, then merge results.")
    p.add_argument("--datasets", nargs="+", default=["ucihar", "wisdm", "pamap2"])
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--jobs", default="auto", help="number of concurrent jobs, or 'auto'")
    p.add_argument("--threads", type=int, default=None, help="CPU threads per job (default: cores / jobs)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--dry_run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    # fold in anything already finished (earlier runs, interrupted parts) first
    merge(dataset_order=args.datasets)
    done = completed_run_keys(RAW_RESULTS_CSV)
    jobs = build_jobs(args.datasets, args.models)
    n_runs = sum(len(j["keys"]) for j in jobs)
    todo = [j for j in jobs if not j["keys"] <= done]
    n_left = sum(len(j["keys"] - done) for j in todo)

    cpus, gpus, ram = os.cpu_count() or 1, gpu_memory_gb(), ram_gb()
    hw = (f"{cpus} CPU threads, RAM {f'{ram:.0f} GB' if ram else 'unknown'}, "
          + (f"{len(gpus)} GPU(s) [{', '.join(f'{g:.0f} GB' for g in gpus)}]" if gpus else "no NVIDIA GPU"))
    if args.jobs == "auto":
        n_jobs, why = auto_jobs(args.device, gpus, cpus, ram)
    else:
        n_jobs, why = max(1, int(args.jobs)), "set by --jobs / JOBS"
    n_jobs = min(n_jobs, max(1, len(todo)))
    threads = args.threads or max(1, cpus // n_jobs)
    n_gpus = len(gpus) if args.device in ("auto", "cuda") else 0

    print(f"hardware: {hw}")
    print(f"grid: {len(jobs)} jobs / {n_runs} runs; already done: {n_runs - n_left} runs; "
          f"to do: {len(todo)} jobs / {n_left} runs")
    print(f"parallel: {n_jobs} jobs x {threads} CPU thread(s) each ({why})")
    print(f"per-job logs: {LOGS_DIR}/<dataset>/<job>.log | per-job results: {PARTS_DIR}/<dataset>/<job>.csv")
    if args.dry_run:
        for j in sorted(todo, key=lambda j: -j["cost"]):
            print("  " + j["name"])
        return 0
    if not todo:
        print("nothing to train")
        return 0

    try:
        failed = run_pool(todo, args, n_jobs, threads, n_gpus)
    except KeyboardInterrupt:
        merge(dataset_order=args.datasets)
        return 130

    merge(dataset_order=args.datasets)
    missing = {k for j in jobs for k in j["keys"]} - completed_run_keys(RAW_RESULTS_CSV)
    if failed or missing:
        print(f"\n{len(failed)} job(s) failed, {len(missing)} run(s) missing from {RAW_RESULTS_CSV}:")
        for j in failed:
            print(f"  {j['name']}  ->  {j['log']}")
        print("fix the cause (see the logs above), then re-run: finished runs are skipped")
        return 1
    print(f"\nall {n_runs} runs finished -> {RAW_RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
