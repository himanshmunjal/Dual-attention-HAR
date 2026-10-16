"""
src/utils.py

Shared utilities used by train.py, profile_efficiency.py, evaluate.py and
stats_test.py:
  - reproducibility (seeding) and device selection
  - label encoding (datasets don't share a contiguous 0..N-1 label space)
  - accuracy / macro-F1
  - efficiency accounting: param count, FLOPs, inference latency
  - checkpointing and per-run prediction files
  - append-only results logging to a single CSV, one row per run
    (model x variant x dataset x split_mode x fold x seed), resume-safe
"""

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

try:
    import fcntl  # POSIX; used to serialise CSV appends from parallel jobs
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt  # Windows equivalent of the fcntl lock above
except ImportError:
    msvcrt = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_CSV = RESULTS_DIR / "tables" / "raw_results.csv"
PREDS_DIR = RESULTS_DIR / "preds"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Evaluation-protocol version, logged with every run and part of its key.
# Bump it whenever folds, validation or model selection change, so results of
# an older protocol are neither resumed from nor mixed into the tables.
#   (none) 2026-10-15 run: 6/6/8 LOSO folds, 3 seeds, early stopping on val loss
#   v2     10/12/8 LOSO folds, 5 seeds, >=2 val subjects (20%), selection on val macro-F1
PROTOCOL = "v2"

# Identity of one run. A run whose key already exists in raw_results.csv is
# skipped by train.py (resume after a crash without duplicating rows).
RUN_KEY = ["model", "variant", "dataset", "split_mode", "fold_id", "seed", "protocol"]

# Fixed column order for raw_results.csv, so rows written by different
# jobs/machines always line up.
RESULT_COLUMNS = RUN_KEY + [
    "accuracy", "macro_f1", "weighted_f1", "n_train", "n_val", "n_test", "num_classes",
    "held_out_subjects", "val_subjects",
    "params", "flops",
    "best_epoch", "epochs_trained", "best_val_loss", "best_val_f1", "train_time_s",
    "device", "device_name", "git_commit", "timestamp", "hparams",
]

NO_VARIANT = "-"  # variant value for models without ablation arms


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN -- slightly slower, but seed-to-seed variance should
    # reflect init/data order, not nondeterministic kernel selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """auto -> CUDA if available, else CPU. MPS (Apple GPU) is only used when
    asked for explicitly: small GRUs are often no faster on MPS than on CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({_cpu_model()})"
    return _cpu_model()


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() is just 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
def encode_labels(y_train: np.ndarray, y_test: np.ndarray):
    """
    Raw label spaces differ across datasets: UCI-HAR is 1-indexed (1..6),
    WISDM is 0-indexed via a hand-written map, PAMAP2 activityIDs are
    non-contiguous device codes (e.g. 1,2,3,4,12,13,16,17,...).
    nn.CrossEntropyLoss needs contiguous 0..num_classes-1 targets, so remap
    using classes observed across BOTH train and test.

    Kept for the notebooks. train.py uses data_loader.iter_folds(), which
    encodes with one dataset-wide class list instead, so a class index means
    the same activity in every fold.
    """
    classes = np.unique(np.concatenate([y_train, y_test]))
    mapping = {c: i for i, c in enumerate(classes)}
    y_train_enc = np.array([mapping[v] for v in y_train], dtype=np.int64)
    y_test_enc = np.array([mapping[v] for v in y_test], dtype=np.int64)
    return y_train_enc, y_test_enc, mapping


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


# ---------------------------------------------------------------------------
# Efficiency accounting
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: torch.nn.Module, input_shape: tuple,
                  device: torch.device = torch.device("cpu")) -> Optional[float]:
    """
    FLOPs via `thop`. Caveat: thop's RNN/GRU FLOP counting is known to be
    approximate (it does not fully model gate-level matrix multiplies for
    all RNN variants) and it does not count torch.fft ops or elementwise
    attention arithmetic -- treat this as a consistent, comparable-across-
    models estimate rather than a ground-truth FLOP count. Report this caveat
    in the paper's efficiency section. Returns None if thop fails.
    """
    try:
        import copy
        from thop import profile
        # thop.profile registers `total_ops`/`total_params` buffers directly
        # onto the model it's given, which then breaks state_dict() for
        # checkpointing -- profile a throwaway deep copy instead.
        probe = copy.deepcopy(model).to(device).eval()
        dummy = torch.randn(1, *input_shape).to(device)
        with torch.no_grad():
            macs, _ = profile(probe, inputs=(dummy,), verbose=False)
        return float(macs) * 2  # MACs -> FLOPs (1 MAC ~= 2 FLOPs)
    except Exception as e:
        print(f"[compute_flops] thop failed ({e}); returning None")
        return None


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency(model: torch.nn.Module, input_shape: tuple, device: torch.device,
                     batch_sizes=(1, 32), n_warmup: int = 20, n_repeats: int = 100) -> dict:
    """
    Median inference latency (ms per batch) per batch size, on the given
    device. Median rather than mean so one OS hiccup doesn't skew a row.
    Weights don't affect latency, so run once per architecture -- in
    profile_efficiency.py, on one fixed machine -- not per training run.
    """
    model = model.to(device).eval()
    results = {}
    with torch.no_grad():
        for bs in batch_sizes:
            dummy = torch.randn(bs, *input_shape).to(device)
            for _ in range(n_warmup):
                model(dummy)
            _sync(device)
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                model(dummy)
                _sync(device)
                times.append(time.perf_counter() - t0)
            results[f"latency_ms_bs{bs}"] = float(np.median(times) * 1000)
    return results


def profile_model(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> dict:
    """Convenience wrapper bundling params + FLOPs + latency into one dict."""
    stats = dict(params=count_parameters(model))
    stats["flops"] = compute_flops(model, input_shape, device)
    stats.update(measure_latency(model, input_shape, device))
    return stats


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stops training when the monitored value hasn't improved for `patience`
    epochs (mode="min": lower is better, e.g. val loss; mode="max": higher is
    better, e.g. val macro-F1). Tracks best state_dict in memory."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.sign = 1.0 if mode == "min" else -1.0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.counter = 0
        self.best_state = None
        self.should_stop = False
        self._epoch = 0

    def step(self, value: float, model: torch.nn.Module):
        self._epoch += 1
        if self.sign * value < self.sign * self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = self._epoch
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Checkpointing / predictions
# ---------------------------------------------------------------------------
def save_checkpoint(model: torch.nn.Module, path: Path, extra: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: Path, map_location=None):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def run_name(model: str, variant: str, dataset: str, split_mode: str,
             fold_id, seed: int) -> str:
    """File-name stem shared by a run's checkpoint and prediction file."""
    parts = [model] + ([variant] if variant != NO_VARIANT else []) + [dataset, split_mode]
    if fold_id is not None and str(fold_id) != "":
        parts.append(f"fold{fold_id}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def save_predictions(path: Path, **arrays):
    """Per-run test predictions (y_true/y_pred/classes/subjects + loss curves)
    so confusion matrices and per-subject analyses never need retraining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Results logging (single append-only CSV, one row per run)
# ---------------------------------------------------------------------------
def _key_str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def run_key(row: dict) -> tuple:
    return tuple(_key_str(row.get(k)) for k in RUN_KEY)


def completed_run_keys(csv_path: Path = RAW_RESULTS_CSV) -> set:
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # rows logged before `protocol` existed get an empty protocol, so they never count as done
    return {tuple(r) for r in df.reindex(columns=RUN_KEY, fill_value="").itertuples(index=False)}


def load_results(csv_path: Path = RAW_RESULTS_CSV, protocol: str = PROTOCOL) -> pd.DataFrame:
    """Read raw_results.csv, keeping only rows of `protocol` and the latest row per run key."""
    df = pd.read_csv(csv_path, dtype={"protocol": str})
    df["variant"] = df["variant"].fillna(NO_VARIANT)
    n_all = len(df)
    df = df[df["protocol"] == protocol] if "protocol" in df else df.iloc[:0]
    if len(df) < n_all:
        print(f"load_results: ignoring {n_all - len(df)} rows from another protocol than '{protocol}' "
              f"(archive them, see results/archive/)")
    return df.drop_duplicates(subset=RUN_KEY, keep="last").reset_index(drop=True)


def append_result_row(row: dict, csv_path: Path = RAW_RESULTS_CSV):
    """
    Appends one row to results/tables/raw_results.csv, creating it with a
    header on first write. Columns follow RESULT_COLUMNS (extra keys are
    appended at the end; missing ones are left empty). A file lock makes
    this safe when several train.py processes run in parallel (POSIX and Windows).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("variant", NO_VARIANT)
    cols = RESULT_COLUMNS + [k for k in row if k not in RESULT_COLUMNS]
    df_row = pd.DataFrame([row]).reindex(columns=cols)

    with open(str(csv_path) + ".lock", "w") as lock:
        _lock_file(lock)
        try:
            new_file = not csv_path.exists() or csv_path.stat().st_size == 0
            if not new_file:
                header = pd.read_csv(csv_path, nrows=0).columns.tolist()
                df_row = df_row.reindex(columns=header)
            df_row.to_csv(csv_path, mode="w" if new_file else "a", header=new_file,
                          index=False, lineterminator="\n", encoding="utf-8")
        finally:
            _unlock_file(lock)


def _lock_file(f):
    """Exclusive, blocking lock on an open file (fcntl on POSIX, msvcrt on Windows)."""
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_EX)
    elif msvcrt is not None:
        f.seek(0)
        while True:  # LK_LOCK gives up after ~10s, so poll the non-blocking form instead
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)


def _unlock_file(f):
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_UN)
    elif msvcrt is not None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def to_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
