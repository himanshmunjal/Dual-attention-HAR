"""
src/profile_efficiency.py

Efficiency accounting (params, FLOPs, inference latency) for every model /
ablation arm on every dataset, measured in ONE controlled pass on ONE
machine -- separately from training, because training runs happen on
whatever hardware is free and their timings are not comparable.

Latency at batch size 1 and 32 is timed in `--rounds` interleaved rounds of
`--repeats` forward passes each (after `--warmup` passes per round); the CSV
holds the median and the 25th/75th percentiles over all timed passes. CPU latency uses a fixed thread count (default 1, a
reasonable proxy for an edge/wearable-class CPU and much more stable than
"all cores"). Report the device names written into the CSV in the paper.

Rows are merged into results/tables/efficiency.csv keyed by
(model, variant, dataset, device), so CPU numbers from one machine and GPU
numbers from another can live in the same file.

Usage (after src/preprocessing.py):
    python -m src.profile_efficiency                         # CPU (+ CUDA if present)
    python -m src.profile_efficiency --devices cpu --cpu_threads 1
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.preprocessing import PROCESSED_DIR
from src.train import MODEL_REGISTRY, DUAL_ATTN_VARIANTS, build_model, config_to_defaults, MODEL_KWARG_NAMES
from src.utils import (count_parameters, compute_flops, measure_latency, device_name,
                       NO_VARIANT, RESULTS_DIR, PROJECT_ROOT)

EFFICIENCY_CSV = RESULTS_DIR / "tables" / "efficiency.csv"
KEY = ["model", "variant", "dataset", "device"]


def dataset_shape(dataset: str):
    meta_path = PROCESSED_DIR / dataset / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (meta["window_samples"], len(meta["channels"])), len(meta["classes"])
    import numpy as np
    X = np.load(PROCESSED_DIR / dataset / "X.npy", mmap_mode="r")
    y = np.load(PROCESSED_DIR / dataset / "y.npy")
    return tuple(X.shape[1:]), len(np.unique(y))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["ucihar", "wisdm", "pamap2"])
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    p.add_argument("--devices", nargs="+", default=None,
                   help="default: cpu, plus cuda if available (mps only if listed)")
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 32])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--out", default=str(EFFICIENCY_CSV))
    args = p.parse_args()

    devices = args.devices or (["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    torch.set_num_threads(args.cpu_threads)
    rows = []

    for ds in args.datasets:
        input_shape, num_classes = dataset_shape(ds)
        cfg_path = PROJECT_ROOT / "configs" / f"{ds}_config.yaml"
        cfg = config_to_defaults(cfg_path) if cfg_path.exists() else {}
        model_kwargs = {k: cfg.get(k) for k in MODEL_KWARG_NAMES}

        for model_name in args.models:
            variants = list(DUAL_ATTN_VARIANTS) if model_name == "dual_attention_bigru" else [NO_VARIANT]
            for variant in variants:
                model = build_model(model_name, input_size=input_shape[-1], num_classes=num_classes,
                                    dataset=ds, dual_attn_variant=variant if variant != NO_VARIANT else "full",
                                    **model_kwargs)
                params = count_parameters(model)
                flops = compute_flops(model, input_shape)
                for dev in devices:
                    device = torch.device(dev)
                    lat = measure_latency(model, input_shape, device, batch_sizes=args.batch_sizes,
                                          n_warmup=args.warmup, n_repeats=args.repeats,
                                          n_rounds=args.rounds)
                    row = dict(model=model_name, variant=variant, dataset=ds, device=dev,
                               device_name=device_name(device),
                               cpu_threads=args.cpu_threads if dev == "cpu" else None,
                               input_T=input_shape[0], input_C=input_shape[1], num_classes=num_classes,
                               params=params, flops=flops, **lat,
                               torch_version=torch.__version__, host=platform.node(),
                               timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                    if "latency_ms_bs32" in lat:
                        row["latency_ms_per_sample_bs32"] = lat["latency_ms_bs32"] / 32
                    rows.append(row)
                    print(f"{ds:7s} {model_name:26s} {variant:14s} {dev:4s} params={params:>7} "
                          f"flops={(flops or 0)/1e6:7.2f}M " +
                          " ".join(f"{b}={lat[f'latency_ms_bs{b}']:.3f}ms [{lat[f'latency_ms_bs{b}_p25']:.3f}, "
                                   f"{lat[f'latency_ms_bs{b}_p75']:.3f}]" for b in args.batch_sizes))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        old["variant"] = old["variant"].fillna(NO_VARIANT)
        merged = pd.concat([old, new]).drop_duplicates(subset=KEY, keep="last")
    else:
        merged = new
    merged.sort_values(KEY).to_csv(out, index=False)
    print(f"wrote {len(new)} rows -> {out}")


if __name__ == "__main__":
    main()
