"""
src/train.py

Generic training driver for all 5 models and the dual-attention ablation arms.

Supports both evaluation protocols from the plan:
  --split_mode official  -> single train/test split, run N seeds
  --split_mode loso      -> subject-independent k-fold, run N seeds per fold
  --split_mode both      -> official, then loso (default)

Every run (one seed on one split/fold) appends one row to
results/tables/raw_results.csv (utils.append_result_row) and writes its test
predictions to results/preds/<run>.npz. A run whose key (model, variant,
dataset, split_mode, fold, seed) is already in the CSV is skipped, so an
interrupted job can simply be re-launched. Parallel jobs (src/run_parallel.py)
each write to their own --results_csv and pass the merged CSV via
--resume_from, so resume still sees every finished run. Latency is NOT measured here
(training machines differ) -- see src/profile_efficiency.py.

Early stopping monitors a SUBJECT-WISE validation set carved out of each
fold's training subjects (data_loader.iter_folds), never the test subjects.
By default the best epoch is picked on validation macro-F1 (--select_metric):
under subject shift, validation loss rises as soon as the model gets
confident, often after 1-3 epochs, while macro-F1 is still improving.

Examples:
    python -m src.train --config configs/ucihar_config.yaml --model bigru_baseline
    python -m src.train --config configs/ucihar_config.yaml --model dual_attention_bigru --dual_attn_variant all
    python -m src.train --config configs/pamap2_config.yaml --model cnn_bigru_reference --split_mode loso
    python -m src.train --config configs/ucihar_config.yaml --model bigru_baseline --smoke
"""

import argparse
import inspect
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data_loader import iter_folds, build_dataloaders, HARDataset
from src.models.bigru_baseline import BiGRUBaseline
from src.models.bilstm_attention_baseline import BiLSTMAttentionBaseline
from src.models.cnn_bigru_reference import CNNBiGRUReference
from src.models.deepconvlstm import DeepConvLSTM
from src.models.dual_attention_bigru import DualAttentionBiGRU
from src.preprocessing import NATIVE_CONFIGS
from src.utils import (
    set_seed, get_device, device_name, git_commit, compute_metrics,
    count_parameters, compute_flops, EarlyStopping, save_checkpoint,
    save_predictions, append_result_row, completed_run_keys, run_key, run_name,
    to_json, NO_VARIANT, PROTOCOL, RAW_RESULTS_CSV, PREDS_DIR, CHECKPOINT_DIR, RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bigru_baseline": BiGRUBaseline,
    "bilstm_attention_baseline": BiLSTMAttentionBaseline,
    "cnn_bigru_reference": CNNBiGRUReference,
    "deepconvlstm": DeepConvLSTM,
    "dual_attention_bigru": DualAttentionBiGRU,
}

# Ablation arms for the proposed model, selected via --dual_attn_variant.
# "full" is the proposed model (spectral-gated dual attention); the others
# isolate each component's contribution -- "plain" reproduces the already-
# closed literature gap (learned channel + temporal attention, no spectral
# prior), so the delta between "plain" and "full" is this project's headline
# architectural-novelty result.
DUAL_ATTN_VARIANTS = {
    "full":            dict(use_channel_attention=True,  use_temporal_attention=True,  use_spectral_gate=True),
    "plain":           dict(use_channel_attention=True,  use_temporal_attention=True,  use_spectral_gate=False),
    "channel_only":    dict(use_channel_attention=True,  use_temporal_attention=False, use_spectral_gate=True),
    "temporal_only":   dict(use_channel_attention=False, use_temporal_attention=True,  use_spectral_gate=True),
}

# architecture knobs that may come from the config / CLI; each model only
# receives the ones its constructor accepts
MODEL_KWARG_NAMES = ["hidden_size", "num_layers", "dropout", "attention_dim",
                     "channel_reduction", "low_band_hz"]


def build_model(model_name: str, input_size: int, num_classes: int,
                 dataset: str = None, dual_attn_variant: str = "full",
                 **model_kwargs) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Registered: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_name]
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in model_kwargs.items() if k in accepted and v is not None}
    if "low_band_hz" in kwargs:
        kwargs["low_band_hz"] = tuple(kwargs["low_band_hz"])
    if model_name == "dual_attention_bigru":
        if dataset is None:
            raise ValueError("dual_attention_bigru needs `dataset` (for the spectral gate's sampling rate)")
        return cls(input_size=input_size, num_classes=num_classes,
                   sampling_rate_hz=NATIVE_CONFIGS[dataset]["sampling_rate"],
                   **DUAL_ATTN_VARIANTS[dual_attn_variant], **kwargs)
    return cls(input_size=input_size, num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# Single train/eval run (one seed, one split or fold)
# ---------------------------------------------------------------------------
def _evaluate_val(model, loader, criterion, device):
    """(mean loss, macro-F1) on a validation loader."""
    model.eval()
    total, n = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += criterion(logits, yb).item() * len(yb)
            n += len(yb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    f1 = compute_metrics(np.concatenate(trues), np.concatenate(preds))["macro_f1"] if n else 0.0
    return total / max(n, 1), f1


def train_one_run(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, epochs: int, lr: float, patience: int,
                   val_fraction: float = 0.1, val_loader: DataLoader = None,
                   grad_clip: float = 1.0, lr_scheduler: str = "plateau",
                   select_metric: str = "val_macro_f1", return_details: bool = False):
    """
    Trains `model`, early-stopping on `select_metric` (val_macro_f1 or
    val_loss; the plateau LR scheduler follows the same metric), restores the
    best epoch, and evaluates on test_loader.

    val_loader should be a subject-wise validation set (train.py always
    passes one). If it is None (old notebook calls), a random window-level
    slice of the training set is used instead -- that leaks overlapping
    windows into validation and is kept only for backwards compatibility.

    Returns the metrics dict, or (metrics, details) with return_details=True,
    where details holds y_true/y_pred and training-curve information.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    mode = "max" if select_metric == "val_macro_f1" else "min"
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=3)
                 if lr_scheduler == "plateau" else None)

    if val_loader is None:
        full_train_ds = train_loader.dataset
        n_val = max(1, int(len(full_train_ds) * val_fraction))
        train_ds, val_ds = torch.utils.data.random_split(
            full_train_ds, [len(full_train_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(0),
        )
        train_loader = DataLoader(train_ds, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=train_loader.batch_size, shuffle=False)

    stopper = EarlyStopping(patience=patience, mode=mode)
    train_losses, val_losses, val_f1s = [], [], []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * len(yb)
            seen += len(yb)
        train_losses.append(running / max(seen, 1))

        val_loss, val_f1 = _evaluate_val(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        monitored = val_f1 if select_metric == "val_macro_f1" else val_loss
        if scheduler is not None:
            scheduler.step(monitored)

        stopper.step(monitored, model)
        # one line per epoch, so a job's log shows it is progressing (and how fast)
        print(f"    epoch {epoch+1:>2}/{epochs}  train_loss={train_losses[-1]:.4f}  val_loss={val_loss:.4f}  "
              f"val_f1={val_f1:.4f}  {time.perf_counter() - t_epoch:.1f}s")
        if stopper.should_stop:
            print(f"    early stop at epoch {epoch+1} (best epoch {stopper.best_epoch}, "
                  f"{select_metric}={stopper.best_value:.4f})")
            break

    train_time = time.perf_counter() - t0
    stopper.restore_best(model)

    # final evaluation on the real (subject-independent) test set
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
            all_true.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    metrics = compute_metrics(y_true, y_pred)
    if not return_details:
        return metrics
    details = dict(
        y_true=y_true, y_pred=y_pred,
        train_losses=np.array(train_losses), val_losses=np.array(val_losses), val_f1s=np.array(val_f1s),
        best_epoch=stopper.best_epoch, epochs_trained=len(train_losses),
        best_val_loss=val_losses[stopper.best_epoch - 1], best_val_f1=val_f1s[stopper.best_epoch - 1],
        train_time_s=train_time,
    )
    return metrics, details


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def model_kwargs_from(args) -> dict:
    return {k: getattr(args, k) for k in MODEL_KWARG_NAMES}


def run(args):
    device = get_device(args.device)
    dev_name = device_name(device)
    commit = git_commit()
    print(f"device: {device} ({dev_name}) | commit {commit}")

    variants = ([NO_VARIANT] if args.model != "dual_attention_bigru" else
                (list(DUAL_ATTN_VARIANTS) if "all" in args.dual_attn_variant else args.dual_attn_variant))
    split_modes = ["official", "loso"] if args.split_mode == "both" else [args.split_mode]
    hparams = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                   grad_clip=args.grad_clip, lr_scheduler=args.lr_scheduler,
                   select_metric=args.select_metric, val_fraction=args.val_fraction,
                   min_val_subjects=args.min_val_subjects, **model_kwargs_from(args))

    done = set()
    if not args.force:
        for csv_path in [args.results_csv, *args.resume_from]:
            done |= completed_run_keys(csv_path)
    flops_cache = {}
    n_run = n_skip = 0

    for split_mode in split_modes:
        for fold in iter_folds(args.dataset, split_mode, n_folds=args.n_folds,
                               val_fraction=args.val_fraction, min_val_subjects=args.min_val_subjects):
            fold_id = fold["fold_id"]
            if split_mode == "loso" and args.folds is not None and fold_id not in args.folds:
                continue
            input_shape = fold["X_train"].shape[1:]  # (T, C)
            num_classes = len(fold["classes"])

            for variant in variants:
                vflag = variant if variant != NO_VARIANT else "full"
                for seed in args.seeds:
                    row_key = dict(model=args.model, variant=variant, dataset=args.dataset,
                                   split_mode=split_mode, fold_id=fold_id, seed=seed, protocol=PROTOCOL)
                    if run_key(row_key) in done:
                        n_skip += 1
                        continue
                    print(f"[{args.model}/{variant}/{args.dataset}/{split_mode}] fold={fold_id} "
                          f"test_subjects={fold['held_out_subjects']} val_subjects={fold['val_subjects']} "
                          f"seed={seed}")

                    set_seed(seed)
                    model = build_model(args.model, input_size=input_shape[-1], num_classes=num_classes,
                                        dataset=args.dataset, dual_attn_variant=vflag,
                                        **model_kwargs_from(args))
                    if variant not in flops_cache:
                        flops_cache[variant] = compute_flops(model, input_shape)
                    train_loader, test_loader = build_dataloaders(
                        fold["X_train"], fold["y_train"], fold["X_test"], fold["y_test"],
                        batch_size=args.batch_size, seed=seed,
                    )
                    val_loader = DataLoader(HARDataset(fold["X_val"], fold["y_val"]),
                                            batch_size=args.batch_size, shuffle=False)
                    metrics, det = train_one_run(
                        model, train_loader, test_loader, device,
                        epochs=args.epochs, lr=args.lr, patience=args.patience,
                        val_loader=val_loader, grad_clip=args.grad_clip,
                        lr_scheduler=args.lr_scheduler, select_metric=args.select_metric,
                        return_details=True,
                    )
                    print(f"    -> acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                          f"(best epoch {det['best_epoch']}/{det['epochs_trained']}, "
                          f"{det['train_time_s']:.0f}s)")

                    name = run_name(args.model, variant, args.dataset, split_mode, fold_id, seed)
                    save_predictions(
                        Path(args.preds_dir) / f"{name}.npz",
                        y_true=det["y_true"], y_pred=det["y_pred"], classes=fold["classes"],
                        subject_test=fold["subject_test"],
                        train_losses=det["train_losses"], val_losses=det["val_losses"], val_f1s=det["val_f1s"],
                    )
                    if not args.no_checkpoints:
                        save_checkpoint(model, Path(args.checkpoint_dir) / args.model / f"{name}.pt",
                                        extra=dict(hparams=hparams, classes=fold["classes"]))

                    append_result_row({
                        **row_key, **metrics,
                        "n_train": len(fold["y_train"]), "n_val": len(fold["y_val"]),
                        "n_test": len(fold["y_test"]), "num_classes": num_classes,
                        "held_out_subjects": " ".join(map(str, fold["held_out_subjects"])),
                        "val_subjects": " ".join(map(str, fold["val_subjects"])),
                        "params": count_parameters(model), "flops": flops_cache[variant],
                        "best_epoch": det["best_epoch"], "epochs_trained": det["epochs_trained"],
                        "best_val_loss": det["best_val_loss"], "best_val_f1": det["best_val_f1"],
                        "train_time_s": det["train_time_s"],
                        "device": device.type, "device_name": dev_name, "git_commit": commit,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "hparams": to_json(hparams),
                    }, csv_path=args.results_csv)
                    n_run += 1
            if args.smoke:
                break  # smoke test: first fold of each split mode only

    print(f"done: {n_run} runs trained, {n_skip} already in {args.results_csv} (skipped)")


# ---------------------------------------------------------------------------
# CLI (+ YAML config: config values become defaults, CLI flags override them)
# ---------------------------------------------------------------------------
CONFIG_KEYS = {
    ("dataset", "name"): "dataset",
    ("evaluation", "seeds"): "seeds",
    ("evaluation", "loso_n_folds"): "n_folds",
    ("evaluation", "val_fraction"): "val_fraction",
    ("evaluation", "min_val_subjects"): "min_val_subjects",
    ("training", "select_metric"): "select_metric",
    ("training", "epochs"): "epochs",
    ("training", "batch_size"): "batch_size",
    ("training", "lr"): "lr",
    ("training", "patience"): "patience",
    ("training", "grad_clip"): "grad_clip",
    ("training", "lr_scheduler"): "lr_scheduler",
    ("model_defaults", "hidden_size"): "hidden_size",
    ("model_defaults", "num_layers"): "num_layers",
    ("model_defaults", "dropout"): "dropout",
    ("model_defaults", "attention_dim"): "attention_dim",
    ("dual_attention_bigru", "channel_reduction"): "channel_reduction",
    ("dual_attention_bigru", "low_band_hz"): "low_band_hz",
}


def config_to_defaults(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = {}
    for (section, key), arg in CONFIG_KEYS.items():
        if key in (cfg.get(section) or {}):
            out[arg] = cfg[section][key]
    return out


def build_parser():
    p = argparse.ArgumentParser(description="Train one model on one dataset (all folds x seeds).")
    p.add_argument("--config", help="YAML config (configs/<dataset>_config.yaml); CLI flags override it")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--dataset", choices=["ucihar", "wisdm", "pamap2"])
    p.add_argument("--split_mode", choices=["official", "loso", "both"], default="both")
    p.add_argument("--dual_attn_variant", nargs="+", default=["full"],
                   choices=list(DUAL_ATTN_VARIANTS.keys()) + ["all"],
                   help="ablation arm(s) for dual_attention_bigru ('all' = every arm); ignored by other models")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_folds", type=int, default=None, help="override LOSO_N_FOLDS default")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="only these LOSO fold ids (default: all); the official split is unaffected")
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="fraction of each fold's training SUBJECTS held out for early stopping")
    p.add_argument("--min_val_subjects", type=int, default=2,
                   help="at least this many validation subjects per fold")
    p.add_argument("--select_metric", choices=["val_macro_f1", "val_loss"], default="val_macro_f1",
                   help="validation metric for early stopping, best-epoch selection and the LR scheduler")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm; 0 disables")
    p.add_argument("--lr_scheduler", choices=["plateau", "none"], default="plateau")
    for k in ("hidden_size", "num_layers", "attention_dim", "channel_reduction"):
        p.add_argument(f"--{k}", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--low_band_hz", type=float, nargs=2, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--results_csv", default=str(RAW_RESULTS_CSV))
    p.add_argument("--resume_from", nargs="*", default=[],
                   help="extra results CSVs whose finished runs are skipped too (not written to)")
    p.add_argument("--preds_dir", default=str(PREDS_DIR))
    p.add_argument("--checkpoint_dir", default=str(CHECKPOINT_DIR))
    p.add_argument("--no_checkpoints", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if the run is already in the CSV")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, first seed, first fold per split; writes to results/smoke/ only")
    return p


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    p = build_parser()
    if known.config:
        p.set_defaults(**config_to_defaults(known.config))
    args = p.parse_args(argv)
    if args.dataset is None:
        p.error("--dataset is required (or pass --config with dataset.name)")
    if args.smoke:
        args.epochs, args.seeds = 1, args.seeds[:1]
        args.results_csv = str(RESULTS_DIR / "smoke" / "raw_results.csv")
        args.preds_dir = str(RESULTS_DIR / "smoke" / "preds")
        args.no_checkpoints, args.force = True, True
    return args


if __name__ == "__main__":
    run(parse_args())
