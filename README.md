# Dual-Attention BiGRU for Human Activity Recognition

A CNN-free, pure recurrent + attention architecture for HAR from wearable/inertial sensor data, evaluated with an emphasis on **efficiency accounting** and **evaluation rigor** — two dimensions largely missing from recent dual-attention HAR literature.

## Novelty

Recent HAR papers (Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer) report accuracy-only comparisons on single train/test splits. No reviewed CNN-free BiGRU + dual-attention model reports parameter/FLOPs/latency accounting alongside subject-independent, multi-seed, statistically validated evaluation across UCI-HAR, WISDM, and PAMAP2. This project fills that combined gap.

See `docs/research_gap_notes.md` for the full gap analysis and `report/literature_review.md` for the complete literature review.

## Models compared (4 total)

| # | Model | Role |
|---|---|---|
| 1 | Plain BiGRU | Baseline 1 |
| 2 | BiLSTM + single attention | Baseline 2 (reproduces prior literature) |
| 3 | CNN + BiGRU | Reference "heavy" baseline for the efficiency claim |
| 4 | Dual-Attention BiGRU (channel + temporal) | Proposed model |

## Datasets

UCI-HAR, WISDM, PAMAP2 — evaluated independently in native mode (own sampling rate/window/channels); optional harmonized mode for a supplementary cross-dataset transfer experiment (see `docs/research_gap_notes.md` §3).

## Evaluation

- Official train/test split (comparability with literature) **and** subject-independent (LOSO) k-fold (headline result).
- 5 seeds per fold, mean ± std reporting.
- Params, FLOPs, and inference latency logged for every model.
- Wilcoxon signed-rank / paired t-test significance testing vs. each baseline.

## Running the full pipeline

### Quick start (fresh machine)

**Prerequisites**
- Git. On Windows install [Git for Windows](https://git-scm.com/download/win), which includes **Git Bash**.
- Python **3.10 or newer** from python.org. On Windows, tick *"Add python.exe to PATH"* or keep the `py` launcher.
  The Microsoft Store `python` stub doesn't count, and the script skips it.
- NVIDIA GPU (optional): a recent NVIDIA driver. `nvidia-smi` must work in a terminal. You don't need the CUDA toolkit,
  because the PyTorch wheel bundles its own CUDA runtime.
- About 1 GB of free disk (datasets are about 320 MB raw and 60 MB processed, plus checkpoints and predictions).

**Run** (macOS/Linux terminal, or **Git Bash** on Windows, not PowerShell or cmd):

```bash
git clone https://github.com/himanshmunjal/Dual-attention-HAR.git
cd Dual-attention-HAR
DEVICE=cuda bash scripts/run_all.sh      # on a machine without an NVIDIA GPU: bash scripts/run_all.sh
```

That single command does everything:
`venv -> download -> preprocess -> smoke -> train -> profile -> stats -> evaluate`.

- **Environment.** On the first run the script creates `.venv/` and installs `requirements.txt`, and reinstalls
  only when that file changes. If `nvidia-smi` finds a GPU, it installs the CUDA build of PyTorch first. Don't
  install anything by hand or activate a venv yourself.
- **Check the GPU.** Near the top of the output (also in `results/logs/setup.log`) you should see
  `cuda available: True (NVIDIA GeForce ...)`. If a GPU is present but CUDA isn't usable, the script
  stops instead of silently training on the CPU.
- **Scale.** 8 arms (4 baselines + 4 ablation arms of the proposed model) × 5 seeds × (1 official split + LOSO folds:
  10 UCI-HAR, 12 WISDM, 8 PAMAP2) gives **1320 training runs** (evaluation protocol v2, see
  `docs/protocol_v2.md`). Keep the laptop plugged in and set it to never sleep.
- **Parallel training.** The 1320 runs are split into 264 jobs (one per dataset × model × variant × split × fold,
  each covering all seeds), and `JOBS` of them run at the same time. `JOBS=auto` (the default) sizes this from the
  hardware: on an NVIDIA GPU up to 12 jobs, limited by CPU cores, GPU memory (~1.5 GB/job) and RAM (~2.5 GB/job).
  For example, a 3090 with 16 cores and 32 GB RAM runs 11 jobs; with 64 GB it runs 12. On a CPU-only machine it's
  about 1 job per 4 cores. Each job writes its own `results/tables/parts/<ds>/<job>.csv` and
  `results/logs/train/<ds>/<job>.log`. When training ends (or on Ctrl-C), all parts are merged into
  `results/tables/raw_results.csv` in a fixed order (dataset → model → split → fold → variant → seed). Progress and
  an ETA are shown on screen and saved in `results/logs/train.log`. A failed job is retried once; if it fails again,
  the others keep going and the script stops with an error after training, listing the failed jobs' logs.
- **Resume.** If the run is interrupted (crash, reboot, Ctrl-C), run the same command again. Downloads that are
  already present are skipped, and so is every run already recorded in `results/tables/raw_results.csv`
  (keyed on model/variant/dataset/split/fold/seed/protocol).

**Options** (environment variables, e.g. `JOBS=10 DEVICE=cuda bash scripts/run_all.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `auto` | `cuda`, `cpu`, `mps` or `auto` (CUDA if available, else CPU). Use `cuda` on a GPU machine so a broken install fails loudly |
| `JOBS` | `auto` | number of training jobs run at the same time (`1` = one after another). Lower it if you hit out-of-memory errors |
| `DATASETS` | `ucihar wisdm pamap2` | subset of datasets |
| `STEPS` | all | subset of `download preprocess smoke train profile stats evaluate` |
| `TORCH_CUDA` | `cu128` | PyTorch CUDA wheel index. Try `cu126` if the driver is too old for the default |
| `BASE_PY` | auto | interpreter used to create the venv, e.g. `python3.12` |
| `VENV` | `.venv` | venv location |
| `NO_VENV=1` + `PY=...` | off | skip venv management and use that interpreter as-is |

**Troubleshooting**
- `$'\r': command not found` / `pipefail\r`: the script has Windows line endings. The repo's `.gitattributes`
  prevents this on a fresh clone. If a copy is already affected, fix it with `sed -i 's/\r$//' scripts/run_all.sh`.
- `no Python >= 3.10 found`: install Python from python.org, or point to it with `BASE_PY=/path/to/python`.
- `GPU is present but this PyTorch cannot use it`: update the NVIDIA driver, then
  `rm .venv/.installed_requirements.txt` and re-run (optionally with `TORCH_CUDA=cu126`).
- `pip` can't find a torch wheel: your Python is too new for the published CUDA wheels. Use Python 3.12 via `BASE_PY`,
  and delete `.venv` first.

**Bringing results back.** Everything the paper needs is in `results/` (`tables/`, `preds/`, `figures/`),
and all of it is tracked by git. Either commit and push from the run machine, or zip the `results/` folder and copy it over.
`checkpoints/` and `results/logs/` are git-ignored, so copy those by hand if you want them.

### Individual steps

Each step can also be run on its own (inside the venv: `source .venv/bin/activate`, or
`source .venv/Scripts/activate` in Git Bash on Windows):

| Step | Command | Output |
|---|---|---|
| Download | `python data/download_data.py --dataset all` | `data/raw/` |
| Preprocess | `python -m src.preprocessing --dataset all` | `data/processed/<ds>/{X,y,subject}.npy`, `meta.json` |
| Train (all, parallel) | `python -m src.run_parallel [--jobs N] [--datasets ...] [--dry_run]` | `results/tables/parts/`, merged into `results/tables/raw_results.csv` |
| Train (one model) | `python -m src.train --config configs/<ds>_config.yaml --model <model> [--dual_attn_variant all] [--folds K]` | `results/tables/raw_results.csv`, `results/preds/*.npz`, `checkpoints/` |
| Merge | `python -m src.merge_results` | `results/tables/raw_results.csv` (parts + existing rows, deduplicated, fixed order) |
| Efficiency | `python -m src.profile_efficiency` | `results/tables/efficiency.csv` (params, FLOPs, CPU/GPU latency) |
| Significance | `python -m src.stats_test` | `results/tables/significance.csv` |
| Tables & figures | `python -m src.evaluate` | `results/tables/master_comparison.{csv,md}`, `results/tables/confusion/`, `results/figures/` |

Protocol details:
- Early stopping monitors a **subject-wise** validation set (10% of each fold's training subjects), never test subjects.
- Labels use one dataset-wide mapping; normalization is fit on each fold's training subjects only.
- Latency is measured in one controlled pass (`profile_efficiency.py`, 1 CPU thread by default), not during training.
- PAMAP2: acc + gyro of all 3 IMUs (18 channels), downsampled 100 → 50 Hz, subject 109 excluded (8-fold LOSO).
- The notebooks are for exploration; use `src.train` for anything that goes into the results table.

Results produced before the 2026-09-22 pipeline fixes are archived in `results/archive/2026-09-22_pre_fix/` (see its README).

## Status

Project underway — see week-by-week timeline in the project plan. Repo committed weekly as provenance evidence.

## Repository structure

```
configs/          per-dataset YAML configs (hyperparameters, seeds, LOSO folds)
data/             download_data.py; raw/ and processed/ are generated (git-ignored)
src/              preprocessing, data_loader, train, profile_efficiency, stats_test, evaluate, utils
src/models/       bigru_baseline, bilstm_attention_baseline, cnn_bigru_reference, deepconvlstm, dual_attention_bigru
scripts/          run_all.sh -- the end-to-end pipeline
notebooks/        exploration only (not used for reported results)
results/          tables/, preds/, figures/ (generated); archive/ holds superseded runs
docs/, report/    research-gap notes and literature review
checkpoints/      model weights (generated, git-ignored)
```
