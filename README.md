# Dual-Attention BiGRU for Human Activity Recognition

A CNN-free, pure recurrent + attention architecture for HAR from wearable/inertial sensor data, evaluated with an emphasis on **efficiency accounting** and **evaluation rigor** — two dimensions largely missing from recent dual-attention HAR literature.

## TL;DR

We set out to propose a spectral-gated dual-attention BiGRU (SG-DA-BiGRU). Under a controlled, subject-independent
evaluation (8 models, 3 benchmarks, 30 folds, 5 seeds, 1,320 runs) **attention gives no measurable gain over a plain
BiGRU**, and the spectral gate is practically equivalent to the same model without it. The paper
(`paper/paper.pdf`) is therefore an empirical study, not a method paper, exactly as the decision rule fixed before the
final run prescribed. The best practical models are the convolutional hybrids (CNN+BiGRU, DeepConvLSTM).

## Motivation and novelty

Recent HAR papers (Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer) report accuracy-only comparisons on single
train/test splits. This project asks whether attention gains survive subject-independent evaluation, with parameter,
FLOPs and measured-latency accounting, multi-seed runs and paired statistical tests across UCI-HAR, WISDM and PAMAP2.

Differences from existing work:

| Aspect | Existing dual-attention HAR | This project |
|---|---|---|
| Channel attention input | learned time-domain statistics | + non-learned **spectral gate** (low-band energy ratio, spectral entropy; 33 parameters, one FFT per window) |
| Front-end | Inception / TCN / CNN branches | **CNN-free** BiGRU, so the attention effect is isolated |
| Evaluation | single split, one run, accuracy | leave-subjects-out CV, 5 seeds per fold, Wilcoxon + Holm, Bayesian ROPE test |
| Cost | rarely reported | parameters, FLOPs and measured CPU/GPU latency |
| Claim discipline | n/a | decision rule committed on 2026-10-16, before the final runs |

See `docs/research_gap_notes.md` for the gap analysis and `report/literature_review.md` for the literature review.

## Models compared (8 total)

All share one training pipeline, hidden size, depth, dropout and head (except DeepConvLSTM, which uses its published configuration).

| Model | Backbone | Channel attn | Spectral gate | Temporal attn | Front-end |
|---|---|---|---|---|---|
| **SG-DA-BiGRU** (proposed, `full`) | BiGRU | yes | yes | yes | none |
| DA-BiGRU (`plain`) | BiGRU | yes | no | yes | none |
| Channel-only | BiGRU | yes | yes | mean-pool | none |
| Temporal-only | BiGRU | no | no | yes | none |
| BiGRU | BiGRU | no | no | mean-pool | none |
| BiLSTM + Attn | BiLSTM | no | no | yes | none |
| CNN + BiGRU | BiGRU | no | no | mean-pool | 3 x Conv1D |
| DeepConvLSTM | LSTM | no | no | last step | 4 x Conv |

`full` vs `plain` isolates the spectral gate. Proposed model layers: input window (T x C, z-scored) -> channel attention
(mean+max pooling, MLP C -> C/4 -> C, sigmoid) -> spectral gate (rFFT descriptors, MLP 2 -> 8 -> 1) -> reweighting
`w = a*g / mean(a*g)` -> 2-layer BiGRU (64 units per direction, dropout 0.3) -> additive temporal attention (64-d) ->
FC 128 + ReLU + dropout -> FC K.

## Datasets

| Dataset | Subjects | Classes | Channels | Rate | Windows | LOSO folds |
|---|---|---|---|---|---|---|
| UCI-HAR | 30 | 6 | 6 | 50 Hz | 10,299 | 10 |
| WISDM v1.1 | 36 | 6 | 3 | 20 Hz | 42,657 | 12 |
| PAMAP2 | 8 (subject 109 excluded) | 12 | 18 | 50 Hz | 14,922 | 8 |

All use 2.56 s windows with about 50% overlap. Each dataset is evaluated independently in native mode.

## Evaluation

- Primary: leave-subjects-out cross-validation (30 folds in total), 20% of training subjects (at least 2) as the subject-wise validation set, normalisation fitted on training subjects only.
- Secondary: fixed split (official for UCI-HAR, seeded 80/20 subject split for WISDM and PAMAP2).
- 5 seeds per fold, averaged within the fold (seeds are never treated as independent samples).
- Primary metric macro-F1; accuracy also reported. Params, FLOPs and latency for every model.
- Pre-specified decision rule: pooled fold-level macro-F1, two-sided Wilcoxon signed-rank, Holm-corrected over 7 comparisons. Method paper only if `full` beats `plain` (p Holm < 0.05) and is not significantly worse than any baseline.
- Training: Adam, lr 1e-3, batch 64, at most 50 epochs, early stopping and LR schedule on validation macro-F1. 27.6 GPU-hours on an RTX 3050 laptop GPU.

## Studies carried out

1. **Literature review and gap analysis** (July 2026). Ten foundation papers plus five 2025-26 papers that overlap the original novelty (Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer, AttnFusionGRU). Result: architecture novelty is closed, so the project was repositioned to evaluation rigor and efficiency accounting.
2. **Pipeline audit and fixes** (2026-09-22). Found and fixed: unlogged ablation variant, duplicated runs, latency measured on mixed hardware, window-level validation leakage, PAMAP2 using one IMU at 100 Hz plus the near-empty subject 109, WISDM session merging, per-fold label encoding. Old numbers are in `results/archive/2026-09-22_pre_fix/` and must not be used.
3. **Protocol v1 run** (2026-10-15, 483 runs). CNN+BiGRU best on all three datasets; `full` vs `plain` gave +0.003 macro-F1 (p = 0.73). Diagnosis from training curves: with checkpoints chosen on validation loss, 43.5% of runs picked epoch <= 3, and 6-fold tests could not reach significance. Archived in `results/archive/2026-10-15_protocol_v1/`.
4. **Protocol v2** (`docs/protocol_v2.md`). Select on validation macro-F1, at least 2 validation subjects, more folds (10/12/8), 5 seeds, DeepConvLSTM added, decision rule committed. Median selected epoch rose from 4 to 9; runs stopping by epoch 3 fell to 13.3%.
5. **Final run** (1,320 runs): main comparison, ablation of the gate and attention components, Friedman + Nemenyi, Bayesian signed-rank test, per-class analysis, variance decomposition, fixed-split vs LOSO rank comparison, efficiency profiling, and a descriptor analysis of what the spectral gate sees.

## Results

### LOSO macro-F1 (mean +/- std across folds, each fold averaged over 5 seeds)

| Model | UCI-HAR (10) | WISDM (12) | PAMAP2 (8) | Pooled (30) |
|---|---|---|---|---|
| BiGRU | 0.920 +/- 0.049 | 0.810 +/- 0.094 | 0.695 +/- 0.198 | 0.816 |
| BiLSTM+Attn | 0.921 +/- 0.049 | 0.820 +/- 0.091 | 0.687 +/- 0.185 | 0.819 |
| CNN+BiGRU | **0.925** +/- 0.042 | **0.838** +/- 0.091 | 0.705 +/- 0.207 | **0.832** |
| DeepConvLSTM | 0.917 +/- 0.051 | 0.820 +/- 0.085 | **0.746** +/- 0.109 | **0.832** |
| SG-DA-BiGRU (proposed) | 0.922 +/- 0.045 | 0.813 +/- 0.091 | 0.685 +/- 0.190 | 0.815 |
| DA-BiGRU | 0.921 +/- 0.047 | 0.820 +/- 0.087 | 0.674 +/- 0.191 | 0.815 |
| Channel-only | 0.916 +/- 0.051 | 0.805 +/- 0.093 | 0.697 +/- 0.184 | 0.813 |
| Temporal-only | 0.918 +/- 0.053 | 0.816 +/- 0.089 | 0.678 +/- 0.178 | 0.813 |

### Pre-specified primary endpoint (SG-DA-BiGRU minus comparator, pooled over 30 folds)

| Comparator | Delta macro-F1 | 95% bootstrap CI | Wins | p Holm |
|---|---|---|---|---|
| DA-BiGRU (gate vs no gate) | +0.0008 | [-0.005, +0.007] | 17/30 | 1.000 |
| BiGRU | -0.0006 | [-0.010, +0.010] | 13/30 | 1.000 |
| BiLSTM+Attn | -0.0031 | [-0.010, +0.003] | 11/30 | 1.000 |
| Channel-only | +0.0020 | [-0.008, +0.013] | 16/30 | 1.000 |
| Temporal-only | +0.0025 | [-0.004, +0.010] | 16/30 | 1.000 |
| CNN+BiGRU | -0.0164 | [-0.029, -0.004] | 9/30 | 0.065 |
| DeepConvLSTM | -0.0171 | [-0.040, +0.001] | 11/30 | 1.000 |

The Bayesian signed-rank test (ROPE +/-0.01) gives 99.5% posterior probability that the gate is practically equivalent
to no gate; the probability that SG-DA-BiGRU is worse than CNN+BiGRU is 0.95.

### Other findings

- **Subjects dominate.** Which subjects are held out explains 94-96% of fold-score variance; the choice among the 8 models explains 0.4-1.5%. Per-subject macro-F1 of the proposed model ranges 0.64-1.00 (UCI-HAR), 0.16-0.97 (WISDM), 0.27-0.84 (PAMAP2).
- **Fixed splits mislead.** Kendall's tau between fixed-split and LOSO model rankings is between -0.14 and 0.43.
- **Where convolutions help.** WISDM: upstairs +5.3, downstairs +3.7 points (CNN+BiGRU vs BiGRU); PAMAP2: walking +10.2, ironing +7.9 (DeepConvLSTM).
- **Why the gate does not help.** Its descriptors mainly separate static from dynamic activities (logistic regression on descriptors alone: macro-F1 0.71 UCI-HAR, 0.51 WISDM), which all models already do almost perfectly; the remaining errors (stairs, sit vs stand, household tasks) need fine temporal structure. A BiGRU can also re-learn channel weighting itself.
- **Checkpoint selection matters.** Selecting on validation loss caused premature stopping under subject shift; selecting on validation macro-F1 fixed it.

### Computational cost (UCI-HAR inputs; latency on RTX 3050 laptop GPU, batch 32)

| Model | Params (k) | MFLOPs / window | GPU latency (ms) |
|---|---|---|---|
| BiGRU | 119.4 | 26.6 | 1.06 |
| BiLSTM+Attn | 161.8 | 37.5 | 1.15 |
| CNN+BiGRU | 259.3 | 18.7 | 2.73 |
| DeepConvLSTM | 458.1 | 175.0 | 3.58 |
| SG-DA-BiGRU | 127.8 | 28.8 | 4.21 |
| DA-BiGRU | 127.8 | 28.8 | 1.64 |

The gate adds 33 parameters and negligible FLOPs (the FFT is not counted) but makes GPU inference about 2.6x slower
than the same model without it. CNN+BiGRU needs the fewest FLOPs because pooling shortens the sequence.

## Conclusion

- The spectral gate is practically equivalent to the same model without it (+0.0008 macro-F1), so the decision rule's condition for a method contribution failed, and the work is reported as an empirical study.
- No attention variant improves on a plain BiGRU under subject-independent evaluation on UCI-HAR, WISDM and PAMAP2.
- Convolutional-recurrent hybrids rank highest (pooled 0.832), though the gap to the proposed model is not significant after Holm correction (p = 0.065). CNN+BiGRU is the best practical choice: top accuracy with the fewest FLOPs.
- Differences between architectures are dwarfed by differences between held-out subjects, and single-split rankings are poor guides to subject-independent ones.
- Recommendations for future HAR papers: evaluate on unseen subjects; select checkpoints on subject-wise validation macro-F1; use enough folds and pool across datasets; ablate against the identical model without the component; report equivalence-aware statistics; fix the decision rule before the final run; report measured latency, not only FLOPs.

## Limitations

- One shared hyper-parameter configuration; no per-model or nested tuning under LOSO.
- Three benchmarks with 6-18 channels and 2.56 s windows; attention may matter with more sensor positions, longer windows or multimodal inputs.
- PAMAP2 has only 8 subjects, so its per-dataset tests are descriptive.
- Protocol revised once (before the final run, identically for all models); the Bayesian, Friedman and per-class analyses were not part of the pre-specified rule.
- Macro-F1 over the union of true and predicted classes lowers absolute PAMAP2 scores for all models.
- Latency measured on one laptop GPU and one CPU thread; FFT cost may differ on embedded hardware.
- The descriptor analysis (gate inputs) covers UCI-HAR and WISDM only, with global rather than per-fold normalisation.

## Paper and presentation

- `paper/paper.pdf` (source `paper/paper.tex`, figures in `paper/figures/`).
- `presentation/HAR_Review_Presentation.pptx`: 22-slide review deck.

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
`venv -> archive -> download -> preprocess -> smoke -> train -> profile -> stats -> evaluate`.

- **Old results.** The `archive` step copies results of an older evaluation protocol (e.g. the 2026-10-15
  v1 run) to `results/archive/<date>_before_v2/` and leaves only current-protocol runs in `results/`, so a
  machine that still holds an earlier run needs no manual cleanup. A half-finished v2 run is kept and resumes.
- **Smoke test.** One epoch of every model on one fold; the pipeline stops at the first model that fails or
  exits non-zero, before any training time is spent.

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
| `STEPS` | all | subset of `archive download preprocess smoke train profile stats evaluate` |
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
- Early stopping monitors a **subject-wise** validation set (20% of each fold's training subjects, at least 2), never test subjects.
- Labels use one dataset-wide mapping; normalization is fit on each fold's training subjects only.
- Latency is measured in one controlled pass (`profile_efficiency.py`, 1 CPU thread by default), not during training.
- PAMAP2: acc + gyro of all 3 IMUs (18 channels), downsampled 100 → 50 Hz, subject 109 excluded (8-fold LOSO).
- The notebooks are for exploration; use `src.train` for anything that goes into the results table.

Results produced before the 2026-09-22 pipeline fixes are archived in `results/archive/2026-09-22_pre_fix/` (see its README).

## Status

Final protocol (v2) run complete (2026-10-21): 1,320 runs, paper drafted as an empirical study. The paper still has
placeholders for affiliation, e-mail, repository URL, acknowledgements and the generative-AI declaration.

## Repository structure

```
configs/          per-dataset YAML configs (hyperparameters, seeds, LOSO folds)
data/             download_data.py; raw/ and processed/ are generated (git-ignored)
src/              preprocessing, data_loader, train, profile_efficiency, stats_test, evaluate, utils
src/models/       bigru_baseline, bilstm_attention_baseline, cnn_bigru_reference, deepconvlstm, dual_attention_bigru
scripts/          run_all.sh -- the end-to-end pipeline
notebooks/        exploration only (not used for reported results)
results/          tables/, preds/, figures/ (generated); archive/ holds superseded runs
docs/, report/    protocol, research-gap notes and literature review
paper/            manuscript, figures and analysis scripts
presentation/     review deck (.pptx)
checkpoints/      model weights (generated, git-ignored)
```
