#!/usr/bin/env bash
# scripts/run_all.sh -- the complete pipeline, end to end.
#
#   bash scripts/run_all.sh                 # everything; training runs JOBS jobs at once (auto-sized)
#   JOBS=10 bash scripts/run_all.sh         # force 10 concurrent training jobs (JOBS=1: one at a time)
#   DATASETS="ucihar" bash scripts/run_all.sh
#   STEPS="train evaluate" bash scripts/run_all.sh   # run only some steps
#
# Steps: download preprocess smoke train profile stats evaluate
# Safe to re-run: downloads are skipped if present and every finished training
# run is skipped (resume from results/tables/raw_results.csv).
# Logs: results/logs/<step>.log; training: results/logs/train.log (progress)
# and results/logs/train/<dataset>/<job>.log (one per job).
#
# Training (src/run_parallel.py) splits the grid into 264 jobs (dataset x model
# x variant x split x fold, all seeds). Each job writes its own
# results/tables/parts/<dataset>/<job>.csv; at the end they are merged into
# results/tables/raw_results.csv in a fixed order. JOBS=auto picks the number
# of concurrent jobs from GPU memory, RAM and CPU cores (up to 12 on a GPU).
#
# Environment: on first run a virtualenv is created in .venv and
# requirements.txt is installed into it (re-installed whenever
# requirements.txt changes). If an NVIDIA GPU is present (nvidia-smi on PATH),
# the CUDA build of PyTorch is installed and the run aborts if CUDA is not
# usable, instead of silently training on the CPU.
#   VENV=.venv            where to put the virtualenv
#   BASE_PY=python3.12    interpreter used to create it (default: first working of py -3 / python3 / python)
#   TORCH_CUDA=cu128      PyTorch CUDA wheel index (https://download.pytorch.org/whl/<TORCH_CUDA>)
#   NO_VENV=1 PY=python   skip venv management and use this interpreter as-is
# Windows: run from Git Bash (bash + tee are needed), or double-click / run
# scripts\run_all.bat from cmd/PowerShell, which starts this script in Git Bash.
# Without Python, Python 3.12 is installed with winget. The PC is kept awake
# while the pipeline runs.

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"ucihar wisdm pamap2"}
STEPS=${STEPS:-"download preprocess smoke train profile stats evaluate"}
JOBS=${JOBS:-auto}
DEVICE=${DEVICE:-auto}
VENV=${VENV:-.venv}
TORCH_CUDA=${TORCH_CUDA:-cu128}
MODELS="bigru_baseline bilstm_attention_baseline cnn_bigru_reference deepconvlstm dual_attention_bigru"
mkdir -p results/logs

# UTF-8 file/console I/O on Windows (default there is cp1252); unbuffered so
# `| tee` logs show progress live instead of in 8KB chunks
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
banner()   { echo; echo "==================== $* ===================="; }

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
is_windows() { [[ $(uname -s) == MINGW* || $(uname -s) == MSYS* || $(uname -s) == CYGWIN* ]]; }

# a usable interpreter >= 3.10 (skips the Windows Store "python" stub, which
# exists on PATH but fails when run). 3.12/3.11 are preferred: the newest
# Python release often has no CUDA PyTorch wheels yet.
find_base_python() {
  local c
  local la=${LOCALAPPDATA:-}; la=${la//\\//}
  for c in ${BASE_PY:-} "py -3.12" "py -3.11" "py -3" python3 python \
           "$la/Programs/Python/Launcher/py.exe -3.12" "$la/Programs/Python/Python312/python.exe"; do
    if $c -c "import sys; sys.exit(sys.version_info < (3, 10))" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# Windows without Python: install 3.12 for the current user with winget
install_python_windows() {
  is_windows && command -v winget >/dev/null 2>&1 || return 1
  echo "no Python >= 3.10 found -> installing Python 3.12 with winget (one time)"
  winget install --id Python.Python.3.12 -e --scope user --silent \
      --accept-package-agreements --accept-source-agreements || true
}

setup_env() {
  banner "0/7 python environment ($VENV)"
  # a venv left half-made by an interrupted first run is rebuilt
  if [[ -d $VENV && ! -x $VENV/Scripts/python.exe && ! -x $VENV/bin/python ]]; then
    echo "$VENV is incomplete -> recreating it"; rm -rf "$VENV"
  fi
  if [[ ! -d $VENV ]]; then
    local base
    base=$(find_base_python) || { install_python_windows && base=$(find_base_python); } \
      || { echo "no Python >= 3.10 found; install one from python.org (or set BASE_PY=...) and re-run"; exit 1; }
    echo "creating $VENV with: $base ($($base --version 2>&1))"
    $base -m venv "$VENV"
  fi
  if [[ -x $VENV/Scripts/python.exe ]]; then PY="$VENV/Scripts/python.exe"   # Windows
  else PY="$VENV/bin/python"; fi

  local has_gpu=0
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && has_gpu=1

  # (re)install only when requirements.txt changed since the last install
  local stamp=$VENV/.installed_requirements.txt
  if ! cmp -s requirements.txt "$stamp"; then
    $PY -m pip install --upgrade pip
    if [[ $has_gpu == 1 ]]; then
      echo "NVIDIA GPU found -> installing CUDA PyTorch from the $TORCH_CUDA index"
      $PY -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    fi
    $PY -m pip install -r requirements.txt
    cp requirements.txt "$stamp"
  else
    echo "requirements already installed (delete $stamp to force a reinstall)"
  fi

  $PY -c "import torch, sys; print(f'python {sys.version.split()[0]} | torch {torch.__version__} | cuda available: {torch.cuda.is_available()}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))"
  if [[ $has_gpu == 1 ]] && ! $PY -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
    echo "An NVIDIA GPU is present but this PyTorch cannot use it (CPU-only build or driver too old)."
    echo "Fix: rm $stamp && $PY -m pip uninstall -y torch, update the NVIDIA driver, then re-run"
    echo "     (or try another wheel index, e.g. TORCH_CUDA=cu126). To train on CPU anyway: DEVICE=cpu."
    [[ $DEVICE == cpu ]] || exit 1
  fi
}

# Windows: keep the PC from going to sleep while this script runs (training
# takes hours; sleep would suspend it). Only a request held by a background
# PowerShell, released when this script exits -- no power settings change.
if is_windows && command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -NonInteractive -Command '
    $s = Add-Type -PassThru -Name P -Namespace W -MemberDefinition "[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);"
    $null = $s::SetThreadExecutionState([uint32]2147483649)
    while ($true) { Start-Sleep 3600 }' >/dev/null 2>&1 &
  AWAKE_PID=$!
  trap 'kill $AWAKE_PID 2>/dev/null || true' EXIT
  echo "keeping Windows awake until the pipeline finishes"
fi

if [[ ${NO_VENV:-0} == 1 ]]; then
  PY=${PY:-$(find_base_python)} || { echo "no Python >= 3.10 found; set PY=..."; exit 1; }
else
  setup_env 2>&1 | tee results/logs/setup.log
  # the pipe runs setup_env in a subshell, so resolve PY again here
  if [[ -x $VENV/Scripts/python.exe ]]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
fi

if has_step download; then
  banner "1/7 download raw datasets"
  $PY data/download_data.py --dataset all 2>&1 | tee results/logs/download.log
fi

if has_step preprocess; then
  banner "2/7 preprocess (window + save .npy)"
  $PY -m src.preprocessing --dataset all 2>&1 | tee results/logs/preprocess.log
  $PY -m src.data_loader 2>&1 | tee -a results/logs/preprocess.log
fi

if has_step smoke; then
  banner "3/7 smoke test (1 epoch, 1 fold; writes results/smoke/ only)"
  for ds in $DATASETS; do
    for m in $MODELS; do
      $PY -m src.train --config "configs/${ds}_config.yaml" --model "$m" --split_mode both \
          --dual_attn_variant all --device "$DEVICE" --smoke
    done
  done 2>&1 | tee results/logs/smoke.log
fi

if has_step train; then
  banner "4/7 train all models x variants x folds x seeds ($JOBS parallel jobs)"
  $PY -m src.run_parallel --datasets $DATASETS --models $MODELS --jobs "$JOBS" --device "$DEVICE" \
      2>&1 | tee results/logs/train.log
fi

if has_step profile; then
  banner "5/7 efficiency profiling (params / FLOPs / latency, one controlled pass)"
  $PY -m src.profile_efficiency --datasets $DATASETS 2>&1 | tee results/logs/profile.log
fi

if has_step stats; then
  banner "6/7 significance tests (Wilcoxon / paired t, LOSO folds)"
  $PY -m src.stats_test 2>&1 | tee results/logs/stats.log
fi

if has_step evaluate; then
  banner "7/7 master table, confusion matrices, figures"
  $PY -m src.evaluate 2>&1 | tee results/logs/evaluate.log
fi

banner "done"
echo "results/tables/raw_results.csv       every run (one row per model/variant/dataset/split/fold/seed)"
echo "results/tables/parts/                per-job results (merged into raw_results.csv)"
echo "results/logs/train/                  per-job training logs"
echo "results/tables/master_comparison.md  paper table   (.csv alongside)"
echo "results/tables/significance.csv      p-values      results/tables/efficiency.csv  latency"
echo "results/figures/                     plots         results/preds/                 per-run predictions"
