# Decision: replace cuDNN multi-layer RNN dropout with `StackedRNN`

**Date:** 2026-10-16
**Affects:** `src/models/stacked_rnn.py` (new), `src/models/bigru_baseline.py`,
`src/models/bilstm_attention_baseline.py`, `src/models/cnn_bigru_reference.py`,
`src/models/dual_attention_bigru.py`
**Environment where the problem occurs:** Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU,
Python 3.11.9, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19.0)

---

## 1. Problem

`scripts/run_all.sh` stopped silently during step 3/7 (smoke test). It printed no error.
The last output was the successful smoke run of `bigru_baseline` on UCI-HAR:

```
[bigru_baseline/-/ucihar/loso] fold=0 ...
    -> acc=0.8018 macro_f1=0.7993 (best epoch 1/1, 1s)
done: 2 runs trained, 0 already in ...\results\smoke\raw_results.csv (skipped)
```

The next model (`bilstm_attention_baseline`) never started, and step 4/7 (train) was never reached.

## 2. Investigation

| # | Check | Result | Conclusion |
|---|-------|--------|------------|
| 1 | Re-ran a smoke run by hand and printed `$?` | Training finished and printed `done:`, then **exit code 127** | The script uses `set -euo pipefail`, so any non-zero exit stops the pipeline silently. |
| 2 | Searched `src/train.py` for `sys.exit` / `os._exit` / `atexit` | None | Python is not choosing this exit code. |
| 3 | Ran bare Python, `import torch`, a CUDA tensor, and a cuDNN GRU/LSTM forward pass | All exit 0 | The CUDA/cuDNN install is not broken in general. |
| 4 | Called `run()` and then `os._exit(0)` | Still 127 | The code is set during native process teardown, not by Python. |
| 5 | Same smoke run with `--device cpu` vs `--device cuda` | CPU → 0, CUDA → 127 | The problem is specific to the GPU path. |
| 6 | Stage-by-stage bisection of the run loop (data, model build, FLOPs, dataloaders, `.to(cuda)`, first forward pass, training) | Exits 0 up to `.to(cuda)`; **127 from the first forward pass onward** | One forward pass of the real model on CUDA is enough to trigger it. |
| 7 | Isolated the layers in `BiGRUBaseline` | `nn.GRU(num_layers=2, dropout=0.3)` in **train mode** → 127; same in eval mode → 0; no dropout → 0; `nn.Linear`, matmul and `F.dropout` on CUDA → 0. `nn.LSTM(num_layers=2, dropout=0.3)` → also crashes | **Trigger: cuDNN's built-in inter-layer RNN dropout in training mode.** |
| 8 | Read the raw exit status from PowerShell instead of Git Bash | `-1073740791` = **`0xC0000409`** (`STATUS_STACK_BUFFER_OVERRUN`, a native fail-fast). Git Bash reports it as 127. Even `TerminateProcess(..., 3)` ends with this status | It is a hard native crash during shutdown, not a normal exit code. |
| 9 | Web search for this PyTorch/cuDNN combination | No matching bug report; only older Windows + cuDNN RNN reports with the same exit code | No upstream fix or workaround was found. |
| 10 | Single-layer GRUs with `nn.Dropout` between them, trained for 5 steps (forward, backward, Adam) on CUDA | Exit 0 | A workaround that keeps the same architecture exists. |

### Why it mattered beyond the smoke test

`src/run_parallel.py` counts a job as failed when its exit code is not 0. It retries once, then
gives up and finally returns 1. Since every model uses a 2-layer RNN with dropout, **every one of the
161 training jobs** would have been marked failed, even though their results were written
correctly. `run_all.sh` would then have stopped before profile, stats and evaluate.

All four models were affected because each one builds its RNN like this:

```python
nn.GRU / nn.LSTM(..., num_layers=num_layers, bidirectional=True,
                 dropout=dropout if num_layers > 1 else 0.0)
```

## 3. Options considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **A. Patch the models:** build the multi-layer RNN from single-layer cuDNN RNNs with `nn.Dropout` between layers | Same architecture and parameter shapes, so the model definition doesn't change; avoids cuDNN dropout entirely; verified to exit cleanly (check 10); works on any OS/GPU | Code change in 4 model files; checkpoint key names change; the dropout masks come from a different RNG stream than cuDNN's | **Chosen** |
| B. Reinstall PyTorch from another wheel index (e.g. `TORCH_CUDA=cu126`) | No code changes | About 2.5 GB download; **not verified** to fix the crash; fixes only this machine and leaves the pipeline fragile elsewhere | Rejected: unverified |
| C. Ignore the exit code, e.g. treat a crash after `done:` as success in `run_parallel.py` / `run_all.sh` | Small change | Hides real crashes; a native fault during teardown could also hide incomplete writes; hacky | Rejected: unsafe |
| D. Disable cuDNN for RNNs (`torch.backends.cudnn.enabled = False`) | No architecture change | Much slower training on hundreds of runs | Rejected: too slow |
| E. Set RNN dropout to 0 | Simple | **Changes the model** (removes regularisation), which invalidates the comparison with the planned hyperparameters | Rejected: changes the experiment |

The user picked option A when asked.

## 4. Change made

### New: `src/models/stacked_rnn.py`

`StackedRNN(cell, input_size, hidden_size, num_layers, bidirectional, dropout)`:

- builds `num_layers` single-layer `nn.GRU` / `nn.LSTM` modules (`batch_first=True`). Layer 0 takes
  `input_size`; the later layers take `hidden_size * num_directions`.
- applies `nn.Dropout(dropout)` to the input of every layer except the first, which is the same as
  dropout on every layer's output except the last. This is the behaviour documented for
  `nn.GRU(num_layers, dropout)`.
- returns `(out, None)` so that existing call sites (`out, _ = self.gru(x)`) keep working. No model
  uses the final hidden state.

### Updated: the 4 models

In each model, `nn.GRU(...)` / `nn.LSTM(...)` became `StackedRNN("gru" | "lstm", ...)` with the same
`input_size`, `hidden_size`, `num_layers`, `bidirectional=True` and
`dropout=dropout if num_layers > 1 else 0.0`. A two-line comment at each site points to
`stacked_rnn.py`. Attribute names (`self.gru`, `self.lstm`) and `forward()` are unchanged.

## 5. Verification

1. **Parameter counts and FLOPs are identical** before and after, for all 4 models on UCI-HAR `(128, 6)`, 6 classes and PAMAP2 `(128, 18)`, 12 classes:

   | Model | Params (ucihar / pamap2) | FLOPs (ucihar / pamap2) |
   |-------|--------------------------|--------------------------|
   | bigru_baseline | 119,430 / 124,812 | 26.64 M / 27.82 M |
   | bilstm_attention_baseline | 161,799 / 168,717 | 37.54 M / 39.11 M |
   | cnn_bigru_reference | 259,270 / 263,884 | 18.73 M / 19.71 M |
   | dual_attention_bigru | 127,803 / 133,332 | 28.76 M / 29.94 M |

2. **Every model exits 0** after a CUDA smoke run (`--split_mode both --dual_attn_variant all --smoke`)
   on UCI-HAR, including all `dual_attention_bigru` ablation variants.

## 6. Consequences

- **Checkpoint keys changed**, e.g. `gru.weight_ih_l0` → `gru.layers.0.weight_ih_l0`. Checkpoints
  saved before this change won't load into the new models. No checkpoints from the full run existed
  yet when the change was made.
- **Numbers are not bit-identical to cuDNN's fused dropout**, because the dropout masks come from
  PyTorch's RNG instead of cuDNN's dropout state. The architecture is the same, and every run in
  `raw_results.csv` comes from the new code, so the comparisons between models stay consistent.
- **Speed:** one cuDNN call per layer instead of one fused call. For these 2-layer models the
  overhead is small compared with data loading and Python overhead.
- **Revisiting:** if a later PyTorch/cuDNN release fixes the teardown crash, the models can go back
  to `nn.GRU/nn.LSTM(num_layers, dropout)`. Check first with:
  ```
  python -c "import torch; m=torch.nn.GRU(6,64,num_layers=2,bidirectional=True,batch_first=True,dropout=0.3).cuda(); m(torch.randn(64,128,6).cuda())"
  ```
  and confirm the exit code is 0 (`echo $?` in Git Bash, `$LASTEXITCODE` in PowerShell).

---

## Appendix: running the pipeline from PowerShell

Two smaller issues came up before this one:

- `JOBS=8 bash scripts/run_all.sh` is bash syntax. In PowerShell, set the variable first: `$env:JOBS = 8`.
- `bash` is not on PowerShell's PATH on this machine. Call Git Bash directly, or use the `.bat` launcher:
  ```powershell
  $env:JOBS = 8; & "C:\Program Files\Git\bin\bash.exe" scripts/run_all.sh
  # or
  $env:JOBS = 8; .\scripts\run_all.bat
  ```
