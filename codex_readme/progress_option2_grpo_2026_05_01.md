# Progress: Option2 GRPO Post-Training (2026-05-01)

## 1) Method Summary

Current main method is **Option 2 surrogate GRPO** post-training on top of official PepFlow checkpoint (`model1.pt`), implemented with **DDP multi-GPU** training.

Core update loop:

1. Use a frozen old policy (`model_old`) to rollout grouped samples.
2. Compute reward using sequence recovery signal (AAR).
3. Build groupwise normalized advantage within each condition group.
4. Compute GRPO clipped surrogate objective (`rho` ratio based on current vs old log-prob scores).
5. Add supervised anchor loss:
   - `L_total = L_grpo + lambda_sup * L_sup`
6. Backprop + grad clip + optimizer step.
7. Periodically sync `model_old` from current model.

Key stabilizations already added:

- Lower LR and lighter supervised anchor weight.
- Non-finite guards on loss / grad / grad norm.
- Safer categorical sampling fallback in `sample_from`.
- Multi-update-per-rollout + delayed old-policy sync to strengthen GRPO signal.

---

## 2) Problem Setting

Current setting follows **fixed-backbone sequence design style** (paper Table 2 direction), specifically:

- `sample_bb=False`
- `sample_ang=False`
- `sample_seq=True`

Interpretation:

- Backbone and angles are not sampled in this setting.
- Training/evaluation mainly optimizes and measures sequence recovery behavior.
- Primary quick metric used in our recent checks: **AAR**.

---

## 3) Training & Debugging Progress

### Implemented

- Added DDP entrypoint for Option2 GRPO.
- Added 4-GPU smoke and normal train shell scripts.
- Added GPU quick-eval script (official ckpt vs post-trained ckpt).

### Debugged Issues

- Previous long run had crash around high iteration with CUDA device assert + NaN gradient behavior.
- Root-cause direction addressed by:
  - safer sampling distribution handling,
  - stricter non-finite guards,
  - stronger/healthier GRPO signal dynamics.

### Current Status

- 4-GPU smoke test: passed.
- 4-GPU full training: completed to iter 1000 successfully.
- No NaN / no DDP deadlock in the completed full run.

---

## 4) Quick Metric Check (GPU, same env)

We ran a quick GPU evaluation comparing:

- `official_model1`
- `option2_grpo_iter1000`

on val split under seq-only setting.

Observed AAR:

- official: `56.78%`
- option2: `60.63%`
- delta: `+3.85` percentage points

This suggests the current post-training direction is **improving** sequence recovery signal (at least in this quick protocol), not showing obvious regression/bug pattern.

---

## 5) Files Involved

### Core Training

- `pepflow/train_ddp_option2.py`
  - DDP Option2 GRPO training logic and logging/checkpointing.

### Stability Fixes

- `pepflow/pepflow/modules/common/layers.py`
  - hardened `sample_from` for invalid probability rows / NaN-safe sampling.

### Run Scripts

- `pepflow/scripts/run_option2_ddp_smoke_4gpu.sh`
- `pepflow/scripts/run_option2_ddp_train_4gpu.sh`
- `pepflow/scripts/run_quick_eval_option2_gpu.sh`

### Outputs / Artifacts

- `pepflow/outputs/option2_grpo_seqonly_train4g_learn_angle_2026_05_01__07_30_53/`
  - full train logs, metrics, checkpoints (`1000.pt` etc.)
- `pepflow/outputs/quick_eval_option2_vs_official.json`
- `pepflow/outputs/quick_eval_option2_vs_official.log`

### Paper Reference (local text)

- `codex_readme/pepflow.txt`
  - Table 1 / Table 2 baseline references for rough comparison context.

