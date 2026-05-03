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
  - now supports difficulty `--level {1,2,3}`:
    - level1: known-backbone seq-only
    - level2: known-backbone seq+angles
    - level3: full design

### Stability Fixes

- `pepflow/pepflow/modules/common/layers.py`
  - hardened `sample_from` for invalid probability rows / NaN-safe sampling.

### Run Scripts

- `pepflow/scripts/run_option2_ddp_smoke_4gpu.sh`
- `pepflow/scripts/run_option2_ddp_train_4gpu.sh`
- `pepflow/scripts/run_quick_eval_option2_gpu.sh`
- `pepflow/scripts/run_option2_ddp_level2_smoke_4gpu.sh`
- `pepflow/scripts/run_option2_ddp_level2_train_4gpu.sh`
- `pepflow/scripts/run_quick_eval_option2_level2_gpu.sh`

### Outputs / Artifacts

- `pepflow/outputs/option2_grpo_seqonly_train4g_learn_angle_2026_05_01__07_30_53/`
  - full train logs, metrics, checkpoints (`1000.pt` etc.)
- `pepflow/outputs/quick_eval_option2_vs_official.json`
- `pepflow/outputs/quick_eval_option2_vs_official.log`

### Paper Reference (local text)

- `codex_readme/pepflow.txt`
  - Table 1 / Table 2 baseline references for rough comparison context.

---

## 6) Level2 (Seq+Angles) Experiment Update

### Setting

- Option2 surrogate GRPO
- level2 mode:
  - `sample_bb=False`
  - `sample_ang=True`
  - `sample_seq=True`
- 4-GPU DDP

### Bring-up and Training

- Level2 4-GPU smoke:
  - passed (`max_iters=2`, no NaN, no deadlock)
- Level2 train run:
  - run dir: `pepflow/outputs/option2_grpo_level2_seqang_train4g_learn_angle_2026_05_01__08_55_13/`
  - reached and exceeded iter 100 (stopped manually after stable progress >120 for fast validation cycle)
  - checkpoint used for eval: `checkpoints/100.pt`
  - training remained numerically stable (no NaN, `skipped_updates` kept at 0 in monitored iterations)

### Level2 Quick Eval (GPU)

Output:
- `pepflow/outputs/quick_eval_option2_level2_vs_official.json`
- `pepflow/outputs/quick_eval_option2_level2_vs_official.log`

AAR comparison under level2 sampling:

- `official_model1_level2_sampling`: `46.90%`
- `option2_level2_latest` (`iter100`): `44.59%`
- delta: `-2.31` percentage points

Interpretation:

- Pipeline and training/eval infra are working normally in level2 (technical path is healthy).
- Current level2 quality after short training (iter100 checkpoint) is **below** official baseline on AAR.
- This indicates we should continue training longer and/or retune hyperparameters for level2 before claiming improvement.

---

## 7) Latest Update: Tuned Level2 Full Run (Completed)

### Run Plan Executed

- Script: `pepflow/scripts/run_option2_ddp_level2_tuned_train_auto_4gpu.sh`
- 4-GPU DDP, level2 mode (`sample_bb=False, sample_ang=True, sample_seq=True`)
- Tuned key args:
  - `lr=5e-5`
  - `lambda_sup=0.2`
  - `updates_per_rollout=1`
  - `old_sync_interval=8`
  - `max_iters=1000`
  - `save_freq=50`

### Training Completion / Stability

- Run dir:
  - `pepflow/outputs/option2_grpo_level2_seqang_tuned_train4g_learn_angle_2026_05_01__09_17_37/`
- Final checkpoint:
  - `checkpoints/1000.pt`
- Auto-train status:
  - **completed successfully** on first attempt (`[auto-train] completed successfully`)
- Monitoring summary:
  - no NaN loss
  - no skipped updates (`skipped_updates=0` throughout monitored progress)
  - no DDP deadlock
  - no forced restart needed

### Post-Train Quick Eval (GPU, level2 protocol)

- Command path:
  - `pepflow/scripts/run_quick_eval_option2_level2_gpu.sh`
- Output files:
  - `pepflow/outputs/quick_eval_option2_level2_vs_official.json`
  - `pepflow/outputs/quick_eval_option2_level2_vs_official_tuned1000.log`

AAR comparison:

- `official_model1_level2_sampling`: `46.90%`
- `option2_level2_latest` (`iter1000`): `39.01%`
- delta: `-7.89` percentage points

Interpretation:

- Infra path is stable and reproducible (training + resume/checkpoint/eval all normal).
- But this tuned level2 run shows **significant degradation** vs official baseline on AAR under current quick protocol.
- Next tuning should focus on reducing policy drift / improving reward signal quality in level2.

---

## 8) Latest Update: Medium-Explore + Mixed Reward (Level2)

### Setting Applied

- Script:
  - `pepflow/scripts/run_option2_ddp_level2_medium_mix_train_auto_4gpu.sh`
- Mode:
  - level2 (`sample_bb=False, sample_ang=True, sample_seq=True`)
- Hyperparameters (medium explore):
  - `lr=2e-5`
  - `lambda_sup=0.4`
  - `group_size=6`
  - `num_steps=16`
  - `updates_per_rollout=1`
  - `old_sync_interval=2`
  - `clip_eps=0.1`
  - `score_clip=5.0`
- Reward:
  - `reward_mode=mix_level2`
  - `0.7 * AAR + 0.2 * angle_consistency + 0.1 * torsion_consistency`

### Run Outcome

- Run dir:
  - `pepflow/outputs/option2_grpo_level2_medium_mix_train4g_learn_angle_2026_05_01__14_47_13/`
- Full training reached:
  - `iter=1000` with `checkpoints/1000.pt`
- Monitoring summary:
  - no NaN
  - `skipped_updates=0` through monitored trajectory
  - no DDP deadlock

### Fixes Discovered During This Run

1. Mixed reward normalization bug
- `angle reward` was unintentionally scaled too large (missing angle-dimension normalization).
- Fixed in `train_ddp_option2.py` by expanding mask to full angle shape before averaging.

2. Auto-restart resume argument bug
- Auto script resume branch passed `--resume` without `--init_ckpt`, while parser requires `--init_ckpt`.
- Fixed script to always pass `--init_ckpt` in resume branch.

3. Resume RNG compatibility bug
- Some resumed checkpoints had RNG state types not directly accepted by `torch.set_rng_state`.
- Added safe conversion/fallback in `set_rng_state` to robustly handle list/ndarray/bytes tensor formats.

### Quick Eval (GPU, level2 protocol)

- Files:
  - `pepflow/outputs/quick_eval_option2_level2_vs_official.json`
  - `pepflow/outputs/quick_eval_option2_level2_vs_official_medium_mix1000.log`
- AAR:
  - official: `46.90%`
  - medium+mix (`iter1000`): `40.80%`
  - delta: `-6.11` percentage points

### Interpretation

- Compared with previous tuned-level2 run (`39.01%`), medium+mix improved to `40.80%` (+1.79 points).
- But it is still below official baseline, so this direction is **partially improved but not yet sufficient**.

---

## 9) Conservative + Mixed Reward (Completed)

### Reason

- Medium+mix improved over tuned-level2 but still lagged official baseline.
- This run tested a more conservative regime to reduce policy drift while keeping mixed reward.

### Config Used

- Script:
  - `pepflow/scripts/run_option2_ddp_level2_conservative_mix_train_auto_4gpu.sh`
- level2 mode:
  - `sample_bb=False`
  - `sample_ang=True`
  - `sample_seq=True`
- GRPO / optimization:
  - `lr=1e-5`
  - `lambda_sup=0.5`
  - `updates_per_rollout=1`
  - `old_sync_interval=1`
  - `clip_eps=0.1`
  - `score_clip=5.0`
  - `group_size=4`
  - `num_steps=20`
- reward:
  - `reward_mode=mix_level2`
  - `reward_w_aar=0.7`
  - `reward_w_ang=0.2`
  - `reward_w_tor=0.1`

### Run Outcome

- Run dir:
  - `pepflow/outputs/option2_grpo_level2_conservative_mix_train4g_learn_angle_2026_05_01__17_17_34/`
- Full training reached:
  - `iter=1000` with `checkpoints/1000.pt`
- Stability summary:
  - no NaN
  - no DDP deadlock
  - `skipped_updates=0` through monitored progression
  - auto-train finished cleanly (`[auto-train] completed successfully`)

### Quick Eval (GPU, level2 protocol)

- Files:
  - `pepflow/outputs/quick_eval_option2_level2_vs_official.json`
  - `pepflow/outputs/quick_eval_option2_level2_vs_official_conservative_mix1000.log`
- AAR:
  - official: `46.90%`
  - conservative+mix (`iter1000`): `45.20%`
  - delta: `-1.70` percentage points

### Comparison vs Earlier Level2 Runs

- tuned-level2 (`39.01%`) -> conservative+mix (`45.20%`): `+6.19` points
- medium+mix (`40.80%`) -> conservative+mix (`45.20%`): `+4.40` points

### Interpretation

- Conservative + mixed reward is currently the best level2 variant tested so far.
- It still underperforms official baseline slightly, but the gap is now much smaller (`-1.70`).

---

## 10) Latest Update (2026-05-03): Aggressive Level2 (Level1-Style) Beats Baseline

### Setting

- Script:
  - `pepflow/scripts/run_option2_ddp_level2_level1style_train_auto_4gpu.sh`
- level2 mode:
  - `sample_bb=False`
  - `sample_ang=True`
  - `sample_seq=True`
- Hyperparameters (aligned with previously strong level1 style):
  - `lr=1e-4`
  - `lambda_sup=0.1`
  - `updates_per_rollout=2`
  - `old_sync_interval=16`
  - `group_size=4`
  - `num_steps=20`
  - `clip_eps=0.2`
  - `score_clip=20.0`
  - `reward_mode=aar`
  - `max_iters=1200`

### Training Outcome

- Run dir:
  - `pepflow/outputs/option2_grpo_level2_level1style_train4g_learn_angle_2026_05_03__05_20_40/`
- Completion:
  - reached `iter=1200` and exited successfully
  - no NaN / no deadlock
  - `skipped_updates=0` through monitored progression

### Quick Eval (GPU, level2 protocol)

- Files:
  - `pepflow/outputs/quick_eval_option2_level2_vs_official_level1style1200_fix.log`
  - `pepflow/outputs/quick_eval_option2_level2_vs_official_level1style1200_fix.json`
- AAR:
  - official: `46.90%`
  - level2 level1-style (`iter1200`): `48.70%`
  - delta: `+1.79` percentage points (about `+3.82%` relative)

### Interpretation

- This is the **first confirmed level2 run that exceeds official checkpoint** in our quick protocol.
- Current best level2 setting is now this aggressive level1-style configuration.

---

## 11) Repository Cleanup (Attempts Archive)

To reduce clutter and keep reproducible entrypoints clear:

- Created:
  - `pepflow/scripts/attempts/`
  - `pepflow/outputs/attempts/`
- Kept at top-level `pepflow/scripts/` (best/active):
  - `run_option2_ddp_train_4gpu.sh` (best level1 entry)
  - `run_option2_ddp_level2_level1style_train_auto_4gpu.sh` (best level2 entry)
  - quick eval utilities (`run_quick_eval_option2*.sh`)
- Moved non-best training/smoke scripts into:
  - `pepflow/scripts/attempts/`
- Moved non-best run directories, smoke outputs, and older eval/sweep artifacts into:
  - `pepflow/outputs/attempts/`

Current top-level outputs are now focused on best checkpoints and their direct eval artifacts.
