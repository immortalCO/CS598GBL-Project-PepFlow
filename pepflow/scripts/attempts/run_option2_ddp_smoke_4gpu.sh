#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Reduce allocator fragmentation for large trajectory tensors.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional knobs for deterministic-ish smoke behavior.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# 4-GPU fast smoke test.
torchrun --standalone --nproc_per_node=4 train_ddp_option2.py \
  --config ./configs/learn_angle_smoke.yaml \
  --init_ckpt ./ckpts/model1.pt \
  --output_root ./outputs \
  --exp_name option2_grpo_seqonly_smoke4g \
  --level 1 \
  --group_size 2 \
  --num_steps 10 \
  --lr 1e-4 \
  --lambda_sup 0.1 \
  --updates_per_rollout 1 \
  --old_sync_interval 4 \
  --clip_eps 0.2 \
  --adv_eps 1e-6 \
  --score_clip 20.0 \
  --max_iters 2 \
  --save_freq 1 \
  --num_workers 2
