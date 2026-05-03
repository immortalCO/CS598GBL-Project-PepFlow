#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

torchrun --standalone --nproc_per_node=4 train_ddp_option2.py \
  --config ./configs/learn_angle_smoke.yaml \
  --init_ckpt ./ckpts/model1.pt \
  --output_root ./outputs \
  --exp_name option2_grpo_level2_medium_mix_smoke4g \
  --level 2 \
  --group_size 6 \
  --num_steps 16 \
  --lr 2e-5 \
  --lambda_sup 0.4 \
  --updates_per_rollout 1 \
  --old_sync_interval 2 \
  --clip_eps 0.1 \
  --adv_eps 1e-6 \
  --score_clip 5.0 \
  --reward_mode mix_level2 \
  --reward_w_aar 0.7 \
  --reward_w_ang 0.2 \
  --reward_w_tor 0.1 \
  --max_iters 2 \
  --save_freq 1 \
  --num_workers 2
