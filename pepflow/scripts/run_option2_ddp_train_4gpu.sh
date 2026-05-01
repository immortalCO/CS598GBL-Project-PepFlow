#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Reduce allocator fragmentation for large trajectory tensors.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# 4-GPU normal training (Balanced defaults from plan).
torchrun --standalone --nproc_per_node=4 train_ddp_option2.py \
  --config ./configs/learn_angle.yaml \
  --init_ckpt ./ckpts/model1.pt \
  --output_root ./outputs \
  --exp_name option2_grpo_seqonly_train4g \
  --group_size 4 \
  --num_steps 20 \
  --lr 1e-4 \
  --lambda_sup 0.1 \
  --updates_per_rollout 2 \
  --old_sync_interval 16 \
  --clip_eps 0.2 \
  --adv_eps 1e-6 \
  --score_clip 20.0 \
  --save_freq 100 \
  --num_workers 4 \
  --sample_bb false \
  --sample_ang false \
  --sample_seq true
