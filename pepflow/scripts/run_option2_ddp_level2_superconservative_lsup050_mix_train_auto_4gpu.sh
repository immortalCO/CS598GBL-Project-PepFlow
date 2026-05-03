#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MAX_RESTARTS="${MAX_RESTARTS:-4}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-8}"

OUTPUT_ROOT="./outputs"
EXP_NAME="option2_grpo_level2_superconservative_lsup050_mix_train4g"

COMMON_ARGS=(
  --config ./configs/learn_angle.yaml
  --output_root "${OUTPUT_ROOT}"
  --exp_name "${EXP_NAME}"
  --level 2
  --group_size 4
  --num_steps 20
  --lr 7e-6
  --lambda_sup 0.50
  --updates_per_rollout 1
  --old_sync_interval 1
  --clip_eps 0.08
  --adv_eps 1e-6
  --score_clip 4.0
  --reward_mode mix_level2
  --reward_w_aar 0.7
  --reward_w_ang 0.2
  --reward_w_tor 0.1
  --save_freq 50
  --max_iters 1200
  --num_workers 4
)

attempt=0
resume_ckpt=""

find_latest_numeric_ckpt() {
  local run_dir="$1"
  if [[ ! -d "${run_dir}/checkpoints" ]]; then
    return 1
  fi
  local latest
  latest="$({
    ls "${run_dir}/checkpoints"/*.pt 2>/dev/null \
      | awk -F'/' '{print $NF}' \
      | sed 's/\.pt$//' \
      | awk '/^[0-9]+$/' \
      | sort -n \
      | tail -n 1
  } || true)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf "%s/checkpoints/%s.pt" "${run_dir}" "${latest}"
}

while true; do
  attempt=$((attempt + 1))
  echo "[auto-train] attempt=${attempt} max_restarts=${MAX_RESTARTS} resume_ckpt=${resume_ckpt:-<none>}"

  cmd=(
    torchrun --standalone --nproc_per_node=4 train_ddp_option2.py
    "${COMMON_ARGS[@]}"
  )

  if [[ -n "${resume_ckpt}" ]]; then
    cmd+=(--init_ckpt ./ckpts/model1.pt --resume "${resume_ckpt}")
  else
    cmd+=(--init_ckpt ./ckpts/model1.pt)
  fi

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[auto-train] completed successfully."
    exit 0
  fi

  if (( attempt > MAX_RESTARTS )); then
    echo "[auto-train] failed after ${MAX_RESTARTS} retries. last_rc=${rc}"
    exit "${rc}"
  fi

  latest_run_dir="$(ls -d ${OUTPUT_ROOT}/${EXP_NAME}_* 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${latest_run_dir}" ]]; then
    echo "[auto-train] no run dir found for ${EXP_NAME}; cannot resume."
    exit "${rc}"
  fi

  latest_ckpt="$(find_latest_numeric_ckpt "${latest_run_dir}" || true)"
  if [[ -z "${latest_ckpt}" ]]; then
    echo "[auto-train] run dir found but no numeric checkpoint yet: ${latest_run_dir}"
    echo "[auto-train] sleeping ${RETRY_SLEEP_SEC}s and retrying fresh."
    sleep "${RETRY_SLEEP_SEC}"
    resume_ckpt=""
    continue
  fi

  resume_ckpt="${latest_ckpt}"
  echo "[auto-train] restart from ${resume_ckpt} after ${RETRY_SLEEP_SEC}s ..."
  sleep "${RETRY_SLEEP_SEC}"
done
