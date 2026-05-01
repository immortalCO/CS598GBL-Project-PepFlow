#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import os
import gc
import glob
import json
import time
from copy import deepcopy

import numpy as np
import torch

from pepflow.utils.misc import load_config, seed_all
from pepflow.utils.train import recursive_to
from pepflow.utils.data import PaddingCollate
from models_con.pep_dataloader import PepDataset
from models_con.flow_model import FlowModel
from models_con.utils import process_dic


def find_latest_ckpt():
    env_ckpt = os.environ.get("NEW_CKPT", "").strip()
    if env_ckpt:
        if not os.path.exists(env_ckpt):
            raise FileNotFoundError(f"NEW_CKPT does not exist: {env_ckpt}")
        return env_ckpt

    run_dirs = sorted(glob.glob("./outputs/option2_grpo_level2_seqang_train4g_*"))
    if not run_dirs:
        raise FileNotFoundError("No level2 run dir found under ./outputs/option2_grpo_level2_seqang_train4g_*")
    latest_run = run_dirs[-1]
    ckpt_dir = os.path.join(latest_run, "checkpoints")
    ckpts = []
    for path in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        name = os.path.basename(path).replace(".pt", "")
        if name.isdigit():
            ckpts.append((int(name), path))
    if not ckpts:
        raise FileNotFoundError(f"No numeric checkpoint found under: {ckpt_dir}")
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def evaluate_ckpt(label, ckpt_path, cfg, dataset, device, num_steps, num_samples):
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = process_dic(state)

    model = FlowModel(cfg.model).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    collate_fn = PaddingCollate(eight=False)
    metrics = {
        "tran": [],
        "rot": [],
        "aar": [],
        "trans_loss": [],
        "rot_loss": [],
        "len": [],
    }

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            data_list = [deepcopy(item) for _ in range(num_samples)]
            batch = recursive_to(collate_fn(data_list), device)

            loss_dic = model(batch)
            traj = model.sample(
                batch,
                num_steps=num_steps,
                sample_bb=False,
                sample_ang=True,
                sample_seq=True,
            )
            final = traj[-1]

            work_device = batch["generate_mask"].device
            gen_mask = batch["generate_mask"].long()
            trans = final["trans"].to(work_device)
            trans_1 = final["trans_1"].to(work_device)
            rotmats = final["rotmats"].to(work_device)
            rotmats_1 = final["rotmats_1"].to(work_device)
            seqs = final["seqs"].to(work_device)
            seqs_1 = final["seqs_1"].to(work_device)
            denom = gen_mask.sum() + 1e-8
            ca_dist = torch.sqrt(
                torch.sum((trans - trans_1) ** 2 * gen_mask[..., None]) / denom
            )
            rot_dist = torch.sqrt(
                torch.sum((rotmats - rotmats_1) ** 2 * gen_mask[..., None, None]) / denom
            )
            aar = torch.sum((seqs == seqs_1).long() * gen_mask) / denom

            metrics["tran"].append(float(ca_dist.item()))
            metrics["rot"].append(float(rot_dist.item()))
            metrics["aar"].append(float(aar.item()))
            metrics["trans_loss"].append(float(loss_dic["trans_loss"].item()))
            metrics["rot_loss"].append(float(loss_dic["rot_loss"].item()))
            metrics["len"].append(int(gen_mask.sum().item() / num_samples))

            del (
                batch,
                loss_dic,
                traj,
                final,
                work_device,
                gen_mask,
                trans,
                trans_1,
                rotmats,
                rotmats_1,
                seqs,
                seqs_1,
                ca_dist,
                rot_dist,
                aar,
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 20 == 0:
                print(f"[{label}] processed {i + 1}/{len(dataset)}", flush=True)

    summary = {}
    for k, v in metrics.items():
        arr = np.array(v)
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"] = float(arr.std())
    summary["elapsed_sec"] = float(time.time() - t0)

    del model, ckpt, state
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def main():
    seed_all(114514)
    cfg, _ = load_config("./configs/learn_angle.yaml")
    dataset = PepDataset(
        structure_dir=cfg.dataset.val.structure_dir,
        dataset_dir=cfg.dataset.val.dataset_dir,
        name=cfg.dataset.val.name,
        transform=None,
        reset=cfg.dataset.val.reset,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_steps = 20
    num_samples = 10
    latest_level2_ckpt = find_latest_ckpt()

    ckpts = [
        ("official_model1_level2_sampling", "./ckpts/model1.pt"),
        ("option2_level2_latest", latest_level2_ckpt),
    ]

    results = {
        "settings": {
            "device": device,
            "num_steps": num_steps,
            "num_samples": num_samples,
            "dataset_len": len(dataset),
            "sample_bb": False,
            "sample_ang": True,
            "sample_seq": True,
            "level": 2,
        },
        "models": {},
    }

    print(
        f"[info] device={device} dataset_len={len(dataset)} "
        f"num_steps={num_steps} num_samples={num_samples}",
        flush=True,
    )
    print(f"[info] auto-resolved level2 checkpoint: {latest_level2_ckpt}", flush=True)
    for label, ckpt_path in ckpts:
        print(f"[info] evaluating {label} from {ckpt_path}", flush=True)
        results["models"][label] = evaluate_ckpt(
            label=label,
            ckpt_path=ckpt_path,
            cfg=cfg,
            dataset=dataset,
            device=device,
            num_steps=num_steps,
            num_samples=num_samples,
        )

    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quick_eval_option2_level2_vs_official.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"RESULT_JSON {out_path}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
PY
