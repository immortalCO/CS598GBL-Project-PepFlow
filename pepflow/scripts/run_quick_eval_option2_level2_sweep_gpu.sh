#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import gc
import glob
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from pepflow.utils.misc import load_config, seed_all
from pepflow.utils.train import recursive_to
from pepflow.utils.data import PaddingCollate
from models_con.pep_dataloader import PepDataset
from models_con.flow_model import FlowModel
from models_con.utils import process_dic


def resolve_run_dir():
    env_dir = os.environ.get("SWEEP_RUN_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        if not p.exists():
            raise FileNotFoundError(f"SWEEP_RUN_DIR not found: {env_dir}")
        return p

    patterns = [
        "./outputs/option2_grpo_level2_superconservative_lsup050_mix_train4g_*",
        "./outputs/option2_grpo_level2_superconservative_mix_train4g_*",
    ]
    run_dirs = []
    for pat in patterns:
        run_dirs.extend(glob.glob(pat))
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise FileNotFoundError("No matching level2 run dirs found for sweep.")
    return Path(run_dirs[-1])


def resolve_steps():
    steps_env = os.environ.get("SWEEP_STEPS", "1000,1050,1100,1150,1200").strip()
    steps = []
    for s in steps_env.split(","):
        s = s.strip()
        if not s:
            continue
        if not s.isdigit():
            raise ValueError(f"Invalid checkpoint step in SWEEP_STEPS: {s}")
        steps.append(int(s))
    if not steps:
        raise ValueError("No checkpoint steps resolved for sweep.")
    return steps


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
            ca_dist = torch.sqrt(torch.sum((trans - trans_1) ** 2 * gen_mask[..., None]) / denom)
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

    run_dir = resolve_run_dir()
    steps = resolve_steps()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_paths = []
    for step in steps:
        p = ckpt_dir / f"{step}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ckpt_paths.append((step, str(p)))

    print(f"[info] device={device} run_dir={run_dir}", flush=True)
    print(f"[info] sweep_steps={steps}", flush=True)

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
            "run_dir": str(run_dir),
            "sweep_steps": steps,
        },
        "models": {},
        "sweep": [],
    }

    official = evaluate_ckpt(
        label="official_model1_level2_sampling",
        ckpt_path="./ckpts/model1.pt",
        cfg=cfg,
        dataset=dataset,
        device=device,
        num_steps=num_steps,
        num_samples=num_samples,
    )
    results["models"]["official_model1_level2_sampling"] = official
    official_aar = official["aar_mean"]
    print(f"[info] official aar_mean={official_aar:.6f}", flush=True)

    best_step = None
    best_aar = -1.0
    for step, ckpt_path in ckpt_paths:
        label = f"option2_step_{step}"
        print(f"[info] evaluating {label} from {ckpt_path}", flush=True)
        model_result = evaluate_ckpt(
            label=label,
            ckpt_path=ckpt_path,
            cfg=cfg,
            dataset=dataset,
            device=device,
            num_steps=num_steps,
            num_samples=num_samples,
        )
        results["models"][label] = model_result
        aar = model_result["aar_mean"]
        delta = aar - official_aar
        results["sweep"].append(
            {
                "step": step,
                "ckpt_path": ckpt_path,
                "aar_mean": aar,
                "delta_vs_official": delta,
                "trans_loss_mean": model_result["trans_loss_mean"],
                "rot_loss_mean": model_result["rot_loss_mean"],
            }
        )
        print(
            f"[sweep] step={step} aar_mean={aar:.6f} delta_vs_official={delta:+.6f}",
            flush=True,
        )
        if aar > best_aar:
            best_aar = aar
            best_step = step

    results["best"] = {
        "step": best_step,
        "aar_mean": best_aar,
        "delta_vs_official": best_aar - official_aar,
    }

    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name
    out_json = out_dir / f"quick_eval_option2_level2_sweep_{run_name}.json"
    out_txt = out_dir / f"quick_eval_option2_level2_sweep_{run_name}.txt"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"official_aar_mean: {official_aar:.6f}")
    lines.append("sweep_results:")
    for item in results["sweep"]:
        lines.append(
            f"  step={item['step']}, aar_mean={item['aar_mean']:.6f}, "
            f"delta_vs_official={item['delta_vs_official']:+.6f}"
        )
    lines.append(
        f"best_step={best_step}, best_aar_mean={best_aar:.6f}, "
        f"best_delta_vs_official={best_aar - official_aar:+.6f}"
    )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"RESULT_SWEEP_JSON {out_json}", flush=True)
    print(f"RESULT_SWEEP_TXT {out_txt}", flush=True)
    print(json.dumps(results["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
PY
