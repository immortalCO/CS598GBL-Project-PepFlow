import argparse
import csv
import os
import random
import shutil
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
import yaml

from models_con.flow_model import FlowModel
from models_con.pep_dataloader import PepDataset
from models_con.utils import process_dic
from pepflow.utils.data import PaddingCollate
from pepflow.utils.misc import BlackHole, current_milli_time, get_logger, load_config, seed_all
from pepflow.utils.train import (
    get_optimizer,
    get_scheduler,
    log_losses,
    recursive_to,
    sum_weighted_losses,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEFAULT_OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs")


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/learn_angle.yaml")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help="Local rank for DDP.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--name", type=str, default="pepflow_option2_grpo")

    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exp_name", type=str, default="option2_grpo_seqonly")

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--lambda_sup", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--updates_per_rollout", type=int, default=2)
    parser.add_argument("--old_sync_interval", type=int, default=16)
    parser.add_argument("--adv_eps", type=float, default=1e-6)
    parser.add_argument("--score_clip", type=float, default=20.0)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=None)

    parser.add_argument("--sample_bb", type=str2bool, default=False)
    parser.add_argument("--sample_ang", type=str2bool, default=False)
    parser.add_argument("--sample_seq", type=str2bool, default=True)
    return parser.parse_args()


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _world_size():
    return dist.get_world_size() if _is_dist() else 1


def _rank():
    return dist.get_rank() if _is_dist() else 0


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    if _is_dist():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y = y / _world_size()
    return y


def reduce_min(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    if _is_dist():
        dist.all_reduce(y, op=dist.ReduceOp.MIN)
    return y


def reduce_max(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    if _is_dist():
        dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def reduce_max_int(flag: int, device: torch.device) -> int:
    t = torch.tensor(float(flag), device=device)
    if _is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item() > 0)


def reduce_mean_finite(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clone()
    finite = torch.isfinite(x)
    num = torch.where(finite, x, torch.zeros_like(x))
    den = finite.float()
    if _is_dist():
        dist.all_reduce(num, op=dist.ReduceOp.SUM)
        dist.all_reduce(den, op=dist.ReduceOp.SUM)
    if den.item() > 0:
        return num / den
    return torch.tensor(float("nan"), device=x.device)


def broadcast_str(value: str, src: int = 0) -> str:
    if not _is_dist():
        return value
    obj_list = [value if _rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def repeat_batch(batch, group_size: int):
    bsz = batch["aa"].shape[0]
    idx = torch.arange(bsz, device=batch["aa"].device).repeat_interleave(group_size)
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.index_select(0, idx)
        elif isinstance(v, list):
            out[k] = [deepcopy(v[i]) for i in idx.tolist()]
        elif isinstance(v, tuple):
            out[k] = tuple(deepcopy(v[i]) for i in idx.tolist())
        else:
            out[k] = deepcopy(v)
    return out


def sync_model_old_from_current(model: DDP, model_old: FlowModel):
    model_old.load_state_dict(model.module.state_dict(), strict=True)
    model_old.eval()
    for p in model_old.parameters():
        p.requires_grad_(False)


def _normalize_model_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        raise ValueError("State dict must be a dictionary.")
    if any(k.startswith("module.") for k in state_dict.keys()):
        return process_dic(state_dict)
    return state_dict


def load_weights_to_model(model: DDP, state_dict: dict):
    normalized = _normalize_model_state_dict(state_dict)
    model.module.load_state_dict(normalized, strict=True)


def extract_model_state_from_ckpt(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def save_args_yaml(args, path: str):
    args_dict = vars(args)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(args_dict, f, sort_keys=True, allow_unicode=True)


def compute_aar_reward(sampled_seqs, gt_seqs, generate_mask, eps=1e-8):
    matches = (sampled_seqs == gt_seqs).float() * generate_mask.float()
    per_sample = matches.sum(dim=-1) / (generate_mask.float().sum(dim=-1) + eps)
    return per_sample


def compute_group_advantages(rewards, batch_size, group_size, eps):
    r = rewards.view(batch_size, group_size)
    mu = r.mean(dim=-1, keepdim=True)
    sigma = r.std(dim=-1, keepdim=True, unbiased=False)
    adv = (r - mu) / (sigma + eps)
    return adv.reshape(-1), mu.reshape(-1), sigma.reshape(-1)


def compute_seq_logprob_score(model_core: FlowModel, batch):
    rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = model_core.encode(batch)
    num_batch = batch["aa"].shape[0]
    t = torch.ones((num_batch, 1), device=batch["aa"].device)
    pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_logits = model_core.ga_encoder(
        t,
        rotmats_1,
        trans_1,
        angles_1,
        seqs_1,
        node_embed,
        edge_embed,
        batch["generate_mask"].long(),
        batch["res_mask"].long(),
    )
    _ = pred_rotmats_1, pred_trans_1, pred_angles_1  # keep explicit unpack for clarity
    pred_seqs_1_logits = torch.nan_to_num(pred_seqs_1_logits, nan=0.0, posinf=20.0, neginf=-20.0)
    log_probs = F.log_softmax(pred_seqs_1_logits, dim=-1)
    tgt = torch.clamp(seqs_1.long(), 0, pred_seqs_1_logits.shape[-1] - 1)
    token_logp = torch.gather(log_probs, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    gen_mask = batch["generate_mask"].float()
    score = (token_logp * gen_mask).sum(dim=-1) / (gen_mask.sum(dim=-1) + 1e-8)
    return score


def get_rng_state(local_rank):
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(local_rank),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state, local_rank):
    if state is None:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device=local_rank)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def setup_run_dir(args, config_name):
    if args.resume:
        ckpt_dir = os.path.dirname(args.resume)
        run_dir = os.path.dirname(ckpt_dir)
    else:
        ts = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
        tag = f"_{args.tag}" if args.tag else ""
        run_name = f"{args.exp_name}_{config_name}_{ts}{tag}"
        run_dir = os.path.join(args.output_root, run_name)
    return run_dir


def main():
    args = parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank if args.local_rank is not None else 0))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = _rank()
    is_main = rank == 0

    config, config_name = load_config(args.config)
    seed_all(config.train.seed + local_rank * 100)
    if args.max_iters is not None:
        config.train.max_iters = int(args.max_iters)

    run_dir_local = setup_run_dir(args, config_name) if is_main else ""
    run_dir = broadcast_str(run_dir_local, src=0)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    metrics_csv = os.path.join(run_dir, "train_metrics.csv")

    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    if _is_dist():
        dist.barrier()

    logger = get_logger("train_option2", run_dir if is_main else None, local_rank)
    writer = BlackHole()

    if is_main and (not args.debug):
        wandb.init(project=args.name, config=config, name=os.path.basename(run_dir))

    if is_main:
        save_args_yaml(args, os.path.join(run_dir, "args.yaml"))
        if os.path.exists(args.config):
            shutil.copyfile(args.config, os.path.join(run_dir, os.path.basename(args.config)))
    if _is_dist():
        dist.barrier()

    logger.info(args)
    logger.info(config)
    logger.info(f"Using run_dir: {run_dir}")
    logger.info(
        "Sampling mode fixed for v1: sample_bb=%s sample_ang=%s sample_seq=%s",
        args.sample_bb,
        args.sample_ang,
        args.sample_seq,
    )

    train_dataset = PepDataset(
        structure_dir=config.dataset.train.structure_dir,
        dataset_dir=config.dataset.train.dataset_dir,
        name=config.dataset.train.name,
        transform=None,
        reset=config.dataset.train.reset,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_sampler.set_epoch(0)
    train_iter = iter(train_loader)
    sampler_epoch = 0

    model = DDP(FlowModel(config.model).to(local_rank), device_ids=[local_rank])
    model_old = FlowModel(config.model).to(local_rank)
    model_old.eval()
    for p in model_old.parameters():
        p.requires_grad_(False)

    optimizer = get_optimizer(config.train.optimizer, model)
    for pg in optimizer.param_groups:
        pg["lr"] = args.lr
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    successful_updates = 0

    it_first = 1
    if args.resume is not None:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}", weights_only=False)
        model_state = extract_model_state_from_ckpt(ckpt)
        load_weights_to_model(model, model_state)
        if "model_old" in ckpt and isinstance(ckpt["model_old"], dict):
            model_old.load_state_dict(_normalize_model_state_dict(ckpt["model_old"]), strict=True)
        else:
            sync_model_old_from_current(model, model_old)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        it_first = int(ckpt["iteration"]) + 1
        set_rng_state(ckpt.get("rng_state", None), local_rank)
        successful_updates = int(ckpt.get("successful_updates", 0))
    else:
        logger.info(f"Loading init checkpoint from {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location=f"cuda:{local_rank}", weights_only=False)
        model_state = extract_model_state_from_ckpt(ckpt)
        try:
            load_weights_to_model(model, model_state)
        except RuntimeError:
            load_weights_to_model(model, process_dic(model_state))
        sync_model_old_from_current(model, model_old)

    if _is_dist():
        dist.barrier()

    scheduler_type = getattr(config.train.scheduler, "type", None)
    if scheduler_type == "plateau" and is_main:
        logger.warning("Scheduler type is plateau; skipping scheduler.step() because no validation loop is used.")

    metric_fields = [
        "iter",
        "loss_total",
        "loss_grpo",
        "loss_sup",
        "reward_mean",
        "reward_std",
        "reward_max",
        "adv_mean",
        "adv_std",
        "rho_mean",
        "rho_min",
        "rho_max",
        "score_delta_mean",
        "score_delta_abs_mean",
        "grad_norm",
        "lr",
        "successful_updates",
        "skipped_updates",
        "time_rollout",
        "time_update",
    ]
    if is_main and (not os.path.exists(metrics_csv)):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=metric_fields)
            writer_csv.writeheader()

    def fetch_batch():
        nonlocal train_iter, sampler_epoch
        try:
            b = next(train_iter)
        except StopIteration:
            sampler_epoch += 1
            train_sampler.set_epoch(sampler_epoch)
            train_iter = iter(train_loader)
            b = next(train_iter)
        return recursive_to(b, local_rank)

    def save_checkpoint(it):
        if not is_main:
            return
        path = os.path.join(ckpt_dir, f"{it}.pt")
        torch.save(
            {
                "config": config,
                "args": vars(args),
                "model": model.module.state_dict(),
                "model_old": model_old.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "iteration": it,
                "successful_updates": successful_updates,
                "rng_state": get_rng_state(local_rank),
                "grpo_state": {
                    "group_size": args.group_size,
                    "num_steps": args.num_steps,
                    "clip_eps": args.clip_eps,
                    "lambda_sup": args.lambda_sup,
                    "lr": args.lr,
                    "updates_per_rollout": args.updates_per_rollout,
                    "old_sync_interval": args.old_sync_interval,
                    "adv_eps": args.adv_eps,
                    "score_clip": args.score_clip,
                    "sampling_mode": {
                        "sample_bb": args.sample_bb,
                        "sample_ang": args.sample_ang,
                        "sample_seq": args.sample_seq,
                    },
                },
                "run_dir": run_dir,
            },
            path,
        )

    for it in range(it_first, config.train.max_iters + 1):
        model.train()
        model_old.eval()
        iter_start = current_milli_time()

        batch = fetch_batch()
        bsz = batch["aa"].shape[0]
        device = batch["aa"].device
        expanded_batch = repeat_batch(batch, args.group_size)

        with torch.no_grad():
            traj = model_old.sample(
                expanded_batch,
                num_steps=args.num_steps,
                sample_bb=args.sample_bb,
                sample_ang=args.sample_ang,
                sample_seq=args.sample_seq,
            )
            final = traj[-1]
            num_classes = int(model.module._interpolant_cfg.seqs.num_classes)
            sampled_seqs = final["seqs"].to(local_rank).long().clamp(min=0, max=num_classes - 1)
            gt_seqs = final["seqs_1"].to(local_rank).long().clamp(min=0, max=num_classes - 1)

            rewards = compute_aar_reward(
                sampled_seqs=sampled_seqs,
                gt_seqs=gt_seqs,
                generate_mask=expanded_batch["generate_mask"],
            )
            adv, _, _ = compute_group_advantages(
                rewards=rewards,
                batch_size=bsz,
                group_size=args.group_size,
                eps=args.adv_eps,
            )
            adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)

        rollout_end = current_milli_time()

        score_batch = repeat_batch(batch, args.group_size)
        score_batch["aa"] = sampled_seqs
        score_batch["aa"] = torch.where(
            score_batch["generate_mask"],
            score_batch["aa"],
            batch["aa"].repeat_interleave(args.group_size, dim=0),
        )

        with torch.no_grad():
            s_old = compute_seq_logprob_score(model_old, score_batch)
            s_old = torch.nan_to_num(s_old, nan=0.0, posinf=0.0, neginf=0.0)

        update_losses = []
        update_grpo_losses = []
        update_sup_losses = []
        update_rho_means = []
        update_rho_mins = []
        update_rho_maxs = []
        update_score_delta_means = []
        update_score_delta_abs_means = []
        update_grad_norms = []
        skipped_updates = 0

        for _ in range(max(1, int(args.updates_per_rollout))):
            s_theta = compute_seq_logprob_score(model.module, score_batch)
            s_theta = torch.nan_to_num(s_theta, nan=0.0, posinf=0.0, neginf=0.0)
            score_delta = s_theta - s_old
            log_ratio = torch.clamp(score_delta, min=-args.score_clip, max=args.score_clip)
            rho = torch.exp(log_ratio)
            rho = torch.nan_to_num(rho, nan=1.0, posinf=1.0 + args.clip_eps, neginf=1.0 - args.clip_eps)
            clipped_rho = torch.clamp(rho, 1.0 - args.clip_eps, 1.0 + args.clip_eps)

            surrogate_unclipped = rho * adv
            surrogate_clipped = clipped_rho * adv
            l_grpo = -torch.mean(torch.min(surrogate_unclipped, surrogate_clipped))

            sup_loss_dict = model(batch)
            l_sup = sum_weighted_losses(sup_loss_dict, config.train.loss_weights)
            loss = l_grpo + args.lambda_sup * l_sup

            local_bad_loss = int(not torch.isfinite(loss).item())
            bad_loss = reduce_max_int(local_bad_loss, device=device)

            optimizer.zero_grad()
            grad_norm_tensor = torch.tensor(float("nan"), device=device)

            if bad_loss:
                skipped_updates += 1
            else:
                loss.backward()

                local_bad_grad = 0
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    if not torch.isfinite(param.grad).all():
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        local_bad_grad = 1
                bad_grad = reduce_max_int(local_bad_grad, device=device)

                if bad_grad:
                    skipped_updates += 1
                    optimizer.zero_grad()
                else:
                    grad_norm = clip_grad_norm_(
                        model.parameters(),
                        config.train.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm_tensor = grad_norm.detach()
                    else:
                        grad_norm_tensor = torch.tensor(float(grad_norm), device=device)
                    bad_norm = reduce_max_int(int(not torch.isfinite(grad_norm_tensor).item()), device=device)
                    if bad_norm:
                        skipped_updates += 1
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        successful_updates += 1
                        if scheduler_type not in (None, "plateau"):
                            scheduler.step()
                        if successful_updates % max(1, int(args.old_sync_interval)) == 0:
                            sync_model_old_from_current(model, model_old)

            update_losses.append(torch.nan_to_num(loss.detach(), nan=0.0, posinf=0.0, neginf=0.0))
            update_grpo_losses.append(torch.nan_to_num(l_grpo.detach(), nan=0.0, posinf=0.0, neginf=0.0))
            update_sup_losses.append(torch.nan_to_num(l_sup.detach(), nan=0.0, posinf=0.0, neginf=0.0))
            update_rho_means.append(torch.nan_to_num(rho.mean().detach(), nan=1.0, posinf=1.0, neginf=1.0))
            update_rho_mins.append(torch.nan_to_num(rho.min().detach(), nan=1.0, posinf=1.0, neginf=1.0))
            update_rho_maxs.append(torch.nan_to_num(rho.max().detach(), nan=1.0, posinf=1.0, neginf=1.0))
            update_score_delta_means.append(torch.nan_to_num(score_delta.mean().detach(), nan=0.0, posinf=0.0, neginf=0.0))
            update_score_delta_abs_means.append(torch.nan_to_num(score_delta.abs().mean().detach(), nan=0.0, posinf=0.0, neginf=0.0))
            update_grad_norms.append(grad_norm_tensor)

        update_end = current_milli_time()

        reward_mean = reduce_mean(rewards.mean())
        reward_std = reduce_mean(rewards.std(unbiased=False))
        reward_max = reduce_max(rewards.max())
        adv_mean = reduce_mean(adv.mean())
        adv_std = reduce_mean(adv.std(unbiased=False))

        loss_grpo_mean = reduce_mean(torch.stack(update_grpo_losses).mean())
        loss_sup_mean = reduce_mean(torch.stack(update_sup_losses).mean())
        loss_total_mean = reduce_mean(torch.stack(update_losses).mean())
        rho_mean = reduce_mean(torch.stack(update_rho_means).mean())
        rho_min = reduce_min(torch.stack(update_rho_mins).min())
        rho_max = reduce_max(torch.stack(update_rho_maxs).max())
        score_delta_mean = reduce_mean(torch.stack(update_score_delta_means).mean())
        score_delta_abs_mean = reduce_mean(torch.stack(update_score_delta_abs_means).mean())

        grad_norm_stack = torch.stack(update_grad_norms)
        finite_grad_norm = grad_norm_stack[torch.isfinite(grad_norm_stack)]
        if finite_grad_norm.numel() > 0:
            grad_norm_local = finite_grad_norm.mean()
        else:
            grad_norm_local = torch.tensor(float("nan"), device=device)
        grad_norm_mean = reduce_mean_finite(grad_norm_local)

        time_rollout = torch.tensor((rollout_end - iter_start) / 1000.0, device=device)
        time_update = torch.tensor((update_end - rollout_end) / 1000.0, device=device)
        time_rollout = reduce_mean(time_rollout)
        time_update = reduce_mean(time_update)

        skipped_updates_t = torch.tensor(float(skipped_updates), device=device)
        skipped_updates_mean = reduce_mean(skipped_updates_t)

        if is_main:
            scalar_dict = {
                "grad_norm": grad_norm_mean,
                "lr": optimizer.param_groups[0]["lr"],
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_max": reward_max,
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "rho_mean": rho_mean,
                "rho_min": rho_min,
                "rho_max": rho_max,
                "score_delta_mean": score_delta_mean,
                "score_delta_abs_mean": score_delta_abs_mean,
                "time_rollout": time_rollout,
                "time_update": time_update,
                "successful_updates": successful_updates,
                "skipped_updates": skipped_updates_mean,
            }
            loss_dict_for_log = {"grpo": loss_grpo_mean, "sup": loss_sup_mean}
            log_losses(loss_total_mean, loss_dict_for_log, scalar_dict, it=it, tag="train", logger=logger, writer=writer)

            row = {
                "iter": it,
                "loss_total": float(loss_total_mean.item()),
                "loss_grpo": float(loss_grpo_mean.item()),
                "loss_sup": float(loss_sup_mean.item()),
                "reward_mean": float(reward_mean.item()),
                "reward_std": float(reward_std.item()),
                "reward_max": float(reward_max.item()),
                "adv_mean": float(adv_mean.item()),
                "adv_std": float(adv_std.item()),
                "rho_mean": float(rho_mean.item()),
                "rho_min": float(rho_min.item()),
                "rho_max": float(rho_max.item()),
                "score_delta_mean": float(score_delta_mean.item()),
                "score_delta_abs_mean": float(score_delta_abs_mean.item()),
                "grad_norm": float(grad_norm_mean.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "successful_updates": int(successful_updates),
                "skipped_updates": float(skipped_updates_mean.item()),
                "time_rollout": float(time_rollout.item()),
                "time_update": float(time_update.item()),
            }
            with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                writer_csv = csv.DictWriter(f, fieldnames=metric_fields)
                writer_csv.writerow(row)

            if it % args.save_freq == 0:
                save_checkpoint(it)

    if is_main:
        save_checkpoint(config.train.max_iters)

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
