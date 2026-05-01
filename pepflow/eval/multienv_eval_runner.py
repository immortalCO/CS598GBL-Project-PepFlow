#!/usr/bin/env python3
"""
Run PepFlow evaluation across multiple conda environments.

Typical usage (from repo root):
  python eval/multienv_eval_runner.py \
    --sample-dir /path/to/SAMPLE_DIR \
    --mode orchestrate
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_MAIN_ENV = "pepflow"
DEFAULT_ENERGY_ENV = "pepflow-eval-energy"


def _run_bash(cmd: str, cwd: Path, dry_run: bool = False) -> None:
    print(f"[RUN] {cmd}", flush=True)
    if dry_run:
        return
    subprocess.run(["bash", "-lc", cmd], cwd=str(cwd), check=True)


def _run_in_conda(env_name: str, inner_cmd: str, cwd: Path, dry_run: bool = False) -> None:
    # Use `bash -c` (not `-lc`): login shells can reset PATH and bypass conda env python.
    cmd = f"conda run --no-capture-output -n {shlex.quote(env_name)} bash -c {shlex.quote(inner_cmd)}"
    _run_bash(cmd, cwd=cwd, dry_run=dry_run)


def _discover_target_dirs(sample_dir: Path) -> List[Path]:
    pdb_root = sample_dir / "pdbs"
    if pdb_root.is_dir():
        candidates = [p for p in sorted(pdb_root.iterdir()) if p.is_dir()]
        if candidates:
            return candidates

    # Fallback: direct subfolders under sample_dir.
    return [p for p in sorted(sample_dir.iterdir()) if p.is_dir() and (p / "gt.pdb").is_file()]


def _iter_sample_pdbs(target_dir: Path) -> List[Path]:
    sample_files = sorted(target_dir.glob("sample_*.pdb"))
    if sample_files:
        return sample_files
    # Fallback: any pdb except gt.
    return sorted([p for p in target_dir.glob("*.pdb") if p.name != "gt.pdb"])


def _safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _mode_geometry(sample_dir: Path, chain_id: str, max_samples_per_target: int | None) -> None:
    from geometry import (
        get_bind_ratio,
        get_chain_from_pdb,
        get_rmsd,
        get_ss,
        get_tm,
        get_traj_chain,
    )

    target_dirs = _discover_target_dirs(sample_dir)
    if not target_dirs:
        raise FileNotFoundError(f"No target folders found under: {sample_dir}")

    out_csv = sample_dir / "eval_geometry.csv"
    rows = []

    for target in target_dirs:
        gt_pdb = target / "gt.pdb"
        if not gt_pdb.is_file():
            continue

        sample_pdbs = _iter_sample_pdbs(target)
        if max_samples_per_target is not None:
            sample_pdbs = sample_pdbs[: max_samples_per_target]

        for sample_pdb in sample_pdbs:
            row = {
                "target": target.name,
                "sample_pdb": str(sample_pdb),
                "gt_pdb": str(gt_pdb),
                "chain_id": chain_id,
                "rmsd_raw": None,
                "rmsd_sup": None,
                "tm": None,
                "ss_match": None,
                "bind_ratio": None,
                "error": "",
            }
            try:
                chain_pred = get_chain_from_pdb(str(sample_pdb), chain_id=chain_id)
                chain_gt = get_chain_from_pdb(str(gt_pdb), chain_id=chain_id)
                if chain_pred is None or chain_gt is None:
                    raise RuntimeError(f"Chain '{chain_id}' not found in sample/gt")

                rmsd_raw, rmsd_sup = get_rmsd(chain_pred, chain_gt)
                tm_score = get_tm(chain_pred, chain_gt)

                try:
                    traj_pred = get_traj_chain(str(sample_pdb), chain_id)
                    traj_gt = get_traj_chain(str(gt_pdb), chain_id)
                    ss_match = float(get_ss(traj_pred, traj_gt))
                except Exception:
                    ss_match = None

                try:
                    bind_ratio = float(get_bind_ratio(str(sample_pdb), str(gt_pdb), chain_id, chain_id))
                except Exception:
                    bind_ratio = None

                row.update(
                    {
                        "rmsd_raw": float(rmsd_raw),
                        "rmsd_sup": float(rmsd_sup),
                        "tm": float(tm_score),
                        "ss_match": ss_match,
                        "bind_ratio": bind_ratio,
                    }
                )
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "sample_pdb",
                "gt_pdb",
                "chain_id",
                "rmsd_raw",
                "rmsd_sup",
                "tm",
                "ss_match",
                "bind_ratio",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Geometry CSV written: {out_csv}")
    print(f"[INFO] rows={len(rows)}")
    print(f"[INFO] mean_rmsd_sup={_safe_mean(r['rmsd_sup'] for r in rows)}")
    print(f"[INFO] mean_tm={_safe_mean(r['tm'] for r in rows)}")


def _mode_energy(sample_dir: Path, chain_id: str, max_samples_per_target: int | None) -> None:
    try:
        from energy import get_rosetta_score
    except Exception as exc:
        raise RuntimeError(
            "Failed to import eval/energy.py. Install PyRosetta in this env first, e.g.:\n"
            "python -c \"import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()\""
        ) from exc

    target_dirs = _discover_target_dirs(sample_dir)
    if not target_dirs:
        raise FileNotFoundError(f"No target folders found under: {sample_dir}")

    out_csv = sample_dir / "eval_energy_rosetta.csv"
    rows = []

    for target in target_dirs:
        sample_pdbs = _iter_sample_pdbs(target)
        if max_samples_per_target is not None:
            sample_pdbs = sample_pdbs[: max_samples_per_target]

        for sample_pdb in sample_pdbs:
            row = {
                "target": target.name,
                "sample_pdb": str(sample_pdb),
                "chain_id": chain_id,
                "stability_score": None,
                "dg_separated": None,
                "error": "",
            }
            try:
                _, stab, dg = get_rosetta_score(str(sample_pdb), chain=chain_id)
                row["stability_score"] = float(stab)
                row["dg_separated"] = float(dg)
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "sample_pdb",
                "chain_id",
                "stability_score",
                "dg_separated",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Energy CSV written: {out_csv}")
    print(f"[INFO] rows={len(rows)}")
    print(f"[INFO] mean_stability={_safe_mean(r['stability_score'] for r in rows)}")
    print(f"[INFO] mean_dg={_safe_mean(r['dg_separated'] for r in rows)}")


def _mode_orchestrate(args: argparse.Namespace, script_path: Path, repo_root: Path) -> None:
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not sample_dir.exists():
        raise FileNotFoundError(f"sample_dir not found: {sample_dir}")

    def run_self_in_env(env_name: str, mode_name: str) -> None:
        cmd = (
            f"python {shlex.quote(str(script_path))}"
            f" --mode {shlex.quote(mode_name)}"
            f" --sample-dir {shlex.quote(str(sample_dir))}"
            f" --chain-id {shlex.quote(args.chain_id)}"
        )
        if args.max_samples_per_target is not None:
            cmd += f" --max-samples-per-target {int(args.max_samples_per_target)}"
        _run_in_conda(env_name, cmd, cwd=repo_root, dry_run=args.dry_run)

    if not args.skip_reconstruct:
        reconstruct_cmd = (
            f"python models_con/sample.py --SAMPLEDIR {shlex.quote(str(sample_dir))}"
        )
        _run_in_conda(args.main_env, reconstruct_cmd, cwd=repo_root, dry_run=args.dry_run)

    if not args.skip_geometry:
        run_self_in_env(args.main_env, "geometry")

    if not args.skip_energy:
        run_self_in_env(args.energy_env, "energy")

    # Optional custom steps for external tools/baselines.
    for step in args.extra_conda_step:
        if ":::" not in step:
            raise ValueError(
                "--extra-conda-step must be in format ENV:::COMMAND, got: "
                + step
            )
        env_name, inner_cmd = step.split(":::", 1)
        _run_in_conda(env_name.strip(), inner_cmd.strip(), cwd=repo_root, dry_run=args.dry_run)

    print("[DONE] Orchestration finished.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PepFlow multi-env evaluation runner.")
    parser.add_argument(
        "--mode",
        choices=["orchestrate", "geometry", "energy"],
        default="orchestrate",
        help="Run orchestrator or a concrete eval mode.",
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        help="SAMPLE_DIR containing outputs/ and/or pdbs/ directories.",
    )
    parser.add_argument("--chain-id", default="A", help="Target peptide chain id.")
    parser.add_argument(
        "--max-samples-per-target",
        type=int,
        default=None,
        help="Optional limit for sample files per target.",
    )

    # Orchestrate options
    parser.add_argument("--main-env", default=DEFAULT_MAIN_ENV, help="Main conda env name.")
    parser.add_argument(
        "--energy-env", default=DEFAULT_ENERGY_ENV, help="Energy conda env name."
    )
    parser.add_argument("--skip-reconstruct", action="store_true", help="Skip PDB reconstruction.")
    parser.add_argument("--skip-geometry", action="store_true", help="Skip geometry eval.")
    parser.add_argument("--skip-energy", action="store_true", help="Skip energy eval.")
    parser.add_argument(
        "--extra-conda-step",
        action="append",
        default=[],
        help="Optional extra step in format ENV:::COMMAND. Can be used multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print commands.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    os.chdir(repo_root)

    sample_dir = Path(args.sample_dir).expanduser().resolve()

    if args.mode == "geometry":
        _mode_geometry(sample_dir, args.chain_id, args.max_samples_per_target)
    elif args.mode == "energy":
        _mode_energy(sample_dir, args.chain_id, args.max_samples_per_target)
    else:
        _mode_orchestrate(args, script_path=script_path, repo_root=repo_root)


if __name__ == "__main__":
    main()
