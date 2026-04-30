# PepFlow Eval & Benchmark Notes

Updated: 2026-04-30 (env split + runner script)

## 1. Overall evaluation logic

Yes, your understanding is basically correct:

1. Use PepFlow `FlowModel` to generate samples (from validation/test split conditions).
2. Save generated outputs (`.pt`).
3. Reconstruct generated outputs to PDB files.
4. Run evaluation metrics on generated results (geometry / sequence / interface / energy).

In this repo, some quick metrics are already computed during generation (`models_con/inference.py`), while richer benchmark metrics are implemented as utility functions in `eval/`.

---

## 2. Built-in quick evaluation in generation stage

File: `models_con/inference.py`

This script:
- loads checkpoint and dataset
- samples with `model.sample(...)`
- computes per-item summary metrics
- writes per-item `.pt` outputs and a CSV

Recorded fields in `outputs.csv`:
- `id`, `len`
- `tran`
- `rot`
- `aar`
- `trans_loss`
- `rot_loss`

Interpretation:
- `tran`: translation RMS-like error on generated residues
- `rot`: rotation-matrix RMS-like error on generated residues
- `aar`: amino-acid recovery ratio on generated residues
- `trans_loss`, `rot_loss`: model forward losses on the sampled batch

Relevant lines:
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/models_con/inference.py:77`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/models_con/inference.py:85`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/models_con/inference.py:86`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/models_con/inference.py:87`

---

## 3. Reconstruction step

File: `models_con/sample.py`

After generation, use the saved `.pt` files to reconstruct PDBs:
- each target gets `sample_i.pdb`
- and ground truth `gt.pdb`

Relevant lines:
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/models_con/sample.py:141`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/models_con/sample.py:143`

---

## 4. Metrics implemented under eval/

### 4.1 Geometry / structure metrics
File: `eval/geometry.py`

- RMSD (`get_rmsd`)
- TM-score via `tmtools` (`get_tm`)
- secondary-structure consistency via DSSP (`get_ss`)
- binding-site overlap ratio (`get_bind_ratio`)

Relevant lines:
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/geometry.py:47`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/geometry.py:61`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/geometry.py:88`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/geometry.py:106`

### 4.2 External TMscore parsing
File: `eval/align.py`

- uses `TMscore` executable output parsing
- extracts RMSD and TM-score from CLI output

Relevant lines:
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/align.py:12`

### 4.3 FoldX binding affinity
File: `eval/foldx.py`

- runs `foldx --command=AnalyseComplex`
- parses `Summary_*_AC.fxout` for binding-affinity field

Relevant lines:
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/foldx.py:61`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/foldx.py:69`

### 4.4 Rosetta energy/interface metrics
File: `eval/energy.py`

- uses PyRosetta `scorefxn` for stability-like score
- uses `InterfaceAnalyzerMover` for `dG_separated`

Relevant lines:
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/energy.py:33`
- `/home/immortalco/CS598GBL-Project-PepFlow/pepflow/eval/energy.py:58`

---

## 5. What “benchmarks” exist in this repo

The repo contains scripts/functions for comparing PepFlow with baseline pipelines:

- ESM-IF sequence design: `eval/run_esmif.py`
- ProteinMPNN design pipeline: `eval/run_mpnn.py`
- RFdiffusion / ProteinGenerator generation wrappers: `eval/run_rfdiffusion.py`
- Scwrl4 side-chain packing baseline: `eval/run_scwrl4.py`
- ESMFold refolding helper: `eval/run_esmfold.py`

These scripts reference many local absolute tool paths from the authors’ environment, so they are templates/helpers rather than fully portable one-click benchmark runners.

---

## 6. Important practical note

There is no single end-to-end benchmark entrypoint in this repo.

Current practical pipeline is:

1. Generate (`models_con/inference.py`)
2. Reconstruct PDB (`models_con/sample.py`)
3. Evaluate via selected functions/scripts in `eval/`
4. Aggregate your own benchmark table

---

## 7. Environment split used in this workspace

Below is the practical mapping we are using now.

### 7.1 Main env (`pepflow`)

Config delta file:
- `pepflow/pepflow-eval-main.yaml`

What to run here:
- generation / reconstruction:
  - `models_con/inference.py`
  - `models_con/sample.py`
- geometry metrics:
  - `eval/geometry.py`
- quick metrics already emitted by inference:
  - `tran`, `rot`, `aar`, `trans_loss`, `rot_loss`

Installed extra in this workspace:
- `tmtools` (was missing; `mdtraj` already existed)

### 7.2 PyRosetta env (`pepflow-eval-energy`)

Config file:
- `pepflow/pepflow-eval-energy.yaml`

What to run here:
- `eval/energy.py`

Status:
- `pyrosetta-installer` is installed
- `pyrosetta` wheel itself is not installed by default (needs explicit installer step)

### 7.3 ESM-IF + ProteinMPNN env (`pepflow-eval-esm-if-mpnn`)

Config file:
- `pepflow/pepflow-eval-esm-if-mpnn.yaml`

What to run here:
- `eval/run_esmif.py`
- `eval/run_mpnn.py`

Status:
- imports validated for `torch`, `torch_scatter`, `torch_geometric`, `esm`, `proteinmpnn`
- scripts still contain hard-coded absolute tool paths and need path patching before full run

### 7.4 RFdiffusion env (`pepflow-eval-rfdiffusion`)

Config file:
- `pepflow/pepflow-eval-rfdiffusion.yaml`

What to run here:
- `eval/run_rfdiffusion.py` (RFdiffusion branch)

Status:
- imports validated for `torch==1.12.1+cu116`, `dgl==1.0.2+cu116`, `hydra`, `e3nn`, `rfdiffusion`, `se3_transformer`
- `run_rfdiffusion.py` also references `protein_generator/inference.py` absolute path; that part needs separate local tool setup

### 7.5 External binary tools (not solved by pip env alone)

- `TMscore` executable for `eval/align.py`
- `FoldX` executable for `eval/foldx.py`
- `Scwrl4` executable for `eval/run_scwrl4.py`

---

## 8. Multi-env evaluation helper script

Created script:
- `pepflow/eval/multienv_eval_runner.py`

Purpose:
- orchestrate eval by calling `conda run --no-capture-output -n <env> ...` from Python
- support split execution across main env and energy env
- provide optional extra conda steps for baseline/external tool commands

Core modes:
1. `orchestrate` (default)
2. `geometry`
3. `energy`

Example (full orchestrate):
```bash
python eval/multienv_eval_runner.py \
  --sample-dir /ABS/PATH/TO/SAMPLE_DIR \
  --mode orchestrate
```

Example (only geometry):
```bash
python eval/multienv_eval_runner.py \
  --mode geometry \
  --sample-dir /ABS/PATH/TO/SAMPLE_DIR \
  --chain-id A
```

Example (append extra cross-env step):
```bash
python eval/multienv_eval_runner.py \
  --sample-dir /ABS/PATH/TO/SAMPLE_DIR \
  --skip-reconstruct --skip-geometry --skip-energy \
  --extra-conda-step "pepflow-eval-esm-if-mpnn:::python your_esmif_cmd.py"
```
