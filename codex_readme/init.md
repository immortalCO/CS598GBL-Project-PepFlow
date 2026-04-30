# PepFlow + GRPO implementation note for Codex

This note has two goals:

1. Make the official PepFlow repo runnable quickly, with the least amount of debugging.
2. Lay out several post-training options for GRPO / DPO / flow-matching alignment, from easiest to hardest, so we can choose based on actual GPU memory and runtime.

Important meta-point:
- Do **not** over-commit to one method before profiling the base model.
- The current PepFlow code already supports partial sampling (`sample_bb`, `sample_ang`, `sample_seq`), so start from the smallest controllable setting first.
- We do **not** yet know the real VRAM cost of training + inference for our hardware. Build the environment, patch the repo, run smoke tests, and profile memory first. Then choose the simplest method that fits.

---

## 1. What the PepFlow repo already provides

### 1.1 Main resources in the repo
Key folders / files:
- `configs/`
- `models_con/`
- `pepflow/`
- `openfold/`
- `eval/`
- `playgrounds/`
- `train.py`
- `train_ddp.py`
- `models_con/inference.py`
- `models_con/sample.py`

### 1.2 Official data / pretrained resources
The README says the official Google Drive contains:
- `PepMerge_release.zip`
- `PepMerge_lmdb.zip`
- `model1.pt`
- `model2.pt`

Interpretation:
- `PepMerge_release.zip` = processed peptide–receptor examples.
- `PepMerge_lmdb.zip` = prebuilt LMDB splits.
- `model1.pt` = suggested for benchmark evaluation.
- `model2.pt` = suggested for larger / more realistic peptide design tasks.

### 1.3 What each script is for
Use the current code, not the README wording, as source of truth.

- `train.py`:
  - single-GPU supervised training
- `train_ddp.py`:
  - multi-GPU DDP supervised training
- `models_con/inference.py`:
  - loads a checkpoint
  - builds `PepDataset`
  - duplicates each item `num_samples` times
  - runs `model.sample(...)`
  - computes a few quick metrics
  - saves final sampled output as `.pt`
- `models_con/sample.py`:
  - loads `.pt` files from `outputs/`
  - reconstructs PDB files
  - writes `sample_i.pdb` and `gt.pdb`

### 1.4 What the model currently exposes that is useful for us
`FlowModel.forward(batch)` returns a dict with:
- `trans_loss`
- `rot_loss`
- `bb_atom_loss`
- `seqs_loss`
- `angle_loss`
- `torsion_loss`

This is very useful, because later we can:
- keep a supervised anchor loss,
- add a reward-based loss,
- or build a surrogate score from the weighted sum of these losses.

`FlowModel.sample(batch, num_steps=..., sample_bb=..., sample_ang=..., sample_seq=...)` already supports modality control:
- `sample_bb=True/False`
- `sample_ang=True/False`
- `sample_seq=True/False`

This gives us natural task slices:
- **known-backbone, sequence-only**: `sample_bb=False, sample_ang=False, sample_seq=True`
- **known-backbone, sequence+angles**: `sample_bb=False, sample_ang=True, sample_seq=True`
- **full design**: all `True`

---

## 2. Fast path to a runnable PepFlow environment

## 2.1 Clone the repo
```bash
git clone https://github.com/Ced3-han/PepFlowww.git
cd PepFlowww
```

## 2.2 Environment strategy

### Option A: closest to the repo
Use the official environment file first, then patch missing pieces.

```bash
conda env create -f environment.yml
conda activate flow

# The README explicitly installs torch-scatter separately.
# IMPORTANT: install the torch-scatter wheel that matches *your* PyTorch/CUDA version.
# The README example uses torch 2.0.0 + cu117, but do not hardcode that if your machine differs.
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Practical extras that the code imports and may need.
pip install joblib lmdb easydict omegaconf wandb
```

### Option B: lean fallback if Option A is painful
If the full conda env is too heavy or fails, create a lean env and add packages incrementally.

```bash
conda create -n pepflow_min python=3.10 -y
conda activate pepflow_min

# Install a PyTorch build matching your machine.
# Example only; replace with the right torch/cuda pair.
# pip install torch torchvision torchaudio --index-url ...

pip install biopython biotite pandas scipy pyyaml tqdm matplotlib joblib lmdb easydict omegaconf wandb
# install torch-scatter with the wheel matching your torch/cuda
```

## 2.3 Make the repo importable
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python setup.py develop
```

If `setup.py develop` is annoying, keeping `PYTHONPATH` is enough for a first pass.

---

## 3. Local data layout that is easiest to work with

Recommended local structure:

```text
PepFlowww/
  Data/
    PepMerge_release/
      1a0n_A/
      ...
    lmdb/
      pep_pocket_train_structure_cache.lmdb
      pep_pocket_test_structure_cache.lmdb
    names.txt
  ckpts/
    model1.pt
    model2.pt
  logs/
```

Why this layout:
- `PepDataset` builds LMDB paths as:
  - `{dataset_dir}/{name}_structure_cache.lmdb`
- the default config uses:
  - `name: pep_pocket_train`
  - `name: pep_pocket_test`

So the easiest setup is:
- `dataset_dir = ./Data/lmdb`
- `structure_dir = ./Data/PepMerge_release`

If you already have the official LMDB zip:
- keep `reset: False`
- do **not** rebuild the cache unless needed

If LMDB is missing:
- set `reset: True` once to preprocess from `structure_dir`

---

## 4. Required repo patches before doing anything serious

These are the first things Codex should patch.

## 4.1 Always pass the config explicitly
`train.py` and `train_ddp.py` default to a path like:
- `./configs/angle/learn_angle.yaml`

But the repo currently ships:
- `configs/learn_angle.yaml`

So do **not** rely on the default.
Always run with:

```bash
--config ./configs/learn_angle.yaml
```

## 4.2 Patch `configs/learn_angle.yaml`
The current config contains absolute local paths from the authors’ machine.

Change the dataset section to something like:

```yaml
dataset:
  train:
    type: peprec
    structure_dir: ./Data/PepMerge_release
    dataset_dir: ./Data/lmdb
    name: pep_pocket_train
    reset: False
  val:
    type: peprec
    structure_dir: ./Data/PepMerge_release
    dataset_dir: ./Data/lmdb
    name: pep_pocket_test
    reset: False
```

For smoke tests, also reduce runtime:
```yaml
train:
  max_iters: 1000
  val_freq: 100
  batch_size: 4
```

## 4.3 Patch `models_con/pep_dataloader.py`
There is a hardcoded absolute path to `names.txt`.

Make it configurable. Example patch:

```python
# BEFORE:
# with open('/abs/path/to/names.txt','r') as f:

# AFTER:
import os

DATA_ROOT = os.environ.get("PEPFLOW_DATA_ROOT", "./Data")
NAMES_PATH = os.environ.get("PEPFLOW_NAMES_PATH", os.path.join(DATA_ROOT, "names.txt"))

names = []
if os.path.exists(NAMES_PATH):
    with open(NAMES_PATH, "r") as f:
        for line in f:
            names.append(line.strip())
```

Practical advice:
- For the very first smoke test, if the names filtering causes trouble, make it optional.
- Since we already have official LMDB splits, we do not need to immediately understand every split-related detail in preprocessing.

## 4.4 Patch `models_con/inference.py` boolean arguments
The current script uses `type=bool` in argparse, which is a bad idea because:
- `--sample_bb False` often still becomes `True` in Python parsing.

Replace with a safe parser:

```python
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
```

Then use:
```python
args.add_argument('--sample_bb', type=str2bool, default=True)
args.add_argument('--sample_ang', type=str2bool, default=True)
args.add_argument('--sample_seq', type=str2bool, default=True)
```

Also delete the duplicated `--num_samples` definitions.

## 4.5 Patch `FlowModel.sample(...)` for efficiency
Right now `sample()` builds a `clean_traj` list and appends CPU copies at every step.
That is okay for inspection, but not ideal for large `num_steps` or RL.

Add a flag:
```python
def sample(..., return_traj=False):
```

Behavior:
- `return_traj=False`:
  - return only the final sample dict
- `return_traj=True`:
  - keep the current full trajectory behavior

For RL, also add a second mode:
```python
def sample_with_cache(...):
```

This should return **minimal** cache needed for training, e.g.:
- current state per step
- time index
- maybe RNG seeds / noises if stochastic sampling is added later
- not full CPU copies of every tensor unless necessary

---

## 5. Smoke tests before any GRPO work

## 5.1 Import test
```bash
python - <<'PY'
import torch
from models_con.pep_dataloader import PepDataset
from pepflow.utils.misc import load_config
print("torch", torch.__version__)
cfg, name = load_config("./configs/learn_angle.yaml")
print("loaded config:", name)
PY
```

## 5.2 Dataset test
```bash
python - <<'PY'
from models_con.pep_dataloader import PepDataset
ds = PepDataset(
    structure_dir="./Data/PepMerge_release",
    dataset_dir="./Data/lmdb",
    name="pep_pocket_train",
    reset=False,
)
print("len(ds) =", len(ds))
x = ds[0]
print("keys:", x.keys())
print("id:", x["id"])
print("generate residues:", x["generate_mask"].sum().item())
PY
```

## 5.3 Single-GPU debug train
Use debug mode first because it avoids the full logging / wandb workflow.

```bash
python train.py \
  --config ./configs/learn_angle.yaml \
  --device cuda:0 \
  --num_workers 0 \
  --debug
```

If that works, try a short real run:
```bash
python train.py \
  --config ./configs/learn_angle.yaml \
  --device cuda:0 \
  --num_workers 4 \
  --name pepflow_local
```

## 5.4 Tiny inference test with a pretrained checkpoint
Start small on purpose:
- low `num_steps`
- low `num_samples`

```bash
mkdir -p ./tmp_outputs/outputs

python models_con/inference.py \
  --config ./configs/learn_angle.yaml \
  --device cuda:0 \
  --ckpt ./ckpts/model1.pt \
  --output ./tmp_outputs \
  --num_steps 20 \
  --num_samples 4 \
  --sample_bb false \
  --sample_ang false \
  --sample_seq true
```

Then reconstruct PDBs:
```bash
python models_con/sample.py --SAMPLEDIR ./tmp_outputs
```

This is the best first test for the **known-backbone, sequence-only** path.

---

## 6. Memory profiling before choosing a post-training method

We do not know yet whether:
- the model forward is cheap,
- the sampler is expensive,
- or full-trajectory storage becomes the bottleneck.

So first profile.

Codex should create a small script:
- vary `num_steps` in `[10, 20, 50, 100]`
- vary `num_samples` in `[1, 2, 4, 8, 16]`
- vary mode:
  - seq-only
  - seq+angles
  - full
- log:
  - wall-clock time
  - `torch.cuda.max_memory_allocated()`
  - optional host RAM if easy

Skeleton idea:
```python
torch.cuda.reset_peak_memory_stats()
# run model.forward(...)
# run model.sample(...)
peak = torch.cuda.max_memory_allocated() / 1024**3
```

Most likely knobs for reducing cost:
1. smaller `num_samples`
2. smaller `num_steps`
3. partial sampling (`sample_bb=False`)
4. `return_traj=False`
5. LoRA / partial fine-tuning only
6. gradient accumulation instead of large batch size

---

## 7. Common notation for post-training

Let:
- `c` = condition (pocket, receptor, maybe backbone)
- `y` = final generated peptide sample
- `pi_theta(y | c)` = current PepFlow generator
- `theta_ref` = frozen pretrained reference model
- `theta_old` = rollout snapshot used to generate on-policy samples
- `G` = group size for GRPO
- `r_i = Oracle(y_i, c)` = scalar reward from an oracle or evaluator

For a group of samples `y_1, ..., y_G`, define normalized group advantage:
\[
A_i = \frac{r_i - \mu_G}{\sigma_G + \varepsilon},
\qquad
\mu_G = \frac{1}{G}\sum_{j=1}^G r_j
\]

PepFlow already returns a modality-decomposed training loss. Define the weighted supervised loss:
\[
\ell_{\text{PF}}(y, c; \theta)
=
\lambda_{\text{trans}} \ell_{\text{trans}}
+ \lambda_{\text{rot}} \ell_{\text{rot}}
+ \lambda_{\text{bb}} \ell_{\text{bb}}
+ \lambda_{\text{seq}} \ell_{\text{seq}}
+ \lambda_{\text{ang}} \ell_{\text{ang}}
+ \lambda_{\text{tors}} \ell_{\text{tors}}
\]

Using the default config weights:
\[
\ell_{\text{PF}}
=
0.5\,\ell_{\text{trans}}
+ 0.5\,\ell_{\text{rot}}
+ 0.25\,\ell_{\text{bb}}
+ 1.0\,\ell_{\text{seq}}
+ 1.0\,\ell_{\text{ang}}
+ 0.5\,\ell_{\text{tors}}
\]

This is useful because we can treat:
\[
s_\theta(y,c) = -\ell_{\text{PF}}(y,c;\theta)
\]
as a surrogate sample score.

---

## 8. Post-training menu, from simplest to hardest

## Option 0: Winner-only self-distillation (easiest sanity baseline)
This is **not** GRPO, but it is the fastest sanity check.

Procedure:
1. For each condition `c`, sample `G` candidates from the current model.
2. Let the oracle pick the best one:
   \[
   y^\star = \arg\max_i r_i
   \]
3. Treat `y^\star` as a pseudo-target and train the model on it using the normal PepFlow loss.

Objective:
\[
L_{\text{winner}} = \ell_{\text{PF}}(y^\star, c; \theta)
\]

Recommended anchored version:
\[
L = \ell_{\text{PF}}(y^\star, c; \theta)
+ \lambda_{\text{sup}} \,\ell_{\text{PF}}(y_{\text{gt}}, c; \theta)
\]

Why do this:
- almost zero RL engineering
- tells us whether “sample best-of-G then imitate it” already helps
- good debug stage before any clipped objective

Why this may fail:
- can collapse diversity
- ignores relative quality among non-winners
- not principled on-policy RL

Use this if:
- the model is expensive
- memory is tight
- we only need a very fast proof of concept

---

## Option 1: Pairwise DPO-style flow matching (still simple, stronger than winner-only)
This is the natural “energy/DPO-style” bridge.

Procedure:
1. For each condition `c`, sample a small group.
2. Use the oracle to pick a winner `y_w` and loser `y_l`.
3. Define a surrogate score from PepFlow’s weighted loss:
   \[
   s_\theta(y,c) = -\ell_{\text{PF}}(y,c;\theta)
   \]
4. Optimize a DPO-style objective.

Objective:
\[
L_{\text{pair}}
=
-\log \sigma \Big(
\beta \Big[
\big(s_\theta(y_w,c)-s_\theta(y_l,c)\big)
-
\big(s_{\theta_{\text{ref}}}(y_w,c)-s_{\theta_{\text{ref}}}(y_l,c)\big)
\Big]
\Big)
\]

Anchored version:
\[
L = L_{\text{pair}} + \lambda_{\text{sup}} \,\ell_{\text{PF}}(y_{\text{gt}}, c; \theta)
\]

Why this is attractive:
- simpler than GRPO
- directly uses relative preference
- does not require modeling stepwise trajectory ratios
- very natural if the oracle can rank or compare samples

Use this if:
- we want something stronger than winner-only
- we are not ready for full GRPO
- we want a clean comparison against “energy-based DPO-like” methods

---

## Option 2: Surrogate GRPO over final samples (recommended first actual GRPO)
This is the recommended first real GRPO-style method for PepFlow.

Core idea:
- Do on-policy grouped rollouts as in GRPO.
- But instead of deriving exact stepwise transition ratios for the mixed continuous+discrete flow process, use a **surrogate ratio** built from the PepFlow loss on the final samples.

### Step 1: rollout
For each condition `c`, sample `G` candidates from `theta_old`:
\[
y_i \sim \pi_{\theta_{\text{old}}}(\cdot \mid c), \quad i=1,\dots,G
\]

### Step 2: reward
Use the oracle to get rewards:
\[
r_i = \text{Oracle}(y_i, c)
\]

Then compute group-relative advantages:
\[
A_i = \frac{r_i - \mu_G}{\sigma_G + \varepsilon}
\]

### Step 3: surrogate score
For each final sample `y_i`, compute a surrogate score from the PepFlow loss:
\[
s_\theta(y_i,c) = -\ell_{\text{PF}}(y_i,c;\theta)
\]

and similarly for the frozen rollout/reference model:
\[
s_{\theta_{\text{old}}}(y_i,c) = -\ell_{\text{PF}}(y_i,c;\theta_{\text{old}})
\]

Define a sample-level ratio:
\[
\rho_i = \exp\left(s_\theta(y_i,c) - s_{\theta_{\text{old}}}(y_i,c)\right)
\]

### Step 4: GRPO clipped objective
\[
L_{\text{GRPO-sur}}
=
-\frac{1}{G}\sum_{i=1}^{G}
\min\left(
\rho_i A_i,\;
\text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i
\right)
\]

Anchored version:
\[
L
=
L_{\text{GRPO-sur}}
+ \lambda_{\text{sup}} \,\ell_{\text{PF}}(y_{\text{gt}}, c; \theta)
+ \beta \, D(\theta, \theta_{\text{ref}})
\]

where `D` can be one of:
- parameter L2 on trainable adapters
- prediction matching to the reference model at random times
- or simply the supervised ground-truth loss term

### Why this is the recommended first GRPO
- no need to derive exact transition densities
- no need to backprop through the whole denoising chain
- no need to store giant autograd graphs
- works naturally with PepFlow’s existing multi-loss training API

### Practical implementation notes
- rollout with `torch.no_grad()`
- update with standard backward only on the surrogate loss
- train only LoRA or a small subset of layers first
- start with:
  - known-backbone
  - sequence-only or sequence+angles
  - small `G`
  - small `num_steps`

This is probably the best balance between:
- “actually GRPO-like”
- and “realistically implementable on a course project timeline”

---

## Option 3: Stepwise GRPO with a stochastic flow sampler (more faithful, harder)
This is the “literal” GRPO route.

Problem:
- PepFlow’s current sampler is basically a deterministic flow / ODE-style iterative update.
- Standard GRPO / PPO likes stochastic rollouts for exploration and for transition-level likelihood ratios.

So we need a stochasticized sampler.

### Stepwise view
Let the trajectory be:
\[
x_{t_0}, x_{t_1}, \dots, x_{t_T}
\]

A stochasticized update can look like:
\[
x_{k+1}
=
x_k + f_\theta(x_k, t_k, c)\Delta t + g_k \sqrt{\Delta t}\,\epsilon_k
\]

where:
- `f_theta` is the learned flow update
- `g_k` is injected noise scale
- `epsilon_k ~ N(0, I)` for continuous parts
- discrete sequence updates may still use sampling from logits

### Group reward
Still only use final reward:
\[
r_i = \text{Oracle}(x_{t_T}^{(i)}, c)
\]
and group advantage:
\[
A_i = \frac{r_i - \mu_G}{\sigma_G + \varepsilon}
\]

### Stepwise ratio
If we can write or approximate transition probabilities:
\[
\rho_{i,k}
=
\frac{
p_\theta\big(x_{k+1}^{(i)} \mid x_k^{(i)}, c\big)
}{
p_{\theta_{\text{old}}}\big(x_{k+1}^{(i)} \mid x_k^{(i)}, c\big)
}
\]

then optimize:
\[
L_{\text{step-GRPO}}
=
-\frac{1}{GT}
\sum_{i=1}^{G}\sum_{k=0}^{T-1}
\min\left(
\rho_{i,k} A_i,\;
\text{clip}(\rho_{i,k},1-\epsilon,1+\epsilon) A_i
\right)
+ \beta \,\text{KL}
\]

### Important engineering rule
Do **not** backprop through the full rollout graph.

Instead:
1. rollout with `torch.no_grad()`
2. cache minimal states
3. recompute the needed quantities during the update step

Otherwise memory will explode.

### Why this is hard for PepFlow
PepFlow has mixed modalities:
- translation
- rotation
- sequence
- angle / torsion

So exact per-step transition modeling is much more annoying than in a simple continuous image flow.

Use this only if:
- surrogate GRPO clearly works and we want a stronger method
- we are willing to patch the sampler carefully

---

## Option 4: Branching / tree-structured GRPO (for efficiency if rollout is too expensive)
If rollout is the bottleneck, branching is the next idea.

Core idea:
- multiple samples share a common prefix of denoising steps
- only branch later
- avoid recomputing the same early steps independently

Naive rollout cost is roughly:
\[
\text{cost}_{\text{naive}} \propto G \cdot T
\]

A branching rollout can reduce this to something more like:
\[
\text{cost}_{\text{branch}}
\propto
T_{\text{shared}} + B\cdot(T - T_{\text{shared}})
\]
where `B < G` if shared prefixes are reused effectively.

Practical version:
1. sample a few initial noises
2. roll a shared prefix
3. branch into multiple continuations
4. evaluate leaves with the oracle
5. optionally prune bad branches early

Why this is interesting:
- much cheaper if `T` is large
- especially helpful if the final reward is sparse and rollout dominates wall-clock time

Use this only after:
- we know rollout is the bottleneck
- and simpler GRPO is already working

---

## Option 5: Training-free inference steering baseline (must-have baseline)
Even if we plan post-training, we should keep a training-free baseline.

Target distribution idea:
\[
p_\alpha(y \mid c)
\propto
p_{\theta_0}(y \mid c)\exp(\alpha r(y,c))
\]

We then approximately sample from this aligned distribution using:
- best-of-`N` reranking
- resampling / SMC-like methods
- reward guidance if the reward is differentiable enough

Why this baseline matters:
- sometimes it gives most of the gain without retraining
- if GRPO is unstable or too expensive, this may still salvage the project
- it is especially useful when oracle evaluation is expensive and we do not want full online updates yet

Minimum inference-steering baseline:
1. sample `N` candidates
2. oracle score all `N`
3. keep the best

Better inference-steering baseline:
- iterative resampling / refinement during denoising
- or test-time alignment / DAS-like search

---

## 9. What **not** to do at first

Do **not** start with:
- full-sequence+structure co-design
- exact stepwise GRPO
- full-parameter finetuning
- huge group sizes
- huge denoising step counts
- backprop through the whole denoising chain

Do **not** assume the current sampler’s memory behavior is acceptable for RL.
Profile first.

Do **not** assume CLI booleans work correctly before patching them.

---

## 10. Recommended implementation order

### Phase A: repo sanity
1. patch config paths
2. patch `names.txt` path
3. patch bool parsing
4. add `return_traj=False`
5. run debug train
6. run tiny inference
7. run tiny PDB reconstruction

### Phase B: profiling
1. profile forward memory
2. profile sample memory
3. profile seq-only / seq+angles / full
4. profile small vs large `num_steps`
5. write down the largest safe setting

### Phase C: baselines
1. best-of-`N` reranking
2. winner-only self-distillation
3. pairwise DPO-style flow matching

### Phase D: first actual GRPO
1. surrogate GRPO over final samples
2. LoRA or last-block training only
3. known-backbone seq-only first
4. then known-backbone seq+angles

### Phase E: harder methods only if needed
1. full PepFlow
2. stochastic stepwise GRPO
3. branching GRPO
4. training-free inference steering comparison

---

## 11. What to log for every experiment

Always log:
- condition ID
- sampling mode (`bb/ang/seq`)
- `num_steps`
- `num_samples`
- runtime
- peak GPU memory
- reward mean / std / max
- diversity proxy
- failure count
- whether outputs can still be reconstructed to PDB cleanly

For training runs also log:
- supervised anchor loss
- RL / DPO / GRPO objective
- reward drift over time
- collapse indicators:
  - repeated sequences
  - identical outputs across the group
  - NaNs
  - empty / invalid reconstructions

---

## 12. If memory is worse than expected, adapt like this

If VRAM is too high:
1. switch to known-backbone seq-only
2. reduce `num_steps`
3. reduce `num_samples`
4. use `return_traj=False`
5. use LoRA / PEFT only
6. use gradient accumulation
7. keep the reward terminal-only
8. prefer DPO-style or surrogate GRPO over stepwise GRPO

If even that is too heavy:
- do inference steering first
- or do winner-only / pairwise DPO before GRPO

---

## 13. The single most likely first method to implement

If nothing else is known yet, implement this first:

**known-backbone, sequence-only PepFlow + surrogate GRPO over final samples**

Concretely:
- patch the repo
- use official LMDB + checkpoint
- set:
  - `sample_bb=False`
  - `sample_ang=False`
  - `sample_seq=True`
- rollout with `torch.no_grad()`
- oracle-score grouped samples
- compute group-normalized advantages
- use the weighted PepFlow loss as a surrogate score
- optimize a clipped GRPO objective plus a supervised anchor

This is the best tradeoff between:
- simplicity,
- being genuinely post-training,
- and not getting buried in stochastic flow math too early.