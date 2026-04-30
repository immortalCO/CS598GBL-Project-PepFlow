# GPU Execution Notes (Shared HOME + tmux train)

Last updated: 2026-04-30

## 1. Core rule for this project

- Two servers share the same `$HOME`.
- So **non-GPU tasks** can run directly on the local machine:
  - create/update conda envs
  - clone repos
  - download/unpack data/checkpoints
  - edit code/config
- **GPU tasks only** should run inside the existing SLURM interactive session in `tmux train`.

## 2. tmux/GPU context we already verified

- `tmux` sessions visible from local:
  - `train`
  - `watcher`
- GPU session is in `train:0.0` and has CUDA access.
- Verified in that pane:
  - `torch==2.2.0+cu121`
  - `torch.cuda.is_available() == True`
  - `torch.version.cuda == 12.1`
  - `torch.cuda.device_count() == 4`
  - all devices: `NVIDIA RTX A6000`

## 3. Important limitation

- Do **not** rely on creating a new tmux window inside that SLURM pane.
- `tmux new-window` executed inside `train:0.0` failed with:
  - `error connecting to /tmp//tmux-1000/default (No such file or directory)`
- Use the existing pane `train:0.0` and inject commands into it.

## 4. Command templates (for another Codex session)

### 4.1 Send a GPU command to SLURM pane

```bash
tmux send-keys -t train:0.0 "cd /home/immortalco/CS598GBL-Project-PepFlow && <YOUR_GPU_COMMAND>" C-m
```

### 4.2 Read recent output from that pane

```bash
tmux capture-pane -pt train:0.0 -S -120
```

### 4.3 Quick connectivity check before long GPU jobs

```bash
tmux send-keys -t train:0.0 "echo __CODEX_TMUX_INJECT_OK__ && hostname && pwd" C-m
sleep 0.5
tmux capture-pane -pt train:0.0 -S -20
```

### 4.4 CUDA sanity check in current conda of train pane

```bash
tmux send-keys -t train:0.0 "python -c \"import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); [print('device_%d' % i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]\"" C-m
sleep 1
tmux capture-pane -pt train:0.0 -S -40
```

## 5. Practical workflow split

1. Local machine:
   - setup `conda` envs (e.g. `pepflow`)
   - clone repo into `./pepflow`
   - prepare data/checkpoints under shared home
   - patch code/config
2. `tmux train:0.0`:
   - run `torch` CUDA checks
   - run training/inference/profiling commands that need GPU
   - monitor logs via `tmux capture-pane`

