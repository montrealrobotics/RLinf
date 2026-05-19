# VLA RL Fine-Tuning Experiments on Mila Cluster

This README documents the steps to reproduce RL fine-tuning experiments using the [RLinf](https://github.com/RLinf/RLinf) framework on a single A100L GPU on the Mila SLURM cluster.

> ⚠️ **Important**: LIBERO and RoboCasa have conflicting `robosuite` dependencies and **cannot share the same virtual environment**. Run them in separate `.venv` environments, or sequentially by reinstalling the appropriate dependencies before each run.

---

## Experiment 0: OpenVLA-OFT + LIBERO-90 (GRPO) — Initial Run

> **Note**: This is the first experiment that was run. It is documented here for reference, but it is **not the recommended starting point**. Training on all 90 tasks simultaneously dilutes the reward signal and makes learning harder. It is better to start with a single task suite like LIBERO-Spatial (see Experiment 1) or ideally a single specific task. This experiment is kept here to document the initial setup process and the issues encountered.

### Overview

| | |
|---|---|
| Model | OpenVLA-OFT with LoRA |
| Environment | LIBERO-90 (90 tasks simultaneously) |
| Algorithm | GRPO |
| Hardware | 1 × A100L (80GB) |
| Config file | `examples/embodiment/config/libero_90_grpo_openvlaoft.yaml` |
| Job script | `job-libero-90.sh` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/montrealrobotics/RLinf.git
cd RLinf
```

### Step 2 — Install dependencies

```bash
bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero --no-root
source .venv/bin/activate
```

Verify the install:

```bash
python -c "from libero.libero.envs import OffScreenRenderEnv; print('libero ok')"
```

### Step 3 — Download the model

```bash
pip install huggingface-hub
hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora \
    --local-dir /path/to/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
```

### Step 4 — Configure

In `examples/embodiment/config/libero_90_grpo_openvlaoft.yaml`, set the model paths:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"

actor:
  model:
    model_path: "/path/to/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"
```

The config is already set up for a single A100L with the following key parameters:

```yaml
algorithm:
  group_size: 2
  rollout_epoch: 1

env:
  train:
    total_num_envs: 2
  eval:
    total_num_envs: 2

actor:
  micro_batch_size: 1
  global_batch_size: 4
  enable_offload: True
  model:
    is_lora: True
    lora_rank: 32

rollout:
  enable_offload: True
```

### Step 5 — Update job script

In `job-libero-90.sh`, adapt the paths:

```bash
cd /path/to/RLinf
```

### Step 6 — Submit

```bash
sbatch job-libero-90.sh
```

### Step 7 — Monitor

```bash
tail -f slurm-JOBID.out
tensorboard --logdir ./logs --port 6006
```

The key metric to watch is `env/success_once`. With 90 tasks, reward spikes are expected but sustained learning is difficult — this is why Experiment 1 (LIBERO-Spatial) is preferred.

> 👉 **Recommendation**: For better results, use [Experiment 1](#experiment-1-openvla-oft--libero-spatial-grpo) which trains on a focused 10-task suite, or ideally configure a single specific task.

---

## Experiment 1: OpenVLA-OFT + LIBERO Spatial (GRPO)

### Overview

| | |
|---|---|
| Model | OpenVLA-OFT with LoRA |
| Environment | LIBERO-Spatial (10 tasks) |
| Algorithm | GRPO |
| Hardware | 1 × A100L (80GB) |
| Config file | `examples/embodiment/config/libero_spatial_grpo_openvlaoft.yaml` |
| Job script | `job-libero-spatial.sh` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/montrealrobotics/RLinf.git
cd RLinf
```

### Step 2 — Install dependencies

```bash
bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero --no-root
source .venv/bin/activate
```

Verify the install:

```bash
python -c "from libero.libero.envs import OffScreenRenderEnv; print('libero ok')"
```

### Step 3 — Download the model

The SFT base model for LIBERO-Spatial is hosted by a third party but referenced
by the official RLinf GRPO model card (https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial).

```bash
pip install huggingface-hub
hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
    --local-dir /path/to/RLinf/Openvla-oft-SFT-libero-spatial-traj1
```

### Step 4 — Configure

In `examples/embodiment/config/libero_spatial_grpo_openvlaoft.yaml`, set:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf/Openvla-oft-SFT-libero-spatial-traj1"

actor:
  model:
    model_path: "/path/to/RLinf/Openvla-oft-SFT-libero-spatial-traj1"
    is_lora: True
    lora_rank: 32
```

Single A100L batch size settings:

```yaml
algorithm:
  group_size: 4
  rollout_epoch: 4

env:
  train:
    total_num_envs: 8
  eval:
    total_num_envs: 50

actor:
  micro_batch_size: 4
  global_batch_size: 128
  enable_offload: True

rollout:
  enable_offload: True
```

### Step 5 — Update job script

In `job-libero-spatial.sh`, adapt the paths:

```bash
cd /path/to/RLinf
```

### Step 6 — Submit

```bash
sbatch job-libero-spatial.sh
```

### Step 7 — Monitor

```bash
tail -f slurm-JOBID.out
tensorboard --logdir ./logs --port 6006
```

The key metric to watch is `env/success_once` — this is the episodic success rate. `env/return` is not informative in LIBERO's sparse-reward setting.

---

## Experiment 2: π0 + RoboCasa CloseDrawer (PPO)

### Overview

| | |
|---|---|
| Model | π0 (PaliGemma + flow matching action head) with LoRA |
| Environment | RoboCasa — CloseDrawer (single task) |
| Algorithm | PPO |
| Hardware | 1 × A100L (80GB) |
| Config file | `examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml` |
| Job script | `job-robocasa.sh` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/montrealrobotics/RLinf.git
cd RLinf
```

### Step 2 — Install dependencies

```bash
bash requirements/install.sh embodied --model openpi --env robocasa --no-root
source .venv/bin/activate
```

Verify the install:

```bash
python -c "import robocasa; print('robocasa ok')"
python -c "import robosuite; print('robosuite ok')"
```

### Step 3 — Download kitchen assets

```bash
python -m robocasa.scripts.download_kitchen_assets
```

> This downloads approximately 5GB of kitchen environment assets. Run on a compute node, not the login node.

### Step 4 — Download the model

```bash
pip install huggingface-hub
hf download RLinf/RLinf-Pi0-RoboCasa \
    --local-dir /path/to/RLinf/RLinf-Pi0-RoboCasa
```

### Step 5 — Configure

In `examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml`, set:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf/RLinf-Pi0-RoboCasa"

actor:
  model:
    model_path: "/path/to/RLinf/RLinf-Pi0-RoboCasa"
    is_lora: True
    lora_rank: 32
```

Single A100L batch size settings (tuned to avoid OOM):

```yaml
algorithm:
  rollout_epoch: 2
  update_epoch: 2

env:
  train:
    total_num_envs: 8
  eval:
    total_num_envs: 5
  enable_offload: True

actor:
  micro_batch_size: 4      # keep at 4 to avoid OOM with LoRA rank 32
  global_batch_size: 64
  enable_offload: True
  optim:
    lr: 1.0e-6             # lower than default, more stable

rollout:
  enable_offload: True

runner:
  max_epochs: 150
  save_interval: 10
```


### Step 6 — Update job script

In `job-robocasa.sh`, adapt the paths:

```bash
cd /path/to/RLinf
```

### Step 7 — Submit

```bash
sbatch job-robocasa.sh
```

### Step 8 — Monitor

```bash
tail -f slurm-JOBID.out
tensorboard --logdir ./logs --port 6006
```

Key metrics to watch:
- `env/success_once` — episodic success rate (primary metric)
- `env/return` — episode return
- `actor/policy_loss` — PPO policy loss
- `critic/value_loss` — critic loss
- `actor/approx_kl` — KL divergence from reference policy

### Resuming from a checkpoint

If the job is cancelled or runs out of time, resume from the last saved checkpoint:

```yaml
runner:
  resume_dir: "/path/to/RLinf/logs/TIMESTAMP-robocasa_closedrawer_ppo_openpi/checkpoints/global_step_X"
```

---

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| LIBERO and RoboCasa can't run in the same venv | Conflicting `robosuite` versions | Use separate venvs or reinstall between runs |
| OOM with `micro_batch_size > 4` | LoRA rank 32 + π0 value head fills A100L | Keep `micro_batch_size: 4` for π0 |
