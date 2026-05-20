# RLinf Setup Progress

## Hardware

- **Machine:** Local workstation (no NVIDIA GPUs)
- **GPU:** AMD Radeon 8060S (gfx1151 / Ryzen AI MAX+ 395 iGPU, Strix Halo)
- **ROCm:** 7.2 installed natively at `/opt/rocm`

## Config fixes

### `examples/embodiment/config/libero_spatial_grpo_openvlaoft.yaml`

- Added `- _self_` to defaults list (fixes Hydra composition warning)
- Changed `env.train.total_num_envs: 8 → 16`

  **Why:** With `component_placement: actor,env,rollout: all` and `env_world_size=4` GPUs,
  the validation requires `total_num_envs // env_world_size // pipeline_stage_num % group_size == 0`.
  `8 // 4 // 1 = 2`, and `2 % 4 ≠ 0`. Setting `total_num_envs: 16` gives `16 // 4 // 1 = 4`,
  which is divisible by `group_size=4`.
  
  On a single AMD GPU, `env_world_size=1`, so `total_num_envs=16` also satisfies the constraint
  (`16 // 1 // 1 = 16`, `16 % 4 = 0`).

## Docker — why it doesn't work on this machine

The pre-built Docker image `rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4` fails with a **SIGSEGV**
during model loading because its PyTorch was compiled for a fixed set of GPU architectures that
does **not** include `gfx1151`:

```
['gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx942',
 'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201']
```

`HSA_OVERRIDE_GFX_VERSION` overrides do not help — the crash happens at the ROCm runtime level
before kernel dispatch.

### Docker run command (for reference, not currently working)

```bash
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 20g \
    --network host \
    --name rlinf-amd-libero \
    -v .:/workspace/RLinf \
    -w /workspace/RLinf \
    rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4 \
    bash -c "source switch_env openvla-oft && MUJOCO_GL=osmesa ROBOT_PLATFORM=LIBERO bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft"
```

To fix Docker, rebuild the image with gfx1151 included:

```bash
docker build \
    --build-arg PLATFORM=amd \
    --build-arg ROCM_VER=6.4 \
    --build-arg BUILD_TARGET=embodied-libero \
    --build-arg 'ROCM_ARCHS=gfx90a;gfx942;gfx1151' \
    -t rlinf-libero-rocm6.4-gfx1151 .
```

(This rebuilds PyTorch from source and takes several hours.)

## Native install — the working path

Uses the same `torch==2.11.0+rocm7.2` build as `/home/gberseth/playground/mini-grp/.venv`,
which correctly sees `AMD Radeon 8060S`.

### Install

```bash
cd /home/gberseth/playground/RLinf
bash requirements/install.sh --platform amd --rocm 7.2 embodied --model openvla-oft --env libero
```

Creates `.venv` at `/home/gberseth/playground/RLinf/.venv`.

### Run training

```bash
cd /home/gberseth/playground/RLinf
source .venv/bin/activate
MUJOCO_GL=osmesa ROBOT_PLATFORM=LIBERO \
bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft
```

> `MUJOCO_GL=osmesa` is required for AMD — CPU-based rendering for LIBERO.
> See: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero_amd.html

## Installation on Onyx

Onyx is a remote workstation with 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 97 GB each),
driver 595.58, CUDA 13.2.

### Docker GPU fix

`nvidia-smi` worked but `docker run --gpus all` failed with:
```
failed to discover GPU vendor from CDI: no known GPU vendor found
```

Root cause: driver 595.58 uses the CDI (Container Device Interface) path, but the CDI spec
file was missing. Fix:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Docker run command

```bash
docker run -it --gpus all \
    --shm-size 100g \
    --net=host \
    --name rlinf \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v .:/workspace/RLinf \
    rlinf/rlinf:agentic-rlinf0.2-maniskill_libero /bin/bash
```

Inside the container, activate the env and run:

```bash
source switch_env openvla-oft
cd /workspace/RLinf
bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft
```

### Config fixes for Onyx (`libero_spatial_grpo_openvlaoft.yaml`)

**Validation constraint:**
`total_num_envs // env_world_size // pipeline_stage_num % group_size == 0`

The onyx config had `pipeline_stage_num: 2` (not 1 as in the local copy), causing:
`4 // 1 // 2 = 2`, `2 % 4 ≠ 0` → AssertionError.

Fixes applied on onyx:

| Field | Before | After | Reason |
|-------|--------|-------|--------|
| `rollout.pipeline_stage_num` | 2 | 1 | Pipelining >1 only useful for multi-GPU rollout |
| `cluster.component_placement` | `all` (4 GPUs) | `"0"` (1 GPU) | Run on single GPU to start |
| `env.train.total_num_envs` | 8 | 4 | `4 // 1 // 1 % 4 == 0` ✓ |
| `defaults` | missing `_self_` | added `- _self_` | Ensures main config overrides defaults (Hydra 1.1+) |

The `_self_` addition silences the Hydra composition order warning and ensures values set in
the main YAML (e.g. `group_size: ${algorithm.group_size}`) take precedence over the base
`env/libero_spatial.yaml` defaults.

## install.sh fixes

### 1. `UV_TORCH_BACKEND` cap for rocm7.2

`uv` only accepts `--torch-backend` values up to `rocm7.1`; passing `rocm7.2` causes:
```
error: invalid value 'rocm7.2' for '--torch-backend <TORCH_BACKEND>'
```

Fix in `requirements/install.sh` `configure_amd()`: cap `UV_TORCH_BACKEND` at `rocm7.1`
when `ROCM_VERSION > 7.1`. Torch itself is installed correctly from the pytorch rocm7.2
index (pinned via `override-dependencies`); the cap only affects subsequent `uv pip install`
calls (flash-attn, openvla-oft) that don't need to re-resolve torch.

## Native install — final working command

```bash
bash requirements/install.sh --platform amd --rocm 7.2 --no-root --no-flash-attn embodied --model openvla-oft --env libero
```

- `--no-root`: skips `sys_deps.sh` (needs passwordless sudo for apt packages; not required on a
  machine where ROCm is already installed natively)
- `--no-flash-attn`: skips flash-attn compilation (ROCm clang++ can't find `<cmath>` on this system;
  flash-attn is optional — standard attention is used instead)

## Run training

```bash
cd /home/gberseth/playground/RLinf
source .venv/bin/activate
MUJOCO_GL=osmesa ROBOT_PLATFORM=LIBERO \
bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft
```

`run_embodiment.sh` was patched to fall back to `logs_local/` automatically if `logs/` is not
writable (e.g. owned by root from prior Docker runs).

## Status

- [x] Config validation fixed (`total_num_envs`, `_self_`)
- [x] Root cause of Docker SIGSEGV identified (gfx1151 not in compiled arch list)
- [x] `install.sh` fixes: UV_EXTRA_INDEX_URL for rocm7.2, UV_TORCH_BACKEND unset for rocm>7.1
- [x] Native install succeeded: `torch==2.11.0+rocm7.2` sees AMD Radeon 8060S
- [x] Training confirmed running: all 4 checkpoint shards load, no SIGSEGV
