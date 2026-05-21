# RLinf Setup Progress

## Hardware

- **Machine:** Local workstation (no NVIDIA GPUs)
- **GPU:** AMD Radeon 8060S (gfx1151 / Ryzen AI MAX+ 395 iGPU, Strix Halo)
- **RAM:** 121.5 GB shared (iGPU uses system RAM — no dedicated VRAM)
- **ROCm:** 7.2 installed natively at `/opt/rocm`

---

## Config changes — `libero_spatial_grpo_openvlaoft.yaml`

| Field | Before | After | Reason |
|-------|--------|-------|--------|
| `defaults` | missing `- _self_` | added | Ensures main YAML overrides base defaults (Hydra 1.1+) |
| `env.train.total_num_envs` | 8 | 16 | `8 // 1 GPU // 1 stage % group_size 4 ≠ 0`; 16 satisfies the constraint |
| `actor.micro_batch_size` | 4 | 1 | Reduce per-step memory pressure |
| `actor.global_batch_size` | 128 | 32 | Reduce memory; match single-GPU scale |

**Validation constraint** (checked in `rlinf/config.py:validate_embodied_cfg`):
```
total_num_envs // env_world_size // pipeline_stage_num % group_size == 0
```

---

## Docker — why it doesn't work on this machine

The pre-built image `rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4` crashes with **SIGSEGV**
during model loading. Its PyTorch was compiled for a fixed arch list that excludes `gfx1151`:

```
['gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx942',
 'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201']
```

`HSA_OVERRIDE_GFX_VERSION` overrides don't help — the crash is at ROCm runtime init, before
kernel dispatch.

### Preferred fix: mount host .venv and ROCm into the container (TODO)

No rebuild required. Docker containers share the host kernel, so the ROCm devices (`/dev/kfd`,
`/dev/dri`) are host interfaces regardless. The working `torch==2.11.0+rocm7.2` from the native
install can be injected at runtime via volume mounts, bypassing the container's broken PyTorch:

```bash
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 20g \
    --network host \
    --name rlinf-amd-libero \
    -v /home/gberseth/playground/RLinf:/workspace/RLinf \
    -v /home/gberseth/playground/RLinf/.venv:/workspace/RLinf/.venv \
    -v /opt/rocm:/opt/rocm \
    -w /workspace/RLinf \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
    rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4 \
    bash -c "source /workspace/RLinf/.venv/bin/activate && MUJOCO_GL=osmesa ROBOT_PLATFORM=LIBERO bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft"
```

What each mount does:
- `.venv` — injects `torch==2.11.0+rocm7.2` (gfx1151-compatible) in place of the container's PyTorch
- `/opt/rocm` — injects host ROCm 7.2 runtime in place of the container's ROCm 6.4
- `LD_LIBRARY_PATH` — ensures the container resolves the mounted ROCm libs first

If linker errors appear (libstdc++, glibc mismatches), mount only `/opt/rocm/lib` instead of all
of `/opt/rocm`.

### Alternative fix: rebuild image with gfx1151 (several hours)

```bash
docker build \
    --build-arg PLATFORM=amd \
    --build-arg ROCM_VER=6.4 \
    --build-arg BUILD_TARGET=embodied-libero \
    --build-arg 'ROCM_ARCHS=gfx90a;gfx942;gfx1151' \
    -t rlinf-libero-rocm6.4-gfx1151 .
```
(Rebuilds PyTorch from source — takes several hours. Only needed if the mount approach has
unresolvable library compatibility issues.)

### Docker run command (for reference)

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

---

## Native install

Uses `torch==2.11.0+rocm7.2` — same build as `/home/gberseth/playground/mini-grp/.venv`,
which correctly sees the AMD Radeon 8060S.

### Install command

```bash
cd /home/gberseth/playground/RLinf
bash requirements/install.sh --platform amd --rocm 7.2 --no-root --no-flash-attn embodied --model openvla-oft --env libero
```

- `--no-root`: skips `sys_deps.sh` (requires passwordless sudo for apt; not needed when ROCm
  is already installed natively)
- `--no-flash-attn`: skips flash-attn source build (ROCm clang++ `<cmath>` not found on this
  system; flash-attn is optional — standard attention is used instead)

Creates `.venv` at `/home/gberseth/playground/RLinf/.venv`.

### install.sh fixes required for rocm7.2

`uv`'s `--torch-backend` only accepts values up to `rocm7.1`. For `rocm7.2`:
- `UV_TORCH_BACKEND` is left **unset** (avoids `invalid value` error)
- `UV_EXTRA_INDEX_URL` is set to the rocm7.2 pytorch index so `uv pip install openvla-oft`
  can resolve `torchaudio==2.11.0+rocm7.2` from that index

Fix is in `configure_amd()` in `requirements/install.sh`.

---

## Run training

```bash
cd /home/gberseth/playground/RLinf
source .venv/bin/activate
RAY_memory_usage_threshold=0.99 MUJOCO_GL=osmesa ROBOT_PLATFORM=LIBERO \
bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft
```

### Why `RAY_memory_usage_threshold=0.99`

The iGPU shares system RAM. Loading two copies of the 7B model (actor + rollout, each ~14 GB
in bf16) during weight sync peaks near 95% of 121 GB RAM. Ray's default kill threshold is 0.95,
which killed the rollout worker before the sync completed. Raising to 0.99 allows the sync to
finish without premature OOM kills.

If RAM remains tight, close other applications (browser, etc.) before running — Chrome/Firefox
were consuming ~3 GB during the OOM run.

### `run_embodiment.sh` patch

The `logs/` directory was owned by root (from prior Docker runs). `run_embodiment.sh` was patched
to fall back to `logs_local/` automatically when `logs/` is not writable. The permissions have
since been fixed, so the standard `logs/` path works again.

---

## Installation on Onyx (remote, NVIDIA)

Onyx is a remote workstation with 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 97 GB each),
driver 595.58, CUDA 13.2.

### Docker GPU fix

`nvidia-smi` worked but `docker run --gpus all` failed with:
```
failed to discover GPU vendor from CDI: no known GPU vendor found
```

Root cause: driver 595.58 requires CDI (Container Device Interface), but `nvidia-container-toolkit`
was not installed. Fix:

```bash
# Install toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Generate CDI spec and configure Docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Docker run command (Onyx)

```bash
docker run -it --gpus all \
    --shm-size 100g \
    --net=host \
    --name rlinf \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /home/gberseth/playground/RLinf:/workspace/RLinf \
    -w /workspace/RLinf \
    rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
    bash -c "source switch_env openvla-oft && bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft"
```

Note: `-v` mounts the host repo so config changes are picked up inside the container.
Config fixes (`_self_`, `pipeline_stage_num: 1`, `component_placement: "0"`, `total_num_envs: 4`)
were committed locally, pushed, and pulled on Onyx — training is now running.

### Docker CUDA architecture mismatch (second failure)

After the CDI fix, the container started but crashed during FSDP model init with:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Root cause: the pre-built image `rlinf/rlinf:agentic-rlinf0.2-maniskill_libero` uses CUDA 12.4.1
as its base, and `pyproject.toml` pins `torch==2.6.0` (cu124 wheels). Neither has compiled
kernels for sm_120 (Blackwell). CUDA 12.8+ is required to target sm_120, and PyTorch 2.7.0 is
the first stable release to ship cu128 wheels with sm_120 support.

Secondary issue: the container exited but was not removed (no `--rm`), so a subsequent
`docker run --name rlinf` fails with a name conflict. Always use `--rm` or `docker rm rlinf`
before re-running.

Fix: rebuild the image from a CUDA 12.8.1 base with PyTorch 2.7.0:

- `docker/Dockerfile`: `CUDA_VER=12.4.1` → `12.8.1`
- `pyproject.toml`: `torch==2.6.0` / `torchvision==0.21.0` / `torchaudio==2.6.0`
  → `torch==2.7.0` / `torchvision==0.22.0` / `torchaudio==2.7.0`

Build command (takes several hours):
```bash
docker build \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg NO_MIRROR=1 \
    -t rlinf-maniskill-libero-cuda128 \
    -f docker/Dockerfile .
```

#### Build failures due to flaky git clones

Two additional Dockerfile changes were needed to get the build to complete:

1. **GnuTLS recv error -54** (`Error in the pull function`) — git was using HTTP/2, which
   multiplexes connections and is sensitive to mid-transfer drops. Fixed by adding
   `http.version HTTP/1.1` to the global git config in the Dockerfile.

2. **All-or-nothing `RUN` step** — the original Dockerfile installed all six venvs
   (`openvla`, `openvla-oft`, `openpi`, `gr00t`, `dexbotic`, `starvla`) in a single
   `RUN` instruction. A git clone failure on any one venv invalidated the entire layer,
   forcing a full restart from scratch. Fixed by splitting into one `RUN` per venv so
   Docker caches each success independently.

3. **`UV_GIT_FETCH_WITH_CLI=true`** — forces uv to use the system git (and therefore the
   HTTP/1.1 config above) instead of its bundled git client, so the fix applies to both
   uv-sourced and install.sh-sourced git operations.

### Docker run command (Onyx, rebuilt image)

```bash
docker run -it --rm --gpus all \
    --shm-size 100g \
    --net=host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /home/gberseth/playground/RLinf:/workspace/RLinf \
    -w /workspace/RLinf \
    rlinf-maniskill-libero-cuda128 \
    bash -c "source switch_env openvla-oft && bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft"
```

Note: added `--rm` to auto-remove the container on exit, avoiding future name conflicts.

---

## Status

- [x] Config validation fixed (`total_num_envs`, `_self_`, `micro_batch_size`, `global_batch_size`)
- [x] Docker SIGSEGV root cause identified: gfx1151 not in compiled arch list
- [x] `install.sh` fixed for rocm7.2 (`UV_EXTRA_INDEX_URL`, unset `UV_TORCH_BACKEND`)
- [x] Native install succeeded: `torch==2.11.0+rocm7.2` sees AMD Radeon 8060S
- [x] Checkpoint shards load successfully (no SIGSEGV)
- [x] OOM fix: `RAY_memory_usage_threshold=0.99` + reduced batch sizes
- [x] Onyx: Docker GPU (CDI) fixed, config synced via git, training running
- [x] Onyx: `docker/Dockerfile` and `pyproject.toml` updated for CUDA 12.8.1 / torch 2.7.0
- [x] Onyx: Docker rebuild `rlinf-maniskill-libero-cuda128` — **complete** (139 GB image built successfully)
- [ ] Confirm full training loop completes a step without crashing (Onyx NVIDIA, rebuilt image)
- [x] Confirm full training loop completes a step without crashing (local AMD) — rollout epoch 1/4 completed, exited cleanly (exit code 0)
- [ ] Rebuild Docker image with gfx1151 support so training can run locally in Docker (see Docker section for build command)
