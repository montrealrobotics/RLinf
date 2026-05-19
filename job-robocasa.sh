#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=120:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@mail.com
#SBATCH --exclude=cn-g011

# Job script for RoboCasa CloseDrawer PPO fine-tuning with π0 on a single A100L
# Requires: RLinf installed in .venv, RLinf-Pi0-RoboCasa model downloaded
# Adapt model paths in examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml before running
# Submit with: sbatch job-robocasa.sh

set -e
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cleanup loop: automatically deletes old checkpoints, keeping only the 2 most recent
# Adapt the path below to match your project directory and experiment name
(
  while true; do
    sleep 1800
    ls -dt /path/to/RLinf/logs/*/robocasa_closedrawer_ppo_openpi/checkpoints/global_step_* 2>/dev/null | tail -n +3 | xargs rm -rf
  done
) &
CLEANUP_PID=$!
trap "kill $CLEANUP_PID 2>/dev/null" EXIT

# Adapt the path below to your project directory
cd /path/to/RLinf
source .venv/bin/activate

exec srun bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_ppo_openpi
