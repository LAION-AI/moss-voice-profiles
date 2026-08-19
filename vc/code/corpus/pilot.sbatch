#!/bin/bash -x
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vcbon-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/logs/%x-%j.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/logs/%x-%j.err
set -x
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
CODE=$DRAMABOX_BASE/vcbon/code
PY=$DRAMABOX_BASE/env_mossaudio/bin/python

# one serial warm-up import before anything parallel touches /e
$PY -c "import torch, transformers, librosa; print('warmup ok', torch.__version__)"

srun $PY -u $CODE/pilot.py "$@"
echo "=== VCBON PILOT DONE ==="
