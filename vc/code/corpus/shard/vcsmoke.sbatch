#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vcsmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%j.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%j.err
set -x
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
CODE=$DRAMABOX_BASE/vcbon/prod/code
PY=$DRAMABOX_BASE/env_mossaudio/bin/python
$PY -c "import torch, transformers, librosa, speechbrain, av, pyloudnorm; print('warmup ok', torch.__version__)"
CUDA_VISIBLE_DEVICES=0 $PY -u $CODE/vcprod.py "$@"
echo "=== VCBON SMOKE DONE $(date -Is) ==="
