#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vcbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%j.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%j.err
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=$DRAMABOX_BASE/env_mossaudio/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u $DRAMABOX_BASE/vcbon/prod/code/vcsidonbench.py "$@"
echo "=== BENCH DONE ==="
