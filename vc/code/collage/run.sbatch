#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vc4arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vc4arm/logs/%x-%j.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vc4arm/logs/%x-%j.err
set -x
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
CODE=$DRAMABOX_BASE/vc4arm/code
PY=$DRAMABOX_BASE/env_mossaudio/bin/python

# ONE serial warm-up import before anything else touches /e (the documented
# 150 s stagger is insufficient on this filesystem).  Single-worker job here,
# but the warm-up also front-loads the page-cache cost out of the timed section.
$PY -c "import torch, transformers, librosa, speechbrain, av, pyloudnorm, chatterbox; print('warmup ok', torch.__version__)"

srun $PY -u $CODE/$1 "${@:2}"
echo "=== VC4ARM STEP DONE $(date -Is) ==="
