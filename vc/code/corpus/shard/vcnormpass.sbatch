#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vcnormpass
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%A_%a.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%A_%a.err
set -x
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
CODE=$DRAMABOX_BASE/vcbon/prod/code
PY=$DRAMABOX_BASE/env_mossaudio/bin/python
$PY -c "import torch, transformers, librosa, speechbrain, av, pyloudnorm; print('warmup ok', torch.__version__)"

SHARDS=$($PY -c "import json;print(','.join(json.load(open('$DRAMABOX_BASE/vcbon/prod/index/normpass_shards.json'))))")
NARR=${VC_NARRAY:-5}
AID=${SLURM_ARRAY_TASK_ID:-1}
WORLD=$((NARR*4))
for R in 0 1 2 3; do
  GR=$(( (AID-1)*4 + R ))
  CUDA_VISIBLE_DEVICES=$R $PY -u $CODE/vcrun.py --rank $GR --world $WORLD \
      --shards "$SHARDS" --run-tag NP1 \
      --out-root $DRAMABOX_BASE/vcbon/prod/smoke \
      --extra "--limit 2000 --ab-presidon 1 --no-marker 0 --norm NONE" &
  sleep 20
done
wait
echo "=== VCBON NORMPASS DONE $(date -Is) ==="
