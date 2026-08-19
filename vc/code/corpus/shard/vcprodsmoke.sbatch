#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vcprodsmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%j.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%j.err
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
CODE=$DRAMABOX_BASE/vcbon/prod/code
PY=$DRAMABOX_BASE/env_mossaudio/bin/python
$PY -c "import torch, transformers, librosa, speechbrain, av, pyloudnorm; print('warmup ok', torch.__version__)"

# Shard list comes from a FILE, never from --export: sbatch splits --export on
# commas, so `--export=ALL,VC_SHARDS=a/000,b/001` silently delivers only `a/000`
# and tries to export `b/001` as a variable name. Cost one wasted job to learn.
SHARDS=$($PY -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))))" "$VC_SHARDFILE")
echo "shards: $SHARDS"
for R in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$R $PY -u $CODE/vcrun.py --rank $R --world 4 --shards "$SHARDS" &
  sleep 20
done
wait
echo "=== VCBON PRODSMOKE DONE $(date -Is) ==="
