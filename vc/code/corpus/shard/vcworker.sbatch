#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=vcbon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%A_%a-%N.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/prod/logs/%x-%A_%a-%N.err
set -x
source /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/env.sh
export HF_HUB_CACHE=$HF_HOME/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
CODE=$DRAMABOX_BASE/vcbon/prod/code
PY=$DRAMABOX_BASE/env_mossaudio/bin/python

# ONE serial warm-up import before anything parallel touches /e.  The documented
# 150 s stagger has proven insufficient on this filesystem: concurrent
# `import transformers` from /e puts every worker in D state.  Doing the import
# once, serially, warms the page cache so the four parallel workers hit RAM.
$PY -c "import torch, transformers, librosa, speechbrain, av, pyloudnorm; print('warmup ok', torch.__version__)"

DEADLINE=${VC_DEADLINE:-34200}   # stop claiming new shards 2.5 h before the 12 h walltime: one full shard measures ~2.1 h, so a later claim would be killed mid-flight
EXTRA=${VC_EXTRA:-}
RUNTAG=${VC_RUNTAG:-VC1}
OUTROOT=${VC_OUTROOT:-$DRAMABOX_BASE/vprof/vc500}

for R in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$R $PY -u $CODE/vcrun.py --rank $R --world 4 \
      --deadline-s $DEADLINE --run-tag $RUNTAG --out-root $OUTROOT \
      ${EXTRA:+--extra "$EXTRA"} &
  sleep 20            # cheap de-sync; the real protection is the warm-up above
done
wait
echo "=== VCBON WORKER DONE $(date -Is) ==="
