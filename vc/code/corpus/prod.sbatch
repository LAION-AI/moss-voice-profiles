#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --job-name=vcprod
#SBATCH --output=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/logs/prod_%A_%a.out
#SBATCH --error=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/vcbon/logs/prod_%A_%a.out
#SBATCH --array=0-47
set -eo pipefail
NB=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox
export HF_HOME=$NB HF_HUB_CACHE=$NB/hfcache/.cache/dramabox
export HUGGINGFACE_HUB_CACHE=$NB/hfcache/.cache/dramabox
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=$NB/tmp
export XDG_CACHE_HOME=$NB/hfcache/xdg TRITON_CACHE_DIR=$NB/hfcache/triton
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0
cd $NB
# 4 workers per node, one per GPU. Voices are claimed with O_EXCL so workers never collide,
# and --walltime-s stops a worker CLAIMING a new voice it cannot finish before the wall.
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $NB/env_transcribe/bin/python -u vcbon/code/prod.py \
    --out $NB/vcbon/prod/run1 --n-cand 4 --walltime-s 18000 \
    > $NB/vcbon/logs/prod_w${SLURM_ARRAY_TASK_ID}_$i.log 2>&1 &
  sleep 150
done
wait
echo VCPROD_NODE_DONE
