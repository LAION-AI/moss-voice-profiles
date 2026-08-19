#!/bin/bash
# Restart wrapper. The sweeper must survive a multi-day wait on the repair gate
# plus a ~23 h run; a login-node hiccup that kills it must not silently end the
# production run. Relaunches on any exit except a clean "all shards complete".
NB=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox
PROD=$NB/vcbon/prod
PY=$NB/env_mossaudio/bin/python
mkdir -p $PROD/logs
while true; do
  echo "[sweepd] starting vcsweep at $(date -Is)" >> $PROD/logs/sweepd.log
  $PY -u $PROD/code/vcsweep.py >> $PROD/logs/sweep.log 2>&1
  rc=$?
  echo "[sweepd] vcsweep exited rc=$rc at $(date -Is)" >> $PROD/logs/sweepd.log
  if [ $rc -eq 0 ] && grep -q "all shards complete" $PROD/logs/sweep.log; then
    echo "[sweepd] run finished, not restarting" >> $PROD/logs/sweepd.log
    break
  fi
  sleep 60
done
