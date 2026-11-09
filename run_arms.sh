#!/usr/bin/env bash
# Continuum-target arms, one at a time, one seed.
#
# Sequential because this box has one GPU and 31 GB of RAM and goes down
# when overcommitted, which needs a physical visit to fix.
#
# PYTHONUNBUFFERED so the logs flush live. Without it Python block-buffers
# stdout into a file and a healthy run reads like a stall.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness
CFG=configs/class_weight.yaml

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1

QUEUE="rl_lowrank:42 rl_kernel:42"

for JOB in $QUEUE; do
  ARM="${JOB%%:*}"
  SEED="${JOB##*:}"
  LOG="$LOGS/${ARM}_seed${SEED}.log"
  STATUS_FILE="checkpoints/${ARM}_seed${SEED}/status.json"

  # Skip only a run that actually FINISHED. Testing for the directory
  # alone would skip a run killed part way through that left a checkpoint.
  if [ -f "$STATUS_FILE" ] && grep -q '"status": "done"' "$STATUS_FILE"; then
    echo "[$(date +%H:%M:%S)] SKIP ${ARM} seed ${SEED} (already done)"
    continue
  fi

  echo "[$(date +%H:%M:%S)] START ${ARM} seed ${SEED} -> ${LOG}"
  "$PY" runner.py --config "$CFG" --experiment "$ARM" \
      --only-seeds "$SEED" > "$LOG" 2>&1
  CODE=$?
  echo "[$(date +%H:%M:%S)] DONE  ${ARM} seed ${SEED} exit=${CODE}"
  if [ "$CODE" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ABORT: ${ARM} seed ${SEED} failed; stopping"
    tail -25 "$LOG"
    exit "$CODE"
  fi
  sleep 20
done

echo "[$(date +%H:%M:%S)] queue finished"
