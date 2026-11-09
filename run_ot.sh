#!/usr/bin/env bash
# Transport-loss arms, queued behind whatever is already training.
#
# These are the first arms that leave the target alone and change how
# errors are priced. Both run at train_subsample_frac 0.9 so they are
# directly comparable to rl_hard_s9, which is the baseline they must beat.
#
# One seed each to start. The pair is the point: if ot_weight 2.0 moves
# nothing that 0.5 moved, the term is inert and no number of seeds will
# help. Seeds come after the direction is established.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness
CFG=configs/class_weight.yaml

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1

run_one () {
  local arm="$1"
  local seed="$2"
  local log="$LOGS/${arm}_seed${seed}.log"
  local status="checkpoints/${arm}_seed${seed}/status.json"
  if [ -f "$status" ] && grep -q '"status": "done"' "$status"; then
    echo "[$(date +%H:%M:%S)] SKIP ${arm} seed ${seed} (done)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${arm} seed ${seed}"
  "$PY" runner.py --config "$CFG" --experiment "$arm" \
      --only-seeds "$seed" > "$log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] DONE  ${arm} seed ${seed} exit=${code}"
  if [ "$code" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ABORT: ${arm} seed ${seed} failed"
    tail -20 "$log"
    exit "$code"
  fi
  sleep 15
}

# Wait for the variance sweep to finish; one GPU, and this box needs a
# physical visit if it goes down.
while pgrep -f "runner.py --config" > /dev/null 2>&1; do sleep 120; done
echo "[$(date +%H:%M:%S)] GPU free, starting transport arms"

run_one rl_ot 42
run_one rl_ot_hi 42

echo "[$(date +%H:%M:%S)] transport arms finished"
grep -h "Transport term active" "$LOGS"/rl_ot_seed42.log \
    "$LOGS"/rl_ot_hi_seed42.log 2>/dev/null
