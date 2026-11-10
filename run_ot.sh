#!/usr/bin/env bash
# Entropy-matched comparison for rl_lowrank.
#
# rl_lowrank gained +0.62 IEMOCAP UAR, but its targets are softer than
# the raw distribution and UAR has tracked softness across every arm
# here. These two put it in a three-point comparison at matched data:
# rl_raw_s9 at entropy 0.486, rl_smooth_s9 at 0.708 with no structure,
# and rl_lowrank_s9 at 0.708 with the projection. A tie against
# rl_smooth_s9 means rl_lowrank is label smoothing with extra steps.
#
# One seed each. Three seeds follow only if the direction is clear.
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
echo "[$(date +%H:%M:%S)] GPU free, starting comparison arms"

run_one rl_raw_s9 42
run_one rl_smooth_s9 42

echo "[$(date +%H:%M:%S)] comparison arms finished"
