#!/usr/bin/env bash
# VAD-matched triplet arms, interleaved by seed.
#
# Order is (vad, rand) for seed 42, then seed 189, then seed 7, so
# stopping early still leaves COMPLETE PAIRED SEEDS. The claim is the
# difference between the two arms at the same seed; three seeds of the
# treatment and one of the control would not support it.
#
# One GPU, so strictly sequential. This box needs a physical visit if RAM
# is overcommitted, which is why nothing here runs in parallel.
#
# Note on `local`: declaring and expanding in one statement, as in
#   local arm="$1" log="${arm}.log"
# is unbound under `set -u`, because the expansion happens before the
# assignment lands. Each declaration gets its own line below.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
CFG="$WT/configs/class_weight.yaml"
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness
DRIVER="$LOGS/trip_driver.log"

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1
mkdir -p "$LOGS"

# Do not start on top of another training run.
#
# Asked by GPU occupancy, not by process name. A pgrep on "runner.py"
# also matches any shell whose command line merely CONTAINS that text,
# including an interactive check for it, which deadlocks this loop
# against an observer. nvidia-smi answers the actual question.
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
          | grep -c .)" -gt 0 ]; do sleep 120; done
echo "[$(date +%H:%M:%S)] GPU free, starting" >> "$DRIVER"

run_one () {
  local arm
  local seed
  local log
  arm="$1"
  seed="$2"
  log="$LOGS/${arm}_seed${seed}.log"
  echo "[$(date +%H:%M:%S)] START $arm seed $seed" >> "$DRIVER"
  "$PY" runner.py --config "$CFG" -e "$arm" --only-seeds "$seed" \
      > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $arm seed $seed exit=$?" >> "$DRIVER"
}

for seed in 42 189 7; do
  run_one rl_trip_vad  "$seed"
  run_one rl_trip_rand "$seed"
  echo "[$(date +%H:%M:%S)] --- seed $seed pair complete ---" >> "$DRIVER"
  "$PY" -m analysis.score_triplets >> "$LOGS/trip_scores.log" 2>&1
done

echo "[$(date +%H:%M:%S)] all seeds complete" >> "$DRIVER"
