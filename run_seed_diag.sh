#!/usr/bin/env bash
# Localise the seeding bug, then run multi-seed results on that basis.
#
# Runs on the official split came out bitwise identical across seeds even
# though set_seed, the length-bucketed sampler and layer initialisation
# each vary with the seed when tested in isolation. Three probes now print
# at startup: the generator state right after set_seed, the sampler's seed
# and first batch, and the sum of the freshly initialised weights.
#
# Stage one is two one-epoch runs on a tenth of the data at different
# seeds, which is minutes rather than hours. Comparing the three probes
# says where the determinism enters:
#
#   probes differ, losses differ   -> seeding is fine, the earlier runs
#                                     must have differed for another reason
#   probes differ, losses match    -> not the data path or the init
#   probes match                   -> the named probe is the culprit
#
# Stage two reads that verdict and launches the multi-seed runs the right
# way: plain seeds if they now vary, otherwise train_subsample_frac so each
# seed draws a different nine tenths of the training rows.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness
CFG=configs/class_weight.yaml

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1

# Wait out any training still in flight; one GPU, and this box dies when
# two jobs share it.
while pgrep -f "runner.py --config" > /dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] GPU free, starting seed diagnostic"

for SEED in 42 189; do
  LOG="$LOGS/seeddiag_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] diag seed ${SEED} -> ${LOG}"
  "$PY" runner.py --config "$CFG" --experiment rl_seeddiag \
      --only-seeds "$SEED" > "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] diag seed ${SEED} exit=$?"
done

echo
echo "================ SEED DIAGNOSTIC ================"
for SEED in 42 189; do
  echo "--- seed ${SEED} ---"
  grep -E "SEED DIAG|Epoch 1/1|train/loss" "$LOGS/seeddiag_seed${SEED}.log" \
      | head -8
done
echo "================================================="
