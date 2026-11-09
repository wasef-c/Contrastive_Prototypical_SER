#!/usr/bin/env bash
# Settle whether the seed alone moves training, then run multi-seed on that
# basis.
#
# Background: full 8-epoch runs on the official split came out bitwise
# identical across seeds. Probing showed the generators, the data order and
# the sampler all vary with the seed, while model initialisation does not.
# A 1-epoch diagnostic at 10 percent of the data DID vary, but that
# subsample is itself seed-drawn, so it could not separate "the seed moves
# training" from "the subsample moves training".
#
# STAGE 1 removes the subsample: one epoch, full training split, two seeds.
# STAGE 2 reads the verdict and picks the queue.
#
#   losses differ -> the seed is enough. Run plain seeds 189 and 7 for both
#                    arms; seed 42 is already on disk and comparable.
#   losses match  -> the seed is not enough. Run the _s9 arms, where each
#                    seed draws a different nine tenths. Three seeds each,
#                    and seed 42 has to be rerun because dropping 10
#                    percent of the data changes the absolute numbers.
#
# Sequential throughout: one GPU, 31 GB of RAM, and this box needs a
# physical visit if it goes down.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness
CFG=configs/class_weight.yaml

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1

run_one () {
  # Declared one per line on purpose. Bash expands every word on a `local`
  # line before the builtin assigns any of them, so referring to $arm in
  # the same declaration that sets it is unbound under `set -u`.
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

while pgrep -f "runner.py --config" > /dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] GPU free"

echo
echo "########## STAGE 1: does the seed alone move training? ##########"
run_one rl_seeddiag_full 42
run_one rl_seeddiag_full 189

VERDICT=$("$PY" - <<'PYEOF'
import re
import sys

LOGS = "/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness"


def loss(seed: int) -> float:
    """Final train loss for one diagnostic seed, from its wandb summary."""
    path = "%s/rl_seeddiag_full_seed%d.log" % (LOGS, seed)
    vals = []
    with open(path, errors="replace") as handle:
        for line in handle:
            m = re.search(r"train/loss\s+([0-9.]+)\s*$", line)
            if m:
                vals.append(float(m.group(1)))
    return vals[-1] if vals else float("nan")


a, b = loss(42), loss(189)
print("LOSS42=%.6f LOSS189=%.6f" % (a, b), file=sys.stderr)
if a != a or b != b:
    print("UNKNOWN")
elif abs(a - b) < 1e-6:
    print("IDENTICAL")
else:
    print("DIFFERENT")
PYEOF
)
echo "[$(date +%H:%M:%S)] verdict: ${VERDICT}"
grep -h "SEED DIAG" "$LOGS"/rl_seeddiag_full_seed42.log \
    "$LOGS"/rl_seeddiag_full_seed189.log 2>/dev/null

echo
echo "########## STAGE 2: multi-seed ##########"
if [ "$VERDICT" = "DIFFERENT" ]; then
  echo "Seed alone is enough. Plain seeds, full data, comparable to disk."
  for S in 189 7; do
    run_one rl_hard "$S"
    run_one rl_lowrank "$S"
  done
elif [ "$VERDICT" = "IDENTICAL" ]; then
  echo "Seed alone is NOT enough. Using the _s9 arms for variance."
  for S in 42 189 7; do
    run_one rl_hard_s9 "$S"
    run_one rl_lowrank_s9 "$S"
  done
else
  echo "Could not read the diagnostic losses; stopping rather than guessing."
  exit 1
fi

echo "[$(date +%H:%M:%S)] queue finished"
