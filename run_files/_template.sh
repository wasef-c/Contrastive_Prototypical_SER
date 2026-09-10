#!/usr/bin/env bash
# Launcher template. Copy this, then append run_arm lines.
#
# Includes the preflight disk guard added after 2026-09-06, when the disk
# filled to 98 percent, torch.save truncated mid-write, four arms died with
# rc=120 and one process hung for 13 hours holding GPU memory. About 20 hours
# of GPU time was lost. The retry loop made it worse rather than better: each
# arm burned three attempts against the same full disk.
set -u
cd /home/rml/Documents/pythontest/Emotion2Vec_Contrastive
source env_fast.sh 2>/dev/null || true
source scripts/disk_guard.sh

PY_BIN=/home/rml/Documents/pythontest/.venv/bin/python
LOG=logs
INHERITED="${1:-}"
if [ -n "$INHERITED" ]; then
  echo "=== $(date '+%F %T')  waiting on inherited pid $INHERITED"
  while [ -d "/proc/$INHERITED" ]; do sleep 60; done
  echo "=== $(date '+%F %T')  inherited pid $INHERITED exited"
fi

run_arm () {
  local cfg="$1"; local arm="$2"; local attempt rc
  # Check before starting, not after failing. A full disk is not transient,
  # so retrying into it wastes the queue rather than recovering it.
  if ! disk_guard 25; then
    echo "=== $(date '+%F %T')  ABORTING QUEUE before $arm"
    exit 1
  fi
  for attempt in 1 2 3; do
    echo "=== $(date '+%F %T')  START $arm (attempt $attempt)"
    for d in checkpoints/${arm}_seed*; do
      [ -d "$d" ] || continue
      if [ -f "$d/results.json" ]; then echo "    keeping completed $d"
      else echo "    clearing incomplete $d"; rm -rf "$d"; fi
    done
    "$PY_BIN" runner.py --config "$cfg" -e "$arm" >> "$LOG/${arm}.log" 2>&1
    rc=$?
    echo "=== $(date '+%F %T')  END   $arm  rc=$rc"
    [ "$rc" -eq 0 ] && break
    # rc 120 has meant a full disk here. Do not spend the other attempts.
    if [ "$rc" -eq 120 ] && ! disk_guard 25; then
      echo "=== $(date '+%F %T')  ABORTING QUEUE: $arm failed and the disk is full"
      exit 1
    fi
    echo "    attempt $attempt failed; pausing 120s"; sleep 120
  done
}

# Before adding arms, check the config does NOT set trace_test_each_epoch.
# It evaluates every test corpus each epoch, costs about 23 percent wall time,
# and is a diagnostic only. Off by default; only the sv_* and rv_* speaker
# validation arms ever used it.
#
# run_arm configs/class_weight.yaml <arm_name>
