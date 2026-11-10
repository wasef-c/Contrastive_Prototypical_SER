#!/usr/bin/env bash
# Analysis queue: prototype modes, the detector, then the within-group probe.
#
# All three are forward passes and analysis, no training. Sequential and
# queued behind any training arm: one GPU, and this box needs a physical
# visit if RAM is overcommitted.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1

while pgrep -f "runner.py --config" > /dev/null 2>&1; do sleep 120; done
echo "[$(date +%H:%M:%S)] GPU free"

stage () {
  local mod="$1"
  local log="$LOGS/${mod}.log"
  echo
  echo "[$(date +%H:%M:%S)] === ${mod} ==="
  "$PY" -m "analysis.${mod}" > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] ${mod} exit=$?"
  tail -46 "$log"
}

stage prototype_modes
stage ambiguity_detector
stage within_group_probe

echo
echo "[$(date +%H:%M:%S)] analysis queue finished"
