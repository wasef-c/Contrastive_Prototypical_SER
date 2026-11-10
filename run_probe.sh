#!/usr/bin/env bash
# Within-group probe, queued behind training.
#
# One GPU, and this box needs a physical visit if RAM is overcommitted, so
# nothing runs concurrently with a training arm.
set -u

WT=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/.claude/worktrees/levers
PY=/home/rml/Documents/pythontest/.venv/bin/python
LOGS=/home/rml/Documents/pythontest/Emotion2Vec_Contrastive/logs_realness

export PYTHONUNBUFFERED=1
cd "$WT" || exit 1

while pgrep -f "runner.py --config" > /dev/null 2>&1; do sleep 120; done
echo "[$(date +%H:%M:%S)] GPU free, running within-group probe"
"$PY" -m analysis.within_group_probe > "$LOGS/within_group_probe.log" 2>&1
echo "[$(date +%H:%M:%S)] probe exit=$?"
tail -32 "$LOGS/within_group_probe.log"
