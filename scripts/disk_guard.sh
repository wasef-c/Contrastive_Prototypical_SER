#!/usr/bin/env bash
# Preflight disk check for training launchers.
#
# Source this and call `disk_guard <gb>` before each arm. On 2026-09-06 the
# disk filled to 98 percent, torch.save began truncating mid-write, and four
# arms failed with rc=120 while one process hung for 13 hours holding GPU
# memory. The retry loop could not help: every attempt hit the same full disk.
# Roughly 20 hours of GPU time was lost before anyone noticed.
#
# One arm needs about 4 GB: a 2.7 GB latest.pt rewritten each epoch plus a
# 1.2 GB final model in saved_models. The default threshold of 25 GB leaves
# room for several arms and for the cache to grow.

disk_guard () {
  local need_gb="${1:-25}"
  local free_gb
  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$need_gb" ]; then
    echo "=== $(date '+%F %T')  DISK GUARD: only ${free_gb} GB free, need ${need_gb} GB"
    echo "    Refusing to start. Free space with:"
    echo "      python scripts/prune_models.py            # show what can go"
    echo "      python scripts/prune_models.py --apply    # delete it"
    return 1
  fi
  echo "    disk ok: ${free_gb} GB free"
  return 0
}
