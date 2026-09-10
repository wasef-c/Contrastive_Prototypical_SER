"""Report and prune saved model weights that no current work references.

`saved_models/` grows by about 1.2 GB per completed run and nothing removes
old entries. By 2026-09-06 it held 557 files totalling 641 GB, which filled
the disk and silently broke four training arms: torch.save truncated
mid-write, runs exited with rc=120, and one process hung for 13 hours holding
GPU memory.

Nothing in the analysis pipeline reads these files. Every table, metric and
confusion matrix comes from checkpoints/<arm>/results.json and
checkpoints/<arm>/predictions/*.npz, which together are about 100 MB. The
weights are only needed to resume training, to evaluate on a new corpus, or
to release a model, so old experiments can go.

Usage:
    python scripts/prune_models.py            # report only, deletes nothing
    python scripts/prune_models.py --apply    # delete the unreferenced files
    python scripts/prune_models.py --keep-prefix foo_   # protect extra arms
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Sequence, Tuple

# Arms referenced by the current paper. Anything whose filename starts with
# one of these is kept. Extend this list rather than editing the logic.
KEEP_PREFIXES: Tuple[str, ...] = (
    "uf4_",           # 4/4 unfrozen, the headline configuration
    "uf8_",           # full-unfreeze probe
    "cw_sqrt_",       # 2/2 headline, now the capacity appendix
    "ctrl_pool", "ctrl_gauss",   # alternative control constructions
    "sens_rawvad", "sens_euclid", "wsh_res", "sens_hidden256",   # ablation ladder
    "wl_sqrt_",       # WavLM encoder comparison
    "auxvad_only", "protoclust_res", "presid_res", "resdisp_res",
    "pcgrad_res", "anneal8_res",                                  # target and delivery variants
    "iemo_", "iemo16_",          # IEMOCAP-trained direction
    "sv_", "sv4_", "rv_",        # speaker-disjoint validation study
)

GIB = 2 ** 30


def classify(paths: Sequence[str],
             keep_prefixes: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split model files into those to keep and those to drop.

    Args:
        paths: Paths to candidate .pt files.
        keep_prefixes: Filename prefixes that mark a file as still needed.

    Returns:
        Tuple of (keep, drop) path lists.
    """
    keep: List[str] = []
    drop: List[str] = []
    for path in paths:
        name = os.path.basename(path)
        (keep if name.startswith(tuple(keep_prefixes)) else drop).append(path)
    return keep, drop


def total_gib(paths: Sequence[str]) -> float:
    """Summed size of the given files in GiB."""
    return sum(os.path.getsize(p) for p in paths) / GIB


def main() -> None:
    """Report, and optionally delete, unreferenced saved models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; default is a dry run")
    parser.add_argument("--keep-prefix", action="append", default=[],
                        help="extra filename prefix to protect, repeatable")
    parser.add_argument("--dir", default="saved_models")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    if not paths:
        print(f"no .pt files under {args.dir}")
        return

    keep, drop = classify(paths, tuple(KEEP_PREFIXES) + tuple(args.keep_prefix))
    print(f"{args.dir}: {len(paths)} files, {total_gib(paths):.0f} GB")
    print(f"  keep {len(keep):4d} files, {total_gib(keep):6.0f} GB  (current paper arms)")
    print(f"  drop {len(drop):4d} files, {total_gib(drop):6.0f} GB  (older experiments)")

    if not drop:
        print("nothing to prune")
        return

    if not args.apply:
        print("\ndry run. Sample of what would be deleted:")
        for path in drop[:10]:
            print("   ", os.path.basename(path))
        if len(drop) > 10:
            print(f"    ... and {len(drop) - 10} more")
        print("\nrerun with --apply to delete")
        return

    freed = 0.0
    for path in drop:
        freed += os.path.getsize(path)
        os.remove(path)
    print(f"\ndeleted {len(drop)} files, {freed / GIB:.0f} GB")


if __name__ == "__main__":
    main()
