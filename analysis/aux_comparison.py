"""Comparison table over auxiliary targets, all in one training setup.

The paper's central exhibit. Every arm shares the backbone, the optimiser, the
schedule, the class weighting and the seeds; only the auxiliary target
changes. For each target the table reports the gain a conventional evaluation
would show (against a no-auxiliary baseline) beside the gain that survives a
capacity-matched control, whose head has the same shape and gradient path but
a target carrying no information about the sample.

The comparison arms are run at three seeds and the core arms at five, so the
paired statistics for each row use only the seeds that row actually has. The
n column makes that explicit rather than hiding it.

Usage:
    python analysis/aux_comparison.py [--corpus IEMO] [--depth uf4]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Dict, List, Sequence, Tuple

Runs = Dict[str, Dict[str, Dict[str, float]]]

# Auxiliary target -> (display name, prior work it stands in for, arm names).
# Arms are listed in the order they should appear in the paper.
ARMS_UF4: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("None (baseline)", "", ("uf4_base", "uf4_base_more")),
    ("Permuted target", "capacity control", ("uf4_ctrl", "uf4_ctrl_s2024")),
    ("Permuted, pooled draw", "capacity control", ("uf4_ctrl_pool",)),
    ("Raw VAD", "Xia and Liu 2017", ("uf4_rawvad",)),
    ("Annotator agreement", "Kim and Provost 2015", ("uf4_agree",)),
    ("Annotator dispersion", "Eyben et al. 2012", ("uf4_disp",)),
    ("Off-label distribution", "secondary labels", ("uf4_offlabel",)),
    ("VAD subtype clusters", "own prior work", ("uf4_auxvad",)),
    ("Residual, unwhitened", "ablation rung", ("uf4_euclid",)),
    ("Residual, whitened (ours)", "", ("uf4_res", "uf4_res_more")),
]

ARMS_CW: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("None (baseline)", "", ("cw_sqrt_base", "cw_sqrt_base_more")),
    ("Permuted target", "capacity control", ("cw_sqrt_ctrl",)),
    ("Raw VAD", "Xia and Liu 2017", ("sens_rawvad_res",)),
    ("Residual, unwhitened", "ablation rung", ("sens_euclid_res",)),
    ("Residual, shared whitening", "ablation rung", ("wsh_res", "wsh_res_more")),
    ("Residual, whitened (ours)", "", ("cw_sqrt_res", "cw_sqrt_res_more")),
]


def load(*arms: str) -> Runs:
    """Load per-seed, per-corpus test metrics, merging arms split across names.

    Args:
        arms: Experiment names whose checkpoint directories are merged.

    Returns:
        Mapping of seed -> corpus -> metric -> value.
    """
    out: Runs = {}
    for arm in arms:
        for path in sorted(glob.glob(f"checkpoints/{arm}_seed*/results.json")):
            seed = os.path.basename(os.path.dirname(path)).split("seed")[-1]
            with open(path) as handle:
                out[seed] = {e["dataset"]: e["results"]
                             for e in json.load(handle)["test_results"]}
    return out


def paired(a: Runs, b: Runs, corpus: str, metric: str,
           seeds: Sequence[str]) -> Tuple[float, float, int]:
    """Mean paired difference, t statistic and win count, scaled by 100."""
    diffs = [100.0 * (a[s][corpus][metric] - b[s][corpus][metric]) for s in seeds]
    mean = st.mean(diffs)
    if len(diffs) < 2:
        return mean, float("nan"), sum(1 for d in diffs if d > 0)
    sd = st.stdev(diffs)
    t = mean / (sd / len(diffs) ** 0.5) if sd else float("nan")
    return mean, t, sum(1 for d in diffs if d > 0)


def main() -> None:
    """Print the auxiliary-target comparison for one corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="IEMO")
    parser.add_argument("--depth", default="uf4", choices=["uf4", "cw"])
    args = parser.parse_args()

    spec = ARMS_UF4 if args.depth == "uf4" else ARMS_CW
    loaded = [(name, src, load(*arms)) for name, src, arms in spec]
    base = loaded[0][2]
    ctrl = loaded[1][2]

    depth = "4/4 unfrozen" if args.depth == "uf4" else "2/2 unfrozen"
    print(f"Auxiliary target comparison, {depth}, corpus {args.corpus}")
    print("Each row: same backbone, optimiser, schedule and seeds; only the "
          "auxiliary target differs.\n")
    head = (f"  {'auxiliary target':27s} {'stands in for':22s} {'n':>2s} "
            f"{'UAR':>13s} {'vs base':>9s} {'vs ctrl':>9s} "
            f"{'AUCneu':>13s} {'vs base':>9s} {'vs ctrl':>9s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, src, runs in loaded:
        if not runs:
            print(f"  {name:27s} {src:22s} {'--':>2s}   not yet run")
            continue
        seeds = sorted(set(runs) & set(base) & set(ctrl), key=int)
        cells = []
        for metric in ("uar", "neutral_auc"):
            vals = [100.0 * runs[s][args.corpus][metric] for s in seeds]
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{st.mean(vals):.2f} ({sd:.2f})")
            for ref in (base, ctrl):
                mean, t, win = paired(runs, ref, args.corpus, metric, seeds)
                mark = "" if abs(t) != abs(t) else ("**" if abs(t) > 4.604
                                                   else "*" if abs(t) > 2.776 else "")
                cells.append(f"{mean:+.2f}{mark}")
        print(f"  {name:27s} {src:22s} {len(seeds):2d} "
              f"{cells[0]:>13s} {cells[1]:>9s} {cells[2]:>9s} "
              f"{cells[3]:>13s} {cells[4]:>9s} {cells[5]:>9s}")
    print("\n  * p<0.05, ** p<0.01 on the paired t test over that row's seeds.")
    print("  'vs ctrl' uses the batch-permutation control.")


if __name__ == "__main__":
    main()
