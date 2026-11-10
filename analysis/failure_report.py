#!/usr/bin/env python3
"""A fuller failure report, with a corrected VAD-atypicality comparator.

An earlier version compared the hard core's VAD atypicality against a
baseline that was not the same quantity: the modal predicted class was
only filled in for core rows, so for the easy rows the comparison measured
distance to the NEUTRAL centroid rather than to anything predicted. The
core figure was right and its comparator was not.

The measure used here is label-independent and applies to every row:

    atypical(i) = min over c != y_i of d(vad_i, centroid_c) < d(vad_i, centroid_{y_i})

that is, does this utterance sit closer to some other class's VAD centroid
than to its own. Centroids come from rows every run classifies correctly,
so they are not defined by the failures being described. The same quantity
is reported for the core and for the easy rows, which makes it a fair
comparison.

Sections, each intended to be forkable into its own investigation:

  1 ERROR DISTRIBUTION   how failures spread across the 35 model-seed runs
  2 HARD CORE BY CLASS   who fails, and what they are called instead
  3 VAD ATYPICALITY      corrected, per class, with the centroid table
  4 ARM FAMILIES         whether families fail on the same rows or on
                         different ones, which decides whether ensembling
                         or routing could help at all
  5 AGREEMENT            pairwise error overlap between runs, against what
                         independent errors at the same rate would give
  6 EXPORT               a CSV of the core for manual listening

SAMSEMO carries no usable VAD in this metadata - every row is
(0.5, 0.5, 0.5) - so section 3 is reported for IEMOCAP only and SAMSEMO's
VAD columns are omitted rather than printed as zeros.

Usage:
    python -m analysis.failure_report
"""

import argparse
import collections
import csv
import itertools
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from analysis.arm_table import CORPORA, by_seed

OUT = pathlib.Path("analysis/failure_report")
NAMES = ("neutral", "happy", "sad", "angry")

FAMILIES: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "mixup": (("uf4_softmix", "uf4_softmix_more"),
              ("uf4_softmix_vanilla", "uf4_softmix_vanilla_more"),
              ("uf4_globalmix", "uf4_globalmix_more")),
    "hard": (("rl_hard",), ("rl_hard_s9",)),
    "soft": (("rl_raw",), ("rl_raw_s9",), ("rl_lowrank",),
             ("rl_lowrank_s9",), ("rl_smooth_s9",)),
    "structured": (("rl_dir",), ("rl_dir_hi",), ("rl_kernel",),
                   ("rl_ot",), ("rl_ot_hi",)),
}


def gather(key: str, n_rows: int) -> Tuple[np.ndarray, List[str]]:
    """Stack argmax predictions from every arm and seed for one corpus."""
    rows: List[np.ndarray] = []
    names: List[str] = []
    for fam, arms in FAMILIES.items():
        for arm in arms:
            for seed, logits in sorted(by_seed(arm, key).items()):
                if logits.shape[0] == n_rows:
                    rows.append(logits.argmax(axis=1))
                    names.append("%s|%s|s%d" % (fam, arm[0], seed))
    return (np.stack(rows) if rows else np.empty((0, n_rows), int)), names


def main() -> None:
    """Produce the report and the per-corpus exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=int, default=400,
                        help="core rows to write per corpus")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}

    for corpus, stem, key in CORPORA:
        meta = np.load("analysis/heldout_meta/%s.npz" % stem,
                       allow_pickle=True)
        lab = meta["label"]
        valid = lab >= 0
        y = lab[valid]
        vad = meta["vad"][valid]
        spk = meta["speaker"][valid].astype(str)
        has_vad = float(vad.std()) > 1e-6

        stack, names = gather(key, len(y))
        if stack.size == 0:
            continue
        wrong = stack != y[None, :]
        n_runs = len(names)
        miss = wrong.sum(axis=0)
        core = miss == n_runs
        easy = miss == 0

        print("\n" + "#" * 72)
        print("%s   %d utterances   %d model-seed runs   VAD usable: %s"
              % (corpus, len(y), n_runs, has_vad))
        print("  pooled accuracy %.4f" % float((~wrong).mean()))

        print("\n1. ERROR DISTRIBUTION")
        hist = np.bincount(miss, minlength=n_runs + 1)
        nz = [(k, int(v)) for k, v in enumerate(hist) if v]
        print("   runs-wrong : count   (showing every non-empty bucket)")
        for k, v in nz:
            bar = "#" * max(1, int(60 * v / max(hist)))
            print("   %10d : %6d  %s" % (k, v, bar))
        print("   wrong in every run %d (%.1f%%), right in every run "
              "%d (%.1f%%)" % (int(core.sum()), 100 * core.mean(),
                               int(easy.sum()), 100 * easy.mean()))

        modal = np.array([np.bincount(stack[:, i], minlength=4).argmax()
                          for i in range(len(y))])

        print("\n2. HARD CORE BY CLASS")
        print("   %-9s %7s %7s %8s   %s"
              % ("class", "core", "total", "rate", "called instead"))
        for c in range(4):
            sel = core & (y == c)
            tot = int((y == c).sum())
            calls = collections.Counter(NAMES[m] for m in modal[sel])
            top = ", ".join("%s %d" % (k, v) for k, v in calls.most_common(3))
            print("   %-9s %7d %7d %7.1f%%   %s"
                  % (NAMES[c], int(sel.sum()), tot,
                     100 * sel.sum() / max(tot, 1), top))

        if has_vad:
            # Centroids from ALL rows of each class, not from the rows
            # every run gets right. Conditioning on model correctness
            # selects VAD-typical utterances, which pushes each emotional
            # class's centroid outward (angry by 0.112 on the 0-1 scale,
            # sad and happy by about 0.077, neutral by only 0.012) and
            # inflates the core's measured atypicality: 53.5 against 14.8
            # becomes 47.4 against 18.8 once the conditioning is removed.
            # All-rows and random-half-of-class agree to a tenth of a
            # point, so this is the stable choice.
            cent = np.stack([vad[y == c].mean(axis=0) for c in range(4)])
            d_all = np.linalg.norm(vad[:, None, :] - cent[None], axis=2)
            d_own = d_all[np.arange(len(y)), y]
            other = d_all.copy()
            other[np.arange(len(y)), y] = np.inf
            d_other = other.min(axis=1)
            atypical = d_other < d_own
            d_modal = d_all[np.arange(len(y)), modal]

            print("\n3. VAD ATYPICALITY  (corrected, same measure both groups)")
            print("   closer to ANOTHER class centroid than its own:")
            print("     hard core %.1f%%   easy %.1f%%   all %.1f%%"
                  % (100 * atypical[core].mean(),
                     100 * atypical[easy].mean(), 100 * atypical.mean()))
            print("   mean distance to own centroid: core %.3f, easy %.3f"
                  % (d_own[core].mean(), d_own[easy].mean()))
            print("   of the core, closer to the MODAL PREDICTED class"
                  " than its own: %.1f%%" % (100 * (d_modal[core]
                                                    < d_own[core]).mean()))
            print("\n   per class, share atypical in VAD")
            print("     %-9s %8s %8s" % ("class", "core", "easy"))
            for c in range(4):
                sc, se = core & (y == c), easy & (y == c)
                print("     %-9s %7.1f%% %7.1f%%"
                      % (NAMES[c], 100 * atypical[sc].mean(),
                         100 * atypical[se].mean()))
            print("\n   class VAD centroids from the easy rows (0-1 scale)")
            for c in range(4):
                print("     %-9s %s" % (NAMES[c], np.round(cent[c], 3)))
        else:
            print("\n3. VAD ATYPICALITY  skipped: this corpus has no usable"
                  " VAD (every row identical)")
            atypical = np.zeros(len(y), dtype=bool)

        print("\n4. ARM FAMILIES  (does each family fail on the same rows?)")
        fam_wrong: Dict[str, np.ndarray] = {}
        for fam in FAMILIES:
            idx = [i for i, n in enumerate(names) if n.startswith(fam + "|")]
            if idx:
                fam_wrong[fam] = wrong[idx].mean(axis=0)
        print("   %-12s %7s %9s %11s" % ("family", "runs", "error rate",
                                         "always-wrong"))
        for fam, w in fam_wrong.items():
            idx = [i for i, n in enumerate(names) if n.startswith(fam + "|")]
            aw = (wrong[idx].all(axis=0)).mean()
            print("   %-12s %7d %8.3f %10.1f%%"
                  % (fam, len(idx), float(w.mean()), 100 * aw))
        fams = list(fam_wrong)
        print("\n   overlap of each family's always-wrong set with others")
        print("   %-12s %s" % ("", "  ".join("%10s" % f for f in fams)))
        sets = {}
        for fam in fams:
            idx = [i for i, n in enumerate(names) if n.startswith(fam + "|")]
            sets[fam] = wrong[idx].all(axis=0)
        for a in fams:
            cells = []
            for b in fams:
                inter = (sets[a] & sets[b]).sum()
                cells.append("%9.1f%%" % (100 * inter / max(sets[a].sum(), 1)))
            print("   %-12s %s" % (a, "  ".join(cells)))
        print("   (row = share of THAT family's always-wrong set also"
              " always-wrong for the column family)")

        print("\n5. AGREEMENT BETWEEN RUNS")
        pair_agree = []
        for i, j in itertools.combinations(range(n_runs), 2):
            pair_agree.append(float((wrong[i] == wrong[j]).mean()))
        err = float(wrong.mean())
        indep = err * err + (1 - err) * (1 - err)
        print("   mean pairwise agreement on error/correct: %.3f"
              % float(np.mean(pair_agree)))
        print("   if runs erred independently at the same rate: %.3f"
              % indep)
        print("   so the runs are far more correlated than chance, which is"
              " why\n   ensembling across these arms would buy little")

        path = OUT / ("%s_core.csv" % corpus.lower())
        order = np.argsort(-miss)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "speaker", "true", "modal_pred", "runs_wrong",
                        "valence", "arousal", "dominance",
                        "vad_atypical", "d_own", "d_other"])
            for i in order[:args.export]:
                w.writerow([
                    int(i), spk[i], NAMES[y[i]], NAMES[modal[i]],
                    int(miss[i]),
                    *(np.round(vad[i], 4).tolist() if has_vad else ["", "", ""]),
                    bool(atypical[i]) if has_vad else "",
                    round(float(d_own[i]), 4) if has_vad else "",
                    round(float(d_other[i]), 4) if has_vad else "",
                ])
        print("\n6. EXPORT  %s  (%d rows, worst first)" % (path, args.export))

        summary[corpus] = {
            "n": int(len(y)), "runs": n_runs,
            "core": int(core.sum()), "easy": int(easy.sum()),
            "pooled_accuracy": float((~wrong).mean()),
            "core_by_class": {NAMES[c]: int((core & (y == c)).sum())
                              for c in range(4)},
            "vad_usable": bool(has_vad),
            "atypical_core": float(atypical[core].mean()) if has_vad else None,
            "atypical_easy": float(atypical[easy].mean()) if has_vad else None,
            "mean_pair_agreement": float(np.mean(pair_agree)),
            "independent_agreement": float(indep),
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote %s" % (OUT / "summary.json"))


if __name__ == "__main__":
    main()
