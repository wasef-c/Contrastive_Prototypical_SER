#!/usr/bin/env python3
"""Running comparison of every mixup arm across the three evaluation corpora.

Regenerate after any run completes. Writes a markdown table and a CSV, and
prints the table to stdout.

Corpora are IEMOFULL (standard 4-class IEMOCAP, excited merged into happy),
SAMSEMO and CMUMOSEI. The superseded IEMO subset is deliberately excluded,
as is MSP-Improv.

Every figure is raw argmax on saved logits with no calibration. Deltas are
paired per training seed against vanilla mixup, on the seed intersection, and
tested with a two-sided one-sample t-test. n is small, so the per-seed sign
pattern matters more than any p-value.
"""

import argparse
import csv
import glob
import pathlib
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

from analysis.mixup_target_comparison import by_seed
from analysis.vad_explained_split import uar

# (display name, meta stem, prediction stem)
ALL_CORPORA = (("IEMOCAP", "test_IEMOFULL", "IEMOFULL"),
               ("SAMSEMO", "test_SAMSEMO", "SAMSEMO"),
               ("CMUMOSEI", "test_CMUMOSEI", "CMUMOSEI"))

# CMU-MOSEI is kept on disk but not reported. The usable subset after label
# and VAD filtering is 1938 rows, and the modality ablation shows the fusion
# scores it with text more than audio (audio only 38.70 UAR against text only
# 39.64), so it cannot support a claim about an affect-geometry mechanism.
# Scoring it still works through ALL_CORPORA for anyone who asks.
HEADLINE_CORPORA = tuple(c for c in ALL_CORPORA if c[0] != "CMUMOSEI")

CORPORA = HEADLINE_CORPORA

REFERENCE = "vanilla mixup"

# label -> (group, arm names pooled into one arm)
ARMS: Sequence[Tuple[str, str, Tuple[str, ...]]] = (
    ("no-mixup baseline", "reference", ("uf4_base", "uf4_base_more")),
    ("hard label", "reference", ("uf4_softmix_ctrl", "uf4_softmix_ctrl_more")),
    ("vanilla mixup", "reference",
     ("uf4_softmix_vanilla", "uf4_softmix_vanilla_more")),
    ("GenLabel PCA-64", "published", ("uf4_genlabel", "uf4_genlabel_more")),
    ("GenLabel raw 1024-d", "published",
     ("uf4_genlabel_raw", "uf4_genlabel_raw_more")),
    ("GEOMETRY (ours)", "method", ("uf4_softmix", "uf4_softmix_more")),
    ("endpoint posterior", "variant",
     ("uf4_softmix_endpt", "uf4_softmix_endpt_more")),
    # The two controls that strip content out of the target while holding its
    # shape fixed. PLACEBO permutes finished targets within source-label
    # groups, so the marginal, the entropy and the mass on the source label
    # all survive and only the per-blend correspondence is destroyed.
    # DIRICHLET never looks at VAD at all, sampling targets whose
    # class-conditional mean and entropy match q by construction. Ours
    # beating neither would mean the affect content is not what pays.
    ("PLACEBO (shuffled target)", "control",
     ("uf4_softmix_shuf", "uf4_softmix_shuf_more")),
    ("VAD-free Dirichlet", "control",
     ("uf4_softmix_dir", "uf4_softmix_dir_more")),
    ("Global Mixup (Xie 2023)", "published",
     ("uf4_globalmix", "uf4_globalmix_more")),
    ("Global Mixup, gamma=1", "published",
     ("uf4_globalmix_g1", "uf4_globalmix_g1_more")),
    ("VAD soft, no mixup", "control", ("uf4_vadsoft",)),
    ("empirical priors", "variant", ("uf4_softmix_prior",)),
    ("calibrated prior", "variant", ("uf4_softmix_cal",)),
    ("consensus-fitted q", "variant", ("uf4_softmix_cons",)),
    ("same-class gate", "variant", ("uf4_softmix_gate",)),
    ("consensus trust", "variant", ("uf4_softmix_trust",)),
    ("consensus trust, placebo", "control", ("uf4_softmix_trust_ctrl",)),
    ("2 blends per sample", "variant", ("uf4_softmix_x2",)),
    ("mixup weight 0.25", "variant", ("uf4_softmix_w025",)),
    ("mixup weight 1.0", "variant", ("uf4_softmix_w10",)),
    ("+ aux VAD", "variant", ("uf4_softmix_aux",)),
    ("+ aux VAD, placebo", "control", ("uf4_softmix_aux_ctrl",)),
    ("pre-fusion, shared lam", "variant", ("uf4_softmix_pm_ctrl",)),
    ("pre-fusion, per-modality", "variant", ("uf4_softmix_pm",)),
    ("blend audio only", "variant", ("uf4_softmix_pmaudio",)),
    ("blend text only", "variant", ("uf4_softmix_pmtext",)),
    ("+ prototype pull", "variant", ("uf4_softmix_pull",)),
    ("+ pull, placebo", "control", ("uf4_softmix_pull_ctrl",)),
    ("pull only, no mixup", "variant", ("uf4_pull_only",)),
    ("toward-centre ratio 0.25", "variant", ("uf4_softmix_r25",)),
    ("toward-centre ratio 0.50", "variant", ("uf4_softmix_r50",)),
    ("16 epochs, ours", "variant", ("uf4_softmix_e16",)),
    ("16 epochs, vanilla", "variant", ("uf4_softmix_vanilla_e16",)),
    ("aggressive lam, ours", "variant", ("uf4_softmix_agg", "uf4_softmix_agg_more")),
    ("aggressive lam, vanilla", "variant",
     ("uf4_softmix_agg_vanilla", "uf4_softmix_agg_vanilla_more")),
    ("aggressive lam, hard", "variant",
     ("uf4_softmix_agg_ctrl", "uf4_softmix_agg_ctrl_more")),
)

OUT_MD = pathlib.Path("paper/reports/arm_table.md")
OUT_CSV = pathlib.Path("paper/reports/arm_table.csv")


def scores(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """UAR, accuracy and macro F1 in percent.

    Args:
        logits: [N, C] raw logits.
        y: [N] integer labels.

    Returns:
        Array of (uar, accuracy, macro_f1).
    """
    p = logits.argmax(1)
    return np.array([100 * uar(y, p), 100 * (p == y).mean(),
                     100 * f1_score(y, p, average="macro")])


def discover_extra() -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Find mixup arms with predictions that the curated list omits.

    Keeps the table current when a new arm is added without editing this file.

    Returns:
        List of (label, group, arm-name tuple) for the unlisted arms.
    """
    listed = {a for _, _, arms in ARMS for a in arms}
    seen = set()
    for f in glob.glob("checkpoints/*/predictions/IEMOFULL.npz"):
        stem = pathlib.Path(f).parts[1]
        arm = re.sub(r"_seed\d+(_seed\d+)?$", "", stem)
        if arm in listed or arm in seen:
            continue
        if arm.startswith(("uf4_softmix", "uf4_pull")):
            seen.add(arm)
    return [(a + "  [unlisted]", "new", (a,)) for a in sorted(seen)]


def main() -> None:
    """Build the table from saved logits and write it out."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="hide arms with fewer finished seeds than this")
    args = ap.parse_args()

    rows = list(ARMS) + discover_extra()
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    ys: Dict[str, np.ndarray] = {}
    for disp, _, _ in rows:
        data[disp] = {}
    for corpus, stem, key in CORPORA:
        meta = np.load("analysis/heldout_meta/%s.npz" % stem, allow_pickle=True)
        lab = meta["label"]
        y = lab[lab >= 0]
        ys[corpus] = y
        for disp, _, arms in rows:
            d = by_seed(arms, key)
            data[disp][corpus] = {s: scores(v, y) for s, v in d.items()
                                  if v.shape[0] == len(y)}

    md: List[str] = []
    md.append("# Mixup arm comparison")
    md.append("")
    md.append("Raw argmax, no calibration. Deltas are paired per training "
              "seed against **%s**, on the seed intersection." % REFERENCE)
    md.append("")
    md.append("| corpus | N | class mix |")
    md.append("|---|---|---|")
    for corpus, _, _ in CORPORA:
        y = ys[corpus]
        mix = " / ".join("%.1f" % (100 * (y == c).mean()) for c in range(4))
        md.append("| %s | %d | %s |" % (corpus, len(y), mix))
    md.append("")

    csv_rows = [("arm", "group", "corpus", "n", "uar", "acc", "f1",
                 "d_uar", "p_uar", "d_acc", "p_acc", "d_f1", "p_f1")]
    for corpus, _, _ in CORPORA:
        md.append("## %s" % corpus)
        md.append("")
        md.append("| arm | n | UAR | acc | F1 | dUAR | p | dACC | p | dF1 | p |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        ref = data[REFERENCE][corpus]
        for disp, grp, _ in rows:
            d = data[disp][corpus]
            s = sorted(d)
            if len(s) < args.min_seeds:
                continue
            m = np.array([d[i] for i in s]).mean(0)
            shared = sorted(set(d) & set(ref))
            cells, nums = [], []
            if disp != REFERENCE and len(shared) >= 2:
                diff = np.array([d[i] - ref[i] for i in shared])
                for j in range(3):
                    mu = diff[:, j].mean()
                    pv = stats.ttest_1samp(diff[:, j], 0).pvalue
                    star = "**" if pv < 0.05 else ""
                    cells += ["%s%+.2f%s" % (star, mu, star), "%.3f" % pv]
                    nums += [mu, pv]
            else:
                cells = ["-"] * 6
                nums = [""] * 6
            bold = "**" if grp == "method" else ""
            md.append("| %s%s%s | %d | %.2f | %.2f | %.2f | %s |"
                      % (bold, disp, bold, len(s), m[0], m[1], m[2],
                         " | ".join(cells)))
            csv_rows.append((disp, grp, corpus, len(s),
                             "%.3f" % m[0], "%.3f" % m[1], "%.3f" % m[2],
                             *["%s" % x if x == "" else "%.4f" % x for x in nums]))
        md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print("\n".join(md))
    print()
    print("wrote %s and %s" % (OUT_MD, OUT_CSV))


if __name__ == "__main__":
    main()
