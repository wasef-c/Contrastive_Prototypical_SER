#!/usr/bin/env python3
"""Coverage check for VAD-matched hard negatives, before any training.

The proposal: alongside the usual classification loss, add a triplet term
whose negative is chosen to have nearly the same VAD as the anchor but a
different categorical label. VAD is used only to pick the negative during
training, never as a model input, so the model has to find whatever cue
distinguishes the two categories beyond their shared VAD.

The first check is whether the data supports those comparisons at all. If
a handful of speakers supply most of the available pairs, repeating their
clips more often will teach speaker identity rather than a cue for
emotion, and no amount of exposure fixes that.

Reported for the happy-versus-neutral case, which is where the failure
analysis finds errors in both directions: happy mistaken for neutral is
the single largest confusion in the hard core on both evaluation corpora.

Everything is measured on the official MSP-Podcast Train split, restricted
to clips whose annotators agreed strongly, since a pair is only
informative if both labels are trustworthy.

Usage:
    python -m analysis.vad_matched_pairs
"""

import argparse
import collections
import glob
import json
import os
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from analysis.vad_label_conflict import load_mspp

BUILD = "/mnt/fast/mspp_build/data"
OUT = pathlib.Path("analysis/vad_matched_pairs")
NAMES = ("neutral", "happy", "sad", "angry")


def meta() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Filename key to official split and to speaker id."""
    shards = sorted(glob.glob(os.path.join(BUILD, "*.parquet")))
    t = pq.read_table(shards, columns=["FileName", "Split_Set", "SpkrID"])
    keys = [os.path.basename(str(f)).rsplit(".", 1)[0]
            for f in t.column("FileName").to_pylist()]
    return (dict(zip(keys, t.column("Split_Set").to_pylist())),
            dict(zip(keys, [str(s) for s in t.column("SpkrID").to_pylist()])))


def concentration(counts: np.ndarray) -> Tuple[float, float]:
    """Top-decile share and Gini of a count distribution."""
    c = np.sort(counts)[::-1]
    total = c.sum()
    if total <= 0:
        return 0.0, 0.0
    top = c[:max(1, len(c) // 10)].sum() / total
    srt = np.sort(counts)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt)
                 / (n * srt.sum())) if srt.sum() > 0 else 0.0
    return float(top), gini


def main() -> None:
    """Report pair availability and speaker concentration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default="happy", choices=list(NAMES))
    parser.add_argument("--negative", default="neutral", choices=list(NAMES))
    parser.add_argument("--min-agree", type=float, default=0.8,
                        help="smallest share of annotators on the majority")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10, 0.15],
                        help="VAD match radii, 0-1 scale")
    parser.add_argument("--low-arousal", type=float, default=0.5,
                        help="arousal below this counts as quiet")
    args = parser.parse_args()

    corpus = load_mspp()
    keep = corpus.n_ann >= 5
    probs = corpus.probs[keep]
    vad = corpus.vad[keep]
    keys = corpus.keys[keep]

    split_of, spk_of = meta()
    split = np.array([split_of.get(k, "") for k in keys])
    spk = np.array([spk_of.get(k, "?") for k in keys])

    inside = probs[:, :4].sum(axis=1)
    agree = probs[:, :4].max(axis=1) / np.maximum(inside, 1e-9)
    lab = probs[:, :4].argmax(axis=1)
    ok = (split == "Train") & (inside >= 0.6) & (agree >= args.min_agree)
    v = (vad[ok] - 1.0) / 6.0
    y, s = lab[ok], spk[ok]
    print("official Train, four-class, agreement >= %.2f: %d clips, "
          "%d speakers" % (args.min_agree, int(ok.sum()), len(set(s))))
    print("  class counts: %s" % {NAMES[c]: int((y == c).sum())
                                  for c in range(4)})

    a_i = NAMES.index(args.anchor)
    n_i = NAMES.index(args.negative)
    anchors = np.where(y == a_i)[0]
    negs = np.where(y == n_i)[0]
    quiet = anchors[v[anchors, 1] < args.low_arousal]
    print("\n%s anchors %d, of which quiet (arousal < %.2f) %d"
          % (args.anchor, len(anchors), args.low_arousal, len(quiet)))
    print("%s candidates %d" % (args.negative, len(negs)))

    tree = cKDTree(v[negs])
    summary: Dict[str, object] = {"n_clips": int(ok.sum()),
                                  "n_speakers": len(set(s)),
                                  "anchors": int(len(anchors)),
                                  "quiet_anchors": int(len(quiet)),
                                  "radii": {}}

    for r in args.radii:
        print("\n" + "=" * 66)
        print("radius %.3f in VAD (0-1 scale)" % r)
        for name, pool in (("all %s" % args.anchor, anchors),
                           ("quiet %s" % args.anchor, quiet)):
            if len(pool) == 0:
                continue
            hits = tree.query_ball_point(v[pool], r)
            # A pair is only usable if the negative is a DIFFERENT speaker.
            usable: List[int] = []
            per_anchor: List[int] = []
            neg_use: collections.Counter = collections.Counter()
            anc_spk: collections.Counter = collections.Counter()
            for p, h in zip(pool, hits):
                cand = [negs[j] for j in h if s[negs[j]] != s[p]]
                per_anchor.append(len(cand))
                if cand:
                    usable.append(p)
                    anc_spk[s[p]] += 1
                    for c in cand:
                        neg_use[s[c]] += 1
            per_anchor = np.array(per_anchor)
            cov = len(usable) / len(pool)
            print("  %-14s anchors with >=1 cross-speaker negative: "
                  "%d of %d (%.1f%%)"
                  % (name, len(usable), len(pool), 100 * cov))
            if not usable:
                continue
            print("  %-14s negatives per anchor: median %d, mean %.1f, "
                  "90th %d" % ("", int(np.median(per_anchor)),
                               per_anchor.mean(),
                               int(np.percentile(per_anchor, 90))))
            a_counts = np.array(list(anc_spk.values()))
            n_counts = np.array(list(neg_use.values()))
            a_top, a_gini = concentration(a_counts)
            n_top, n_gini = concentration(n_counts)
            print("  %-14s anchor speakers %d; top decile supplies %.1f%% "
                  "(Gini %.2f)" % ("", len(anc_spk), 100 * a_top, a_gini))
            print("  %-14s negative speakers %d; top decile supplies %.1f%% "
                  "(Gini %.2f)" % ("", len(neg_use), 100 * n_top, n_gini))
            if name.startswith("quiet"):
                summary["radii"]["%.3f" % r] = {
                    "coverage": float(cov),
                    "usable_anchors": int(len(usable)),
                    "anchor_speakers": int(len(anc_spk)),
                    "anchor_top_decile_share": float(a_top),
                    "anchor_gini": float(a_gini),
                    "negatives_per_anchor_median": float(
                        np.median(per_anchor)),
                }

    print("\n" + "=" * 66)
    print("VERDICT GUIDE")
    print("  Coverage below about half means most anchors have no partner")
    print("  and the term would apply to a minority of the data.")
    print("  An anchor Gini above roughly 0.6, or a top decile above 50")
    print("  percent, means a few speakers dominate the comparisons; the")
    print("  model can then satisfy the objective by learning those")
    print("  speakers rather than a cue for emotion.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "%s_vs_%s.json" % (args.anchor, args.negative)).write_text(
        json.dumps(summary, indent=2))
    print("\nwrote %s" % (OUT / ("%s_vs_%s.json"
                                 % (args.anchor, args.negative))))


if __name__ == "__main__":
    main()
