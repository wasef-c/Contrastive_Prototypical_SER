#!/usr/bin/env python3
"""Pre-training checks for the VAD-matched triplet experiment.

Three things have to be settled before any GPU time, because each one can
make the result uninterpretable rather than merely negative.

1 SAMPLING WEIGHT. Drawing uniformly from eligible pairs gives an anchor
  with 200 negatives twenty times the weight of one with 10. Drawing the
  anchor first and then one of its negatives equalises it. Reported here
  as the effective anchor weight under both schemes.

2 PER-SPEAKER CAP. Chosen by counting how many anchors survive and how
  often clips repeat in an epoch, not from the Gini alone.

3 CONTROL CONTRAST. The control reuses the matched arm's negative clips
  and shuffles their assignment, within negative speaker where possible so
  speaker pairings survive. That only tests the hypothesis if the shuffled
  negatives are actually further away in VAD. Speakers have characteristic
  VAD ranges, so within-speaker shuffling may leave most negatives inside
  the matching radius, in which case the control is not a control.

Also fixes the evaluation subgroup on IEMOCAP, since the definition has
been stated inconsistently: the MSP coverage check used arousal < 0.5
while a figure of 46.9 percent was attributed to arousal <= 0.5. That
46.9 does not correspond to anything measured in this project; the closest
real number is 47.4 percent, which is the VAD-atypical share of the hard
core and a different quantity entirely. Counts for all three definitions
are printed so the subgroup can be fixed before training.

Usage:
    python -m analysis.triplet_design
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

from analysis.arm_table import by_seed
from analysis.vad_label_conflict import load_mspp

BUILD = "/mnt/fast/mspp_build/data"
OUT = pathlib.Path("analysis/triplet_design")
NAMES = ("neutral", "happy", "sad", "angry")


def meta() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Filename key to official split and speaker id."""
    shards = sorted(glob.glob(os.path.join(BUILD, "*.parquet")))
    t = pq.read_table(shards, columns=["FileName", "Split_Set", "SpkrID"])
    keys = [os.path.basename(str(f)).rsplit(".", 1)[0]
            for f in t.column("FileName").to_pylist()]
    return (dict(zip(keys, t.column("Split_Set").to_pylist())),
            dict(zip(keys, [str(s) for s in t.column("SpkrID").to_pylist()])))


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a count vector."""
    srt = np.sort(np.asarray(counts, dtype=float))
    n = len(srt)
    if n == 0 or srt.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(srt)
                 / (n * srt.sum()))


def main() -> None:
    """Run the three checks and fix the evaluation subgroup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--min-agree", type=float, default=0.8)
    parser.add_argument("--low-arousal", type=float, default=0.5)
    parser.add_argument("--caps", type=int, nargs="+",
                        default=[2, 4, 8, 16, 0],
                        help="per-speaker anchor caps; 0 means uncapped")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    corpus = load_mspp()
    keep = corpus.n_ann >= 5
    probs, vad, keys = corpus.probs[keep], corpus.vad[keep], corpus.keys[keep]
    split_of, spk_of = meta()
    split = np.array([split_of.get(k, "") for k in keys])
    spk = np.array([spk_of.get(k, "?") for k in keys])

    inside = probs[:, :4].sum(axis=1)
    agree = probs[:, :4].max(axis=1) / np.maximum(inside, 1e-9)
    lab = probs[:, :4].argmax(axis=1)
    ok = (split == "Train") & (inside >= 0.6) & (agree >= args.min_agree)
    v = (vad[ok] - 1.0) / 6.0
    y, s = lab[ok], spk[ok]

    anchors = np.where((y == 1) & (v[:, 1] < args.low_arousal))[0]
    negs = np.where(y == 0)[0]
    tree = cKDTree(v[negs])
    elig: Dict[int, List[int]] = {}
    for a in anchors:
        cand = [negs[j] for j in tree.query_ball_point(v[a], args.radius)
                if s[negs[j]] != s[a]]
        if cand:
            elig[int(a)] = cand
    print("quiet happy anchors with >=1 cross-speaker neutral negative: %d"
          % len(elig))

    print("\n" + "=" * 68)
    print("1. SAMPLING WEIGHT")
    sizes = np.array([len(c) for c in elig.values()])
    w_pair = sizes / sizes.sum()
    print("   negatives per anchor: median %d, mean %.1f, max %d"
          % (int(np.median(sizes)), sizes.mean(), sizes.max()))
    print("   uniform-over-pairs weight: top anchor %.3f%% of the loss,"
          " ratio max/min %.0fx"
          % (100 * w_pair.max(), sizes.max() / max(sizes.min(), 1)))
    print("   anchor-first weight: every anchor %.3f%%, ratio 1x"
          % (100.0 / len(elig)))
    print("   -> anchor-first, as proposed. Uniform-over-pairs would give")
    print("      the best-connected anchor %.0f times the weight."
          % (sizes.max() / max(sizes.min(), 1)))

    print("\n" + "=" * 68)
    print("2. PER-SPEAKER CAP")
    by_spk: Dict[str, List[int]] = collections.defaultdict(list)
    for a in elig:
        by_spk[s[a]].append(a)
    print("   %-10s %9s %9s %9s %10s"
          % ("cap", "anchors", "speakers", "gini", "clip reps/epoch"))
    cap_rows: Dict[str, object] = {}
    for cap in args.caps:
        kept: List[int] = []
        for sp, rows in by_spk.items():
            take = rows if cap == 0 else rows[:cap]
            kept.extend(take)
        cnt = np.array([min(len(r), cap) if cap else len(r)
                        for r in by_spk.values()])
        label = "uncapped" if cap == 0 else str(cap)
        # One triplet per anchor per epoch, so clip repetition is 1 for
        # anchors; the question is how often a NEGATIVE clip recurs.
        draws = collections.Counter()
        for a in kept:
            draws[int(rng.choice(elig[a]))] += 1
        reps = np.array(list(draws.values()))
        print("   %-10s %9d %9d %9.2f %10s"
              % (label, len(kept), sum(1 for r in cnt if r > 0), gini(cnt),
                 "median %d, max %d" % (int(np.median(reps)), reps.max())))
        cap_rows[label] = {"anchors": len(kept), "gini": gini(cnt),
                           "neg_rep_median": float(np.median(reps)),
                           "neg_rep_max": int(reps.max())}

    print("\n" + "=" * 68)
    print("3. CONTROL CONTRAST")
    chosen = {a: int(rng.choice(c)) for a, c in elig.items()}
    a_idx = np.array(list(chosen))
    n_idx = np.array([chosen[a] for a in a_idx])
    d_matched = np.linalg.norm(v[a_idx] - v[n_idx], axis=1)

    perm = rng.permutation(len(n_idx))
    d_global = np.linalg.norm(v[a_idx] - v[n_idx[perm]], axis=1)

    # Within negative speaker: permute only among negatives sharing a
    # speaker, which preserves every anchor-speaker pairing.
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for pos, n in enumerate(n_idx):
        groups[s[n]].append(pos)
    n_within = n_idx.copy()
    for sp, pos in groups.items():
        if len(pos) > 1:
            n_within[pos] = n_idx[rng.permutation(pos)]
    d_within = np.linalg.norm(v[a_idx] - v[n_within], axis=1)

    print("   %-26s %8s %8s %10s" % ("assignment", "median", "mean",
                                     "within r"))
    for label, d in (("VAD-matched", d_matched),
                     ("shuffled within speaker", d_within),
                     ("shuffled globally", d_global)):
        print("   %-26s %8.4f %8.4f %9.1f%%"
              % (label, float(np.median(d)), float(d.mean()),
                 100 * float((d <= args.radius).mean())))
    frac_within = float((d_within <= args.radius).mean())
    print("\n   singleton negative speakers (nothing to shuffle with): %d"
          % sum(1 for p in groups.values() if len(p) == 1))
    if frac_within > 0.5:
        print("   WARNING: %.0f%% of within-speaker shuffles stay inside the"
              " radius." % (100 * frac_within))
        print("   Speakers have characteristic VAD ranges, so that control")
        print("   barely differs from the matched arm. Use the global")
        print("   shuffle, or shuffle within speaker but reject draws that")
        print("   land inside the radius.")
    else:
        print("   Within-speaker shuffling does create contrast (%.0f%% stay"
              " inside)." % (100 * frac_within))

    print("\n" + "=" * 68)
    print("4. EVALUATION SUBGROUP ON IEMOCAP")
    m = np.load("analysis/heldout_meta/test_IEMOFULL.npz", allow_pickle=True)
    lab_t = m["label"]
    valid = lab_t >= 0
    yt = lab_t[valid]
    vt = m["vad"][valid]
    happy = yt == 1
    ar = vt[:, 1]
    print("   originally happy clips: %d" % int(happy.sum()))
    print("   median arousal among them: %.3f" % float(np.median(ar[happy])))
    for label, sel in (("arousal <  0.5", happy & (ar < 0.5)),
                       ("arousal == 0.5", happy & (ar == 0.5)),
                       ("arousal <= 0.5", happy & (ar <= 0.5))):
        print("   %-16s %5d clips (%.1f%% of happy)"
              % (label, int(sel.sum()), 100 * sel.sum() / happy.sum()))

    d = by_seed(("rl_hard_s9",), "IEMOFULL")
    if 42 in d and d[42].shape[0] == len(yt):
        p = d[42].argmax(axis=1)
        print("\n   current rl_hard_s9 recall on the full subgroup")
        for label, sel in (("arousal <  0.5", happy & (ar < 0.5)),
                           ("arousal == 0.5", happy & (ar == 0.5)),
                           ("arousal <= 0.5", happy & (ar <= 0.5)),
                           ("all happy", happy)):
            if sel.sum():
                print("     %-16s %5.1f%%  (n=%d)"
                      % (label, 100 * (p[sel] == 1).mean(), int(sel.sum())))
        print("   reverse error, neutral called happy: %.1f%% of %d neutral"
              % (100 * (p[yt == 0] == 1).mean(), int((yt == 0).sum())))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps({
        "eligible_anchors": len(elig),
        "caps": cap_rows,
        "control_within_speaker_inside_radius": frac_within,
        "control_global_inside_radius": float(
            (d_global <= args.radius).mean()),
        "matched_median_distance": float(np.median(d_matched)),
    }, indent=2))
    print("\nwrote %s" % (OUT / "checks.json"))


if __name__ == "__main__":
    main()
