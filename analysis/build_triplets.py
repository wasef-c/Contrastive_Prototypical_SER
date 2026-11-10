#!/usr/bin/env python3
"""Build the triplet assignments for the VAD-matched experiment.

Emits, for each of two arms, a list of (anchor, positive, negative)
filename keys drawn from the official MSP-Podcast Train split:

  matched  negative is a clip of the confusable class whose VAD is within
           RADIUS of the anchor, from a different speaker
  random   the SAME multiset of negative clips, reassigned to anchors by a
           shuffle within direction

Design decisions, each settled by measurement rather than by taste:

BIDIRECTIONAL. An earlier version used happy anchors against neutral
negatives only. That is satisfiable by sliding the happy/neutral decision
boundary toward happy: quiet-happy recall rises, neutral-called-happy
false alarms rise with it, and no cue has been learned. Anchoring both
directions removes that escape, since moving the boundary now costs as
much as it gains.

SYMMETRIC ANCHOR RULE. An anchor is a clip that sits closer to the OTHER
class's VAD centroid than to its own. The earlier happy-side rule
(arousal < 0.5) does not transfer to neutral, and the centroid rule gives
more happy anchors anyway: 2,142 usable against 1,810, covering 1,416 of
those 1,810, so the low-arousal IEMOCAP endpoint stays relevant.

PAIR CHOICE. Only happy/neutral and neutral/sad have both coverage and
anchors. Angry is unavailable rather than untried: just 184 angry clips
are VAD-atypical at all and happy-to-angry covers 13.6 percent, which is
the same fact as angry being VAD-atypical only 12.0 percent of the time.
Sad has coverage but 214 to 325 anchors after capping, too few to move a
20k-clip training set.

ANCHOR-FIRST. Negatives per anchor run from 1 to several hundred, so
sampling uniformly over eligible pairs would give the best-connected
anchor hundreds of times the weight of the worst. One negative is drawn
per anchor instead, which equalises anchor weight.

CAPS FOR PARITY. Happy anchors cap at 8 per speaker (1,048 anchors, Gini
0.44); neutral anchors cap at 2 (1,191, Gini 0.15). The ratio is 1.14 to
1, so neither direction dominates the term. Uncapped the neutral side
would outweigh the happy side close to 3 to 1.

GLOBAL SHUFFLE FOR THE CONTROL. Shuffling within negative speaker was the
first choice, because it preserves speaker pairings, but speakers have
characteristic VAD ranges and 53.4 percent of such shuffles land back
inside the radius - the control would be most of the way to the treatment.
A global shuffle leaves about 5 percent inside. The cost is that anchor
and negative speaker pairings change, which is a confound to state rather
than one that can be removed here. The shuffle runs within direction, so
the control keeps the treatment's class composition exactly.

POSITIVE FROM A THIRD SPEAKER. A clip of the anchor's own class from a
speaker that is neither the anchor's nor the negative's, so the triplet
cannot be satisfied by speaker identity alone.

Usage:
    python -m analysis.build_triplets
"""

import argparse
import collections
import glob
import json
import os
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from analysis.vad_label_conflict import load_mspp

BUILD = "/mnt/fast/mspp_build/data"
OUT = pathlib.Path("analysis/triplets/mspp_triplets.npz")
NAMES = ("neutral", "happy", "sad", "angry")

# (anchor class, negative class, per-speaker anchor cap)
DIRECTIONS: Tuple[Tuple[int, int, int], ...] = ((1, 0, 8), (0, 1, 2))


def meta() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Filename key to official split and speaker id."""
    shards = sorted(glob.glob(os.path.join(BUILD, "*.parquet")))
    t = pq.read_table(shards, columns=["FileName", "Split_Set", "SpkrID"])
    keys = [os.path.basename(str(f)).rsplit(".", 1)[0]
            for f in t.column("FileName").to_pylist()]
    return (dict(zip(keys, t.column("Split_Set").to_pylist())),
            dict(zip(keys, [str(s) for s in t.column("SpkrID").to_pylist()])))


def gini(counts: Sequence[float]) -> float:
    """Gini coefficient of a count vector."""
    srt = np.sort(np.asarray(counts, dtype=float))
    n = len(srt)
    if n == 0 or srt.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(srt)
                 / (n * srt.sum()))


def main() -> None:
    """Build both assignments and write them with their diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--min-agree", type=float, default=0.8)
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
    y, s, k = lab[ok], spk[ok], keys[ok]
    centroid = np.stack([v[y == c].mean(axis=0) for c in range(4)])

    triples: Dict[str, List[Tuple[str, str, str]]] = {"matched": [],
                                                      "random": []}
    direction: Dict[str, List[int]] = {"matched": [], "random": []}
    per_dir: Dict[str, object] = {}
    dropped = 0

    for anchor_cls, neg_cls, cap in DIRECTIONS:
        tag = "%s_to_%s" % (NAMES[anchor_cls], NAMES[neg_cls])
        d_own = np.linalg.norm(v - centroid[anchor_cls], axis=1)
        d_oth = np.linalg.norm(v - centroid[neg_cls], axis=1)
        pool = np.where((y == anchor_cls) & (d_oth < d_own))[0]
        negs = np.where(y == neg_cls)[0]
        same = np.where(y == anchor_cls)[0]
        tree = cKDTree(v[negs])

        elig: Dict[int, List[int]] = {}
        for a in pool:
            cand = [int(negs[j])
                    for j in tree.query_ball_point(v[a], args.radius)
                    if s[negs[j]] != s[a]]
            if cand:
                elig[int(a)] = cand

        by_spk: Dict[str, List[int]] = collections.defaultdict(list)
        for a in elig:
            by_spk[s[a]].append(a)
        kept: List[int] = []
        for sp in sorted(by_spk):
            order = list(by_spk[sp])
            rng.shuffle(order)
            kept.extend(order[:cap] if cap > 0 else order)
        kept.sort()
        counts = [min(len(r), cap) if cap > 0 else len(r)
                  for r in by_spk.values()]
        print("%s: pool %d, usable %d, kept %d over %d speakers (cap %d, "
              "gini %.2f)" % (tag, len(pool), len(elig), len(kept),
                              len(by_spk), cap, gini(counts)))

        neg_of = {a: int(rng.choice(elig[a])) for a in kept}
        neg_list = np.array([neg_of[a] for a in kept])
        shuffled = neg_list[rng.permutation(len(kept))]

        def pick_positive(a: int, n: int) -> int:
            """A clip of the anchor's class from a third speaker, so
            speaker identity alone cannot satisfy the triplet."""
            for _ in range(64):
                c = int(rng.choice(same))
                if s[c] != s[a] and s[c] != s[n] and c != a:
                    return c
            return -1

        for i, a in enumerate(kept):
            for mode, n in (("matched", int(neg_list[i])),
                            ("random", int(shuffled[i]))):
                if s[n] == s[a]:
                    dropped += 1
                    continue
                p = pick_positive(a, n)
                if p < 0:
                    dropped += 1
                    continue
                triples[mode].append((str(k[a]), str(k[p]), str(k[n])))
                direction[mode].append(anchor_cls)

        d_m = np.linalg.norm(v[kept] - v[neg_list], axis=1)
        d_r = np.linalg.norm(v[kept] - v[shuffled], axis=1)
        print("   matched  median %.4f  within radius %.1f%%"
              % (np.median(d_m), 100 * (d_m <= args.radius).mean()))
        print("   random   median %.4f  within radius %.1f%%"
              % (np.median(d_r), 100 * (d_r <= args.radius).mean()))
        per_dir[tag] = {
            "anchor_pool": int(len(pool)),
            "usable_anchors": int(len(elig)),
            "kept_anchors": int(len(kept)),
            "anchor_speakers": int(len(by_spk)),
            "cap": int(cap),
            "anchor_gini": gini(counts),
            "matched_median_dist": float(np.median(d_m)),
            "random_median_dist": float(np.median(d_r)),
            "matched_within_radius": float((d_m <= args.radius).mean()),
            "random_within_radius": float((d_r <= args.radius).mean()),
        }

    print("\ntriples built: matched %d, random %d (dropped %d)"
          % (len(triples["matched"]), len(triples["random"]), dropped))
    for mode in ("matched", "random"):
        byd = collections.Counter(direction[mode])
        print("   %-8s %s" % (mode, {NAMES[c]: n for c, n in
                                     sorted(byd.items())}))
        print("   %-8s distinct negative clips %d"
              % ("", len(set(t[2] for t in triples[mode]))))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        matched=np.array(triples["matched"], dtype=object),
        random=np.array(triples["random"], dtype=object),
        matched_dir=np.array(direction["matched"], dtype=np.int64),
        random_dir=np.array(direction["random"], dtype=np.int64),
        radius=np.float32(args.radius),
    )
    diag: Dict[str, object] = {
        "directions": per_dir,
        "args": vars(args),
        "triples_matched": len(triples["matched"]),
        "triples_random": len(triples["random"]),
        "dropped": dropped,
    }
    (OUT.parent / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
