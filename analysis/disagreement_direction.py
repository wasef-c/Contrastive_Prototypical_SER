#!/usr/bin/env python3
"""Split each panel's own disagreement into real-axis and noise-axis parts.

Agreement magnitude cannot distinguish a panel that split because the
utterance genuinely sits between two emotions from one that split because
the annotators were careless. Both read as "three of five agreed". The
direction of the split can distinguish them, because the matched-VAD
analysis measures which label confusions carry real structure in which
region of VAD space.

A panel splitting neutral against happy at VAD (3,3,3) is splitting along
an axis that is measurably real there: utterances at that coordinate are
54 percent neutral and 42 percent happy even when every panel is
unanimous. A panel splitting angry against happy at the same coordinate is
splitting along an axis that carries no structure anywhere near it.

The decomposition is per label pair. An utterance's own disagreement mass
on the unordered pair (a, b) is 2 pi[a] pi[b], estimated without bias by
2 p[a] p[b] n / (n - 1). Its region supplies the share of the (a, b)
confusion that survives correction for annotation noise, from
analysis.vad_label_conflict. Weighting the first by the second splits the
utterance's disagreement into the part that lies along locally real axes
and the part that does not.

This script answers whether the mechanism has anything to work with. If
nearly all disagreement mass already lies along real axes there is little
to correct. If a large share lies along noise axes, those are rows whose
soft label should be sharpened rather than trusted.

Regions are quantile cells rather than a fitted smooth map, so that the
realness estimate in each cell rests on pairs drawn from that cell alone
and nothing is borrowed across the space.

Usage:
    python -m analysis.disagreement_direction
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from analysis.vad_label_conflict import (Corpus, label_pair_excess, load_mspp,
                                         subset, true_agreement)

OUT_DIR = pathlib.Path("analysis/disagreement_direction")
FOUR = [0, 1, 2, 3]
NAMES = ("neutral", "happy", "sad", "angry")


def four_class_population(corpus: Corpus, min_inside: float,
                          min_ann: int) -> Corpus:
    """Restrict to the population the model is actually trained on.

    The classifier has four outputs, so an utterance whose votes mostly
    name labels outside that scheme is not a training row and its
    disagreement is not something the model could represent. Votes inside
    the scheme are renormalised, which is what the training label already
    does implicitly by taking the majority over the four.

    Args:
        corpus: the full corpus.
        min_inside: smallest share of votes that must fall inside the four.
        min_ann: smallest surviving panel size.

    Returns:
        A Corpus restricted to four classes.
    """
    counts4 = corpus.counts[:, FOUR]
    inside = counts4.sum(axis=1) / np.maximum(corpus.n_ann, 1)
    keep = inside >= min_inside
    out = subset(corpus, keep)
    out.counts = out.counts[:, FOUR]
    out.labels = list(NAMES)
    second = out.counts.sum(axis=1) >= min_ann
    out = subset(out, second)
    return out


def pair_mass(probs: np.ndarray, n_ann: np.ndarray) -> np.ndarray:
    """Unbiased per-utterance disagreement mass on each unordered label pair.

    Summing over pairs recovers the unbiased estimate of
    1 - ||pi||^2, the chance two of this utterance's annotators differ.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.

    Returns:
        [N, C, C] symmetric array with a zero diagonal.
    """
    n = np.maximum(np.asarray(n_ann, dtype=np.float64), 2.0)
    scale = (n / (n - 1.0))[:, None, None]
    outer = probs[:, :, None] * probs[:, None, :] * scale
    idx = np.arange(probs.shape[1])
    outer[:, idx, idx] = 0.0
    return outer


def cell_ids(vad: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign utterances to quantile cells of VAD space.

    Args:
        vad: [N, 3] VAD means.
        bins: bins per dimension.

    Returns:
        Tuple of [N] cell ids and the [bins - 1, 3] cut points used.
    """
    qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    cuts = np.quantile(vad, qs, axis=0)
    digits = np.stack([np.searchsorted(cuts[:, d], vad[:, d], side="right")
                       for d in range(3)], axis=1)
    ids = (digits[:, 0] * bins + digits[:, 1]) * bins + digits[:, 2]
    return ids, cuts


def realness_map(probs: np.ndarray, n_ann: np.ndarray, ids: np.ndarray,
                 min_cell: int, max_pairs: int, rng: np.random.Generator
                 ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Share of each label pair's confusion that is real, per VAD cell.

    Within a cell every utterance shares a VAD neighbourhood, so pairs
    drawn inside it are matched on VAD by construction and the correction
    for annotation noise applies as in the main analysis.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.
        ids: [N] cell ids.
        min_cell: smallest cell worth estimating.
        max_pairs: cap on pairs sampled per cell.
        rng: random generator.

    Returns:
        Tuple of cell id -> [C, C] real-share matrix, and cell id -> size.
    """
    out: Dict[int, np.ndarray] = {}
    sizes: Dict[int, int] = {}
    for cell in np.unique(ids):
        members = np.where(ids == cell)[0]
        if members.size < min_cell:
            continue
        take = int(min(max_pairs, members.size * (members.size - 1)))
        li = rng.choice(members, size=take, replace=True)
        lj = rng.choice(members, size=take, replace=True)
        good = li != lj
        li, lj = li[good], lj[good]
        exc, cross = label_pair_excess(probs, n_ann, li, lj)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(cross > 1e-9, exc / cross, 0.0)
        out[int(cell)] = np.clip(share, 0.0, 1.0)
        sizes[int(cell)] = int(members.size)
    return out, sizes


def main() -> None:
    """Measure how much panel disagreement runs along noise axes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=4,
                        help="quantile bins per VAD dimension")
    parser.add_argument("--min-cell", type=int, default=300,
                        help="smallest cell worth estimating")
    parser.add_argument("--max-pairs", type=int, default=60000,
                        help="pairs sampled per cell")
    parser.add_argument("--min-inside", type=float, default=0.6,
                        help="share of votes that must name one of the four")
    parser.add_argument("--min-annotators", type=int, default=4,
                        help="smallest surviving panel")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    full = load_mspp()
    corpus = four_class_population(full, args.min_inside, args.min_annotators)
    probs = corpus.probs
    n_ann = corpus.n_ann
    total = probs.shape[0]
    print("cache covers %d utterances; four-class training population %d "
          "(%.1f%%)" % (full.counts.shape[0], total,
                        100.0 * total / full.counts.shape[0]))

    agree = true_agreement(probs, n_ann)
    unanimous = probs.max(axis=1) >= 1.0 - 1e-9
    print("panels that were unanimous on the four-class label: %.1f%%"
          % (100.0 * unanimous.mean()))
    print("chance two annotators of the same utterance differ: %.3f"
          % float(1.0 - agree.mean()))

    ids, _ = cell_ids(corpus.vad, args.bins)
    shares, sizes = realness_map(probs, n_ann, ids, args.min_cell,
                                 args.max_pairs, rng)
    covered = np.isin(ids, list(shares))
    print("VAD cells estimated: %d of %d possible, covering %.1f%% of rows"
          % (len(shares), args.bins ** 3, 100.0 * covered.mean()))

    mass = pair_mass(probs, n_ann)
    real = np.zeros(total)
    for cell, share in shares.items():
        sel = ids == cell
        real[sel] = (mass[sel] * share[None, :, :]).sum(axis=(1, 2))
    total_mass = mass.sum(axis=(1, 2))

    use = covered & (total_mass > 1e-9)
    gross = float(total_mass[use].sum())
    real_sum = float(real[use].sum())
    print("\nTHE GATE")
    print("  rows with a split panel in an estimated cell: %d" % int(use.sum()))
    print("  their total disagreement mass:        %.1f" % gross)
    print("  along locally REAL axes:              %.1f  (%.1f%%)"
          % (real_sum, 100.0 * real_sum / gross))
    print("  along NOISE axes:                     %.1f  (%.1f%%)"
          % (gross - real_sum, 100.0 * (gross - real_sum) / gross))

    frac = np.full(total, np.nan)
    frac[use] = real[use] / total_mass[use]
    ok = np.isfinite(frac)
    print("\n  per-utterance real-axis fraction, distribution:")
    for q in (5, 25, 50, 75, 95):
        print("    p%-3d %.3f" % (q, float(np.percentile(frac[ok], q))))
    for lo, hi, name in ((0.0, 0.25, "mostly noise"),
                         (0.25, 0.75, "mixed"),
                         (0.75, 1.01, "mostly real")):
        sel = ok & (frac >= lo) & (frac < hi)
        print("    %-14s %6d rows (%.1f%% of split panels), "
              "mean own disagreement %.3f"
              % (name, int(sel.sum()), 100.0 * sel.sum() / ok.sum(),
                 float(total_mass[sel].mean()) if sel.sum() else float("nan")))

    print("\n  which pairs carry the NOISE-axis mass:")
    noise_by_pair = np.zeros((4, 4))
    real_by_pair = np.zeros((4, 4))
    for cell, share in shares.items():
        sel = ids == cell
        m = mass[sel].sum(axis=0)
        real_by_pair += m * share
        noise_by_pair += m * (1.0 - share)
    rows: List[Tuple[float, str, float, float]] = []
    for a in range(4):
        for b in range(a + 1, 4):
            rows.append((float(noise_by_pair[a, b] * 2),
                         "%s / %s" % (NAMES[a], NAMES[b]),
                         float(real_by_pair[a, b] * 2),
                         float((real_by_pair[a, b] + noise_by_pair[a, b]) * 2)))
    rows.sort(reverse=True)
    print("    %-18s %10s %10s %7s" % ("pair", "noise", "real", "real%"))
    for noise, name, realv, tot in rows:
        print("    %-18s %10.1f %10.1f %6.0f%%"
              % (name, noise, realv, 100.0 * realv / tot if tot else 0.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mspp.json"
    out.write_text(json.dumps({
        "n_cache": int(full.counts.shape[0]),
        "n_population": int(total),
        "frac_unanimous": float(unanimous.mean()),
        "p_disagree_same_utterance": float(1.0 - agree.mean()),
        "cells_estimated": len(shares),
        "row_coverage": float(covered.mean()),
        "mass_total": gross,
        "mass_real": real_sum,
        "real_share": real_sum / gross,
        "cell_sizes": sizes,
        "pairs": [{"pair": n, "noise": v, "real": r, "total": t}
                  for v, n, r, t in rows],
        "args": vars(args),
    }, indent=2))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
