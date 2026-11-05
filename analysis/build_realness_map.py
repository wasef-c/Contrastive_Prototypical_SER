#!/usr/bin/env python3
"""Fit the label-pair realness map and emit per-utterance trust weights.

Produces the lookup the two levers need at training time. For each
utterance it writes alpha, the share of that panel's own disagreement
running along label-pair axes that carry real structure where the
utterance sits in VAD space. Lever one interpolates the target between the
hard majority label and the annotator distribution by alpha; lever two
scales the row's loss by it.

Everything is fitted on the official MSP-Podcast Train split alone. The
map is a function of the training labels, so fitting it on rows that later
serve as validation would let the validation metric see its own labels
through the target. Development rows still receive an alpha, computed by
applying the Train-fitted cell boundaries and Train-fitted realness
values, so the column is defined everywhere without Development having
influenced it.

Two maps are written. The real one, and a placebo whose realness values
are permuted across VAD cells while each label pair keeps its
corpus-average value. The placebo preserves how soft the targets are on
average and destroys only the claim that softness belongs in particular
places, which is the mechanism under test.

Usage:
    python -m analysis.build_realness_map
"""

import argparse
import glob
import os
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

from analysis.disagreement_direction import (NAMES, four_class_population,
                                             pair_mass, realness_map)
from analysis.vad_label_conflict import Corpus, load_mspp, subset

BUILD_DIR = "/mnt/fast/mspp_build/data"
OUT = pathlib.Path("analysis/realness_map/mspp_official_train.npz")
NUM_CLASSES = 4


def official_splits(build_dir: str = BUILD_DIR) -> Dict[str, str]:
    """Map extension-stripped filename to its official MSP-Podcast split.

    Only the two metadata columns are read, so the audio payload in the
    shards is never decoded.

    Args:
        build_dir: directory holding the build's parquet shards.

    Returns:
        Dict from filename key to "Train" or "Development".
    """
    shards = sorted(glob.glob(os.path.join(build_dir, "*.parquet")))
    if not shards:
        raise FileNotFoundError("no parquet shards under %s" % build_dir)
    table = pq.read_table(shards, columns=["FileName", "Split_Set"])
    keys = [os.path.basename(str(f)).rsplit(".", 1)[0]
            for f in table.column("FileName").to_pylist()]
    return dict(zip(keys, table.column("Split_Set").to_pylist()))


def cells_from_cuts(vad: np.ndarray, cuts: np.ndarray,
                    bins: int) -> np.ndarray:
    """Assign utterances to cells using boundaries fitted elsewhere.

    Args:
        vad: [N, 3] VAD means.
        cuts: [bins - 1, 3] boundaries from the training split.
        bins: bins per dimension.

    Returns:
        [N] cell ids.
    """
    digits = np.stack([np.searchsorted(cuts[:, d], vad[:, d], side="right")
                       for d in range(3)], axis=1)
    return (digits[:, 0] * bins + digits[:, 1]) * bins + digits[:, 2]


def alpha_from_map(probs: np.ndarray, n_ann: np.ndarray, ids: np.ndarray,
                   shares: Dict[int, np.ndarray]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Real-axis fraction of each panel's own disagreement.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.
        ids: [N] cell ids.
        shares: cell id -> [C, C] realness matrix.

    Returns:
        Tuple of [N] alpha and [N] boolean mask of rows the map covers.
        Unanimous panels have no disagreement to apportion and are given
        alpha zero, which leaves their target at the hard label where it
        already sits.
    """
    mass = pair_mass(probs, n_ann)
    total = mass.sum(axis=(1, 2))
    real = np.zeros(probs.shape[0])
    for cell, share in shares.items():
        sel = ids == cell
        if sel.any():
            real[sel] = (mass[sel] * share[None, :, :]).sum(axis=(1, 2))
    covered = np.isin(ids, list(shares))
    alpha = np.zeros(probs.shape[0])
    live = covered & (total > 1e-9)
    alpha[live] = np.clip(real[live] / total[live], 0.0, 1.0)
    return alpha, covered


def shuffle_map(shares: Dict[int, np.ndarray], rng: np.random.Generator
                ) -> Dict[int, np.ndarray]:
    """Permute each label pair's realness across cells, keeping its mean.

    The placebo has to remove the spatial claim without also making the
    targets harder or softer on average, or a difference against it would
    only show that softness matters somewhere. Permuting within each label
    pair holds that pair's corpus-average realness exactly fixed.

    Args:
        shares: the fitted map.
        rng: random generator.

    Returns:
        A map with the same values in different cells.
    """
    cells = sorted(shares)
    out = {c: np.zeros_like(shares[c]) for c in cells}
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            vals = np.array([shares[c][a, b] for c in cells])
            rng.shuffle(vals)
            for c, v in zip(cells, vals):
                out[c][a, b] = v
                out[c][b, a] = v
    return out


def main() -> None:
    """Fit on official Train and write the per-utterance lookup."""
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
    corpus = four_class_population(load_mspp(), args.min_inside,
                                   args.min_annotators)
    split_of = official_splits()
    split = np.array([split_of.get(k, "") for k in corpus.keys])
    is_train = split == "Train"
    print("four-class population %d rows: %d official Train, %d Development, "
          "%d unmatched" % (corpus.keys.size, int(is_train.sum()),
                            int((split == "Development").sum()),
                            int((split == "").sum())))
    if not is_train.any():
        raise RuntimeError("no rows matched the official Train split")

    train = subset(corpus, is_train)
    probs_tr, n_tr = train.probs, train.n_ann

    # Cell boundaries come from the training split only, for the same
    # reason the realness values do.
    qs = np.linspace(0.0, 1.0, args.bins + 1)[1:-1]
    cuts = np.quantile(train.vad, qs, axis=0)
    ids_tr = cells_from_cuts(train.vad, cuts, args.bins)
    shares, sizes = realness_map(probs_tr, n_tr, ids_tr, args.min_cell,
                                 args.max_pairs, rng)
    print("fitted %d of %d cells on Train, covering %.1f%% of Train rows"
          % (len(shares), args.bins ** 3,
             100.0 * np.isin(ids_tr, list(shares)).mean()))

    placebo = shuffle_map(shares, rng)

    # Apply to every row, Train-fitted boundaries and values throughout.
    ids_all = cells_from_cuts(corpus.vad, cuts, args.bins)
    alpha, covered = alpha_from_map(corpus.probs, corpus.n_ann, ids_all,
                                   shares)
    alpha_pl, _ = alpha_from_map(corpus.probs, corpus.n_ann, ids_all, placebo)

    tr = is_train & covered
    print("\nalpha on official Train rows the map covers (%d):" % int(tr.sum()))
    print("  real map    mean %.3f  p90 %.3f" % (alpha[tr].mean(),
                                                 np.percentile(alpha[tr], 90)))
    print("  placebo     mean %.3f  p90 %.3f"
          % (alpha_pl[tr].mean(), np.percentile(alpha_pl[tr], 90)))
    print("  correlation between them %.3f"
          % np.corrcoef(alpha[tr], alpha_pl[tr])[0, 1])

    print("\n  realness per label pair on Train (mean over cells, weighted):")
    w = np.array([sizes[c] for c in sorted(shares)], dtype=float)
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            vals = np.array([shares[c][a, b] for c in sorted(shares)])
            print("    %-18s %.3f  (range %.2f to %.2f)"
                  % ("%s / %s" % (NAMES[a], NAMES[b]),
                     float((vals * w).sum() / w.sum()), vals.min(),
                     vals.max()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cell_ids_sorted = np.array(sorted(shares), dtype=np.int64)
    np.savez_compressed(
        OUT,
        keys=corpus.keys,
        split=split,
        alpha=alpha.astype(np.float32),
        alpha_placebo=alpha_pl.astype(np.float32),
        covered=covered,
        probs=corpus.probs.astype(np.float32),
        n_annotators=corpus.n_ann.astype(np.int32),
        vad=corpus.vad.astype(np.float32),
        cell_id=ids_all.astype(np.int64),
        cuts=cuts.astype(np.float32),
        bins=np.int64(args.bins),
        map_cells=cell_ids_sorted,
        map_values=np.stack([shares[int(c)] for c in cell_ids_sorted]),
        map_placebo=np.stack([placebo[int(c)] for c in cell_ids_sorted]),
        map_sizes=np.array([sizes[int(c)] for c in cell_ids_sorted]),
        labels=np.array(NAMES),
    )
    print("\nwrote %s" % OUT)
    print("training reads alpha by filename key; rows absent from the file, "
          "or with covered=False, fall back to the hard label at weight 1.")


if __name__ == "__main__":
    main()
