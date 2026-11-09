#!/usr/bin/env python3
"""Build the continuum targets: PC coordinates, low-rank labels, and a kernel.

Three target families, all fitted on the official MSP-Podcast Train split
alone so no validation row influences a training target.

PC COORDINATES. The per-utterance annotator label distribution lives near
a three-dimensional surface inside the nine-label simplex: the first three
principal components hold about 91 percent of the variance and the audio
recovers them at R^2 0.64, 0.50 and 0.30, while the fourth and fifth come
back negative. Those coordinates are the regression target for the
auxiliary-head arms, and the comparison arm regresses VAD instead, so a
gain can be attributed to this space rather than to having an auxiliary
regression at all.

LOW-RANK LABELS. Training on the raw distribution spends part of every
target on components four and up, which the measurement says the signal
does not carry. Projecting onto the first three components and
reconstructing removes that without changing anything else, so the arm is
one line different from the soft-label arm that already works.

KERNEL. A parameter-free label-smoothing kernel: given that one annotator
of an utterance chose a, how often did another annotator of the SAME
utterance choose b. That conditional is measured rather than tuned, it
applies to every utterance including unanimous ones, and it is a single
small matrix, so it carries over to corpora with no panel data.

Usage:
    python -m analysis.build_continuum_targets
"""

import argparse
import glob
import os
import pathlib
from typing import Dict, Tuple

import numpy as np
import pyarrow.parquet as pq

from analysis.vad_label_conflict import MSPP_LABELS, load_mspp

BUILD = "/mnt/fast/mspp_build/data"
OUT = pathlib.Path("analysis/continuum/mspp_continuum.npz")
RANK = 3
NUM_CLASSES = 4


def official_splits(build_dir: str = BUILD) -> Dict[str, str]:
    """Filename key to official split, reading only metadata columns."""
    shards = sorted(glob.glob(os.path.join(build_dir, "*.parquet")))
    table = pq.read_table(shards, columns=["FileName", "Split_Set"])
    keys = [os.path.basename(str(f)).rsplit(".", 1)[0]
            for f in table.column("FileName").to_pylist()]
    return dict(zip(keys, table.column("Split_Set").to_pylist()))


def joint_label_matrix(probs: np.ndarray, n_ann: np.ndarray) -> np.ndarray:
    """Unbiased sum over utterances of the outer product of true shares.

    For two annotators drawn from one panel, the chance they pick a and b
    is pi_a pi_b off the diagonal and pi_a squared on it. The observed
    product is biased for both, upward on the diagonal and downward off
    it, so each is corrected separately before summing.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.

    Returns:
        [C, C] symmetric non-negative matrix.
    """
    n = np.maximum(n_ann.astype(np.float64), 2.0)
    off = (probs[:, :, None] * probs[:, None, :]
           * (n / (n - 1.0))[:, None, None]).sum(axis=0)
    diag = ((n[:, None] * probs ** 2 - probs)
            / (n - 1.0)[:, None]).sum(axis=0)
    np.fill_diagonal(off, 0.0)
    out = off + np.diag(diag)
    return np.maximum(0.5 * (out + out.T), 0.0)


def main() -> None:
    """Fit all three target families on official Train and write them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-annotators", type=int, default=5,
                        help="smallest panel contributing to the fits")
    args = parser.parse_args()

    corpus = load_mspp()
    keep = corpus.n_ann >= args.min_annotators
    probs = corpus.probs[keep]
    n_ann = corpus.n_ann[keep]
    keys = corpus.keys[keep]

    split_of = official_splits()
    split = np.array([split_of.get(k, "") for k in keys])
    is_train = split == "Train"
    print("utterances %d: %d Train, %d Development, %d unmatched"
          % (len(keys), int(is_train.sum()),
             int((split == "Development").sum()), int((split == "").sum())))

    # ---- principal subspace, fitted on Train only -------------------
    train_probs = probs[is_train]
    mean = train_probs.mean(axis=0)
    centred = train_probs - mean
    cov = (centred.T @ centred) / (centred.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    basis = vecs[:, :RANK]
    ratio = vals / vals.sum()
    print("\nprincipal components fitted on Train (%d rows)"
          % train_probs.shape[0])
    for p in range(min(6, len(vals))):
        print("  PC%-2d %6.1f%% cumulative %5.1f%%  top: %s"
              % (p + 1, 100 * ratio[p], 100 * ratio[:p + 1].sum(),
                 ", ".join("%s %+.2f" % (MSPP_LABELS[t], vecs[t, p])
                           for t in np.argsort(-np.abs(vecs[:, p]))[:2])))
    print("  rank %d retains %.1f%% of the variance"
          % (RANK, 100 * ratio[:RANK].sum()))

    coords = (probs - mean) @ basis
    # Standardise the coordinates using Train statistics, so the auxiliary
    # regression loss is not dominated by whichever component happens to
    # have the largest scale.
    c_mean = coords[is_train].mean(axis=0)
    c_std = coords[is_train].std(axis=0)
    coords_z = (coords - c_mean) / np.maximum(c_std, 1e-8)
    print("  coordinate scales on Train: %s" % np.round(c_std, 4))

    # ---- low-rank soft labels ---------------------------------------
    recon = mean + (probs - mean) @ basis @ basis.T
    four = np.clip(recon[:, :NUM_CLASSES], 1e-6, None)
    lowrank = four / four.sum(axis=1, keepdims=True)
    raw4 = np.clip(probs[:, :NUM_CLASSES], 0.0, None)
    raw4 = raw4 / np.maximum(raw4.sum(axis=1, keepdims=True), 1e-9)
    shift = np.abs(lowrank - raw4).sum(axis=1).mean() / 2.0
    print("\nlow-rank soft labels")
    print("  mean total-variation shift from the raw distribution: %.4f"
          % shift)
    print("  rows whose argmax changes: %.2f%%"
          % (100 * (lowrank.argmax(1) != raw4.argmax(1)).mean()))

    # ---- substitution kernel ----------------------------------------
    joint = joint_label_matrix(probs[is_train], n_ann[is_train])
    joint4 = joint[:NUM_CLASSES, :NUM_CLASSES]
    kernel = joint4 / joint4.sum(axis=1, keepdims=True)
    print("\nsubstitution kernel, P(another annotator says b | one said a)")
    print("  %-9s %s" % ("given", "  ".join("%9s" % MSPP_LABELS[b]
                                            for b in range(NUM_CLASSES))))
    for a in range(NUM_CLASSES):
        print("  %-9s %s" % (MSPP_LABELS[a],
                             "  ".join("%9.3f" % kernel[a, b]
                                       for b in range(NUM_CLASSES))))
    print("  diagonal mean %.3f, so smoothing strength is %.3f"
          % (np.diag(kernel).mean(), 1.0 - np.diag(kernel).mean()))

    # ---- ground cost for the transport loss -------------------------
    # Distance between labels, from how often annotators substitute them.
    # Used to price WHICH wrong class the model picks: confusing the pair
    # people actually confuse should cost less than confusing a pair they
    # never do.
    #
    # Against a one-hot target the transport distance has a closed form,
    # since all mass must move to the true class: W(p, delta_y) = sum_a
    # p[a] C[a, y]. That is minimised at p = delta_y with cost zero, so
    # the term agrees with cross-entropy at the optimum and cannot starve
    # a class the way a smoothing kernel can.
    freq4 = probs[is_train][:, :NUM_CLASSES].sum(axis=0)
    rate4 = joint4 / np.maximum(np.sqrt(np.outer(freq4, freq4)), 1e-9)
    cost = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            if a != b:
                cost[a, b] = -np.log(max(rate4[a, b], 1e-9))
    off = cost[~np.eye(NUM_CLASSES, dtype=bool)]
    cost = cost / off.mean()
    print("\nground cost for the transport loss (mean off-diagonal 1.0)")
    print("  %-9s %s" % ("", "  ".join("%8s" % MSPP_LABELS[b]
                                       for b in range(NUM_CLASSES))))
    for a in range(NUM_CLASSES):
        print("  %-9s %s" % (MSPP_LABELS[a],
                             "  ".join("%8.3f" % cost[a, b]
                                       for b in range(NUM_CLASSES))))
    # Report the NORMALISED costs; off was captured before the
    # division, so printing it directly mislabels the scale.
    off_n = cost[~np.eye(NUM_CLASSES, dtype=bool)]
    print("  spread: cheapest %.3f, dearest %.3f, ratio %.2f"
          % (off_n.min(), off_n.max(), off_n.max() / off_n.min()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        ot_cost=cost.astype(np.float32),
        keys=keys,
        split=split,
        pc_coords=coords_z.astype(np.float32),
        pc_basis=basis.astype(np.float32),
        pc_mean=mean.astype(np.float32),
        pc_coord_mean=c_mean.astype(np.float32),
        pc_coord_std=c_std.astype(np.float32),
        pc_var_ratio=ratio[:RANK].astype(np.float32),
        lowrank_target=lowrank.astype(np.float32),
        raw_target=raw4.astype(np.float32),
        kernel=kernel.astype(np.float32),
        labels=np.array(MSPP_LABELS[:NUM_CLASSES]),
    )
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
