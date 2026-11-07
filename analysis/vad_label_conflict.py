#!/usr/bin/env python3
"""Separate true categorical structure from annotation noise at matched VAD.

Two utterances can sit at the same point in valence-arousal-dominance
space and still be given different categorical labels. The difference is
either information the categorical label carries that the VAD coordinates
do not, or it is noise, in the sense that a different panel of annotators
would have produced a different answer. Those two readings have opposite
consequences. The first says the VAD coordinates are an incomplete
description of the emotion and a label assigned from geometry alone has an
error floor. The second says the conflict is an artefact of how few people
rated each utterance and will shrink as panels grow.

The two can be told apart because the same panel measures both. The spread
of labels ACROSS utterances at matched VAD is the apparent conflict. The
spread of labels WITHIN one utterance, across its own annotators, is that
utterance's own noise level. Sampling a finite panel inflates the first by
an amount the second lets us estimate, and subtracting it leaves an
unbiased estimate of the conflict that would survive an infinite panel.

Two estimators are computed and should agree.

Analytic. For an utterance with true label distribution pi rated by n
annotators, the observed p is multinomial, and
    E[1 - ||p||^2] = (1 - ||pi||^2) (n - 1) / n,
so u = (1 - ||p||^2) / (n - 1) is unbiased for the sampling variance
(1 - ||pi||^2) / n. Since
    E||p_i - p_j||^2 = ||pi_i - pi_j||^2 + (1-||pi_i||^2)/n_i
                                         + (1-||pi_j||^2)/n_j,
the quantity
    Dhat = ||p_i - p_j||^2 - u_i - u_j
is unbiased for the true squared divergence ||pi_i - pi_j||^2. It uses
every annotator and needs n >= 2.

Split-half. Each panel is split at random into halves A and B. Crossing
the halves makes the two sampling errors independent, so
    ||p_i^A - p_j^B||^2 - 0.5||p_i^A - p_i^B||^2 - 0.5||p_j^A - p_j^B||^2
is unbiased for the same target without assuming a multinomial. It is the
model-free check on the analytic estimator and needs n >= 4.

Reference quantities. The same estimators applied to randomly drawn pairs
give the total reliable between-utterance label variance. The ratio of the
VAD-matched value to that total is the share of reliable categorical
structure that the VAD coordinates fail to explain, which is the error
floor of any rule that assigns a label from VAD geometry alone.

Measurement error in VAD is handled explicitly. Each utterance's VAD is
itself an annotator mean whose standard error is known, and on MSP-Podcast
that error is about a third of the between-utterance spread. Matched pairs
are therefore reported across a sweep of radii rather than at a single
threshold, and the unbiased estimate of the true squared VAD distance is
reported beside each shell so the reader can see how far the smallest
shell really is from coincident.

Usage:
    python -m analysis.vad_label_conflict --corpus mspp
    python -m analysis.vad_label_conflict --corpus iemocap
"""

import argparse
import collections
import glob
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

REPO = pathlib.Path("/home/rml/Documents/pythontest/Emotion2Vec_Contrastive")
MSPP_CACHE = REPO / "cache/annotator_meta/mspp_build_subtypes.npz"
IEMOCAP_GLOB = ("/mnt/fast/huggingface/hub/datasets--cairocode--iemocap/"
                "snapshots/*/data/*.parquet")
OUT_DIR = pathlib.Path("analysis/vad_label_conflict")

MSPP_LABELS: List[str] = [
    "Neutral", "Happy", "Sad", "Angry", "Contempt", "Surprise", "Disgust",
    "Fear", "Other",
]
MSPP_SUBTYPES: List[str] = [
    "Neutral", "Happy", "Concerned", "Frustrated", "Angry", "Sad", "Amused",
    "Annoyed", "Excited", "Disappointed", "Contempt", "Surprise", "Disgust",
    "Confused", "Depressed", "Fear",
]
VAD_NAMES = ("valence", "arousal", "dominance")


@dataclass
class Corpus:
    """Annotator-level label and VAD data for one corpus.

    Attributes:
        name: corpus identifier.
        counts: [N, C] number of annotators choosing each categorical label.
        vad: [N, 3] annotator-mean VAD on the corpus's native scale.
        vad_std: [N, 3] annotator standard deviation of VAD.
        n_vad: [N] number of annotators contributing to the VAD mean.
        labels: names of the C categorical labels.
        keys: [N] utterance identifiers, so example pairs can be listened to.
        subtypes: optional [N, S] secondary label distribution.
        subtype_names: names of the S subtypes, if present.
    """

    name: str
    counts: np.ndarray
    vad: np.ndarray
    vad_std: np.ndarray
    n_vad: np.ndarray
    labels: List[str]
    keys: np.ndarray
    subtypes: Optional[np.ndarray] = None
    subtype_names: Optional[List[str]] = None

    @property
    def n_ann(self) -> np.ndarray:
        """Number of categorical annotators per utterance."""
        return self.counts.sum(axis=1)

    @property
    def probs(self) -> np.ndarray:
        """Observed label distribution per utterance."""
        return self.counts / np.maximum(self.n_ann, 1)[:, None]


def load_mspp(path: pathlib.Path = MSPP_CACHE) -> Corpus:
    """Load MSP-Podcast annotator statistics from the cached npz.

    Args:
        path: npz written by utils.annotator_meta.build_subtype_cache.

    Returns:
        A Corpus with integer annotator counts recovered from label_dist.
    """
    with np.load(path, allow_pickle=True) as z:
        n = np.asarray(z["n_annotators"], dtype=np.int64)
        counts = np.rint(np.asarray(z["label_dist"], dtype=np.float64)
                         * n[:, None]).astype(np.float64)
        corpus = Corpus(
            name="mspp",
            counts=counts,
            vad=np.asarray(z["vad_mean"], dtype=np.float64),
            vad_std=np.asarray(z["vad_std"], dtype=np.float64),
            n_vad=n,
            labels=list(MSPP_LABELS),
            keys=np.asarray(z["keys"]).astype(str),
            subtypes=np.asarray(z["subtype_dist"], dtype=np.float64),
            subtype_names=list(MSPP_SUBTYPES),
        )
    return corpus


def load_iemocap(pattern: str = IEMOCAP_GLOB,
                 single_choice: bool = True) -> Corpus:
    """Load IEMOCAP per-annotator categorical labels and VAD ratings.

    The release stores one JSON object per utterance mapping evaluator id
    to the categories that evaluator chose, and every utterance has
    exactly three evaluators. An evaluator may name more than one
    category, which 4.2 percent of votes do.

    Splitting such a vote evenly across the categories named would leave
    fractional counts, and the correction for sampling noise is derived
    for a panel of whole annotators: fractional counts break the
    split-half estimator outright, because rounding them back to integers
    no longer reconstructs the panel. Only the first category each
    evaluator named is therefore kept, which keeps every panel an exact
    three whole votes at the cost of discarding the secondary choice on
    those 4.2 percent of votes.

    Args:
        pattern: glob for the dataset's parquet shards.
        single_choice: keep only each evaluator's first named category.

    Returns:
        A Corpus with integer annotator counts.
    """
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError("no IEMOCAP parquet shards at %s" % pattern)
    cols = ["utterance_id", "annotators_categorical", "annotators_attribute",
            "valence", "activation", "dominance"]
    rows: List[Dict[str, object]] = []
    for shard in shards:
        rows.extend(pq.read_table(shard, columns=cols).to_pylist())

    vocab: Dict[str, int] = {}
    per_row: List[Dict[str, float]] = []
    vads: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    nvad: List[int] = []
    for row in rows:
        votes = json.loads(row["annotators_categorical"] or "{}")
        tally: Dict[str, float] = collections.defaultdict(float)
        for _, picks in votes.items():
            picks = [p for p in (picks or []) if p]
            if not picks:
                continue
            if single_choice:
                picks = picks[:1]
            for pick in picks:
                vocab.setdefault(pick, len(vocab))
                tally[pick] += 1.0 / len(picks)
        per_row.append(dict(tally))

        attr = json.loads(row["annotators_attribute"] or "{}")
        vals = np.array([v for v in attr.values() if v and len(v) == 3],
                        dtype=np.float64)
        if len(vals) == 0:
            vals = np.array([[row["valence"], row["activation"],
                              row["dominance"]]], dtype=np.float64)
        vads.append(vals.mean(axis=0))
        stds.append(vals.std(axis=0, ddof=1) if len(vals) > 1
                    else np.zeros(3))
        nvad.append(len(vals))

    order = sorted(vocab)
    index = {name: i for i, name in enumerate(order)}
    counts = np.zeros((len(per_row), len(order)), dtype=np.float64)
    for i, tally in enumerate(per_row):
        for name, weight in tally.items():
            counts[i, index[name]] = weight

    return Corpus(
        name="iemocap",
        counts=counts,
        vad=np.asarray(vads, dtype=np.float64),
        vad_std=np.asarray(stds, dtype=np.float64),
        n_vad=np.asarray(nvad, dtype=np.int64),
        labels=order,
        keys=np.asarray([str(r["utterance_id"]) for r in rows]),
    )


def within_dispersion(probs: np.ndarray, n_ann: np.ndarray) -> np.ndarray:
    """Unbiased per-utterance sampling variance of the label distribution.

    Estimates (1 - ||pi||^2) / n, the expected squared error of the
    observed distribution about the true one.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.

    Returns:
        [N] variance estimates.
    """
    n = np.maximum(np.asarray(n_ann, dtype=np.float64), 2.0)
    return (1.0 - (probs ** 2).sum(axis=1)) / (n - 1.0)


def split_half(counts: np.ndarray, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Split each utterance's annotator panel at random into two halves.

    Each half is drawn without replacement from the panel, so the two are
    independent given the true label distribution, which is what makes the
    crossed split-half estimator unbiased.

    Args:
        counts: [N, C] annotator counts, rounded to integers internally.
        rng: random generator.

    A panel of fewer than four cannot be split into two halves that each
    contain at least two annotators, and a panel whose counts are not
    whole numbers cannot be reconstructed from them at all. Both cases
    return NaN rather than an approximation, so that a corpus where this
    estimator does not apply reports nothing instead of reporting
    nonsense.

    Returns:
        Tuple of [N, C] distributions for half A and half B, NaN for any
        row that cannot be split.
    """
    if not np.allclose(counts, np.rint(counts)):
        return (np.full(counts.shape, np.nan), np.full(counts.shape, np.nan))
    ints = np.rint(counts).astype(np.int64)
    totals = ints.sum(axis=1)
    out_a = np.full(counts.shape, np.nan)
    out_b = np.full(counts.shape, np.nan)
    for i in range(counts.shape[0]):
        total = int(totals[i])
        if total < 4:
            continue
        half = total // 2
        urn = np.repeat(np.arange(counts.shape[1]), ints[i])
        rng.shuffle(urn)
        a = np.bincount(urn[:half], minlength=counts.shape[1])
        b = np.bincount(urn[half:2 * half], minlength=counts.shape[1])
        out_a[i] = a / half
        out_b[i] = b / half
    return out_a, out_b


def analytic_divergence(probs: np.ndarray, disp: np.ndarray,
                        idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    """Unbiased squared divergence between the true distributions of pairs.

    Args:
        probs: [N, C] observed label distributions.
        disp: [N] per-utterance sampling variance from within_dispersion.
        idx_i: [P] left indices of each pair.
        idx_j: [P] right indices of each pair.

    Returns:
        [P] unbiased estimates of ||pi_i - pi_j||^2. Individual values may
        be negative; only their mean over a set of pairs is meaningful.
    """
    diff = probs[idx_i] - probs[idx_j]
    return (diff ** 2).sum(axis=1) - disp[idx_i] - disp[idx_j]


def split_half_divergence(pa: np.ndarray, pb: np.ndarray, idx_i: np.ndarray,
                          idx_j: np.ndarray) -> np.ndarray:
    """Model-free unbiased squared divergence using crossed panel halves.

    Args:
        pa: [N, C] half-A distributions.
        pb: [N, C] half-B distributions.
        idx_i: [P] left indices of each pair.
        idx_j: [P] right indices of each pair.

    Returns:
        [P] estimates, NaN where either panel was too small to split.
    """
    cross = ((pa[idx_i] - pb[idx_j]) ** 2).sum(axis=1)
    own_i = ((pa[idx_i] - pb[idx_i]) ** 2).sum(axis=1)
    own_j = ((pa[idx_j] - pb[idx_j]) ** 2).sum(axis=1)
    return cross - 0.5 * own_i - 0.5 * own_j


def neighbour_pairs(vad_z: np.ndarray, max_radius: float, anchors: int,
                    per_anchor: int, rng: np.random.Generator
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample pairs of utterances that are close in standardised VAD space.

    Querying every point against every other would return hundreds of
    millions of pairs in the dense neutral region alone, so a bounded
    sample is taken: a fixed number of anchors, each contributing at most
    per_anchor neighbours drawn uniformly from its ball. Subsampling
    inside the ball rather than truncating keeps the pair set
    representative of the neighbourhood instead of of whichever points the
    tree happened to list first.

    Args:
        vad_z: [N, 3] VAD standardised by its between-utterance deviation.
        max_radius: largest neighbourhood radius to collect.
        anchors: how many anchor utterances to draw.
        per_anchor: cap on neighbours kept per anchor.
        rng: random generator.

    Returns:
        Tuple of left indices, right indices and their VAD distances.
    """
    tree = cKDTree(vad_z)
    total = vad_z.shape[0]
    picks = rng.choice(total, size=min(anchors, total), replace=False)
    left: List[np.ndarray] = []
    right: List[np.ndarray] = []
    for anchor in picks:
        ball = tree.query_ball_point(vad_z[anchor], max_radius)
        ball = np.asarray([b for b in ball if b != anchor], dtype=np.int64)
        if ball.size == 0:
            continue
        if ball.size > per_anchor:
            ball = rng.choice(ball, size=per_anchor, replace=False)
        left.append(np.full(ball.size, anchor, dtype=np.int64))
        right.append(ball)
    idx_i = np.concatenate(left)
    idx_j = np.concatenate(right)
    dist = np.linalg.norm(vad_z[idx_i] - vad_z[idx_j], axis=1)
    return idx_i, idx_j, dist


def random_pairs(total: int, count: int, rng: np.random.Generator
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Draw pairs of distinct utterances uniformly at random.

    Args:
        total: number of utterances.
        count: number of pairs to draw.
        rng: random generator.

    Returns:
        Tuple of left and right indices.
    """
    idx_i = rng.integers(0, total, size=count)
    idx_j = rng.integers(0, total, size=count)
    keep = idx_i != idx_j
    return idx_i[keep], idx_j[keep]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator,
                   draws: int = 400) -> Tuple[float, float]:
    """Mean of a pair statistic with a bootstrap standard error.

    Pairs sharing an anchor are not independent, so the usual standard
    error of the mean understates the uncertainty. Resampling the pairs
    themselves does not fix that either, but it is the honest minimum and
    the sample sizes here are large enough that it is not the binding
    constraint on any claim made from these numbers.

    Args:
        values: [P] per-pair statistics, NaN entries ignored.
        rng: random generator.
        draws: bootstrap resamples.

    Returns:
        Tuple of mean and bootstrap standard error.
    """
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan"), float("nan")
    means = np.empty(draws)
    for d in range(draws):
        idx = rng.integers(0, vals.size, size=vals.size)
        means[d] = vals[idx].mean()
    return float(vals.mean()), float(means.std(ddof=1))


def true_agreement(probs: np.ndarray, n_ann: np.ndarray) -> np.ndarray:
    """Unbiased estimate of the chance two annotators of one utterance agree.

    E[||p||^2] = ||pi||^2 + (1 - ||pi||^2) / n, so the observed collision
    probability overstates the true one and the inflation is removable.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.

    Returns:
        [N] estimates of ||pi||^2, clipped to [0, 1].
    """
    n = np.maximum(np.asarray(n_ann, dtype=np.float64), 2.0)
    est = (n * (probs ** 2).sum(axis=1) - 1.0) / (n - 1.0)
    return np.clip(est, 0.0, 1.0)


def true_vad_distance_sq(vad_z: np.ndarray, se2_z: np.ndarray,
                         idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    """Unbiased estimate of the squared VAD distance between true means.

    The measured separation of two annotator means exceeds the separation
    of the quantities they estimate by the two standard errors, so the
    measured distance alone would make every shell look tighter than it is.

    Args:
        vad_z: [N, 3] standardised VAD means.
        se2_z: [N, 3] squared standard errors on the same scale.
        idx_i: [P] left indices of each pair.
        idx_j: [P] right indices of each pair.

    Returns:
        [P] estimates, which may be negative for individual pairs.
    """
    d2 = ((vad_z[idx_i] - vad_z[idx_j]) ** 2).sum(axis=1)
    return d2 - se2_z[idx_i].sum(axis=1) - se2_z[idx_j].sum(axis=1)


def cross_disagreement(probs: np.ndarray, idx_i: np.ndarray,
                       idx_j: np.ndarray) -> np.ndarray:
    """Chance that one annotator of i and one of j give different labels.

    Needs no bias correction: the two panels are independent, so the
    product of the two observed distributions is already unbiased for the
    product of the true ones.

    Args:
        probs: [N, C] observed label distributions.
        idx_i: [P] left indices of each pair.
        idx_j: [P] right indices of each pair.

    Returns:
        [P] disagreement probabilities.
    """
    return 1.0 - (probs[idx_i] * probs[idx_j]).sum(axis=1)


def label_pair_excess(probs: np.ndarray, n_ann: np.ndarray,
                      idx_i: np.ndarray, idx_j: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose the excess disagreement across ordered label pairs.

    The off-diagonal entries of the excess sum to exactly the excess
    disagreement probability, so this says which label confusions at
    matched VAD survive the correction for annotation noise and which do
    not. The cross matrix is returned alongside because absolute excess
    scales with how often a confusion arises at all: a rare label pair
    shows a small excess because it is rare, not because it is noise, and
    only the ratio of the two separates those readings.

    Args:
        probs: [N, C] observed label distributions.
        n_ann: [N] annotators per utterance.
        idx_i: [P] left indices of each pair.
        idx_j: [P] right indices of each pair.

    Returns:
        Tuple of the [C, C] symmetric excess matrix and the [C, C]
        observed cross-confusion matrix.
    """
    cross = probs[idx_i].T @ probs[idx_j] / idx_i.size
    cross = 0.5 * (cross + cross.T)
    used = np.unique(np.concatenate([idx_i, idx_j]))
    n = np.maximum(np.asarray(n_ann[used], dtype=np.float64), 2.0)
    scale = (n / (n - 1.0))[:, None]
    within = (probs[used] * scale).T @ probs[used] / used.size
    within = 0.5 * (within + within.T)
    excess = cross - within
    np.fill_diagonal(excess, 0.0)
    return excess, cross


def summarise_shell(corpus: Corpus, probs: np.ndarray, disp: np.ndarray,
                    agree: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                    t2: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray,
                    mask: np.ndarray, rng: np.random.Generator
                    ) -> Dict[str, float]:
    """Compute every headline quantity for one set of pairs.

    All three excess readings are on the probability scale: the amount by
    which two annotators rating DIFFERENT utterances at matched VAD
    disagree more often than two annotators rating the SAME utterance.
    Half the squared divergence equals that excess exactly, so
    excess_direct and excess_analytic are the same estimator written two
    ways and agreeing is only an implementation check. excess_split_half
    is the independent one: it assumes no sampling model, only that a
    random half of a panel is exchangeable with the other half.

    Args:
        corpus: the corpus being analysed.
        probs: [N, C] observed label distributions.
        disp: [N] per-utterance sampling variance.
        agree: [N] unbiased within-utterance agreement.
        pa: [N, C] half-A distributions.
        pb: [N, C] half-B distributions.
        t2: [P] unbiased squared true VAD distances.
        idx_i: [P] left indices.
        idx_j: [P] right indices.
        mask: [P] boolean selecting the pairs in this shell.
        rng: random generator.

    Returns:
        Dictionary of pair count, VAD geometry and the divergence readings.
    """
    li, lj = idx_i[mask], idx_j[mask]
    if li.size == 0:
        return {"n_pairs": 0}
    cross = cross_disagreement(probs, li, lj)
    within = 1.0 - 0.5 * (agree[li] + agree[lj])
    analytic = analytic_divergence(probs, disp, li, lj)
    halves = split_half_divergence(pa, pb, li, lj)
    cross_m, cross_e = bootstrap_mean(cross, rng)
    exc_m, exc_e = bootstrap_mean(cross - within, rng)
    ana_m, ana_e = bootstrap_mean(0.5 * analytic, rng)
    half_m, half_e = bootstrap_mean(0.5 * halves, rng)
    return {
        "n_pairs": int(li.size),
        "mean_true_vad_dist_sq": float(t2[mask].mean()),
        "p_disagree_within": float(within.mean()),
        "p_disagree_cross": cross_m,
        "p_disagree_cross_se": cross_e,
        "excess_direct": exc_m,
        "excess_direct_se": exc_e,
        "excess_analytic": ana_m,
        "excess_analytic_se": ana_e,
        "excess_split_half": half_m,
        "excess_split_half_se": half_e,
    }


def excess_disagreement(probs: np.ndarray, agree: np.ndarray,
                        idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    """Excess chance of disagreement across utterances over within one.

    Equals half the unbiased squared divergence between the two true label
    distributions, and is zero when the two utterances are labelled from
    the same underlying distribution however noisy that distribution is.

    Args:
        probs: [N, C] observed label distributions.
        agree: [N] unbiased within-utterance agreement.
        idx_i: [P] left indices.
        idx_j: [P] right indices.

    Returns:
        [P] per-pair excess, meaningful only in the mean.
    """
    cross = 1.0 - (probs[idx_i] * probs[idx_j]).sum(axis=1)
    within = 1.0 - 0.5 * (agree[idx_i] + agree[idx_j])
    return cross - within


def shuffle_control(probs: np.ndarray, agree: np.ndarray, idx_i: np.ndarray,
                    idx_j: np.ndarray, rng: np.random.Generator,
                    draws: int = 20) -> Tuple[float, float]:
    """Excess disagreement after destroying the VAD correspondence only.

    A global random-pair reference is the wrong denominator here, because
    matching tightly on VAD is not a neutral filter: annotators who agree
    on an utterance's VAD tend to agree on its category too, so the
    tightest shells are populated by unusually clean utterances whose
    confident labels sit further apart than average. Permuting the right
    hand side of the shell's own pairs keeps both margins exactly as they
    are and removes nothing but the VAD matching, so the comparison is
    between the same utterances paired two ways.

    Args:
        probs: [N, C] observed label distributions.
        agree: [N] unbiased within-utterance agreement.
        idx_i: [P] left indices.
        idx_j: [P] right indices.
        rng: random generator.
        draws: permutations to average over.

    Returns:
        Tuple of mean excess under permutation and its spread across draws.
    """
    if idx_i.size == 0:
        return float("nan"), float("nan")
    means = np.empty(draws)
    for d in range(draws):
        perm = rng.permutation(idx_j)
        keep = perm != idx_i
        means[d] = excess_disagreement(probs, agree, idx_i[keep],
                                       perm[keep]).mean()
    return float(means.mean()), float(means.std(ddof=1))


def report_label_pairs(excess: np.ndarray, cross: np.ndarray,
                       labels: List[str], min_mass: float = 0.002,
                       top: int = 12) -> List[Dict[str, object]]:
    """Rank label pairs by how much of their conflict is irreducible.

    Two readings are returned per pair. The excess is the absolute
    contribution to irreducible conflict, and summing it over pairs
    recovers the headline. The real share is that excess divided by how
    often the confusion arises at all, which is what distinguishes a
    distinction annotators cannot make reliably from one that is simply
    rare. A share near zero means two utterances at matched VAD confuse
    these labels no more than one utterance's own annotators do, so the
    distinction carries no information beyond the panel's own variability.

    Args:
        excess: [C, C] symmetric excess matrix.
        cross: [C, C] observed cross-confusion matrix.
        labels: label names.
        min_mass: smallest cross-confusion mass worth reporting a share
            for, since the ratio is unstable for vanishingly rare pairs.
        top: how many pairs to list.

    Returns:
        List of records sorted by excess, largest first.
    """
    out: List[Dict[str, object]] = []
    size = len(labels)
    for a in range(size):
        for b in range(a + 1, size):
            mass = float(2.0 * cross[a, b])
            exc = float(2.0 * excess[a, b])
            out.append({
                "a": labels[a], "b": labels[b], "excess": exc, "mass": mass,
                "real_share": exc / mass if mass >= min_mass else None,
            })
    out.sort(key=lambda row: row["excess"], reverse=True)
    return out[:top]


def confident_conflict_examples(corpus: Corpus, probs: np.ndarray,
                                agree: np.ndarray, idx_i: np.ndarray,
                                idx_j: np.ndarray, mask: np.ndarray,
                                min_agree: float, top: int = 15
                                ) -> List[Dict[str, object]]:
    """List matched pairs whose conflict cannot be blamed on either panel.

    Both utterances must be individually well agreed on, so neither side
    of the disagreement is an utterance the annotators argued about. What
    is left is two confidently labelled utterances that occupy the same
    VAD coordinates and carry different names, which is the concrete form
    of the edge case this analysis is looking for.

    Args:
        corpus: the corpus being analysed.
        probs: [N, C] observed label distributions.
        agree: [N] unbiased within-utterance agreement.
        idx_i: [P] left indices.
        idx_j: [P] right indices.
        mask: [P] boolean selecting the tightest shell.
        min_agree: agreement floor both utterances must clear.
        top: how many examples to list.

    Returns:
        List of example records, most divergent first.
    """
    li, lj = idx_i[mask], idx_j[mask]
    keep = (agree[li] >= min_agree) & (agree[lj] >= min_agree)
    li, lj = li[keep], lj[keep]
    if li.size == 0:
        return []
    div = ((probs[li] - probs[lj]) ** 2).sum(axis=1)
    order = np.argsort(-div)[:top]
    records: List[Dict[str, object]] = []
    for pos in order:
        a, b = int(li[pos]), int(lj[pos])
        records.append({
            "key_a": str(corpus.keys[a]),
            "key_b": str(corpus.keys[b]),
            "label_a": corpus.labels[int(probs[a].argmax())],
            "label_b": corpus.labels[int(probs[b].argmax())],
            "votes_a": "%d/%d" % (int(round(corpus.counts[a].max())),
                                  int(round(corpus.n_ann[a]))),
            "votes_b": "%d/%d" % (int(round(corpus.counts[b].max())),
                                  int(round(corpus.n_ann[b]))),
            "vad_a": [round(float(v), 2) for v in corpus.vad[a]],
            "vad_b": [round(float(v), 2) for v in corpus.vad[b]],
        })
    return records


def region_map(corpus: Corpus, probs: np.ndarray, agree: np.ndarray,
               idx_i: np.ndarray, idx_j: np.ndarray, mask: np.ndarray
               ) -> List[Dict[str, object]]:
    """Excess conflict broken down by where in VAD space the pair sits.

    Cells are the octants cut at the median of each dimension, which keeps
    every cell populated. A high-excess cell is a region where the VAD
    coordinates are a weak description of the emotion; a near-zero cell is
    a region where they are close to sufficient.

    Args:
        corpus: the corpus being analysed.
        probs: [N, C] observed label distributions.
        agree: [N] unbiased within-utterance agreement.
        idx_i: [P] left indices.
        idx_j: [P] right indices.
        mask: [P] boolean selecting the tightest shell.

    Returns:
        One record per octant, highest excess first.
    """
    li, lj = idx_i[mask], idx_j[mask]
    med = np.median(corpus.vad, axis=0)
    side = (corpus.vad >= med).astype(np.int64)
    cell = side[:, 0] * 4 + side[:, 1] * 2 + side[:, 2]
    same = cell[li] == cell[lj]
    li, lj = li[same], lj[same]
    cross = cross_disagreement(probs, li, lj)
    within = 1.0 - 0.5 * (agree[li] + agree[lj])
    excess = cross - within
    out: List[Dict[str, object]] = []
    for code in range(8):
        sel = cell[li] == code
        if sel.sum() < 200:
            continue
        name = "%s-V %s-A %s-D" % (
            "hi" if code & 4 else "lo",
            "hi" if code & 2 else "lo",
            "hi" if code & 1 else "lo")
        modal = collections.Counter(
            corpus.labels[int(probs[i].argmax())] for i in li[sel])
        out.append({
            "cell": name,
            "n_pairs": int(sel.sum()),
            "p_disagree_within": float(within[sel].mean()),
            "excess": float(excess[sel].mean()),
            "top_labels": [n for n, _ in modal.most_common(3)],
        })
    out.sort(key=lambda row: row["excess"], reverse=True)
    return out


def subtype_profiles(corpus: Corpus, probs: np.ndarray, agree: np.ndarray,
                     idx_i: np.ndarray, idx_j: np.ndarray, mask: np.ndarray,
                     min_agree: float, top: int = 4
                     ) -> List[Dict[str, object]]:
    """Describe the secondary labels on either side of a surviving conflict.

    Where two confidently labelled utterances sit at the same VAD and
    disagree, the secondary labels are the obvious place to look for what
    the primary label is responding to. This is descriptive only: the
    subtype field is multi-label and its annotator counts are not
    recoverable from the cache, so no bias correction is applied and no
    significance is claimed.

    Args:
        corpus: the corpus being analysed.
        probs: [N, C] observed label distributions.
        agree: [N] unbiased within-utterance agreement.
        idx_i: [P] left indices.
        idx_j: [P] right indices.
        mask: [P] boolean selecting the tightest shell.
        min_agree: agreement floor both utterances must clear.
        top: how many label pairs to profile.

    Returns:
        One record per label pair, each carrying the mean subtype profile
        of the two sides.
    """
    if corpus.subtypes is None or corpus.subtype_names is None:
        return []
    li, lj = idx_i[mask], idx_j[mask]
    keep = (agree[li] >= min_agree) & (agree[lj] >= min_agree)
    li, lj = li[keep], lj[keep]
    mod_i = probs[li].argmax(axis=1)
    mod_j = probs[lj].argmax(axis=1)
    differ = mod_i != mod_j
    li, lj, mod_i, mod_j = li[differ], lj[differ], mod_i[differ], mod_j[differ]
    pairs = collections.Counter(
        (min(a, b), max(a, b)) for a, b in zip(mod_i.tolist(), mod_j.tolist()))
    out: List[Dict[str, object]] = []
    for (a, b), count in pairs.most_common(top):
        sel_a = np.concatenate([li[mod_i == a], lj[mod_j == a]])
        sel_b = np.concatenate([li[mod_i == b], lj[mod_j == b]])
        if sel_a.size == 0 or sel_b.size == 0:
            continue
        prof_a = corpus.subtypes[sel_a].mean(axis=0)
        prof_b = corpus.subtypes[sel_b].mean(axis=0)
        gap = prof_a - prof_b
        order = np.argsort(-np.abs(gap))[:5]
        out.append({
            "label_a": corpus.labels[a],
            "label_b": corpus.labels[b],
            "n_pairs": int(count),
            "subtype_gaps": [
                {"subtype": corpus.subtype_names[k],
                 "mean_a": round(float(prof_a[k]), 3),
                 "mean_b": round(float(prof_b[k]), 3)}
                for k in order],
        })
    return out


def coincident_unanimous_conflict(corpus: Corpus
                                  ) -> Dict[str, object]:
    """Count label conflict with no estimator and no correction at all.

    Restricting to utterances whose annotators were unanimous on the
    category AND gave identical VAD removes the need for any of the
    machinery above. Those panels have no internal disagreement to
    correct for, and their VAD is known exactly rather than estimated, so
    grouping them by exact VAD coordinate and counting how often two
    members carry different labels is a direct measurement. It is the
    plainest version of the result and a check that the corrected
    estimates are not an artefact of the correction.

    The baseline is the same quantity for random pairs drawn from this
    same set of utterances, so the comparison holds the population fixed
    and varies only whether the two utterances share VAD coordinates.

    Args:
        corpus: the corpus to measure.

    Returns:
        Dictionary carrying the counts, the conflict rate at coincident
        VAD, the same-population baseline, and the largest groups.
    """
    probs = corpus.probs
    unanimous = probs.max(axis=1) >= 1.0 - 1e-9
    exact_vad = corpus.vad_std.mean(axis=1) == 0.0
    sel = np.where(unanimous & exact_vad)[0]
    if sel.size < 2:
        return {"n_utterances": int(sel.size)}

    labels = probs[sel].argmax(axis=1)
    groups: Dict[Tuple[float, ...], List[int]] = collections.defaultdict(list)
    for pos, idx in enumerate(sel):
        groups[tuple(corpus.vad[idx])].append(int(labels[pos]))

    n_pairs = 0
    n_differ = 0
    rows: List[Tuple[int, Tuple[float, ...], List[int]]] = []
    for coord, members in groups.items():
        size = len(members)
        if size < 2:
            continue
        counts = np.bincount(members, minlength=len(corpus.labels))
        pairs = size * (size - 1) // 2
        same = int((counts * (counts - 1) // 2).sum())
        n_pairs += pairs
        n_differ += pairs - same
        rows.append((size, coord, counts.tolist()))

    share = np.bincount(labels, minlength=len(corpus.labels)).astype(float)
    share /= share.sum()
    rows.sort(reverse=True)
    return {
        "n_utterances": int(sel.size),
        "n_groups": len(rows),
        "n_pairs": int(n_pairs),
        "conflict_at_coincident_vad": float(n_differ / n_pairs) if n_pairs
        else float("nan"),
        "same_population_baseline": float(1.0 - (share ** 2).sum()),
        "largest_groups": [
            {"vad": [round(float(v), 1) for v in coord],
             "n": size,
             "labels": {corpus.labels[k]: int(c)
                        for k, c in enumerate(counts) if c}}
            for size, coord, counts in rows[:10]],
    }


def subset(corpus: Corpus, keep: np.ndarray) -> Corpus:
    """Restrict a corpus to a boolean selection of its utterances.

    Args:
        corpus: the corpus to restrict.
        keep: [N] boolean mask.

    Returns:
        A new Corpus holding only the selected rows.
    """
    return Corpus(
        name=corpus.name, counts=corpus.counts[keep], vad=corpus.vad[keep],
        vad_std=corpus.vad_std[keep], n_vad=corpus.n_vad[keep],
        labels=corpus.labels, keys=corpus.keys[keep],
        subtypes=None if corpus.subtypes is None else corpus.subtypes[keep],
        subtype_names=corpus.subtype_names)


def analyse(corpus: Corpus, args: argparse.Namespace,
            rng: np.random.Generator, verbose: bool = True
            ) -> Dict[str, object]:
    """Run the matched-VAD conflict analysis on one set of utterances.

    Args:
        corpus: utterances to analyse, already filtered to a stratum.
        args: parsed command line arguments.
        rng: random generator.
        verbose: print the tables as well as returning them.

    Returns:
        Dictionary of every quantity computed, suitable for JSON.
    """
    total = corpus.counts.shape[0]
    probs = corpus.probs
    n_ann = corpus.n_ann
    disp = within_dispersion(probs, n_ann)
    agree = true_agreement(probs, n_ann)
    pa, pb = split_half(corpus.counts, rng)

    scale = corpus.vad.std(axis=0)
    vad_z = (corpus.vad - corpus.vad.mean(axis=0)) / scale
    se2_z = (corpus.vad_std ** 2) / np.maximum(corpus.n_vad, 1)[:, None]
    se2_z = se2_z / (scale ** 2)

    idx_i, idx_j, dist = neighbour_pairs(vad_z, args.max_radius, args.anchors,
                                         args.per_anchor, rng)
    t2 = true_vad_distance_sq(vad_z, se2_z, idx_i, idx_j)

    if verbose:
        print("  %d utterances, VAD standard error %s of a deviation"
              % (total, np.round(np.sqrt(se2_z.mean(axis=0)), 2)))
        print("  P(two annotators of the same utterance differ) = %.3f"
              % float(1.0 - agree.mean()))
        print("  %d matched pairs sampled" % idx_i.size)
        print("\n  %-11s %8s %8s %8s %8s %8s %7s" % (
            "VAD radius", "pairs", "within", "matched", "split-h",
            "shuffled", "share"))

    edges = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    shells: List[Dict[str, object]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() < 200:
            continue
        stats = summarise_shell(corpus, probs, disp, agree, pa, pb, t2,
                                idx_i, idx_j, mask, rng)
        shuf, shuf_sd = shuffle_control(probs, agree, idx_i[mask],
                                        idx_j[mask], rng)
        stats["radius_lo"] = lo
        stats["radius_hi"] = hi
        stats["excess_shuffled"] = shuf
        stats["excess_shuffled_sd"] = shuf_sd
        stats["share_unexplained"] = (stats["excess_direct"] / shuf
                                      if shuf else float("nan"))
        shells.append(stats)
        if verbose:
            print("  %-11s %8d %8.3f %8.3f %8.3f %8.3f %6.0f%%" % (
                "%.2f-%.2f" % (lo, hi), stats["n_pairs"],
                stats["p_disagree_within"], stats["excess_direct"],
                stats["excess_split_half"], shuf,
                100.0 * stats["share_unexplained"]))

    tight = dist < edges[1]
    out: Dict[str, object] = {
        "n_utterances": int(total),
        "p_disagree_same_utterance": float(1.0 - agree.mean()),
        "vad_se_in_sd_units": np.sqrt(se2_z.mean(axis=0)).tolist(),
        "n_matched_pairs": int(idx_i.size),
        "shells": shells,
    }
    if tight.sum() < 200:
        return out

    exc_mat, cross_mat = label_pair_excess(probs, n_ann, idx_i[tight],
                                           idx_j[tight])
    out["label_pairs"] = report_label_pairs(exc_mat, cross_mat, corpus.labels)
    if verbose:
        print("\n  label pairs at matched VAD (excess, how often the "
              "confusion arises, share of it that is real):")
        for row in out["label_pairs"]:
            share = ("  -  " if row["real_share"] is None
                     else "%4.0f%%" % (100.0 * row["real_share"]))
            print("    %-9s vs %-9s  %+.4f   %.4f   %s"
                  % (row["a"], row["b"], row["excess"], row["mass"], share))
    out["regions"] = region_map(corpus, probs, agree, idx_i, idx_j, tight)
    out["examples"] = confident_conflict_examples(
        corpus, probs, agree, idx_i, idx_j, tight, args.min_agree)
    out["subtype_profiles"] = subtype_profiles(
        corpus, probs, agree, idx_i, idx_j, tight, args.min_agree)
    if not verbose:
        return out

    print("\n  by VAD octant, tightest shell:")
    for row in out["regions"]:
        print("    %-16s n=%-6d within=%.3f excess=%+.3f  %s"
              % (row["cell"], row["n_pairs"], row["p_disagree_within"],
                 row["excess"], ", ".join(row["top_labels"])))
    if out["examples"]:
        print("\n  unanimously labelled utterances at matched VAD that "
              "disagree:")
        for row in out["examples"][:6]:
            print("    %-24s %-9s %-5s vs %-24s %-9s %-5s  VAD %s"
                  % (row["key_a"], row["label_a"], row["votes_a"],
                     row["key_b"], row["label_b"], row["votes_b"],
                     row["vad_a"]))
    for row in out["subtype_profiles"]:
        print("\n  secondary labels behind %s vs %s (%d pairs):"
              % (row["label_a"], row["label_b"], row["n_pairs"]))
        for gap in row["subtype_gaps"]:
            print("    %-14s %-9s %.3f   %-9s %.3f"
                  % (gap["subtype"], row["label_a"], gap["mean_a"],
                     row["label_b"], gap["mean_b"]))
    return out


def main() -> None:
    """Run the analysis for one corpus, whole and split by VAD dispersion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="mspp",
                        choices=["mspp", "iemocap"], help="corpus to analyse")
    parser.add_argument("--anchors", type=int, default=20000,
                        help="anchor utterances for neighbour sampling")
    parser.add_argument("--per-anchor", type=int, default=40,
                        help="neighbours kept per anchor")
    parser.add_argument("--max-radius", type=float, default=1.0,
                        help="largest neighbourhood radius, in VAD deviations")
    parser.add_argument("--min-annotators", type=int, default=5,
                        help="drop utterances rated by fewer than this many")
    parser.add_argument("--min-agree", type=float, default=0.6,
                        help="agreement floor for the confident-pair subsets")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    corpus = load_mspp() if args.corpus == "mspp" else load_iemocap()
    corpus = subset(corpus, corpus.n_ann >= args.min_annotators)
    print("corpus %s: %d utterances, %d labels, panels of %d to %d"
          % (corpus.name, corpus.counts.shape[0], len(corpus.labels),
             int(corpus.n_ann.min()), int(corpus.n_ann.max())))

    # Strata are cut on annotator dispersion in VAD, which is measured on a
    # different channel from the categorical outcome. Cutting on label
    # agreement instead would select on the outcome and inflate the very
    # contrast being measured.
    spread = corpus.vad_std.mean(axis=1)
    cuts = np.quantile(spread, [1.0 / 3.0, 2.0 / 3.0])
    strata: List[Tuple[str, np.ndarray]] = [
        ("all", np.ones(spread.shape[0], dtype=bool)),
        ("clear (low VAD dispersion)", spread <= cuts[0]),
        ("middle", (spread > cuts[0]) & (spread <= cuts[1])),
        ("contested (high VAD dispersion)", spread > cuts[1]),
    ]

    results: Dict[str, object] = {"corpus": corpus.name,
                                  "labels": corpus.labels,
                                  "args": vars(args),
                                  "vad_dispersion_cuts": cuts.tolist()}
    plain = coincident_unanimous_conflict(corpus)
    results["coincident_unanimous"] = plain
    if plain.get("n_pairs"):
        print("\nplain count, no correction applied: %d utterances whose "
              "annotators were\nunanimous on the label and identical on "
              "VAD, in %d coincident groups."
              % (plain["n_utterances"], plain["n_groups"]))
        print("  two at the SAME VAD coordinate carry different labels  %.3f"
              % plain["conflict_at_coincident_vad"])
        print("  two drawn at random from the same utterances           %.3f"
              % plain["same_population_baseline"])
        print("  largest coincident groups:")
        for row in plain["largest_groups"][:6]:
            print("    %-16s n=%-4d %s" % (row["vad"], row["n"], row["labels"]))

    for name, mask in strata:
        print("\n=== %s ===" % name)
        results[name] = analyse(subset(corpus, mask), args, rng)

    whole = results["all"]
    if whole["shells"]:
        tight = whole["shells"][0]
        print("\nheadline, tightest VAD shell of the whole corpus:")
        print("  two annotators of the same utterance differ      %.3f"
              % whole["p_disagree_same_utterance"])
        print("  excess for a VAD-matched different utterance     %.3f"
              % tight["excess_direct"])
        print("  the same utterances paired without VAD matching  %.3f"
              % tight["excess_shuffled"])
        print("  share of reliable label structure VAD misses     %.0f%%"
              % (100.0 * tight["share_unexplained"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("%s.json" % corpus.name)
    out.write_text(json.dumps(results, indent=2))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
