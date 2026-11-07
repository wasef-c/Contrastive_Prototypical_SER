#!/usr/bin/env python3
"""
Prototypicality calculation based on VAD distance
"""

import math

import numpy as np
import torch
import torch.nn as nn


def calculate_difficulty(valence, arousal, dominance, label, expected_vad):
    """
    Calculate prototypicality (difficulty) as Euclidean distance in VAD space

    Lower difficulty = more prototypical (close to class prototype)
    Higher difficulty = more atypical (far from class prototype)

    Args:
        valence: float (0-1 normalized)
        arousal: float (0-1 normalized)
        dominance: float (0-1 normalized)
        label: int (0-3)
        expected_vad: dict mapping label to [V, A, D] prototype

    Returns:
        float: Euclidean distance (0-3 range typically)
    """
    actual_vad = [valence, arousal, dominance]
    expected = expected_vad.get(label)

    if expected is None:
        return 0.0

    # Euclidean distance
    distance = math.sqrt(sum((float(a) - float(e)) ** 2 for a, e in zip(actual_vad, expected)))

    # Guard against NaN or inf
    if math.isnan(distance) or math.isinf(distance):
        return 0.0

    return distance


# Datasets with real VAD annotations (others get default 0.5/0.5/0.5 which is meaningless)
DATASETS_WITH_VAD = {"IEMO", "MSPI", "MSPP"}


_CENTROID_CACHE: "dict[int, torch.Tensor]" = {}


def _centroids_from_dict(expected_vad: dict) -> torch.Tensor:
    """Build (and cache) a [num_classes, 3] CPU tensor from an expected_vad dict.

    Cached by id(expected_vad) since the dict comes from config and is reused
    every batch. Avoids rebuilding the tensor on each call.

    Args:
        expected_vad: dict mapping label (int) to [V, A, D] prototype list.

    Returns:
        [num_classes, 3] float32 tensor of class centroids.
    """
    key = id(expected_vad)
    cached = _CENTROID_CACHE.get(key)
    if cached is not None:
        return cached

    num_classes = max(expected_vad.keys()) + 1
    centroids = torch.zeros(num_classes, 3, dtype=torch.float32)
    for label, vad in expected_vad.items():
        centroids[label] = torch.tensor(vad, dtype=torch.float32)
    _CENTROID_CACHE[key] = centroids
    return centroids


def batch_calculate_difficulty(batch, expected_vad):
    """
    Return per-sample difficulty for a batch.

    Fast path: if the batch carries 'difficulty' (precomputed at dataset init),
    return it directly. Samples with static centroids never need to recompute.

    Fallback (vectorized): compute from VAD + expected_vad. Used when the batch
    has no precomputed difficulty (e.g. older collate, ad-hoc batches).

    Samples from datasets without VAD annotations (CMUMOSEI, SAMSEMO) get
    difficulty=0 (fully prototypical) since their VAD values are fake defaults.

    Args:
        batch: dict with 'valence', 'arousal', 'dominance', 'label' tensors
               and 'dataset' list of corpus name strings.
        expected_vad: dict mapping label -> [V, A, D] prototype.

    Returns:
        tensor: [batch_size] difficulty scores (CPU float32).
    """
    if 'difficulty' in batch and isinstance(batch['difficulty'], torch.Tensor):
        return batch['difficulty']

    labels = batch['label']
    valence = batch['valence'].float()
    arousal = batch['arousal'].float()
    dominance = batch['dominance'].float()

    centroids = _centroids_from_dict(expected_vad)
    expected = centroids[labels]
    actual = torch.stack([valence, arousal, dominance], dim=1)
    diff = ((actual - expected) ** 2).sum(dim=1).sqrt()

    dataset_names = batch.get('dataset', None)
    if dataset_names is not None:
        mask = torch.tensor(
            [n in DATASETS_WITH_VAD for n in dataset_names],
            dtype=torch.float32,
        )
        diff = diff * mask

    return torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)


# Default class prototypes (normalized to 0-1 range)
DEFAULT_EXPECTED_VAD = {
    0: [0.5, 0.375, 0.5],   # neutral (mid valence, low-mid arousal, mid dominance)
    1: [0.75, 0.7, 0.7],    # happy (high valence, high arousal, high dominance)
    2: [0.2, 0.3, 0.25],    # sad (low valence, low arousal, low dominance)
    3: [0.2, 0.8, 0.75],    # anger (low valence, high arousal, high dominance)
}


def batch_difficulty_tensor(batch, centroids, device):
    """
    Tensor-based, on-graph difficulty. Gradients flow from difficulty back into
    `centroids` (when it is a learnable Parameter).

    Args:
        batch: dict with 'valence','arousal','dominance','label' tensors and 'dataset' list
        centroids: [C, 3] tensor (Parameter or buffer)
        device: torch device

    Returns:
        [B] tensor of Euclidean distances. Non-VAD-dataset samples get 0.
    """
    labels = batch['label'].to(device)  # [B]
    B = labels.shape[0]

    valence = batch['valence'].float().to(device)
    arousal = batch['arousal'].float().to(device)
    dominance = batch['dominance'].float().to(device)
    actual = torch.stack([valence, arousal, dominance], dim=1)  # [B, 3]

    expected = centroids.to(device)[labels]  # [B, 3], gradient flows here in grad mode
    diff = ((actual - expected) ** 2).sum(dim=1).sqrt()  # [B]

    # Zero-out samples from non-VAD datasets (fake VAD = 0.5/0.5/0.5)
    ds_names = batch.get('dataset', None)
    if ds_names is not None:
        mask = torch.tensor(
            [n in DATASETS_WITH_VAD for n in ds_names],
            dtype=torch.float32, device=device,
        )
        diff = diff * mask

    # Guard
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    return diff


class LearnableCentroids(nn.Module):
    """
    Per-class VAD centroids. Two modes:
      - "ema": buffer updated by running mean of batch per-class means (no gradient)
      - "grad": nn.Parameter updated via optimizer (gradient flows through difficulty)

    Initialized from `expected_vad` dict (label -> [V,A,D]).
    """

    def __init__(self, expected_vad, num_classes=4, mode="ema", momentum=0.9):
        super().__init__()
        assert mode in ("ema", "grad"), f"Unknown mode: {mode}"
        self.mode = mode
        self.momentum = momentum
        self.num_classes = num_classes

        init = torch.zeros(num_classes, 3)
        for c in range(num_classes):
            vad = expected_vad.get(c, [0.5, 0.5, 0.5])
            init[c] = torch.tensor(vad, dtype=torch.float32)

        if mode == "grad":
            self.centroids = nn.Parameter(init)
        else:  # ema
            self.register_buffer("centroids", init)

    def forward(self):
        return self.centroids

    @torch.no_grad()
    def ema_update(self, batch, device):
        """Update centroids from this batch (EMA). Only uses samples from VAD datasets."""
        if self.mode != "ema":
            return
        labels = batch['label'].to(device)
        valence = batch['valence'].float().to(device)
        arousal = batch['arousal'].float().to(device)
        dominance = batch['dominance'].float().to(device)
        actual = torch.stack([valence, arousal, dominance], dim=1)  # [B, 3]

        ds_names = batch.get('dataset', None)
        if ds_names is not None:
            mask = torch.tensor(
                [n in DATASETS_WITH_VAD for n in ds_names],
                dtype=torch.bool, device=device,
            )
        else:
            mask = torch.ones(labels.shape[0], dtype=torch.bool, device=device)

        for c in range(self.num_classes):
            class_mask = (labels == c) & mask
            n = class_mask.sum().item()
            if n == 0:
                continue
            batch_mean = actual[class_mask].mean(dim=0)  # [3]
            self.centroids[c] = self.momentum * self.centroids[c] + (1.0 - self.momentum) * batch_mean


def class_vad_stats(train_data, num_classes: int, consensus_q: float = 0.0,
                    dims=(0, 1, 2), consensus_source: str = 'class'):
    """Per-class VAD mean and inverse covariance from the training split.

    The configured prototypes are hardcoded constants. Measured against the
    data they are close for neutral, happy and sad but off by 0.066 for angry,
    which is 40 percent of that class's own scatter. Fitting them here removes
    that error, and the covariance is what makes a non-Euclidean distance
    possible.

    Args:
        train_data: sequence of sample dicts with valence/arousal/dominance
            and label.
        num_classes: number of classes.
        consensus_source: "class" selects on categorical agreement (the
            fraction of annotators choosing the majority label); "vad" selects
            on dimensional agreement (inverse mean annotator VAD dispersion).
        consensus_q: when above zero, keep only the samples whose
            'annot_consensus' sits at or above this within-class quantile
            before fitting. A prototype is meant to be what the emotion
            sounds like when it is unambiguous, and an utterance annotators
            split over contributes a VAD value that summarises the
            disagreement rather than measuring the emotion. Measured on
            MSP-Podcast, filtering at 0.75 moves the sad and angry centres by
            0.64 and 0.58 within-class standard deviations and widens the
            sad/angry prototype separation by 26.5 percent, which is the
            boundary that collapses on MSP-Improv.

    Returns:
        (means, inv_covs) as numpy arrays of shape [C, 3] and [C, 3, 3].
    """
    dims = tuple(dims)
    k = len(dims)
    means = np.zeros((num_classes, k), dtype=np.float64)
    inv_covs = np.zeros((num_classes, k, k), dtype=np.float64)
    for c in range(num_classes):
        rows = [r for r in train_data if r.get('label') == c]
        if consensus_q > 0.0:
            if consensus_source == 'vad':
                # Dimensional agreement: how tightly the annotators agreed on
                # the VAD numbers for this clip. Distinct from categorical
                # agreement, which asks how many chose the majority LABEL.
                # The two correlate at only r=+0.433 and pick centroids 0.045
                # apart on average, diverging most on happy (0.090).
                scores = np.array(
                    [1.0 / (1.0 + float(np.mean(np.maximum(
                        r.get('annot_vad_std', [0.0, 0.0, 0.0]), 1e-6))))
                     for r in rows], dtype=np.float64)
            else:
                scores = np.array([r.get('annot_consensus', 0.0) for r in rows],
                                  dtype=np.float64)
            if (scores > 0).any():
                thr = np.quantile(scores[scores > 0], consensus_q)
                sel = [r for r, sc in zip(rows, scores) if sc >= thr]
                # Keep enough points that the 3x3 covariance stays estimable.
                if len(sel) >= 50:
                    rows = sel
        pts = np.array([[r.get('valence', 0.0), r.get('arousal', 0.0),
                         r.get('dominance', 0.0)] for r in rows],
                       dtype=np.float64)[:, list(dims)]
        if len(pts) < 4:
            means[c] = 0.0
            inv_covs[c] = np.eye(k)
            continue
        means[c] = pts.mean(axis=0)
        cov = np.atleast_2d(np.cov(pts, rowvar=False))
        # Ridge so a near-degenerate class cannot blow the inverse up.
        inv_covs[c] = np.linalg.inv(cov + 1e-4 * np.eye(k))
    return means, inv_covs


def mahalanobis_distance(vad, labels, means, inv_covs):
    """Per-sample Mahalanobis distance to the sample's own class centre.

    Euclidean distance treats valence, arousal and dominance as equally
    scaled and uncorrelated. Measured per-class standard deviations range
    from 0.09 to 0.15 across the three axes and the axes are correlated, so
    that isotropy assumption is wrong; this is the standard correction and
    the direct answer to the criticism of the Euclidean definition.

    Args:
        vad: [N, 3] array.
        labels: [N] class ids.
        means: [C, 3] class centres.
        inv_covs: [C, 3, 3] inverse covariances.

    Returns:
        [N] distances.
    """
    out = np.zeros(len(labels), dtype=np.float64)
    for c in np.unique(labels):
        m = labels == c
        d = vad[m] - means[int(c)]
        out[m] = np.sqrt(np.maximum(
            np.einsum('ij,jk,ik->i', d, inv_covs[int(c)], d), 0.0))
    return out


def whitening_matrices(inv_covs):
    """Cholesky factors of the inverse covariances.

    Whitening turns the residual into a vector whose norm is the Mahalanobis
    distance, so a whitened residual fixes both problems at once: the metric,
    which Euclidean gets wrong because the per-class VAD ellipsoids have
    condition numbers of 6 to 9, and the collapse to a scalar, which throws
    away the direction the sample is atypical in.

    Args:
        inv_covs: [C, 3, 3] inverse covariances.

    Returns:
        [C, 3, 3] lower-triangular factors L with L L^T = inv_cov.
    """
    out = np.zeros_like(inv_covs)
    k = inv_covs.shape[-1]
    for c in range(inv_covs.shape[0]):
        try:
            out[c] = np.linalg.cholesky(inv_covs[c])
        except np.linalg.LinAlgError:
            # Not positive definite after the ridge; fall back to no whitening
            # so the arm degrades to a plain residual rather than crashing.
            out[c] = np.eye(k)
    return out


def residual_targets(vad, labels, means, whiten=None):
    """Signed displacement of each sample from its own class centre.

    Args:
        vad: [N, 3] array.
        labels: [N] class ids.
        means: [C, 3] class centres.
        whiten: optional [C, 3, 3] Cholesky factors. When given the residual
            is whitened, so its norm equals the Mahalanobis distance.

    Returns:
        [N, 3] residuals.
    """
    out = vad - means[labels]
    if whiten is not None:
        for c in np.unique(labels):
            m = labels == c
            out[m] = out[m] @ whiten[int(c)]
    return out


def symmetric_whitening_matrices(inv_covs):
    """Symmetric inverse square roots of the covariances.

    whitening_matrices returns Cholesky factors, which are triangular. That
    makes the whitened components depend on the order the VAD axes are listed
    in: with a lower-triangular factor the last component is a scaled
    dominance alone while the first mixes all three. The symmetric root
    treats the axes evenly.

    Both satisfy W W^T = inv_cov, so the norm of the whitened residual is the
    Mahalanobis distance either way; only the individual components differ,
    and those are what the auxiliary head regresses.

    Args:
        inv_covs: [C, 3, 3] inverse covariances.

    Returns:
        [C, 3, 3] symmetric matrices W with W W = inv_cov.
    """
    out = np.zeros_like(inv_covs)
    k = inv_covs.shape[-1]
    for c in range(inv_covs.shape[0]):
        try:
            vals, vecs = np.linalg.eigh(inv_covs[c])
            vals = np.maximum(vals, 1e-12)
            out[c] = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        except np.linalg.LinAlgError:
            out[c] = np.eye(k)
    return out


def signed_axis_whitened(vad, labels, means, whiten, neutral_index: int = 0):
    """signed_axis_targets, but computed in the whitened space.

    The original works on raw VAD, which confounds the decomposition with the
    absence of whitening; the unwhitened residual is null (+0.07 on IEMOCAP
    neutral AUC against +1.33 whitened), so an unwhitened signed target would
    most likely be null for that reason rather than because the decomposition
    fails.

    Here the residual and the class axis are both whitened before projecting,
    so the only difference from the plain whitened residual is the change of
    basis: one signed coordinate along the class's own direction away from
    neutral, and the magnitude of what is left.

    Args:
        vad: [N, 3] array.
        labels: [N] class ids.
        means: [C, 3] class centres.
        whiten: [C, 3, 3] per-class whitening matrices.
        neutral_index: which class is neutral.

    Returns:
        [N, 2] targets, column 0 signed along the axis, column 1 off-axis
        magnitude.
    """
    out = np.zeros((len(labels), 2), dtype=np.float64)
    neutral = means[neutral_index]
    for c in np.unique(labels):
        m = labels == c
        w = whiten[int(c)]
        d = (vad[m] - means[c]) @ w
        axis = (means[c] - neutral) @ w
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            # Neutral itself has no direction away from neutral. Fall back to
            # the plain magnitude so the column is still defined.
            out[m, 1] = np.linalg.norm(d, axis=1)
            continue
        axis = axis / norm
        along = d @ axis
        out[m, 0] = along
        out[m, 1] = np.linalg.norm(d - np.outer(along, axis), axis=1)
    return out


def all_class_residuals(vad, labels, means, whiten=None):
    """Residual to every class centre, not only the sample's own.

    A residual from the sample's own centre says where inside its class the
    sample sits, but nothing about its position relative to the other classes.
    That is the information the angry/sad boundary needs: both are low-valence
    and are separated mainly by arousal and dominance, and measurement shows
    the own-class target leaves that confusion untouched and can worsen it
    (true-angry predicted sad rises from 0.228 to 0.265 on MSP-Improv).

    Args:
        vad: [N, 3] array.
        labels: [N] class ids, unused except for shape, kept for symmetry.
        means: [C, 3] class centres.
        whiten: optional [C, 3, 3] Cholesky factors, applied per target class.

    Returns:
        [N, 3 * C] residuals, class-major.
    """
    parts = []
    for c in range(means.shape[0]):
        d = vad - means[c]
        if whiten is not None:
            d = d @ whiten[c]
        parts.append(d)
    return np.concatenate(parts, axis=1)


def signed_axis_targets(vad, labels, means, neutral_index: int = 0):
    """Decompose the residual along the class's own emotional axis.

    Distance-based prototypicality counts every deviation as atypical, so an
    utterance that is MORE angry than the average angry scores as far from the
    prototype as one drifting toward neutral. That is the wrong functional
    form: intensifying along a class's characteristic direction makes a sample
    more canonical, not less. Here the residual is split into

        along: signed projection onto (mu_class - mu_neutral), positive means
            further from neutral in the class's own direction
        off:   magnitude of what is left, deviation the class direction does
            not explain

    so the head is asked for a signed quantity where the sign carries meaning,
    rather than a magnitude that conflates the two cases.

    Args:
        vad: [N, 3] array.
        labels: [N] class ids.
        means: [C, 3] class centres.
        neutral_index: which class is neutral.

    Returns:
        [N, 2] targets, columns (along, off).
    """
    out = np.zeros((len(labels), 2), dtype=np.float64)
    neutral = means[neutral_index]
    for c in np.unique(labels):
        c = int(c)
        m = labels == c
        axis = means[c] - neutral
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            # Neutral itself has no direction away from neutral; use raw
            # distance so the column is still defined.
            out[m, 0] = 0.0
            out[m, 1] = np.linalg.norm(vad[m] - means[c], axis=1)
            continue
        axis = axis / norm
        d = vad[m] - means[c]
        along = d @ axis
        off = np.linalg.norm(d - np.outer(along, axis), axis=1)
        out[m, 0] = along
        out[m, 1] = off
    return out


def pooled_whitening(train_data, means: np.ndarray, num_classes: int,
                     consensus_q: float = 0.0) -> np.ndarray:
    """One whitening matrix shared by every class, from the pooled covariance.

    Whitening each class by its OWN covariance puts each class's residual in a
    different coordinate system: measured on MSP-Podcast the class whitening
    matrices differ by up to 49 percent (happy against angry), with condition
    numbers of 5.7 to 8.6. A regression head then emits three numbers whose
    meaning depends on the sample's class, which is exactly what is unknown at
    inference, so the target is defined in a latent sample-dependent basis.

    Pooling the within-class scatter keeps the class-relative centering, which
    is the part that carries information, while putting every residual in one
    consistent space. This is the whitening used by linear discriminant
    analysis for the same reason.

    Args:
        train_data: sequence of sample dicts with valence/arousal/dominance
            and label.
        means: [C, 3] class centres.
        num_classes: number of classes.
        consensus_q: optional within-class agreement quantile filter, matching
            class_vad_stats, so the pooled scatter is estimated from the same
            rows that defined the centres.

    Returns:
        [C, 3, 3] Cholesky factors, identical for every class.
    """
    resid = []
    for c in range(num_classes):
        rows = [r for r in train_data if r.get('label') == c]
        if consensus_q > 0.0:
            scores = np.array([r.get('annot_consensus', 0.0) for r in rows],
                              dtype=np.float64)
            if (scores > 0).any():
                thr = np.quantile(scores[scores > 0], consensus_q)
                sel = [r for r, sc in zip(rows, scores) if sc >= thr]
                if len(sel) >= 50:
                    rows = sel
        if not rows:
            continue
        pts = np.array([[r.get('valence', 0.0), r.get('arousal', 0.0),
                         r.get('dominance', 0.0)] for r in rows],
                       dtype=np.float64)
        resid.append(pts - means[c])
    if not resid:
        return np.tile(np.eye(3), (num_classes, 1, 1))
    allr = np.concatenate(resid, axis=0)
    cov = np.cov(allr, rowvar=False) + 1e-4 * np.eye(3)
    try:
        w = np.linalg.cholesky(np.linalg.inv(cov))
    except np.linalg.LinAlgError:
        w = np.eye(3)
    return np.tile(w, (num_classes, 1, 1))

def marginal_whitening(train_data, dims=(0, 1, 2)):
    """Global mean and whitening from the MARGINAL VAD covariance.

    Deliberately ignores the class labels. Where class_vad_stats fits one
    ellipsoid per class, this fits a single ellipsoid to the whole training
    pool, so the resulting transform carries no class-relative information at
    all.

    This exists as a control. The map v -> W (v - mu) is affine and
    invertible, so its output carries exactly the same information as raw
    VAD; only the conditioning of the regression target changes. An arm using
    it therefore separates "rescaling the target helps for optimisation
    reasons" from "rescaling helps because it expresses position relative to
    a class prototype". Any difference from the raw VAD arm cannot be
    informational.

    Args:
        train_data: sequence of sample dicts with valence/arousal/dominance.
        dims: which VAD axes to keep.

    Returns:
        (mean [k], whitening [k, k]) with whiten = cholesky(inv(cov)), applied
        on the right as `(v - mean) @ whiten` to match residual_targets.
    """
    dims = tuple(dims)
    k = len(dims)
    pts = np.array([[r.get('valence', 0.0), r.get('arousal', 0.0),
                     r.get('dominance', 0.0)] for r in train_data],
                   dtype=np.float64)[:, list(dims)]
    if len(pts) < 4:
        return np.zeros(k), np.eye(k)
    mean = pts.mean(axis=0)
    cov = np.atleast_2d(np.cov(pts, rowvar=False)) + 1e-4 * np.eye(k)
    try:
        whiten = np.linalg.cholesky(np.linalg.inv(cov))
    except np.linalg.LinAlgError:
        whiten = np.eye(k)
    return mean, whiten

def diagonal_whitening(train_data, means: np.ndarray, num_classes: int,
                       dims=(0, 1, 2), ridge: float = 1e-4) -> np.ndarray:
    """Per-class diagonal scaling: divide each axis by its within-class std.

    The scale-matched control. Full whitening does two things at once, it puts
    every coordinate on unit variance AND it decorrelates the axes by rotating
    the class ellipsoid onto its principal directions. The unwhitened residual
    does neither, so comparing it against the whitened residual cannot say
    which of the two carries the effect.

    This does only the first. The target has approximately unit variance per
    coordinate, like the whitened residual, but the class ellipsoid keeps its
    tilt: off-diagonal correlation survives untouched. Whitened against
    scale-matched therefore isolates the decorrelation step with scale held
    fixed, which is the contrast that answers whether the gain is only target
    rescaling.

    Args:
        train_data: sequence of sample dicts with valence/arousal/dominance
            and label.
        means: [C, k] class centres.
        num_classes: number of classes.
        dims: which VAD axes to keep.
        ridge: added to the variance before the square root, matching the
            ridge class_vad_stats applies to the full covariance.

    Returns:
        [C, k, k] diagonal matrices with 1 / sigma on the diagonal, applied on
        the right as `(v - mean) @ W` to match residual_targets.
    """
    dims = tuple(dims)
    k = len(dims)
    out = np.zeros((num_classes, k, k), dtype=np.float64)
    for c in range(num_classes):
        rows = [r for r in train_data if r.get('label') == c]
        if len(rows) < 4:
            out[c] = np.eye(k)
            continue
        pts = np.array([[r.get('valence', 0.0), r.get('arousal', 0.0),
                         r.get('dominance', 0.0)] for r in rows],
                       dtype=np.float64)[:, list(dims)]
        var = pts.var(axis=0, ddof=1) + ridge
        out[c] = np.diag(1.0 / np.sqrt(var))
    return out


def vad_trust_weights(vad: np.ndarray, labels: np.ndarray,
                      means: np.ndarray, inv_covs: np.ndarray) -> np.ndarray:
    """Per-sample confidence that VAD alone accounts for the given label.

    Returns the Gaussian class posterior evaluated at each sample's own label,
    under uniform class priors. A row whose VAD sits squarely inside its own
    class ellipsoid scores near 1; a row whose VAD looks like a different
    class scores near 0.

    The motivation is measured rather than assumed. Fitting these Gaussians on
    MSP-Podcast and transferring them to IEMOCAP leaves 34.8 percent of that
    corpus misclassified from VAD alone, and across sixteen arm and control
    pairs twelve arms LOSE accuracy on exactly those rows while gaining on the
    rest. The loud-sad and muted-happy utterances live there, so regressing
    VAD on them teaches the encoder something the classifier has to undo.
    Using this as an auxiliary loss weight confines the VAD target to the rows
    where VAD and the label agree.

    Uniform priors are deliberate. Fitted source priors would let a row profit
    from source-to-target prior shift, which is not the quantity wanted here.

    Args:
        vad: [N, D] VAD coordinates.
        labels: [N] class labels indexing means and inv_covs.
        means: [C, D] per-class means.
        inv_covs: [C, D, D] per-class inverse covariances.

    Returns:
        [N] posterior weights in (0, 1], NOT normalised. Callers that feed an
        auxiliary loss should rescale to mean one so the loss magnitude and
        proto_predictor_weight stay comparable across arms.
    """
    vad = np.asarray(vad, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    num_classes = len(means)
    scores = np.empty((len(vad), num_classes), dtype=np.float64)
    for c in range(num_classes):
        delta = vad - means[c]
        quad = np.einsum('ni,ij,nj->n', delta, inv_covs[c], delta)
        sign, logdet = np.linalg.slogdet(inv_covs[c])
        # logdet of the inverse covariance, so its sign contributes positively.
        scores[:, c] = -0.5 * quad + 0.5 * logdet
    scores -= scores.max(axis=1, keepdims=True)
    post = np.exp(scores)
    post /= post.sum(axis=1, keepdims=True)
    return post[np.arange(len(vad)), labels]


def vad_explainable(vad: np.ndarray, labels: np.ndarray, means: np.ndarray,
                    inv_covs: np.ndarray) -> np.ndarray:
    """Whether VAD alone accounts for each row's label.

    A row is explainable when the Gaussian class model, fitted on the training
    VAD and evaluated under uniform priors, assigns it to its own class. A row
    is unexplainable when some RIVAL class is closer, which is the competitive
    condition rather than mere distance from the row's own centre.

    That distinction is measured, not assumed. On IEMOCAP the toward-rival
    component of the residual predicts this at AUC 0.931 while the residual
    NORM reaches only 0.654, and the orthogonal component is exactly chance at
    0.500. Distance from the class centre is therefore a poor proxy: half of
    the most displaced quintile is still explainable, because those rows
    drifted in directions no rival occupies.

    Uniform priors are deliberate. Fitted source priors would let a row profit
    from source-to-target prior shift, which is not the quantity wanted here.

    Args:
        vad: [N, D] VAD coordinates.
        labels: [N] class labels indexing means and inv_covs.
        means: [C, D] per-class means.
        inv_covs: [C, D, D] per-class inverse covariances.

    Returns:
        [N] float array, 1.0 where VAD accounts for the label and 0.0 where a
        rival class is closer. Float rather than bool so it can be used
        directly as a BCE target.
    """
    vad = np.asarray(vad, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    scores = np.empty((len(vad), len(means)), dtype=np.float64)
    for c in range(len(means)):
        delta = vad - means[c]
        quad = np.einsum('ni,ij,nj->n', delta, inv_covs[c], delta)
        _, logdet = np.linalg.slogdet(inv_covs[c])
        scores[:, c] = -0.5 * quad + 0.5 * logdet
    return (scores.argmax(axis=1) == labels).astype(np.float64)
