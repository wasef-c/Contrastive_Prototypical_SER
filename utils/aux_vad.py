#!/usr/bin/env python3
"""
Auxiliary VAD-cluster multitask helper.

Idea: fit k-means over the VAD points of the training set (across all
VAD-annotated corpora combined), snapshot the centroids, and add a light
classification head on top of the fused embedding that predicts each sample's
nearest cluster ID. The classifier gets the emotion label; the aux head gets
the VAD cluster ID. Total loss is a weighted sum. At eval the aux head is
unused, so test corpora do not need VAD.

Clustering across all training corpora (not per corpus) gives a shared target
that any corpus with VAD contributed to. In a cross-corpus training setup
this becomes a domain-shared signal, which is the point.
"""

from typing import Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from utils.prototypicality import DATASETS_WITH_VAD


def _class_prototype_seeds(
    points_np: np.ndarray,
    labels_np: np.ndarray,
    num_classes: int,
    atypical_percentile: float = 75.0,
) -> np.ndarray:
    """Build (proto, atypical) seed pairs per class for KMeans init.

    For each class c:
      seed[2c]   = class VAD mean (prototype seed)
      seed[2c+1] = the sample whose distance to the class mean sits at the
                   `atypical_percentile` (atypical exemplar seed)

    Classes with no samples get their two seeds set to the global mean so
    KMeans can still start. Duplicate seeds (e.g., a class with a single
    sample) are perturbed by a small deterministic offset so KMeans does
    not degenerate.

    Args:
        points_np: [M, 3] VAD points of all VAD-annotated training samples.
        labels_np: [M] int class labels aligned with points_np.
        num_classes: number of primary classes.
        atypical_percentile: percentile of within-class distance used to pick
            the atypical exemplar. 75 is a reasonable "far but not outlier"
            pick; 100 would be the extreme outlier.

    Returns:
        [2 * num_classes, 3] float32 seeds.
    """
    seeds = np.zeros((2 * num_classes, 3), dtype=np.float32)
    global_mean = points_np.mean(axis=0)

    for c in range(num_classes):
        class_mask = labels_np == c
        if not class_mask.any():
            seeds[2 * c] = global_mean
            seeds[2 * c + 1] = global_mean + np.array([1e-3, 0.0, 0.0], dtype=np.float32)
            continue
        class_points = points_np[class_mask]
        class_mean = class_points.mean(axis=0).astype(np.float32)
        diffs = class_points - class_mean
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        if dists.size == 1:
            atyp = class_mean + np.array([1e-3, 0.0, 0.0], dtype=np.float32)
        else:
            thr = float(np.percentile(dists, atypical_percentile))
            candidates = np.abs(dists - thr)
            atyp_local = int(np.argmin(candidates))
            atyp = class_points[atyp_local].astype(np.float32)
            if float(np.linalg.norm(atyp - class_mean)) < 1e-6:
                atyp = class_mean + np.array([1e-3, 0.0, 0.0], dtype=np.float32)
        seeds[2 * c] = class_mean
        seeds[2 * c + 1] = atyp
    return seeds


def build_vad_centroids(
    train_data: Sequence[dict],
    k: int,
    seed: int = 42,
    init: str = "random",
    num_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit k-means over VAD points from VAD-annotated training samples.

    Args:
        train_data: list of dataset item dicts (train_dataset.data indexed by
            the train_indices split).
        k: number of clusters. When init == "class_prototypes" this is
            expected to equal 2 * num_classes; a mismatch is a warning and
            the code falls back to random init.
        seed: random state for KMeans reproducibility.
        init: "random" (sklearn kmeans++), "class_prototypes" (per-class
            proto + atyp exemplar seeds; k-means refines), or
            "random_partition" (k centroids sampled uniformly inside the
            VAD bounding box and never fitted; an arbitrary Voronoi
            partition used as a placebo control).
        num_classes: number of primary classes; used only for
            "class_prototypes" init to size the seed array.

    Returns:
        centroids: [k, 3] float32 numpy array.
        assignments: [N] int64 array of cluster IDs for each sample in
            train_data. Samples without real VAD get -1 so callers can skip
            them at loss time.

    Notes:
        Uses sklearn KMeans; falls back to all-first-centroid centroids if the
        number of usable VAD samples is < k (very degenerate).
    """
    points = []
    used_indices = []
    used_labels = []
    for i, item in enumerate(train_data):
        if item.get("dataset") not in DATASETS_WITH_VAD:
            continue
        v = float(item["valence"])
        a = float(item["arousal"])
        d = float(item["dominance"])
        points.append([v, a, d])
        used_indices.append(i)
        used_labels.append(int(item["label"]))

    n = len(train_data)
    assignments = np.full(n, -1, dtype=np.int64)

    if len(points) == 0:
        centroids = np.zeros((k, 3), dtype=np.float32)
        return centroids, assignments

    points_np = np.asarray(points, dtype=np.float32)
    labels_np = np.asarray(used_labels, dtype=np.int64)

    if len(points) < k:
        centroids = np.tile(points_np[0:1], (k, 1)).astype(np.float32)
        for j, idx in enumerate(used_indices):
            assignments[idx] = j % k
        return centroids, assignments

    if init == "random_partition":
        # Placebo partition: uniform random centroids in the data bounding
        # box, no fitting. Still a deterministic function of VAD (nearest
        # centroid) but the geometry is arbitrary.
        rng = np.random.RandomState(seed)
        lo = points_np.min(axis=0)
        hi = points_np.max(axis=0)
        centroids = rng.uniform(lo, hi, size=(k, 3)).astype(np.float32)
        diff = points_np[:, None, :] - centroids[None, :, :]
        labels = (diff * diff).sum(axis=-1).argmin(axis=-1)
        for local_i, idx in enumerate(used_indices):
            assignments[idx] = int(labels[local_i])
        return centroids, assignments

    if init == "class_prototypes":
        if k != 2 * num_classes:
            print(f"  WARNING: aux_vad_cluster_init=class_prototypes expects "
                  f"k = 2 * num_classes ({2 * num_classes}) but got k={k}. "
                  f"Falling back to random init.")
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        else:
            seeds = _class_prototype_seeds(points_np, labels_np, num_classes)
            km = KMeans(n_clusters=k, init=seeds, n_init=1, random_state=seed)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)

    labels = km.fit_predict(points_np)
    centroids = km.cluster_centers_.astype(np.float32)

    for local_i, idx in enumerate(used_indices):
        assignments[idx] = int(labels[local_i])

    return centroids, assignments


def build_per_class_vad_centroids(
    train_data: Sequence[dict],
    clusters_per_class: int,
    num_classes: int,
    seed: int = 42,
    init: str = "random",
) -> np.ndarray:
    """Fit a separate k-means within each emotion class's VAD points.

    Unlike build_vad_centroids, which pools all classes and lets cells
    straddle class boundaries, this carves subtypes strictly inside each
    class: happy-1..happy-n, angry-1..angry-n, and so on. The aux label
    space is num_classes * clusters_per_class, with sample label

        aux_id = class_label * clusters_per_class + local_cluster

    so the aux task is "which subtype of which class", a strict refinement
    of the primary label.

    Args:
        train_data: list of dataset item dicts.
        clusters_per_class: number of subtypes to fit inside each class.
        num_classes: number of primary classes.
        seed: random state for KMeans reproducibility.
        init: "random" (sklearn kmeans++) or "random_partition" (unfitted
            uniform centroids inside the class's VAD bounding box; the
            placebo control at per-class scope).

    Returns:
        centroids: [num_classes, clusters_per_class, 3] float32 array.
            Classes with too few VAD samples get their centroids filled
            with that class's mean (or the global mean if empty), which
            makes their subtype assignment arbitrary but harmless.
    """
    by_class = {c: [] for c in range(num_classes)}
    for item in train_data:
        if item.get("dataset") not in DATASETS_WITH_VAD:
            continue
        c = int(item["label"])
        if c < 0 or c >= num_classes:
            continue
        by_class[c].append([
            float(item["valence"]),
            float(item["arousal"]),
            float(item["dominance"]),
        ])

    all_points = [p for pts in by_class.values() for p in pts]
    global_mean = (np.asarray(all_points, dtype=np.float32).mean(axis=0)
                   if all_points else np.zeros(3, dtype=np.float32))

    centroids = np.zeros(
        (num_classes, clusters_per_class, 3), dtype=np.float32,
    )
    rng = np.random.RandomState(seed)

    for c in range(num_classes):
        pts = np.asarray(by_class[c], dtype=np.float32)
        if pts.shape[0] == 0:
            centroids[c] = np.tile(global_mean, (clusters_per_class, 1))
            continue
        if pts.shape[0] < clusters_per_class:
            centroids[c] = np.tile(pts.mean(axis=0), (clusters_per_class, 1))
            continue

        if init == "random_partition":
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            centroids[c] = rng.uniform(
                lo, hi, size=(clusters_per_class, 3),
            ).astype(np.float32)
        else:
            km = KMeans(
                n_clusters=clusters_per_class,
                random_state=seed + c,
                n_init=10,
            )
            km.fit(pts)
            centroids[c] = km.cluster_centers_.astype(np.float32)

    return centroids


def batch_per_class_cluster_ids_and_mask(
    batch: dict,
    centroids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (per-class subtype IDs, mask) for a batch.

    Each sample is assigned to the nearest centroid *within its own class*,
    then mapped to a flat label class_label * clusters_per_class + local.
    The primary label is used only to pick which class's centroids apply;
    the aux head still has to infer the subtype from the embedding.

    Args:
        batch: collated batch dict with 'valence', 'arousal', 'dominance',
            'label', 'dataset' keys.
        centroids: [num_classes, clusters_per_class, 3] float tensor on
            `device`.
        device: torch device for outputs.

    Returns:
        cluster_ids: [B] LongTensor of flat subtype IDs.
        mask: [B] float tensor, 1.0 for samples with real VAD.
    """
    valence = batch["valence"].to(device).float()
    arousal = batch["arousal"].to(device).float()
    dominance = batch["dominance"].to(device).float()
    points = torch.stack([valence, arousal, dominance], dim=1)  # [B, 3]

    labels = batch["label"].to(device).long()  # [B]
    clusters_per_class = centroids.shape[1]

    # Gather each sample's own class centroids: [B, clusters_per_class, 3]
    own = centroids[labels]
    diff = points.unsqueeze(1) - own  # [B, kpc, 3]
    local = (diff * diff).sum(dim=-1).argmin(dim=-1).long()  # [B]
    cluster_ids = labels * clusters_per_class + local

    mask = _batch_vad_mask(batch, cluster_ids.shape[0], device)
    return cluster_ids, mask


def batch_cluster_ids_and_mask(
    batch: dict,
    centroids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cluster_ids, mask) for a batch, computed from VAD.

    Args:
        batch: collated batch dict with 'valence', 'arousal', 'dominance',
            'dataset' keys.
        centroids: [k, 3] float tensor on `device`.
        device: torch device for outputs.

    Returns:
        cluster_ids: [B] LongTensor of argmin distance to centroids. Samples
            without real VAD get 0 (masked out anyway).
        mask: [B] float tensor, 1.0 for samples with real VAD (usable in aux
            loss) and 0.0 for the fake-VAD ones.
    """
    valence = batch["valence"].to(device).float()
    arousal = batch["arousal"].to(device).float()
    dominance = batch["dominance"].to(device).float()
    points = torch.stack([valence, arousal, dominance], dim=1)  # [B, 3]

    # Squared distance to each centroid, argmin
    diff = points.unsqueeze(1) - centroids.unsqueeze(0)  # [B, k, 3]
    dist_sq = (diff * diff).sum(dim=-1)  # [B, k]
    cluster_ids = torch.argmin(dist_sq, dim=-1).long()  # [B]

    ds_names = batch.get("dataset", None)
    if ds_names is not None:
        mask = torch.tensor(
            [n in DATASETS_WITH_VAD for n in ds_names],
            dtype=torch.float32, device=device,
        )
    else:
        mask = torch.ones(points.shape[0], dtype=torch.float32, device=device)

    return cluster_ids, mask


def _batch_vad_mask(batch: dict, n: int, device: torch.device) -> torch.Tensor:
    """Return a [n] float mask, 1.0 for samples from VAD-annotated corpora."""
    ds_names = batch.get("dataset", None)
    if ds_names is None:
        return torch.ones(n, dtype=torch.float32, device=device)
    return torch.tensor(
        [name in DATASETS_WITH_VAD for name in ds_names],
        dtype=torch.float32, device=device,
    )


def batch_vad_targets_and_mask(
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ([B, 3] VAD regression targets, [B] mask) for the aux head.

    Args:
        batch: collated batch dict with 'valence', 'arousal', 'dominance',
            'dataset' keys.
        device: torch device for outputs.

    Returns:
        targets: [B, 3] float tensor of (valence, arousal, dominance).
        mask: [B] float tensor, 1.0 for samples with real VAD.
    """
    valence = batch["valence"].to(device).float()
    arousal = batch["arousal"].to(device).float()
    dominance = batch["dominance"].to(device).float()
    targets = torch.stack([valence, arousal, dominance], dim=1)
    mask = _batch_vad_mask(batch, targets.shape[0], device)
    return targets, mask


def batch_scrambled_ids_and_mask(
    batch: dict,
    k: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pseudo-random cluster IDs, mask) for the permuted control.

    Each sample's label is a deterministic hash of its VAD values, so the
    same utterance gets the same label every epoch, but nearby VAD points
    get unrelated labels. The aux task keeps its shape and difficulty while
    carrying no usable VAD structure; fitting it requires per-sample
    memorization. This is the pure-regularization control arm.

    Args:
        batch: collated batch dict with 'valence', 'arousal', 'dominance',
            'dataset' keys.
        k: number of pseudo-cluster labels.
        seed: offsets the hash so different seeds give different labelings.
        device: torch device for outputs.

    Returns:
        cluster_ids: [B] LongTensor of pseudo-random labels in [0, k).
        mask: [B] float tensor, 1.0 for samples with real VAD.
    """
    valence = batch["valence"].to(device).float()
    arousal = batch["arousal"].to(device).float()
    dominance = batch["dominance"].to(device).float()

    # Sin-hash: high-frequency mixing of the three coordinates gives a
    # spatially white value in [0, 1), then bucket into k labels.
    h = torch.sin(
        valence * 12.9898 + arousal * 78.233 + dominance * 37.719
        + float(seed) * 0.618
    ) * 43758.5453
    frac = h - torch.floor(h)
    cluster_ids = (frac * k).long().clamp_(0, k - 1)

    mask = _batch_vad_mask(batch, cluster_ids.shape[0], device)
    return cluster_ids, mask
