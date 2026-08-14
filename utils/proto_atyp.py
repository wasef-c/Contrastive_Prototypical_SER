#!/usr/bin/env python3
"""
Prototypical / atypical sub-label helpers.

At train init we widen the classification head from num_classes to 2 * num_classes.
Each training sample is assigned a sub-label

    sub = 2 * class + is_atypical

where is_atypical in {0, 1} is derived from the sample's VAD Euclidean distance
to its class centroid. The centroid source is picked per experiment (theoretical
from config.expected_vad, or empirical class means from the train set).

Two split criteria are supported:
    per_class_median: each class is split 50/50 by its within-class median
        distance. Sub-label distribution stays balanced across atyp/proto.
    global_median: one median across all VAD-annotated training samples.
        Classes with wider VAD spread contribute more atypical samples.

At eval time we do not compute sub-labels. The 2C logits are collapsed to C
class probabilities via softmax pair sum, so test corpora without VAD still
work.

Samples from datasets without real VAD annotations (e.g. CMUMOSEI, SAMSEMO)
have fake 0.5/0.5/0.5 VAD in this codebase, so their distance would be
misleading. During center estimation and threshold estimation we skip them.
During batch-time sub-label assignment we default them to is_atypical=0
(prototype). In the current MSPI-only training setup this never fires; it is
a safety net for future multi-corpus runs.
"""

import hashlib
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.prototypicality import DATASETS_WITH_VAD


def build_class_centers(
    source: str,
    expected_vad: dict,
    train_data: Sequence[dict],
    num_classes: int,
) -> np.ndarray:
    """Build a [num_classes, 3] array of VAD class centers.

    Args:
        source: "theoretical" (use expected_vad from config) or "class_means"
            (empirical per-class VAD means from VAD-annotated train samples).
        expected_vad: dict mapping class index to [V, A, D]. Used as-is for
            "theoretical" and as a fallback for classes with no samples under
            "class_means".
        train_data: list of dataset item dicts (train_dataset.data).
        num_classes: number of primary classes (typically 4).

    Returns:
        [num_classes, 3] float32 numpy array.

    Raises:
        ValueError: if source is not recognized.
    """
    if source == "theoretical":
        centers = np.zeros((num_classes, 3), dtype=np.float32)
        for c in range(num_classes):
            vad = expected_vad.get(c, [0.5, 0.5, 0.5])
            centers[c] = np.array(vad, dtype=np.float32)
        return centers

    if source == "class_means":
        sums = np.zeros((num_classes, 3), dtype=np.float64)
        counts = np.zeros(num_classes, dtype=np.int64)
        for item in train_data:
            if item.get("dataset") not in DATASETS_WITH_VAD:
                continue
            c = int(item["label"])
            if c < 0 or c >= num_classes:
                continue
            sums[c, 0] += float(item["valence"])
            sums[c, 1] += float(item["arousal"])
            sums[c, 2] += float(item["dominance"])
            counts[c] += 1
        centers = np.zeros((num_classes, 3), dtype=np.float32)
        for c in range(num_classes):
            if counts[c] > 0:
                centers[c] = (sums[c] / counts[c]).astype(np.float32)
            else:
                vad = expected_vad.get(c, [0.5, 0.5, 0.5])
                centers[c] = np.array(vad, dtype=np.float32)
        return centers

    raise ValueError(f"Unknown proto_atyp_center_source: {source}")


def _per_sample_vad_distance(
    train_data: Sequence[dict],
    centers: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute per-sample Euclidean distance to the sample's class centroid.

    Samples from datasets without real VAD get NaN so callers can skip them.

    Args:
        train_data: list of dataset item dicts.
        centers: [num_classes, 3] centers.
        num_classes: number of primary classes.

    Returns:
        [N] float32 array of distances; NaN for non-VAD samples.
    """
    n = len(train_data)
    distances = np.full(n, np.nan, dtype=np.float32)
    for i, item in enumerate(train_data):
        if item.get("dataset") not in DATASETS_WITH_VAD:
            continue
        c = int(item["label"])
        if c < 0 or c >= num_classes:
            continue
        v = float(item["valence"])
        a = float(item["arousal"])
        d = float(item["dominance"])
        diff = np.array([v, a, d], dtype=np.float32) - centers[c]
        distances[i] = float(np.sqrt(np.sum(diff * diff)))
    return distances


def compute_split_thresholds(
    train_data: Sequence[dict],
    centers: np.ndarray,
    num_classes: int,
    criterion: str,
) -> np.ndarray:
    """Return threshold(s) used to decide is_atypical from a VAD distance.

    Args:
        train_data: list of dataset item dicts (train_dataset.data).
        centers: [num_classes, 3] centers (matches the source used at split time).
        num_classes: number of primary classes.
        criterion: "per_class_median" or "global_median".

    Returns:
        [num_classes] float32 array. For global_median the same value is
        broadcast across all classes, so downstream code always looks up
        thresholds[label].

    Raises:
        ValueError: if criterion is not recognized.
    """
    distances = _per_sample_vad_distance(train_data, centers, num_classes)

    if criterion == "per_class_median":
        thresholds = np.zeros(num_classes, dtype=np.float32)
        labels = np.array(
            [int(item["label"]) for item in train_data], dtype=np.int64
        )
        for c in range(num_classes):
            class_mask = (labels == c) & ~np.isnan(distances)
            if class_mask.any():
                thresholds[c] = float(np.median(distances[class_mask]))
            else:
                thresholds[c] = 0.0
        return thresholds

    if criterion == "global_median":
        valid = distances[~np.isnan(distances)]
        if valid.size > 0:
            global_thr = float(np.median(valid))
        else:
            global_thr = 0.0
        return np.full(num_classes, global_thr, dtype=np.float32)

    raise ValueError(f"Unknown proto_atyp_split_criterion: {criterion}")


def batch_sub_labels(
    batch: dict,
    centers: torch.Tensor,
    thresholds: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute sub-labels for a training batch.

    sub = 2 * class + is_atypical, where is_atypical = int(distance > threshold[class]).
    Samples from non-VAD datasets are forced to is_atypical=0.

    Args:
        batch: collated batch dict with 'label', 'valence', 'arousal',
            'dominance', 'dataset' keys.
        centers: [num_classes, 3] float tensor of class centers on `device`.
        thresholds: [num_classes] float tensor of per-class thresholds on `device`.
        num_classes: number of primary classes.
        device: torch device for output tensors.

    Returns:
        [B] LongTensor of sub-labels on `device`.
    """
    labels = batch["label"].to(device).long()
    valence = batch["valence"].to(device).float()
    arousal = batch["arousal"].to(device).float()
    dominance = batch["dominance"].to(device).float()
    actual = torch.stack([valence, arousal, dominance], dim=1)  # [B, 3]

    expected = centers[labels]  # [B, 3]
    distance = torch.sqrt(((actual - expected) ** 2).sum(dim=1))  # [B]

    class_thresh = thresholds[labels]  # [B]
    is_atypical = (distance > class_thresh).long()  # [B]

    ds_names = batch.get("dataset", None)
    if ds_names is not None:
        vad_mask = torch.tensor(
            [n in DATASETS_WITH_VAD for n in ds_names],
            dtype=torch.long, device=device,
        )
        is_atypical = is_atypical * vad_mask

    return labels * 2 + is_atypical


def batch_random_sub_labels(
    batch: dict,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Placebo sub-labels: the split carries no information about the sample.

    Assigns is_atypical by hashing the transcript, so a given utterance
    always lands on the same side across epochs while the split itself is
    unrelated to anything about the emotion. The head is still widened to
    2 * num_classes and the sub-label CE still runs, so this arm isolates
    how much of any gain comes from the split's content rather than from
    head widening plus a harder objective.

    This control exists because the aux VAD ablation found that a permuted
    label assignment matched the fitted one. A prototypicality split needs
    the same test before its result can be believed.

    Args:
        batch: collated batch dict with 'label' and 'transcript'.
        num_classes: number of primary classes.
        seed: offsets the hash so different seeds give different splits.
        device: torch device for output tensors.

    Returns:
        [B] LongTensor of sub-labels on `device`.
    """
    labels = batch["label"].to(device).long()
    texts = batch.get("transcript", None)
    if texts is None:
        return labels * 2

    flags = []
    for text in texts:
        digest = hashlib.md5(f"{seed}:{text}".encode("utf-8")).digest()
        flags.append(digest[0] & 1)
    is_atypical = torch.tensor(flags, dtype=torch.long, device=device)
    return labels * 2 + is_atypical


def collapse_sub_logits_to_class_probs(
    sub_logits: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Collapse [B, 2C] sub-logits into [B, C] class probabilities.

    Uses softmax over the 2C sub-classes, then sums the (proto, atyp) pair
    per class. Suitable for argmax at eval time.

    Args:
        sub_logits: [B, 2 * num_classes] raw logits.
        num_classes: number of primary classes.

    Returns:
        [B, num_classes] float tensor of class probabilities.
    """
    probs = F.softmax(sub_logits, dim=-1)  # [B, 2C]
    reshaped = probs.reshape(probs.shape[0], num_classes, 2)  # [B, C, 2]
    return reshaped.sum(dim=-1)  # [B, C]
