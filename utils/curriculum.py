#!/usr/bin/env python3
"""
Curriculum learning helpers.

Ports the pacing and subset-selection logic from Emotion2Vec_Text/functions.py
into a form that operates on an integer index list (a Subset of the training
dataset). The caller precomputes:
  - train_indices  : list of ints into train_dataset.data
  - train_labels   : list of ints, one per index
  - train_difficulties : np.ndarray of floats, one per index (lower = easier)

Each epoch during warm-up, create_curriculum_subset returns the subset of
train_indices to train on for that epoch. After curriculum_epochs epochs the
full train_indices set is returned.

Design notes:
  - Difficulty here means VAD Euclidean distance to the class centroid (see
    utils/prototypicality.calculate_difficulty). Lower values are "closer to
    prototype" and considered easier under the curriculum framing used in the
    Emotion2Vec_Text repo.
  - The "difficulty" curriculum sorts by ascending difficulty (easy first).
  - The "inverse_difficulty" curriculum sorts by descending difficulty
    (hard first). It exists as an ablation to test whether easy-first actually
    matters, or whether just seeing fewer samples early is what helps.
  - "class_balance" starts with class 0 (neutral) and grows the target class
    set as the pacing fraction increases.
  - "preset_order" assumes train_indices are already ordered by curriculum
    priority and just takes the head.
"""

import math
import random
from typing import Callable, List, Optional, Sequence

import numpy as np


def get_curriculum_pacing_function(pacing_type: str) -> Callable[[int, int], float]:
    """Return a pacing function fraction(epoch, total_epochs) in [0, 1].

    Args:
        pacing_type: "linear", "sqrt", "log", or anything else for a no-op
            (always returns 1.0).

    Returns:
        Callable that maps (current_epoch_index, total_curriculum_epochs) to
        the fraction of training data to include this epoch. Capped at 1.0
        (the original repo capped at 2.0 which was a bug for this use).
    """
    if pacing_type == "linear":
        return lambda epoch, total_epochs: min(1.0, (epoch + 1) / max(1, total_epochs))
    if pacing_type == "sqrt":
        return lambda epoch, total_epochs: min(
            1.0, math.sqrt((epoch + 1) / max(1, total_epochs))
        )
    if pacing_type == "log":
        return lambda epoch, total_epochs: min(
            1.0, math.log(epoch + 2) / math.log(max(2, total_epochs) + 1)
        )
    return lambda epoch, total_epochs: 1.0


def create_curriculum_subset(
    train_indices: Sequence[int],
    train_labels: Sequence[int],
    train_difficulties: np.ndarray,
    epoch: int,
    total_curriculum_epochs: int,
    pacing_function: Callable[[int, int], float],
    curriculum_type: str = "difficulty",
    class_order: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Pick the subset of train_indices to train on for this epoch.

    Args:
        train_indices: absolute indices into train_dataset.data, length N.
        train_labels: integer class label for each index in train_indices.
        train_difficulties: per-index difficulty (lower = easier). Length N.
        epoch: current epoch index (0-based). If >= total_curriculum_epochs,
            the full train_indices list is returned unchanged.
        total_curriculum_epochs: number of warm-up epochs.
        pacing_function: fraction schedule from get_curriculum_pacing_function.
        curriculum_type: one of "difficulty", "inverse_difficulty",
            "class_balance", "random", "none", "preset_order".
        class_order: For "class_balance" only, the order in which classes are
            introduced. Defaults to [0, 1, 2, 3].
        rng: Optional random.Random for reproducibility. Falls back to the
            module-level random state if None.

    Returns:
        List of absolute indices into train_dataset.data (a subset of
        train_indices, in the order they should be fed to the DataLoader; the
        caller is expected to wrap them in Subset and set shuffle=True or
        pass a sampler).

    Raises:
        ValueError: if curriculum_type is unknown.
    """
    if rng is None:
        rng = random

    n = len(train_indices)
    if n == 0:
        return []

    if epoch >= total_curriculum_epochs or curriculum_type == "none":
        return list(train_indices)

    fraction = pacing_function(epoch, total_curriculum_epochs)
    num_samples = max(1, int(n * fraction))

    if curriculum_type == "random":
        pool = list(range(n))
        rng.shuffle(pool)
        return [train_indices[i] for i in pool[:num_samples]]

    if curriculum_type == "class_balance":
        order = class_order if class_order is not None else [0, 1, 2, 3]
        max_classes = max(1, min(len(order), int(fraction * len(order)) + 1))
        target_classes = set(order[:max_classes])
        valid_local = [i for i in range(n) if train_labels[i] in target_classes]
        if not valid_local:
            valid_local = list(range(n))
        rng.shuffle(valid_local)
        take = min(num_samples, len(valid_local))
        return [train_indices[i] for i in valid_local[:take]]

    if curriculum_type == "inverse_difficulty":
        local = list(range(n))
        rng.shuffle(local)
        local.sort(key=lambda i: -float(train_difficulties[i]))
        return [train_indices[i] for i in local[:num_samples]]

    if curriculum_type == "preset_order":
        return [train_indices[i] for i in range(min(num_samples, n))]

    if curriculum_type == "difficulty":
        local = list(range(n))
        rng.shuffle(local)
        local.sort(key=lambda i: float(train_difficulties[i]))
        return [train_indices[i] for i in local[:num_samples]]

    raise ValueError(f"Unknown curriculum_type: {curriculum_type}")
