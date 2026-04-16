#!/usr/bin/env python3
"""
Prototypicality calculation based on VAD distance
"""

import math
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
        expected_vad: dict mapping label → [V, A, D] prototype

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


def batch_calculate_difficulty(batch, expected_vad):
    """
    Calculate difficulty for a batch.

    Samples from datasets without VAD annotations (CMUMOSEI, SAMSEMO) get
    difficulty=0 (fully prototypical) since their VAD values are fake defaults.

    Args:
        batch: dict with 'valence', 'arousal', 'dominance', 'label' tensors
               and 'dataset' list of corpus name strings
        expected_vad: dict mapping label → [V, A, D] prototype

    Returns:
        tensor: [batch_size] difficulty scores
    """
    batch_size = batch['label'].shape[0]
    dataset_names = batch.get('dataset', [None] * batch_size)
    difficulties = []

    for i in range(batch_size):
        ds_name = dataset_names[i] if dataset_names else None
        if ds_name is not None and ds_name not in DATASETS_WITH_VAD:
            # No real VAD → assign neutral difficulty
            difficulties.append(0.0)
            continue

        label = batch['label'][i].item()
        valence = batch['valence'][i].item()
        arousal = batch['arousal'][i].item()
        dominance = batch['dominance'][i].item()

        diff = calculate_difficulty(valence, arousal, dominance, label, expected_vad)
        difficulties.append(diff)

    return torch.tensor(difficulties, dtype=torch.float32)


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
