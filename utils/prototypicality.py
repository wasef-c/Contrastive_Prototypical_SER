#!/usr/bin/env python3
"""
Prototypicality calculation based on VAD distance
"""

import math
import torch


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


def batch_calculate_difficulty(batch, expected_vad):
    """
    Calculate difficulty for a batch

    Args:
        batch: dict with 'valence', 'arousal', 'dominance', 'label' tensors
        expected_vad: dict mapping label → [V, A, D] prototype

    Returns:
        tensor: [batch_size] difficulty scores
    """
    batch_size = batch['label'].shape[0]
    difficulties = []

    for i in range(batch_size):
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
