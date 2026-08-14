#!/usr/bin/env python3
"""Salience-based difficulty weighting.

The existing prototypical weighting scores a sample by its Euclidean
distance to its own class's VAD prototype. Measured against the baseline's
actual errors that quantity is close to useless: Spearman rho of -0.03 to
+0.17, flipping sign between IEMOCAP and MSP-Improv. The reason is that
the distance is unsigned, so an unusually loud and unusually positive
happy sample scores as atypical when such samples are in fact the easiest.

The directional version predicts error in every class and both corpora
(rho up to +0.36). An emotional sample is difficult when it sits close to
the neutral region of VAD space, and a neutral sample is difficult when it
sits far from it:

    difficulty = -distance(sample, neutral centroid)   for emotional
    difficulty = +distance(sample, neutral centroid)   for neutral

so higher always means harder. See motivation/README.md for the evidence.

Two implementation points that the earlier weighting got wrong:

  * Weights are normalised to mean 1 **within each class**, not globally.
    Difficulty correlates with class, so a global normalisation silently
    changes the effective class balance and confounds with the
    inverse-frequency class weights already in the loss.
  * The placebo draws a difficulty with the same per-class mean and
    variance from a stable hash of the utterance, so the weight
    distribution and its per-sample stability are preserved while the
    correspondence to the sample is destroyed. Without that arm a gain
    cannot be attributed to salience rather than to reweighting itself.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_salience_stats(vad: np.ndarray, labels: np.ndarray,
                           num_classes: int = 4) -> Dict[str, np.ndarray]:
    """Derive the constants needed to score salience difficulty.

    All statistics come from the training corpus only, so nothing about
    the evaluation corpora enters the weighting.

    Args:
        vad: [N, 3] training valence, arousal, dominance.
        labels: [N] training class labels, with 0 as neutral.
        num_classes: number of classes.

    Returns:
        Dict with the standardisation mean and scale, the neutral centroid
        in standardised space, and the per-class mean and standard
        deviation of the resulting difficulty.
    """
    mean = vad.mean(axis=0)
    scale = np.maximum(vad.std(axis=0), 1e-8)
    z = (vad - mean) / scale
    neutral_centre = z[labels == 0].mean(axis=0)

    dist = np.linalg.norm(z - neutral_centre, axis=1)
    difficulty = np.where(labels == 0, dist, -dist)

    class_mean = np.zeros(num_classes, dtype=np.float64)
    class_std = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        m = labels == c
        if m.sum() > 1:
            class_mean[c] = difficulty[m].mean()
            class_std[c] = max(difficulty[m].std(), 1e-8)

    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "neutral_centre": neutral_centre.astype(np.float32),
        "class_mean": class_mean.astype(np.float32),
        "class_std": class_std.astype(np.float32),
    }


def salience_difficulty(vad: torch.Tensor, labels: torch.Tensor,
                        stats: Dict[str, np.ndarray]) -> torch.Tensor:
    """Standardised salience difficulty for a batch, higher meaning harder.

    Args:
        vad: [B, 3] valence, arousal, dominance.
        labels: [B] class labels, with 0 as neutral.
        stats: output of compute_salience_stats.

    Returns:
        [B] difficulty, standardised within class so a value of 1 means
        one standard deviation harder than that class's average.
    """
    device = vad.device
    mean = torch.as_tensor(stats["mean"], device=device, dtype=vad.dtype)
    scale = torch.as_tensor(stats["scale"], device=device, dtype=vad.dtype)
    centre = torch.as_tensor(stats["neutral_centre"], device=device,
                             dtype=vad.dtype)

    z = (vad - mean) / scale
    dist = torch.linalg.norm(z - centre, dim=1)
    signed = torch.where(labels == 0, dist, -dist)

    c_mean = torch.as_tensor(stats["class_mean"], device=device, dtype=vad.dtype)
    c_std = torch.as_tensor(stats["class_std"], device=device, dtype=vad.dtype)
    idx = labels.long().clamp(0, len(c_mean) - 1)
    return (signed - c_mean[idx]) / c_std[idx]


def placebo_difficulty(keys: Tuple[str, ...], labels: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """Difficulty with the same distribution as the real score but no link to the sample.

    Drawn from a standard normal seeded by a stable hash of the utterance,
    matching the standardised real difficulty in mean and variance. Stable
    per utterance across epochs and seeds, exactly as the real score is.

    Args:
        keys: per-sample stable identifiers, typically transcripts.
        labels: [B] class labels, used only for shape.
        device: device for the returned tensor.

    Returns:
        [B] placebo difficulty.
    """
    values = []
    for key in keys:
        seed = abs(hash(("salience-placebo", key))) % (2 ** 31)
        values.append(np.random.default_rng(seed).standard_normal())
    return torch.tensor(values, device=device, dtype=torch.float32)


def salience_weights(difficulty: torch.Tensor, labels: torch.Tensor,
                     beta: float, num_classes: int = 4,
                     clip: float = 3.0, scope: str = "both") -> torch.Tensor:
    """Per-sample loss weights that emphasise difficult samples.

    The symmetric rule (scope "both") upweights two groups that teach
    opposite lessons: low-salience emotional samples, which say quiet
    speech can carry emotion, and high-salience neutral samples, which say
    activated speech can still be neutral. Measured at beta 0.5 the net
    effect was to widen neutral, raising neutral recall 2.6 points and
    dropping emotional recall 2.0 for -0.88 UAR, so the second group won.
    The scope argument exists to apply each half on its own and find out
    whether that reading is right.

    Args:
        difficulty: [B] standardised difficulty, higher meaning harder.
        labels: [B] class labels, with 0 as neutral.
        beta: strength. 0 reproduces uniform weighting; negative values
            downweight difficult samples instead, which is the sensible
            direction if those errors are irreducible rather than merely
            underweighted.
        num_classes: number of classes.
        clip: maximum ratio between the largest and smallest weight,
            bounding the variance that a handful of extreme samples can
            inject. Low-salience anger is rare, so without this a few
            hundred utterances would dominate the gradient.
        scope: "both", "emotional" (leave neutral samples at weight 1) or
            "neutral" (leave emotional samples at weight 1).

    Returns:
        [B] weights, mean 1 within each class present in the batch.
    """
    if scope not in ("both", "emotional", "neutral"):
        raise ValueError(f"unknown salience scope {scope!r}")

    w = torch.exp(beta * difficulty).clamp(1.0 / clip, clip)
    if scope == "emotional":
        w = torch.where(labels == 0, torch.ones_like(w), w)
    elif scope == "neutral":
        w = torch.where(labels == 0, w, torch.ones_like(w))

    # Mean 1 per class, so the effective class balance set by the
    # inverse-frequency class weights is left untouched.
    out = w.clone()
    for c in range(num_classes):
        m = labels == c
        if m.any():
            out[m] = w[m] / w[m].mean().clamp_min(1e-8)
    return out
