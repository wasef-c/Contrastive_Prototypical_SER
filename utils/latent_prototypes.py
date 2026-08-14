#!/usr/bin/env python3
"""
Latent sub-prototypes: let the model discover its own subtypes.

Every prototypicality split tested in this project was imposed and then
supervised. Distance to a VAD centroid, distance to a lexical profile,
the model's own frozen confidence: each produced a per-sample sub-label
that the widened head was trained to predict. All of them matched or lost
to a random hash, and the reason is visible in the transfer numbers:
predicting the split from audio gave cross-corpus AUC of 0.50 to 0.56, so
the split named different samples in the target corpus than in the source.
An imposed label that does not transfer cannot carry information, whatever
its content.

This inverts that. The head still widens to num_classes * K, but no
sub-label is ever supervised. Training uses the collapsed class
probability

    p(class c) = sum over that class's K prototypes of softmax(logits)

so the loss only knows the parent emotion. How samples distribute across
a class's prototypes is a latent variable the optimiser settles however
best reduces the parent loss.

Three failure modes we measured are avoided by construction:

  * no transfer problem, because no external definition has to mean the
    same thing in IEMOCAP as in MSP-Podcast;
  * no rediscovery of speaker identity. k-means on emotion2vec finds
    speaker at 2.7x the emotion signal because clustering explains
    variance; prototypes trained through a classification loss are pulled
    toward whatever discriminates emotion, and variance that does not
    help is ignored;
  * no boundary shift. Collapsing at eval leaves the four-way decision
    surface unchanged in form, so this sidesteps the zero-sum trade that
    cost every boundary mechanism we tried (neutral recall fell to 0.128
    under the strongest of them, for -2.89 UAR).

The degenerate outcome is real and worth naming in advance: nothing stops
the model putting almost every sample of a class into one prototype and
leaving the rest empty. That is a clean negative answer rather than a bug,
which is why prototype usage is reported rather than forced. An optional
usage-entropy bonus can encourage spread, but it is off by default because
turning it on reintroduces an imposed structure.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def collapse_prototype_logits(
    logits: torch.Tensor,
    num_classes: int,
    protos_per_class: int,
) -> torch.Tensor:
    """Sum each class's prototype probabilities into a class distribution.

    Logits are class-major: prototypes of class 0 first, then class 1, and
    so on.

    Args:
        logits: [B, num_classes * protos_per_class] raw logits.
        num_classes: number of primary classes.
        protos_per_class: prototypes K per class.

    Returns:
        [B, num_classes] probabilities summing to 1.
    """
    probs = F.softmax(logits.float(), dim=-1)
    return probs.view(-1, num_classes, protos_per_class).sum(dim=2)


def prototype_assignments(
    logits: torch.Tensor,
    num_classes: int,
    protos_per_class: int,
) -> torch.Tensor:
    """Which prototype within its predicted class each sample activates.

    Args:
        logits: [B, num_classes * protos_per_class] raw logits.
        num_classes: number of primary classes.
        protos_per_class: prototypes K per class.

    Returns:
        [B] index in [0, num_classes * protos_per_class) of the strongest
        prototype overall.
    """
    return logits.float().argmax(dim=-1)


def latent_prototype_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    protos_per_class: int,
    class_weights: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    usage_entropy_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Negative log-likelihood of the collapsed class probability.

    Only the parent emotion is supervised. The split across a class's
    prototypes is never told to the model and emerges from optimisation.

    Args:
        logits: [B, num_classes * protos_per_class] raw logits.
        labels: [B] parent class labels.
        num_classes: number of primary classes.
        protos_per_class: prototypes K per class.
        class_weights: [num_classes] inverse-frequency weights, applied
            per sample exactly as in the plain softmax path so the
            imbalance correction is unchanged.
        sample_weights: [B] additional per-sample weights.
        usage_entropy_weight: if > 0, subtract this times the batch-level
            entropy of prototype usage, encouraging samples to spread
            across prototypes rather than collapsing onto one. Off by
            default: forcing spread imposes exactly the kind of structure
            this design is meant to avoid assuming.

    Returns:
        (scalar loss, [B, num_classes] collapsed class probabilities).
    """
    class_probs = collapse_prototype_logits(logits, num_classes, protos_per_class)
    per_sample = -torch.log(
        class_probs.gather(1, labels.view(-1, 1).long()).squeeze(1).clamp_min(1e-12)
    )

    if class_weights is not None:
        per_sample = per_sample * class_weights[labels.long()]
    if sample_weights is not None:
        per_sample = per_sample * sample_weights

    loss = per_sample.sum() / max(labels.size(0), 1)

    if usage_entropy_weight > 0.0:
        # Mean prototype usage over the batch; high entropy means samples
        # are spread across prototypes rather than piled on one.
        usage = F.softmax(logits.float(), dim=-1).mean(dim=0)
        entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        loss = loss - usage_entropy_weight * entropy

    return loss, class_probs


def usage_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    protos_per_class: int,
) -> torch.Tensor:
    """Count how often each prototype is the argmax, per true class.

    Used to detect the degenerate outcome where a class routes nearly all
    of its samples through a single prototype, which would mean no latent
    subtype structure was found.

    Args:
        logits: [B, num_classes * protos_per_class] raw logits.
        labels: [B] true class labels.
        num_classes: number of primary classes.
        protos_per_class: prototypes K per class.

    Returns:
        [num_classes, protos_per_class] counts.
    """
    winners = prototype_assignments(logits, num_classes, protos_per_class)
    counts = torch.zeros(num_classes, protos_per_class,
                         dtype=torch.long, device=logits.device)
    for c in range(num_classes):
        m = labels == c
        if not m.any():
            continue
        w = winners[m]
        for k in range(protos_per_class):
            counts[c, k] = int(((w // protos_per_class == c)
                                & (w % protos_per_class == k)).sum())
    return counts
