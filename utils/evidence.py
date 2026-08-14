#!/usr/bin/env python3
"""
Neutral as absence: evidence heads and the anti-neutral margin.

Two mechanisms aimed at the same observed failure from the SAMSEMO
analysis: 19.5 percent of emotional utterances were predicted neutral,
those utterances sat at low confidence margin (d = -1.2 to -1.5), and the
soft-label attempt to fix it (pa_neutral) made transfer sharply worse
(-1.58 points) because it moved target mass toward neutral and taught the
model that uncertain means neutral. Both mechanisms here push the other
way: they never give neutral any positive evidence to learn.

Evidence heads
--------------
A 4-way softmax treats neutral as a peer class with its own prototype,
so the model learns "what neutral sounds like" and weak emotional
utterances compete against that prototype. If neutral is the absence of
emotion rather than an emotion, the matching structure is one sigmoid
evidence head per emotional class and no neutral head at all:

    training   BCE per head. An emotional sample is the positive for its
               head and a negative for the others. A neutral sample is a
               negative for every head; it only ever teaches "no
               evidence here", never "this is what neutral looks like".
    inference  argmax over heads if the strongest evidence clears a
               threshold, else neutral.

Softmax must put its probability mass somewhere; sigmoids are allowed to
say nothing is present. A muted happy utterance needs only slightly
positive happy evidence to escape neutral, instead of having to look
unlike the neutral prototype.

Anti-neutral margin
-------------------
For the standard softmax head: a hinge penalty on emotional samples
whose true-class logit fails to clear the neutral logit by a margin,
weighted per sample by the model's own accumulated uncertainty (the
ConfidenceTracker EMA). Ambiguous emotional samples get the strongest
push away from neutral; confident ones are left alone. This is the
mirror image of the failed soft-label scheme: same samples, opposite
force direction.

The placebo for the ambiguity weighting is a fixed random permutation of
the per-sample weights (ambiguity_shuffle), which preserves the weight
distribution while destroying its relationship to the samples.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def evidence_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build per-head BCE targets from class labels.

    Head j corresponds to class j + 1 (class 0, neutral, has no head).

    Args:
        labels: [B] class labels in [0, num_classes).
        num_classes: number of primary classes including neutral.

    Returns:
        [B, num_classes - 1] float targets; all-zero rows for neutral.
    """
    onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
    return onehot[:, 1:]


def evidence_class_probs(evidence_logits: torch.Tensor) -> torch.Tensor:
    """Derive a normalized 4-way distribution from evidence logits.

    Neutral's probability is the joint probability that no emotion is
    present, treating heads as independent: prod(1 - p_j). The result is
    renormalized so downstream consumers (confidence tracking, NLL-style
    reporting) see a proper distribution.

    Args:
        evidence_logits: [B, C-1] sigmoid logits.

    Returns:
        [B, C] probabilities with neutral in column 0.
    """
    p = torch.sigmoid(evidence_logits.float())
    none = (1.0 - p).clamp_min(1e-6).prod(dim=1, keepdim=True)
    joint = torch.cat([none, p], dim=1)
    return joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-12)


def evidence_predictions(
    evidence_logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Decision rule: strongest evidence wins if it clears the threshold.

    Args:
        evidence_logits: [B, C-1] sigmoid logits.
        threshold: minimum sigmoid probability for any emotion to fire.

    Returns:
        [B] predicted class labels, 0 (neutral) when nothing fires.
    """
    p = torch.sigmoid(evidence_logits.float())
    best, idx = p.max(dim=1)
    return torch.where(best >= threshold, idx + 1,
                       torch.zeros_like(idx))


def calibrate_evidence_threshold(
    evidence_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    grid: Optional[Sequence[float]] = None,
) -> tuple:
    """Pick the evidence threshold that maximizes UAR on held-out data.

    A fixed 0.5 is an arbitrary operating point and it interacts with
    pos_weight: balancing the heads shifts every sigmoid upward, so the
    same threshold that under-fires without pos_weight over-fires with it.
    Both regimes were measured here. Without pos_weight the heads sat low,
    only 3 emotional heads in 10 cleared 0.5, and neutral recall rose to
    0.578 while emotional fell to 0.512. With pos_weight the heads sit near
    0.5, almost everything fires, and neutral recall collapsed to 0.036
    against emotional 0.703.

    Sweeping the threshold on validation data decouples the two: the loss
    decides how well each head separates its class, the threshold decides
    where to cut. Only the operating point is fitted, and it is fitted on
    data never used for testing.

    Args:
        evidence_logits: [N, C-1] logits over a held-out split.
        labels: [N] true class labels including neutral.
        num_classes: number of primary classes including neutral.
        grid: candidate thresholds; defaults to 0.05..0.95 in 0.025 steps.

    Returns:
        (best threshold, best UAR at that threshold).
    """
    if grid is None:
        grid = [round(0.05 + 0.025 * i, 4) for i in range(37)]

    y_true = labels.detach().cpu().numpy()
    best_thr, best_uar = 0.5, -1.0

    for thr in grid:
        preds = evidence_predictions(evidence_logits, thr).detach().cpu().numpy()
        recalls = []
        for c in range(num_classes):
            mask = y_true == c
            if mask.sum() == 0:
                continue
            recalls.append(float((preds[mask] == c).mean()))
        uar = float(np.mean(recalls)) if recalls else 0.0
        if uar > best_uar:
            best_thr, best_uar = float(thr), uar

    return best_thr, best_uar


def head_pos_weights(
    class_counts: Sequence[int],
    cap: float = 10.0,
) -> torch.Tensor:
    """Positive-class weights that balance each one-vs-rest head.

    Each emotional head sees its own class as positive and every other
    sample as negative, so on MSP-Podcast the positive rate runs 10 to 28
    percent. Plain BCE therefore biases every head toward "absent", the
    sigmoids sit low, few clear the decision threshold, and samples fall
    through to neutral. That is the opposite of what evidence heads are
    for, and it is what the first ev_heads run measured: neutral recall
    rose from 0.462 to 0.578 while emotional recall fell.

    A per-sample class weight cannot fix this. It scales a sample's whole
    loss, leaving the positive/negative ratio inside each head untouched.
    The correction has to be pos_weight = n_negative / n_positive.

    Args:
        class_counts: per-class training counts, index 0 = neutral.
        cap: upper bound on the ratio, so a rare class cannot dominate
            the gradient.

    Returns:
        [C-1] float tensor of positive weights, one per emotional head.
    """
    total = float(sum(class_counts))
    weights = []
    for c in range(1, len(class_counts)):
        pos = float(class_counts[c])
        neg = total - pos
        weights.append(min(neg / pos, cap) if pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def evidence_bce_loss(
    evidence_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-head BCE with imbalance, class and per-sample weighting.

    Args:
        evidence_logits: [B, C-1] head logits.
        labels: [B] class labels including neutral.
        num_classes: number of primary classes including neutral.
        class_weights: [C] inverse-frequency weights over the full label
            set. Applied per sample. NOTE this does not correct the
            within-head positive/negative imbalance; pass pos_weight for
            that.
        sample_weights: [B] extra multiplicative weights (ambiguity).
        pos_weight: [C-1] weight on the positive term of each head, from
            head_pos_weights(). Without it the heads under-fire and
            everything defaults to neutral.

    Returns:
        Scalar loss, normalized by batch size.
    """
    targets = evidence_targets(labels, num_classes)
    per_head = F.binary_cross_entropy_with_logits(
        evidence_logits.float(), targets, reduction="none",
        pos_weight=(pos_weight.to(evidence_logits.device)
                    if pos_weight is not None else None),
    )  # [B, C-1]
    per_sample = per_head.mean(dim=1)  # [B]

    if class_weights is not None:
        per_sample = per_sample * class_weights[labels.long()]
    if sample_weights is not None:
        per_sample = per_sample * sample_weights

    return per_sample.sum() / max(labels.size(0), 1)


def anti_neutral_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    neutral_index: int = 0,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hinge pushing emotional samples' logits above neutral's by a margin.

    Only emotional samples contribute; neutral samples are untouched, so
    the term cannot degrade neutral recall directly.

    Args:
        logits: [B, C] softmax-head logits.
        labels: [B] class labels.
        margin: required gap between the true-class and neutral logits.
        neutral_index: which class is neutral.
        sample_weights: [B] per-sample weights (ambiguity); weights on
            neutral rows are ignored.

    Returns:
        Scalar loss, averaged over emotional samples.
    """
    emotional = labels != neutral_index
    if not emotional.any():
        return logits.new_zeros(())

    lg = logits.float()
    true_logit = lg.gather(1, labels.view(-1, 1).long()).squeeze(1)
    neutral_logit = lg[:, neutral_index]
    hinge = F.relu(margin - (true_logit - neutral_logit))  # [B]

    hinge = hinge[emotional]
    if sample_weights is not None:
        w = sample_weights[emotional]
        return (hinge * w).sum() / w.sum().clamp_min(1e-6)
    return hinge.mean()


class AmbiguityWeights:
    """Per-sample loss weights from the model's own uncertainty.

    Wraps a ConfidenceTracker's EMA values into multiplicative weights:

        w_i = 1 + beta * (1 - v_i)

    where v_i is the sample's tracked confidence rescaled to [0, 1] over
    the seen population. The most ambiguous samples get weight 1 + beta,
    the most confident get 1. Unseen samples (first epochs, val split)
    get 1.

    The shuffle option is the placebo: a fixed seeded permutation of the
    value array is applied at lookup, preserving the weight distribution
    while destroying its per-sample meaning. If shuffled weights work as
    well as real ones, the benefit was generic re-weighting noise, not
    the ambiguity signal.

    Args:
        tracker: a ConfidenceTracker being updated during training.
        beta: maximum extra weight for a fully ambiguous sample.
        warmup_epochs: epochs before weights activate; until then every
            weight is 1 so early, uninformative confidence does not steer.
        shuffle: enable the placebo permutation.
        seed: permutation seed.
    """

    def __init__(
        self,
        tracker,
        beta: float = 1.0,
        warmup_epochs: int = 3,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.tracker = tracker
        self.beta = float(beta)
        self.warmup_epochs = int(warmup_epochs)
        self.shuffle = bool(shuffle)
        self._perm = None
        if shuffle:
            rng = np.random.RandomState(seed)
            self._perm = rng.permutation(len(tracker.values))

    def __call__(
        self,
        sample_index: torch.Tensor,
        current_epoch: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Weights for a batch, or None during warmup.

        Args:
            sample_index: [B] dataset positions.
            current_epoch: 1-based epoch counter.
            device: output device.

        Returns:
            [B] float weights, or None while warming up.
        """
        if current_epoch <= self.warmup_epochs:
            return None

        idx = sample_index.detach().cpu().numpy()
        if self._perm is not None:
            idx = self._perm[idx]
        vals = self.tracker.values[idx]

        seen = self.tracker.values[np.isfinite(self.tracker.values)]
        if seen.size < 10:
            return None
        lo, hi = float(seen.min()), float(seen.max())
        span = max(hi - lo, 1e-6)

        norm = (vals - lo) / span
        norm = np.where(np.isfinite(norm), np.clip(norm, 0.0, 1.0), 1.0)
        weights = 1.0 + self.beta * (1.0 - norm)
        return torch.tensor(weights, dtype=torch.float32, device=device)
