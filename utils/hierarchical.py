#!/usr/bin/env python3
"""
Hierarchical head: detect emotion presence, then identify which emotion.

The 4-way softmax treats neutral as a peer of happy, sad and angry, so a
weak emotional utterance has to out-compete a learned neutral prototype
whose mass is inflated by neutral being 52 percent of MSP-Podcast. The
evidence-head alternative removed neutral entirely but introduced a free
decision threshold, and that threshold interacted badly with the per-head
class weighting: two separate bugs, one where nothing fired (neutral
recall 0.578) and one where everything did (neutral recall 0.036).

This factors the problem instead:

    P(neutral) = 1 - P(emotional)
    P(c)       = P(emotional) * P(c | emotional)      for c in {hap, sad, ang}

One sigmoid answers "is there emotion here", one 3-way softmax answers
"which emotion", and the product is a proper 4-way distribution. Argmax
over it needs no threshold.

Why this shape suits the problem:

  * the detector is a balanced binary task (48 percent positive) rather
    than the 11 to 28 percent positive rates that destabilised the
    one-vs-rest heads;
  * the emotion head never models neutral, so it is not forced to draw a
    boundary between "weak happiness" and "no emotion";
  * neutral samples supply gradient to the detector but none to the
    emotion head, since their likelihood does not involve it. Neutral
    still gets learned, without becoming a competing class;
  * composition is soft, so a sample the detector is unsure about keeps
    scaled-down emotion probability instead of being hard-gated away.

P(emotional) is also a learned salience score. Loud sad and muted happy
differ in it, and unlike the VAD-derived intensity measures tried earlier
it is predicted from the input, so it is defined on any corpus.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def compose_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Build the 4-way distribution from detector and emotion logits.

    Args:
        logits: [B, C] where column 0 is the emotion-presence logit and
            columns 1: are the emotion logits.

    Returns:
        [B, C] probabilities, column 0 neutral. Rows sum to 1 by
        construction.
    """
    p_emo = torch.sigmoid(logits[:, :1].float())          # [B, 1]
    which = F.softmax(logits[:, 1:].float(), dim=1)       # [B, C-1]
    return torch.cat([1.0 - p_emo, p_emo * which], dim=1)


def emotion_class_weights(class_counts: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency weights over the emotional classes only.

    The identity head is a 3-way problem restricted to emotional samples,
    so its imbalance is measured within that subset, not over all four
    classes. On MSP-Podcast happy is 52 percent of emotional utterances
    while sad is 20 percent.

    Args:
        class_counts: per-class training counts, index 0 = neutral.

    Returns:
        [C-1] float tensor normalized to sum to C-1.
    """
    emo = [float(c) for c in class_counts[1:]]
    total = sum(emo)
    raw = [(total / c) if c > 0 else 1.0 for c in emo]
    scale = sum(raw)
    return torch.tensor([r / scale * len(emo) for r in raw], dtype=torch.float32)


def hierarchical_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    neutral_index: int = 0,
    detector_pos_weight: Optional[float] = None,
    class_weights: Optional[torch.Tensor] = None,
    emotion_weights: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    detector_weight: float = 1.0,
) -> torch.Tensor:
    """Negative log-likelihood of the composed distribution.

    Written as two terms rather than one NLL so each can be weighted, but
    it is the same quantity:

        neutral sample   -log(1 - p_emo)
        emotional sample -log(p_emo) - log(softmax_c)

    The emotion term is masked to emotional samples, which is what keeps
    the emotion head from ever seeing neutral as a target.

    The 4-class inverse-frequency weights belong on BOTH terms, including
    the detector. It is tempting to argue otherwise: the detector is a
    near-balanced binary problem (48 percent positive), so the 4-class
    weights appear to bias it 2.5x toward "emotional" for no reason. That
    argument is wrong, and removing them costs 4.8 UAR points with neutral
    recall rising to 0.807 against emotional 0.414.

    The reason is the metric. UAR gives each of the four classes equal
    weight, but neutral holds 52 percent of the data while happy, sad and
    angry hold 28, 11 and 15. Maximising UAR therefore requires predicting
    emotional more often than the natural prior suggests, and the "bias"
    the 4-class weights introduce into the detector is exactly that
    correction. Judge the detector against the reported metric, not
    against its own binary class balance.

    Args:
        logits: [B, C] detector logit in column 0, emotion logits after.
        labels: [B] class labels including neutral.
        neutral_index: which label is neutral.
        detector_pos_weight: extra weight on the positive (emotional) term
            of the detector, on top of class_weights. None is normal.
        class_weights: [C] inverse-frequency weights applied per sample to
            both terms. This is the UAR correction described above.
        emotion_weights: [C-1] optional extra weights on the identity term
            only. Normally None, since class_weights already covers it.
        sample_weights: [B] extra weights (ambiguity).
        detector_weight: multiplier on the presence term relative to the
            identity term.

    Returns:
        Scalar loss averaged over the batch.
    """
    labels = labels.long()
    is_emotional = (labels != neutral_index).float()

    detector_logit = logits[:, 0].float()
    detector = F.binary_cross_entropy_with_logits(
        detector_logit, is_emotional, reduction="none",
        pos_weight=(torch.tensor(detector_pos_weight, device=logits.device)
                    if detector_pos_weight is not None else None),
    )  # [B]

    # Emotion identity: index into the emotion block, so class c maps to
    # column c - 1. Neutral rows are masked out entirely.
    emo_targets = (labels - 1).clamp_min(0)
    identity = F.cross_entropy(
        logits[:, 1:].float(), emo_targets, reduction="none",
        weight=(emotion_weights.to(logits.device)
                if emotion_weights is not None else None),
    ) * is_emotional  # [B]

    per_sample = detector_weight * detector + identity

    # Applied to both terms. See the note above on why the detector needs
    # this despite being near-balanced in its own right.
    if class_weights is not None:
        per_sample = per_sample * class_weights[labels]
    if sample_weights is not None:
        per_sample = per_sample * sample_weights

    return per_sample.sum() / max(labels.size(0), 1)


def hierarchical_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the composed 4-way distribution. No threshold.

    Args:
        logits: [B, C] detector logit in column 0, emotion logits after.

    Returns:
        [B] predicted class labels.
    """
    return compose_class_probs(logits).argmax(dim=1)


def emotion_salience(logits: torch.Tensor) -> torch.Tensor:
    """P(emotional): how strongly any emotion is present.

    Args:
        logits: [B, C] head output.

    Returns:
        [B] salience in [0, 1].
    """
    return torch.sigmoid(logits[:, 0].float())
