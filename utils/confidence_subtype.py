#!/usr/bin/env python3
"""
Subtypes defined by the model's own uncertainty, and soft neutral targets.

Two mechanisms, both annotation-free, both motivated by the SAMSEMO
failure analysis.

ConfidenceTracker
-----------------
Every prototypicality definition tested so far was static and externally
defined: distance from a VAD centroid, distance from a lexical profile.
Both failed the same way. Predicting the resulting typical/atypical split
from audio gave a cross-corpus AUC of 0.50 to 0.56, meaning the split
named different samples in the target corpus than in the training corpus,
so no scheme built on it could transfer. As training signals they were
indistinguishable from a random hash, and the lexical one was worse.

Model confidence does not have that problem structurally. It is computed
from the input at inference time, so it is defined on any corpus without
annotations or re-estimated centroids. It was also by far the strongest
signal in the failure analysis: utterances the model leaked into neutral
had a confidence margin near 0.25 against 0.6 for those it classified
correctly, an effect size of d = -1.2 to -1.5, larger than any acoustic
or annotator feature.

The tracker accumulates an exponential moving average of per-sample
confidence across epochs, then freezes a per-class median split into
typical and atypical. Freezing matters: confidence in early epochs
reflects optimization state as much as sample difficulty, and a split
that churns every epoch is a moving target. Because the split comes from
the model at epoch N and is applied from N+1 onward, it acts as a
stop-gradient teacher signal rather than instantaneous self-prediction,
which keeps it closer to self-distillation than to a degenerate loop.

There is no escape hatch for the model. Evaluation collapses the proto
and atyp logits of a class by summing their probabilities, so routing an
utterance to "atypical happy" still commits to predicting happy. The
model cannot reduce loss by dumping hard samples somewhere harmless.

neutral_soft_targets
--------------------
The confusion analysis found 19.5 percent of emotional utterances
predicted neutral, and those utterances were acoustically weaker: mean
energy d = -0.41 for angry and -0.29 for happy, mean pitch d = -0.84 for
angry. They were also twice as likely to have a human annotator who
independently called them neutral (28.3 vs 14.0 percent, odds ratio 2.43,
p = 1.2e-13).

That points at neutral not being a peer category but the region where no
emotion is strong enough to register. A quiet happy utterance is
genuinely part neutral, and a one-hot target asserts otherwise. This
function moves a portion of the target mass onto neutral in inverse
proportion to the sample's intensity, so weak instances get a blended
target and strong ones stay effectively one-hot.

Intensity is used only to shape the training target. The model never has
to estimate it at test time, which is why using VAD here does not inherit
the transfer failure that sank the VAD-based prototypicality splits.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class ConfidenceTracker:
    """Per-sample EMA of model confidence, frozen into a proto/atyp split.

    Args:
        num_samples: size of the training dataset (indexes are positions in
            train_dataset.data).
        num_classes: number of primary classes.
        momentum: EMA weight on the running value. Higher means slower
            adaptation.
        metric: "margin" (top-1 minus top-2 probability) or "true_prob"
            (probability assigned to the correct class). Margin measures
            decision-boundary proximity; true_prob measures how wrong the
            model is.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        momentum: float = 0.9,
        metric: str = "margin",
    ) -> None:
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.metric = metric
        self.values = np.full(num_samples, np.nan, dtype=np.float32)
        self.labels = np.full(num_samples, -1, dtype=np.int64)
        self.thresholds: Optional[np.ndarray] = None
        self.frozen = False

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_index: torch.Tensor,
    ) -> None:
        """Fold one batch of predictions into the running averages.

        Args:
            logits: [B, C] class logits. When the head is widened these must
                already be collapsed to per-class probabilities.
            labels: [B] true class labels.
            sample_index: [B] positions into the training dataset.
        """
        if self.frozen:
            return
        probs = F.softmax(logits.float(), dim=-1)
        if self.metric == "true_prob":
            score = probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
        else:
            top2 = probs.topk(2, dim=-1).values
            score = top2[:, 0] - top2[:, 1]

        idx = sample_index.detach().cpu().numpy()
        val = score.detach().cpu().numpy().astype(np.float32)
        lab = labels.detach().cpu().numpy().astype(np.int64)

        current = self.values[idx]
        fresh = np.isnan(current)
        updated = np.where(
            fresh, val, self.momentum * current + (1.0 - self.momentum) * val,
        )
        self.values[idx] = updated
        self.labels[idx] = lab

    def freeze(self, criterion: str = "per_class_median") -> np.ndarray:
        """Fix the typical/atypical thresholds from the accumulated averages.

        Args:
            criterion: "per_class_median" splits each class at its own
                median confidence, keeping the sub-label balance even.
                "global_median" uses one threshold everywhere, so classes
                the model finds harder contribute more atypical samples.

        Returns:
            [num_classes] float32 thresholds. Samples with confidence at or
            below the threshold are atypical.
        """
        thresholds = np.zeros(self.num_classes, dtype=np.float32)
        seen = ~np.isnan(self.values)

        if criterion == "global_median":
            pooled = self.values[seen]
            value = float(np.median(pooled)) if pooled.size else 0.0
            thresholds[:] = value
        else:
            for c in range(self.num_classes):
                mask = seen & (self.labels == c)
                thresholds[c] = (float(np.median(self.values[mask]))
                                 if mask.any() else 0.0)

        self.thresholds = thresholds
        self.frozen = True
        return thresholds

    def coverage(self) -> float:
        """Fraction of training samples that have received a confidence value."""
        return float(np.isfinite(self.values).mean())

    def sub_labels(
        self,
        labels: torch.Tensor,
        sample_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Look up frozen sub-labels for a batch.

        sub = 2 * class + is_atypical, where is_atypical marks a sample whose
        tracked confidence sits at or below its class threshold. Samples never
        seen during the accumulation phase default to typical.

        Args:
            labels: [B] true class labels.
            sample_index: [B] positions into the training dataset.
            device: torch device for the output.

        Returns:
            [B] LongTensor of sub-labels.

        Raises:
            RuntimeError: if called before freeze().
        """
        if self.thresholds is None:
            raise RuntimeError("ConfidenceTracker.freeze() must run first")

        idx = sample_index.detach().cpu().numpy()
        lab = labels.detach().cpu().numpy().astype(np.int64)
        vals = self.values[idx]
        thr = self.thresholds[np.clip(lab, 0, self.num_classes - 1)]
        atypical = np.where(np.isnan(vals), 0, (vals <= thr).astype(np.int64))

        return (labels.to(device).long() * 2
                + torch.from_numpy(atypical).to(device).long())


def neutral_soft_targets(
    labels: torch.Tensor,
    intensity: torch.Tensor,
    num_classes: int,
    alpha: float = 0.3,
    neutral_index: int = 0,
) -> torch.Tensor:
    """Blend a share of each emotional target onto neutral, by intensity.

    A sample at the weakest end of its class's intensity range gives up
    `alpha` of its target mass to neutral; one at the strongest end gives up
    nothing. Samples already labelled neutral are left one-hot, since the
    construction only expresses "this emotion is too weak to fully register"
    and there is no weaker state for neutral to decay toward.

    Args:
        labels: [B] true class labels.
        intensity: [B] per-sample intensity already normalized to [0, 1]
            within its class, where 0 is weakest. Non-finite entries are
            treated as full intensity, leaving those targets one-hot.
        num_classes: number of primary classes.
        alpha: maximum share of target mass moved onto neutral.
        neutral_index: which class index is neutral.

    Returns:
        [B, num_classes] float target distribution, each row summing to 1.
    """
    targets = F.one_hot(labels.long(), num_classes=num_classes).float()

    weakness = 1.0 - intensity.clamp(0.0, 1.0)
    weakness = torch.nan_to_num(weakness, nan=0.0)
    share = (alpha * weakness).unsqueeze(1)  # [B, 1]

    is_neutral = (labels == neutral_index).unsqueeze(1)
    share = torch.where(is_neutral, torch.zeros_like(share), share)

    targets = targets * (1.0 - share)
    targets[:, neutral_index] = targets[:, neutral_index] + share.squeeze(1)
    return targets


def class_relative_intensity(
    vad: torch.Tensor,
    labels: torch.Tensor,
    class_ranges: torch.Tensor,
    neutral_vad: torch.Tensor,
) -> torch.Tensor:
    """Rank a sample's VAD distance from neutral within its own class.

    Intensity is measured relative to the class rather than globally,
    because the classes sit at different distances from neutral by
    construction: angry is far from the neutral point in the expected_vad
    table while sad is much closer. A global scale would mark most sad
    samples weak purely because sadness is a low-arousal state.

    Args:
        vad: [B, 3] valence, arousal, dominance.
        labels: [B] class labels.
        class_ranges: [num_classes, 2] per-class (min, max) raw distance,
            precomputed over the training split.
        neutral_vad: [3] neutral reference point.

    Returns:
        [B] intensity in [0, 1], where 1 is the strongest instance of that
        class seen during training.
    """
    dist = torch.sqrt(((vad - neutral_vad.unsqueeze(0)) ** 2).sum(dim=1))
    lo = class_ranges[labels, 0]
    hi = class_ranges[labels, 1]
    span = (hi - lo).clamp_min(1e-6)
    return ((dist - lo) / span).clamp(0.0, 1.0)
