#!/usr/bin/env python3
"""Class-dependent margin softmax (LDAM) for converting ranking into decisions.

Measured on the auxiliary prototypicality arms: emotion-vs-emotion AUC rose on
all four corpora, frozen and unfrozen, while UAR did not move. That pattern
means the correct class is climbing in score without overtaking its competitor,
so the improvement sits just under the decision boundary where a per-sample
argmax loss cannot see it.

Plain cross entropy only asks whether the true class is highest. A margin asks
whether it is highest BY AT LEAST m, so sub-threshold gains in ranking become
actual changes in the decision.

The margin is class-dependent following Cao et al. 2019 (Learning Imbalanced
Datasets with Label-Distribution-Aware Margin Loss), where m_c is proportional
to n_c^(-1/4). That schedule comes from minimising a bound on the BALANCED
error rate, which is what UAR measures, so it is the right form here rather
than a uniform margin.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class LDAMLoss(torch.nn.Module):
    """Label-distribution-aware margin cross entropy.

    Args:
        class_counts: samples per class in the training split.
        max_margin: margin assigned to the rarest class; all others scale
            down as n_c^(-1/4).
        scale: logit scale applied after the margin, as in the original.
        weight: optional per-class weights, passed through to cross entropy
            so this composes with inverse-frequency weighting.
    """

    def __init__(self, class_counts, max_margin: float = 0.5,
                 scale: float = 30.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        m = 1.0 / np.power(counts, 0.25)
        m = m * (max_margin / m.max())
        self.register_buffer("margins", torch.tensor(m, dtype=torch.float32))
        self.scale = float(scale)
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """Cross entropy after subtracting the true class's margin.

        Args:
            logits: [B, C] raw scores.
            labels: [B] class ids.

        Returns:
            Scalar loss.
        """
        idx = torch.zeros_like(logits, dtype=torch.bool)
        idx.scatter_(1, labels.view(-1, 1).long(), True)
        # Only the true class is penalised, so the model must clear the margin
        # rather than merely win.
        adjusted = logits - idx.float() * self.margins.to(logits.device)
        return F.cross_entropy(self.scale * adjusted, labels,
                               weight=self.weight)
