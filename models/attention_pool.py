#!/usr/bin/env python3
"""
Attention pooling over frame-level emotion2vec features.

Every pooling mode tried so far applies a fixed rule over time: the mean,
the standard deviation, or the means of fixed segments. The seg8 result
suggests fixed high-resolution slicing is the wrong direction. It kept
the neutral-recall gain (+8.3 points) but lost 6 points of emotional
recall for -2.42 UAR overall, which is what over-resolved, unweighted
temporal detail looks like when it overfits one corpus.

Attention pooling differs in that the weighting is learned and
utterance-specific. A query vector scores every frame, and the output is
the softmax-weighted sum. The model decides which moments matter instead
of averaging them all equally or chopping them into fixed bins. Multiple
heads let it attend to several things at once (onset, peak, decay) and
concatenate the results.

This is attentive pooling as used in speaker verification (Okabe et al.,
2018); it is standard machinery, not a novel mechanism. It is included
because it is the natural way to keep the time axis without the fixed
slicing that seg8 showed to be harmful.

Note on the emotion2vec paper: its Table 9 shows frame-level features
beating the model's own learned utterance embedding for downstream tasks
by nine points (71.79 vs 62.77 WA). It does not say how frame features
are aggregated for utterance-level prediction, citing SUPERB practice,
which is mean pooling. So the paper endorses using frame output but takes
no position on how to pool it.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Multi-head attention pooling over a frame sequence.

    Each head owns a learned query. Scores come from a small MLP over the
    frames, are softmaxed across time, and produce one weighted sum per
    head; the heads are concatenated. With num_heads=1 and a degenerate
    scorer this reduces to mean pooling, so the mode we already trust is
    inside the hypothesis space rather than excluded by it.

    Args:
        input_dim: frame feature width (768 for emotion2vec).
        num_heads: number of independent attention queries. Output width
            is input_dim * num_heads.
        hidden_dim: width of the scoring MLP.
        dropout: dropout on the scoring MLP.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.output_dim = input_dim * num_heads

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool a frame sequence into a fixed-width vector.

        Args:
            frames: [B, T, D] frame features.
            mask: [B, T] with 1 for real frames and 0 for padding. Padded
                positions are set to -inf before the softmax so they
                receive exactly zero weight.

        Returns:
            [B, D * num_heads] pooled features.
        """
        scores = self.score(frames)                      # [B, T, H]
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.bool().unsqueeze(-1), float("-inf"),
            )
        weights = F.softmax(scores, dim=1)               # [B, T, H]
        # [B, H, T] @ [B, T, D] -> [B, H, D], then flatten the head axis.
        pooled = torch.bmm(weights.transpose(1, 2), frames)
        return pooled.reshape(frames.shape[0], -1)

    def attention_weights(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the per-frame weights, for inspecting what the model attends to.

        Args:
            frames: [B, T, D] frame features.
            mask: [B, T] real-frame mask.

        Returns:
            [B, T, num_heads] softmax weights over time.
        """
        scores = self.score(frames)
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.bool().unsqueeze(-1), float("-inf"),
            )
        return F.softmax(scores, dim=1)


class MeanPoolControl(nn.Module):
    """Capacity-matched control for AttentionPool.

    Produces the same output width by repeating the mean num_heads times,
    and carries a scoring MLP of identical shape whose output is discarded.
    So the parameter count and the downstream fusion width match the
    attention arm exactly while the pooled content is just the mean.

    This is the same logic as the dup<K> pooling control, which showed
    that 76 percent of the apparent mean_std_halves UAR gain came from a
    wider fusion layer rather than from temporal information. Without it,
    any attention-pooling result is uninterpretable.

    Args:
        input_dim: frame feature width.
        num_heads: repetition factor, matching AttentionPool.
        hidden_dim: width of the unused scoring MLP.
        dropout: dropout in the unused MLP.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.output_dim = input_dim * num_heads
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-pool and repeat, ignoring the scoring MLP.

        Args:
            frames: [B, T, D] frame features.
            mask: [B, T] real-frame mask.

        Returns:
            [B, D * num_heads] pooled features.
        """
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            mean = (frames * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            mean = frames.mean(dim=1)
        # Touch the unused parameters so they receive gradient and the two
        # arms remain optimisation-comparable rather than merely
        # parameter-count-comparable.
        _ = self.score(frames).sum() * 0.0
        return mean.repeat(1, self.num_heads) + _
