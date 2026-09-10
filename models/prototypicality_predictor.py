#!/usr/bin/env python3
"""
Auxiliary Prototypicality Prediction Head

Multi-task learning: force the shared backbone to encode WHERE a sample sits
relative to its class prototype. The prediction head regresses the prototypicality
score (VAD-based difficulty) from the shared embedding.

At test time, the head is discarded — its value is in regularizing the backbone
to encode subjectivity/annotation-confidence structure.
"""

import torch
import torch.nn as nn


class PrototypicalityPredictor(nn.Module):
    """
    Small MLP that predicts prototypicality score from shared embedding.

    Input: shared embedding [batch, hidden_dim] (1024)
    Output: predicted prototypicality [batch, 1]
    Target: actual difficulty = euclidean_dist(sample_VAD, class_centroid_VAD)
    """

    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=1):
        """
        Args:
            input_dim: width of the shared embedding.
            hidden_dim: width of the hidden layer.
            output_dim: 1 for a scalar distance, 3 for the VAD residual,
                or num_classes * clusters_per_class for subtype logits.
                A scalar collapses a 3-D position into one number, so a
                sample that is atypical because it is loud and one that is
                atypical because it is quiet receive the same target; the
                wider variants keep that direction.
        """
        super().__init__()
        self.output_dim = int(output_dim)
        if int(hidden_dim) <= 0:
            # Linear probe. The head's job is not to predict the target well,
            # it is to force the SHARED embedding to carry the information. A
            # wider head can fit the target from whatever the embedding
            # already holds, which shrinks the gradient reaching the trunk and
            # lets the head absorb the task. A linear head cannot, so the only
            # way to reduce the loss is for the embedding itself to become
            # more informative.
            self.head = nn.Linear(input_dim, self.output_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.output_dim),
            )

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, input_dim] shared backbone embeddings

        Returns:
            [batch] when output_dim is 1, otherwise [batch, output_dim].
        """
        out = self.head(embeddings)
        return out.squeeze(-1) if self.output_dim == 1 else out
