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

    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, input_dim] shared backbone embeddings

        Returns:
            [batch] predicted prototypicality scores
        """
        return self.head(embeddings).squeeze(-1)
