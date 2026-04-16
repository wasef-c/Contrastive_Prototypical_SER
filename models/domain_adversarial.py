#!/usr/bin/env python3
"""
Domain Adversarial Training with Prototypicality Weighting

Standard domain adversarial: make embeddings domain-invariant via gradient reversal.
Novel twist: weight adversarial signal by prototypicality.
  - Prototypical samples SHOULD be domain-invariant → strong adversarial weight
  - Atypical samples are inherently corpus-specific → weak adversarial weight
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    Forward: identity
    Backward: negate gradients and scale by lambda

    During forward pass, acts as identity.
    During backward pass, reverses gradient direction so the feature extractor
    learns to FOOL the domain discriminator.
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal"""

    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def set_lambda(self, lambda_val):
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator that predicts which corpus a sample comes from.

    With gradient reversal, the feature extractor learns to produce embeddings
    that the discriminator CANNOT distinguish by corpus → domain-invariant features.
    """

    def __init__(self, input_dim=1024, hidden_dim=256, num_domains=3):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def set_lambda(self, lambda_val):
        """Update GRL lambda (typically scheduled during training)"""
        self.grl.set_lambda(lambda_val)

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, input_dim] - shared embeddings from model

        Returns:
            domain_logits: [batch_size, num_domains]
        """
        reversed_embeddings = self.grl(embeddings)
        return self.classifier(reversed_embeddings)


class ModalityDomainDiscriminator(nn.Module):
    """
    Per-modality domain discriminator applied to PRE-FUSION features.

    Each modality (audio, text) has its own GRL + classifier head predicting
    the source corpus. Forces each encoder to produce corpus-invariant
    representations INDEPENDENTLY, before fusion gets a chance to mask
    modality-specific corpus signatures.

    Novel angle: previous domain adversarial work applies GRL to fused/shared
    embeddings, which may hide corpus-specific features that only exist in one
    modality (e.g. microphone characteristics in audio, scripted language in text).
    """

    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=256, num_domains=3):
        super().__init__()
        self.audio_grl = GradientReversalLayer()
        self.text_grl = GradientReversalLayer()

        self.audio_classifier = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def set_lambda(self, lambda_val):
        """Update GRL lambda for both modalities (typically scheduled during training)."""
        self.audio_grl.set_lambda(lambda_val)
        self.text_grl.set_lambda(lambda_val)

    def forward(self, audio_features, text_features):
        """
        Args:
            audio_features: [B, audio_dim] - pre-fusion audio features (WITH gradient)
            text_features: [B, text_dim] - pre-fusion text features (WITH gradient)

        Returns:
            audio_domain_logits: [B, num_domains]
            text_domain_logits: [B, num_domains]
        """
        audio_reversed = self.audio_grl(audio_features)
        text_reversed = self.text_grl(text_features)
        return self.audio_classifier(audio_reversed), self.text_classifier(text_reversed)


class PrototypicalDomainAdversarialLoss(nn.Module):
    """
    Domain adversarial loss weighted by prototypicality.

    Standard domain adversarial: L_adv = CE(domain_pred, domain_label)
    Ours: L_adv = mean_i[ w_i * CE_i(domain_pred_i, domain_label_i) ]

    where w_i = exp(-alpha * difficulty_i)

    Prototypical samples (low difficulty) → high weight → MUST be domain-invariant
    Atypical samples (high difficulty) → low weight → allowed to retain domain info
    """

    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, domain_logits, domain_labels, difficulties=None):
        """
        Args:
            domain_logits: [batch_size, num_domains] - discriminator output
            domain_labels: [batch_size] - corpus IDs (0, 1, 2, ...)
            difficulties: [batch_size] - prototypicality scores (optional)

        Returns:
            loss: scalar
        """
        per_sample_loss = self.ce_loss(domain_logits, domain_labels)

        if difficulties is not None:
            # Weight by prototypicality: canonical samples must be domain-invariant
            weights = torch.exp(-self.alpha * difficulties)
            # Normalize weights so they sum to batch_size (preserves loss scale)
            weights = weights * (weights.numel() / (weights.sum() + 1e-8))
            loss = (weights * per_sample_loss).mean()
        else:
            loss = per_sample_loss.mean()

        return loss
