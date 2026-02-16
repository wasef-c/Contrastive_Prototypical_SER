#!/usr/bin/env python3
"""
Contrastive loss implementations with prototypicality weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)

    Pulls samples from the same class together, pushes different classes apart.
    Based on: https://arxiv.org/abs/2004.11362

    Formula:
        L_i = -1/|P(i)| * Σ_{p∈P(i)} log[exp(sim(z_i,z_p)/τ) / Σ_a exp(sim(z_i,z_a)/τ)]

    where:
        - z_i, z_p: L2-normalized embeddings
        - P(i): set of positives (same class as i, excluding i)
        - τ: temperature parameter
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, weights=None):
        """
        Args:
            embeddings: [batch_size, embedding_dim] - L2 normalized
            labels: [batch_size] - class labels
            weights: [batch_size] - optional sample weights (for prototypicality)

        Returns:
            loss: scalar tensor
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Create positive mask (same class, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        mask_positive.fill_diagonal_(0)  # Exclude self

        # Compute log probabilities
        exp_logits = torch.exp(logits)
        # Mask out self-similarity
        exp_logits = exp_logits * (1 - torch.eye(batch_size, device=device))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Compute mean log-likelihood over positives
        num_positives = mask_positive.sum(dim=1)
        # Avoid division by zero
        num_positives = torch.clamp(num_positives, min=1.0)

        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / num_positives

        # Loss (negative log-likelihood)
        loss = -mean_log_prob_pos

        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights

        return loss.mean()


class PrototypicalContrastiveLoss_V1(nn.Module):
    """
    Prototypical Contrastive Loss - Variant 1: Sample Weighting

    Weights each sample by its prototypicality:
        weight_i = exp(-alpha * difficulty_i)

    Prototypical samples (low difficulty) get higher weight → pulled harder
    Atypical samples (high difficulty) get lower weight → less influence
    """

    def __init__(self, temperature=0.07, alpha=1.0):
        super().__init__()
        self.base_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.alpha = alpha

    def forward(self, embeddings, labels, difficulties):
        """
        Args:
            embeddings: [batch_size, embedding_dim] - L2 normalized
            labels: [batch_size]
            difficulties: [batch_size] - prototypicality scores

        Returns:
            loss: scalar
        """
        # Compute weights: prototypical samples get higher weight
        weights = torch.exp(-self.alpha * difficulties)

        return self.base_loss(embeddings, labels, weights=weights)


class PrototypicalContrastiveLoss_V2(nn.Module):
    """
    Prototypical Contrastive Loss - Variant 2: Pair Weighting

    Weights each positive pair by combined difficulty:
        weight_ij = exp(-beta * (difficulty_i + difficulty_j))

    Both prototypical → strongest pull
    Mixed → medium pull
    Both atypical → weak pull
    """

    def __init__(self, temperature=0.07, beta=0.5):
        super().__init__()
        self.temperature = temperature
        self.beta = beta

    def forward(self, embeddings, labels, difficulties):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
            difficulties: [batch_size]

        Returns:
            loss: scalar
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Create positive mask
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        mask_positive.fill_diagonal_(0)

        # Compute pair-wise difficulty weights
        # [batch_size, 1] + [1, batch_size] → [batch_size, batch_size]
        difficulty_sum = difficulties.unsqueeze(1) + difficulties.unsqueeze(0)
        pair_weights = torch.exp(-self.beta * difficulty_sum)

        # Apply weights to positive pairs only
        weighted_mask_positive = mask_positive * pair_weights

        # Compute log probabilities
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (1 - torch.eye(batch_size, device=device))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Weighted mean over positives
        num_positives = weighted_mask_positive.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1e-6)

        mean_log_prob_pos = (weighted_mask_positive * log_prob).sum(dim=1) / num_positives

        loss = -mean_log_prob_pos
        return loss.mean()


class PrototypicalContrastiveLoss_V3(nn.Module):
    """
    Prototypical Contrastive Loss - Variant 3: Threshold Separation

    Binary separation of prototypical vs atypical:
        weight_i = 1.0 if difficulty_i < threshold else 0.1

    Focuses learning on prototypical samples, reduces atypical influence
    """

    def __init__(self, temperature=0.07, threshold=1.0, atypical_weight=0.1):
        super().__init__()
        self.base_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.threshold = threshold
        self.atypical_weight = atypical_weight

    def forward(self, embeddings, labels, difficulties):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
            difficulties: [batch_size]

        Returns:
            loss: scalar
        """
        # Binary weights
        weights = torch.where(
            difficulties < self.threshold,
            torch.ones_like(difficulties),
            torch.ones_like(difficulties) * self.atypical_weight
        )

        return self.base_loss(embeddings, labels, weights=weights)


def create_contrastive_loss(loss_type, **kwargs):
    """
    Factory function to create contrastive loss

    Args:
        loss_type: str - "supervised", "prototypical_v1", "prototypical_v2", "prototypical_v3"
        **kwargs: loss-specific parameters

    Returns:
        Contrastive loss module
    """
    temperature = kwargs.get('temperature', 0.07)

    if loss_type == "supervised":
        return SupervisedContrastiveLoss(temperature=temperature)

    elif loss_type == "prototypical_v1":
        alpha = kwargs.get('alpha', 1.0)
        return PrototypicalContrastiveLoss_V1(temperature=temperature, alpha=alpha)

    elif loss_type == "prototypical_v2":
        beta = kwargs.get('beta', 0.5)
        return PrototypicalContrastiveLoss_V2(temperature=temperature, beta=beta)

    elif loss_type == "prototypical_v3":
        threshold = kwargs.get('threshold', 1.0)
        atypical_weight = kwargs.get('atypical_weight', 0.1)
        return PrototypicalContrastiveLoss_V3(
            temperature=temperature,
            threshold=threshold,
            atypical_weight=atypical_weight
        )

    else:
        raise ValueError(f"Unknown contrastive loss type: {loss_type}")
