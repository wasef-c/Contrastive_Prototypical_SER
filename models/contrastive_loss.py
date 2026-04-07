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


class PrototypeAnchoredLoss(nn.Module):
    """
    Prototype-Anchored Loss with VAD-Guided Centers (Novel)

    Instead of pairwise contrastive learning, this approach:
    1. Maintains learnable class prototypes in embedding space, initialized from VAD centroids
    2. Pulls each sample toward its class prototype with strength proportional to prototypicality
    3. Pushes class prototypes apart from each other
    4. Uses adaptive margins: prototypical samples must be close, atypical ones get slack

    The key insight: prototypicality (VAD distance) defines a continuous measure of how
    "canonical" a sample is. Canonical samples anchor the embedding space, creating
    domain-invariant class cores that generalize across corpora.

    Loss = L_anchor + lambda_sep * L_separation

    L_anchor = mean_i[ w_i * max(0, ||embed_i - proto_{y_i}||^2 - margin_i) ]
        where w_i = exp(-alpha * difficulty_i)   (prototypical → high weight)
              margin_i = margin_base + beta * difficulty_i  (atypical → relaxed margin)

    L_separation = sum_{c != c'} max(0, delta - ||proto_c - proto_{c'}||^2)
        pushes prototypes at least delta apart
    """

    def __init__(
        self,
        embedding_dim=128,
        num_classes=4,
        expected_vad=None,
        alpha=2.0,
        beta=0.5,
        margin_base=0.1,
        separation_margin=2.0,
        separation_weight=0.5,
        learnable_prototypes=True,
        use_adaptive_difficulty=False,
        use_hard_negative_mining=False,
        hard_negative_weight=0.3,
        use_memory_bank=False,
        bank_size=64,
        bank_momentum=0.5,
        bank_threshold=0.5,
    ):
        """
        Args:
            embedding_dim: dimension of projected embeddings
            num_classes: number of emotion classes
            expected_vad: dict {label: [v, a, d]} - used to initialize prototype relationships
            alpha: controls prototypicality weight decay (higher = more focus on prototypical)
            beta: controls margin scaling with difficulty (higher = more slack for atypical)
            margin_base: minimum margin for most prototypical samples
            separation_margin: minimum distance between class prototypes
            separation_weight: weight for prototype separation loss
            learnable_prototypes: if True, prototypes are updated via gradient descent
            use_adaptive_difficulty: if True, compute difficulty from embedding-space
                distances to learned prototypes instead of fixed VAD centroids.
            use_hard_negative_mining: if True, add prototypicality-weighted repulsion
                from wrong-class prototypes. Prototypical samples near wrong prototypes
                get stronger repulsion (informative negatives).
            hard_negative_weight: weight for hard negative mining loss term
            use_memory_bank: if True, maintain per-class FIFO queue of prototypical
                embeddings and blend with learnable prototypes
            bank_size: number of embeddings per class in memory bank
            bank_momentum: blend factor (1.0 = all learnable, 0.0 = all bank centroid)
            bank_threshold: only store embeddings with difficulty < threshold
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin_base = margin_base
        self.separation_margin = separation_margin
        self.separation_weight = separation_weight
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_adaptive_difficulty = use_adaptive_difficulty
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_negative_weight = hard_negative_weight
        self.use_memory_bank = use_memory_bank
        self.bank_size = bank_size
        self.bank_momentum = bank_momentum
        self.bank_threshold = bank_threshold

        # Initialize prototypes
        # We can't directly map 3D VAD to embedding_dim, but we can use VAD structure
        # to initialize prototypes with meaningful relative distances
        init_prototypes = self._init_from_vad(expected_vad, embedding_dim, num_classes)
        if learnable_prototypes:
            self.prototypes = nn.Parameter(init_prototypes)
        else:
            self.register_buffer('prototypes', init_prototypes)

        # Memory bank: per-class FIFO queue of prototypical embeddings
        if use_memory_bank:
            self.register_buffer('memory_bank', torch.zeros(num_classes, bank_size, embedding_dim))
            self.register_buffer('bank_ptr', torch.zeros(num_classes, dtype=torch.long))
            self.register_buffer('bank_filled', torch.zeros(num_classes, dtype=torch.bool))

        print(f"   PrototypeAnchoredLoss: {num_classes} prototypes in {embedding_dim}D")
        print(f"   alpha={alpha}, beta={beta}, margin_base={margin_base}")
        if use_adaptive_difficulty:
            print(f"   Adaptive difficulty: ON (embedding-space distances)")
        if use_hard_negative_mining:
            print(f"   Hard negative mining: ON (weight={hard_negative_weight})")
        if use_memory_bank:
            print(f"   Memory bank: ON (size={bank_size}, momentum={bank_momentum}, threshold={bank_threshold})")

    def _init_from_vad(self, expected_vad, embedding_dim, num_classes):
        """
        Initialize prototype embeddings preserving VAD-space distances.
        Projects 3D VAD centroids into embedding_dim space, then adds noise
        to fill remaining dimensions while preserving relative structure.
        """
        prototypes = torch.randn(num_classes, embedding_dim) * 0.01

        if expected_vad is not None:
            # Place first 3 dims according to VAD centroids (scaled up)
            for label in range(num_classes):
                if label in expected_vad:
                    vad = expected_vad[label]
                    # Scale VAD [0,1] to [-1, 1] range and amplify
                    prototypes[label, 0] = (vad[0] - 0.5) * 4  # valence
                    prototypes[label, 1] = (vad[1] - 0.5) * 4  # arousal
                    prototypes[label, 2] = (vad[2] - 0.5) * 4  # dominance

        # L2 normalize
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return prototypes

    def forward(self, embeddings, labels, difficulties):
        """
        Args:
            embeddings: [batch_size, embedding_dim] - L2 normalized projected embeddings
            labels: [batch_size] - class labels (0 to num_classes-1)
            difficulties: [batch_size] - prototypicality scores (VAD distance).
                          Ignored when use_adaptive_difficulty=True.

        Returns:
            loss: scalar
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Normalize prototypes (keep on unit sphere)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # [num_classes, embed_dim]

        # Memory bank: blend learnable prototypes with bank centroids
        if self.use_memory_bank and self.bank_filled.any():
            # Clone buffers to avoid in-place modification conflicts with autograd
            bank_snapshot = self.memory_bank.clone().detach()
            filled_snapshot = self.bank_filled.clone().detach()
            bank_centroids = bank_snapshot.mean(dim=1)  # [C, D]
            bank_centroids = F.normalize(bank_centroids, p=2, dim=1)
            # Blend: momentum * learnable + (1 - momentum) * bank centroid
            blended = self.bank_momentum * prototypes_norm + (1 - self.bank_momentum) * bank_centroids
            # Only blend for classes that have bank entries
            prototypes_norm = torch.where(
                filled_snapshot.unsqueeze(1).expand_as(prototypes_norm),
                F.normalize(blended, p=2, dim=1),
                prototypes_norm,
            )

        # Get each sample's class prototype
        class_prototypes = prototypes_norm[labels]  # [batch_size, embed_dim]

        # Compute squared distances to own class prototype
        distances_sq = ((embeddings - class_prototypes) ** 2).sum(dim=1)  # [batch_size]

        if self.use_adaptive_difficulty:
            # Adaptive: difficulty = distance to learned prototype in embedding space.
            # Detached so weights/margins don't backprop through the distance
            # (prevents degenerate solutions where model collapses all distances).
            # Normalized to [0, 1] range per batch for stable weighting.
            adaptive_diff = distances_sq.detach().sqrt()
            adaptive_diff = adaptive_diff / (adaptive_diff.max() + 1e-8)
            difficulties = adaptive_diff

        # Prototypicality-based weights: prototypical samples → high weight
        weights = torch.exp(-self.alpha * difficulties)

        # Adaptive margins: prototypical samples → tight margin, atypical → relaxed
        margins = self.margin_base + self.beta * difficulties

        # Anchor loss: weighted hinge on distance to own prototype
        anchor_loss = weights * F.relu(distances_sq - margins)
        L_anchor = anchor_loss.mean()

        # Prototype separation loss: push prototypes apart
        L_separation = torch.tensor(0.0, device=device)
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                dist_sq = ((prototypes_norm[i] - prototypes_norm[j]) ** 2).sum()
                # Hinge: penalize if prototypes are too close
                L_separation = L_separation + F.relu(self.separation_margin - dist_sq)

        # Hard negative mining: prototypicality-weighted repulsion from wrong prototypes
        L_hard_neg = torch.tensor(0.0, device=device)
        if self.use_hard_negative_mining:
            # Distance to ALL prototypes: [B, C]
            all_dist_sq = torch.cdist(embeddings, prototypes_norm).pow(2)

            # Mask out correct class (only repel from wrong prototypes)
            wrong_mask = torch.ones(batch_size, self.num_classes, device=device)
            wrong_mask.scatter_(1, labels.unsqueeze(1), 0)

            # Hinge: penalize if too close to wrong prototypes
            repulsion = F.relu(self.separation_margin - all_dist_sq) * wrong_mask  # [B, C]

            # Weight by prototypicality: prototypical samples get stronger repulsion
            # (they're informative negatives), atypical samples get weaker (noisy)
            weighted_repulsion = weights.unsqueeze(1) * repulsion  # [B, C]
            L_hard_neg = weighted_repulsion.sum(dim=1).mean()

        # Update memory bank with prototypical samples from this batch
        if self.use_memory_bank:
            with torch.no_grad():
                proto_mask = difficulties < self.bank_threshold
                for c in range(self.num_classes):
                    class_mask = (labels == c) & proto_mask
                    if class_mask.sum() > 0:
                        class_embeds = embeddings[class_mask].detach()
                        for emb in class_embeds:
                            ptr = self.bank_ptr[c].long()
                            self.memory_bank[c, ptr] = emb
                            self.bank_ptr[c] = (ptr + 1) % self.bank_size
                        self.bank_filled[c] = True

        loss = L_anchor + self.separation_weight * L_separation + self.hard_negative_weight * L_hard_neg
        return loss


class PrototypeDivergenceLoss(nn.Module):
    """
    Prototype Divergence Loss (Novel)

    Complementary to PrototypeAnchoredLoss - this pushes samples AWAY from
    wrong-class prototypes, scaled by how prototypical the sample is.

    A highly prototypical "happy" sample should be far from the "sad" prototype.
    An atypical sample (ambiguous emotion) gets less repulsion since it may genuinely
    share characteristics with multiple emotions.

    L_diverge = mean_i[ w_i * sum_{c != y_i} max(0, delta - ||embed_i - proto_c||^2) ]
    """

    def __init__(self, embedding_dim=128, num_classes=4, alpha=2.0, divergence_margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.divergence_margin = divergence_margin
        self.num_classes = num_classes

    def forward(self, embeddings, labels, difficulties, prototypes):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
            difficulties: [batch_size]
            prototypes: [num_classes, embedding_dim] - from PrototypeAnchoredLoss

        Returns:
            loss: scalar
        """
        batch_size = embeddings.shape[0]
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        # Weights
        weights = torch.exp(-self.alpha * difficulties)

        # Compute distance to ALL prototypes: [batch_size, num_classes]
        # embeddings: [B, D], prototypes: [C, D] → [B, C]
        all_distances_sq = torch.cdist(embeddings, prototypes_norm).pow(2)

        # Create mask for wrong classes
        wrong_class_mask = torch.ones(batch_size, self.num_classes, device=embeddings.device)
        wrong_class_mask.scatter_(1, labels.unsqueeze(1), 0)  # Zero out correct class

        # Repulsion: penalize if too close to wrong prototypes
        repulsion = F.relu(self.divergence_margin - all_distances_sq) * wrong_class_mask
        repulsion_per_sample = repulsion.sum(dim=1)  # [batch_size]

        loss = (weights * repulsion_per_sample).mean()
        return loss


class PrototypeAnchoredMultiDSLoss(nn.Module):
    """
    Prototype-Anchored Loss with Cross-Domain Alignment (Novel)

    Extends PrototypeAnchoredLoss for multi-corpus training by adding an explicit
    cross-domain alignment term. When a batch contains samples from multiple corpora,
    this loss ensures that same-class samples from different corpora converge to the
    same region in embedding space.

    Loss = L_anchor + w_sep * L_separation + w_align * L_cross_domain

    L_cross_domain: For each class, compute per-corpus centroids of embeddings in the batch.
    Then minimize the pairwise distance between centroids from different corpora.
    Weighted by prototypicality so that alignment is driven by canonical samples.

    This explicitly forces domain invariance: a prototypical "happy" from IEMO and
    a prototypical "happy" from MSPP are pulled toward each other, not just toward
    a shared prototype.
    """

    def __init__(
        self,
        embedding_dim=128,
        num_classes=4,
        expected_vad=None,
        alpha=2.0,
        beta=0.5,
        margin_base=0.1,
        separation_margin=2.0,
        separation_weight=0.5,
        alignment_weight=1.0,
        learnable_prototypes=True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin_base = margin_base
        self.separation_margin = separation_margin
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Initialize prototypes from VAD (reuse same logic)
        init_prototypes = self._init_from_vad(expected_vad, embedding_dim, num_classes)
        if learnable_prototypes:
            self.prototypes = nn.Parameter(init_prototypes)
        else:
            self.register_buffer('prototypes', init_prototypes)

        print(f"   PrototypeAnchoredMultiDSLoss: {num_classes} prototypes in {embedding_dim}D")
        print(f"   alpha={alpha}, beta={beta}, margin_base={margin_base}, alignment_weight={alignment_weight}")

    def _init_from_vad(self, expected_vad, embedding_dim, num_classes):
        prototypes = torch.randn(num_classes, embedding_dim) * 0.01
        if expected_vad is not None:
            for label in range(num_classes):
                if label in expected_vad:
                    vad = expected_vad[label]
                    prototypes[label, 0] = (vad[0] - 0.5) * 4
                    prototypes[label, 1] = (vad[1] - 0.5) * 4
                    prototypes[label, 2] = (vad[2] - 0.5) * 4
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return prototypes

    def forward(self, embeddings, labels, difficulties, dataset_ids=None):
        """
        Args:
            embeddings: [batch_size, embedding_dim] - L2 normalized
            labels: [batch_size] - class labels
            difficulties: [batch_size] - prototypicality scores
            dataset_ids: [batch_size] - integer corpus IDs (0, 1, 2, ...)
                         If None, falls back to standard prototype-anchored loss.

        Returns:
            loss: scalar
        """
        device = embeddings.device

        # Normalize prototypes
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

        # === L_anchor (same as PrototypeAnchoredLoss) ===
        class_prototypes = prototypes_norm[labels]
        distances_sq = ((embeddings - class_prototypes) ** 2).sum(dim=1)
        weights = torch.exp(-self.alpha * difficulties)
        margins = self.margin_base + self.beta * difficulties
        anchor_loss = weights * F.relu(distances_sq - margins)
        L_anchor = anchor_loss.mean()

        # === L_separation (same as PrototypeAnchoredLoss) ===
        L_separation = torch.tensor(0.0, device=device)
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                dist_sq = ((prototypes_norm[i] - prototypes_norm[j]) ** 2).sum()
                L_separation = L_separation + F.relu(self.separation_margin - dist_sq)

        # === L_cross_domain (NEW: explicit cross-corpus alignment) ===
        L_cross_domain = torch.tensor(0.0, device=device)

        if dataset_ids is not None:
            unique_classes = labels.unique()
            unique_domains = dataset_ids.unique()

            # Only compute if we have multiple domains in the batch
            if len(unique_domains) > 1:
                for c in unique_classes:
                    # Collect prototypicality-weighted centroids per domain for this class
                    class_mask = (labels == c)
                    domain_centroids = []

                    for d in unique_domains:
                        domain_class_mask = class_mask & (dataset_ids == d)
                        if domain_class_mask.sum() < 2:
                            continue  # Skip if too few samples

                        # Get embeddings and weights for this class+domain
                        dc_embeddings = embeddings[domain_class_mask]
                        dc_weights = weights[domain_class_mask]

                        # Weighted centroid: prototypical samples contribute more
                        dc_weights_norm = dc_weights / (dc_weights.sum() + 1e-8)
                        centroid = (dc_embeddings * dc_weights_norm.unsqueeze(1)).sum(dim=0)
                        domain_centroids.append(centroid)

                    # Pull all domain centroids for this class together
                    if len(domain_centroids) >= 2:
                        for i in range(len(domain_centroids)):
                            for j in range(i + 1, len(domain_centroids)):
                                L_cross_domain = L_cross_domain + ((domain_centroids[i] - domain_centroids[j]) ** 2).sum()

        loss = L_anchor + self.separation_weight * L_separation + self.alignment_weight * L_cross_domain
        return loss


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

    elif loss_type == "prototype_anchored":
        return PrototypeAnchoredLoss(
            embedding_dim=kwargs.get('projection_dim', 128),
            num_classes=kwargs.get('num_classes', 4),
            expected_vad=kwargs.get('expected_vad', None),
            alpha=kwargs.get('alpha', 2.0),
            beta=kwargs.get('beta', 0.5),
            margin_base=kwargs.get('margin_base', 0.1),
            separation_margin=kwargs.get('separation_margin', 2.0),
            separation_weight=kwargs.get('separation_weight', 0.5),
            use_adaptive_difficulty=kwargs.get('use_adaptive_difficulty', False),
            use_hard_negative_mining=kwargs.get('use_hard_negative_mining', False),
            hard_negative_weight=kwargs.get('hard_negative_weight', 0.3),
            use_memory_bank=kwargs.get('use_memory_bank', False),
            bank_size=kwargs.get('bank_size', 64),
            bank_momentum=kwargs.get('bank_momentum', 0.5),
            bank_threshold=kwargs.get('bank_threshold', 0.5),
        )

    elif loss_type == "prototype_anchored_multiDS":
        return PrototypeAnchoredMultiDSLoss(
            embedding_dim=kwargs.get('projection_dim', 128),
            num_classes=kwargs.get('num_classes', 4),
            expected_vad=kwargs.get('expected_vad', None),
            alpha=kwargs.get('alpha', 2.0),
            beta=kwargs.get('beta', 0.5),
            margin_base=kwargs.get('margin_base', 0.1),
            separation_margin=kwargs.get('separation_margin', 2.0),
            separation_weight=kwargs.get('separation_weight', 0.5),
            alignment_weight=kwargs.get('alignment_weight', 1.0),
        )

    else:
        raise ValueError(f"Unknown contrastive loss type: {loss_type}")
