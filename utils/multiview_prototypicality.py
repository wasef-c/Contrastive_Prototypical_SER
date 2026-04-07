#!/usr/bin/env python3
"""
Multi-View Prototypicality: combines VAD distance, cross-modal agreement,
and embedding-space distance into a unified difficulty score.
"""

import torch
import torch.nn.functional as F


def compute_vad_difficulty(batch, expected_vad, device):
    """
    Compute prototypicality as Euclidean distance in VAD space (normalized to [0,1]).

    Args:
        batch: dict with 'valence', 'arousal', 'dominance', 'label' tensors
        expected_vad: dict mapping label -> [V, A, D] prototype
        device: torch device

    Returns:
        [B] tensor of difficulty scores in [0, 1], or None if VAD data missing
    """
    if 'valence' not in batch or 'arousal' not in batch or 'dominance' not in batch:
        return None

    labels = batch['label']
    B = labels.shape[0]

    valence = batch['valence'].float()
    arousal = batch['arousal'].float()
    dominance = batch['dominance'].float()
    actual_vad = torch.stack([valence, arousal, dominance], dim=1)  # [B, 3]

    # Build expected VAD tensor for each sample's class
    expected = torch.zeros(B, 3)
    for i in range(B):
        label = labels[i].item()
        if label in expected_vad:
            vad = expected_vad[label]
            expected[i] = torch.tensor(vad, dtype=torch.float32)

    expected = expected.to(device)
    actual_vad = actual_vad.to(device)

    # Euclidean distance
    distances = ((actual_vad - expected) ** 2).sum(dim=1).sqrt()  # [B]

    # Normalize to [0, 1] per batch
    d_max = distances.max()
    if d_max > 1e-8:
        distances = distances / d_max

    return distances


def compute_crossmodal_agreement(modal_features, device):
    """
    Compute cross-modal disagreement as difficulty.
    High cosine similarity between audio and text = agreement = prototypical.
    Low similarity = disagreement = atypical.

    Args:
        modal_features: dict with 'audio' [B, D_a] and 'text' [B, D_t] tensors,
                        or None if unimodal
        device: torch device

    Returns:
        [B] tensor of difficulty scores in [0, 1], or None if unimodal
    """
    if modal_features is None:
        return None
    if 'audio' not in modal_features or 'text' not in modal_features:
        return None

    audio = modal_features['audio'].to(device)  # [B, D_a]
    text = modal_features['text'].to(device)     # [B, D_t]

    # Project to same dimensionality if needed (use the smaller dim)
    # For cosine similarity we need same dim — use a simple linear projection
    # But to keep this parameter-free, we truncate to min dim
    min_dim = min(audio.shape[1], text.shape[1])
    audio = audio[:, :min_dim]
    text = text[:, :min_dim]

    # L2 normalize
    audio_norm = F.normalize(audio, p=2, dim=1)
    text_norm = F.normalize(text, p=2, dim=1)

    # Cosine similarity per sample
    cos_sim = (audio_norm * text_norm).sum(dim=1)  # [B] in [-1, 1]

    # Convert to difficulty: (1 - cos_sim) / 2 maps [-1,1] -> [0,1]
    # High agreement (cos_sim=1) -> difficulty=0 (prototypical)
    # Low agreement (cos_sim=-1) -> difficulty=1 (atypical)
    difficulty = (1.0 - cos_sim) / 2.0

    return difficulty


def compute_embedding_difficulty(embeddings_norm, labels, prototypes, device):
    """
    Compute difficulty as distance to learned class prototype in embedding space.
    Detached from gradient to prevent degenerate collapse.

    Args:
        embeddings_norm: [B, D] L2-normalized embeddings
        labels: [B] class labels
        prototypes: [C, D] prototype parameters (will be detached and normalized)
        device: torch device

    Returns:
        [B] tensor of difficulty scores in [0, 1]
    """
    # Detach and normalize prototypes
    proto_norm = F.normalize(prototypes.detach(), p=2, dim=1)  # [C, D]

    # Get each sample's class prototype
    class_protos = proto_norm[labels]  # [B, D]

    # Squared distance to own prototype
    distances = ((embeddings_norm.detach() - class_protos) ** 2).sum(dim=1).sqrt()  # [B]

    # Normalize to [0, 1]
    d_max = distances.max()
    if d_max > 1e-8:
        distances = distances / d_max

    return distances


def compute_multiview_difficulty(
    batch, expected_vad, modal_features, embeddings_norm, labels,
    prototypes, device, w_vad=0.4, w_cross=0.3, w_embed=0.3
):
    """
    Combine multiple views into a single difficulty score.
    Missing views are skipped with weight renormalization.

    Args:
        batch: dict with VAD fields and labels
        expected_vad: dict mapping label -> [V, A, D]
        modal_features: dict with 'audio'/'text' tensors, or None
        embeddings_norm: [B, D] normalized embeddings
        labels: [B] class labels
        prototypes: [C, D] prototype parameters
        device: torch device
        w_vad: weight for VAD view
        w_cross: weight for cross-modal view
        w_embed: weight for embedding-space view

    Returns:
        [B] tensor of combined difficulty scores in [0, 1]
    """
    views = []
    weights = []

    # VAD view
    d_vad = compute_vad_difficulty(batch, expected_vad, device)
    if d_vad is not None:
        views.append(d_vad)
        weights.append(w_vad)

    # Cross-modal agreement view
    d_cross = compute_crossmodal_agreement(modal_features, device)
    if d_cross is not None:
        views.append(d_cross)
        weights.append(w_cross)

    # Embedding-space view
    if embeddings_norm is not None and prototypes is not None:
        d_embed = compute_embedding_difficulty(embeddings_norm, labels, prototypes, device)
        views.append(d_embed)
        weights.append(w_embed)

    if len(views) == 0:
        # Fallback: uniform difficulty
        return torch.zeros(labels.shape[0], device=device)

    # Renormalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Weighted sum
    combined = torch.zeros_like(views[0])
    for view, w in zip(views, weights):
        combined = combined + w * view

    return combined
