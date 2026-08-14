#!/usr/bin/env python3
"""
VADmix: cross-corpus feature-level mixup with VAD-derived soft targets.

Mixes post-fusion embeddings between samples from DIFFERENT training corpora,
and produces soft class targets by mapping the blended VAD coordinate through
the class prototype dictionary. The cross-corpus pairing forces the encoder to
learn features that interpolate across distribution shifts; the VAD-soft target
keeps the supervision signal aligned with the prototype geometry rather than
arbitrary one-hot indices.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def cross_corpus_permutation(
    corpora: List[str],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build a permutation that pairs each sample with a partner from a
    different corpus when possible.

    For sample i with corpus c_i, picks a partner j uniformly at random from
    {j : c_j != c_i}. If the batch is single-corpus, falls back to a random
    permutation that excludes self-pairing.

    Args:
        corpora: List of corpus name strings, length B.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        LongTensor of shape [B] with partner indices.
    """
    n = len(corpora)
    perm = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        candidates = [j for j in range(n) if corpora[j] != corpora[i]]
        if not candidates:
            candidates = [j for j in range(n) if j != i] or [i]
        idx = torch.randint(len(candidates), (1,), generator=generator).item()
        perm[i] = candidates[idx]
    return perm


def vad_to_soft_labels(
    vad: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Convert continuous VAD coordinates into a soft class distribution by
    distance to class prototypes.

    soft_label[c] = softmax_c(-||vad - prototype_c||_2 / temperature)

    Args:
        vad: [B, 3] tensor of (V, A, D) coordinates.
        prototypes: [C, 3] tensor of class prototype VAD coordinates.
        temperature: Softmax temperature. Lower -> sharper (closer to one-hot
            on nearest prototype). Higher -> flatter.

    Returns:
        [B, C] soft target distribution that sums to 1 along dim=1.
    """
    # Compute in fp32 for autocast safety: cdist and softmax with small
    # temperatures can produce NaN/Inf in fp16.
    vad_f = vad.float()
    proto_f = prototypes.float()
    distances = torch.cdist(vad_f, proto_f)  # [B, C]
    return F.softmax(-distances / max(temperature, 1e-6), dim=1)


def vadmix_step(
    embeddings: torch.Tensor,
    vad: torch.Tensor,
    corpora: List[str],
    prototypes: torch.Tensor,
    output_layer: torch.nn.Module,
    alpha: float = 0.2,
    temperature: float = 0.5,
    cross_corpus_only: bool = True,
) -> tuple:
    """Run one VADmix step: pair samples cross-corpus, blend embeddings, blend
    VAD targets, classify the blended embedding, build soft target.

    Args:
        embeddings: [B, hidden_dim] post-fusion embeddings.
        vad: [B, 3] VAD coordinates for each sample.
        corpora: List of corpus name strings, length B.
        prototypes: [C, 3] class prototype VAD coordinates.
        output_layer: nn.Module that maps [B, hidden_dim] -> [B, num_classes].
        alpha: Beta(alpha, alpha) parameter. 0.2 is the SimCLR-mixup default.
        temperature: VAD-to-soft-label softmax temperature.
        cross_corpus_only: If True, prefer cross-corpus pairing. If False, use
            a fully random permutation (vanilla mixup with VAD targets).

    Returns:
        mixed_logits: [B, num_classes]
        soft_targets: [B, num_classes]
        lam: float - the mixing coefficient drawn from Beta(alpha, alpha)
    """
    if cross_corpus_only:
        perm = cross_corpus_permutation(corpora).to(embeddings.device)
    else:
        perm = torch.randperm(len(corpora), device=embeddings.device)

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())

    mixed_emb = lam * embeddings + (1.0 - lam) * embeddings[perm]
    mixed_vad = lam * vad + (1.0 - lam) * vad[perm]

    mixed_logits = output_layer(mixed_emb)
    soft_targets = vad_to_soft_labels(mixed_vad, prototypes, temperature=temperature)

    return mixed_logits, soft_targets, lam


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy with soft targets. Optionally per-class weighted.

    Args:
        logits: [B, C]
        soft_targets: [B, C], rows sum to 1.
        class_weights: Optional [C] per-class weights (frequency-based).

    Returns:
        Scalar loss (mean over batch).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weights is not None:
        # Apply per-class weight to each component of the soft target
        weighted = soft_targets * class_weights.unsqueeze(0)
        loss = -(weighted * log_probs).sum(dim=-1)
        # Normalize by per-sample target weight so loss scale stays comparable
        norm = soft_targets.sum(dim=-1).clamp(min=1e-8)
        return (loss / norm).mean()
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def build_prototype_tensor(expected_vad: dict, num_classes: int) -> torch.Tensor:
    """Build a [num_classes, 3] CPU tensor from the expected_vad dict.

    Args:
        expected_vad: dict mapping label (int) -> [V, A, D] list.
        num_classes: total number of classes.

    Returns:
        [num_classes, 3] float32 tensor.
    """
    proto = torch.zeros(num_classes, 3, dtype=torch.float32)
    for c in range(num_classes):
        if c in expected_vad:
            proto[c] = torch.tensor(expected_vad[c], dtype=torch.float32)
        else:
            proto[c] = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    return proto
