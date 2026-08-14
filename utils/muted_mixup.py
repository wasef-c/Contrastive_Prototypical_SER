#!/usr/bin/env python3
"""Synthesise muted emotional samples by mixing toward neutral.

The per-sample diagnosis found that upweighting low-salience emotional
samples does nothing: sal_emo moved its target group from 0.487 to 0.491
error, i.e. not at all. Those samples are not underweighted, they are
underrepresented and unsupported by the representation. MSP-Podcast
contains 524 quiet angry utterances in total, 5.9 percent of the angry
class, so there is very little to learn the muted region from.

Reweighting cannot create data. This does: each emotional sample is
blended in embedding space toward a randomly chosen neutral sample, which
produces an example that sits closer to the neutral region while still
being an instance of its emotion. The blend is emotion-dominant, keeping
lambda in [0.5, 1], and the label stays the hard emotional one. That is
the claim the project is built on stated as training data: moving toward
neutral in feature space does not make an utterance neutral.

The placebo pairs each emotional sample with a different *emotional*
sample instead, matching the amount of mixing, the lambda distribution
and the extra gradient path exactly, while removing the one thing under
test, which is the direction of the blend. Without it a gain cannot be
told apart from mixup acting as a generic regulariser.

Mixing happens on post-fusion embeddings rather than waveforms because
the encoders are frozen, so embedding space is the only place a new
example can be made cheaply.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sample_lambda(n: int, alpha: float, device: torch.device) -> torch.Tensor:
    """Draw emotion-dominant mixing coefficients.

    Values are folded into [0.5, 1] so the emotional sample always
    dominates the blend. A blend where neutral dominated would be an
    utterance whose label is genuinely doubtful, which is a different
    and unwanted experiment.

    Args:
        n: number of coefficients.
        alpha: Beta(alpha, alpha) parameter.
        device: device for the returned tensor.

    Returns:
        [n] coefficients in [0.5, 1].
    """
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((n,)).to(device)
    return torch.maximum(lam, 1.0 - lam)


def symmetric_mixup_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    alpha: float = 2.0,
    within_class_control: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Blend in both directions so the decision boundary has no reason to move.

    The one-directional version synthesises only emotional-labelled
    samples, which adds emotional evidence and shifts the prior. Measured
    at weight 0.5 it drove neutral recall from 0.545 to 0.297 while
    emotional recall rose from 0.580 to 0.655: the mechanism teaches muted
    emotion very effectively and pays for it entirely out of neutral.

    Doing it symmetrically states the invariance instead of a preference:

        emotional blended toward neutral  -> still that emotion
        neutral blended toward emotional  -> still neutral

    Equal numbers of samples are added to both sides, so the prior is
    untouched and what remains is the claim that intensity does not
    determine the label.

    The control blends within class instead (emotional with emotional,
    neutral with neutral), adding the same volume to the same classes
    without ever crossing the intensity axis. That isolates crossing the
    axis from adding synthetic data, which the one-directional control
    could not do because it shifted the prior too.

    Args:
        embeddings: [B, hidden_dim] post-fusion embeddings.
        labels: [B] class labels, with 0 as neutral.
        output_layer: maps [B, hidden_dim] to [B, num_classes].
        alpha: Beta(alpha, alpha) parameter for the mixing coefficient.
        within_class_control: if True, pair each sample with one of its own
            class rather than across the neutral boundary.
        generator: optional RNG for reproducible pairing.

    Returns:
        (logits, targets) for the synthesised samples, or (None, None) when
        the batch lacks both a neutral and an emotional sample.
    """
    device = embeddings.device
    neu = torch.nonzero(labels == 0, as_tuple=False).flatten()
    emo = torch.nonzero(labels != 0, as_tuple=False).flatten()
    if neu.numel() == 0 or emo.numel() == 0:
        return None, None

    chunks, targets = [], []
    for src, partner_pool in ((emo, emo if within_class_control else neu),
                              (neu, neu if within_class_control else emo)):
        if src.numel() == 0 or partner_pool.numel() == 0:
            continue
        if within_class_control and partner_pool.numel() < 2:
            continue
        pick = torch.randint(partner_pool.numel(), (src.numel(),),
                             device=device, generator=generator)
        partner = partner_pool[pick]
        if within_class_control:
            clash = partner == src
            if clash.any():
                partner[clash] = partner_pool[
                    (pick[clash] + 1) % partner_pool.numel()]
        lam = sample_lambda(src.numel(), alpha, device).unsqueeze(1)
        chunks.append(lam * embeddings[src] + (1.0 - lam) * embeddings[partner])
        targets.append(labels[src])

    if not chunks:
        return None, None
    mixed = torch.cat(chunks, dim=0)
    return output_layer(mixed), torch.cat(targets, dim=0)


def muted_mixup_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    alpha: float = 2.0,
    shuffle_control: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Blend emotional embeddings toward neutral and classify the result.

    Args:
        embeddings: [B, hidden_dim] post-fusion embeddings.
        labels: [B] class labels, with 0 as neutral.
        output_layer: maps [B, hidden_dim] to [B, num_classes].
        alpha: Beta(alpha, alpha) parameter. Larger keeps lambda near 0.5,
            producing more strongly muted examples; 2.0 is a mild default.
        shuffle_control: if True, pair each emotional sample with another
            emotional sample instead of a neutral one. This is the placebo.
        generator: optional RNG for reproducible pairing.

    Returns:
        (logits, targets, lam) for the synthesised samples, or (None, None,
        None) when the batch lacks the pairing partners needed.
    """
    device = embeddings.device
    emo_idx = torch.nonzero(labels != 0, as_tuple=False).flatten()
    pool_idx = (torch.nonzero(labels != 0, as_tuple=False).flatten()
                if shuffle_control
                else torch.nonzero(labels == 0, as_tuple=False).flatten())

    # Need emotional samples to mute and partners to mute them toward. The
    # control additionally needs more than one emotional sample, otherwise
    # every sample would pair with itself and the blend would be a no-op.
    if emo_idx.numel() == 0 or pool_idx.numel() == 0:
        return None, None, None
    if shuffle_control and pool_idx.numel() < 2:
        return None, None, None

    pick = torch.randint(pool_idx.numel(), (emo_idx.numel(),),
                         device=device, generator=generator)
    partner = pool_idx[pick]
    if shuffle_control:
        # Avoid self-pairing, which would leave the embedding unchanged and
        # silently weaken the control.
        clash = partner == emo_idx
        if clash.any():
            partner[clash] = pool_idx[(pick[clash] + 1) % pool_idx.numel()]

    lam = sample_lambda(emo_idx.numel(), alpha, device).unsqueeze(1)
    mixed = lam * embeddings[emo_idx] + (1.0 - lam) * embeddings[partner]
    return output_layer(mixed), labels[emo_idx], lam.squeeze(1)


def muted_mixup_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy on the synthesised muted samples.

    The label is the emotional one, unblended, so the loss asserts that a
    muted instance of an emotion is still that emotion.

    Args:
        logits: [M, num_classes] logits for synthesised samples.
        targets: [M] emotional class labels.
        class_weights: optional [num_classes] inverse-frequency weights,
            applied exactly as in the primary loss so the synthesised
            samples do not reintroduce the class imbalance.

    Returns:
        Scalar loss.
    """
    return F.cross_entropy(logits, targets.long(), weight=class_weights)
