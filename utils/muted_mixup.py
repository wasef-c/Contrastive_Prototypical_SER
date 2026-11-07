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


def vad_calibrated_lambda(
    vad: torch.Tensor,
    neutral_centre: torch.Tensor,
    target_quantile: float,
    shuffle_control: bool = False,
    target_span: Optional[float] = None,
    floor: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Choose each blend coefficient from the sample's measured intensity.

    Random lambda mutes every sample by an arbitrary amount, so a already
    quiet utterance can be muted as hard as a shouted one and the synthetic
    points land wherever the draw puts them. Here intensity is measured as
    distance from the neutral centre in VAD space, d_i, and blending toward a
    neutral partner moves a sample roughly linearly toward that centre, so a
    blend at lambda lands near intensity lambda * d_i. Solving for a target
    intensity t gives lambda_i = t / d_i: loud samples are muted hard, already
    quiet ones are barely touched. Targets are drawn from the low-intensity
    end, which is the region the training set barely covers, so abundant
    high-intensity samples are converted into scarce low-intensity ones.

    The control permutes the finished coefficients across samples. That keeps
    the marginal distribution of lambda, the number of synthetic samples and
    the gradient path identical, and removes only the correspondence between a
    sample's real intensity and how far it is muted, which is the thing under
    test.

    Args:
        vad: [B, 3] valence/arousal/dominance for the samples being muted.
        neutral_centre: [3] VAD centre of the neutral class.
        target_quantile: upper bound of the target intensity band, as a
            quantile of the batch's own intensity spread. Only used when
            target_span is None. Lower pushes further into the quiet region.
        target_span: fixed upper bound of the target band, precomputed over
            the training split. Preferred over target_quantile, which is
            noisy at realistic batch sizes.
        shuffle_control: permute the coefficients across samples, destroying
            the intensity correspondence while matching everything else.
        floor: smallest allowed lambda, so a very loud sample cannot be
            blended into something indistinguishable from neutral.
        generator: optional RNG for reproducible draws.

    Returns:
        [B] blend coefficients in [floor, 1].
    """
    distance = torch.linalg.norm(vad - neutral_centre.unsqueeze(0), dim=1)
    # A sample sitting on the neutral centre has no intensity to remove; the
    # clamp keeps its coefficient finite and it ends up close to unchanged.
    distance = distance.clamp_min(1e-6)

    if target_span is not None:
        # Fixed span computed once over the training split. A batch holds only
        # a handful of emotional samples, so a batch-local quantile is a noisy
        # target that drifts with batch composition.
        span = torch.as_tensor(target_span, device=vad.device, dtype=distance.dtype)
    elif distance.numel() > 1:
        span = torch.quantile(distance, float(target_quantile))
    else:
        span = distance.mean()
    targets = torch.rand(distance.shape, device=vad.device,
                         generator=generator) * span

    lam = (targets / distance).clamp(floor, 1.0)
    if shuffle_control:
        perm = torch.randperm(lam.numel(), device=lam.device, generator=generator)
        lam = lam[perm]
    return lam


def sparse_fill_mixup_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    vad: torch.Tensor,
    neutral_centre: torch.Tensor,
    class_direction: torch.Tensor,
    class_target: torch.Tensor,
    shuffle_control: bool = False,
    floor: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Synthesise samples into whichever intensity region its class lacks.

    Muted mixup moves every emotional sample toward neutral, but the training
    set is only thin at the quiet end for one class. On MSP-Podcast the
    emotional samples split across intensity quartiles as

        happy  6966 / 6162 / 5554 / 4379   (thin when loud)
        sad    3000 / 2622 / 1827 / 1198   (thin when loud)
        angry  1090 / 2287 / 3698 / 5492   (thin when quiet)

    so muting uniformly fills a real gap for angry, adds to an already dense
    region for happy, and for sad pushes samples toward the neutral boundary
    where sad and neutral are hardest to separate. Here each class moves
    toward its own sparse end instead: quiet classes get amplified, loud
    classes get muted. The claim under test is unchanged, that intensity does
    not determine identity, but it is applied where the data is missing.

    Muting blends toward a neutral sample. Amplifying blends toward a
    high-intensity sample of the SAME class, so the label stays unambiguous in
    both directions.

    Args:
        embeddings: [B, hidden_dim] post-fusion embeddings.
        labels: [B] class labels, 0 is neutral.
        output_layer: maps embeddings to logits.
        vad: [B, 3] valence/arousal/dominance.
        neutral_centre: [3] VAD centre of the neutral class.
        class_direction: [num_classes] with -1 to mute, +1 to amplify, 0 to
            skip. Entry 0 (neutral) is ignored.
        class_target: [num_classes] target intensity per class.
        shuffle_control: permute the blend coefficients across samples,
            matching their distribution while removing the correspondence
            with measured intensity.
        floor: smallest allowed blend coefficient.
        generator: optional RNG.

    Returns:
        (logits, targets, lam), or (None, None, None) when no eligible pairs.
    """
    device = embeddings.device
    distance = torch.linalg.norm(
        vad - neutral_centre.unsqueeze(0), dim=1).clamp_min(1e-6)

    src_list, partner_list, lam_list = [], [], []
    neutral_pool = torch.nonzero(labels == 0, as_tuple=False).flatten()

    for cls in torch.unique(labels):
        cls_i = int(cls)
        if cls_i == 0 or class_direction[cls_i] == 0:
            continue
        idx = torch.nonzero(labels == cls_i, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        if class_direction[cls_i] < 0:
            # Mute: blend toward neutral, target below current intensity.
            if neutral_pool.numel() == 0:
                continue
            pool = neutral_pool
            pick = torch.randint(pool.numel(), (idx.numel(),), device=device,
                                 generator=generator)
            partner = pool[pick]
            target = torch.rand(idx.shape, device=device,
                                generator=generator) * class_target[cls_i]
            lam = (target / distance[idx]).clamp(floor, 1.0)
        else:
            # Amplify: blend toward the loudest members of the same class.
            # Needs at least two so a sample is not paired with itself.
            if idx.numel() < 2:
                continue
            order = torch.argsort(distance[idx], descending=True)
            loud = idx[order[:max(1, idx.numel() // 2)]]
            pick = torch.randint(loud.numel(), (idx.numel(),), device=device,
                                 generator=generator)
            partner = loud[pick]
            clash = partner == idx
            if clash.any():
                partner[clash] = loud[(pick[clash] + 1) % loud.numel()]
            # Resulting intensity is roughly lam*d_i + (1-lam)*d_partner, so
            # aim past the current value toward the class target.
            d_i, d_p = distance[idx], distance[partner]
            target = class_target[cls_i].expand_as(d_i)
            denom = (d_p - d_i)
            lam = torch.where(denom.abs() < 1e-6,
                              torch.full_like(d_i, 0.5),
                              (d_p - target) / denom).clamp(floor, 1.0)

        src_list.append(idx)
        partner_list.append(partner)
        lam_list.append(lam)

    if not src_list:
        return None, None, None

    src = torch.cat(src_list)
    partner = torch.cat(partner_list)
    lam = torch.cat(lam_list)
    if shuffle_control:
        lam = lam[torch.randperm(lam.numel(), device=device, generator=generator)]

    lam = lam.unsqueeze(1)
    mixed = lam * embeddings[src] + (1.0 - lam) * embeddings[partner]
    return output_layer(mixed), labels[src], lam.squeeze(1)


def vad_gated_symmetric_mixup(
    audio: torch.Tensor,
    text: torch.Tensor,
    labels: torch.Tensor,
    vad: torch.Tensor,
    neutral_centre: torch.Tensor,
    mix_audio: bool = True,
    mix_text: bool = True,
    gate_quantile: float = 0.5,
    shuffle_control: bool = False,
    floor: float = 0.05,
    generator: Optional[torch.Generator] = None,
):
    """Blend both ways, but only for samples whose VAD says they can afford it.

    Two problems with the earlier arms are addressed at once.

    The one-directional version synthesised only emotional samples, which
    shifted the prior and cost neutral recall roughly four times what it
    bought. The label-symmetric version fixed the prior but transformed every
    sample regardless of where it sat, and lost accuracy monotonically with
    weight. Here direction is decided per sample from measured intensity:
    only emotional samples that are LOUD for their class get muted, and only
    neutral samples that are clearly central get an emotion blended in. Both
    sides gain samples, so the prior is untouched, and samples near the
    boundary, whose labels are already doubtful, are left alone.

    Mixing can be restricted to one modality. Intensity is prosodic, so
    attenuating the fused embedding also attenuates lexical evidence, which is
    not what the hypothesis claims: the same angry words spoken flatly are
    still angry. Audio-only mixing states that claim exactly; text-only is its
    mirror-image control.

    Args:
        audio: [B, D_a] pooled audio features.
        text: [B, D_t] text features.
        labels: [B] class labels, 0 is neutral.
        vad: [B, 3] valence/arousal/dominance.
        neutral_centre: [3] VAD centre of the neutral class.
        mix_audio: blend the audio stream.
        mix_text: blend the text stream.
        gate_quantile: per-class intensity quantile above which an emotional
            sample is eligible for muting, and below which a neutral sample is
            eligible for the reverse.
        shuffle_control: permute the coefficients across samples, matching
            their distribution while removing the intensity correspondence.
        floor: smallest allowed blend coefficient.
        generator: optional RNG.

    Returns:
        (mixed_audio, mixed_text, targets, lam), or (None, None, None, None)
        when no eligible pairs exist in the batch.
    """
    device = audio.device
    distance = torch.linalg.norm(
        vad - neutral_centre.unsqueeze(0), dim=1).clamp_min(1e-6)

    emo_all = torch.nonzero(labels != 0, as_tuple=False).flatten()
    neu_all = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if emo_all.numel() == 0 or neu_all.numel() == 0:
        return None, None, None, None

    src_list, partner_list, lam_list = [], [], []

    # Loud emotional samples blend toward neutral, keeping their emotion.
    for cls in torch.unique(labels[labels != 0]):
        idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        d_cls = distance[idx]
        thresh = (torch.quantile(d_cls, gate_quantile) if d_cls.numel() > 1
                  else d_cls.mean())
        loud = idx[d_cls >= thresh]
        if loud.numel() == 0:
            continue
        pick = torch.randint(neu_all.numel(), (loud.numel(),), device=device,
                             generator=generator)
        # Aim at the quiet end of this class rather than at neutral itself.
        target = torch.rand(loud.shape, device=device, generator=generator) * thresh
        src_list.append(loud)
        partner_list.append(neu_all[pick])
        lam_list.append((target / distance[loud]).clamp(floor, 1.0))

    # Clearly-central neutral samples take on a little emotion, staying neutral.
    d_neu = distance[neu_all]
    if d_neu.numel() > 1:
        neu_thresh = torch.quantile(d_neu, 1.0 - gate_quantile)
        central = neu_all[d_neu <= neu_thresh]
        if central.numel() > 0:
            pick = torch.randint(emo_all.numel(), (central.numel(),),
                                 device=device, generator=generator)
            # Neutral must stay dominant, otherwise the label is a lie.
            lam = 1.0 - torch.rand(central.shape, device=device,
                                   generator=generator) * 0.5
            src_list.append(central)
            partner_list.append(emo_all[pick])
            lam_list.append(lam.clamp(0.5, 1.0))

    if not src_list:
        return None, None, None, None

    src = torch.cat(src_list)
    partner = torch.cat(partner_list)
    lam = torch.cat(lam_list)
    if shuffle_control:
        lam = lam[torch.randperm(lam.numel(), device=device, generator=generator)]
    lam = lam.unsqueeze(1)

    mixed_audio = (lam * audio[src] + (1.0 - lam) * audio[partner]
                   if mix_audio else audio[src])
    mixed_text = (lam * text[src] + (1.0 - lam) * text[partner]
                  if mix_text else text[src])
    return mixed_audio, mixed_text, labels[src], lam.squeeze(1)


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
    vad: Optional[torch.Tensor] = None,
    neutral_centre: Optional[torch.Tensor] = None,
    vad_target_quantile: float = 0.5,
    vad_shuffle: bool = False,
    vad_target_span: Optional[float] = None,
    derive_labels: bool = False,
    proto_means: Optional[torch.Tensor] = None,
    proto_inv_covs: Optional[torch.Tensor] = None,
    derive_min_posterior: float = 0.0,
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
        vad: optional [B, 3] VAD for the batch. When given together with
            neutral_centre, lambda is calibrated from measured intensity
            instead of drawn from Beta, and alpha is ignored.
        neutral_centre: optional [3] VAD centre of the neutral class.
        vad_target_quantile: target intensity band for the calibrated blend.
        vad_shuffle: permute the calibrated coefficients across samples. The
            control for the calibrated variant.
        vad_target_span: fixed target-intensity bound from the training
            split; overrides vad_target_quantile when given.

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

    if vad is not None and neutral_centre is not None:
        # Intensity-calibrated blend. Deliberately not folded into [0.5, 1]:
        # the point is to reach the quiet region, which needs the neutral
        # partner to dominate for the loudest samples.
        lam = vad_calibrated_lambda(
            vad[emo_idx], neutral_centre,
            target_quantile=vad_target_quantile,
            shuffle_control=vad_shuffle,
            target_span=vad_target_span,
            generator=generator,
        ).unsqueeze(1)
    else:
        lam = sample_lambda(emo_idx.numel(), alpha, device).unsqueeze(1)
    mixed = lam * embeddings[emo_idx] + (1.0 - lam) * embeddings[partner]

    # Label the blend by where it LANDS rather than where it came from. See
    # vad_derived_labels for why: inheriting the parent's label puts an
    # emotional target on a neutral-region point, which is what collapses
    # neutral recall in the inherited-label arms.
    if derive_labels:
        if vad is None or proto_means is None or proto_inv_covs is None:
            raise ValueError(
                "derive_labels needs vad, proto_means and proto_inv_covs; "
                "without the fitted class statistics the blend cannot be "
                "placed in VAD space.")
        targets, keep = vad_derived_labels(
            vad[emo_idx], vad[partner], lam, proto_means, proto_inv_covs,
            min_posterior=derive_min_posterior)
        if not bool(keep.any()):
            return None, None, None
        mixed, targets = mixed[keep], targets[keep]
        return output_layer(mixed), targets, lam.squeeze(1)[keep]

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


def consistent_muted_mixup(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    vad: torch.Tensor,
    classify_fn,
    gate_quantile: float = 0.5,
    floor: float = 0.05,
    shuffle_control: bool = False,
    generator: Optional[torch.Generator] = None,
):
    """Blend both ways and report where the blend landed in VAD space.

    Muted mixup and the auxiliary prototypicality head currently contradict
    each other. The mixup moves an embedding toward a neutral partner and
    keeps the hard emotional label, while the auxiliary head is asked for the
    prototypicality of the ORIGINAL sample, so its target describes a point
    the embedding has left. Interpolating the VAD by the same coefficient
    removes the contradiction and states the thesis as two supervisions that
    agree: the category is invariant under attenuation, the position within
    the category is not.

    Blending is bidirectional and gated, so loud emotional samples move toward
    neutral, central neutral samples take on a little emotion, and both sides
    gain in proportion, leaving the prior where it was.

    Args:
        embeddings: [B, H] post-fusion embeddings.
        labels: [B] class labels, 0 is neutral.
        vad: [B, 3] valence/arousal/dominance.
        classify_fn: maps [M, H] to [M, num_classes].
        gate_quantile: per-class intensity quantile deciding eligibility.
        floor: smallest blend coefficient.
        shuffle_control: permute the coefficients, matching their marginal
            while removing the correspondence with measured intensity.
        generator: optional RNG.

    Returns:
        (logits, labels, mixed_vad, lam, mixed_emb) for the synthesised
        samples, or five Nones when the batch has no eligible pairs.
        mixed_vad is where each synthetic sample landed, so a caller can
        build a consistent auxiliary target, and mixed_emb is the blended
        embedding the auxiliary head should be applied to.
    """
    device = embeddings.device
    emo_all = torch.nonzero(labels != 0, as_tuple=False).flatten()
    neu_all = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if emo_all.numel() == 0 or neu_all.numel() == 0:
        return None, None, None, None, None

    # Intensity here is distance from the neutral class mean in the batch,
    # which needs no fitted constant and keeps the gate self-contained.
    centre = vad[neu_all].mean(dim=0)
    distance = torch.linalg.norm(vad - centre.unsqueeze(0), dim=1).clamp_min(1e-6)

    src_list, partner_list, lam_list = [], [], []
    for cls in torch.unique(labels[labels != 0]):
        idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        d_cls = distance[idx]
        thresh = (torch.quantile(d_cls, gate_quantile) if d_cls.numel() > 1
                  else d_cls.mean())
        loud = idx[d_cls >= thresh]
        if loud.numel() == 0:
            continue
        pick = torch.randint(neu_all.numel(), (loud.numel(),), device=device,
                             generator=generator)
        target = torch.rand(loud.shape, device=device, generator=generator) * thresh
        src_list.append(loud)
        partner_list.append(neu_all[pick])
        lam_list.append((target / distance[loud]).clamp(floor, 1.0))

    d_neu = distance[neu_all]
    if d_neu.numel() > 1:
        central = neu_all[d_neu <= torch.quantile(d_neu, 1.0 - gate_quantile)]
        if central.numel() > 0:
            pick = torch.randint(emo_all.numel(), (central.numel(),),
                                 device=device, generator=generator)
            # Neutral stays dominant, otherwise the preserved label is a lie.
            lam = 1.0 - torch.rand(central.shape, device=device,
                                   generator=generator) * 0.5
            src_list.append(central)
            partner_list.append(emo_all[pick])
            lam_list.append(lam.clamp(0.5, 1.0))

    if not src_list:
        return None, None, None, None, None

    src = torch.cat(src_list)
    partner = torch.cat(partner_list)
    lam = torch.cat(lam_list)
    if shuffle_control:
        lam = lam[torch.randperm(lam.numel(), device=device, generator=generator)]
    lam = lam.unsqueeze(1)

    mixed_emb = lam * embeddings[src] + (1.0 - lam) * embeddings[partner]
    # The same coefficient applied to VAD, which is what makes the auxiliary
    # target describe the synthetic point rather than the one it came from.
    mixed_vad = lam * vad[src] + (1.0 - lam) * vad[partner]
    return classify_fn(mixed_emb), labels[src], mixed_vad, lam.squeeze(1), mixed_emb


def vad_derived_labels(
    vad_src: torch.Tensor,
    vad_partner: torch.Tensor,
    lam: torch.Tensor,
    means: torch.Tensor,
    inv_covs: torch.Tensor,
    min_posterior: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Label a blended sample by where its blended VAD lands, not by its parent.

    The existing muted mixup keeps the parent's hard emotional label on a
    point blended toward neutral. Measured on IEMOCAP that is what destroys
    neutral: recall falls 54.3 -> 36.6 -> 27.4 as the amount of blending
    rises, and the mass moves into happy and sad, because the model is being
    told that neutral-region points carry emotional labels.

    Here the label follows the geometry instead. The blend's VAD is the same
    convex combination as the blend's embedding, which is exact rather than
    assumed: the auxiliary head is a bare nn.Linear (hidden_dim 0), and a
    linear map of a convex combination is the convex combination of the maps.
    A blend whose VAD lands in neutral's region therefore becomes neutral,
    and no synthetic point is ever mislabelled by construction.

    The residual risk is that neutral and sad are the closest pair in VAD
    (0.165, the smallest of the six, and the ratio to the mean pair distance
    is 0.52 under every affine transform tried), so blends near that boundary
    get noisy labels. min_posterior discards them rather than guessing.

    Args:
        vad_src: [M, 3] VAD of the source samples.
        vad_partner: [M, 3] VAD of the blend partners.
        lam: [M] or [M, 1] blend coefficients on the source.
        means: [C, 3] class VAD centres fitted on the training split.
        inv_covs: [C, 3, 3] per-class inverse covariances.
        min_posterior: discard blends whose top posterior falls below this.
            0.0 keeps everything.

    Returns:
        Tuple of the derived labels [M] and a keep mask [M].
    """
    if lam.dim() == 2:
        lam = lam.squeeze(1)
    lam_c = lam.unsqueeze(1)
    vad_mix = lam_c * vad_src + (1.0 - lam_c) * vad_partner

    scores = []
    for c in range(means.shape[0]):
        delta = vad_mix - means[c].unsqueeze(0)
        quad = torch.einsum('ni,ij,nj->n', delta, inv_covs[c], delta)
        sign, logdet = torch.linalg.slogdet(inv_covs[c])
        scores.append(-0.5 * quad + 0.5 * logdet)
    scores = torch.stack(scores, dim=1)
    post = torch.softmax(scores, dim=1)
    conf, derived = post.max(dim=1)
    keep = conf >= float(min_posterior)
    return derived.long(), keep


def gaussian_posterior(points: torch.Tensor, means: torch.Tensor,
                       inv_covs: torch.Tensor,
                       log_priors: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Class posterior at each point under fitted class-conditional Gaussians.

    Priors default to uniform, matching fit_class_gaussians in the
    diagnostic and matching GenLabel, whose label is a plain normalised
    likelihood with no prior term.

    Uniform priors are not a neutral choice here. Neutral is roughly 45
    percent of MSP-Podcast, so dropping the prior systematically discounts
    it: a neutral sample's target averages only 0.407 on neutral. That is
    the same direction as the entire measured effect, every row of the
    confusion matrix losing neutral mass, which makes "this is implicit
    class rebalancing" a live competing explanation. Passing empirical
    log priors here is the control that separates the two.

    Args:
        points: [N, D] coordinates to score.
        means: [C, D] class centres.
        inv_covs: [C, D, D] per-class inverse covariances.
        log_priors: [C] log class priors, or None for uniform.

    Returns:
        [N, C] posterior, rows summing to one.
    """
    scores = []
    for c in range(means.shape[0]):
        delta = points - means[c].unsqueeze(0)
        quad = torch.einsum('ni,ij,nj->n', delta, inv_covs[c], delta)
        _, logdet = torch.linalg.slogdet(inv_covs[c])
        scores.append(-0.5 * quad + 0.5 * logdet)
    stacked = torch.stack(scores, dim=1)
    if log_priors is not None:
        stacked = stacked + log_priors.unsqueeze(0)
    return torch.softmax(stacked, dim=1)


def solve_calibrated_prior(
    vad: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    inv_covs: torch.Tensor,
    class_prior: torch.Tensor,
    alpha: float = 2.0,
    fold_lambda: bool = True,
    n_samples: int = 120000,
    iters: int = 60,
    seed: int = 0,
) -> torch.Tensor:
    """Log-prior offset that makes the BLEND marginal match the class prior.

    Uniform priors are the standard generative-classifier choice, but they
    are not calibrated to what training actually sees. Measured on
    MSP-Podcast, the target marginal over blends injects neutral at 0.340
    against a true 0.453 and sad at 0.195 against 0.107, and that skew is
    the neutral-recall loss at its origin. Empirical class priors overshoot
    the other way (neutral 0.510) and disagree with the uniform-prior
    target's argmax on a quarter of blends, which is plausibly why they
    removed the effect entirely.

    This solves instead for the offset under which the marginal matches,
    by iterative proportional fitting. Nothing is tuned: the offset is
    determined by the fitted Gaussians, the class prior and the lambda
    distribution. Measured, it lands the marginal exactly on the class
    prior while preserving 81 percent of the per-sample argmax, against
    74 percent for empirical priors.

    Args:
        vad: [N, 3] training VAD.
        labels: [N] training labels, used only to draw representative pairs.
        means: [C, 3] fitted class centres.
        inv_covs: [C, 3, 3] fitted inverse covariances.
        class_prior: [C] target marginal, normally the class frequencies.
        alpha: Beta(alpha, alpha) parameter, matching the arm.
        fold_lambda: whether lambda is folded into [0.5, 1], matching the arm.
        n_samples: number of synthetic blends to calibrate on.
        iters: iterative proportional fitting steps.
        seed: RNG seed, so the offset is reproducible.

    Returns:
        [C] log-prior offset to pass to gaussian_posterior.
    """
    device = vad.device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n = vad.shape[0]
    idx = torch.randint(n, (n_samples,), generator=gen).to(device)
    pick = torch.randint(n, (n_samples,), generator=gen).to(device)
    lam = torch.distributions.Beta(alpha, alpha).sample((n_samples,))
    if fold_lambda:
        lam = torch.maximum(lam, 1.0 - lam)
    lam = lam.to(device).unsqueeze(1)
    v_mix = lam * vad[idx] + (1.0 - lam) * vad[pick]

    prior = (class_prior / class_prior.sum()).to(device)
    offset = torch.zeros(means.shape[0], device=device)
    for _ in range(int(iters)):
        marg = gaussian_posterior(v_mix, means, inv_covs, offset).mean(0)
        offset = offset + torch.log(prior / marg.clamp_min(1e-12))
    return offset


def general_soft_mixup_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    vad: torch.Tensor,
    means: torch.Tensor,
    inv_covs: torch.Tensor,
    alpha: float = 2.0,
    hard_control: bool = False,
    vanilla_target: bool = False,
    endpoint_posterior: bool = False,
    same_class_hard: bool = False,
    log_priors: Optional[torch.Tensor] = None,
    trust: Optional[torch.Tensor] = None,
    trust_min: float = 0.2,
    repeats: int = 1,
    pull_fraction: float = 0.0,
    pull_tracker=None,
    target_shuffle: bool = False,
    dirichlet_conc: Optional[torch.Tensor] = None,
    fold_lambda: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Blend any pair and supervise with the blend's own class posterior.

    Three changes from the directed variants, each answering a measured
    failure rather than a preference.

    ANY PAIR. muted_mixup only ever blends emotional samples toward neutral,
    so neutral is a partner and never a source and receives no synthetic data
    at all. symmetric_mixup_step adds the reverse direction but still only
    crosses the neutral boundary. Here every sample is a source and the
    partner is drawn from the whole batch, so the augmentation covers the
    space instead of one corridor.

    CONTROLLED LAMBDA. The VAD-calibrated coefficient has median 0.47 and a
    tenth of its blends below 0.09, i.e. 91 percent partner. sample_lambda
    already folds Beta(alpha, alpha) into [0.5, 1] for exactly this reason:
    a blend the partner dominates is an utterance whose label is genuinely
    doubtful. This uses that.

    SOFT TARGET. Both existing variants assert a hard label. Inheriting it
    puts an emotional target on a neutral-region point, which drove neutral
    recall from 54.3 to 27.4 percent. Taking the argmax of the blended VAD
    instead fixes the direction but still asserts confidence: measured over
    30,000 real blends the top posterior is below 0.5 on 27.7 percent of
    them, and at the class flip points it is 0.33 to 0.46. Here the posterior
    itself is the target, so a blend sitting between neutral and sad is told
    it is uncertain, which it is, and one deep inside angry gets a near
    one-hot target. No threshold and no switch point are chosen anywhere.

    The blended VAD and the blended embedding correspond exactly rather than
    approximately: the auxiliary head is a bare nn.Linear, and a linear map of
    a convex combination is the convex combination of the maps.

    Args:
        embeddings: [B, hidden_dim] post-fusion embeddings.
        labels: [B] class labels, used only by the control.
        output_layer: maps embeddings to logits.
        vad: [B, 3] valence/arousal/dominance.
        means: [C, 3] class VAD centres fitted on the training split.
        inv_covs: [C, C, C] per-class inverse covariances.
        alpha: Beta(alpha, alpha) parameter for the mixing coefficient.
        hard_control: supervise with the dominant endpoint's hard label
            instead of the posterior.
        vanilla_target: supervise with standard mixup's soft label,
            lam * y_source + (1 - lam) * y_partner. THIS is the control the
            novelty claim needs. Standard mixup is already soft, so comparing
            the geometry-derived target only against a HARD label conflates
            two effects: soft beating hard, which is textbook, and geometry
            beating lambda-interpolation, which is the actual claim.
            Measured over 40,000 blends the two soft targets differ by a
            median total-variation distance of 0.41 and their argmax agrees
            on only 63 percent, so they are genuinely different supervision.
            The vanilla target can only place mass on the two endpoint
            classes; the geometry target places more than a quarter of its
            mass on a third class in 37 percent of blends.
        endpoint_posterior: supervise with lam * q(v_a) + (1 - lam) * q(v_b),
            the VAD posteriors of the two ENDPOINTS interpolated by lambda,
            instead of q evaluated at the blended VAD point. This isolates
            the one thing the method claims: that evaluating the mixed
            affect location matters beyond interpolating VAD-derived soft
            labels. Both arms see identical blends, identical lambda and
            identical fitted q, and both targets are soft and VAD-derived,
            so a difference between them cannot be soft-versus-hard, cannot
            be the use of VAD, and cannot be the lambda distribution. The
            two coincide only where q is affine along the segment joining
            the endpoints, which a Gaussian posterior is not.
        same_class_hard: when both parents carry the same label, supervise
            with that label as a one-hot instead of with q. Measured over
            60,000 real blends, 32.7 percent of pairs share a class, and on
            those the blended-point target places only 0.549 mass on a
            label that is not in doubt: a blend of two sad utterances is
            sad. That is a certainty being discarded, not an ambiguity
            being modelled. The gate is a fact about the pair, not a tuned
            threshold, and it confines q to the cross-class blends it is
            actually motivated for.
        log_priors: [C] log class priors for q, or None for uniform. See
            gaussian_posterior for why uniform is not neutral.
        trust: [B] per-sample annotator consensus, the fraction of
            annotators voting the majority class. When given, it decides
            how far the geometry is allowed to override the parents'
            labels:

                c   = lam * trust_a + (1 - lam) * trust_b
                w   = (1 - c) / (1 - trust_min)
                tgt = w * q(v_mix) + (1 - w) * vanilla

            A blend of two unanimously annotated utterances keeps the
            vanilla target; one whose parents annotators disagreed about is
            labelled by where it lands. The motivation is measured: q
            contradicts the annotation on 33.6 percent of training rows,
            and consensus separates those rows (0.634 against 0.696) while
            annotator VAD dispersion does not (0.787 against 0.786). So
            disagreement tracks label reliability, not VAD reliability.
            Note this also damps the average geometry strength to about a
            third, which is why the shuffled placebo, carrying an identical
            w distribution, is the arm it must be judged against.
        trust_min: smallest consensus in the training split, used to map
            consensus onto [0, 1] without introducing a tuned constant.
        repeats: number of blends drawn per real sample. The encoder pass is
            already paid for, so additional blends cost only a forward and
            backward through the classifier head.
        pull_fraction: fraction of blends whose partner is REPLACED by the
            row's own class centroid, so the blend moves toward the class
            prototype rather than toward another sample. Count-fixed: the
            number of synthetic samples and the gradient scale are unchanged,
            which makes this a genuine mixing ratio rather than an added loss
            term. Those rows get a one-hot target, both endpoints being the
            same class. 0.0 reproduces the original behaviour exactly.
        pull_tracker: ClassCentroidEMA supplying the prototypes. Required
            when pull_fraction is above zero, ignored otherwise.
        target_shuffle: THE PLACEBO for the geometry target. Permutes the
            finished targets among blends that share a SOURCE label, so the
            marginal, the entropy and the per-class target distribution are
            all preserved exactly while the correspondence between a blend
            and its own affect position is destroyed.

            This is the control the mechanism needs and that vanilla mixup
            does not supply. Vanilla's target has entropy 0.39 against this
            method's 0.85 and can only place mass on the two parents'
            labels, so beating vanilla does not rule out the possibility
            that any sufficiently diffuse soft target would do. Only a
            matched-distribution shuffle rules that out: if the gain
            survives here, the target's CONTENT is irrelevant and the
            mechanism is just label smoothing with an unusual shape.
        dirichlet_conc: [C, C] concentrations from solve_dirichlet_match. The
            VAD-FREE control: the target is SAMPLED from a Dirichlet matched
            to q's class-conditional mean and entropy, so it carries no affect
            information at all, not even a real q value attached to the wrong
            blend. Separates "VAD supplies the right target family" from
            "VAD is unnecessary".
        fold_lambda: fold the coefficient into [0.5, 1] so the source always
            dominates. False allows the partner to dominate, reaching the
            aggressive regime where mislabelling actually occurs.
        generator: optional RNG.

    Returns:
        (logits, targets, lam). targets is [M, C] soft when hard_control is
        False and [M] long otherwise. (None, None, None) for a batch of one.
    """
    device = embeddings.device
    n = embeddings.size(0)
    if n < 2:
        return None, None, None

    # More than one blend per real sample. The batch is a fixed cost in
    # encoder forward/backward; extra blends reuse those embeddings and only
    # add a classifier-head pass, so this buys synthetic supervision cheaply.
    # repeats=1 reproduces the original behaviour exactly.
    if repeats > 1:
        embeddings = embeddings.repeat(repeats, 1)
        labels = labels.repeat(repeats)
        vad = vad.repeat(repeats, 1)
        if trust is not None:
            trust = trust.repeat(repeats)
        n = embeddings.size(0)

    pick = torch.randint(n, (n,), device=device, generator=generator)
    clash = pick == torch.arange(n, device=device)
    if clash.any():
        pick[clash] = (pick[clash] + 1) % n

    if fold_lambda:
        lam = sample_lambda(n, alpha, device)
    else:
        # Unfolded: the partner is allowed to dominate. With a hard label that
        # is mislabelling by construction, which is what drove neutral recall
        # to 48.3 percent. With a soft target it is safe, because the target
        # follows the blend rather than its source, so this is the only
        # setting in which aggressive blending can be stated honestly.
        lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device)
    lam = lam.unsqueeze(1)

    # Count-fixed prototype pull: on a fraction of rows the partner is the
    # row's own class centroid instead of another sample. Applied before the
    # blend so neither the number of synthetic samples nor the gradient scale
    # changes, which is what makes this a ratio rather than an extra term.
    pull_mask = None
    if pull_fraction > 0.0 and pull_tracker is not None and pull_tracker.ready:
        sel = torch.rand(n, device=device, generator=generator) < pull_fraction
        if bool(sel.any()):
            pull_mask = sel
            partner = embeddings[pick].clone()
            partner[sel] = pull_tracker.centroids[labels.long()][sel]
        else:
            partner = embeddings[pick]
    else:
        partner = embeddings[pick]
    mixed = lam * embeddings + (1.0 - lam) * partner
    logits = output_layer(mixed)

    if hard_control:
        # Label from whichever endpoint dominates the blend. With fold_lambda
        # that is always the source; unfolded it is not, so choose per sample
        # rather than assuming.
        dominant = torch.where(lam.squeeze(1) >= 0.5, labels, labels[pick])
        return logits, dominant.long(), lam.squeeze(1)

    if vanilla_target:
        n_cls = means.shape[0]
        eye = torch.eye(n_cls, device=device, dtype=logits.dtype)
        target = lam * eye[labels.long()] + (1.0 - lam) * eye[labels[pick].long()]
        return logits, target, lam.squeeze(1)

    if endpoint_posterior:
        # Interpolate the endpoints' own posteriors. Same q, same lambda,
        # same blends: the only change is that q is never evaluated at the
        # mixed location.
        q_src = gaussian_posterior(vad, means, inv_covs, log_priors)
        q_par = gaussian_posterior(vad[pick], means, inv_covs, log_priors)
        post = lam * q_src + (1.0 - lam) * q_par
    else:
        vad_mix = lam * vad + (1.0 - lam) * vad[pick]
        post = gaussian_posterior(vad_mix, means, inv_covs, log_priors)

    if trust is not None:
        # Per-sample trust in the annotated label decides how much the
        # geometry may override it. vanilla is rebuilt here rather than
        # reused because the vanilla_target branch returns early.
        n_cls = means.shape[0]
        eye_t = torch.eye(n_cls, device=device, dtype=post.dtype)
        vanilla = (lam * eye_t[labels.long()]
                   + (1.0 - lam) * eye_t[labels[pick].long()])
        c_pair = lam * trust.unsqueeze(1) + (1.0 - lam) * trust[pick].unsqueeze(1)
        w = ((1.0 - c_pair) / max(1e-6, 1.0 - trust_min)).clamp(0.0, 1.0)
        post = w * post + (1.0 - w) * vanilla

    if same_class_hard:
        same = labels == labels[pick]
        if bool(same.any()):
            eye = torch.eye(means.shape[0], device=device, dtype=post.dtype)
            post = torch.where(same.unsqueeze(1), eye[labels.long()], post)

    if pull_mask is not None:
        # A blend between a sample and its own class prototype belongs to that
        # class without ambiguity, so the geometry target does not apply.
        eye_p = torch.eye(means.shape[0], device=device, dtype=post.dtype)
        post = torch.where(pull_mask.unsqueeze(1), eye_p[labels.long()], post)

    if dirichlet_conc is not None:
        conc = dirichlet_conc.to(device=device, dtype=torch.float32)[labels.long()]
        post = torch.distributions.Dirichlet(conc).sample().to(post.dtype)

    if target_shuffle:
        # Permute targets WITHIN source-label groups: a blend from a happy
        # source still receives a target typical of happy-source blends, so
        # every distributional property is held fixed and only the per-blend
        # correspondence is broken.
        perm = torch.arange(n, device=device)
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel(), device=device,
                                               generator=generator)]
        post = post[perm]

    return logits, post, lam.squeeze(1)


def soft_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy against a soft target distribution.

    Args:
        logits: [M, C] logits for the synthesised samples.
        targets: [M, C] target distribution, or [M] hard labels.
        class_weights: optional [C] weights, applied as in the primary loss
            so the synthetic samples do not reintroduce the class imbalance.

    Returns:
        Scalar loss.
    """
    if targets.dim() == 1:
        return F.cross_entropy(logits, targets.long(), weight=class_weights)
    logp = F.log_softmax(logits, dim=1)
    per_sample = -(targets * logp).sum(dim=1)
    if class_weights is not None:
        # Weight each synthetic sample by the weight of its dominant class,
        # mirroring how the hard-label path applies class weights.
        w = class_weights[targets.argmax(dim=1)]
        return (per_sample * w).sum() / w.sum().clamp_min(1e-8)
    return per_sample.mean()


def per_modality_mixup_step(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    vad: torch.Tensor,
    classify_from_modalities,
    means: torch.Tensor,
    inv_covs: torch.Tensor,
    alpha: float = 2.0,
    w_audio: float = 0.638,
    independent: bool = True,
    mode: str = "independent",
    fold_lambda: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Blend each modality with its own coefficient, before fusion.

    Every other arm blends the POST-fusion embedding, so audio and text move
    together under one lambda. This blends the two branch outputs separately
    and lets the fusion recombine them, which reaches points no joint blend
    can produce: the audio of one utterance with mostly the text of another.

    THE TARGET IS AN APPROXIMATION, unlike every other arm here, and that is
    the honest cost of this variant. With independent coefficients the blend
    is not lam * z_a + (1 - lam) * z_b for any single lam, and cross-attention
    fusion is nonlinear, so the exact correspondence between the embedding
    blend and the VAD blend is gone. Human annotations also cannot supply the
    missing quantity: a rater scored the utterance, not its modalities, so
    affect is not decomposable by modality from the labels.

    What is used instead is an effective coefficient,

        lam_eff = w_audio * lam_audio + (1 - w_audio) * lam_text
        target  = q(lam_eff * v_a + (1 - lam_eff) * v_b)

    with w_audio the audio branch's share of above-chance accuracy, measured
    once by modality ablation (audio-only 64.4 percent and text-only 47.4
    against 25 chance, giving 39.4 and 22.4 above chance, so 0.638). It is a
    single global constant from an existing measurement, not a per-sample
    pseudo-label, which keeps the target anchored to annotated VAD rather
    than to the model's own predictions.

    The control sets independent=False, so both modalities share one lambda.
    That is algebraically the joint blend, routed through the same pre-fusion
    path, and it isolates independence per modality from the change of where
    the blend happens.

    Args:
        audio_features: [B, D_a] audio branch output, gradients attached.
        text_features: [B, D_t] text branch output, gradients attached.
        labels: [B] class labels, unused except by the caller.
        vad: [B, 3] annotated VAD.
        classify_from_modalities: callable mapping (audio, text) to logits.
        means: [C, 3] fitted class centres.
        inv_covs: [C, 3, 3] fitted inverse covariances.
        alpha: Beta(alpha, alpha) parameter.
        w_audio: audio's share in the effective coefficient.
        independent: draw a separate lambda per modality. Superseded by
            mode; kept so existing callers behave unchanged.
        mode: "independent" draws a lambda per modality; "joint" shares one;
            "audio_only" blends audio and takes text wholly from the source;
            "text_only" is its mirror. With audio_only the effective
            coefficient becomes w*lam_audio + (1 - w), so the target leans on
            the source, which is correct because the text is entirely the
            source's.
        fold_lambda: fold each lambda into [0.5, 1].
        generator: optional RNG.

    Returns:
        (logits, target, lam_eff). (None, None, None) for a batch of one.
    """
    device = audio_features.device
    n = audio_features.size(0)
    if n < 2:
        return None, None, None

    pick = torch.randint(n, (n,), device=device, generator=generator)
    clash = pick == torch.arange(n, device=device)
    if clash.any():
        pick[clash] = (pick[clash] + 1) % n

    def draw():
        lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device)
        if fold_lambda:
            lam = torch.maximum(lam, 1.0 - lam)
        return lam

    if mode == "audio_only":
        # Blend the audio branch; the text branch is taken wholly from the
        # source. Motivated by the modality ablation: text contributes only
        # +11 happy recall, so blending it may mostly inject noise while
        # blending audio acts where the signal is.
        lam_a = draw()
        lam_t = torch.ones(n, device=device)
    elif mode == "text_only":
        lam_a = torch.ones(n, device=device)
        lam_t = draw()
    elif mode == "joint":
        lam_a = draw()
        lam_t = lam_a
    else:
        lam_a = draw()
        lam_t = draw() if independent else lam_a

    a_mix = lam_a.unsqueeze(1) * audio_features + (1.0 - lam_a).unsqueeze(1) * audio_features[pick]
    t_mix = lam_t.unsqueeze(1) * text_features + (1.0 - lam_t).unsqueeze(1) * text_features[pick]
    logits = classify_from_modalities(a_mix, t_mix)

    lam_eff = (w_audio * lam_a + (1.0 - w_audio) * lam_t).unsqueeze(1)
    v_mix = lam_eff * vad + (1.0 - lam_eff) * vad[pick]
    return logits, gaussian_posterior(v_mix, means, inv_covs), lam_eff.squeeze(1)


class ClassCentroidEMA:
    """Running per-class mean of post-fusion embeddings.

    Blending a sample toward a same-class PARTNER is unreliable at batch size
    16: sad is 10.7 percent of the pool, so a sad row often has no same-class
    partner in its batch at all. A running centroid is always available and is
    a better estimate of the class prototype than any single partner.

    Attributes:
        ready: True once every class has been seen at least once.
    """

    def __init__(self, num_classes: int, dim: int, device: torch.device,
                 momentum: float = 0.9) -> None:
        """Create the tracker.

        Args:
            num_classes: number of classes.
            dim: embedding dimensionality.
            device: device the centroids live on.
            momentum: EMA momentum; 0.9 converges within a few dozen batches
                against the ~3,800 batches in an epoch here.
        """
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.centroids = torch.zeros(num_classes, dim, device=device)
        self.seen = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @property
    def ready(self) -> bool:
        """True once every class has contributed at least one batch."""
        return bool(self.seen.all())

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Fold one batch into the running centroids.

        Args:
            embeddings: [B, D] post-fusion embeddings.
            labels: [B] class labels.
        """
        z = embeddings.detach()
        for c in range(self.num_classes):
            m = labels == c
            if not bool(m.any()):
                continue
            batch_mean = z[m].mean(0)
            if self.seen[c]:
                self.centroids[c].mul_(self.momentum).add_(
                    batch_mean, alpha=1.0 - self.momentum)
            else:
                self.centroids[c] = batch_mean
                self.seen[c] = True


def prototype_pull_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    tracker: "ClassCentroidEMA",
    alpha: float = 2.0,
    global_target: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Blend each sample toward its own class prototype, keeping its label.

    Motivated by a measured association rather than a hunch: across 44 mixup
    runs, the share of embedding variance that is WITHIN class correlates with
    UAR at r = -0.451 (p = 0.002), so tighter classes go with better transfer.
    The same correlation is absent in the auxiliary family (r = +0.212,
    p = 0.26), so this is specific to the mixup mechanism.

    Unlike every other blend here the label is unambiguous: both endpoints
    belong to the same class, so the target is a plain one-hot and none of the
    labelling machinery applies.

    Args:
        embeddings: [B, D] post-fusion embeddings.
        labels: [B] class labels.
        output_layer: maps embeddings to logits.
        tracker: running class centroids.
        alpha: Beta(alpha, alpha) parameter, folded so the real sample
            always dominates its prototype.
        global_target: PLACEBO. Pull toward the class-agnostic mean of the
            centroids instead of the sample's own class centroid, with the
            displacement rescaled to the magnitude the real pull would have
            applied. Magnitude matching is not cosmetic: unscaled, the global
            mean sits far from any class centroid and moves samples 5.6 times
            further than the real pull, so the placebo would lose on
            intervention size rather than on class-specificity. Rescaled, it
            keeps the structure (contract toward a fixed attractor) and the
            step size, and removes only the one thing under test.
        generator: optional RNG.

    Returns:
        (logits, labels) for the pulled samples, or (None, None) when the
        centroids are not yet populated for every class.
    """
    if not tracker.ready:
        return None, None
    device = embeddings.device
    n = embeddings.size(0)
    lam = sample_lambda(n, alpha, device).unsqueeze(1)
    own = tracker.centroids[labels.long()]
    if global_target:
        glob = tracker.centroids.mean(0, keepdim=True)
        step = (1.0 - lam) * (own - embeddings)
        direction = glob - embeddings
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        pulled = embeddings + direction * step.norm(dim=1, keepdim=True)
    else:
        pulled = lam * embeddings + (1.0 - lam) * own
    return output_layer(pulled), labels.long()


def global_mixup_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    num_classes: int,
    alpha: float = 2.0,
    gamma: float = 4.0,
    theta: Optional[float] = None,
    top_s: int = 0,
    fold_lambda: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Global Mixup relabeling (Xie et al., AAAI 2023, arXiv 2206.02734).

    The closest published method to ours in motivation: it states the same
    problem, that mixup's label "is simply combined from the labels of the
    original sample pairs without reasonable judgment, resulting in ambiguous
    labels", and relabels the blend from global relationships instead. The
    signal differs, which is exactly the comparison worth making: Global
    Mixup reads structure out of the UNSUPERVISED feature space, ours reads
    it out of ANNOTATED affect space.

    Their equations, implemented directly:

        y*            = sum_t P(B_t | D(B_t, B')) * y_t                (3)
        P(B_t | D)    = exp(D(B_t, B')) / sum_i exp(D(B', B_i))        (4)
        D(B_t, B')    = gamma * cosine(B_t, B')                        (5)

    so the blend's label is a softmax-weighted vote over the real samples'
    one-hot labels, weighted by embedding-space cosine similarity. Only
    blends whose vanilla label is "overconfident", argmax y_tilde >= theta,
    are relabeled; the rest keep the vanilla target.

    Adaptations, each documented rather than silent:

      SPACE. Their D flattens a word-embedding MATRIX per sentence. Our
      samples are single post-fusion vectors, so equation 5 reduces to plain
      cosine similarity between vectors, with gamma retained.

      T. They set T, the number of samples whose relationships are computed,
      to the batch size. Kept: every real sample in the batch votes.

      THETA. With c = 4 the maximum of any distribution over four classes is
      at least 1/c = 0.25, so theta = 1/c relabels every blend. That is the
      setting used here, because it needs no threshold to be chosen and it
      matches our own arm, which also supplies a target for every blend.

      GAMMA. Tuned in the paper over {1, 2, 4, 6}. Tuning it here would mean
      selecting on test, so it is fixed a priori and the arm is run at two
      values, the same treatment GenLabel received with its PCA dimension.

      LOSS. Their equation 7 trains on original plus vanilla-labelled plus
      global-labelled blends. Here the global target simply replaces our
      geometry target at the same mixup weight, so the only thing that
      differs between the two arms is the labelling rule. Adding their extra
      vanilla term would change the blend budget and confound the
      comparison.

    Args:
        embeddings: [B, D] post-fusion embeddings.
        labels: [B] class labels.
        output_layer: maps embeddings to logits.
        num_classes: number of classes.
        alpha: Beta(alpha, alpha) parameter for the mixing coefficient.
        gamma: relationship-correction scale on the cosine similarity.
        theta: overconfidence threshold; defaults to 1/num_classes, which
            relabels every blend.
        top_s: keep only this many nearest voters, zeroing the rest; 0 means
            every sample in the batch votes.
        fold_lambda: fold the coefficient into [0.5, 1], matching our arms.
        generator: optional RNG.

    Returns:
        (logits, target, lam), target [B, C] soft. (None, None, None) for a
        batch of one.
    """
    device = embeddings.device
    n = embeddings.size(0)
    if n < 2:
        return None, None, None
    if theta is None:
        theta = 1.0 / float(num_classes)

    pick = torch.randint(n, (n,), device=device, generator=generator)
    clash = pick == torch.arange(n, device=device)
    if clash.any():
        pick[clash] = (pick[clash] + 1) % n

    if fold_lambda:
        lam = sample_lambda(n, alpha, device)
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device)
    lam = lam.unsqueeze(1)
    mixed = lam * embeddings + (1.0 - lam) * embeddings[pick]
    logits = output_layer(mixed)

    eye = torch.eye(num_classes, device=device, dtype=logits.dtype)
    vanilla = lam * eye[labels.long()] + (1.0 - lam) * eye[labels[pick].long()]

    # Equation 5: cosine similarity between each blend and every real sample.
    zn = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    mn = mixed / mixed.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = gamma * (mn @ zn.t())

    if top_s and 0 < top_s < n:
        # "the largest top-s from all computed D will be used for P, and P of
        # the rest will be set to 0"
        thresh = sim.topk(top_s, dim=1).values[:, -1:]
        sim = sim.masked_fill(sim < thresh, float("-inf"))

    # Equations 4 and 3: softmax over similarities, then a weighted vote.
    p = torch.softmax(sim, dim=1)
    y_star = p @ eye[labels.long()]

    # Relabel only the overconfident blends.
    overconf = (vanilla.max(dim=1).values >= theta).unsqueeze(1)
    target = torch.where(overconf, y_star, vanilla)
    return logits, target, lam.squeeze(1)


def solve_dirichlet_match(
    vad: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    inv_covs: torch.Tensor,
    num_classes: int,
    alpha: float = 2.0,
    fold_lambda: bool = True,
    n_samples: int = 60000,
    seed: int = 0,
) -> torch.Tensor:
    """Dirichlet concentrations reproducing q's class-conditional statistics.

    The VAD-FREE control. The shuffled ablation still draws its targets from
    q, so it still needs the annotations; this asks whether the target has to
    come from VAD at all. For each source class it fits a Dirichlet whose
    mean equals q's mean target for that class and whose entropy matches q's,
    so a sampled target is statistically indistinguishable from a real one in
    both first moment and dispersion, while carrying no affect information
    about any particular blend.

    Entropy is matched by a scan over the concentration scale, which is the
    only free quantity once the mean is fixed.

    Args:
        vad: [N, 3] training VAD.
        labels: [N] training labels.
        means: [C, 3] fitted class centres.
        inv_covs: [C, 3, 3] fitted inverse covariances.
        num_classes: number of classes.
        alpha: Beta(alpha, alpha) parameter, matching the arm.
        fold_lambda: whether lambda is folded, matching the arm.
        n_samples: blends used to estimate the statistics.
        seed: RNG seed, so the fit is reproducible.

    Returns:
        [C, C] concentration matrix; row c is the Dirichlet for a blend whose
        SOURCE is class c.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n = vad.shape[0]
    i1 = torch.randint(n, (n_samples,), generator=gen)
    i2 = torch.randint(n, (n_samples,), generator=gen)
    lam = torch.distributions.Beta(alpha, alpha).sample((n_samples,))
    if fold_lambda:
        lam = torch.maximum(lam, 1.0 - lam)
    q = gaussian_posterior(lam.unsqueeze(1) * vad[i1]
                           + (1.0 - lam).unsqueeze(1) * vad[i2], means, inv_covs)
    src = labels[i1]

    def entropy(t):
        return -(t.clamp_min(1e-12).log() * t).sum(1)

    out = torch.ones(num_classes, num_classes)
    for c in range(num_classes):
        m = src == c
        if int(m.sum()) < 50:
            continue
        mean_t = q[m].mean(0).clamp_min(1e-4)
        mean_t = mean_t / mean_t.sum()
        want = float(entropy(q[m]).mean())
        best, best_conc = None, 4.0
        for conc in torch.arange(0.5, 30.0, 0.1):
            samp = torch.distributions.Dirichlet(
                (mean_t * conc).clamp_min(1e-3)).sample((3000,))
            gap = abs(float(entropy(samp).mean()) - want)
            if best is None or gap < best:
                best, best_conc = gap, float(conc)
        out[c] = (mean_t * best_conc).clamp_min(1e-3)
    return out
