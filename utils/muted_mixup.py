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
