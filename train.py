#!/usr/bin/env python3
"""
Clean training script for emotion recognition with contrastive learning
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hashlib
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import numpy as np
import random
import wandb
from pathlib import Path
import argparse
import json
import os

from utils.annotator_meta import SUBTYPE_NAMES
from utils.config import Config, RAW_AUDIO_ENCODERS
from utils.continuum import load_continuum
from data.dataset import create_datasets
from data.collate import vad_collate_fn
from models.classifier import create_model
from models.contrastive_loss import create_contrastive_loss
from models.domain_adversarial import (
    DomainDiscriminator,
    ModalityDomainDiscriminator,
    PrototypicalDomainAdversarialLoss,
)
from models.prototypicality_predictor import PrototypicalityPredictor
from utils.length_bucket import (LengthBucketedBatchSampler,
                                 load_or_compute_durations)
from utils.margin_loss import LDAMLoss
from utils.metrics import threshold_free_metrics
from utils.metrics import calculate_classification_metrics, calculate_vad_metrics
from utils.prototypicality import (
    diagonal_whitening,
    marginal_whitening,
    signed_axis_whitened,
    symmetric_whitening_matrices,
    pooled_whitening,
    DATASETS_WITH_VAD,
    LearnableCentroids,
    batch_calculate_difficulty,
    class_vad_stats,
    mahalanobis_distance,
    all_class_residuals,
    residual_targets,
    vad_trust_weights,
    vad_explainable,
    signed_axis_targets,
    whitening_matrices,
    batch_difficulty_tensor,
    calculate_difficulty,
)
from utils.genlabel import EmbeddingClassDensity, genlabel_mixup_step
from utils.multiview_prototypicality import compute_multiview_difficulty, compute_crossmodal_agreement
from utils.muted_mixup import (ClassCentroidEMA, consistent_muted_mixup,
                               solve_dirichlet_match,
                               global_mixup_step,
                               gaussian_posterior, prototype_pull_step,
                               per_modality_mixup_step,
                               solve_calibrated_prior,
                               muted_mixup_loss, muted_mixup_step,
                               general_soft_mixup_step, soft_target_loss,
                               sparse_fill_mixup_step,
                               vad_gated_symmetric_mixup,
                               symmetric_mixup_step)
from utils.salience import (
    compute_salience_stats,
    placebo_difficulty,
    salience_difficulty,
    salience_weights,
)
from utils.vadmix import vadmix_step, build_prototype_tensor, soft_cross_entropy as vadmix_soft_ce
from utils.curriculum import create_curriculum_subset, get_curriculum_pacing_function
from utils.proto_atyp import (
    build_class_centers,
    compute_split_thresholds,
    batch_random_sub_labels,
    batch_sub_labels,
    collapse_sub_logits_to_class_probs,
)
from utils.confidence_subtype import (
    ConfidenceTracker,
    class_relative_intensity,
    neutral_soft_targets,
)
from utils.latent_prototypes import (
    collapse_prototype_logits,
    latent_prototype_loss,
    usage_statistics,
)
from utils.hierarchical import (
    compose_class_probs,
    emotion_class_weights,
    emotion_salience,
    hierarchical_loss,
    hierarchical_predictions,
)
from utils.evidence import (
    AmbiguityWeights,
    anti_neutral_margin_loss,
    evidence_bce_loss,
    evidence_class_probs,
    calibrate_evidence_threshold,
    evidence_predictions,
    head_pos_weights,
)
from utils.nrc_lexicon import (
    NRC_CATEGORIES,
    batch_nrc_sub_labels,
    build_nrc_class_centers,
    compute_nrc_thresholds,
    load_lexicon,
    profile_for_text,
)
from utils.aux_vad import (
    batch_cluster_ids_and_mask,
    batch_per_class_cluster_ids_and_mask,
    batch_residual_cluster_ids_and_mask,
    batch_scrambled_ids_and_mask,
    batch_vad_targets_and_mask,
    build_per_class_vad_centroids,
    build_residual_centroids,
    build_vad_centroids,
)
from utils.aux_vad_viz import save_cluster_visualization
from utils.embed_collapse import collapse_metrics


def soft_cross_entropy(logits, soft_targets, class_weights=None):
    """
    Cross-entropy with per-sample soft targets and optional class weights.

    Args:
        logits: [batch_size, num_classes] raw logits
        soft_targets: [batch_size, num_classes] soft probability targets
        class_weights: [num_classes] optional class weights

    Returns:
        [batch_size] per-sample loss (unreduced)
    """
    log_probs = F.log_softmax(logits, dim=1)  # [B, C]
    per_sample = -(soft_targets * log_probs).sum(dim=1)  # [B]

    # Apply class weights: weight each sample by the weight of its dominant class
    if class_weights is not None:
        hard_labels = soft_targets.argmax(dim=1)  # [B]
        weights = class_weights[hard_labels]  # [B]
        per_sample = per_sample * weights

    return per_sample


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_model_inputs(batch, config, model, device):
    """
    Prepare model inputs from a batch based on modality and audio encoder type.

    Returns:
        dict of model inputs ready for model(**inputs)
    """
    audio_encoder_type = getattr(config, 'audio_encoder_type', 'preextracted')

    # Prefer cached features whenever the batch carries them. wav2vec2 and
    # emotion2vec encoders both cache their frozen output when unfreeze
    # layers is 0, so at that point the batch has 'features' and no
    # 'waveforms' regardless of the configured encoder_type.
    has_features = 'features' in batch
    uses_raw_audio = audio_encoder_type in RAW_AUDIO_ENCODERS and not has_features

    if config.modality == "audio":
        if uses_raw_audio:
            waveforms = batch['waveforms'].to(device)
            audio_mask = batch['audio_attention_mask'].to(device)
            return {'audio_waveforms': waveforms, 'audio_attention_mask': audio_mask}
        else:
            features = batch['features'].to(device)
            return {'audio_features': features}

    elif config.modality == "text":
        transcripts = batch['transcript']
        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            input_ids, attention_mask = model.text_encoder.tokenize_batch(
                transcripts, max_length=config.text_max_length, device=device
            )
            return {'text_input_ids': input_ids, 'text_attention_mask': attention_mask}

    elif config.modality == "both":
        transcripts = batch['transcript']
        model_inputs = {}

        if uses_raw_audio:
            model_inputs['audio_waveforms'] = batch['waveforms'].to(device)
            model_inputs['audio_attention_mask'] = batch['audio_attention_mask'].to(device)
        else:
            model_inputs['audio_features'] = batch['features'].to(device)

        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            input_ids, attention_mask = model.text_encoder.tokenize_batch(
                transcripts, max_length=config.text_max_length, device=device
            )
            model_inputs['text_input_ids'] = input_ids
            model_inputs['text_attention_mask'] = attention_mask

        return model_inputs

    return {}


class _PerSampleProject(torch.autograd.Function):
    """Project this branch's gradient off a reference gradient, per sample.

    Batch-level PCGrad (proto_pcgrad) takes gradients of the batch-MEAN losses,
    so its conflict test runs on the aggregate. At batch 16 with roughly 65
    percent of rows VAD-explained, the auxiliary gradient is dominated by those
    rows before any projection happens, so it cannot protect the minority where
    the conflict actually lives.

    This applies the same projection per ROW of the shared representation. A
    row whose auxiliary gradient opposes its classification gradient keeps only
    the orthogonal part; a row where they agree passes through untouched.

    The reference gradient is supplied through a one-element list filled in
    after the forward pass but before backward, because the classification
    gradient is not known when the auxiliary head runs.

    Scaling-invariant in the reference: multiplying ref by s scales dot by s
    and denom by s squared, so the subtracted term is unchanged. AMP loss
    scaling on the reference therefore does not alter the projection.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, ref_holder: list) -> torch.Tensor:
        """Identity forward that remembers where to find the reference.

        Args:
            x: shared representation, [B, D].
            ref_holder: one-element list, filled with the [B, D] reference
                gradient before backward runs.

        Returns:
            x unchanged.
        """
        ctx.ref_holder = ref_holder
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Remove the component of grad_out that opposes the reference.

        Args:
            grad_out: [B, D] gradient arriving from the auxiliary branch.

        Returns:
            Tuple of the projected gradient and None for ref_holder.
        """
        ref = ctx.ref_holder[0]
        if ref is None:
            return grad_out, None
        g = grad_out.flatten(1)
        r = ref.flatten(1).to(g.dtype)
        dot = (g * r).sum(dim=1, keepdim=True)
        denom = r.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
        g = torch.where(dot < 0, g - (dot / denom) * r, g)
        return g.view_as(grad_out), None


def train_epoch(model, dataloader, criterion, optimizer, device, config,
                contrastive_criterion=None, domain_discriminator=None, domain_adv_loss=None,
                proto_predictor=None, scaler=None, global_step=0, current_epoch=0,
                proto_predictor_audio=None, proto_predictor_text=None,
                modality_discriminator=None, centroid_tracker=None,
                proto_atyp_ctx=None, aux_vad_centroids=None, logit_adjust=None,
                confidence_tracker=None, intensity_ctx=None,
                ambiguity_weights=None, evidence_pos_weight=None,
                hierarchical_emotion_weights=None, salience_stats=None,
                genlabel_density=None, ot_cost=None):
    """
    Train for one epoch

    Args:
        model: EmotionClassifier
        dataloader: DataLoader
        criterion: Primary loss (CrossEntropy or MSE)
        optimizer: Optimizer
        device: torch device
        config: Config object
        contrastive_criterion: Optional contrastive loss
        domain_discriminator: Optional DomainDiscriminator for adversarial training
        domain_adv_loss: Optional PrototypicalDomainAdversarialLoss
        proto_predictor: Optional PrototypicalityPredictor for auxiliary task
        scaler: Optional GradScaler for mixed precision
        global_step: int - running step count for wandb logging

    Returns:
        dict with loss, metrics, and updated global_step
    """
    use_amp = scaler is not None
    accum_steps = max(1, getattr(config, 'gradient_accumulation_steps', 1))
    model.train()
    if domain_discriminator is not None:
        domain_discriminator.train()
    if modality_discriminator is not None:
        modality_discriminator.train()
    if proto_predictor is not None:
        proto_predictor.train()

    muted_synth_count = 0
    # Mean blend coefficient, reported so a miscalibrated VAD rule is visible
    # in the log rather than only in the final metrics.
    muted_lam_sum = 0.0
    muted_lam_n = 0
    total_loss = 0
    total_primary_loss = 0
    total_contrastive_loss = 0
    total_adversarial_loss = 0
    total_modality_adversarial_loss = 0
    total_proto_pred_loss = 0
    total_vadmix_loss = 0
    total_aux_vad_loss = 0
    total_vrex_var = 0.0
    vrex_var_steps = 0
    last_vrex_var = 0.0

    # Build prototype tensor once per epoch for VADmix (CPU; moved to device per step)
    use_vadmix = getattr(config, 'use_vadmix', False)
    vadmix_prototypes = None
    if use_vadmix:
        vadmix_prototypes = build_prototype_tensor(config.expected_vad, config.num_classes)

    all_predictions = []
    all_labels = []
    all_vad_preds = []
    all_vad_targets = []

    # Contrastive weight warm-up: linear ramp from 0 to target over warmup epochs
    warmup_epochs = getattr(config, 'contrastive_warmup_epochs', 5)
    if warmup_epochs > 0 and current_epoch < warmup_epochs:
        warmup_factor = current_epoch / warmup_epochs
    else:
        warmup_factor = 1.0
    effective_contrastive_weight = config.contrastive_weight * warmup_factor

    optimizer.zero_grad()
    num_batches = len(dataloader)
    aux_mix_count = 0

    # Class VAD statistics for the soft-label-without-mixup arm. Built here
    # rather than per batch so the conversion does not repeat, and before
    # the loop because the primary loss needs them, which runs earlier in
    # the iteration than the mixup block that builds its own copy.
    _vs_means = _vs_inv = None
    if getattr(config, 'use_vad_soft_labels', False):
        if getattr(config, '_proto_means', None) is None:
            raise ValueError(
                "use_vad_soft_labels needs the fitted class statistics; "
                "enable use_proto_predictor so they are computed.")
        for _incompatible in ('use_vrex', 'use_prototypical_weighting',
                              'use_salience_weighting',
                              'use_prototypical_label_smoothing'):
            if getattr(config, _incompatible, False):
                raise ValueError(
                    f"use_vad_soft_labels cannot be combined with "
                    f"{_incompatible}: both decide how the primary loss is "
                    f"reduced, and the combination would silently change it.")
        _vs_means = torch.as_tensor(config._proto_means, dtype=torch.float32,
                                    device=device)
        _vs_inv = torch.as_tensor(config._proto_inv_covs,
                                  dtype=torch.float32, device=device)

    for batch_idx, batch in enumerate(dataloader):
        # Holder for the per-sample PCGrad reference gradient. Defined
        # at loop scope: it is written inside the auxiliary branch and
        # read again at backward, so a conditional definition would be
        # a NameError on any batch that skips that branch.
        _ps_ref = [None]
        is_last_batch = (batch_idx == num_batches - 1)
        should_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch
        # Move to device
        labels = batch['label'].to(device)

        # Prepare inputs based on modality and audio encoder type
        model_inputs = _prepare_model_inputs(batch, config, model, device)

        # Forward pass with embedding extraction (inside autocast for mixed precision)
        _ce_src = getattr(config, 'ce_weight_source', 'vad')
        _ce_needs_modal = (getattr(config, 'use_prototypical_weighting', False)
                           and _ce_src in ('agreement', 'both')
                           and config.modality == 'both')
        use_embeddings = (config.use_contrastive
                          or getattr(config, 'use_domain_adversarial', False)
                          or getattr(config, 'use_modality_adversarial', False)
                          or getattr(config, 'use_proto_predictor', False)
                          or getattr(config, 'use_vadmix', False)
                          or getattr(config, 'use_aux_vad_cluster', False)
                          or getattr(config, 'use_muted_mixup', False)
                          or getattr(config, 'use_detector_fusion', False)
                          or _ce_needs_modal)
        raw_embeddings = None
        embeddings_norm = None
        modal_features = None

        with autocast('cuda', enabled=use_amp):
            if use_embeddings:
                # When modality adversarial is active, keep gradients flowing into
                # the audio/text encoders via modal_features (no detach)
                if (modality_discriminator is not None
                        or proto_predictor_audio is not None
                        or proto_predictor_text is not None
                        or bool(getattr(config, 'muted_mixup_per_modality',
                                        False))):
                    model_inputs['detach_modal_features'] = False
                result = model(**model_inputs, return_embeddings=True)
                logits, projected_embeddings, raw_embeddings, modal_features = result
                embeddings_norm = F.normalize(projected_embeddings, p=2, dim=1)
            else:
                logits = model(**model_inputs)

            # Primary loss
            if config.task_type == "regression":
                vad_targets = batch['vad'].to(device)
                loss_primary = criterion(logits, vad_targets)
                all_vad_preds.append(logits.detach().cpu().float().numpy())
                all_vad_targets.append(vad_targets.cpu().numpy())
            else:
                # Feed the confidence tracker before any branch decides how
                # to build the target. Probabilities are collapsed first when
                # the head is widened so the tracker always sees a 4-way
                # distribution.
                if confidence_tracker is not None and 'sample_index' in batch:
                    n_cls = int(getattr(config, 'num_classes', 4))
                    track_logits = logits
                    if logits.shape[-1] == 2 * n_cls:
                        track_logits = torch.log(
                            collapse_sub_logits_to_class_probs(
                                logits, n_cls).clamp_min(1e-12)
                        )
                    elif getattr(config, 'use_latent_prototypes', False):
                        track_logits = torch.log(
                            collapse_prototype_logits(
                                logits, n_cls,
                                int(getattr(config, 'prototypes_per_class', 2)),
                            ).clamp_min(1e-12)
                        )
                    elif getattr(config, 'use_hierarchical_head', False):
                        track_logits = torch.log(
                            compose_class_probs(logits).clamp_min(1e-12)
                        )
                    elif logits.shape[-1] == n_cls - 1:
                        # Evidence heads: synthesize a 4-way distribution.
                        # softmax(log p) recovers p exactly, so the tracker
                        # sees the same probabilities the decision rule uses.
                        track_logits = torch.log(
                            evidence_class_probs(logits).clamp_min(1e-12)
                        )
                    confidence_tracker.update(
                        track_logits, labels, batch['sample_index'],
                    )

                # Per-sample ambiguity weights from the tracker EMA. None
                # during warmup, so early uninformative confidence does not
                # steer the loss.
                amb_w = None
                if ambiguity_weights is not None and 'sample_index' in batch:
                    amb_w = ambiguity_weights(
                        batch['sample_index'], current_epoch, device,
                    )

                use_proto_atyp = proto_atyp_ctx is not None
                use_proto_weight = getattr(config, 'use_prototypical_weighting', False)
                use_label_smooth = getattr(config, 'use_prototypical_label_smoothing', False)
                use_vad_soft = getattr(config, 'use_vad_soft_labels', False)
                # Direction-aware soft labels. soft_label_mode picks the
                # target ("hard", "raw" or "realness") and
                # realness_loss_weight switches lever two independently, so
                # either lever can be run alone or both together.
                soft_mode = str(getattr(config, 'soft_label_mode', 'hard'))
                realness_weight_on = bool(
                    getattr(config, 'realness_loss_weight', False))
                realness_floor = float(
                    getattr(config, 'realness_weight_floor', 0.25))
                use_realness = (soft_mode in ('raw', 'realness')
                                or realness_weight_on)
                ot_weight = float(getattr(config, 'ot_weight', 0.0))
                use_salience_weight = getattr(config, 'use_salience_weighting', False)
                use_vrex = getattr(config, 'use_vrex', False)

                if use_proto_atyp:
                    # Sub-label CE on the widened 2 * num_classes head. Class
                    # weights are duplicated across the (proto, atyp) pair for
                    # each class so imbalance rebalance still holds.
                    _src = proto_atyp_ctx.get('center_source')
                    _pending = (_src == 'confidence'
                                and (confidence_tracker is None
                                     or not confidence_tracker.frozen))
                    if _pending:
                        # Split not frozen yet. Train the parent task only,
                        # by collapsing the widened head to class
                        # probabilities. The atypical slots stay unused
                        # until the split exists.
                        sub_labels_batch = None
                        class_probs_pre = collapse_sub_logits_to_class_probs(
                            logits, proto_atyp_ctx['num_classes'],
                        )
                        cls_w_pre = (criterion.weight
                                     if hasattr(criterion, 'weight') else None)
                        loss_primary = F.nll_loss(
                            torch.log(class_probs_pre.clamp_min(1e-12)),
                            labels, weight=cls_w_pre,
                        )
                    elif _src == 'confidence':
                        sub_labels_batch = confidence_tracker.sub_labels(
                            labels, batch['sample_index'], device,
                        )
                    elif _src == 'random':
                        sub_labels_batch = batch_random_sub_labels(
                            batch, proto_atyp_ctx['num_classes'],
                            proto_atyp_ctx['seed'], device,
                        )
                    elif _src == 'nrc':
                        sub_labels_batch = batch_nrc_sub_labels(
                            batch, proto_atyp_ctx['centers'],
                            proto_atyp_ctx['thresholds'],
                            proto_atyp_ctx['num_classes'], device,
                        )
                    else:
                        sub_labels_batch = batch_sub_labels(
                            batch, proto_atyp_ctx['centers'],
                            proto_atyp_ctx['thresholds'],
                            proto_atyp_ctx['num_classes'], device,
                        )
                    cls_weights_pa = criterion.weight if hasattr(criterion, 'weight') else None
                    sub_weights = (
                        cls_weights_pa.repeat_interleave(2)
                        if cls_weights_pa is not None else None
                    )
                    sibling_w = float(getattr(config, 'proto_atyp_sibling_weight', 0.0))
                    if sub_labels_batch is None:
                        pass  # loss_primary already set by the pre-freeze path
                    elif sibling_w > 0.0:
                        # Two-hot soft target: put (1 - w) mass on the correct
                        # sub-label and w on the same-class sibling. Since
                        # sub = 2*class + is_atypical, flipping bit 0 gives the
                        # sibling. Cross-class mistakes still get zero mass.
                        num_sub = 2 * proto_atyp_ctx['num_classes']
                        soft = torch.zeros(
                            labels.size(0), num_sub, device=device, dtype=logits.dtype,
                        )
                        row = torch.arange(labels.size(0), device=device)
                        sibling = sub_labels_batch ^ 1
                        soft[row, sub_labels_batch] = 1.0 - sibling_w
                        soft[row, sibling] = sibling_w
                        loss_primary = soft_cross_entropy(logits, soft, sub_weights)
                        if loss_primary.dim() > 0:
                            loss_primary = loss_primary.mean()
                    else:
                        loss_primary = F.cross_entropy(
                            logits, sub_labels_batch, weight=sub_weights, reduction='sum'
                        ) / labels.size(0)
                elif getattr(config, 'use_latent_prototypes', False):
                    # Only the parent class is supervised; which prototype a
                    # sample routes through is latent.
                    cls_w_lp = (criterion.weight
                                if hasattr(criterion, 'weight') else None)
                    loss_primary, _ = latent_prototype_loss(
                        logits, labels,
                        int(getattr(config, 'num_classes', 4)),
                        int(getattr(config, 'prototypes_per_class', 2)),
                        class_weights=cls_w_lp,
                        sample_weights=amb_w,
                        usage_entropy_weight=float(getattr(
                            config, 'prototype_usage_entropy', 0.0)),
                    )
                elif getattr(config, 'use_hierarchical_head', False):
                    # P(neutral) = 1 - P(emotional); P(c) = P(emotional) *
                    # P(c | emotional). Neutral samples train only the
                    # presence detector, never the emotion head.
                    cls_w_h = (criterion.weight
                               if hasattr(criterion, 'weight') else None)
                    loss_primary = hierarchical_loss(
                        logits, labels,
                        class_weights=cls_w_h,
                        sample_weights=amb_w,
                        detector_pos_weight=getattr(
                            config, 'hierarchical_detector_pos_weight', None),
                        detector_weight=float(getattr(
                            config, 'hierarchical_detector_weight', 1.0)),
                    )
                elif getattr(config, 'use_evidence_heads', False):
                    # Neutral as absence: per-emotion BCE. Neutral samples
                    # are negatives for every head; ambiguity weights (when
                    # active) upweight the samples the model finds hard.
                    # pos_weight already balances each head's positive and
                    # negative terms at n_neg/n_pos. The inverse-frequency
                    # class weight corrects the same imbalance a second
                    # time: together they gave the sad head a 35.5x pos:neg
                    # ratio where 8.4x is correct, every head over-fired,
                    # and neutral recall collapsed to 0.036. Use one, not
                    # both. pos_weight is the right one because it acts
                    # inside the binary problem the head actually solves.
                    loss_primary = evidence_bce_loss(
                        logits, labels,
                        int(getattr(config, 'num_classes', 4)),
                        class_weights=None,
                        sample_weights=amb_w,
                        pos_weight=evidence_pos_weight,
                    )
                elif use_vad_soft:
                    # VAD-derived soft labels WITHOUT any mixup. The target
                    # is the class posterior at the sample's OWN annotated
                    # VAD point, the same q the mixup arms use.
                    #
                    # This separates two things the mixup arms confound:
                    # whether soft labels derived from the affect
                    # annotations are what helps, and whether blending is
                    # needed to deliver them. The endpoint-posterior target
                    # lam * q(v_a) + (1 - lam) * q(v_b) is exactly this
                    # relabeling followed by STANDARD mixup, because mixup's
                    # label rule is linear and the two operations commute.
                    # So if this arm alone reproduces the gain, the blending
                    # is not part of the mechanism at all.
                    vad_now = batch['vad'].to(device).float()
                    soft_targets = gaussian_posterior(vad_now, _vs_means,
                                                      _vs_inv)
                    if bool(getattr(config, 'vad_soft_label_shuffle', False)):
                        # Placebo. Permute the soft labels WITHIN each class,
                        # so every sample still gets a soft label typical of
                        # its own class, with the same marginal distribution
                        # and the same entropy, but not the one its own VAD
                        # implies. Removes the per-sample affect information
                        # and nothing else. A shuffle across classes would
                        # instead be mislabelling, which is a different and
                        # uninformative experiment.
                        perm = torch.arange(labels.size(0), device=device)
                        for c in labels.unique():
                            idx = (labels == c).nonzero(as_tuple=True)[0]
                            if idx.numel() > 1:
                                perm[idx] = idx[torch.randperm(
                                    idx.numel(), device=device)]
                        soft_targets = soft_targets[perm]
                    # Rows carrying no VAD annotation keep their hard label.
                    has_vad = vad_now.abs().sum(dim=1) > 0
                    if not bool(has_vad.all()):
                        one_hot = F.one_hot(
                            labels, num_classes=config.num_classes).float()
                        soft_targets = torch.where(has_vad.unsqueeze(1),
                                                   soft_targets, one_hot)
                    cls_weights = (criterion.weight
                                   if hasattr(criterion, 'weight') else None)
                    per_sample_loss = soft_cross_entropy(logits, soft_targets,
                                                         cls_weights)
                    # Reduce here. The per-sample weighting chain further
                    # down (vrex / salience / proto weighting) has no final
                    # else, so a target kind outside its cases would leave
                    # loss_primary unset. Combining this arm with those
                    # schemes is refused at setup rather than silently
                    # taking a different reduction.
                    loss_primary = per_sample_loss.mean()
                elif soft_mode in ('lowrank', 'kernel', 'smooth'):
                    # Continuous targets drawn from the label space itself
                    # rather than from VAD.
                    #
                    # lowrank: the panel's distribution projected onto the
                    # leading principal components of the label
                    # distribution and reconstructed. The raw distribution
                    # spends part of every target on directions a
                    # class-balanced probe found unrecoverable from audio,
                    # and this removes them. Per utterance, so rows the
                    # cache misses keep their hard label.
                    #
                    # kernel: how often another annotator of the same
                    # utterance chose each class, given one chose this
                    # one. Measured, nothing to tune, identical for every
                    # row of a class, so it covers unanimous rows too and
                    # carries to corpora with no panel data.
                    one_hot = F.one_hot(
                        labels, num_classes=config.num_classes).float()
                    if soft_mode == 'kernel':
                        soft_targets = batch['cont_kernel'].to(device).float()
                    elif soft_mode == 'smooth':
                        # Entropy-matched control: the raw annotator
                        # distribution mixed toward uniform by a single
                        # epsilon, solved so its mean entropy equals the
                        # low-rank target's. Plain label smoothing, so a
                        # tie against rl_lowrank says that arm's gain is
                        # softness rather than structure.
                        sm = batch['cont_smooth'].to(device).float()
                        keep = (batch['cont_mask'].to(device).float()
                                > 0).unsqueeze(1)
                        soft_targets = torch.where(keep, sm, one_hot)
                    else:
                        low = batch['cont_lowrank'].to(device).float()
                        keep = (batch['cont_mask'].to(device).float()
                                > 0).unsqueeze(1)
                        soft_targets = torch.where(keep, low, one_hot)
                    # Guard against a target that does not normalise, which
                    # would silently rescale this row's loss.
                    soft_targets = soft_targets / soft_targets.sum(
                        dim=1, keepdim=True).clamp_min(1e-8)
                    cls_weights = (criterion.weight
                                   if hasattr(criterion, 'weight') else None)
                    per_sample_loss = soft_cross_entropy(logits, soft_targets,
                                                         cls_weights)
                    loss_primary = per_sample_loss.mean()
                elif use_realness:
                    # Direction-aware soft labels. A panel that split three
                    # to two tells you it disagreed, not whether the
                    # utterance genuinely sits between two emotions or the
                    # annotators were unreliable about that distinction.
                    # alpha is the share of this panel's own disagreement
                    # running along label-pair axes that carry real
                    # structure where the utterance sits in VAD space,
                    # fitted by analysis.build_realness_map on the official
                    # Train split alone.
                    #
                    # Two levers, independently switchable, both reading the
                    # same alpha. "realness" interpolates the target toward
                    # the annotator distribution by alpha; "raw" trusts that
                    # distribution everywhere and is the comparison the
                    # claim rests on, since beating hard labels alone would
                    # only show that soft labels help.
                    r_alpha = batch['realness_alpha'].to(device).float()
                    r_probs = batch['realness_probs'].to(device).float()
                    r_mask = batch['realness_mask'].to(device).float()

                    one_hot = F.one_hot(
                        labels, num_classes=config.num_classes).float()
                    totals = r_probs.sum(dim=1, keepdim=True)
                    soft = torch.where(totals > 0, r_probs / totals.clamp_min(1e-8),
                                       one_hot)
                    if soft_mode == 'raw':
                        blend = soft
                    else:
                        a = r_alpha.clamp(0.0, 1.0).unsqueeze(1)
                        blend = (1.0 - a) * one_hot + a * soft
                    # A row the map does not cover is unmeasured, not
                    # measured as noisy, so it keeps the hard label rather
                    # than a fabricated distribution.
                    keep = (r_mask > 0).unsqueeze(1)
                    soft_targets = torch.where(keep, blend, one_hot)

                    cls_weights = (criterion.weight
                                   if hasattr(criterion, 'weight') else None)
                    per_sample_loss = soft_cross_entropy(logits, soft_targets,
                                                         cls_weights)

                    if realness_weight_on:
                        # Lever two. A panel whose split runs along noise
                        # axes is evidence about the annotators rather than
                        # the utterance, so it earns less influence. The
                        # floor keeps it contributing, because its majority
                        # label is still the best guess available and its
                        # audio still teaches something.
                        #
                        # Normalised so the effective learning rate matches
                        # the unweighted arms. The normaliser is computed
                        # over the COVERED rows only, and uncovered rows are
                        # pinned at one. Scaling everything by the batch
                        # mean instead would push uncovered rows above every
                        # measured row, since the covered weights average
                        # below one, and rows with missing metadata would
                        # end up with the most influence in the batch.
                        covered = r_mask > 0
                        raw_w = realness_floor + (1.0 - realness_floor) * \
                            r_alpha.clamp(0.0, 1.0)
                        if bool(covered.any()):
                            scale = (covered.sum().float()
                                     / raw_w[covered].sum().clamp_min(1e-8))
                            raw_w = raw_w * scale
                        raw_w = torch.where(covered, raw_w,
                                            torch.ones_like(raw_w))
                        loss_primary = (per_sample_loss * raw_w).mean()
                    else:
                        loss_primary = per_sample_loss.mean()
                elif use_label_smooth:
                    # Prototypicality-based label smoothing: atypical samples get softer labels
                    difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                    beta = getattr(config, 'label_smoothing_beta', 0.5)
                    max_smooth = getattr(config, 'label_smoothing_max', 0.6)
                    smoothing = (beta * difficulties).clamp(0.0, max_smooth)  # [B]

                    # Build soft targets: (1 - smoothing) * one_hot + smoothing / num_classes
                    one_hot = F.one_hot(labels, num_classes=config.num_classes).float()  # [B, C]
                    smoothing = smoothing.unsqueeze(1)  # [B, 1]
                    soft_targets = (1.0 - smoothing) * one_hot + smoothing / config.num_classes

                    # Get class weights from criterion if available
                    cls_weights = criterion.weight if hasattr(criterion, 'weight') else None
                    per_sample_loss = soft_cross_entropy(logits, soft_targets, cls_weights)
                elif use_proto_weight or use_salience_weight:
                    # Standard CE with reduction='none'
                    per_sample_loss = criterion(logits, labels)  # [B]
                elif use_vrex:
                    # Per-sample CE so we can group by corpus and compute variance
                    cls_weights = criterion.weight if hasattr(criterion, 'weight') else None
                    per_sample_loss = F.cross_entropy(
                        logits, labels, weight=cls_weights, reduction='none'
                    )  # [B]
                elif intensity_ctx is not None:
                    # Intensity-dependent soft targets. Weak instances of an
                    # emotion give up a share of their target mass to
                    # neutral; strong ones stay effectively one-hot. Neutral
                    # samples are untouched.
                    vad_batch = torch.stack([
                        batch['valence'].to(device).float(),
                        batch['arousal'].to(device).float(),
                        batch['dominance'].to(device).float(),
                    ], dim=1)
                    rel_intensity = class_relative_intensity(
                        vad_batch, labels,
                        intensity_ctx['class_ranges'],
                        intensity_ctx['neutral_vad'],
                    )
                    soft_targets = neutral_soft_targets(
                        labels, rel_intensity,
                        int(getattr(config, 'num_classes', 4)),
                        alpha=float(getattr(config, 'neutral_soft_alpha', 0.3)),
                    )
                    cls_weights_ns = (criterion.weight
                                      if hasattr(criterion, 'weight') else None)
                    adj_logits = (logits if logit_adjust is None
                                  else logits + logit_adjust)
                    per_sample_ns = soft_cross_entropy(
                        adj_logits, soft_targets.to(adj_logits.dtype),
                        cls_weights_ns,
                    )
                    loss_primary = (per_sample_ns.mean()
                                    if per_sample_ns.dim() > 0 else per_sample_ns)
                else:
                    # Sum-based normalization: sum(w*CE) / B instead of sum(w*CE) / sum(w).
                    # When divided by accum_steps at backward, gives sum(w*CE) / (accum_steps * B)
                    # which is identical across micro-batches regardless of class composition.
                    cls_weights = criterion.weight if hasattr(criterion, 'weight') else None
                    # Logit adjustment shifts the decision boundary by the log
                    # class prior during training only. Eval uses raw logits,
                    # which yields a prior-free (balanced) classifier without
                    # amplifying minority-class gradients the way reweighting
                    # does. See Menon et al., ICLR 2021.
                    adj_logits = logits if logit_adjust is None else logits + logit_adjust
                    # A class-dependent margin has to be applied to the logits
                    # the loss actually sees. This branch calls F.cross_entropy
                    # directly and only borrows criterion.weight, so a margin
                    # living inside the criterion object would be constructed,
                    # printed, and silently ignored.
                    ldam_margins = getattr(criterion, 'margins', None)
                    if ldam_margins is not None:
                        adj_logits = adj_logits - (
                            F.one_hot(labels.long(), adj_logits.size(1)).float()
                            * ldam_margins.to(adj_logits.device))
                        adj_logits = adj_logits * float(
                            getattr(criterion, 'scale', 1.0))
                    if amb_w is None:
                        loss_primary = F.cross_entropy(
                            adj_logits, labels, weight=cls_weights, reduction='sum'
                        ) / labels.size(0)
                    else:
                        # Ambiguity weighting on the plain softmax head. This
                        # branch previously ignored amb_w entirely, so an
                        # ambiguity arm with no other mechanism was bit-identical
                        # to its baseline (sd 0.00 across seeds) rather than
                        # merely null. Keep the sum/B normalisation so the
                        # gradient scale matches the unweighted path.
                        per_sample = F.cross_entropy(
                            adj_logits, labels, weight=cls_weights,
                            reduction='none',
                        )
                        loss_primary = (per_sample * amb_w).sum() / labels.size(0)

                if use_vrex:
                    # VREx: penalize variance of per-corpus mean losses.
                    # Corpora are treated as IRM environments. Features that work
                    # consistently across corpora are preferred over corpus-specific shortcuts.
                    corpus_names = batch['dataset']
                    unique_corpora = sorted(set(corpus_names))
                    per_corpus_losses = []
                    for c in unique_corpora:
                        mask = torch.tensor(
                            [n == c for n in corpus_names], device=device, dtype=torch.bool
                        )
                        if mask.any():
                            per_corpus_losses.append(per_sample_loss[mask].mean())
                    if len(per_corpus_losses) > 1:
                        per_corpus_tensor = torch.stack(per_corpus_losses)
                        loss_mean_env = per_corpus_tensor.mean()
                        loss_var_env = per_corpus_tensor.var(unbiased=False)
                        warmup_frac = getattr(config, 'vrex_warmup_frac', 0.3)
                        warmup_epochs_v = max(1, int(warmup_frac * config.num_epochs))
                        warm = min(1.0, current_epoch / warmup_epochs_v)
                        lam_vrex = getattr(config, 'vrex_lambda', 10.0) * warm
                        loss_primary = loss_mean_env + lam_vrex * loss_var_env
                        last_vrex_var = float(loss_var_env.detach().item())
                    else:
                        # Single corpus in batch: no variance signal, fall back to mean CE
                        loss_primary = per_sample_loss.mean()
                        last_vrex_var = 0.0
                elif use_salience_weight:
                    # Emphasise samples displaced toward the neutral region:
                    # quiet emotion and activated neutral. Weights are mean 1
                    # within each class, so the inverse-frequency class
                    # balance already applied in the CE is left untouched and
                    # the only thing varying is which samples inside a class
                    # carry the gradient.
                    if getattr(config, 'salience_shuffle', False):
                        sal_d = placebo_difficulty(
                            tuple(batch.get('transcript', [''] * len(labels))),
                            labels, device,
                        )
                    else:
                        sal_d = salience_difficulty(
                            batch['vad'].to(device), labels, salience_stats,
                        )
                    sal_w = salience_weights(
                        sal_d, labels,
                        beta=float(getattr(config, 'salience_beta', 0.5)),
                        num_classes=int(getattr(config, 'num_classes', 4)),
                        clip=float(getattr(config, 'salience_clip', 3.0)),
                        scope=str(getattr(config, 'salience_scope', 'both')),
                    )
                    loss_primary = (per_sample_loss * sal_w).mean()

                elif use_proto_weight or use_label_smooth:
                    if use_proto_weight:
                        # Weight per-sample loss by prototypicality
                        if not use_label_smooth:
                            ce_src = getattr(config, 'ce_weight_source', 'vad')
                            if centroid_tracker is not None:
                                # On-graph: gradients flow back to centroids in grad mode
                                d_vad = batch_difficulty_tensor(batch, centroid_tracker(), device)
                            else:
                                d_vad = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                            if ce_src == 'mahal':
                                # Atypicality as this project defines it: the
                                # Mahalanobis distance from the sample's own
                                # class centre, using covariances fitted on
                                # the training split. Distinct from d_vad,
                                # which is a Euclidean distance to hardcoded
                                # centroids.
                                #
                                # This enters as a WEIGHT on the main loss,
                                # not as a regression target, so it is not
                                # something an auxiliary VAD head can imitate,
                                # and being a norm it is not an affine
                                # function of VAD.
                                _v = batch['vad'].numpy().astype(np.float64)
                                _pd = tuple(getattr(config, 'proto_vad_dims',
                                                    (0, 1, 2)))
                                if len(_pd) != _v.shape[1]:
                                    _v = _v[:, list(_pd)]
                                difficulties = torch.tensor(
                                    mahalanobis_distance(
                                        _v, batch['label'].numpy(),
                                        config._ce_means, config._ce_inv_covs),
                                    dtype=torch.float32, device=device)
                            elif ce_src == 'vad':
                                difficulties = d_vad
                            else:
                                d_agree = compute_crossmodal_agreement(modal_features, device)
                                if d_agree is None:
                                    difficulties = d_vad
                                elif ce_src == 'agreement':
                                    difficulties = d_agree
                                else:  # "both"
                                    w = getattr(config, 'ce_weight_both_vad_w', 0.5)
                                    difficulties = w * d_vad + (1.0 - w) * d_agree
                        # else: reuse difficulties already computed by label smoothing block above ([B])
                        alpha = getattr(config, 'prototypical_weighting_alpha', 2.0)
                        sign = 1.0 if getattr(config, 'ce_weight_invert', False) else -1.0
                        sample_weights = torch.exp(sign * alpha * difficulties)  # [B]
                        # Normalize so weights have mean 1 (preserves gradient scale)
                        sample_weights = sample_weights * (sample_weights.numel() / (sample_weights.sum() + 1e-8))
                        loss_primary = (per_sample_loss * sample_weights).mean()
                    else:
                        loss_primary = per_sample_loss.mean()

                # Transport term: price WHICH wrong class is predicted.
                #
                # Cross-entropy reads only the mass on the true class, so
                # it charges the same for every mistake. Every target-side
                # arm in this file therefore spent its structure somewhere
                # the loss is blind to. This charges by distance instead,
                # using a cost measured from how often annotators
                # substitute each pair: confusing neutral with sad (0.698)
                # costs less than confusing happy with angry (1.228).
                #
                # Against a one-hot target the transport distance is exact
                # in closed form, because all mass must move to the true
                # class: W(p, delta_y) = sum_a p[a] C[a, y]. It is zero at
                # p = delta_y, so it agrees with cross-entropy at the
                # optimum and cannot starve a class, which is how the
                # measured smoothing kernel destroyed sad recall.
                #
                # Prediction on record: UAR averages per-class recall, and
                # recall is unaffected by WHERE a row's errors land. So
                # this should move F1 and accuracy more than UAR, and a
                # flat UAR here is the expected result rather than a null.
                if ot_weight > 0.0 and ot_cost is not None:
                    cost_rows = ot_cost.to(logits.device)[labels]
                    loss_ot = (F.softmax(logits, dim=1)
                               * cost_rows).sum(dim=1).mean()
                    loss_primary = loss_primary + ot_weight * loss_ot

                # Anti-neutral margin: push emotional samples' true logit
                # past the neutral logit, hardest on the ambiguous ones.
                # Softmax head only; evidence heads have no neutral logit.
                if (getattr(config, 'use_anti_neutral_margin', False)
                        and not use_proto_atyp
                        and not getattr(config, 'use_evidence_heads', False)):
                    loss_anm = anti_neutral_margin_loss(
                        logits, labels,
                        margin=float(getattr(config, 'anti_neutral_margin', 1.0)),
                        sample_weights=amb_w,
                    )
                    loss_primary = loss_primary + (
                        float(getattr(config, 'anti_neutral_weight', 0.5))
                        * loss_anm
                    )

                if use_proto_atyp:
                    class_probs = collapse_sub_logits_to_class_probs(
                        logits, proto_atyp_ctx['num_classes']
                    )
                    preds = torch.argmax(class_probs, dim=-1).cpu().numpy()
                elif getattr(config, 'use_latent_prototypes', False):
                    preds = collapse_prototype_logits(
                        logits, int(getattr(config, 'num_classes', 4)),
                        int(getattr(config, 'prototypes_per_class', 2)),
                    ).argmax(dim=-1).cpu().numpy()
                elif getattr(config, 'use_hierarchical_head', False):
                    preds = hierarchical_predictions(logits).cpu().numpy()
                elif getattr(config, 'use_evidence_heads', False):
                    preds = evidence_predictions(
                        logits,
                        float(getattr(config, 'evidence_threshold', 0.5)),
                    ).cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_predictions.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                # EMA update for learned centroids (no gradient)
                if centroid_tracker is not None and centroid_tracker.mode == "ema":
                    centroid_tracker.ema_update(batch, device)

            # Contrastive loss
            loss_contrastive = torch.tensor(0.0, device=device)
            if config.use_contrastive and contrastive_criterion is not None and embeddings_norm is not None:
                loss_type = config.contrastive_loss_type

                if loss_type == 'prototype_anchored_multiDS':
                    difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                    corpus_names = batch['dataset']
                    unique_corpora = sorted(set(corpus_names))
                    corpus_to_id = {name: i for i, name in enumerate(unique_corpora)}
                    dataset_ids = torch.tensor([corpus_to_id[n] for n in corpus_names], device=device)
                    loss_contrastive = contrastive_criterion(embeddings_norm, labels, difficulties, dataset_ids)
                elif loss_type.startswith('prototypical') or loss_type == 'prototype_anchored':
                    # Multi-view prototypicality: combine VAD + cross-modal + embedding views
                    use_multiview = getattr(config, 'use_multiview_prototypicality', False)
                    if use_multiview and loss_type == 'prototype_anchored':
                        prototypes = contrastive_criterion.prototypes
                        difficulties = compute_multiview_difficulty(
                            batch, config.expected_vad, modal_features,
                            embeddings_norm, labels, prototypes, device,
                            w_vad=getattr(config, 'mv_weight_vad', 0.4),
                            w_cross=getattr(config, 'mv_weight_cross', 0.3),
                            w_embed=getattr(config, 'mv_weight_embed', 0.3),
                        )
                    else:
                        difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                    loss_contrastive = contrastive_criterion(embeddings_norm, labels, difficulties)
                else:
                    loss_contrastive = contrastive_criterion(embeddings_norm, labels)

            # Domain adversarial loss
            loss_adversarial = torch.tensor(0.0, device=device)
            if domain_discriminator is not None and domain_adv_loss is not None and raw_embeddings is not None:
                corpus_names = batch['dataset']
                unique_corpora = sorted(set(corpus_names))
                corpus_to_id = {name: i for i, name in enumerate(unique_corpora)}
                domain_labels = torch.tensor([corpus_to_id[n] for n in corpus_names], device=device)

                # Only compute if batch has multiple domains
                if len(unique_corpora) > 1:
                    domain_logits = domain_discriminator(raw_embeddings)

                    # Prototypicality-weighted adversarial loss
                    if getattr(config, 'use_prototypical_adversarial', True):
                        difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                        loss_adversarial = domain_adv_loss(domain_logits, domain_labels, difficulties)
                    else:
                        loss_adversarial = domain_adv_loss(domain_logits, domain_labels)

            # Modality-level adversarial loss: GRL applied pre-fusion, per modality
            loss_modality_adversarial = torch.tensor(0.0, device=device)
            if (modality_discriminator is not None and domain_adv_loss is not None
                    and modal_features is not None):
                corpus_names = batch['dataset']
                unique_corpora = sorted(set(corpus_names))
                corpus_to_id = {name: i for i, name in enumerate(unique_corpora)}
                domain_labels_mod = torch.tensor(
                    [corpus_to_id[n] for n in corpus_names], device=device
                )

                if len(unique_corpora) > 1:
                    audio_feat = modal_features['audio']
                    text_feat = modal_features['text']
                    audio_dom_logits, text_dom_logits = modality_discriminator(
                        audio_feat, text_feat
                    )

                    if getattr(config, 'use_prototypical_adversarial', True):
                        difficulties_ma = batch_calculate_difficulty(
                            batch, config.expected_vad
                        ).to(device)
                        loss_audio_adv = domain_adv_loss(
                            audio_dom_logits, domain_labels_mod, difficulties_ma
                        )
                        loss_text_adv = domain_adv_loss(
                            text_dom_logits, domain_labels_mod, difficulties_ma
                        )
                    else:
                        loss_audio_adv = domain_adv_loss(audio_dom_logits, domain_labels_mod)
                        loss_text_adv = domain_adv_loss(text_dom_logits, domain_labels_mod)

                    loss_modality_adversarial = 0.5 * (loss_audio_adv + loss_text_adv)

            # Auxiliary prototypicality prediction loss
            loss_proto_pred = torch.tensor(0.0, device=device)
            if proto_predictor is not None and raw_embeddings is not None:
                target_kind = str(getattr(config, 'proto_target', 'scalar'))
                if target_kind == 'scalar' and getattr(
                        config, '_proto_means', None) is None:
                    difficulties = batch_calculate_difficulty(
                        batch, config.expected_vad).to(device)
                else:
                    vad_np = batch['vad'].numpy().astype(np.float64)
                    _pdims = getattr(config, '_proto_dims', (0, 1, 2))
                    if len(_pdims) != vad_np.shape[1]:
                        vad_np = vad_np[:, list(_pdims)]
                    lab_np = batch['label'].numpy()
                    means = config._proto_means
                    if target_kind == 'allclass':
                        difficulties = torch.tensor(
                            all_class_residuals(vad_np, lab_np, means,
                                                whiten=config._proto_whiten),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'signed':
                        difficulties = torch.tensor(
                            signed_axis_targets(vad_np, lab_np, means),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'signedw':
                        # Same decomposition, computed after whitening, so it
                        # is compared against the whitened residual on equal
                        # terms rather than against an unwhitened one.
                        _w = getattr(config, '_proto_whiten', None)
                        if _w is None:
                            difficulties = torch.tensor(
                                signed_axis_targets(vad_np, lab_np, means),
                                dtype=torch.float32, device=device)
                        else:
                            difficulties = torch.tensor(
                                signed_axis_whitened(vad_np, lab_np, means, _w),
                                dtype=torch.float32, device=device)
                    elif target_kind == 'protoscore':
                        # Bounded prototypicality: 1 at the class centre,
                        # decaying towards 0 as the sample becomes atypical.
                        # Unlike 'invresidual' this is NOT affine, so it
                        # changes the geometry of the loss rather than only
                        # relabelling it. Whitened distances have a long right
                        # tail, so MSE on the raw distance is dominated by the
                        # most atypical samples; exp(-d) compresses that tail
                        # and spends the head's resolution on samples near the
                        # centre, which is where the neutral against emotional
                        # boundary sits. Must precede the generic mahalanobis
                        # branch below, which would otherwise capture it.
                        difficulties = torch.tensor(
                            np.exp(-mahalanobis_distance(
                                vad_np, lab_np, means,
                                config._proto_inv_covs)),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'offlabel':
                        # The secondary-label target: where annotators who did
                        # NOT vote the majority class went instead.
                        #
                        # Take the annotator vote distribution over the four
                        # classes, zero the sample's own class, renormalise.
                        # For a sad utterance this asks "given it is sad,
                        # which way do raters drift", and the neutral
                        # component of that answer is the neutral boundary
                        # supervised directly rather than inferred through VAD
                        # geometry.
                        #
                        # Distinct from soft-label training (Ando et al. 2018)
                        # and from label smoothing: removing the true-class
                        # mass means the target carries no information about
                        # the label itself, only about the direction of
                        # disagreement. That has to be said explicitly in the
                        # paper or a reader will call it label smoothing.
                        #
                        # label_dist is 9 wide (Neutral, Happy, Sad, Angry,
                        # Contempt, Surprise, Disgust, Fear, Other); the first
                        # four are this project's classes in the same order.
                        # Rows with no annotator metadata, and rows where the
                        # off-label mass is zero because every annotator
                        # agreed, fall back to a uniform target over the other
                        # three classes.
                        _ncls = int(getattr(config, 'num_classes', 4))
                        _ld = batch['label_dist'].numpy().astype(
                            np.float64)[:, :_ncls]
                        _ld[np.arange(len(lab_np)), lab_np] = 0.0
                        _tot = _ld.sum(axis=1, keepdims=True)
                        _uniform = np.full_like(_ld, 1.0 / max(1, _ncls - 1))
                        _uniform[np.arange(len(lab_np)), lab_np] = 0.0
                        _have = (batch['annot_mask'].numpy() > 0)[:, None] & (_tot > 0)
                        _ld = np.where(_have, _ld / np.maximum(_tot, 1e-8), _uniform)
                        difficulties = torch.tensor(
                            _ld, dtype=torch.float32, device=device)
                    elif target_kind == 'dispersion':
                        # Inter-rater standard deviation as the auxiliary
                        # target, three dimensions, one per VAD axis. This is
                        # the Eyben et al. 2012 signal: they add annotator
                        # standard deviation as an extra multi-task target
                        # alongside five affective dimensions. Isolating it
                        # here gives a direct comparison in our setup.
                        #
                        # Note resdisp already tested dispersion CONCATENATED
                        # with the residual and lost 1.83 UAR against the
                        # control, but that confounds two things: dispersion
                        # being unhelpful, and dispersion diluting a target
                        # that works. This arm separates them.
                        #
                        # Rows the annotator join did not cover carry a std of
                        # zero, which reads as perfect agreement rather than
                        # as missing, so they take the batch mean instead.
                        _sd = batch['annot_vad_std'].numpy().astype(np.float64)
                        if len(_pdims) != _sd.shape[1]:
                            _sd = _sd[:, list(_pdims)]
                        _have = batch['annot_mask'].numpy() > 0
                        if _have.any() and not _have.all():
                            _sd[~_have] = _sd[_have].mean(axis=0)
                        difficulties = torch.tensor(
                            _sd, dtype=torch.float32, device=device)
                    elif target_kind == 'agreement':
                        # Annotator agreement AS the auxiliary target: the
                        # fraction of annotators choosing the majority label.
                        #
                        # This is the closest prior art's quantity. Kim and
                        # Provost 2015 define "prototypicality" as annotator
                        # agreement and use it to weight training on IEMOCAP
                        # four-class, with the gain concentrated on neutral.
                        # Running it here as an auxiliary target makes the
                        # distinction empirical rather than rhetorical.
                        #
                        # Two reasons to expect it to do less than the
                        # residual. It is a scalar, and every scalar target
                        # tried here has been null (protoscore, exp of minus
                        # the Mahalanobis distance, discards direction and
                        # does nothing), while the direction-preserving
                        # three-vector works. And it measures something nearly
                        # orthogonal to our target: across 80,941 training
                        # rows the correlation between class agreement and
                        # Mahalanobis atypicality is +0.027, with per-axis
                        # signed correlations no larger than 0.086.
                        #
                        # Rows the annotator join did not cover carry a
                        # consensus of zero, which would read as total
                        # disagreement rather than as missing. They take the
                        # batch mean of the covered rows instead.
                        _ag = batch['annot_consensus'].numpy().astype(np.float64)
                        _have = batch['annot_mask'].numpy() > 0
                        if _have.any() and not _have.all():
                            _ag[~_have] = _ag[_have].mean()
                        elif not _have.any():
                            _ag = np.full_like(_ag, 0.5)
                        difficulties = torch.tensor(
                            _ag[:, None], dtype=torch.float32, device=device)
                    elif target_kind == 'presidual':
                        # Precision-weighted residual: the whitened residual
                        # divided per axis by one plus the annotator
                        # dispersion on that axis.
                        #
                        #     r = W_c (v - mu_c) / (1 + s)
                        #
                        # Section 4 of the results log measures roughly 29
                        # percent of within-class VAD spread as annotator
                        # measurement noise. The plain residual cannot tell a
                        # speaker who is genuinely atypical from raters who
                        # could not agree, and treats both as large targets.
                        # Dividing by dispersion shrinks the second case, so
                        # the magnitude comes to mean RELIABLY atypical.
                        #
                        # This modulates the working target instead of
                        # widening it. The 'resdisp' arm appended dispersion
                        # as three extra outputs and lost 1.83 UAR against the
                        # placebo on IEMOCAP, for two compounding reasons:
                        # dispersion appears not to be recoverable from audio
                        # and text, so half the gradient fitted noise, and a
                        # standard deviation is a magnitude with no direction,
                        # the same defect that made the scalar 'protoscore'
                        # null. Here dispersion never has to be predicted and
                        # the target keeps all three signed components.
                        #
                        # Rows the annotator join did not cover carry a std of
                        # zero, which would read as perfect agreement and
                        # leave those residuals undivided. They take the batch
                        # mean of the covered rows instead.
                        _r = residual_targets(vad_np, lab_np, means,
                                              whiten=config._proto_whiten)
                        _sd = batch['annot_vad_std'].numpy().astype(np.float64)
                        if len(_pdims) != _sd.shape[1]:
                            _sd = _sd[:, list(_pdims)]
                        _have = batch['annot_mask'].numpy() > 0
                        if _have.any() and not _have.all():
                            _sd[~_have] = _sd[_have].mean(axis=0)
                        elif not _have.any():
                            _sd = np.zeros_like(_sd)
                        difficulties = torch.tensor(
                            _r / (1.0 + np.maximum(_sd, 0.0)),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'resdisp':
                        # Whitened residual concatenated with the annotator
                        # dispersion for the same utterance, in the same
                        # class-whitened coordinates. Six outputs.
                        #
                        # Every other target here is a deterministic function
                        # of (VAD, label), so none can carry more information
                        # than (VAD, label) already does; the whitened
                        # residual works by presenting that information in a
                        # class-comparable form, not by adding to it, which is
                        # why the dose response is flat across 8x and why an
                        # affine reparameterisation reproduces it exactly.
                        # Annotator dispersion is the first component that is
                        # NOT recoverable from the consensus VAD or the label:
                        # two utterances with identical consensus ratings can
                        # have had very different levels of rater agreement.
                        #
                        # It is whitened by the same per-class matrix so the
                        # two halves share units and the class-conditional
                        # structure that the ladder shows is load-bearing.
                        # Rows the annotator join did not cover carry a std of
                        # zero, which would read as perfect agreement, so they
                        # are set to the batch mean of the covered rows;
                        # absent evidence reads as typical rather than as
                        # certainty.
                        _r = residual_targets(vad_np, lab_np, means,
                                              whiten=config._proto_whiten)
                        _sd = batch['annot_vad_std'].numpy().astype(np.float64)
                        if len(_pdims) != _sd.shape[1]:
                            _sd = _sd[:, list(_pdims)]
                        _have = batch['annot_mask'].numpy() > 0
                        if _have.any() and not _have.all():
                            _sd[~_have] = _sd[_have].mean(axis=0)
                        _w = getattr(config, '_proto_whiten', None)
                        if _w is not None:
                            _sd = np.einsum('nij,nj->ni', _w[lab_np], _sd)
                        difficulties = torch.tensor(
                            np.concatenate([_r, _sd], axis=1),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'resagree':
                        # Prototypicality AND annotator agreement together.
                        #
                        # Q2 found the two predict model difficulty through
                        # independent routes: in a joint logistic with class
                        # fixed effects, disagreement carries +0.434 and
                        # residual magnitude +0.336, each surviving the
                        # other. They are also near orthogonal within class
                        # (Spearman -0.003). If that complementarity carries
                        # into the auxiliary task, the pair should beat
                        # either alone.
                        #
                        # Agreement is standardised with training statistics
                        # so both halves of the target have unit variance.
                        # The existing resdisp kind does not do this: passing
                        # raw positive dispersion values through the class
                        # whitening leaves that half carrying 18.8x the
                        # residual half's weight, which makes it a dispersion
                        # arm rather than a combination. Dispersion is also
                        # the wrong half by our own measurement, being inert
                        # in the joint model.
                        _r = residual_targets(vad_np, lab_np, means,
                                              whiten=config._proto_whiten)
                        _ag = batch['annot_consensus'].numpy().astype(
                            np.float64)
                        _have = batch['annot_mask'].numpy() > 0
                        if _have.any() and not _have.all():
                            _ag[~_have] = _ag[_have].mean()
                        elif not _have.any():
                            _ag = np.full_like(_ag, config._agree_mu)
                        _ag = (_ag - config._agree_mu) / config._agree_sd
                        difficulties = torch.tensor(
                            np.concatenate([_r, _ag[:, None]], axis=1),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'randfixed':
                        # A FIXED random target per utterance: learnable, but
                        # carrying no information about the emotion.
                        #
                        # The decisive test of the within-class-variance
                        # story. The shuffled controls redraw their target
                        # every batch, so the head can never fit them and the
                        # trunk gets only noise; they gain +0.28 AUCneu. This
                        # target is drawn once per utterance and held fixed
                        # across all epochs, so the head CAN fit it, but only
                        # by making the shared embedding keep utterances
                        # individually distinguishable. That is precisely the
                        # within-class spread cross-entropy collapses and
                        # every arm that helps restores.
                        #
                        # If this reproduces the gain, the mechanism is that
                        # the auxiliary task preserves within-class geometry
                        # and the emotional content of the target is
                        # irrelevant. If it does not, content matters and the
                        # regulariser reading is wrong.
                        #
                        # Drawn from a standard normal, matching the whitened
                        # residual's unit covariance, so the effective
                        # auxiliary weight is comparable to uf4_res.
                        _si = batch['sample_index'].numpy()
                        difficulties = torch.tensor(
                            config._rand_table[_si],
                            dtype=torch.float32, device=device)
                    elif target_kind == 'vadagree':
                        # Globally whitened VAD plus standardised agreement.
                        #
                        # The fourth cell of the 2x2 completed by resagree:
                        # with and without class-relative centring, crossed
                        # with and without annotator agreement. Needed to say
                        # whether centring contributes anything once agreement
                        # is already in the target, which resagree against res
                        # alone cannot answer.
                        _wv = (vad_np - config._marginal_mean) @ \
                            config._marginal_whiten
                        _ag = batch['annot_consensus'].numpy().astype(
                            np.float64)
                        _have = batch['annot_mask'].numpy() > 0
                        if _have.any() and not _have.all():
                            _ag[~_have] = _ag[_have].mean()
                        elif not _have.any():
                            _ag = np.full_like(_ag, config._agree_mu)
                        _ag = (_ag - config._agree_mu) / config._agree_sd
                        difficulties = torch.tensor(
                            np.concatenate([_wv, _ag[:, None]], axis=1),
                            dtype=torch.float32, device=device)
                    elif target_kind == 'whitevad':
                        # CONTROL: whitening WITHOUT class-relative centering.
                        #
                        # The global mean and the marginal covariance ignore
                        # the label entirely, so this target is an invertible
                        # affine map of raw VAD and carries exactly the same
                        # information the 'rawvad' arm does. Only the
                        # conditioning of the regression target changes: the
                        # axes are decorrelated and put on a common scale.
                        #
                        # It separates the two explanations the ablation
                        # ladder otherwise confounds. The whitened residual
                        # differs from raw VAD in BOTH class-relative
                        # centering and conditioning, so a reviewer can argue
                        # the gain is only that MSE on raw VAD weights three
                        # correlated axes of unequal scale unevenly. If this
                        # arm matches 'rawvad', conditioning explains nothing
                        # and the rescaling works only in combination with
                        # centering. If it matches the residual arm, the
                        # class-relative reading is wrong.
                        difficulties = torch.tensor(
                            (vad_np - config._marginal_mean)
                            @ config._marginal_whiten,
                            dtype=torch.float32, device=device)
                    elif target_kind == 'rawvad':
                        # Raw VAD, with no centroid subtraction and no
                        # whitening. The direct test of whether
                        # 'prototypicality' does any work: residual is an
                        # affine function of VAD given the class, and the
                        # class is what the main head already computes, so a
                        # reviewer will ask whether predicting VAD itself is
                        # equivalent. Same head, same output width, same
                        # loss weight; only the target differs.
                        difficulties = torch.tensor(
                            vad_np, dtype=torch.float32, device=device)
                    elif target_kind in ('residual', 'absresidual',
                                         'invresidual'):
                        _r = residual_targets(vad_np, lab_np, means,
                                              whiten=config._proto_whiten)
                        if target_kind == 'invresidual':
                            # 1 - residual. Affine, and therefore an exact
                            # reparameterisation of 'residual': the head ends
                            # in a bare Linear, so its final layer can absorb
                            # the negation and the shift, and MSE is
                            # symmetric. The optimum and the gradient reaching
                            # the shared trunk are unchanged. This arm exists
                            # to audit the invertibility argument that retired
                            # the consensus centroids; a difference beyond
                            # seed noise would falsify that argument.
                            _r = 1.0 - _r
                        if target_kind == 'absresidual':
                            # Magnitude without direction. The signed residual
                            # plus the label reconstructs VAD at R^2 0.983, so
                            # it is VAD regression in a class-shifted frame;
                            # the absolute value drops that to 0.406, making it
                            # the first variant that is not recoverable from
                            # VAD. The cost is the sign, which carries whether
                            # a sample is more extreme than its class or
                            # drifting toward neutral.
                            _r = np.abs(_r)
                        difficulties = torch.tensor(
                            _r, dtype=torch.float32, device=device)
                    elif target_kind == 'subtype':
                        # Nearest per-class subtype centroid, as a class id in
                        # [0, num_classes * clusters_per_class).
                        cents = config._proto_subtype_centroids
                        kpc = cents.shape[1]
                        d = np.linalg.norm(
                            vad_np[:, None, :] - cents[lab_np], axis=2)
                        difficulties = torch.tensor(
                            lab_np * kpc + d.argmin(axis=1),
                            dtype=torch.long, device=device)
                    elif str(getattr(config, 'proto_metric',
                                     'euclidean')) == 'mahalanobis':
                        difficulties = torch.tensor(
                            mahalanobis_distance(vad_np, lab_np, means,
                                                 config._proto_inv_covs),
                            dtype=torch.float32, device=device)
                    else:
                        difficulties = torch.tensor(
                            np.linalg.norm(vad_np - means[lab_np], axis=1),
                            dtype=torch.float32, device=device)
                if bool(getattr(config, 'proto_predictor_shuffle', False)):
                    # Control task: same head, same parameters, same gradient
                    # path, but the target no longer belongs to the sample.
                    # Anything the real arm gains over this is attributable to
                    # prototypicality rather than to the extra capacity.
                    #
                    # "batch" permutes within the batch, the original design.
                    # "pool" draws from the training set's residuals, removing
                    # both the fixed points a permutation always has and any
                    # dependence on batch composition.
                    # "gauss" draws from a Gaussian matched to the pool's mean
                    # and covariance, so the control does not reuse any real
                    # row at all. Agreement across the three says the result
                    # does not depend on how the control was built.
                    _mode = str(getattr(config, 'proto_shuffle_mode', 'batch'))
                    _pool_t = getattr(config, '_proto_pool', None)
                    if _mode == 'batch' or _pool_t is None:
                        difficulties = difficulties[torch.randperm(
                            difficulties.size(0), device=difficulties.device)]
                    elif _mode == 'pool':
                        pick = torch.randint(_pool_t.size(0),
                                             (difficulties.size(0),))
                        difficulties = _pool_t[pick].to(difficulties.device)
                    elif _mode == 'gauss':
                        if not hasattr(config, '_proto_pool_chol'):
                            _mu = _pool_t.mean(dim=0)
                            _cov = torch.from_numpy(np.cov(
                                _pool_t.numpy().T)).float()
                            _cov += 1e-6 * torch.eye(_cov.size(0))
                            config._proto_pool_mu = _mu
                            config._proto_pool_chol = torch.linalg.cholesky(_cov)
                        z = torch.randn(difficulties.size(0),
                                        config._proto_pool_mu.size(0))
                        difficulties = (config._proto_pool_mu
                                        + z @ config._proto_pool_chol.T
                                        ).to(difficulties.device)
                    else:
                        raise ValueError(
                            f"unknown proto_shuffle_mode: {_mode}. Use batch, "
                            f"pool or gauss.")
                # Per-sample gradient surgery: wrap the auxiliary head's
                # input so its gradient into the shared trunk can be projected
                # row by row against the classification gradient. The reference
                # is filled in at backward time.
                if bool(getattr(config, 'proto_pcgrad_per_sample', False)):
                    _aux_in = _PerSampleProject.apply(raw_embeddings, _ps_ref)
                else:
                    _aux_in = raw_embeddings
                pred_proto = proto_predictor(_aux_in)
                # Optional per-sample weighting by annotator agreement. A
                # contested utterance's consensus VAD summarises a
                # disagreement rather than measuring the emotion, so its
                # residual is a noisy target the head is asked to fit anyway.
                # Annotator dispersion is one of the few quantities available
                # that is not a deterministic function of VAD, so unlike the
                # geometric variants this can add information rather than
                # restate it. Rows the annotator join did not cover carry a
                # std of zero, which the floor would turn into the LARGEST
                # weight; they are held at the mean of the covered rows so
                # absent evidence reads as neutral rather than as certainty.
                agree_w = None
                if (bool(getattr(config, 'proto_agreement_weight', False))
                        and 'annot_vad_std' in batch):
                    spread = torch.clamp(
                        batch['annot_vad_std'].to(device), min=0.25).mean(dim=1)
                    agree_w = 1.0 / (1.0 + spread)
                    has_meta = batch['annot_mask'].to(device) > 0
                    if has_meta.any():
                        agree_w = torch.where(has_meta, agree_w,
                                              agree_w[has_meta].mean())
                    agree_w = agree_w / agree_w.mean().clamp_min(1e-6)

                # Optional: class-balance the auxiliary loss.
                #
                # Neutral is 45 percent of the training pool, so it dominates
                # the auxiliary gradient by count alone, whatever its target
                # is worth. Masking it entirely (proto_skip_neutral) tests the
                # extreme and overshoots: it removes half the neutral-AUC gain
                # on IEMOCAP, so the neutral residual does carry information.
                # This is the middle ground. Rows are weighted by the same
                # class weights the classifier already uses, so neutral's
                # share of the auxiliary gradient falls from 45 to about 31
                # percent without being eliminated, and no new hyperparameter
                # is introduced.
                #
                # Weights are renormalised to mean one over the batch, so the
                # auxiliary loss keeps its magnitude and
                # proto_predictor_weight stays comparable across arms.
                if (bool(getattr(config, 'proto_class_balanced', False))
                        and getattr(criterion, 'weight', None) is not None):
                    _cw = criterion.weight.to(pred_proto.device)[
                        batch['label'].to(pred_proto.device)]
                    _cw = _cw.to(pred_proto.dtype)
                    _cw = _cw / _cw.mean().clamp_min(1e-6)
                    agree_w = _cw if agree_w is None else agree_w * _cw

                # Optional: weight the auxiliary loss by how well VAD alone
                # accounts for the row's label.
                #
                # Fitting a Gaussian class-conditional VAD classifier on
                # MSP-Podcast and transferring it to IEMOCAP leaves 34.8
                # percent of that corpus misclassified from VAD alone. Scored
                # against matched controls on that split, twelve of fifteen
                # arms LOSE accuracy on those rows while gaining on the rest;
                # whitevad is +0.85 on the explained half and -1.06 on the
                # unexplained half, both at p<0.05 over five seeds. The head
                # capacity alone HELPS the unexplained half (ctrl_pool is
                # +0.98 over base), so it is the VAD content specifically that
                # costs, not the extra parameters.
                #
                # The loud-sad and muted-happy utterances are that population.
                # Regressing VAD there pushes the encoder toward a VAD-shaped
                # solution the classifier then has to undo. This weight
                # confines the auxiliary target to rows where VAD and the
                # label agree, and is computed from statistics already fitted
                # on the training split.
                #
                # Modes:
                #   global     rescale to mean one over the batch.
                #   per_class  rescale to mean one WITHIN each class. The raw
                #              posterior averages 0.41 on neutral and 0.78 on
                #              angry, so the global mode silently reweights
                #              classes and is confounded with clsbal, which is
                #              already known not to help. per_class removes
                #              that component and isolates within-class trust.
                _trust_mode = str(getattr(config, 'proto_vad_trust', 'off'))
                if (_trust_mode not in ('off', 'false', 'False')
                        and getattr(config, '_proto_means', None) is not None
                        and 'vad' in batch):
                    _tv = batch['vad'].numpy().astype(np.float64)
                    _tdims = getattr(config, '_proto_dims', (0, 1, 2))
                    if len(_tdims) != _tv.shape[1]:
                        _tv = _tv[:, list(_tdims)]
                    _tl = batch['label'].numpy()
                    _tw = vad_trust_weights(_tv, _tl, config._proto_means,
                                            config._proto_inv_covs)
                    _tw = torch.tensor(_tw, dtype=pred_proto.dtype,
                                       device=pred_proto.device)
                    if _trust_mode == 'per_class':
                        _tl_t = batch['label'].to(pred_proto.device)
                        for _c in _tl_t.unique():
                            _m = _tl_t == _c
                            _tw[_m] = _tw[_m] / _tw[_m].mean().clamp_min(1e-6)
                    # The control permutes the weights across the batch. The
                    # weight DISTRIBUTION is preserved exactly and only its
                    # pairing with the sample is destroyed, so any gain that
                    # survives it is not the mechanism.
                    if bool(getattr(config, 'proto_vad_trust_shuffle', False)):
                        _tw = _tw[torch.randperm(_tw.numel(),
                                                 device=_tw.device)]
                    _tw = _tw / _tw.mean().clamp_min(1e-6)
                    agree_w = _tw if agree_w is None else agree_w * _tw

                # Optional: drop neutral rows from the auxiliary loss.
                #
                # The target asks where a sample sits inside its own class,
                # measured in that class's units. For the emotional classes
                # that is well defined. For neutral it may not be: if
                # neutrality is the absence of emotion rather than an emotion,
                # a neutral utterance far from the neutral centroid is
                # drifting toward SOME emotion, and the whitened residual
                # expresses that drift in neutral's own coordinate system,
                # which carries no information about which emotion it is
                # drifting toward. Neutral is 45 percent of the training pool,
                # so that possibly incoherent target dominates the auxiliary
                # gradient.
                #
                # The surviving weights are rescaled to mean one, so the
                # auxiliary loss keeps the same magnitude as the unmasked arms
                # and proto_predictor_weight stays comparable across them.
                if bool(getattr(config, 'proto_skip_neutral', False)):
                    _keep = (batch['label'] != 0).to(
                        device=pred_proto.device, dtype=pred_proto.dtype)
                    if _keep.sum() > 0:
                        _keep = _keep * (_keep.numel() / _keep.sum())
                        agree_w = _keep if agree_w is None else agree_w * _keep

                # Optional: score the head only on WITHIN-class variation.
                #
                # residual_targets centres each class on its own training mean,
                # so the target's conditional mean given class is zero by
                # construction. Measured on the held-out split the target's
                # per-class means are 0.00 to 0.03, while the head predicts
                # -0.29 to +0.26, and 8.5 percent of its output variance is
                # explained by class alone (26 percent on IEMOCAP). That is
                # the head reading class identity off the embedding and
                # emitting it, which is error rather than signal.
                #
                # Removing the per-class mean from both sides makes that
                # shortcut worthless: the head can only reduce its loss by
                # predicting where a sample sits INSIDE its class.
                #
                # CAVEAT the mean is estimated per batch, and at batch 16 each
                # class holds about four rows, so the estimate is noisy and
                # removes some genuine within-class variance too. A running
                # estimate would be better if this is pursued.
                if bool(getattr(config, 'proto_centre_pred', False)):
                    _lab_t = batch['label'].to(pred_proto.device)
                    _pc = torch.zeros_like(pred_proto)
                    _dc = torch.zeros_like(difficulties)
                    for _c in _lab_t.unique():
                        _m = _lab_t == _c
                        if int(_m.sum()) > 1:
                            _pc[_m] = pred_proto[_m].mean(dim=0, keepdim=True)
                            _dc[_m] = difficulties[_m].mean(dim=0, keepdim=True)
                    pred_proto = pred_proto - _pc
                    difficulties = difficulties - _dc

                # Second auxiliary task: is this row's label accounted
                # for by its VAD at all?
                #
                # The regression target asks WHERE a sample sits inside its
                # class. This asks whether a RIVAL class sits closer, which is
                # the condition that actually separates the two populations:
                # the toward-rival component of the residual predicts it at AUC
                # 0.931, the residual NORM reaches only 0.654, and the
                # orthogonal component is exactly chance at 0.500. mahal and
                # protoscore are both functions of the norm, so they target the
                # weaker quantity.
                #
                # Non-affine in VAD and dependent on ALL class centres, so a
                # globally standardised VAD target cannot reproduce it. That is
                # the critique that sank the class-relative residual, where
                # whitevad matched res at p=0.918, and this is immune to it.
                #
                # Binary under BCE, so there is no target variance to match and
                # the proto_predictor_weight calibration confound does not
                # arise here. expl_head_weight is relative to the regression
                # term, so the effective weight is proto_predictor_weight times
                # expl_head_weight.
                #
                # Split BEFORE the target-kind branch below, so every kind sees
                # a correctly shaped pred_proto.
                _expl_loss = None
                if bool(getattr(config, 'use_expl_head', False)):
                    _expl_logit = pred_proto[:, -1]
                    pred_proto = pred_proto[:, :-1]
                    _ev = batch['vad'].numpy().astype(np.float64)
                    _edims = getattr(config, '_proto_dims', (0, 1, 2))
                    if len(_edims) != _ev.shape[1]:
                        _ev = _ev[:, list(_edims)]
                    _et = vad_explainable(_ev, batch['label'].numpy(),
                                          config._proto_means,
                                          config._proto_inv_covs)
                    _et = torch.tensor(_et, dtype=_expl_logit.dtype,
                                       device=_expl_logit.device)
                    # Control: permute the targets across the batch. The class
                    # balance is preserved exactly and only the pairing with
                    # the sample is destroyed, so a gain that survives it is
                    # not the mechanism.
                    if bool(getattr(config, 'expl_head_shuffle', False)):
                        _et = _et[torch.randperm(_et.numel(),
                                                 device=_et.device)]
                    _expl_loss = F.binary_cross_entropy_with_logits(
                        _expl_logit, _et)

                if str(getattr(config, 'proto_target', 'scalar')) == 'subtype':
                    if agree_w is None:
                        loss_proto_pred = F.cross_entropy(pred_proto, difficulties)
                    else:
                        per = F.cross_entropy(pred_proto, difficulties,
                                              reduction='none')
                        loss_proto_pred = (per * agree_w).mean()
                elif agree_w is None:
                    loss_proto_pred = F.mse_loss(pred_proto, difficulties)
                else:
                    per = F.mse_loss(pred_proto, difficulties,
                                     reduction='none').mean(dim=1)
                    loss_proto_pred = (per * agree_w).mean()
                if _expl_loss is not None:
                    loss_proto_pred = loss_proto_pred + (
                        float(getattr(config, 'expl_head_weight', 1.0))
                        * _expl_loss)

                # VAD-consistent mixup on the AUXILIARY target.
                #
                # For a continuous target, mixup is exactly correct rather
                # than assumed: the blend of two utterances genuinely has
                # the blended affect coordinates, and the whitevad target is
                # an affine map of VAD, so the blended target equals the
                # blend of the endpoint targets identically. None of the
                # difficulties that afflict the categorical target apply.
                #
                # This is only informative when the head is NONLINEAR. With
                # proto_predictor_hidden_dim 0 the head is a bare linear map
                # W, so W(lam*z_a + (1-lam)*z_b) - (lam*t_a + (1-lam)*t_b)
                # collapses to lam*r_a + (1-lam)*r_b, a strictly weaker copy
                # of the endpoint loss whose gradient on each sample is
                # contaminated by a random partner's residual. With a hidden
                # layer the identity breaks and the term becomes a real
                # constraint: affect varies linearly along segments between
                # training points in embedding space.
                if (bool(getattr(config, 'aux_vad_mixup', False))
                        and torch.is_floating_point(difficulties)):
                    # Only meaningful for a CONTINUOUS target. A classification
                    # target kind (allclass, subtype) carries integer class
                    # indices, which cannot be blended, so the guard refuses
                    # rather than silently averaging label ids.
                    _n_av = raw_embeddings.size(0)
                    if _n_av >= 2:
                        _pick_av = torch.randint(_n_av, (_n_av,), device=device)
                        _clash = _pick_av == torch.arange(_n_av, device=device)
                        if _clash.any():
                            _pick_av[_clash] = (_pick_av[_clash] + 1) % _n_av
                        _alpha_av = float(getattr(config, 'aux_vad_mixup_alpha', 2.0))
                        _lam_av = torch.distributions.Beta(
                            _alpha_av, _alpha_av).sample((_n_av,)).to(device)
                        if bool(getattr(config, 'aux_vad_mixup_fold', True)):
                            _lam_av = torch.maximum(_lam_av, 1.0 - _lam_av)
                        _lam_av = _lam_av.unsqueeze(1)
                        _z_av = (_lam_av * raw_embeddings
                                 + (1.0 - _lam_av) * raw_embeddings[_pick_av])
                        _t_av = (_lam_av * difficulties
                                 + (1.0 - _lam_av) * difficulties[_pick_av])
                        if bool(getattr(config, 'aux_vad_mixup_shuffle', False)):
                            # Placebo: keep the blended embeddings and the
                            # exact distribution of blended targets, but
                            # permute which target goes with which blend.
                            # Same loss scale, correspondence destroyed.
                            _t_av = _t_av[torch.randperm(_n_av, device=device)]
                        _loss_av = F.mse_loss(proto_predictor(_z_av), _t_av)
                        loss_proto_pred = loss_proto_pred + (
                            float(getattr(config, 'aux_vad_mixup_weight', 1.0))
                            * _loss_av)
                        aux_mix_count += int(_z_av.size(0))

                # Per-modality auxiliary losses, averaged with the fused one
                # so the total keeps the same magnitude and
                # proto_predictor_weight stays comparable across arms.
                _extra_losses = []
                for _head, _key in ((proto_predictor_audio, 'audio'),
                                    (proto_predictor_text, 'text')):
                    if _head is None or not isinstance(modal_features, dict):
                        continue
                    _feat = modal_features.get(_key)
                    if _feat is None:
                        continue
                    _pred = _head(_feat)
                    if agree_w is None:
                        _extra_losses.append(F.mse_loss(_pred, difficulties))
                    else:
                        _per = F.mse_loss(_pred, difficulties,
                                          reduction='none').mean(dim=1)
                        _extra_losses.append((_per * agree_w).mean())
                if _extra_losses:
                    loss_proto_pred = ((loss_proto_pred + sum(_extra_losses))
                                       / (1 + len(_extra_losses)))

                # Gradient-alignment diagnostic, sampled every N steps.
                # Reuses the loss already computed above rather than running a
                # second forward: the predictor contains dropout, so an extra
                # forward would advance the RNG and make arms differ for a
                # reason unrelated to the mechanism. autograd.grad returns
                # gradients without writing .grad, so the real backward that
                # follows is unaffected.
                cos_every = int(getattr(config, 'proto_grad_cosine_every', 0))
                if cos_every > 0 and (global_step % cos_every == 0):
                    try:
                        shared = [q for q in model.fusion_module.parameters()
                                  if q.requires_grad]
                        if shared:
                            g_main = torch.autograd.grad(
                                loss_primary, shared, retain_graph=True,
                                allow_unused=True)
                            g_aux = torch.autograd.grad(
                                loss_proto_pred, shared, retain_graph=True,
                                allow_unused=True)
                            fm = torch.cat([g.reshape(-1) for g in g_main
                                            if g is not None])
                            fa = torch.cat([g.reshape(-1) for g in g_aux
                                            if g is not None])
                            wandb.log({
                                'aux/grad_cosine': float(F.cosine_similarity(
                                    fm.unsqueeze(0), fa.unsqueeze(0)).item()),
                                'aux/global_step': global_step})
                    except Exception:
                        pass

            # Cross-modal alignment loss: train learned projections
            loss_cross_modal = torch.tensor(0.0, device=device)
            if modal_features is not None and hasattr(model, 'audio_cross_proj'):
                audio_proj = F.normalize(modal_features['audio'], p=2, dim=1)
                text_proj = F.normalize(modal_features['text'], p=2, dim=1)
                loss_cross_modal = 1.0 - F.cosine_similarity(audio_proj, text_proj).mean()

            # VADmix loss: cross-corpus feature mixup with VAD-derived soft targets
            loss_vadmix = torch.tensor(0.0, device=device)
            if use_vadmix and raw_embeddings is not None and config.task_type == "classification":
                vad_batch = batch['vad'].to(device).float()
                corpora_batch = batch['dataset']
                protos_dev = vadmix_prototypes.to(device)
                mixed_logits, soft_targets, _lam = vadmix_step(
                    embeddings=raw_embeddings,
                    vad=vad_batch,
                    corpora=corpora_batch,
                    prototypes=protos_dev,
                    output_layer=model.output_layer,
                    alpha=getattr(config, 'vadmix_alpha', 0.2),
                    temperature=getattr(config, 'vadmix_temperature', 0.5),
                    cross_corpus_only=getattr(config, 'vadmix_cross_corpus_only', True),
                )
                cls_weights_vm = criterion.weight if hasattr(criterion, 'weight') else None
                loss_vadmix = vadmix_soft_ce(mixed_logits, soft_targets, cls_weights_vm)

            # Muted-emotion mixup: synthesise low-salience emotional
            # examples by blending emotional embeddings toward neutral
            # ones, keeping the hard emotional label. Targets the region
            # the diagnosis found empty (524 quiet angry utterances in the
            # whole training set) rather than reweighting what is there,
            # which was measured to do nothing.
            loss_muted = torch.tensor(0.0, device=device)
            loss_pull = torch.tensor(0.0, device=device)
            if ((bool(getattr(config, 'use_proto_pull', False))
                 or float(getattr(config, 'muted_mixup_pull_fraction', 0.0)) > 0)
                    and getattr(config, '_pull_tracker', None) is None):
                config._pull_tracker = ClassCentroidEMA(
                    int(getattr(config, 'num_classes', 4)),
                    int(getattr(config, 'hidden_dim', 1024)), device,
                    momentum=float(getattr(config, 'proto_pull_momentum', 0.9)))
                print("   Prototype pull: EMA class centroids in embedding "
                      "space, momentum %.2f, weight %.2f%s"
                      % (float(getattr(config, 'proto_pull_momentum', 0.9)),
                         float(getattr(config, 'proto_pull_weight', 0.5)),
                         "  [GLOBAL-MEAN PLACEBO]"
                         if bool(getattr(config, 'proto_pull_global', False))
                         else ""))
            # Fitted class statistics as tensors for the derived-label
            # path. Built here rather than per call so the conversion does not
            # repeat inside the batch loop.
            _mm_means = _mm_inv = _mm_logpri = None
            if (bool(getattr(config, 'muted_mixup_derive_labels', False))
                    or bool(getattr(config, 'muted_mixup_general_soft', False))
                    or bool(getattr(config, 'muted_mixup_per_modality', False))):
                if getattr(config, '_proto_means', None) is not None:
                    _mm_means = torch.as_tensor(
                        config._proto_means, dtype=torch.float32, device=device)
                    _mm_inv = torch.as_tensor(
                        config._proto_inv_covs, dtype=torch.float32,
                        device=device)
                    if (bool(getattr(config, 'muted_mixup_calibrated_prior',
                                     False))
                            and getattr(config, '_proto_cal_prior',
                                        None) is not None):
                        _mm_logpri = torch.as_tensor(
                            config._proto_cal_prior, dtype=torch.float32,
                            device=device)
                    elif (bool(getattr(config, 'muted_mixup_class_priors', False))
                            and getattr(config, '_proto_log_priors',
                                        None) is not None):
                        _mm_logpri = torch.as_tensor(
                            config._proto_log_priors, dtype=torch.float32,
                            device=device)

            if (getattr(config, 'use_muted_mixup', False)
                    and raw_embeddings is not None
                    and config.task_type == "classification"):
                if bool(getattr(config, 'use_global_mixup', False)):
                    # Global Mixup (Xie et al., AAAI 2023): relabel the blend
                    # by a similarity-weighted vote over the real samples in
                    # the batch. Same family as ours, unsupervised signal.
                    mm_logits, mm_targets, mm_lam = global_mixup_step(
                        raw_embeddings, labels,
                        model.classify_from_embeddings,
                        num_classes=int(getattr(config, 'num_classes', 4)),
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        gamma=float(getattr(config, 'global_mixup_gamma', 4.0)),
                        theta=getattr(config, 'global_mixup_theta', None),
                        top_s=int(getattr(config, 'global_mixup_top_s', 0)),
                        fold_lambda=bool(getattr(
                            config, 'muted_mixup_fold_lambda', True)))
                elif (bool(getattr(config, 'muted_mixup_per_modality', False))
                        and isinstance(modal_features, dict)
                        and _mm_means is not None):
                    # Pre-fusion blend, one coefficient per modality. The
                    # target uses an effective coefficient rather than an
                    # exact correspondence; see per_modality_mixup_step.
                    mm_logits, mm_targets, mm_lam = per_modality_mixup_step(
                        modal_features['audio'], modal_features['text'],
                        labels, batch['vad'].to(device).float(),
                        model.classify_from_modalities,
                        means=_mm_means, inv_covs=_mm_inv,
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        w_audio=float(getattr(config, 'muted_mixup_w_audio',
                                              0.638)),
                        independent=bool(getattr(
                            config, 'muted_mixup_modality_independent', True)),
                        mode=str(getattr(config, 'muted_mixup_modality_mode',
                                         'independent')),
                        fold_lambda=bool(getattr(
                            config, 'muted_mixup_fold_lambda', True)))
                elif genlabel_density is not None:
                    # GenLabel baseline: the target comes from class
                    # densities over EMBEDDINGS, not over VAD. Statistics
                    # for the next refit are accumulated from the same
                    # embeddings, costing one matmul and no extra pass.
                    genlabel_density.accumulate(raw_embeddings, labels)
                    mm_logits, mm_targets, mm_lam = genlabel_mixup_step(
                        raw_embeddings, labels,
                        model.classify_from_embeddings,
                        density=genlabel_density,
                        num_classes=int(getattr(config, 'num_classes', 4)),
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        gamma=float(getattr(config, 'genlabel_gamma', 1.0)),
                        fold_lambda=bool(
                            getattr(config, 'muted_mixup_fold_lambda', True)),
                    )
                elif bool(getattr(config, 'muted_mixup_general_soft', False)):
                    # Any pair, Beta lambda folded to [0.5, 1], and the
                    # blend's own class posterior as the target. No switch
                    # point and no threshold: see general_soft_mixup_step.
                    if _mm_means is None:
                        raise ValueError(
                            "muted_mixup_general_soft needs the fitted class "
                            "statistics; enable use_proto_predictor so they "
                            "are computed.")
                    _mm_trust = None
                    if bool(getattr(config, 'muted_mixup_trust_target', False)):
                        if 'annot_consensus' not in batch:
                            raise ValueError(
                                "muted_mixup_trust_target needs annotator "
                                "metadata; enable use_annotator_meta.")
                        _mm_trust = batch['annot_consensus'].to(device).float()
                        if bool(getattr(config,
                                        'muted_mixup_trust_shuffle', False)):
                            # Placebo: permute consensus WITHIN each class so
                            # the distribution of trust weights is identical
                            # and only its correspondence to the sample is
                            # destroyed.
                            _perm = torch.arange(labels.size(0), device=device)
                            for _c in labels.unique():
                                _ix = (labels == _c).nonzero(as_tuple=True)[0]
                                if _ix.numel() > 1:
                                    _perm[_ix] = _ix[torch.randperm(
                                        _ix.numel(), device=device)]
                            _mm_trust = _mm_trust[_perm]
                    mm_logits, mm_targets, mm_lam = general_soft_mixup_step(
                        raw_embeddings, labels,
                        model.classify_from_embeddings,
                        vad=batch['vad'].to(device).float(),
                        means=_mm_means, inv_covs=_mm_inv,
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        hard_control=bool(
                            getattr(config, 'muted_mixup_hard_control', False)),
                        vanilla_target=bool(
                            getattr(config, 'muted_mixup_vanilla_target', False)),
                        endpoint_posterior=bool(
                            getattr(config, 'muted_mixup_endpoint_posterior',
                                    False)),
                        same_class_hard=bool(
                            getattr(config, 'muted_mixup_same_class_hard',
                                    False)),
                        log_priors=_mm_logpri,
                        trust=_mm_trust,
                        trust_min=float(getattr(config, '_consensus_min', 0.2)),
                        repeats=int(getattr(config, 'muted_mixup_repeats', 1)),
                        pull_fraction=float(getattr(
                            config, 'muted_mixup_pull_fraction', 0.0)),
                        pull_tracker=getattr(config, '_pull_tracker', None),
                        target_shuffle=bool(getattr(
                            config, 'muted_mixup_target_shuffle', False)),
                        dirichlet_conc=(
                            torch.as_tensor(config._dirichlet_conc)
                            if getattr(config, '_dirichlet_conc', None) is not None
                            else None),
                        fold_lambda=bool(
                            getattr(config, 'muted_mixup_fold_lambda', True)),
                    )
                elif getattr(config, 'muted_mixup_symmetric', False):
                    # Blend both ways so equal numbers of synthetic samples
                    # land on each side of the neutral boundary and the
                    # prior is left where it was.
                    mm_lam = None
                    mm_logits, mm_targets = symmetric_mixup_step(
                        raw_embeddings, labels, model.classify_from_embeddings,
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        within_class_control=bool(
                            getattr(config, 'muted_mixup_control', False)),
                    )
                else:
                    # VAD-calibrated variant needs the batch's VAD and the
                    # neutral centre; expected_vad[0] is that centre already.
                    mm_vad = None
                    mm_centre = None
                    span = None
                    if bool(getattr(config, 'muted_mixup_vad_calibrated', False)):
                        exp_vad = getattr(config, 'expected_vad', None) or {}
                        centre = exp_vad.get(0) or exp_vad.get('0')
                        if centre is not None and 'vad' in batch:
                            span = getattr(config, '_muted_vad_span', None)
                            mm_vad = batch['vad'].to(device).float()
                            mm_centre = torch.tensor(
                                centre, dtype=torch.float32, device=device)
                    if (bool(getattr(config, 'muted_mixup_gated_symmetric', False))
                            and mm_vad is not None
                            and 'audio_features' in model_inputs
                            and 'text_input_ids' in model_inputs):
                        # Mix before fusion so a modality can be left intact.
                        # Intensity is prosodic, so attenuating the fused
                        # embedding also erases lexical evidence, which is not
                        # what the hypothesis claims.
                        a_feat, t_feat = model.modality_features(
                            model_inputs['audio_features'],
                            model_inputs['text_input_ids'],
                            model_inputs['text_attention_mask'])
                        mix_a, mix_t, mm_targets, mm_lam = vad_gated_symmetric_mixup(
                            a_feat, t_feat, labels, mm_vad, mm_centre,
                            mix_audio=bool(getattr(config, 'muted_mixup_mix_audio', True)),
                            mix_text=bool(getattr(config, 'muted_mixup_mix_text', True)),
                            gate_quantile=float(
                                getattr(config, 'muted_mixup_gate_quantile', 0.5)),
                            shuffle_control=bool(
                                getattr(config, 'muted_mixup_vad_shuffle', False)),
                        )
                        mm_logits = (model.classify_from_modalities(mix_a, mix_t)
                                     if mix_a is not None else None)
                    elif (bool(getattr(config, 'muted_mixup_sparse_fill', False))
                            and mm_vad is not None
                            and getattr(config, '_sparse_direction', None) is not None):
                        # Each class moves toward the intensity end its own
                        # training data is thin at, rather than every class
                        # muting toward neutral.
                        mm_logits, mm_targets, mm_lam = sparse_fill_mixup_step(
                            raw_embeddings, labels,
                            model.classify_from_embeddings,
                            vad=mm_vad,
                            neutral_centre=mm_centre,
                            class_direction=torch.tensor(
                                config._sparse_direction, device=device),
                            class_target=torch.tensor(
                                config._sparse_target, dtype=torch.float32,
                                device=device),
                            shuffle_control=bool(
                                getattr(config, 'muted_mixup_vad_shuffle', False)),
                        )
                    else:
                        mm_logits, mm_targets, mm_lam = muted_mixup_step(
                            raw_embeddings, labels,
                            model.classify_from_embeddings,
                            alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                            shuffle_control=bool(
                                getattr(config, 'muted_mixup_control', False)),
                            vad=mm_vad,
                            neutral_centre=mm_centre,
                            vad_target_quantile=float(
                                getattr(config, 'muted_mixup_vad_quantile', 0.5)),
                            vad_shuffle=bool(
                                getattr(config, 'muted_mixup_vad_shuffle', False)),
                            vad_target_span=span if mm_vad is not None else None,
                            # Label the blend from its own position in VAD
                            # space instead of inheriting the parent's class.
                            derive_labels=bool(
                                getattr(config, 'muted_mixup_derive_labels',
                                        False)),
                            proto_means=_mm_means,
                            proto_inv_covs=_mm_inv,
                            derive_min_posterior=float(
                                getattr(config,
                                        'muted_mixup_derive_min_posterior',
                                        0.0)),
                        )
                    if mm_lam is not None:
                        muted_lam_sum += float(mm_lam.sum())
                        muted_lam_n += int(mm_lam.numel())
                if mm_logits is not None:
                    cls_w_mm = criterion.weight if hasattr(criterion, 'weight') else None
                    loss_muted = (
                        soft_target_loss(mm_logits, mm_targets, cls_w_mm)
                        if mm_targets.dim() == 2
                        else muted_mixup_loss(mm_logits, mm_targets, cls_w_mm))
                    muted_synth_count += int(mm_logits.size(0))

            if ((bool(getattr(config, 'use_proto_pull', False))
                 or float(getattr(config, 'muted_mixup_pull_fraction', 0.0)) > 0)
                    and raw_embeddings is not None
                    and config.task_type == "classification"):
                # Blend each sample toward its own class prototype. Both
                # endpoints share a class, so the target is a plain one-hot
                # and none of the geometry labelling applies.
                _tr = config._pull_tracker
                _pull_logits, _pull_y = (None, None)
                if bool(getattr(config, 'use_proto_pull', False)):
                    _pull_logits, _pull_y = prototype_pull_step(
                        raw_embeddings, labels,
                        model.classify_from_embeddings, _tr,
                        alpha=float(getattr(config, 'proto_pull_alpha', 2.0)),
                        global_target=bool(getattr(config, 'proto_pull_global',
                                                   False)))
                if _pull_logits is not None:
                    _cw = (criterion.weight
                           if hasattr(criterion, 'weight') else None)
                    loss_pull = F.cross_entropy(_pull_logits, _pull_y,
                                                weight=_cw)
                # Update AFTER the loss so the centroids used as targets are
                # not the same step's own embeddings.
                _tr.update(raw_embeddings, labels)

            # Consistency-coupled mixup. The classification target for a
            # blended sample stays the hard original label, while its
            # prototypicality target moves with the blend, so the two
            # supervisions state the same claim instead of contradicting each
            # other: the category is invariant under attenuation, the position
            # within the category is not.
            loss_consistent = torch.tensor(0.0, device=device)
            if (bool(getattr(config, 'use_consistent_mixup', False))
                    and raw_embeddings is not None
                    and proto_predictor is not None
                    and config.task_type == "classification"
                    and 'vad' in batch):
                cm_vad = batch['vad'].to(device).float()
                cm_logits, cm_labels, cm_mixed_vad, _cm_lam, cm_emb = (
                    consistent_muted_mixup(
                        raw_embeddings, labels, cm_vad,
                        model.classify_from_embeddings,
                        gate_quantile=float(
                            getattr(config, 'muted_mixup_gate_quantile', 0.5)),
                        shuffle_control=bool(
                            getattr(config, 'muted_mixup_vad_shuffle', False)),
                    ))
                if cm_logits is not None:
                    cls_w_cm = (criterion.weight
                                if hasattr(criterion, 'weight') else None)
                    loss_consistent = muted_mixup_loss(cm_logits, cm_labels,
                                                       cls_w_cm)
                    # Auxiliary supervision on the synthetic points, using
                    # where the blend actually landed in VAD space.
                    means = getattr(config, '_proto_means', None)
                    if means is not None:
                        tgt = torch.tensor(
                            residual_targets(
                                cm_mixed_vad.detach().cpu().numpy().astype(
                                    np.float64),
                                cm_labels.cpu().numpy(), means,
                                whiten=config._proto_whiten),
                            dtype=torch.float32, device=device)
                        loss_consistent = loss_consistent + F.mse_loss(
                            proto_predictor(cm_emb), tgt)
                    muted_synth_count += int(cm_logits.size(0))

            # Presence loss for feature-level detector fusion. The branch is
            # already inside the classification path, so this only supervises
            # what it encodes. The placebo shuffles the targets, leaving the
            # branch as capacity that carries no presence information.
            loss_det_fusion = torch.tensor(0.0, device=device)
            if (getattr(config, 'use_detector_fusion', False)
                    and raw_embeddings is not None
                    and config.task_type == "classification"):
                det_logits = model.detector_logits_from_embeddings(raw_embeddings)
                det_targets = (labels != 0).long()
                if getattr(config, 'detector_fusion_shuffle', False):
                    perm = torch.randperm(det_targets.size(0), device=det_targets.device)
                    det_targets = det_targets[perm]
                loss_det_fusion = F.cross_entropy(det_logits, det_targets)

            # Auxiliary VAD-cluster loss. Aux head lives on the model; we
            # already have raw_embeddings (use_embeddings was forced True when
            # this flag is on). Non-VAD samples are masked out.
            loss_aux_vad = torch.tensor(0.0, device=device)
            aux_vad_task = getattr(config, 'aux_vad_task', 'cluster')
            use_aux_vad_flag = (getattr(config, 'use_aux_vad_cluster', False)
                                and raw_embeddings is not None
                                and getattr(model, 'use_aux_vad_cluster', False)
                                and (aux_vad_task in ('regression', 'subtype')
                                     or aux_vad_centroids is not None))
            if use_aux_vad_flag:
                aux_out = model.aux_vad_forward(raw_embeddings)
                if aux_vad_task == 'regression':
                    # Predict raw (V, A, D); per-sample MSE over the 3 dims.
                    vad_targets, aux_mask = batch_vad_targets_and_mask(
                        batch, device,
                    )
                    if getattr(config, 'aux_vad_shuffle', False):
                        # Placebo: keep the auxiliary head, its parameters and
                        # the gradient noise of a second objective, but break
                        # the correspondence between features and targets. A
                        # gain that survives this is regularisation from having
                        # any auxiliary regression, not from learning VAD.
                        vad_targets = vad_targets[torch.randperm(
                            vad_targets.size(0), device=vad_targets.device)]
                    per_sample_aux = F.mse_loss(
                        aux_out, vad_targets, reduction='none',
                    ).mean(dim=1)
                elif aux_vad_task == 'resid_cluster':
                    # Target is the sample's deviation mode: which way it
                    # departs from its own class prototype, in whitened VAD
                    # space. Under the shared scope the same mode id spans
                    # classes, so "louder than this emotion normally is" is
                    # one target whether the emotion is sadness or happiness.
                    cluster_ids, aux_mask = batch_residual_cluster_ids_and_mask(
                        batch,
                        aux_vad_centroids,
                        config._resid_means_t,
                        config._resid_whiten_t,
                        device,
                        scope=getattr(config, 'aux_vad_cluster_scope', 'shared'),
                    )
                    if getattr(config, 'aux_vad_shuffle', False):
                        # Placebo: permute the mode ids across the supervised
                        # rows. Same head, same label marginal, same number of
                        # supervised samples; only the correspondence between
                        # utterance and deviation mode is broken.
                        valid = aux_mask.nonzero(as_tuple=True)[0]
                        if valid.numel() > 1:
                            perm = valid[torch.randperm(valid.numel(),
                                                        device=device)]
                            shuffled = cluster_ids.clone()
                            shuffled[valid] = cluster_ids[perm]
                            cluster_ids = shuffled
                    per_sample_aux = F.cross_entropy(
                        aux_out, cluster_ids, reduction='none',
                    )
                elif aux_vad_task == 'subtype':
                    # Annotator-supplied secondary tags. Where k-means invents
                    # subtypes from VAD geometry, these are the nuance the
                    # annotators actually named: Depressed against Frustrated
                    # inside sad, Amused against Excited inside happy. Several
                    # tags can apply at once, so the target is a per-tag rate
                    # and the loss is multi-label BCE rather than a softmax.
                    subtype_targets = batch['subtype_dist'].to(device)
                    aux_mask = batch['annot_mask'].to(device)
                    if getattr(config, 'aux_vad_shuffle', False):
                        # Placebo, permuted within the covered rows only so the
                        # control keeps the same number of supervised samples
                        # as the real arm and differs solely in whether the
                        # target belongs to the utterance.
                        valid = aux_mask.nonzero(as_tuple=True)[0]
                        if valid.numel() > 1:
                            perm = valid[torch.randperm(valid.numel(),
                                                        device=device)]
                            shuffled = subtype_targets.clone()
                            shuffled[valid] = subtype_targets[perm]
                            subtype_targets = shuffled
                    per_sample_aux = F.binary_cross_entropy_with_logits(
                        aux_out, subtype_targets, reduction='none',
                    ).mean(dim=1)
                elif getattr(config, 'aux_vad_cluster_scope', 'global') == 'per_class':
                    # Subtype-within-class target: happy-1, happy-2, angry-1...
                    cluster_ids, aux_mask = batch_per_class_cluster_ids_and_mask(
                        batch, aux_vad_centroids, device,
                    )
                    if getattr(config, 'aux_vad_shuffle', False):
                        # Placebo for the per-class subtype target. Permuting
                        # within the supervised rows preserves the head, its
                        # parameter count, the label marginal and the number
                        # of supervised samples, so the only thing removed is
                        # whether the subtype belongs to the utterance.
                        valid = aux_mask.nonzero(as_tuple=True)[0]
                        if valid.numel() > 1:
                            perm = valid[torch.randperm(valid.numel(),
                                                        device=device)]
                            shuffled = cluster_ids.clone()
                            shuffled[valid] = cluster_ids[perm]
                            cluster_ids = shuffled
                    per_sample_aux = F.cross_entropy(
                        aux_out, cluster_ids, reduction='none',
                    )
                elif getattr(config, 'aux_vad_permute_clusters', False):
                    # Permuted-label control: pseudo-random per-sample labels
                    # with no usable VAD structure.
                    cluster_ids, aux_mask = batch_scrambled_ids_and_mask(
                        batch,
                        int(getattr(config, 'aux_vad_cluster_k', 8)),
                        int(getattr(config, 'seed', 42)),
                        device,
                    )
                    per_sample_aux = F.cross_entropy(
                        aux_out, cluster_ids, reduction='none',
                    )
                else:
                    cluster_ids, aux_mask = batch_cluster_ids_and_mask(
                        batch, aux_vad_centroids, device,
                    )
                    per_sample_aux = F.cross_entropy(
                        aux_out, cluster_ids, reduction='none',
                    )
                if (getattr(config, 'aux_vad_agreement_weight', False)
                        and 'annot_vad_std' in batch):
                    # A contested utterance carries a VAD target that summarises
                    # an argument rather than measuring anything, so the head is
                    # asked to fit noise. Down-weight by annotator spread, with
                    # a floor for the unanimous rows whose std is exactly zero.
                    spread = torch.clamp(
                        batch['annot_vad_std'].to(device), min=0.25,
                    ).mean(dim=1)
                    agree_w = 1.0 / (1.0 + spread)
                    # Rows the annotator join did not cover carry a std of
                    # zero because the data is absent, not because everyone
                    # agreed. The floor turns that into the largest weight in
                    # the batch, so absent evidence would read as perfect
                    # agreement and get up-weighted 1.44x over real rows.
                    # Give them the mean weight of the covered rows instead,
                    # which leaves them exactly neutral.
                    has_meta = batch['annot_mask'].to(device) > 0
                    if has_meta.any():
                        agree_w = torch.where(has_meta, agree_w,
                                              agree_w[has_meta].mean())
                    covered = aux_mask > 0
                    if covered.any():
                        # Renormalise over the covered rows so the auxiliary
                        # term keeps the magnitude of the unweighted arm and
                        # the comparison is about where capacity goes, not how
                        # much of it there is.
                        agree_w = agree_w / agree_w[covered].mean().clamp_min(1e-6)
                    aux_mask = aux_mask * agree_w

                # Average over samples with real VAD only.
                denom = aux_mask.sum().clamp_min(1.0)
                loss_aux_vad = (per_sample_aux * aux_mask).sum() / denom

            # Combined loss
            adv_weight = getattr(config, 'adversarial_weight', 0.0)
            modality_adv_weight = getattr(config, 'modality_adv_weight', 0.0)
            proto_pred_weight = getattr(config, 'proto_predictor_weight', 0.0)
            # Optional linear ramp. At full weight from step one the auxiliary
            # head competes with classification while the shared embedding is
            # still random, so the target it fits is noise; ramping lets the
            # embedding form first.
            pp_warm = int(getattr(config, 'proto_predictor_warmup_epochs', 0))
            if pp_warm > 0 and current_epoch < pp_warm:
                proto_pred_weight *= (current_epoch + 1) / pp_warm
            # Anneal to zero over the final N epochs. If the auxiliary task
            # helps by shaping the encoder early and only adds noise later,
            # this keeps the benefit and drops the cost; it also makes the
            # SWA window pure main-task, so a gain that survives cannot be a
            # late-training calibration nudge.
            pp_anneal = int(getattr(config, 'proto_predictor_anneal_epochs', 0))
            if pp_anneal > 0:
                total = int(getattr(config, 'num_epochs', 20))
                start = max(0, total - pp_anneal)
                if current_epoch >= start:
                    frac = 1.0 - (current_epoch - start + 1) / pp_anneal
                    proto_pred_weight *= max(0.0, frac)
            cross_modal_weight = getattr(config, 'cross_modal_weight', 0.1)
            vadmix_weight = getattr(config, 'vadmix_weight', 1.0) if use_vadmix else 0.0
            aux_vad_weight = (float(getattr(config, 'aux_vad_cluster_weight', 0.2))
                              if use_aux_vad_flag else 0.0)
            loss = (loss_primary
                    + effective_contrastive_weight * loss_contrastive
                    + adv_weight * loss_adversarial
                    + modality_adv_weight * loss_modality_adversarial
                    + proto_pred_weight * loss_proto_pred
                    + float(getattr(config, 'proto_pull_weight', 0.0))
                    * loss_pull
                    + cross_modal_weight * loss_cross_modal
                    + vadmix_weight * loss_vadmix
                    + aux_vad_weight * loss_aux_vad
                    + float(getattr(config, 'muted_mixup_weight', 0.5)) * loss_muted
                    + float(getattr(config, 'detector_fusion_weight', 0.5))
                    * loss_det_fusion
                    + float(getattr(config, 'consistent_mixup_weight', 0.5))
                    * loss_consistent)

        # Track losses (unscaled values)
        total_loss += loss.item()
        total_primary_loss += loss_primary.item()
        if config.use_contrastive:
            total_contrastive_loss += loss_contrastive.item()
        if domain_discriminator is not None:
            total_adversarial_loss += loss_adversarial.item()
        if modality_discriminator is not None:
            total_modality_adversarial_loss += loss_modality_adversarial.item()
        if proto_predictor is not None:
            total_proto_pred_loss += loss_proto_pred.item()
        if use_vadmix:
            total_vadmix_loss += loss_vadmix.item()
        if getattr(config, 'use_aux_vad_cluster', False):
            total_aux_vad_loss += loss_aux_vad.item()
        if getattr(config, 'use_vrex', False):
            total_vrex_var += last_vrex_var
            vrex_var_steps += 1

        # Backward (scale loss for gradient accumulation)
        scaled_loss = loss / accum_steps
        # Per-sample PCGrad: capture the CLASSIFICATION gradient at the shared
        # representation, so the auxiliary branch can be projected against it
        # during the single backward that follows. One extra partial backward
        # through the classifier head only, not through the encoder.
        if (bool(getattr(config, 'proto_pcgrad_per_sample', False))
                and proto_predictor is not None
                and raw_embeddings is not None
                and torch.is_tensor(loss_primary)
                and loss_primary.requires_grad):
            _ps_ref[0] = torch.autograd.grad(
                loss_primary, raw_embeddings,
                retain_graph=True, allow_unused=True)[0]
        use_pcgrad = (bool(getattr(config, 'proto_pcgrad', False))
                      and proto_predictor is not None
                      and proto_pred_weight > 0.0
                      and torch.is_tensor(loss_proto_pred)
                      and loss_proto_pred.requires_grad)
        if use_pcgrad:
            # Gradient surgery (PCGrad, Yu et al. 2020). The permuted control
            # costs 0.3 to 0.9 accuracy on every corpus, which is interference:
            # its target carries no information, so the only thing its gradient
            # can do to the shared trunk is fight the classification loss.
            #
            # Here the auxiliary gradient is projected onto the orthogonal
            # complement of the main gradient wherever the two conflict, so the
            # auxiliary task can only move the representation in directions
            # that do not oppose classification. Where they already agree the
            # auxiliary gradient passes through untouched.
            #
            # Costs one extra backward pass over the shared trunk per step.
            # Applied per parameter tensor rather than on one flattened vector:
            # the flattened version lets a large conflict in one layer cancel
            # agreement in another, which is not the intent.
            main_loss = scaled_loss - (proto_pred_weight * loss_proto_pred) / accum_steps
            aux_loss = (proto_pred_weight * loss_proto_pred) / accum_steps
            params = [p_ for p_ in model.parameters() if p_.requires_grad]
            g_main = torch.autograd.grad(
                scaler.scale(main_loss) if use_amp else main_loss,
                params, retain_graph=True, allow_unused=True)
            g_aux = torch.autograd.grad(
                scaler.scale(aux_loss) if use_amp else aux_loss,
                params, retain_graph=False, allow_unused=True)
            for p_, gm, ga in zip(params, g_main, g_aux):
                if gm is None and ga is None:
                    continue
                if gm is None:
                    combined = ga
                elif ga is None:
                    combined = gm
                else:
                    dot = torch.sum(ga * gm)
                    if dot < 0:
                        ga = ga - dot / gm.pow(2).sum().clamp_min(1e-12) * gm
                    combined = gm + ga
                p_.grad = combined if p_.grad is None else p_.grad + combined
        elif use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Optimizer step at accumulation boundary
        if should_step:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            wandb.log({'train/step_loss': loss.item(), 'train/global_step': global_step})

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_primary_loss = total_primary_loss / len(dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(dataloader) if config.use_contrastive else 0.0
    avg_adversarial_loss = total_adversarial_loss / len(dataloader) if domain_discriminator is not None else 0.0
    avg_modality_adversarial_loss = total_modality_adversarial_loss / len(dataloader) if modality_discriminator is not None else 0.0
    avg_proto_pred_loss = total_proto_pred_loss / len(dataloader) if proto_predictor is not None else 0.0
    avg_vadmix_loss = total_vadmix_loss / len(dataloader) if use_vadmix else 0.0
    avg_aux_vad_loss = (total_aux_vad_loss / len(dataloader)
                        if getattr(config, 'use_aux_vad_cluster', False) else 0.0)
    avg_vrex_var = (total_vrex_var / vrex_var_steps) if vrex_var_steps > 0 else 0.0

    if config.task_type == "regression":
        vad_preds = np.concatenate(all_vad_preds, axis=0)
        vad_targets = np.concatenate(all_vad_targets, axis=0)
        metrics = calculate_vad_metrics(vad_preds, vad_targets)
    else:
        metrics = calculate_classification_metrics(all_predictions, all_labels)

    if getattr(config, "use_muted_mixup", False):
        mean_lam = (muted_lam_sum / muted_lam_n) if muted_lam_n else float('nan')
        print(f"   muted mixup: {muted_synth_count} synthetic samples this epoch, "
              f"mean lambda={mean_lam:.3f}")
    return {
        'loss': avg_loss,
        'primary_loss': avg_primary_loss,
        'contrastive_loss': avg_contrastive_loss,
        'adversarial_loss': avg_adversarial_loss,
        'modality_adversarial_loss': avg_modality_adversarial_loss,
        'proto_pred_loss': avg_proto_pred_loss,
        'vadmix_loss': avg_vadmix_loss,
        'aux_vad_loss': avg_aux_vad_loss,
        'vrex_var': avg_vrex_var,
        'effective_contrastive_weight': effective_contrastive_weight,
        'global_step': global_step,
        **metrics
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config, use_amp=False):
    """
    Evaluate model

    Args:
        model: EmotionClassifier
        dataloader: DataLoader
        criterion: Loss function
        device: torch device
        config: Config object
        use_amp: bool - whether to use mixed precision

    Returns:
        dict with loss and metrics
    """
    model.eval()

    total_loss = 0
    all_predictions = []
    all_eval_logits = []
    all_labels = []
    all_vad_preds = []
    all_vad_targets = []

    use_proto_atyp = bool(getattr(config, 'use_proto_atyp_split', False))
    num_classes_pa = int(getattr(config, 'num_classes', 4))

    for batch in dataloader:
        labels = batch['label'].to(device)

        # Prepare inputs
        model_inputs = _prepare_model_inputs(batch, config, model, device)

        with autocast('cuda', enabled=use_amp):
            # Forward pass
            logits = model(**model_inputs)

            # Calculate loss
            if config.task_type == "regression":
                vad_targets = batch['vad'].to(device)
                loss = criterion(logits, vad_targets)
                all_vad_preds.append(logits.cpu().float().numpy())
                all_vad_targets.append(vad_targets.cpu().numpy())
            else:
                if getattr(config, 'use_latent_prototypes', False):
                    cls_w_lp = (criterion.weight
                                if hasattr(criterion, 'weight') else None)
                    loss, cprobs = latent_prototype_loss(
                        logits, labels,
                        int(getattr(config, 'num_classes', 4)),
                        int(getattr(config, 'prototypes_per_class', 2)),
                        class_weights=cls_w_lp,
                    )
                    preds = cprobs.argmax(dim=-1).cpu().numpy()
                elif getattr(config, 'use_hierarchical_head', False):
                    loss = hierarchical_loss(logits, labels)  # reporting only
                    preds = hierarchical_predictions(logits).cpu().numpy()
                elif getattr(config, 'use_evidence_heads', False):
                    # Evidence heads: BCE loss, threshold decision rule.
                    # Reporting only; matches the training weighting scheme
                    # (pos_weight balances the heads, no class weight).
                    loss = evidence_bce_loss(
                        logits, labels,
                        int(getattr(config, 'num_classes', 4)),
                        class_weights=None,
                    )
                    preds = evidence_predictions(
                        logits,
                        float(getattr(config, 'evidence_threshold', 0.5)),
                    ).cpu().numpy()
                elif use_proto_atyp:
                    # Collapse [B, 2C] to [B, C] class probs, then use NLL on
                    # the 4-class labels. No VAD is used at eval, so the same
                    # path works for cross-corpus test sets without VAD.
                    class_probs = collapse_sub_logits_to_class_probs(logits, num_classes_pa)
                    class_log_probs = torch.log(class_probs.clamp_min(1e-12))
                    cls_weights_eval = criterion.weight if hasattr(criterion, 'weight') else None
                    loss = F.nll_loss(class_log_probs, labels, weight=cls_weights_eval)
                    preds = torch.argmax(class_probs, dim=-1).cpu().numpy()
                else:
                    loss = criterion(logits, labels)
                    # Handle reduction='none' case (when prototypical weighting/smoothing is on)
                    if loss.dim() > 0:
                        loss = loss.mean()
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_predictions.extend(preds)
                # Kept so neutral_auc / emo_auc can be computed downstream;
                # both need scores, not the argmax.
                all_eval_logits.append(logits.detach().cpu().float().numpy())
                all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)

    if config.task_type == "regression":
        vad_preds = np.concatenate(all_vad_preds, axis=0)
        vad_targets = np.concatenate(all_vad_targets, axis=0)
        metrics = calculate_vad_metrics(vad_preds, vad_targets)
    else:
        metrics = calculate_classification_metrics(all_predictions, all_labels)
        if all_eval_logits:
            metrics['logits'] = np.concatenate(all_eval_logits, axis=0)

    result = {
        'loss': avg_loss,
        **metrics
    }
    if config.task_type == "classification":
        result['predictions'] = all_predictions
        result['labels'] = all_labels
    return result


def _calibrate_threshold_on_val(model, val_loader, config, device, use_amp=False):
    """Choose the evidence threshold that maximizes UAR on the val split.

    Run after training, before test evaluation. Only the operating point is
    fitted, and it is fitted on data the test corpora never see. A fixed
    0.5 is arbitrary and interacts with pos_weight, which is what produced
    the two degenerate regimes measured earlier (neutral recall 0.578
    without pos_weight, 0.036 with it).

    Args:
        model: trained classifier with evidence heads.
        val_loader: validation DataLoader.
        config: Config object.
        device: torch device.
        use_amp: whether to run under autocast.

    Returns:
        (threshold, val UAR at that threshold).
    """
    model.eval()
    chunks, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = _prepare_model_inputs(batch, config, model, device)
            with autocast('cuda', enabled=use_amp):
                logits = model(**inputs)
            chunks.append(logits.detach().float().cpu())
            labels.append(batch['label'])
    if not chunks:
        return float(getattr(config, 'evidence_threshold', 0.5)), float('nan')

    thr, uar = calibrate_evidence_threshold(
        torch.cat(chunks, dim=0), torch.cat(labels, dim=0),
        int(getattr(config, 'num_classes', 4)),
    )
    return thr, uar


def _collect_embeddings(model, dataloader, config, device, use_amp=False):
    """Collect raw fused embeddings and labels over a dataloader.

    Args:
        model: EmotionClassifier
        dataloader: DataLoader over a labeled evaluation set
        config: Config object
        device: torch device
        use_amp: bool - whether to use mixed precision

    Returns:
        (embeddings [N, D] float32 np array, labels [N] int64 np array)
    """
    model.eval()
    all_emb = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            model_inputs = _prepare_model_inputs(batch, config, model, device)
            with autocast('cuda', enabled=use_amp):
                result = model(**model_inputs, return_embeddings=True)
            _logits, _proj, raw_embeddings, _modal = result
            all_emb.append(raw_embeddings.detach().cpu().float().numpy())
            all_labels.append(batch['label'].numpy())
    if len(all_emb) == 0:
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros(0, dtype=np.int64))
    return np.concatenate(all_emb, axis=0), np.concatenate(all_labels, axis=0)


def _cm_class_names(config):
    """Class names for confusion-matrix logging, sized to the label set.

    Binary neutral-detector arms have two classes, so the four-way emotion
    names would be rejected by the plotting call.

    Args:
        config: Config object.

    Returns:
        List of class-name strings of length num_classes.
    """
    if bool(getattr(config, 'binary_neutral', False)):
        return ['neutral', 'emotional']
    names = ['neutral', 'happy', 'sad', 'angry']
    if bool(getattr(config, 'drop_neutral', False)):
        # Neutral is gone and the rest shifted down, so slicing from the
        # front would silently mislabel every class.
        names = names[1:]
    return names[:int(getattr(config, 'num_classes', 4))]


def _dump_split_predictions(model, dataloader, config, device, split_name,
                            use_amp=False, proto_predictor=None):
    """Write per-sample logits for one split to the run's checkpoint dir.

    Stacking, error-correlation analysis and threshold sweeps all need the
    raw model outputs rather than aggregate metrics. Writing them once at
    the end of a run turns every one of those into offline array work
    instead of a fresh GPU pass, and keeps the ensemble reproducible from
    the exact split the base model actually held out.

    Rows follow dataloader order with shuffling disabled, so files from
    different arms over the same split align row for row.

    Args:
        model: trained EmotionClassifier in eval mode.
        dataloader: DataLoader over a labeled split, shuffle disabled.
        config: Config object.
        device: torch device.
        split_name: name used in the output filename.
        use_amp: whether to use mixed precision.
        proto_predictor: optional auxiliary head. When given, its predictions
            and the batch VAD are dumped alongside the logits, so the head's
            fit to its own target can be measured offline on every split
            rather than only inferred from the training loss. The head is
            discarded after training, so this is the only record of it.

    Returns:
        Path to the written .npz file, or None when the split is empty.
    """
    model.eval()
    all_logits, all_labels, all_label4, all_speakers = [], [], [], []
    all_aux, all_vad = [], []
    want_aux = proto_predictor is not None
    if want_aux:
        proto_predictor.eval()
    with torch.no_grad():
        for batch in dataloader:
            model_inputs = _prepare_model_inputs(batch, config, model, device)
            with autocast('cuda', enabled=use_amp):
                if want_aux:
                    result = model(**model_inputs, return_embeddings=True)
                else:
                    result = model(**model_inputs)
            logits = result[0] if isinstance(result, tuple) else result
            if want_aux:
                # (logits, projected, raw, modal); raw is what the head reads.
                raw_emb = result[2]
                all_aux.append(
                    proto_predictor(raw_emb).detach().cpu().float().numpy())
                if 'vad' in batch:
                    all_vad.append(batch['vad'].numpy())
            all_logits.append(logits.detach().cpu().float().numpy())
            all_labels.append(batch['label'].numpy())
            if 'label4' in batch:
                all_label4.append(batch['label4'].numpy())
            all_speakers.extend(
                batch.get('speaker', ['unknown'] * len(batch['label'])),
            )

    if len(all_logits) == 0:
        return None

    ckpt_dir, _, _, _ = get_checkpoint_paths(config)
    pred_dir = ckpt_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    dest = pred_dir / f"{split_name}.npz"

    labels = np.concatenate(all_labels, axis=0)
    payload = {
        'logits': np.concatenate(all_logits, axis=0).astype(np.float32),
        'labels': labels.astype(np.int64),
        'speakers': np.array(all_speakers),
        'arm': np.array(config.experiment_name),
        'seed': np.array(int(config.seed)),
        'split': np.array(split_name),
        'binary_neutral': np.array(bool(getattr(config, 'binary_neutral', False))),
        'hierarchical': np.array(
            bool(getattr(config, 'use_hierarchical_head', False)),
        ),
    }
    if all_label4:
        payload['label4'] = np.concatenate(all_label4, axis=0).astype(np.int64)
    if all_aux:
        payload['aux_pred'] = np.concatenate(all_aux, axis=0).astype(np.float32)
    if all_vad:
        payload['vad'] = np.concatenate(all_vad, axis=0).astype(np.float32)
    # The fitted class statistics, so the target can be reconstructed exactly
    # offline for any split. They are per-run: the prototypes are fitted on
    # this seed's training rows, so they cannot be recovered from elsewhere.
    for _attr in ('_proto_means', '_proto_whiten', '_proto_inv_covs',
                  '_marginal_mean', '_marginal_whiten'):
        _val = getattr(config, _attr, None)
        if _val is not None:
            payload[_attr.lstrip('_')] = np.asarray(_val, dtype=np.float64)
    payload['proto_target'] = np.array(
        str(getattr(config, 'proto_target', 'none')))

    np.savez_compressed(dest, **payload)
    print(f"      predictions -> {dest} {payload['logits'].shape}")
    return dest


def _compute_per_sample_difficulty(train_dataset, config):
    """
    Compute per-sample difficulty scores for the training dataset.

    Returns:
        np.array of difficulty scores, one per sample in train_dataset.data
    """
    difficulties = []
    for item in train_dataset.data:
        diff = calculate_difficulty(
            item['valence'], item['arousal'], item['dominance'],
            item['label'], config.expected_vad
        )
        difficulties.append(diff)
    return np.array(difficulties)


def get_checkpoint_paths(config):
    """Return (ckpt_dir, latest_path, status_path, results_path) for this run."""
    ckpt_dir = Path("checkpoints") / f"{config.experiment_name}_seed{config.seed}"
    return (
        ckpt_dir,
        ckpt_dir / "latest.pt",
        ckpt_dir / "status.json",
        ckpt_dir / "results.json",
    )


# Config fields that do not affect the trained weights. Everything else
# feeds the fingerprint below, so changing it invalidates a cached run.
_FINGERPRINT_EXCLUDE = {
    'test_datasets',      # eval-only; handled separately as a subset check
    'wandb_project', 'experiment_name', 'experiment_id',
    # Gradient checkpointing trades compute for activation memory and does
    # not change the computation. Verified directly on 2026-09-13: one batch
    # through the 4/4 model with identical seeds gave the same loss to ten
    # decimals and bitwise identical gradients across all 89,183,496
    # trainable elements. Excluded so toggling it does not mark finished runs
    # stale and trigger a retrain.
    'use_gradient_checkpointing',
}


def _session_of(speaker):
    """Recover the IEMOCAP recording session number from a speaker id.

    IEMOCAP speaker ids are of the form 'Ses01F', optionally carrying a
    corpus prefix such as 'IEMOFULL:Ses01F'. The session number is the only
    recording-condition grouping the corpus exposes, and each session holds
    exactly two speakers, so holding one out is both speaker-disjoint and
    condition-disjoint.

    Args:
        speaker: Speaker id string, possibly prefixed with a corpus name.

    Returns:
        Session number as an int, or None if the id carries no session tag.
    """
    match = re.search(r'Ses(\d+)', str(speaker))
    return int(match.group(1)) if match else None


def _config_fingerprint(config):
    """Stable hash of the training-relevant parts of a config.

    Cached runs are keyed only by experiment name and seed, so re-running a
    changed config under an unchanged name silently reuses stale results.
    That has already produced one corrupted sweep here, where arms trained
    under early stopping were compared against arms trained on a fixed
    epoch budget. The fingerprint turns that into a loud refusal.

    Args:
        config: Config object or dict.

    Returns:
        Hex digest string over the sorted training-relevant key/value pairs.
    """
    cfg = config.__dict__ if hasattr(config, '__dict__') else dict(config)
    items = []
    for key in sorted(cfg):
        if key in _FINGERPRINT_EXCLUDE or key.startswith('_'):
            continue
        try:
            items.append(f"{key}={json.dumps(cfg[key], sort_keys=True, default=str)}")
        except (TypeError, ValueError):
            items.append(f"{key}={cfg[key]!r}")
    return hashlib.sha256("|".join(items).encode('utf-8')).hexdigest()[:16]


def is_run_finished(config):
    """Check if this experiment+seed completed under an equivalent config.

    Returns False when the stored fingerprint disagrees with the current
    config, so the run is redone rather than silently reused. Also requires
    the cached run to have evaluated a superset of the requested test
    corpora, since a cache missing a corpus cannot answer for it.
    """
    _, _, status_path, results_path = get_checkpoint_paths(config)
    if not status_path.exists() or not results_path.exists():
        return False
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    if status.get('status') != 'done':
        return False

    name = getattr(config, 'experiment_name', '?')
    cached_fp = status.get('config_fingerprint')
    if cached_fp is None:
        print(f"  NOTE: cached run {name} predates config fingerprinting, so "
              f"it cannot be\n        verified against the current config. "
              f"Delete its checkpoint directory to\n        force a retrain "
              f"if the config has changed since it ran.")
    elif cached_fp != _config_fingerprint(config):
        print(f"  Cached run {name} was produced by a DIFFERENT config "
              f"(fingerprint\n    {cached_fp} vs {_config_fingerprint(config)}). "
              f"Retraining rather than reusing it.")
        return False

    requested = set(getattr(config, 'test_datasets', []) or [])
    cached_sets = status.get('test_datasets')
    if cached_sets is not None and not requested.issubset(set(cached_sets)):
        missing = sorted(requested - set(cached_sets))
        print(f"  Cached run {name} never evaluated {missing}. "
              f"Retraining rather than reusing it.")
        return False

    return True


def load_finished_results(config):
    """Load cached results.json from a previously finished run."""
    _, _, _, results_path = get_checkpoint_paths(config)
    with open(results_path, 'r') as f:
        return json.load(f)


def save_checkpoint(config, epoch, global_step, model, optimizer, scheduler,
                    scaler, contrastive_criterion, domain_discriminator,
                    modality_discriminator, proto_predictor,
                    best_val_metric, best_model_state, best_contrastive_state,
                    epochs_without_improvement, wandb_run_id,
                    swa_window=None):
    """Atomic checkpoint save: write to .tmp then rename.

    swa_window is persisted because it is not recoverable after the fact. It
    holds the trailing per-epoch weight copies that model_selection
    'swa_last_n' averages at the end of training. A resumed run skips every
    completed epoch, so without this the window would be empty, the SWA branch
    would not fire, and selection would fall back to best_val silently, giving
    a model chosen by a different rule than its sibling seeds.
    """
    ckpt_dir, latest_path, status_path, _ = get_checkpoint_paths(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_metric': best_val_metric,
        'best_model_state': best_model_state,
        'epochs_without_improvement': epochs_without_improvement,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'config': config.to_dict(),
        'wandb_run_id': wandb_run_id,
    }
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    if contrastive_criterion is not None and hasattr(contrastive_criterion, 'state_dict'):
        state['contrastive_state_dict'] = contrastive_criterion.state_dict()
    if best_contrastive_state is not None:
        state['best_contrastive_state'] = best_contrastive_state
    if domain_discriminator is not None:
        state['domain_discriminator_state_dict'] = domain_discriminator.state_dict()
    if modality_discriminator is not None:
        state['modality_discriminator_state_dict'] = modality_discriminator.state_dict()
    if proto_predictor is not None:
        state['proto_predictor_state_dict'] = proto_predictor.state_dict()
    if swa_window is not None:
        state['swa_window'] = swa_window

    tmp_path = latest_path.with_suffix('.pt.tmp')
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)

    status = {
        'status': 'running',
        'epoch': epoch,
        'total_epochs': config.num_epochs,
        'experiment_name': config.experiment_name,
        'seed': config.seed,
    }
    status_tmp = status_path.with_suffix('.json.tmp')
    with open(status_tmp, 'w') as f:
        json.dump(status, f, indent=2)
    os.replace(status_tmp, status_path)


def load_checkpoint(config, model, optimizer, scheduler, scaler,
                    contrastive_criterion, domain_discriminator,
                    modality_discriminator, proto_predictor, device):
    """Load checkpoint if it exists. Returns dict with resume state or None."""
    _, latest_path, status_path, _ = get_checkpoint_paths(config)
    if not latest_path.exists():
        return None

    print(f"\n  Found checkpoint: {latest_path}")
    state = torch.load(latest_path, map_location=device, weights_only=False)

    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in state:
        scaler.load_state_dict(state['scaler_state_dict'])
    if contrastive_criterion is not None and 'contrastive_state_dict' in state:
        contrastive_criterion.load_state_dict(state['contrastive_state_dict'])
    if domain_discriminator is not None and 'domain_discriminator_state_dict' in state:
        domain_discriminator.load_state_dict(state['domain_discriminator_state_dict'])
    if modality_discriminator is not None and 'modality_discriminator_state_dict' in state:
        modality_discriminator.load_state_dict(state['modality_discriminator_state_dict'])
    if proto_predictor is not None and 'proto_predictor_state_dict' in state:
        proto_predictor.load_state_dict(state['proto_predictor_state_dict'])

    # Restore RNG state so continuation is deterministic
    torch.set_rng_state(state['torch_rng_state'].cpu())
    if state.get('cuda_rng_state') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng_state']])
    np.random.set_state(state['numpy_rng_state'])
    random.setstate(state['python_rng_state'])

    print(f"  Resuming from epoch {state['epoch']}/{config.num_epochs}, "
          f"best_val_metric={state['best_val_metric']:.4f}")

    return {
        'start_epoch': state['epoch'],
        'global_step': state['global_step'],
        'best_val_metric': state['best_val_metric'],
        'best_model_state': state['best_model_state'],
        'best_contrastive_state': state.get('best_contrastive_state'),
        'epochs_without_improvement': state['epochs_without_improvement'],
        'wandb_run_id': state.get('wandb_run_id'),
        'swa_window': state.get('swa_window'),
    }


def mark_run_done(config, results):
    """Write results.json and update status to done. Clean up latest.pt."""
    ckpt_dir, latest_path, status_path, results_path = get_checkpoint_paths(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Serialize results - strip tensors/non-JSON by converting to plain python
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if not isinstance(v, (torch.Tensor,)) and k not in ('predictions', 'labels')}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(_clean(results), f, indent=2)

    status = {
        'status': 'done',
        'epoch': config.num_epochs,
        'total_epochs': config.num_epochs,
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        # Recorded so a later run under the same name can tell whether this
        # cached result was produced by an equivalent config.
        'config_fingerprint': _config_fingerprint(config),
        'test_datasets': list(getattr(config, 'test_datasets', []) or []),
    }
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)

    # Keep latest.pt around in case user wants to resume from last state -
    # but it's large, so delete to save disk. Best model is in saved_models/.
    if latest_path.exists():
        latest_path.unlink()


def train(config, datasets=None):
    """
    Main training function

    Args:
        config: Config object
        datasets: Optional tuple of (train_dataset, test_datasets) to reuse pre-loaded data
    """
    # Set seed
    set_seed(config.seed)

    # Ground cost for the transport term, loaded once per run. Absent file
    # means the term is skipped rather than the run failing.
    _ot_cost = None
    if float(getattr(config, 'ot_weight', 0.0)) > 0.0:
        _cont = load_continuum(getattr(config, 'continuum_path', None))
        if _cont is None:
            raise FileNotFoundError(
                "ot_weight is set but analysis/continuum/"
                "mspp_continuum.npz is missing. Build it with "
                "python -m analysis.build_continuum_targets")
        _ot_cost = torch.from_numpy(_cont.ot_cost)
        print("  Transport term active: ot_weight=%.2f, cost spread %.3f "
              "to %.3f" % (float(config.ot_weight),
                           float(_ot_cost[_ot_cost > 0].min()),
                           float(_ot_cost.max())))

    # Seed diagnostic. Two runs differing only in config.seed produced
    # bitwise identical logits on the official split, so confirm the seed
    # actually reached the generators at this point rather than assuming it.
    print("  SEED DIAG: config.seed=%s torch=%.6f numpy=%.6f"
          % (config.seed, float(torch.rand(1)), float(np.random.rand())))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # Create or reuse datasets
    if datasets is not None:
        train_dataset, test_datasets = datasets
        print(f"\n  Reusing pre-loaded datasets (train: {len(train_dataset)}, test: {len(test_datasets)} sets)")
    else:
        print(f"\n  Loading datasets...")
        train_dataset, test_datasets = create_datasets(config)

    # Stratified train/val split (preserves class distribution).
    #
    # Stratify on the 4-way label and draw from a dedicated RNG seeded by the
    # run seed. Both details matter for stacking: a binary arm has two label
    # groups instead of four and would otherwise consume randomness in a
    # different order, and the global stream's position depends on whatever
    # dataset construction did first. Either would hand different arms
    # different held-out sets, and a stacked combiner needs the members'
    # held-out rows to correspond.
    total_samples = len(train_dataset)
    labels = [train_dataset.data[i]['label'] for i in range(total_samples)]
    strat_labels = [train_dataset.data[i].get('label4', labels[i])
                    for i in range(total_samples)]

    # Group indices by class
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(strat_labels):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    split_rng = np.random.RandomState(int(config.seed))
    heldout_session = getattr(config, 'heldout_session', None)
    if heldout_session is not None:
        # Session-independent cross-validation fold (standard IEMOCAP
        # protocol). Train on four of the five recording sessions and
        # evaluate on the fifth, so neither speaker nor recording condition
        # is shared across the split. Each session contributes exactly two
        # speakers, which makes the fold speaker-disjoint by construction
        # rather than approximately so, unlike the sampling-based option
        # below.
        #
        # The held-out session serves as the validation split and is also the
        # reported evaluation set under this protocol. That is only honest
        # because model selection here averages the final epochs rather than
        # picking the best validation epoch, so nothing is selected on the
        # rows being scored. Do not combine this with best-epoch selection.
        want = int(heldout_session)
        sessions = [_session_of(train_dataset.data[idx].get('speaker', ''))
                    for idx in range(total_samples)]
        untagged = sum(1 for s in sessions if s is None)
        if untagged:
            example = train_dataset.data[0].get('speaker', '?')
            raise ValueError(
                f"heldout_session={want} was requested but {untagged} of "
                f"{total_samples} training rows carry no 'Ses<NN>' session "
                f"tag, so the fold would not be session-independent. Example "
                f"speaker id: {example!r}.")
        available = sorted(set(sessions))
        if want not in available:
            raise ValueError(
                f"heldout_session={want} matched no rows; the training pool "
                f"holds sessions {available}.")
        for idx, sess in enumerate(sessions):
            (val_indices if sess == want else train_indices).append(idx)
        val_dist = {}
        for idx in val_indices:
            val_dist[strat_labels[idx]] = val_dist.get(strat_labels[idx], 0) + 1
        print(f"  Session-independent fold: holding out session {want} of "
              f"{available}")
        print(f"   {len(train_indices)} train rows, {len(val_indices)} "
              f"evaluation rows")
        print(f"   Held-out class counts: {dict(sorted(val_dist.items()))}")
    elif bool(getattr(config, 'val_split_official', False)):
        # The corpus's own split assignment, read from the official_split
        # field the loader keeps. MSP-Podcast's Train and Development splits
        # are speaker-disjoint, so this is the standard protocol and gives a
        # validation set with no speaker shared with training.
        #
        # It costs training data, and not evenly: angry is 26.6 percent of
        # Development against 11.4 percent of Train, so holding Development
        # out drops 46 percent of the corpus's angry four-class rows against
        # roughly 27 percent of happy and sad. Chosen deliberately here for
        # the direction-aware arms, whose realness map is fitted on Train
        # alone; a map fitted across a merged pool would let validation rows
        # influence training targets through their own labels.
        #
        # Runs made with this are not comparable to runs made without it,
        # because it changes which rows are trained on.
        for idx in range(total_samples):
            which = str(train_dataset.data[idx].get('official_split', ''))
            if which == 'Train':
                train_indices.append(idx)
            elif which:
                val_indices.append(idx)
        if not train_indices or not val_indices:
            raise ValueError(
                "val_split_official is set but the corpus carries no usable "
                "official_split field. MSP-Podcast supplies Split_Set; other "
                "corpora do not, so use val_split_by_speaker or the default "
                "stratified split instead.")
        val_dist = {}
        for idx in val_indices:
            val_dist[strat_labels[idx]] = val_dist.get(strat_labels[idx], 0) + 1
        train_dist = {}
        for idx in train_indices:
            train_dist[strat_labels[idx]] = train_dist.get(strat_labels[idx], 0) + 1
        print(f"  Official corpus split: {len(train_indices)} train rows, "
              f"{len(val_indices)} validation rows")
        print(f"   Train class counts: {dict(sorted(train_dist.items()))}")
        print(f"   Val class counts: {dict(sorted(val_dist.items()))}")
    elif bool(getattr(config, 'val_split_by_speaker', False)):
        # Speaker-disjoint validation. The class-stratified split below puts
        # utterances from the same speaker on both sides, so validation
        # measures performance on speakers the model has already fitted and
        # cannot be an honest generalisation estimate. Here whole speakers go
        # to one side or the other.
        #
        # Speakers are consumed in random order until the validation share is
        # reached, so class balance is approximate rather than exact. With
        # 2,748 speakers on MSP-Podcast the law of large numbers keeps it
        # close; on a corpus with few speakers it will not, which is why this
        # is opt-in rather than the default.
        #
        # Note this changes which rows are TRAINED on, so runs made with it
        # are not comparable to runs made without it.
        speakers = defaultdict(list)
        for idx in range(total_samples):
            speakers[train_dataset.data[idx].get('speaker', f'row{idx}')].append(idx)
        order = sorted(speakers.keys())
        split_rng.shuffle(order)
        target = int(total_samples * config.val_split)
        taken = 0
        val_speakers = set()
        for spk in order:
            if taken >= target:
                break
            val_speakers.add(spk)
            taken += len(speakers[spk])
        for spk, idxs in speakers.items():
            (val_indices if spk in val_speakers else train_indices).extend(idxs)
        val_dist = {}
        for idx in val_indices:
            val_dist[strat_labels[idx]] = val_dist.get(strat_labels[idx], 0) + 1
        print(f"  Speaker-disjoint split: {len(val_speakers)} of {len(speakers)} "
              f"speakers held out, {len(val_indices)} rows "
              f"({100.0 * len(val_indices) / max(1, total_samples):.1f} percent)")
        print(f"   Val class counts: {dict(sorted(val_dist.items()))}")
    else:
        for label in sorted(class_indices.keys()):
            indices = class_indices[label]
            split_rng.shuffle(indices)
            val_size = max(1, int(len(indices) * config.val_split))
            val_indices.extend(indices[:val_size])
            train_indices.extend(indices[val_size:])

    # Optional class-stratified subsample of the TRAINING pool only.
    #
    # Unfrozen runs carry no frame cache and push raw audio through
    # emotion2vec forward and backward every step, measured at 47 minutes per
    # epoch on the full 80,941-row pool, which makes a multi-arm sweep
    # impossible. Subsampling is legitimate for a controlled comparison as
    # long as every arm draws the same rows, which the fixed seed guarantees.
    # Validation is deliberately left at full size so model selection and the
    # reported val metrics stay comparable to the full-data runs.
    frac = float(getattr(config, 'train_subsample_frac', 1.0))
    if 0.0 < frac < 1.0:
        sub_rng = np.random.RandomState(int(config.seed) + 12345)
        by_class = defaultdict(list)
        for idx in train_indices:
            by_class[strat_labels[idx]].append(idx)
        kept = []
        for label in sorted(by_class):
            pool = list(by_class[label])
            sub_rng.shuffle(pool)
            kept.extend(pool[:max(1, int(round(len(pool) * frac)))])
        print(f"  Train subsample: {len(train_indices)} -> {len(kept)} rows "
              f"(frac={frac}, class-stratified, seed-fixed)")
        train_indices = kept

    # Deterministic order so dumped prediction rows align across arms.
    train_indices.sort()
    val_indices.sort()

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Print class distribution in val set
    val_labels = [labels[i] for i in val_indices]
    val_dist = {l: val_labels.count(l) for l in sorted(set(val_labels))}
    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")
    print(f"   Val class distribution: {val_dist}")

    # Two-stage training: compute difficulty and split train indices
    use_two_stage = getattr(config, 'use_two_stage_training', False)
    proto_train_indices = None

    # Curriculum learning setup: precompute difficulties/labels for the sorting
    # step in create_curriculum_subset. Only enabled in the single-stage path;
    # if both flags are on, two-stage wins and curriculum is disabled with a
    # printed warning (they are incompatible: two-stage already changes the
    # dataloader per stage).
    use_curriculum = bool(getattr(config, 'use_curriculum_learning', False))
    if use_curriculum and use_two_stage:
        print("  WARNING: use_curriculum_learning and use_two_stage_training both set. "
              "Disabling curriculum for this run.")
        use_curriculum = False

    curriculum_train_labels = None
    curriculum_train_difficulties = None
    curriculum_pacing_fn = None
    if use_curriculum or use_two_stage:
        difficulties = _compute_per_sample_difficulty(train_dataset, config)

    if use_two_stage:
        # Compute threshold on training indices only
        train_difficulties = difficulties[train_indices]
        threshold = np.percentile(train_difficulties, config.stage1_percentile)
        # Prototypical subset: samples with difficulty <= threshold
        proto_train_indices = [idx for idx in train_indices if difficulties[idx] <= threshold]
        print(f"  Two-stage training enabled:")
        print(f"   Stage 1: {len(proto_train_indices)}/{len(train_indices)} prototypical samples (percentile={config.stage1_percentile}, threshold={threshold:.4f})")
        print(f"   Stage 1 epochs: {config.stage1_epochs}, Stage 2 LR factor: {config.stage2_lr_factor}")

    if use_curriculum:
        curriculum_train_labels = [train_dataset.data[i]['label'] for i in train_indices]
        curriculum_train_difficulties = difficulties[train_indices]
        curriculum_pacing_fn = get_curriculum_pacing_function(
            getattr(config, 'curriculum_pacing', 'linear')
        )
        print(f"  Curriculum learning enabled:")
        print(f"   type={config.curriculum_type}, pacing={config.curriculum_pacing}, "
              f"epochs={config.curriculum_epochs}, "
              f"post_dropout={config.post_curriculum_dropout}")

    # Proto/atypical sub-label split precompute. The head is already widened
    # inside create_model when config.use_proto_atyp_split is True; here we
    # build the centers and per-class thresholds that batch_sub_labels needs.
    use_proto_atyp = bool(getattr(config, 'use_proto_atyp_split', False))
    proto_atyp_ctx = None
    if use_proto_atyp:
        for _flag in ('use_prototypical_weighting', 'use_prototypical_label_smoothing',
                      'use_vrex', 'use_vadmix'):
            if getattr(config, _flag, False):
                print(f"  WARNING: {_flag} is on with use_proto_atyp_split. Disabling "
                      f"{_flag} for this run (they are mutually exclusive).")
                setattr(config, _flag, False)

        center_source = getattr(config, 'proto_atyp_center_source', 'theoretical')
        split_criterion = getattr(
            config, 'proto_atyp_split_criterion', 'per_class_median',
        )
        proto_train_data = [train_dataset.data[i] for i in train_indices]

        if center_source == 'confidence':
            # The split comes from the model's own EMA confidence and does
            # not exist yet. Centers and thresholds stay unused.
            centers_np = np.zeros((config.num_classes, 3), dtype=np.float32)
            thresholds_np = np.zeros(config.num_classes, dtype=np.float32)
        elif center_source == 'random':
            # Placebo arm: the split is a hash of the transcript, so it is
            # stable per utterance but unrelated to the emotion. Centers and
            # thresholds are unused; zeros keep the context shape uniform.
            centers_np = np.zeros((config.num_classes, 3), dtype=np.float32)
            thresholds_np = np.zeros(config.num_classes, dtype=np.float32)
        elif center_source == 'nrc':
            # Lexical prototypicality: distance from the class mean NRC
            # emotion profile of the transcript. Measured to be uncorrelated
            # with the VAD measure (rho = -0.02) and to transfer better
            # across corpora (AUC 0.561 vs 0.521), which is why it is worth
            # a separate arm rather than another VAD center definition.
            centers_np = build_nrc_class_centers(
                train_data=proto_train_data,
                num_classes=config.num_classes,
            )
            thresholds_np = compute_nrc_thresholds(
                train_data=proto_train_data,
                centers=centers_np,
                num_classes=config.num_classes,
                criterion=split_criterion,
            )
        else:
            centers_np = build_class_centers(
                source=center_source,
                expected_vad=config.expected_vad,
                train_data=proto_train_data,
                num_classes=config.num_classes,
            )
            thresholds_np = compute_split_thresholds(
                train_data=proto_train_data,
                centers=centers_np,
                num_classes=config.num_classes,
                criterion=split_criterion,
            )

        proto_atyp_ctx = {
            'centers': torch.tensor(centers_np, dtype=torch.float32).to(device),
            'thresholds': torch.tensor(thresholds_np, dtype=torch.float32).to(device),
            'num_classes': config.num_classes,
            'center_source': center_source,
            'seed': int(getattr(config, 'seed', 42)),
        }
        print(f"  Proto/atypical split enabled:")
        print(f"   center_source={center_source}, "
              f"split_criterion={split_criterion}")
        if center_source == 'confidence':
            print(f"   split will be frozen from the model's own EMA "
                  f"confidence at epoch "
                  f"{int(getattr(config, 'confidence_freeze_epoch', 5))}\n"
                  f"   metric={getattr(config, 'confidence_metric', 'margin')}, "
                  f"momentum={getattr(config, 'confidence_momentum', 0.9)}; "
                  f"until then only the\n   parent 4-way task is trained via "
                  f"collapsed class probabilities.")
        elif center_source == 'random':
            print("   PLACEBO arm: sub-labels are a hash of the transcript "
                  "and carry no\n   information about the emotion. Any gain "
                  "here is head widening, not\n   prototypicality.")
        elif center_source == 'nrc':
            n_scored = sum(
                1 for it in proto_train_data
                if profile_for_text(it.get('transcript', ''), load_lexicon())
                is not None
            )
            print(f"   {n_scored}/{len(proto_train_data)} training utterances "
                  f"contain NRC affect words; the rest are treated as "
                  f"prototypical")
            for c in range(config.num_classes):
                top = np.argsort(-centers_np[c])[:3]
                desc = ", ".join(
                    f"{NRC_CATEGORIES[i]} {centers_np[c, i]:.2f}" for i in top
                )
                print(f"     class {c}: top categories {desc}, "
                      f"median_dist={thresholds_np[c]:.4f}")
        else:
            print(f"   Centers:")
            for c in range(config.num_classes):
                print(f"     class {c}: V={centers_np[c, 0]:.4f} "
                      f"A={centers_np[c, 1]:.4f} "
                      f"D={centers_np[c, 2]:.4f}, "
                      f"median_dist={thresholds_np[c]:.4f}")

    # Confidence tracker: needed by the self-derived proto/atyp split and
    # by ambiguity weighting. One tracker serves both.
    confidence_tracker = None
    use_amb = bool(getattr(config, 'use_ambiguity_weighting', False))
    if use_amb or (use_proto_atyp and getattr(
            config, 'proto_atyp_center_source', '') == 'confidence'):
        confidence_tracker = ConfidenceTracker(
            num_samples=len(train_dataset.data),
            num_classes=int(config.num_classes),
            momentum=float(getattr(config, 'confidence_momentum', 0.9)),
            metric=str(getattr(config, 'confidence_metric', 'margin')),
        )

    # Ambiguity-weighted losses (evidence BCE or anti-neutral margin).
    ambiguity_weights = None
    if use_amb:
        ambiguity_weights = AmbiguityWeights(
            tracker=confidence_tracker,
            beta=float(getattr(config, 'ambiguity_beta', 1.0)),
            warmup_epochs=int(getattr(config, 'ambiguity_warmup_epochs', 3)),
            shuffle=bool(getattr(config, 'ambiguity_shuffle', False)),
            seed=int(getattr(config, 'seed', 42)),
        )
        shuffled = " (PLACEBO: weights shuffled)" if getattr(
            config, 'ambiguity_shuffle', False) else ""
        print(f"  Ambiguity weighting enabled: beta="
              f"{getattr(config, 'ambiguity_beta', 1.0)}, warmup="
              f"{getattr(config, 'ambiguity_warmup_epochs', 3)} epochs"
              f"{shuffled}")

    if bool(getattr(config, 'use_evidence_heads', False)):
        for _flag in ('use_neutral_soft_labels', 'use_logit_adjustment',
                      'use_prototypical_weighting',
                      'use_prototypical_label_smoothing'):
            if getattr(config, _flag, False):
                print(f"  WARNING: {_flag} has no effect with "
                      f"use_evidence_heads; disabling it for this run.")
                setattr(config, _flag, False)
        print(f"  Evidence heads enabled: threshold="
              f"{getattr(config, 'evidence_threshold', 0.5)} "
              f"(neutral = no head fires)")
    if bool(getattr(config, 'use_anti_neutral_margin', False)):
        print(f"  Anti-neutral margin enabled: m="
              f"{getattr(config, 'anti_neutral_margin', 1.0)}, weight="
              f"{getattr(config, 'anti_neutral_weight', 0.5)}")

    # Intensity-dependent soft targets toward neutral. The per-class
    # distance range is measured once over the training split: classes sit
    # at different distances from neutral by construction (angry is far,
    # sad is close), so a global scale would mark most sad samples weak
    # purely because sadness is low-arousal.
    intensity_ctx = None
    if bool(getattr(config, 'use_neutral_soft_labels', False)):
        neutral_vad_np = np.asarray(
            config.expected_vad[0], dtype=np.float32,
        )
        dists_by_class = {c: [] for c in range(config.num_classes)}
        for i in train_indices:
            item = train_dataset.data[i]
            if item.get('dataset') not in DATASETS_WITH_VAD:
                continue
            c = int(item['label'])
            if c < 0 or c >= config.num_classes:
                continue
            point = np.array([item['valence'], item['arousal'],
                              item['dominance']], dtype=np.float32)
            dists_by_class[c].append(
                float(np.linalg.norm(point - neutral_vad_np))
            )
        ranges = np.zeros((config.num_classes, 2), dtype=np.float32)
        for c in range(config.num_classes):
            vals = dists_by_class[c]
            if vals:
                # Use percentiles rather than min/max so a single outlier
                # does not compress the whole class into the low end.
                ranges[c, 0] = float(np.percentile(vals, 5))
                ranges[c, 1] = float(np.percentile(vals, 95))
            else:
                ranges[c] = (0.0, 1.0)
        intensity_ctx = {
            'class_ranges': torch.tensor(ranges, dtype=torch.float32).to(device),
            'neutral_vad': torch.tensor(
                neutral_vad_np, dtype=torch.float32).to(device),
        }
        print(f"  Neutral soft labels enabled: alpha="
              f"{getattr(config, 'neutral_soft_alpha', 0.3)}")
        for c in range(config.num_classes):
            print(f"     class {c}: intensity range "
                  f"[{ranges[c, 0]:.3f}, {ranges[c, 1]:.3f}]")

    # Auxiliary VAD-cluster multitask precompute. K-means centroids are fit
    # once over VAD points from the training split (across all VAD-annotated
    # corpora combined). The centroids are stored as a tensor and the aux
    # head lives inside the model (added conditionally by create_model).
    # Target-intensity bound for VAD-calibrated muted mixup, computed once over
    # the training split. A batch holds only a handful of emotional samples, so
    # a batch-local quantile would make the target drift with batch composition.
    if (bool(getattr(config, 'use_muted_mixup', False))
            and bool(getattr(config, 'muted_mixup_vad_calibrated', False))):
        exp_vad_cfg = getattr(config, 'expected_vad', None) or {}
        neutral_centre_cfg = exp_vad_cfg.get(0) or exp_vad_cfg.get('0')
        if neutral_centre_cfg is None:
            raise ValueError(
                "muted_mixup_vad_calibrated needs expected_vad[0] as the "
                "neutral centre, but expected_vad is missing or has no key 0."
            )
        centre_np = np.asarray(neutral_centre_cfg, dtype=np.float64)
        dists = []
        for i in train_indices:
            row = train_dataset.data[i]
            if row.get('label', 0) == 0:
                continue
            dists.append(np.linalg.norm(
                np.array([row.get('valence', 0.0), row.get('arousal', 0.0),
                          row.get('dominance', 0.0)], dtype=np.float64) - centre_np))
        q = float(getattr(config, 'muted_mixup_vad_quantile', 0.5))
        config._muted_vad_span = float(np.quantile(dists, q)) if dists else None
        print(f"  Muted mixup VAD calibration: {len(dists)} emotional train "
              f"samples, intensity q{q:.2f} span={config._muted_vad_span:.4f}, "
              f"shuffle_control={bool(getattr(config, 'muted_mixup_vad_shuffle', False))}")

        if bool(getattr(config, 'muted_mixup_sparse_fill', False)):
            # Per class, decide which intensity end is thin and aim there.
            n_cls = int(getattr(config, 'num_classes', 4))
            per_class = {c: [] for c in range(1, n_cls)}
            for i in train_indices:
                row = train_dataset.data[i]
                lab = row.get('label', 0)
                if lab == 0 or lab not in per_class:
                    continue
                per_class[lab].append(np.linalg.norm(
                    np.array([row.get('valence', 0.0), row.get('arousal', 0.0),
                              row.get('dominance', 0.0)],
                             dtype=np.float64) - centre_np))
            only = set(getattr(config, 'muted_mixup_sparse_classes', []) or [])
            tq = float(getattr(config, 'muted_mixup_sparse_target_q', 0.1))
            direction = [0] * n_cls
            target = [0.0] * n_cls
            for c, vals in per_class.items():
                if not vals or (only and c not in only):
                    continue
                arr = np.asarray(vals)
                lo_q, hi_q = np.quantile(arr, 0.25), np.quantile(arr, 0.75)
                n_lo = int((arr <= lo_q).sum())
                n_hi = int((arr >= hi_q).sum())
                # Thin at the quiet end means mute; thin when loud means amplify.
                if n_lo <= n_hi:
                    direction[c] = -1
                    target[c] = float(np.quantile(arr, tq))
                else:
                    direction[c] = +1
                    target[c] = float(np.quantile(arr, 1.0 - tq))
            config._sparse_direction = direction
            config._sparse_target = target
            names = ['neutral', 'happy', 'sad', 'angry']
            print("  Sparse-fill mixup targets:")
            for c in range(1, n_cls):
                nm = names[c] if c < len(names) else f"class{c}"
                verb = {-1: 'mute', 1: 'amplify', 0: 'skip'}[direction[c]]
                print(f"    {nm:8s} n={len(per_class.get(c, [])):>6d} "
                      f"{verb:8s} target={target[c]:.4f}")

    use_aux_vad = bool(getattr(config, 'use_aux_vad_cluster', False))
    aux_vad_centroids = None
    if use_aux_vad:
        aux_task = getattr(config, 'aux_vad_task', 'cluster')
        if aux_task == 'regression':
            # Regression arm predicts raw VAD; no centroids to fit.
            print(f"  Aux VAD regression multitask enabled: "
                  f"weight={getattr(config, 'aux_vad_cluster_weight', 0.2)}")
        elif aux_task == 'resid_cluster':
            # Cluster whitened residuals rather than raw VAD. Needs the class
            # centres and covariances fitted from the training split, which is
            # also what makes the residual well defined.
            n_cls = int(getattr(config, 'num_classes', 4))
            k_res = int(getattr(config, 'aux_resid_cluster_k', 6))
            scope = getattr(config, 'aux_vad_cluster_scope', 'shared')
            _rm, _ri = class_vad_stats(
                [train_dataset.data[i] for i in train_indices], n_cls,
                consensus_q=float(getattr(config, 'proto_centroid_consensus_q', 0.0)))
            _rw = whitening_matrices(_ri)
            config._resid_means = _rm
            config._resid_whiten = _rw
            # Stashed on config rather than threaded through train_epoch's
            # signature: they are fixed constants fitted once from the
            # training split, and every call site already receives config.
            config._resid_means_t = torch.tensor(
                _rm, dtype=torch.float32).to(device)
            config._resid_whiten_t = torch.tensor(
                _rw, dtype=torch.float32).to(device)
            centroids_np = build_residual_centroids(
                train_data=[train_dataset.data[i] for i in train_indices],
                means=_rm, whiten=_rw, k=k_res, num_classes=n_cls,
                scope=scope, seed=int(getattr(config, 'seed', 42)),
            )
            aux_vad_centroids = torch.tensor(
                centroids_np, dtype=torch.float32).to(device)
            n_labels = (n_cls * k_res if scope == 'per_class' else k_res)
            print(f"  Aux residual-cluster multitask enabled:")
            print(f"   scope={scope}, k={k_res} -> {n_labels} labels, "
                  f"weight={getattr(config, 'aux_vad_cluster_weight', 0.2)}, "
                  f"shuffle={getattr(config, 'aux_vad_shuffle', False)}")
            print(f"   fitted class centres: {np.round(_rm, 4).tolist()}")
            if scope != 'per_class':
                for j in range(k_res):
                    print(f"     mode {j}: dV={centroids_np[j, 0]:+.3f} "
                          f"dA={centroids_np[j, 1]:+.3f} "
                          f"dD={centroids_np[j, 2]:+.3f}")
        elif aux_task == 'subtype':
            # Targets come from the annotator metadata join, so there is
            # nothing to fit here either.
            print(f"  Aux annotator-subtype multitask enabled: "
                  f"{len(SUBTYPE_NAMES)} tags, "
                  f"weight={getattr(config, 'aux_vad_cluster_weight', 0.2)}, "
                  f"shuffle={getattr(config, 'aux_vad_shuffle', False)}")
        elif getattr(config, 'aux_vad_cluster_scope', 'global') == 'per_class':
            # One k-means inside each class: happy-1..happy-n, angry-1..angry-n.
            kpc = int(getattr(config, 'aux_vad_clusters_per_class', 2))
            n_cls = int(getattr(config, 'num_classes', 4))
            aux_init = getattr(config, 'aux_vad_cluster_init', 'random')
            centroids_np = build_per_class_vad_centroids(
                train_data=[train_dataset.data[i] for i in train_indices],
                clusters_per_class=kpc,
                num_classes=n_cls,
                seed=int(getattr(config, 'seed', 42)),
                init=aux_init,
            )
            aux_vad_centroids = torch.tensor(centroids_np, dtype=torch.float32).to(device)
            print(f"  Aux VAD per-class subtype multitask enabled:")
            print(f"   {kpc} subtypes x {n_cls} classes = {kpc * n_cls} labels, "
                  f"init={aux_init}, "
                  f"weight={getattr(config, 'aux_vad_cluster_weight', 0.2)}")
            class_names = ['neutral', 'happy', 'sad', 'angry']
            for c in range(n_cls):
                name = class_names[c] if c < len(class_names) else f"class{c}"
                for j in range(kpc):
                    print(f"     {name}-{j + 1}: V={centroids_np[c, j, 0]:.4f} "
                          f"A={centroids_np[c, j, 1]:.4f} "
                          f"D={centroids_np[c, j, 2]:.4f}")
        else:
            k = int(getattr(config, 'aux_vad_cluster_k', 8))
            aux_init = getattr(config, 'aux_vad_cluster_init', 'random')
            permuted = bool(getattr(config, 'aux_vad_permute_clusters', False))
            centroids_np, _assignments = build_vad_centroids(
                train_data=[train_dataset.data[i] for i in train_indices],
                k=k,
                seed=int(getattr(config, 'seed', 42)),
                init=aux_init,
                num_classes=int(getattr(config, 'num_classes', 4)),
            )
            aux_vad_centroids = torch.tensor(centroids_np, dtype=torch.float32).to(device)
            print(f"  Aux VAD cluster multitask enabled:")
            print(f"   k={k}, init={aux_init}, permuted={permuted}, "
                  f"weight={getattr(config, 'aux_vad_cluster_weight', 0.2)}")
            if permuted:
                print("   NOTE: permuted-label control arm; centroids are "
                      "fit but per-sample labels are scrambled at loss time.")
            for j in range(k):
                print(f"     cluster {j}: V={centroids_np[j, 0]:.4f} "
                      f"A={centroids_np[j, 1]:.4f} D={centroids_np[j, 2]:.4f}")

    # Create dataloaders
    num_workers = getattr(config, 'num_workers', 2)
    eval_batch_size = getattr(config, 'eval_batch_size', None) or config.batch_size
    eval_batch_size = int(eval_batch_size)
    pin_memory = device.type == 'cuda'
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    # Optional corpus-balanced sampling: WeightedRandomSampler that draws
    # samples so each batch has roughly equal share per corpus regardless of
    # natural dataset size. Required for VREx (per-corpus loss estimates need
    # enough samples to be stable). Use corpus_sample_weights to set explicit
    # per-corpus oversampling factors, otherwise weights are uniform across corpora.
    train_sampler = None
    if getattr(config, 'use_corpus_balanced_sampler', False):
        train_corpora = [train_dataset.data[i]['dataset'] for i in train_indices]
        corpus_counts = {}
        for c in train_corpora:
            corpus_counts[c] = corpus_counts.get(c, 0) + 1
        explicit_w = getattr(config, 'corpus_sample_weights', None) or {}
        # Per-sample weight = corpus_factor / corpus_count[corpus]
        sample_weights = []
        for c in train_corpora:
            corpus_factor = float(explicit_w.get(c, 1.0))
            sample_weights.append(corpus_factor / corpus_counts[c])
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
        )
        print(f"  Corpus-balanced sampler enabled. Counts: {corpus_counts}")
        if explicit_w:
            print(f"   Explicit per-corpus weights: {explicit_w}")

    # Length-bucketed batching for raw-audio runs. The collate pads to the
    # longest waveform in the batch, so random batching wastes ~40% of the
    # compute on padding and the waste grows with batch size. Grouping similar
    # durations cuts it to under 2%.
    length_sampler = None
    if (bool(getattr(config, 'use_length_bucketing', False))
            and getattr(train_dataset, 'uses_raw_audio', False)
            and train_sampler is None):
        try:
            durations = load_or_compute_durations(
                train_dataset.dataset_name, train_dataset.hf_dataset)
            # Map each position in the Subset back to its row duration.
            subset_durs = np.array([
                durations[train_dataset.data[i]['hf_idx']]
                for i in train_indices], dtype=np.float32)
            length_sampler = LengthBucketedBatchSampler(
                durations=subset_durs,
                indices=range(len(train_indices)),
                batch_size=config.batch_size,
                window_batches=int(getattr(config, 'length_bucket_window', 50)),
                drop_last=True,
                seed=int(getattr(config, 'seed', 42)),
            )
            # Seed diagnostic. Runs on the official split came out bitwise
            # identical across seeds although every component varies with
            # the seed in isolation, so log what the sampler was actually
            # built with and the order it actually yields. If these match
            # across two runs with different config.seed, the data path is
            # the culprit; if they differ and the results still match, it
            # is not.
            _first = next(iter(length_sampler))
            print(f"  SEED DIAG: config.seed={config.seed} "
                  f"sampler.seed={length_sampler.seed} "
                  f"first_batch={_first[:6]}")
            print(f"  Length bucketing enabled: {len(length_sampler)} batches, "
                  f"durations {subset_durs.min():.1f}-{subset_durs.max():.1f}s "
                  f"(mean {subset_durs.mean():.1f}s)")
        except Exception as e:
            print(f"  Length bucketing unavailable ({e}); falling back to "
                  f"random batching")
            length_sampler = None

    if length_sampler is not None:
        train_loader = DataLoader(
            train_subset,
            batch_sampler=length_sampler,
            collate_fn=vad_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            collate_fn=vad_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=vad_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    test_loaders = []
    for test_dataset in test_datasets:
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=vad_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        test_loaders.append(test_loader)

    # Create model
    print(f"\n  Creating model...")
    model = create_model(config).to(device)

    # Seed diagnostic. Sum the freshly initialised trainable weights. If
    # this matches across two runs with different config.seed, the model is
    # not being initialised from the seeded generator, whatever the data
    # order does.
    with torch.no_grad():
        _init_sum = float(sum(float(p.detach().double().sum())
                              for p in model.parameters()
                              if p.requires_grad))
    print("  SEED DIAG: initial trainable weight sum = %.6f" % _init_sum)
    print(f"   Modality: {config.modality}")
    print(f"   Task: {config.task_type}")
    print(f"   Audio encoder: {getattr(config, 'audio_encoder_type', 'preextracted')}")
    print(f"   Contrastive: {config.use_contrastive}")

    # Cache frozen encoder features: compute once, reuse every epoch (~100x faster)
    audio_encoder_type = getattr(config, 'audio_encoder_type', 'preextracted')
    if (audio_encoder_type in RAW_AUDIO_ENCODERS
            and getattr(config, 'unfreeze_audio_layers', 0) == 0
            and hasattr(model, 'audio_encoder') and model.audio_encoder is not None):
        print(f"\n  Caching frozen encoder features (one-time cost)...")
        train_dataset.cache_encoder_features(model.audio_encoder, device, batch_size=config.batch_size)
        for td in test_datasets:
            td.cache_encoder_features(model.audio_encoder, device, batch_size=config.batch_size)

    # Create loss functions
    logit_adjust = None
    evidence_pos_weight = None
    hierarchical_emotion_weights = None
    if config.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        # Compute inverse-frequency class weights for imbalanced data
        class_counts = [0, 0, 0, 0]
        for item in train_dataset.data:
            label = item["label"]
            if label < 4:
                class_counts[label] += 1

        total_samples = sum(class_counts)
        # How hard to correct for the training-corpus class imbalance.
        #   inverse_freq      w ~ 1 / freq       (matches balanced risk, but
        #                                         amplifies minority gradients)
        #   sqrt_inverse_freq w ~ 1 / sqrt(freq) (lower variance, under-corrects)
        #   none              w = 1              (use with logit adjustment)
        weight_mode = str(getattr(config, 'class_weight_mode', 'inverse_freq'))
        freq_weights = []
        for i in range(config.num_classes):
            if class_counts[i] > 0:
                freq_ratio = class_counts[i] / total_samples
                if weight_mode == 'none':
                    freq_weight = 1.0
                elif weight_mode == 'sqrt_inverse_freq':
                    freq_weight = (1.0 / math.sqrt(freq_ratio)) / config.num_classes
                else:
                    freq_weight = (1.0 / freq_ratio) / config.num_classes
            else:
                freq_weight = 1.0
            freq_weights.append(freq_weight)

        # Normalize so weights sum to num_classes
        total_weight = sum(freq_weights)
        freq_weights = [w / total_weight * config.num_classes for w in freq_weights]

        # Evidence heads need the within-head positive/negative ratio
        # corrected, which the per-sample class weight above cannot do.
        # Identity-term weights for the hierarchical head: inverse
        # frequency over emotional classes only. The detector is left
        # unweighted because it is already near-balanced.
        evidence_pos_weight = None
        if bool(getattr(config, 'use_evidence_heads', False)):
            evidence_pos_weight = head_pos_weights(
                class_counts[:config.num_classes],
                cap=float(getattr(config, 'evidence_pos_weight_cap', 10.0)),
            ).to(device)
            print(f"   Evidence head pos_weight: "
                  f"{[round(float(w), 2) for w in evidence_pos_weight]}")

        print(f"   Class counts: {class_counts}")
        print(f"   Class weight mode: {weight_mode}")
        print(f"   Class weights: {[f'{w:.3f}' for w in freq_weights]}")

        freq_weights_tensor = torch.tensor(freq_weights, dtype=torch.float32).to(device)

        # Logit adjustment: add tau * log(prior) to the logits during training
        # only. Fisher-consistent for balanced error (which is what UAR
        # measures) and, unlike reweighting, it shifts the decision boundary
        # instead of scaling minority-class gradients.
        if bool(getattr(config, 'use_logit_adjustment', False)):
            tau = float(getattr(config, 'logit_adjustment_tau', 1.0))
            priors = [
                max(class_counts[i], 1) / total_samples
                for i in range(config.num_classes)
            ]
            logit_adjust = torch.tensor(
                [tau * math.log(p) for p in priors],
                dtype=torch.float32,
            ).to(device)
            print(f"   Logit adjustment: tau={tau}, "
                  f"offsets={[f'{v:.3f}' for v in logit_adjust.tolist()]}")
            if weight_mode != 'none':
                print("   WARNING: logit adjustment and class reweighting both "
                      "correct for imbalance. Set class_weight_mode: none to "
                      "avoid double-correcting.")

        # Mahalanobis atypicality as a CE weight needs the fitted class
        # statistics, which are otherwise only built when the auxiliary head
        # is on. Fit them here so a weighting-only arm carries no extra head
        # and stays capacity matched to the plain baseline.
        if str(getattr(config, 'ce_weight_source', 'vad')) == 'mahal':
            _wm, _wi = class_vad_stats(
                [train_dataset.data[i] for i in train_indices],
                int(config.num_classes),
                dims=tuple(getattr(config, 'proto_vad_dims', (0, 1, 2))))
            config._ce_means = _wm
            config._ce_inv_covs = _wi
            print(f"   CE weighting by Mahalanobis atypicality: class stats "
                  f"fitted on {len(train_indices)} training rows")

        # Use reduction='none' if we need per-sample weighting or label smoothing
        use_proto_weight = getattr(config, 'use_prototypical_weighting', False)
        use_label_smooth = getattr(config, 'use_prototypical_label_smoothing', False)
        use_salience_weight = getattr(config, 'use_salience_weighting', False)
        if use_proto_weight or use_label_smooth or use_salience_weight:
            criterion = nn.CrossEntropyLoss(weight=freq_weights_tensor, reduction='none')
            print(f"   Prototypical weighting: {use_proto_weight} (alpha={getattr(config, 'prototypical_weighting_alpha', 2.0)})")
            print(f"   Label smoothing: {use_label_smooth} (beta={getattr(config, 'label_smoothing_beta', 0.5)}, max={getattr(config, 'label_smoothing_max', 0.6)})")
        elif bool(getattr(config, 'use_ldam_margin', False)):
            # Class-dependent margin. Inverse-frequency weighting corrects the
            # PRIOR, which a model can satisfy by sliding the decision
            # boundary; a margin requires the true class to beat the others by
            # a gap, which a uniform shift cannot produce. Measured motivation:
            # the auxiliary arms raised emotion-vs-emotion AUC on 4/4 corpora
            # while UAR stayed flat, i.e. the gain sat just under the boundary.
            _counts = np.bincount(
                [train_dataset.data[i]['label'] for i in train_indices],
                minlength=int(config.num_classes))
            criterion = LDAMLoss(
                class_counts=_counts,
                max_margin=float(getattr(config, 'ldam_max_margin', 0.5)),
                scale=float(getattr(config, 'ldam_scale', 30.0)),
                weight=freq_weights_tensor,
            ).to(device)
            print(f"   LDAM margin: max={getattr(config, 'ldam_max_margin', 0.5)}, "
                  f"scale={getattr(config, 'ldam_scale', 30.0)}, "
                  f"counts={_counts.tolist()}")
            print(f"   per-class margins: "
                  f"{criterion.margins.cpu().numpy().round(4).tolist()}")
        else:
            criterion = nn.CrossEntropyLoss(weight=freq_weights_tensor)

        # Salience weighting constants come from the training split only, so
        # neither the validation split nor any evaluation corpus influences
        # which samples get emphasised.
        salience_stats = None
        if use_salience_weight or getattr(config, 'use_salience_gate', False):
            _sal_rows = [train_dataset.data[i] for i in train_indices]
            _sal_vad = np.array([[r['valence'], r['arousal'], r['dominance']]
                                 for r in _sal_rows], dtype=np.float32)
            _sal_lab = np.array([int(r['label']) for r in _sal_rows])
            salience_stats = compute_salience_stats(
                _sal_vad, _sal_lab, int(getattr(config, 'num_classes', 4)))
            if use_salience_weight:
                _mode = ("SHUFFLED placebo"
                         if getattr(config, 'salience_shuffle', False) else "real")
                print(f"   Salience weighting: {_mode}, "
                      f"beta={getattr(config, 'salience_beta', 0.5)}, "
                      f"clip={getattr(config, 'salience_clip', 3.0)}")
            if getattr(config, 'use_salience_gate', False):
                # The gate standardises predicted VAD with training-split
                # constants, so they must be installed before the first step.
                model.set_salience_reference(
                    salience_stats["mean"], salience_stats["scale"],
                    salience_stats["neutral_centre"])
                print("   Salience gate reference installed from training split")

    contrastive_criterion = None
    if config.use_contrastive:
        contrastive_criterion = create_contrastive_loss(
            config.contrastive_loss_type,
            temperature=config.contrastive_temperature,
            alpha=config.prototypical_alpha,
            beta=config.prototypical_beta,
            threshold=config.prototypical_threshold,
            # For prototype_anchored loss
            projection_dim=getattr(config, 'projection_dim', 128),
            num_classes=config.num_classes,
            expected_vad=getattr(config, 'expected_vad', None),
            margin_base=getattr(config, 'margin_base', 0.1),
            separation_margin=getattr(config, 'separation_margin', 2.0),
            separation_weight=getattr(config, 'separation_weight', 0.5),
            alignment_weight=getattr(config, 'alignment_weight', 1.0),
            use_adaptive_difficulty=getattr(config, 'use_adaptive_difficulty', False),
            use_hard_negative_mining=getattr(config, 'use_hard_negative_mining', False),
            hard_negative_weight=getattr(config, 'hard_negative_weight', 0.3),
            use_memory_bank=getattr(config, 'use_memory_bank', False),
            bank_size=int(getattr(config, 'bank_size', 64)),
            bank_momentum=getattr(config, 'bank_momentum', 0.5),
            bank_threshold=getattr(config, 'bank_threshold', 0.5),
        ).to(device)
        print(f"   Contrastive loss: {config.contrastive_loss_type}")

    # Domain adversarial training
    domain_discriminator = None
    modality_discriminator = None
    domain_adv_loss = None
    use_adversarial = getattr(config, 'use_domain_adversarial', False)
    use_modality_adversarial = getattr(config, 'use_modality_adversarial', False)

    # Determine number of domains from training data
    train_names = config.train_dataset
    if isinstance(train_names, str):
        train_names = [train_names]
    num_domains = len(train_names)

    if (use_adversarial or use_modality_adversarial) and num_domains < 2:
        print("   WARNING: Domain adversarial requires multi-corpus training. Disabling.")
        use_adversarial = False
        use_modality_adversarial = False

    if use_adversarial or use_modality_adversarial:
        adv_alpha = getattr(config, 'adversarial_alpha', 2.0)
        adv_hidden = getattr(config, 'adversarial_hidden_dim', 256)
        domain_adv_loss = PrototypicalDomainAdversarialLoss(alpha=adv_alpha).to(device)
        print(f"   Domain adversarial: {num_domains} domains, alpha={adv_alpha}")
        print(f"   Prototypicality-weighted: {getattr(config, 'use_prototypical_adversarial', True)}")

    if use_adversarial:
        adv_hidden = getattr(config, 'adversarial_hidden_dim', 256)
        domain_discriminator = DomainDiscriminator(
            input_dim=config.hidden_dim,
            hidden_dim=adv_hidden,
            num_domains=num_domains,
        ).to(device)
        print(f"   Embedding-level discriminator: hidden={adv_hidden}")
        print(f"   Adversarial weight: {getattr(config, 'adversarial_weight', 0.1)}")

    if use_modality_adversarial:
        if config.modality != "both":
            print("   WARNING: Modality adversarial requires modality='both'. Disabling.")
            use_modality_adversarial = False
        else:
            adv_hidden = getattr(config, 'adversarial_hidden_dim', 256)
            # Pre-fusion audio/text feature dims (before the fusion module)
            audio_feat_dim = model.audio_dim if hasattr(model, 'audio_dim') else 768
            text_feat_dim = model.text_dim if getattr(model, 'text_dim', None) else 768
            modality_discriminator = ModalityDomainDiscriminator(
                audio_dim=audio_feat_dim,
                text_dim=text_feat_dim,
                hidden_dim=adv_hidden,
                num_domains=num_domains,
            ).to(device)
            print(f"   Modality-level discriminator: audio={audio_feat_dim}, text={text_feat_dim}")
            print(f"   Modality adversarial weight: {getattr(config, 'modality_adv_weight', 0.1)}")

    # Learnable VAD centroids (for CE prototypicality weighting)
    centroid_tracker = None
    if getattr(config, 'use_learned_centroids', False):
        centroid_tracker = LearnableCentroids(
            expected_vad=config.expected_vad,
            num_classes=config.num_classes,
            mode=getattr(config, 'learned_centroid_mode', 'ema'),
            momentum=getattr(config, 'learned_centroid_momentum', 0.9),
        ).to(device)
        print(f"   Learnable centroids: mode={centroid_tracker.mode}, momentum={centroid_tracker.momentum}")

    if bool(getattr(config, 'use_proto_dist_logits', False)):
        n_cls = int(getattr(config, 'num_classes', 4))
        _dm, _di = class_vad_stats(
            [train_dataset.data[i] for i in train_indices], n_cls,
            consensus_q=float(getattr(config, 'proto_centroid_consensus_q', 0.0)))
        _dw = whitening_matrices(_di)
        if bool(getattr(config, 'proto_dist_permute', False)):
            # Control: give each class another class's prototype and
            # covariance. The stream keeps its shape, its parameter and its
            # gradient path; only the geometry stops corresponding to the
            # class it scores. A gain that survives this is not prototype
            # structure.
            rng = np.random.RandomState(int(getattr(config, 'seed', 42)))
            perm = rng.permutation(n_cls)
            while n_cls > 1 and (perm == np.arange(n_cls)).all():
                perm = rng.permutation(n_cls)
            _dm, _dw = _dm[perm], _dw[perm]
            print(f"   Prototype-distance PERMUTED control: mapping {perm.tolist()}")
        model.set_proto_dist_reference(_dm, _dw)
        print(f"   Prototype-distance alpha floor: "
              f"{float(getattr(config, 'proto_dist_alpha_min', 0.0)):g}")
        print(f"   Prototype-distance reference installed: "
              f"centres {np.round(_dm, 4).tolist()}")


    # GenLabel density bank over learned embeddings. Built once and
    # refitted at every epoch boundary; see utils/genlabel.py for why the
    # authors' frozen-encoder recipe could not be used verbatim.
    genlabel_density = None
    if getattr(config, 'use_genlabel', False):
        genlabel_density = EmbeddingClassDensity(
            num_classes=int(getattr(config, 'num_classes', 4)),
            dim=int(getattr(config, 'hidden_dim', 1024)),
            device=device,
            pca_dim=int(getattr(config, 'genlabel_pca_dim', 0)),
            shrinkage=float(getattr(config, 'genlabel_shrinkage', 0.0)),
            tol=float(getattr(config, 'genlabel_tol', 1e-4)),
            proj_mode=str(getattr(config, 'genlabel_proj_mode', 'within')))
        print(f"   GenLabel density: dim {genlabel_density.dim} -> "
              f"pca {genlabel_density.pca_dim or 'none'}, "
              f"proj {genlabel_density.proj_mode}, "
              f"shrinkage {genlabel_density.shrinkage:g}, "
              f"tol {genlabel_density.tol:g}, "
              f"gamma {float(getattr(config, 'genlabel_gamma', 1.0)):g}")

    # Auxiliary prototypicality predictor
    proto_predictor = None
    proto_predictor_audio = None
    proto_predictor_text = None
    if getattr(config, 'use_proto_predictor', False):
        proto_hidden = getattr(config, 'proto_predictor_hidden_dim', 256)
        # Fit the class centres and covariances from the training split. The
        # configured prototypes are constants and are off by 0.066 for angry,
        # which is 40 percent of that class's own scatter.
        proto_target = str(getattr(config, 'proto_target', 'scalar'))
        proto_metric = str(getattr(config, 'proto_metric', 'euclidean'))
        n_cls = int(getattr(config, 'num_classes', 4))
        # Which VAD axes the prototypicality target uses. Dominance is the
        # axis annotators compress into the narrowest range (p5-p95 of 2.80
        # against 3.25 for valence) while disagreeing about as much in
        # absolute terms, so it is the least reliable of the three relative to
        # its own spread, and whitening by class scatter UP-weights it because
        # its variance is smallest. Dropping it is the direct test of whether
        # it contributes or costs.
        _dims = tuple(getattr(config, 'proto_vad_dims', (0, 1, 2)))
        config._proto_dims = _dims
        _pm, _pi = class_vad_stats(
            [train_dataset.data[i] for i in train_indices], n_cls,
            consensus_q=float(getattr(config, 'proto_centroid_consensus_q', 0.0)),
            dims=_dims,
            consensus_source=str(getattr(config, 'proto_consensus_source', 'class')))
        config._proto_means = _pm
        config._proto_inv_covs = _pi
        # Empirical log class priors over the same rows the Gaussians were
        # fitted on. Only consumed when an arm asks for prior-corrected
        # posteriors; uniform priors remain the default everywhere else.
        _pc = np.bincount(
            [int(train_dataset.data[i].get('label4',
                                           train_dataset.data[i]['label']))
             for i in train_indices],
            minlength=n_cls).astype(np.float64)
        config._proto_log_priors = np.log(_pc / max(1.0, _pc.sum()) + 1e-12)
        # Smallest annotator consensus on the same rows. Used to map
        # consensus onto [0, 1] for the trust-weighted target without
        # introducing a tuned constant.
        # Calibrated prior offset: the log-prior under which the target
        # marginal over BLENDS matches the class distribution. Solved by
        # iterative proportional fitting, so no constant is tuned. Uniform
        # priors under-inject neutral (0.340 vs 0.453) and over-inject sad
        # (0.195 vs 0.107); empirical priors overshoot to 0.510.
        # VAD-free Dirichlet control: concentrations matching q's
        # class-conditional mean and entropy over blends.
        config._dirichlet_conc = None
        if bool(getattr(config, 'muted_mixup_dirichlet', False)):
            _dr = [train_dataset.data[i] for i in train_indices]
            _dv = torch.tensor([[r.get('valence', 0.0), r.get('arousal', 0.0),
                                 r.get('dominance', 0.0)] for r in _dr],
                               dtype=torch.float32)
            _dl = torch.tensor([int(r.get('label4', r['label'])) for r in _dr],
                               dtype=torch.long)
            _dk = _dv.abs().sum(1) > 0
            config._dirichlet_conc = solve_dirichlet_match(
                _dv[_dk], _dl[_dk],
                torch.as_tensor(_pm, dtype=torch.float32),
                torch.as_tensor(_pi, dtype=torch.float32),
                num_classes=n_cls,
                alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                fold_lambda=bool(getattr(config, 'muted_mixup_fold_lambda', True)),
                seed=int(getattr(config, 'seed', 42))).numpy()
            print("   Dirichlet control concentrations:\n%s"
                  % np.round(config._dirichlet_conc, 3))

        config._proto_cal_prior = None
        if bool(getattr(config, 'muted_mixup_calibrated_prior', False)):
            _cal_rows = [train_dataset.data[i] for i in train_indices]
            _cal_vad = torch.tensor(
                [[r.get('valence', 0.0), r.get('arousal', 0.0),
                  r.get('dominance', 0.0)] for r in _cal_rows],
                dtype=torch.float32)
            _cal_lab = torch.tensor(
                [int(r.get('label4', r['label'])) for r in _cal_rows],
                dtype=torch.long)
            _cal_keep = _cal_vad.abs().sum(1) > 0
            _cal_prior = torch.bincount(
                _cal_lab[_cal_keep], minlength=n_cls).float()
            _cal_off = solve_calibrated_prior(
                _cal_vad[_cal_keep], _cal_lab[_cal_keep],
                torch.as_tensor(_pm, dtype=torch.float32),
                torch.as_tensor(_pi, dtype=torch.float32),
                _cal_prior,
                alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                fold_lambda=bool(
                    getattr(config, 'muted_mixup_fold_lambda', True)),
                seed=int(getattr(config, 'seed', 42)))
            config._proto_cal_prior = _cal_off.cpu().numpy()
            print(f"   Calibrated prior offset: "
                  f"{np.round(config._proto_cal_prior, 4).tolist()}")

        _cons = [float(train_dataset.data[i].get('annot_consensus', 0.0))
                 for i in train_indices
                 if float(train_dataset.data[i].get('annot_mask', 0.0)) > 0]
        config._consensus_min = float(min(_cons)) if _cons else 0.0
        print(f"   Annotator consensus on fitting rows: min "
              f"{config._consensus_min:.3f}, mean "
              f"{(sum(_cons) / max(1, len(_cons))):.3f}")
        print(f"   Class priors on the fitting rows: "
              f"{np.round(_pc / max(1.0, _pc.sum()), 4).tolist()}")
        _wscope = str(getattr(config, 'proto_whiten_scope', 'per_class'))
        if proto_metric != 'mahalanobis':
            config._proto_whiten = None
        elif _wscope == 'shared':
            config._proto_whiten = pooled_whitening(
                [train_dataset.data[i] for i in train_indices], _pm, n_cls,
                consensus_q=float(getattr(config, 'proto_centroid_consensus_q', 0.0)))
        elif str(getattr(config, 'proto_whiten_form', 'cholesky')) == 'diagonal':
            # Scale-matched control: unit variance per coordinate, but the
            # class ellipsoid keeps its tilt. Isolates decorrelation from
            # rescaling when read against the full whitening.
            config._proto_whiten = diagonal_whitening(
                [train_dataset.data[i] for i in train_indices], _pm, n_cls,
                dims=_dims)
        elif str(getattr(config, 'proto_whiten_form', 'cholesky')) == 'symmetric':
            # Cholesky factors are triangular, so the whitened components
            # depend on the order of the VAD axes. The symmetric root does
            # not, and gives the same Mahalanobis norm.
            config._proto_whiten = symmetric_whitening_matrices(_pi)
        else:
            config._proto_whiten = whitening_matrices(_pi)

        # Fixed random target table, for the 'randfixed' arm. One draw per
        # row of the underlying dataset, indexed by sample_index so a given
        # utterance receives the same target in every epoch. Seeded from the
        # run seed so the arm is reproducible.
        if proto_target == 'randfixed':
            _n_rows = len(train_dataset.data)
            _rng = np.random.RandomState(int(config.seed) + 9973)
            config._rand_table = _rng.randn(_n_rows, len(_dims))
            print(f"   Fixed random target table: {_n_rows} rows x "
                  f"{len(_dims)}, N(0,1), seeded {int(config.seed) + 9973}")

        # Training-split agreement statistics, for the 'resagree' target.
        if proto_target in ('resagree', 'vadagree'):
            _ag_tr = np.array(
                [train_dataset.data[i].get('annot_consensus', 0.0)
                 for i in train_indices], dtype=np.float64)
            _ag_m = _ag_tr[_ag_tr > 0]
            config._agree_mu = float(_ag_m.mean()) if _ag_m.size else 0.5
            config._agree_sd = float(_ag_m.std()) if _ag_m.size else 1.0
            config._agree_sd = max(config._agree_sd, 1e-6)
            print(f"   Agreement standardisation: mean {config._agree_mu:.4f}, "
                  f"sd {config._agree_sd:.4f}")

        # Marginal (label-free) whitening, for the 'whitevad' control arm.
        # Fitted on the same training rows as the class statistics so the two
        # arms differ only in whether the label is used.
        if proto_target in ('whitevad', 'vadagree'):
            _mm, _mw = marginal_whitening(
                [train_dataset.data[i] for i in train_indices], dims=_dims)
            config._marginal_mean = _mm
            config._marginal_whiten = _mw
            print(f"   Marginal whitening: mean {np.round(_mm, 4).tolist()}, "
                  f"label-free")

        # Pool of real residuals over the whole TRAINING split, for the
        # control variants that draw a target unrelated to the sample.
        #
        # The default control permutes targets within the batch, which has two
        # cosmetic defects: a random permutation has exactly one fixed point in
        # expectation regardless of size, so about 6 percent of samples at
        # batch 16 receive their own target, and batches are length-bucketed
        # rather than random. Measurement says the second does not matter here
        # (duration against residual magnitude correlates -0.031, and mean
        # duration varies by 3 percent across the four classes), but drawing
        # from the full pool removes both concerns and is simpler to state:
        # each control target is a real residual drawn independently of the
        # input.
        _pool = None
        if str(getattr(config, 'proto_shuffle_mode', 'batch')) != 'batch':
            _pool_vad = np.array(
                [[train_dataset.data[i].get('valence', 0.0),
                  train_dataset.data[i].get('arousal', 0.0),
                  train_dataset.data[i].get('dominance', 0.0)]
                 for i in train_indices], dtype=np.float64)[:, list(_dims)]
            _pool_lab = np.array(
                [train_dataset.data[i]['label'] for i in train_indices])
            _finite = np.isfinite(_pool_vad).all(axis=1)
            _pool_rows = residual_targets(
                _pool_vad[_finite], _pool_lab[_finite], _pm,
                whiten=config._proto_whiten)
            _pool = torch.tensor(_pool_rows, dtype=torch.float32)
            print(f"   Control pool: {_pool.shape[0]} training residuals, "
                  f"mode={getattr(config, 'proto_shuffle_mode')}")
        config._proto_pool = _pool

        if proto_target == 'offlabel':
            proto_out = n_cls
        elif proto_target == 'dispersion':
            proto_out = len(_dims)
        elif proto_target == 'agreement':
            proto_out = 1
        elif proto_target == 'presidual':
            proto_out = len(_dims)
        elif proto_target == 'resdisp':
            proto_out = len(_dims) * 2
        elif proto_target in ('residual', 'absresidual', 'invresidual',
                              'rawvad', 'whitevad'):
            proto_out = len(_dims)
        elif proto_target in ('resagree', 'vadagree'):
            proto_out = len(_dims) + 1
        elif proto_target == 'randfixed':
            proto_out = len(_dims)
        elif proto_target == 'allclass':
            proto_out = len(_dims) * n_cls
        elif proto_target in ('signed', 'signedw'):
            proto_out = 2
        elif proto_target == 'subtype':
            kpc = int(getattr(config, 'aux_vad_clusters_per_class', 2))
            centroids = build_per_class_vad_centroids(
                train_data=[train_dataset.data[i] for i in train_indices],
                clusters_per_class=kpc, num_classes=n_cls,
                seed=int(getattr(config, 'seed', 42)))
            config._proto_subtype_centroids = centroids
            proto_out = n_cls * kpc
        else:
            proto_out = 1

        # Optional SECOND auxiliary task sharing the same module.
        #
        # Every arm runs proto_predictor_hidden_dim 0, so the head is a bare
        # nn.Linear whose output rows share no parameters. One Linear(1024,
        # N+1) is therefore mathematically identical to two independent heads,
        # and needs no extra plumbing through the optimizer, the checkpoint or
        # the evaluation dumps. With a hidden layer the two tasks WOULD share
        # it, which is a different model, so refuse that case rather than
        # quietly change what the flag means.
        _use_expl = bool(getattr(config, 'use_expl_head', False))
        if _use_expl:
            if int(proto_hidden) > 0:
                raise ValueError(
                    "use_expl_head requires proto_predictor_hidden_dim 0. With "
                    "a hidden layer the regression and explainability tasks "
                    "would share it, which is not the intended two-head model.")
            proto_out += 1

        proto_predictor = PrototypicalityPredictor(
            input_dim=config.hidden_dim,
            hidden_dim=proto_hidden,
            output_dim=proto_out,
        ).to(device)
        if _use_expl:
            print(f"   Explainability head: +1 output, weight "
                  f"{getattr(config, 'expl_head_weight', 1.0)}, "
                  f"shuffle={bool(getattr(config, 'expl_head_shuffle', False))}")
        print(f"   Proto predictor: {config.hidden_dim} -> {proto_hidden} "
              f"-> {proto_out}  (target={proto_target}, metric={proto_metric})")

        # Per-modality auxiliary heads.
        #
        # The fused representation is what the classifier reads, so supervising
        # only it lets the fusion layer satisfy the auxiliary task without
        # either encoder having to encode prototypicality. Attaching the same
        # target to the audio and text embeddings separately forces both
        # streams to carry it.
        #
        # NOTE this changes the head count from one to three, so the arm is NOT
        # capacity matched to uf4_ctrl_pool and needs its own control before a
        # gain here can be attributed to the per-modality supervision rather
        # than to the extra parameters and gradient noise.
        if bool(getattr(config, 'proto_modality_heads', False)):
            _adim = getattr(model, 'audio_dim', None)
            _tdim = getattr(model, 'text_dim', None)
            if _adim and _tdim:
                proto_predictor_audio = PrototypicalityPredictor(
                    input_dim=int(_adim), hidden_dim=proto_hidden,
                    output_dim=proto_out).to(device)
                proto_predictor_text = PrototypicalityPredictor(
                    input_dim=int(_tdim), hidden_dim=proto_hidden,
                    output_dim=proto_out).to(device)
                print(f"   Per-modality proto heads: audio {_adim} -> "
                      f"{proto_out}, text {_tdim} -> {proto_out}")
            else:
                print("   proto_modality_heads set but modality dims are "
                      "unavailable; falling back to the fused head only")
        print(f"   Proto predictor weight: {getattr(config, 'proto_predictor_weight', 0.5)}")
        _cq = float(getattr(config, 'proto_centroid_consensus_q', 0.0))
        print(f"   Prototype fitting: consensus_q={_cq}"
              f"{' (all samples)' if _cq <= 0 else ' (high-agreement only)'}")
        print(f"   Fitted class centres: "
              f"{np.round(_pm, 4).tolist()}")

    # Optimizer - differential LR for unfrozen BERT and Wav2Vec2 layers
    unfreeze_bert = getattr(config, 'unfreeze_bert_layers', 0)
    bert_lr = getattr(config, 'bert_learning_rate', config.learning_rate * 0.1)
    audio_encoder_type = getattr(config, 'audio_encoder_type', 'preextracted')
    unfreeze_audio = getattr(config, 'unfreeze_audio_layers', 0)
    audio_lr = getattr(config, 'audio_learning_rate', config.learning_rate * 0.1)

    # Collect special param IDs (BERT + Wav2Vec2) for differential LR
    special_param_ids = set()
    param_groups = []

    # BERT differential LR
    if unfreeze_bert > 0 and hasattr(model, 'text_encoder') and model.text_encoder is not None:
        bert_params = model.text_encoder.get_bert_params()
        special_param_ids.update(id(p) for p in bert_params)
        param_groups.append({'params': bert_params, 'lr': bert_lr})
        print(f"   BERT LR: {bert_lr:.2e}")

    # Wav2Vec2 / Emotion2Vec differential LR
    if audio_encoder_type in RAW_AUDIO_ENCODERS and unfreeze_audio > 0 and hasattr(model, 'audio_encoder') and model.audio_encoder is not None:
        audio_params = model.audio_encoder.get_audio_params()
        special_param_ids.update(id(p) for p in audio_params)
        param_groups.append({'params': audio_params, 'lr': audio_lr})
        print(f"   {audio_encoder_type} LR: {audio_lr:.2e}")

    # The salience gate is two scalars that start at zero and must reach
    # O(0.1) to shift a logit at all. At the main LR of 5e-6 a scalar moves
    # about 1e-4 over 20 epochs, so it cannot turn on within the budget no
    # matter how useful the signal is, and a learned weight near zero would
    # say nothing about the signal. Give it its own much larger LR so that
    # the final weight is a real readout.
    gate_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith('salience_gate_')]
    if gate_params:
        gate_lr = float(getattr(config, 'salience_gate_lr', 1e-2))
        special_param_ids.update(id(p) for p in gate_params)
        param_groups.append({'params': gate_params, 'lr': gate_lr})
        print(f"   Salience gate LR: {gate_lr:.2e} ({len(gate_params)} params)")

    # All other trainable model params (exclude frozen encoder params)
    other_model_params = [p for p in model.parameters() if p.requires_grad and id(p) not in special_param_ids]
    param_groups.append({'params': other_model_params, 'lr': config.learning_rate})
    print(f"   Main LR: {config.learning_rate:.2e}")

    # Add contrastive criterion + domain discriminator + proto predictor params
    if contrastive_criterion is not None:
        param_groups.append({'params': list(contrastive_criterion.parameters()), 'lr': config.learning_rate})
    if domain_discriminator is not None:
        param_groups.append({'params': list(domain_discriminator.parameters()), 'lr': config.learning_rate})
    if modality_discriminator is not None:
        param_groups.append({'params': list(modality_discriminator.parameters()), 'lr': config.learning_rate})
    if proto_predictor is not None:
        # The auxiliary head is randomly initialised, unlike the pretrained
        # trunk it sits on, so the trunk's fine-tuning rate is not obviously
        # the right rate for it. Measured: at the shared 5e-6 a 1024-256-3
        # head reaches train MSE 0.543 while a 1024-3 head reaches 0.527,
        # i.e. the higher-capacity head fits WORSE, which is an optimisation
        # failure rather than a capacity one. Defaults to the shared rate so
        # every existing arm and fingerprint is unchanged.
        _pp_lr = float(getattr(config, 'proto_predictor_lr',
                               config.learning_rate))
        param_groups.append({'params': list(proto_predictor.parameters()),
                             'lr': _pp_lr})
        if _pp_lr != config.learning_rate:
            print(f"   Proto predictor LR: {_pp_lr:.2e} "
                  f"(trunk {config.learning_rate:.2e})")
    for _extra in (proto_predictor_audio, proto_predictor_text):
        if _extra is not None:
            param_groups.append({'params': list(_extra.parameters()),
                                 'lr': config.learning_rate})
    if centroid_tracker is not None and centroid_tracker.mode == "grad":
        centroid_lr = getattr(config, 'learned_centroid_lr', 1e-3)
        param_groups.append({'params': list(centroid_tracker.parameters()), 'lr': centroid_lr})
        print(f"   Centroid LR (grad mode): {centroid_lr:.2e}")

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler (CosineAnnealing like old repo)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    print(f"   Scheduler: CosineAnnealingLR (T_max={config.num_epochs})")

    # Mixed precision (fp16): speeds up forward pass and saves VRAM.
    # GradScaler only needed when unfreezing (backward through encoder).
    use_amp = device.type == 'cuda'
    unfreeze_audio_for_amp = getattr(config, 'unfreeze_audio_layers', 0) > 0
    scaler = GradScaler('cuda') if (use_amp and unfreeze_audio_for_amp) else None
    if use_amp:
        print(f"   Mixed precision (fp16): enabled (scaler={'on' if scaler else 'off'})")

    # Try to load checkpoint (for resume)
    resume_state = load_checkpoint(
        config, model, optimizer, scheduler, scaler,
        contrastive_criterion, domain_discriminator,
        modality_discriminator, proto_predictor, device,
    )

    # Initialize WandB - resume existing run if we have a run ID
    wandb_init_kwargs = {
        'project': config.wandb_project,
        'name': config.experiment_name,
        'config': config.to_dict(),
    }
    if resume_state is not None and resume_state.get('wandb_run_id'):
        wandb_init_kwargs['id'] = resume_state['wandb_run_id']
        wandb_init_kwargs['resume'] = 'allow'
    wandb.init(**wandb_init_kwargs)
    # Per-step x-axis for training loss
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/step_loss", step_metric="train/global_step")
    # Per-epoch x-axis for everything else
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss",         step_metric="epoch")
    wandb.define_metric("train/primary_loss", step_metric="epoch")
    wandb.define_metric("train/uar",          step_metric="epoch")
    wandb.define_metric("train/accuracy",     step_metric="epoch")
    wandb.define_metric("train/mae",          step_metric="epoch")
    wandb.define_metric("train/ccc",          step_metric="epoch")
    wandb.define_metric("val/*",              step_metric="epoch")
    wandb.define_metric("test/*",             step_metric="epoch")
    wandb.define_metric("stage",              step_metric="epoch")

    # Early stopping setup
    early_stopping_patience = int(getattr(config, 'early_stopping_patience', 10))
    epochs_without_improvement = 0
    use_early_stopping = bool(getattr(config, 'use_early_stopping', True))

    # Which weights get evaluated on the test corpora at the end.
    model_selection = str(getattr(config, 'model_selection', 'best_val'))
    swa_last_n = int(getattr(config, 'swa_last_n', 5))
    swa_window = [] if model_selection == 'swa_last_n' else None
    if not use_early_stopping:
        print(f"   Early stopping DISABLED: fixed budget of "
              f"{config.num_epochs} epochs")
    if model_selection != 'best_val':
        detail = (f" (last {swa_last_n} epochs)"
                  if model_selection == 'swa_last_n' else "")
        print(f"   Model selection: {model_selection}{detail}")
    if swa_window is not None:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        win_mb = n_trainable * 4 * swa_last_n / 1e6
        print(f"   SWA window: {n_trainable/1e6:.1f}M trainable of "
              f"{n_total/1e6:.1f}M total params, ~{win_mb:.0f} MB held on CPU")

    # Best model tracking (across both stages)
    best_val_metric = 0.0
    best_model_state = None
    best_contrastive_state = None

    # Track current wandb run id for checkpoint metadata
    wandb_run_id = wandb.run.id if wandb.run is not None else None

    # Apply resume state if we loaded a checkpoint
    resume_start_epoch = 0
    resume_global_step = 0
    if resume_state is not None:
        resume_start_epoch = resume_state['start_epoch']
        resume_global_step = resume_state['global_step']
        best_val_metric = resume_state['best_val_metric']
        best_model_state = resume_state['best_model_state']
        best_contrastive_state = resume_state.get('best_contrastive_state')
        epochs_without_improvement = resume_state['epochs_without_improvement']
        # Restore the trailing weight window used by model_selection
        # 'swa_last_n'. A resumed run skips every completed epoch, so without
        # this the window stays empty, the averaging branch does not fire, and
        # selection falls back to best_val without saying so, producing a model
        # chosen by a different rule than its sibling seeds.
        if swa_window is not None and resume_state.get('swa_window'):
            swa_window = list(resume_state['swa_window'])
            print(f"   Restored SWA window: {len(swa_window)} epoch(s) of "
                  f"trailing weights")

    # Determine training stages
    if use_two_stage:
        stages = [
            {
                'name': 'Stage 1 (prototypical)',
                'indices': proto_train_indices,
                'epochs': config.stage1_epochs,
                'lr_factor': 1.0,
            },
            {
                'name': 'Stage 2 (full data)',
                'indices': train_indices,
                'epochs': config.num_epochs - config.stage1_epochs,
                'lr_factor': config.stage2_lr_factor,
            },
        ]
    else:
        stages = [
            {
                'name': 'Training',
                'indices': train_indices,
                'epochs': config.num_epochs,
                'lr_factor': 1.0,
            },
        ]

    # Starts at 0, not at resume_start_epoch. The per-stage loop below
    # replays every epoch index and advances global_epoch itself when it
    # skips an already-completed one, so seeding it with the resume point
    # would count those epochs twice and run num_epochs extra epochs after
    # a resume (a run resumed at 2/2 went on to train epochs 3 and 4).
    global_epoch = 0
    global_step = resume_global_step

    for stage in stages:
        stage_name = stage['name']
        stage_epochs = stage['epochs']
        stage_indices = stage['indices']
        lr_factor = stage['lr_factor']

        if stage_epochs <= 0:
            continue

        print(f"\n  {stage_name}: {stage_epochs} epochs, {len(stage_indices)} samples")

        # Create stage-specific dataloader
        stage_subset = Subset(train_dataset, stage_indices)
        stage_loader = DataLoader(
            stage_subset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=vad_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )

        # Apply LR factor for stage 2 (skip on resume - already in optimizer state)
        if lr_factor != 1.0 and resume_state is None:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * lr_factor
            print(f"   LR reduced by {lr_factor}x")

        # Reset early stopping between stages
        if use_two_stage:
            epochs_without_improvement = 0

        for epoch_in_stage in range(stage_epochs):
            # Skip epochs already completed in a previous run
            if (global_epoch + 1) <= resume_start_epoch:
                global_epoch += 1
                continue

            global_epoch += 1
            print(f"\n  Epoch {global_epoch}/{config.num_epochs} ({stage_name})")

            # Reshuffle the length buckets so batch composition varies across
            # epochs rather than repeating the same groupings every time.
            if length_sampler is not None:
                length_sampler.set_epoch(global_epoch)

            # Curriculum learning: rebuild the training loader each epoch from
            # a paced subset of stage_indices. global_epoch is 1-based here, so
            # the create_curriculum_subset epoch arg is (global_epoch - 1).
            curriculum_fraction = 1.0
            if use_curriculum:
                cur_epoch_zero = global_epoch - 1
                cur_indices = create_curriculum_subset(
                    train_indices=stage_indices,
                    train_labels=curriculum_train_labels,
                    train_difficulties=curriculum_train_difficulties,
                    epoch=cur_epoch_zero,
                    total_curriculum_epochs=config.curriculum_epochs,
                    pacing_function=curriculum_pacing_fn,
                    curriculum_type=config.curriculum_type,
                )
                if cur_epoch_zero < config.curriculum_epochs:
                    curriculum_fraction = curriculum_pacing_fn(
                        cur_epoch_zero, config.curriculum_epochs
                    )
                cur_subset = Subset(train_dataset, cur_indices)
                stage_loader = DataLoader(
                    cur_subset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=vad_collate_fn,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent,
                )
                print(f"   Curriculum: {len(cur_indices)}/{len(stage_indices)} "
                      f"samples (fraction={curriculum_fraction:.3f})")

                # Bump dropout on the first epoch after curriculum completes.
                if global_epoch == config.curriculum_epochs + 1:
                    post_dropout = float(getattr(config, 'post_curriculum_dropout', 0.6))
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            m.p = post_dropout
                    print(f"   Curriculum complete: dropout bumped to {post_dropout}")

            # Schedule GRL lambda (sigmoid ramp, scaled by peak lambda)
            grl_lambda = 0.0
            if domain_discriminator is not None or modality_discriminator is not None:
                peak_lambda = getattr(config, 'adversarial_peak_lambda', 1.0)
                warmup_frac = getattr(config, 'adversarial_warmup_frac', 0.5)
                warmup_frac = max(1e-6, warmup_frac)
                p = min(1.0, global_epoch / (config.num_epochs * warmup_frac))
                grl_lambda = peak_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
                if domain_discriminator is not None:
                    domain_discriminator.set_lambda(grl_lambda)
                if modality_discriminator is not None:
                    modality_discriminator.set_lambda(grl_lambda)

            # Train
            train_metrics = train_epoch(
                model, stage_loader, criterion, optimizer, device, config,
                contrastive_criterion, domain_discriminator, domain_adv_loss,
                proto_predictor, scaler, global_step, current_epoch=global_epoch,
                proto_predictor_audio=proto_predictor_audio,
                proto_predictor_text=proto_predictor_text,
                modality_discriminator=modality_discriminator,
                centroid_tracker=centroid_tracker,
                proto_atyp_ctx=proto_atyp_ctx,
                aux_vad_centroids=aux_vad_centroids,
                logit_adjust=logit_adjust,
                confidence_tracker=confidence_tracker,
                intensity_ctx=intensity_ctx,
                ambiguity_weights=ambiguity_weights,
                evidence_pos_weight=evidence_pos_weight,
                hierarchical_emotion_weights=hierarchical_emotion_weights,
                salience_stats=salience_stats,
                genlabel_density=genlabel_density,
                ot_cost=_ot_cost,
            )
            global_step = train_metrics['global_step']

            if genlabel_density is not None:
                # Refit on the epoch that just ended. A failed refit leaves
                # the previous densities in place, or keeps the arm on the
                # vanilla target if none exist yet.
                ok = genlabel_density.refit()
                print(f"   GenLabel refit: {'ok' if ok else 'SKIPPED'}"
                      f" (ready={genlabel_density.ready})")

            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   LR: {current_lr:.2e}")

            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device, config, use_amp)

            # Log to WandB
            log_dict = {
                'epoch': global_epoch,
                'train/loss': train_metrics['loss'],
                'train/primary_loss': train_metrics['primary_loss'],
                'val/loss': val_metrics['loss'],
            }

            if use_two_stage:
                log_dict['stage'] = 1 if lr_factor == 1.0 else 2

            if use_curriculum:
                log_dict['train/curriculum_fraction'] = curriculum_fraction

            if config.use_contrastive:
                log_dict['train/contrastive_loss'] = train_metrics['contrastive_loss']
                log_dict['train/contrastive_weight'] = train_metrics['effective_contrastive_weight']

            if domain_discriminator is not None:
                log_dict['train/adversarial_loss'] = train_metrics['adversarial_loss']
                log_dict['train/grl_lambda'] = grl_lambda

            if modality_discriminator is not None:
                log_dict['train/modality_adversarial_loss'] = train_metrics['modality_adversarial_loss']
                log_dict['train/grl_lambda'] = grl_lambda

            if proto_predictor is not None:
                log_dict['train/proto_pred_loss'] = train_metrics['proto_pred_loss']

            if config.task_type == "regression":
                log_dict.update({
                    'train/mae': train_metrics['overall_mae'],
                    'train/ccc': train_metrics['overall_ccc'],
                    'val/mae': val_metrics['overall_mae'],
                    'val/ccc': val_metrics['overall_ccc'],
                })
                current_metric = val_metrics['overall_ccc']
                metric_name = "CCC"
            else:
                log_dict.update({
                    'train/accuracy': train_metrics['accuracy'],
                    'train/uar': train_metrics['uar'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/uar': val_metrics['uar'],
                })
                current_metric = val_metrics['uar']
                metric_name = "UAR"

            wandb.log(log_dict)

            # Print progress
            if config.task_type == "regression":
                print(f"   Train: MAE={train_metrics['overall_mae']:.4f}, CCC={train_metrics['overall_ccc']:.4f}")
                print(f"   Val:   MAE={val_metrics['overall_mae']:.4f}, CCC={val_metrics['overall_ccc']:.4f}")
            else:
                print(f"   Train: Acc={train_metrics['accuracy']:.4f}, UAR={train_metrics['uar']:.4f}")
                print(f"   Val:   Acc={val_metrics['accuracy']:.4f}, UAR={val_metrics['uar']:.4f}")

            # Save best model + early stopping
            early_stop_triggered = False
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_model_state = model.state_dict().copy()
                if contrastive_criterion is not None and hasattr(contrastive_criterion, 'state_dict'):
                    best_contrastive_state = contrastive_criterion.state_dict().copy()
                epochs_without_improvement = 0
                print(f"   New best {metric_name}: {current_metric:.4f}")
            else:
                # Disable early stopping (and the patience counter) while the
                # curriculum is active: val on a partial training set is not
                # comparable to val on the full set.
                curriculum_active = use_curriculum and global_epoch <= config.curriculum_epochs
                if not curriculum_active and use_early_stopping:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"   Early stopping: no improvement for {early_stopping_patience} epochs")
                        early_stop_triggered = True

            # Freeze the confidence-derived split once enough epochs have
            # accumulated. From here on the sub-labels stop moving and the
            # widened head trains against a fixed target, which makes this a
            # stop-gradient teacher signal rather than self-prediction.
            # Freeze only when the proto/atyp split consumes the tracker.
            # Ambiguity weighting wants the live EMA, never a frozen one.
            if (confidence_tracker is not None
                    and use_proto_atyp
                    and getattr(config, 'proto_atyp_center_source', '')
                    == 'confidence'
                    and not confidence_tracker.frozen
                    and global_epoch >= int(getattr(
                        config, 'confidence_freeze_epoch', 5))):
                thr = confidence_tracker.freeze(
                    getattr(config, 'proto_atyp_split_criterion',
                            'per_class_median'),
                )
                print(f"   Confidence split frozen at epoch {global_epoch}: "
                      f"coverage={confidence_tracker.coverage() * 100:.1f}% "
                      f"of training samples")
                print(f"     thresholds per class: "
                      f"{[round(float(t), 4) for t in thr]}")
                if wandb.run is not None:
                    wandb.log({'proto_atyp/confidence_frozen_epoch': global_epoch})

            # Optional per-epoch evaluation on the test corpora. OFF BY
            # DEFAULT and it should stay off for production runs: it costs an
            # extra pass over every test set each epoch, measured at about 23
            # percent wall time (sv_base took 3h05m against the usual 2h30m at
            # 2/2), and used carelessly it is test-set peeking.
            #
            # It exists to answer one question: does the validation score
            # track cross-corpus transfer AS TRAINING PROCEEDS. That is what
            # early stopping needs and it cannot be answered from end-of-run
            # numbers, which give one point per run. The trace is written to
            # the log for offline correlation and is never read by training,
            # model selection, or the reported results.
            if bool(getattr(config, 'trace_test_each_epoch', False)):
                _trace = {}
                for _tl, _td in zip(test_loaders, test_datasets):
                    _tm = evaluate(model, _tl, criterion, device, config, use_amp)
                    _trace[_td.dataset_name] = (
                        float(_tm.get('uar', float('nan'))),
                        float(_tm.get('neutral_auc', float('nan'))),
                    )
                _vu = float(val_metrics.get('uar', float('nan')))
                print("   EPOCHTRACE epoch=%d val_uar=%.4f " % (global_epoch, _vu)
                      + " ".join(f"{k}_uar={v[0]:.4f} {k}_nauc={v[1]:.4f}"
                                 for k, v in sorted(_trace.items())))
                model.train()

            # Running tail of weights for the SWA selection mode. Only
            # trainable parameters are kept: the audio and text encoders are
            # frozen, so their weights are identical every epoch and holding
            # copies of them costs gigabytes of RAM for nothing.
            if swa_window is not None:
                swa_window.append({
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters() if p.requires_grad
                })
                if len(swa_window) > swa_last_n:
                    swa_window.pop(0)

            # Checkpoint after each epoch (atomic write)
            save_checkpoint(
                config, global_epoch, global_step,
                model, optimizer, scheduler, scaler,
                contrastive_criterion, domain_discriminator,
                modality_discriminator, proto_predictor,
                best_val_metric, best_model_state, best_contrastive_state,
                epochs_without_improvement, wandb_run_id,
                swa_window=swa_window,
            )

            if early_stop_triggered:
                break

    # Pick the weights to evaluate. "best_val" is the historical behavior;
    # the other modes exist because val UAR on the training corpus does not
    # track cross-corpus test UAR, so picking the val-best checkpoint adds
    # selection noise to every arm comparison.
    print(f"\n  Evaluating on test datasets...")
    if model_selection == 'final':
        print("   Using final-epoch weights (model_selection=final)")
    elif model_selection == 'swa_last_n' and swa_window:
        n_avg = len(swa_window)
        print(f"   Averaging weights over the last {n_avg} epochs "
              f"(model_selection=swa_last_n)")
        # Average in place, one parameter at a time, so peak memory stays at
        # one extra copy of a single tensor rather than a whole state dict.
        averaged = {}
        for key, ref in swa_window[-1].items():
            acc = swa_window[0][key].float().clone()
            for sd in swa_window[1:]:
                acc += sd[key].float()
            averaged[key] = (acc / n_avg).to(ref.dtype)
        # strict=False: the window holds trainable parameters only, so the
        # frozen encoder weights already in the model are left untouched.
        model.load_state_dict(averaged, strict=False)
        del averaged
        swa_window.clear()
    else:
        model.load_state_dict(best_model_state)
    # Refresh the saved state so the checkpoint matches what was evaluated.
    if model_selection != 'best_val':
        best_model_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    # Calibrate the evidence threshold on validation before testing. The
    # loss decides how well each head separates its class; this decides
    # where to cut. Fitted only on val, never on the test corpora.
    if bool(getattr(config, 'use_evidence_heads', False)):
        cal_thr, cal_uar = _calibrate_threshold_on_val(
            model, val_loader, config, device, use_amp,
        )
        print(f"   Evidence threshold calibrated on val: "
              f"{cal_thr:.3f} (val UAR {cal_uar:.4f}, was "
              f"{getattr(config, 'evidence_threshold', 0.5)})")
        config.evidence_threshold = cal_thr
        if wandb.run is not None:
            wandb.log({'evidence/calibrated_threshold': cal_thr,
                       'evidence/calibrated_val_uar': cal_uar})

    # Collect results for multi-seed averaging
    results = {
        'validation': val_metrics,
        'test_results': [],
    }

    for test_loader, test_dataset in zip(test_loaders, test_datasets):
        test_metrics = evaluate(model, test_loader, criterion, device, config, use_amp)
        dataset_name = test_dataset.dataset_name

        results['test_results'].append({
            'dataset': dataset_name,
            'results': test_metrics,
        })

        train_name = config.train_dataset if isinstance(config.train_dataset, str) else "+".join(config.train_dataset)
        print(f"\n   {train_name} -> {dataset_name}:")
        if config.task_type == "regression":
            print(f"      MAE: {test_metrics['overall_mae']:.4f}")
            print(f"      CCC: {test_metrics['overall_ccc']:.4f}")
            wandb.log({
                f'test/{dataset_name}_mae': test_metrics['overall_mae'],
                f'test/{dataset_name}_ccc': test_metrics['overall_ccc'],
            })
        else:
            # Within-class collapse metrics on the raw fused embeddings.
            # These test the mechanism hypothesis: aux VAD arms should keep
            # higher within-class variance / effective rank than baseline
            # and permuted arms, and that retention should track UAR.
            test_emb, test_emb_labels = _collect_embeddings(
                model, test_loader, config, device, use_amp,
            )
            emb_metrics = collapse_metrics(
                test_emb, test_emb_labels, int(config.num_classes),
            )
            # Threshold-free split of UAR: how well neutral is ranked apart
            # from emotional, versus how well the emotions are separated from
            # each other. A mechanism that only moves the operating point
            # changes UAR but leaves both of these untouched.
            tf_metrics = threshold_free_metrics(
                np.asarray(test_metrics.get('logits',
                                            np.zeros((0, config.num_classes)))),
                np.asarray(test_metrics['labels']),
                int(config.num_classes),
            )
            # Merge so multi-seed averaging in runner.py picks them up.
            test_metrics.update(emb_metrics)
            test_metrics.update(tf_metrics)
            print(f"      Acc: {test_metrics['accuracy']:.4f}")
            print(f"      UAR: {test_metrics['uar']:.4f}")
            print(f"      neutral_auc: {tf_metrics['neutral_auc']:.4f}, "
                  f"emo_auc: {tf_metrics['emo_auc']:.4f}")
            print(f"      within_var_ratio: {emb_metrics['within_var_ratio']:.4f}, "
                  f"embed_rank: {emb_metrics['effective_rank']:.1f}")
            wandb.log({
                f'test/{dataset_name}_acc': test_metrics['accuracy'],
                f'test/{dataset_name}_uar': test_metrics['uar'],
                f'test/{dataset_name}_neutral_auc': tf_metrics['neutral_auc'],
                f'test/{dataset_name}_emo_auc': tf_metrics['emo_auc'],
                f'test/{dataset_name}_within_var_ratio': emb_metrics['within_var_ratio'],
                f'test/{dataset_name}_embed_rank': emb_metrics['effective_rank'],
                f'cm/test_{dataset_name}': wandb.plot.confusion_matrix(
                    preds=test_metrics['predictions'],
                    y_true=test_metrics['labels'],
                    class_names=_cm_class_names(config),
                    title=f'{dataset_name} (best model)',
                ),
            })

    # Per-sample logits for offline ensembling. The held-out split is written
    # alongside the test corpora because the stacked combiner has to be fitted
    # on data the base models never trained on, and that split is chosen with
    # the run seed so it cannot be reconstructed reliably from outside.
    if bool(getattr(config, 'save_predictions', False)):
        print("\n   Dumping per-sample predictions...")
        _dump_split_predictions(model, val_loader, config, device,
                                'heldout', use_amp,
                                proto_predictor=proto_predictor)
        for test_loader, test_dataset in zip(test_loaders, test_datasets):
            _dump_split_predictions(model, test_loader, config, device,
                                    test_dataset.dataset_name, use_amp,
                                    proto_predictor=proto_predictor)

    # Val confusion matrix (best model)
    if config.task_type == "classification":
        val_best = evaluate(model, val_loader, criterion, device, config, use_amp)
        wandb.log({
            'cm/val': wandb.plot.confusion_matrix(
                preds=val_best['predictions'],
                y_true=val_best['labels'],
                class_names=_cm_class_names(config),
                title='Validation (best model)',
            ),
        })

    # Save model
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{config.experiment_name}_seed{config.seed}.pt"

    save_dict = {
        'model_state_dict': best_model_state,
        'config': config.to_dict(),
        'best_val_metric': best_val_metric,
    }
    if best_contrastive_state is not None:
        save_dict['contrastive_state_dict'] = best_contrastive_state

    torch.save(save_dict, save_path)

    print(f"\n  Model saved to: {save_path}")

    # Aux VAD cluster visualization: 4-panel figure showing the fixed VAD
    # centroids against the training points and a cluster-vs-class heatmap.
    # Saved next to the checkpoint for this seed.
    if use_aux_vad and aux_vad_centroids is not None:
        try:
            ckpt_dir, _, _, _ = get_checkpoint_paths(config)
            viz_path = ckpt_dir / "aux_vad_clusters.png"
            per_class_scope = (
                getattr(config, 'aux_vad_cluster_scope', 'global') == 'per_class'
            )
            if per_class_scope:
                k_desc = (f"{int(getattr(config, 'aux_vad_clusters_per_class', 2))}"
                          f" per class")
            else:
                k_desc = f"k={int(getattr(config, 'aux_vad_cluster_k', 8))}"
            save_cluster_visualization(
                train_data=[train_dataset.data[i] for i in train_indices],
                centroids=aux_vad_centroids.cpu().numpy(),
                num_classes=config.num_classes,
                out_path=viz_path,
                title=f"{config.experiment_name} seed{config.seed} "
                      f"({k_desc}, "
                      f"init={getattr(config, 'aux_vad_cluster_init', 'random')})",
                expected_vad=getattr(config, 'expected_vad', None),
                per_class_scope=per_class_scope,
            )
            wandb.log({"aux_vad/cluster_viz": wandb.Image(str(viz_path))})
        except Exception as viz_e:
            print(f"  aux_vad_viz failed: {viz_e}")

    # Mark run as finished - writes results.json and status=done, removes latest.pt
    mark_run_done(config, results)

    wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config)
