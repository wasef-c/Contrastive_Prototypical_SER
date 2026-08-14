#!/usr/bin/env python3
"""
Clean training script for emotion recognition with contrastive learning
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hashlib
import math
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

from utils.config import Config
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
from utils.metrics import calculate_classification_metrics, calculate_vad_metrics
from utils.prototypicality import (
    DATASETS_WITH_VAD,
    LearnableCentroids,
    batch_calculate_difficulty,
    batch_difficulty_tensor,
    calculate_difficulty,
)
from utils.multiview_prototypicality import compute_multiview_difficulty, compute_crossmodal_agreement
from utils.muted_mixup import (muted_mixup_loss, muted_mixup_step,
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
    batch_scrambled_ids_and_mask,
    batch_vad_targets_and_mask,
    build_per_class_vad_centroids,
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
    uses_raw_audio = audio_encoder_type in ("wav2vec2", "emotion2vec") and not has_features

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


def train_epoch(model, dataloader, criterion, optimizer, device, config,
                contrastive_criterion=None, domain_discriminator=None, domain_adv_loss=None,
                proto_predictor=None, scaler=None, global_step=0, current_epoch=0,
                modality_discriminator=None, centroid_tracker=None,
                proto_atyp_ctx=None, aux_vad_centroids=None, logit_adjust=None,
                confidence_tracker=None, intensity_ctx=None,
                ambiguity_weights=None, evidence_pos_weight=None,
                hierarchical_emotion_weights=None, salience_stats=None):
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

    # Contrastive weight warm-up: linear ramp from 0 → target over warmup epochs
    warmup_epochs = getattr(config, 'contrastive_warmup_epochs', 5)
    if warmup_epochs > 0 and current_epoch < warmup_epochs:
        warmup_factor = current_epoch / warmup_epochs
    else:
        warmup_factor = 1.0
    effective_contrastive_weight = config.contrastive_weight * warmup_factor

    optimizer.zero_grad()
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
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
                          or _ce_needs_modal)
        raw_embeddings = None
        embeddings_norm = None
        modal_features = None

        with autocast('cuda', enabled=use_amp):
            if use_embeddings:
                # When modality adversarial is active, keep gradients flowing into
                # the audio/text encoders via modal_features (no detach)
                if modality_discriminator is not None:
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
                            if ce_src == 'vad':
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
                difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                pred_proto = proto_predictor(raw_embeddings)
                loss_proto_pred = F.mse_loss(pred_proto, difficulties)

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
            if (getattr(config, 'use_muted_mixup', False)
                    and raw_embeddings is not None
                    and config.task_type == "classification"):
                if getattr(config, 'muted_mixup_symmetric', False):
                    # Blend both ways so equal numbers of synthetic samples
                    # land on each side of the neutral boundary and the
                    # prior is left where it was.
                    mm_logits, mm_targets = symmetric_mixup_step(
                        raw_embeddings, labels, model.output_layer,
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        within_class_control=bool(
                            getattr(config, 'muted_mixup_control', False)),
                    )
                else:
                    mm_logits, mm_targets, _mm_lam = muted_mixup_step(
                        raw_embeddings, labels, model.output_layer,
                        alpha=float(getattr(config, 'muted_mixup_alpha', 2.0)),
                        shuffle_control=bool(getattr(config, 'muted_mixup_control', False)),
                    )
                if mm_logits is not None:
                    cls_w_mm = criterion.weight if hasattr(criterion, 'weight') else None
                    loss_muted = muted_mixup_loss(mm_logits, mm_targets, cls_w_mm)
                    muted_synth_count += int(mm_logits.size(0))

            # Auxiliary VAD-cluster loss. Aux head lives on the model; we
            # already have raw_embeddings (use_embeddings was forced True when
            # this flag is on). Non-VAD samples are masked out.
            loss_aux_vad = torch.tensor(0.0, device=device)
            aux_vad_task = getattr(config, 'aux_vad_task', 'cluster')
            use_aux_vad_flag = (getattr(config, 'use_aux_vad_cluster', False)
                                and raw_embeddings is not None
                                and getattr(model, 'use_aux_vad_cluster', False)
                                and (aux_vad_task == 'regression'
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
                elif getattr(config, 'aux_vad_cluster_scope', 'global') == 'per_class':
                    # Subtype-within-class target: happy-1, happy-2, angry-1...
                    cluster_ids, aux_mask = batch_per_class_cluster_ids_and_mask(
                        batch, aux_vad_centroids, device,
                    )
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
                # Average over samples with real VAD only.
                denom = aux_mask.sum().clamp_min(1.0)
                loss_aux_vad = (per_sample_aux * aux_mask).sum() / denom

            # Combined loss
            adv_weight = getattr(config, 'adversarial_weight', 0.0)
            modality_adv_weight = getattr(config, 'modality_adv_weight', 0.0)
            proto_pred_weight = getattr(config, 'proto_predictor_weight', 0.0)
            cross_modal_weight = getattr(config, 'cross_modal_weight', 0.1)
            vadmix_weight = getattr(config, 'vadmix_weight', 1.0) if use_vadmix else 0.0
            aux_vad_weight = (float(getattr(config, 'aux_vad_cluster_weight', 0.2))
                              if use_aux_vad_flag else 0.0)
            loss = (loss_primary
                    + effective_contrastive_weight * loss_contrastive
                    + adv_weight * loss_adversarial
                    + modality_adv_weight * loss_modality_adversarial
                    + proto_pred_weight * loss_proto_pred
                    + cross_modal_weight * loss_cross_modal
                    + vadmix_weight * loss_vadmix
                    + aux_vad_weight * loss_aux_vad
                    + float(getattr(config, 'muted_mixup_weight', 0.5)) * loss_muted)

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
        if use_amp:
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
        print(f"   muted mixup: {muted_synth_count} synthetic samples this epoch")
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
    return names[:int(getattr(config, 'num_classes', 4))]


def _dump_split_predictions(model, dataloader, config, device, split_name,
                            use_amp=False):
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

    Returns:
        Path to the written .npz file, or None when the split is empty.
    """
    model.eval()
    all_logits, all_labels, all_label4, all_speakers = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            model_inputs = _prepare_model_inputs(batch, config, model, device)
            with autocast('cuda', enabled=use_amp):
                result = model(**model_inputs)
            logits = result[0] if isinstance(result, tuple) else result
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
}


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
                    epochs_without_improvement, wandb_run_id):
    """Atomic checkpoint save: write to .tmp then rename."""
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
    }


def mark_run_done(config, results):
    """Write results.json and update status to done. Clean up latest.pt."""
    ckpt_dir, latest_path, status_path, results_path = get_checkpoint_paths(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Serialize results — strip tensors/non-JSON by converting to plain python
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

    # Keep latest.pt around in case user wants to resume from last state —
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
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        split_rng.shuffle(indices)
        val_size = max(1, int(len(indices) * config.val_split))
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])

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
    use_aux_vad = bool(getattr(config, 'use_aux_vad_cluster', False))
    aux_vad_centroids = None
    if use_aux_vad:
        aux_task = getattr(config, 'aux_vad_task', 'cluster')
        if aux_task == 'regression':
            # Regression arm predicts raw VAD; no centroids to fit.
            print(f"  Aux VAD regression multitask enabled: "
                  f"weight={getattr(config, 'aux_vad_cluster_weight', 0.2)}")
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
    print(f"   Modality: {config.modality}")
    print(f"   Task: {config.task_type}")
    print(f"   Audio encoder: {getattr(config, 'audio_encoder_type', 'preextracted')}")
    print(f"   Contrastive: {config.use_contrastive}")

    # Cache frozen encoder features: compute once, reuse every epoch (~100x faster)
    audio_encoder_type = getattr(config, 'audio_encoder_type', 'preextracted')
    if (audio_encoder_type in ('wav2vec2', 'emotion2vec')
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

        # Use reduction='none' if we need per-sample weighting or label smoothing
        use_proto_weight = getattr(config, 'use_prototypical_weighting', False)
        use_label_smooth = getattr(config, 'use_prototypical_label_smoothing', False)
        use_salience_weight = getattr(config, 'use_salience_weighting', False)
        if use_proto_weight or use_label_smooth or use_salience_weight:
            criterion = nn.CrossEntropyLoss(weight=freq_weights_tensor, reduction='none')
            print(f"   Prototypical weighting: {use_proto_weight} (alpha={getattr(config, 'prototypical_weighting_alpha', 2.0)})")
            print(f"   Label smoothing: {use_label_smooth} (beta={getattr(config, 'label_smoothing_beta', 0.5)}, max={getattr(config, 'label_smoothing_max', 0.6)})")
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

    # Auxiliary prototypicality predictor
    proto_predictor = None
    if getattr(config, 'use_proto_predictor', False):
        proto_hidden = getattr(config, 'proto_predictor_hidden_dim', 256)
        proto_predictor = PrototypicalityPredictor(
            input_dim=config.hidden_dim,
            hidden_dim=proto_hidden,
        ).to(device)
        print(f"   Proto predictor: {config.hidden_dim} -> {proto_hidden} -> 1")
        print(f"   Proto predictor weight: {getattr(config, 'proto_predictor_weight', 0.5)}")

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
    if audio_encoder_type in ("wav2vec2", "emotion2vec") and unfreeze_audio > 0 and hasattr(model, 'audio_encoder') and model.audio_encoder is not None:
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
        param_groups.append({'params': list(proto_predictor.parameters()), 'lr': config.learning_rate})
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

    # Initialize WandB — resume existing run if we have a run ID
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

        # Apply LR factor for stage 2 (skip on resume — already in optimizer state)
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
            )
            global_step = train_metrics['global_step']

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
            # Merge so multi-seed averaging in runner.py picks them up.
            test_metrics.update(emb_metrics)
            print(f"      Acc: {test_metrics['accuracy']:.4f}")
            print(f"      UAR: {test_metrics['uar']:.4f}")
            print(f"      within_var_ratio: {emb_metrics['within_var_ratio']:.4f}, "
                  f"embed_rank: {emb_metrics['effective_rank']:.1f}")
            wandb.log({
                f'test/{dataset_name}_acc': test_metrics['accuracy'],
                f'test/{dataset_name}_uar': test_metrics['uar'],
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
                                'heldout', use_amp)
        for test_loader, test_dataset in zip(test_loaders, test_datasets):
            _dump_split_predictions(model, test_loader, config, device,
                                    test_dataset.dataset_name, use_amp)

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

    # Mark run as finished — writes results.json and status=done, removes latest.pt
    mark_run_done(config, results)

    wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config)
