#!/usr/bin/env python3
"""
Clean training script for emotion recognition with contrastive learning
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
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
from utils.prototypicality import batch_calculate_difficulty, calculate_difficulty, batch_difficulty_tensor, LearnableCentroids
from utils.multiview_prototypicality import compute_multiview_difficulty, compute_crossmodal_agreement


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

    uses_raw_audio = audio_encoder_type in ("wav2vec2", "emotion2vec")

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
                modality_discriminator=None, centroid_tracker=None):
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

    total_loss = 0
    total_primary_loss = 0
    total_contrastive_loss = 0
    total_adversarial_loss = 0
    total_modality_adversarial_loss = 0
    total_proto_pred_loss = 0

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
                use_proto_weight = getattr(config, 'use_prototypical_weighting', False)
                use_label_smooth = getattr(config, 'use_prototypical_label_smoothing', False)

                if use_label_smooth:
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
                elif use_proto_weight:
                    # Standard CE with reduction='none'
                    per_sample_loss = criterion(logits, labels)  # [B]
                else:
                    # Sum-based normalization: sum(w*CE) / B instead of sum(w*CE) / sum(w).
                    # When divided by accum_steps at backward, gives sum(w*CE) / (accum_steps * B)
                    # which is identical across micro-batches regardless of class composition.
                    cls_weights = criterion.weight if hasattr(criterion, 'weight') else None
                    loss_primary = F.cross_entropy(
                        logits, labels, weight=cls_weights, reduction='sum'
                    ) / labels.size(0)

                if use_proto_weight or use_label_smooth:
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
                        else:
                            difficulties = difficulties.squeeze(1)  # undo unsqueeze from label smoothing
                        alpha = getattr(config, 'prototypical_weighting_alpha', 2.0)
                        sign = 1.0 if getattr(config, 'ce_weight_invert', False) else -1.0
                        sample_weights = torch.exp(sign * alpha * difficulties)  # [B]
                        # Normalize so weights have mean 1 (preserves gradient scale)
                        sample_weights = sample_weights * (sample_weights.numel() / (sample_weights.sum() + 1e-8))
                        loss_primary = (per_sample_loss * sample_weights).mean()
                    else:
                        loss_primary = per_sample_loss.mean()

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

            # Combined loss
            adv_weight = getattr(config, 'adversarial_weight', 0.0)
            modality_adv_weight = getattr(config, 'modality_adv_weight', 0.0)
            proto_pred_weight = getattr(config, 'proto_predictor_weight', 0.0)
            cross_modal_weight = getattr(config, 'cross_modal_weight', 0.1)
            loss = (loss_primary
                    + effective_contrastive_weight * loss_contrastive
                    + adv_weight * loss_adversarial
                    + modality_adv_weight * loss_modality_adversarial
                    + proto_pred_weight * loss_proto_pred
                    + cross_modal_weight * loss_cross_modal)

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

    if config.task_type == "regression":
        vad_preds = np.concatenate(all_vad_preds, axis=0)
        vad_targets = np.concatenate(all_vad_targets, axis=0)
        metrics = calculate_vad_metrics(vad_preds, vad_targets)
    else:
        metrics = calculate_classification_metrics(all_predictions, all_labels)

    return {
        'loss': avg_loss,
        'primary_loss': avg_primary_loss,
        'contrastive_loss': avg_contrastive_loss,
        'adversarial_loss': avg_adversarial_loss,
        'modality_adversarial_loss': avg_modality_adversarial_loss,
        'proto_pred_loss': avg_proto_pred_loss,
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


def is_run_finished(config):
    """Check if this experiment+seed has already completed."""
    _, _, status_path, results_path = get_checkpoint_paths(config)
    if not status_path.exists() or not results_path.exists():
        return False
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
        return status.get('status') == 'done'
    except (json.JSONDecodeError, IOError):
        return False


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

    # Stratified train/val split (preserves class distribution)
    total_samples = len(train_dataset)
    labels = [train_dataset.data[i]['label'] for i in range(total_samples)]

    # Group indices by class
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        np.random.shuffle(indices)
        val_size = max(1, int(len(indices) * config.val_split))
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])

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

    if use_two_stage:
        difficulties = _compute_per_sample_difficulty(train_dataset, config)
        # Compute threshold on training indices only
        train_difficulties = difficulties[train_indices]
        threshold = np.percentile(train_difficulties, config.stage1_percentile)
        # Prototypical subset: samples with difficulty <= threshold
        proto_train_indices = [idx for idx in train_indices if difficulties[idx] <= threshold]
        print(f"  Two-stage training enabled:")
        print(f"   Stage 1: {len(proto_train_indices)}/{len(train_indices)} prototypical samples (percentile={config.stage1_percentile}, threshold={threshold:.4f})")
        print(f"   Stage 1 epochs: {config.stage1_epochs}, Stage 2 LR factor: {config.stage2_lr_factor}")

    # Create dataloaders
    num_workers = getattr(config, 'num_workers', 2)
    eval_batch_size = getattr(config, 'eval_batch_size', None) or config.batch_size
    eval_batch_size = int(eval_batch_size)
    pin_memory = device.type == 'cuda'
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
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
        freq_weights = []
        for i in range(config.num_classes):
            if class_counts[i] > 0:
                freq_ratio = class_counts[i] / total_samples
                freq_weight = (1.0 / freq_ratio) / config.num_classes
            else:
                freq_weight = 1.0
            freq_weights.append(freq_weight)

        # Normalize so weights sum to num_classes
        total_weight = sum(freq_weights)
        freq_weights = [w / total_weight * config.num_classes for w in freq_weights]

        print(f"   Class counts: {class_counts}")
        print(f"   Class weights: {[f'{w:.3f}' for w in freq_weights]}")

        freq_weights_tensor = torch.tensor(freq_weights, dtype=torch.float32).to(device)

        # Use reduction='none' if we need per-sample weighting or label smoothing
        use_proto_weight = getattr(config, 'use_prototypical_weighting', False)
        use_label_smooth = getattr(config, 'use_prototypical_label_smoothing', False)
        if use_proto_weight or use_label_smooth:
            criterion = nn.CrossEntropyLoss(weight=freq_weights_tensor, reduction='none')
            print(f"   Prototypical weighting: {use_proto_weight} (alpha={getattr(config, 'prototypical_weighting_alpha', 2.0)})")
            print(f"   Label smoothing: {use_label_smooth} (beta={getattr(config, 'label_smoothing_beta', 0.5)}, max={getattr(config, 'label_smoothing_max', 0.6)})")
        else:
            criterion = nn.CrossEntropyLoss(weight=freq_weights_tensor)

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

    global_epoch = resume_start_epoch
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
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"   Early stopping: no improvement for {early_stopping_patience} epochs")
                    early_stop_triggered = True

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

    # Load best model and evaluate on test sets
    print(f"\n  Evaluating on test datasets...")
    model.load_state_dict(best_model_state)

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
            print(f"      Acc: {test_metrics['accuracy']:.4f}")
            print(f"      UAR: {test_metrics['uar']:.4f}")
            wandb.log({
                f'test/{dataset_name}_acc': test_metrics['accuracy'],
                f'test/{dataset_name}_uar': test_metrics['uar'],
                f'cm/test_{dataset_name}': wandb.plot.confusion_matrix(
                    preds=test_metrics['predictions'],
                    y_true=test_metrics['labels'],
                    class_names=['neutral', 'happy', 'sad', 'angry'],
                    title=f'{dataset_name} (best model)',
                ),
            })

    # Val confusion matrix (best model)
    if config.task_type == "classification":
        val_best = evaluate(model, val_loader, criterion, device, config, use_amp)
        wandb.log({
            'cm/val': wandb.plot.confusion_matrix(
                preds=val_best['predictions'],
                y_true=val_best['labels'],
                class_names=['neutral', 'happy', 'sad', 'angry'],
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
