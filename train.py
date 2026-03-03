#!/usr/bin/env python3
"""
Clean training script for emotion recognition with contrastive learning
"""

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

from utils.config import Config
from data.dataset import create_datasets
from data.collate import vad_collate_fn
from models.classifier import create_model
from models.contrastive_loss import create_contrastive_loss
from models.domain_adversarial import DomainDiscriminator, PrototypicalDomainAdversarialLoss
from models.prototypicality_predictor import PrototypicalityPredictor
from utils.metrics import calculate_classification_metrics, calculate_vad_metrics
from utils.prototypicality import batch_calculate_difficulty, calculate_difficulty


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

    if config.modality == "audio":
        if audio_encoder_type == "wav2vec2":
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

        if audio_encoder_type == "wav2vec2":
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
                proto_predictor=None, scaler=None):
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

    Returns:
        dict with loss and metrics
    """
    use_amp = scaler is not None
    model.train()
    if domain_discriminator is not None:
        domain_discriminator.train()
    if proto_predictor is not None:
        proto_predictor.train()

    total_loss = 0
    total_primary_loss = 0
    total_contrastive_loss = 0
    total_adversarial_loss = 0
    total_proto_pred_loss = 0

    all_predictions = []
    all_labels = []
    all_vad_preds = []
    all_vad_targets = []

    for batch in dataloader:
        # Move to device
        labels = batch['label'].to(device)

        # Prepare inputs based on modality and audio encoder type
        model_inputs = _prepare_model_inputs(batch, config, model, device)

        # Forward pass with embedding extraction (inside autocast for mixed precision)
        use_embeddings = config.use_contrastive or getattr(config, 'use_domain_adversarial', False) or getattr(config, 'use_proto_predictor', False)
        raw_embeddings = None
        embeddings_norm = None

        with autocast('cuda', enabled=use_amp):
            if use_embeddings:
                logits, projected_embeddings, raw_embeddings = model(**model_inputs, return_embeddings=True)
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
                    # Standard CE (already reduced)
                    loss_primary = criterion(logits, labels)

                if use_proto_weight or use_label_smooth:
                    if use_proto_weight:
                        # Weight per-sample loss by prototypicality
                        if not use_label_smooth:
                            difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                        else:
                            difficulties = difficulties.squeeze(1)  # undo unsqueeze from label smoothing
                        alpha = getattr(config, 'prototypical_weighting_alpha', 2.0)
                        sample_weights = torch.exp(-alpha * difficulties)  # [B]
                        # Normalize so weights have mean 1 (preserves gradient scale)
                        sample_weights = sample_weights * (sample_weights.numel() / (sample_weights.sum() + 1e-8))
                        loss_primary = (per_sample_loss * sample_weights).mean()
                    else:
                        loss_primary = per_sample_loss.mean()

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_predictions.extend(preds)
                all_labels.extend(labels.cpu().numpy())

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

            # Auxiliary prototypicality prediction loss
            loss_proto_pred = torch.tensor(0.0, device=device)
            if proto_predictor is not None and raw_embeddings is not None:
                difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                pred_proto = proto_predictor(raw_embeddings)
                loss_proto_pred = F.mse_loss(pred_proto, difficulties)

            # Combined loss
            adv_weight = getattr(config, 'adversarial_weight', 0.0)
            proto_pred_weight = getattr(config, 'proto_predictor_weight', 0.0)
            loss = loss_primary + config.contrastive_weight * loss_contrastive + adv_weight * loss_adversarial + proto_pred_weight * loss_proto_pred

        # Backward (with scaler if mixed precision)
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_primary_loss += loss_primary.item()
        if config.use_contrastive:
            total_contrastive_loss += loss_contrastive.item()
        if domain_discriminator is not None:
            total_adversarial_loss += loss_adversarial.item()
        if proto_predictor is not None:
            total_proto_pred_loss += loss_proto_pred.item()

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_primary_loss = total_primary_loss / len(dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(dataloader) if config.use_contrastive else 0.0
    avg_adversarial_loss = total_adversarial_loss / len(dataloader) if domain_discriminator is not None else 0.0
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
        'proto_pred_loss': avg_proto_pred_loss,
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

    return {
        'loss': avg_loss,
        **metrics
    }


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
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=vad_collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=vad_collate_fn
    )

    test_loaders = []
    for test_dataset in test_datasets:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=vad_collate_fn
        )
        test_loaders.append(test_loader)

    # Create model
    print(f"\n  Creating model...")
    model = create_model(config).to(device)
    print(f"   Modality: {config.modality}")
    print(f"   Task: {config.task_type}")
    print(f"   Audio encoder: {getattr(config, 'audio_encoder_type', 'preextracted')}")
    print(f"   Contrastive: {config.use_contrastive}")

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
        ).to(device)
        print(f"   Contrastive loss: {config.contrastive_loss_type}")

    # Domain adversarial training
    domain_discriminator = None
    domain_adv_loss = None
    use_adversarial = getattr(config, 'use_domain_adversarial', False)

    if use_adversarial:
        # Determine number of domains from training data
        train_names = config.train_dataset
        if isinstance(train_names, str):
            train_names = [train_names]
        num_domains = len(train_names)

        if num_domains < 2:
            print("   WARNING: Domain adversarial requires multi-corpus training. Disabling.")
            use_adversarial = False
        else:
            adv_alpha = getattr(config, 'adversarial_alpha', 2.0)
            adv_hidden = getattr(config, 'adversarial_hidden_dim', 256)

            domain_discriminator = DomainDiscriminator(
                input_dim=config.hidden_dim,
                hidden_dim=adv_hidden,
                num_domains=num_domains,
            ).to(device)

            domain_adv_loss = PrototypicalDomainAdversarialLoss(alpha=adv_alpha).to(device)

            print(f"   Domain adversarial: {num_domains} domains, alpha={adv_alpha}")
            print(f"   Adversarial weight: {getattr(config, 'adversarial_weight', 0.1)}")
            print(f"   Prototypicality-weighted: {getattr(config, 'use_prototypical_adversarial', True)}")

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

    # Wav2Vec2 differential LR
    if audio_encoder_type == "wav2vec2" and unfreeze_audio > 0 and hasattr(model, 'audio_encoder') and model.audio_encoder is not None:
        audio_params = model.audio_encoder.get_audio_params()
        special_param_ids.update(id(p) for p in audio_params)
        param_groups.append({'params': audio_params, 'lr': audio_lr})
        print(f"   Wav2Vec2 LR: {audio_lr:.2e}")

    # All other model params
    other_model_params = [p for p in model.parameters() if id(p) not in special_param_ids]
    param_groups.append({'params': other_model_params, 'lr': config.learning_rate})
    print(f"   Main LR: {config.learning_rate:.2e}")

    # Add contrastive criterion + domain discriminator + proto predictor params
    if contrastive_criterion is not None:
        param_groups.append({'params': list(contrastive_criterion.parameters()), 'lr': config.learning_rate})
    if domain_discriminator is not None:
        param_groups.append({'params': list(domain_discriminator.parameters()), 'lr': config.learning_rate})
    if proto_predictor is not None:
        param_groups.append({'params': list(proto_predictor.parameters()), 'lr': config.learning_rate})

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler (CosineAnnealing like old repo)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    print(f"   Scheduler: CosineAnnealingLR (T_max={config.num_epochs})")

    # Mixed precision scaler (enabled for CUDA with wav2vec2 or when explicitly requested)
    use_amp = device.type == 'cuda' and audio_encoder_type == 'wav2vec2'
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print(f"   Mixed precision (fp16): enabled")

    # Initialize WandB
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=config.to_dict()
    )

    # Early stopping setup
    early_stopping_patience = getattr(config, 'early_stopping_patience', 10)
    epochs_without_improvement = 0

    # Best model tracking (across both stages)
    best_val_metric = 0.0
    best_model_state = None
    best_contrastive_state = None

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

    global_epoch = 0

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
            collate_fn=vad_collate_fn
        )

        # Apply LR factor for stage 2
        if lr_factor != 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * lr_factor
            print(f"   LR reduced by {lr_factor}x")

        # Reset early stopping between stages
        if use_two_stage:
            epochs_without_improvement = 0

        for epoch_in_stage in range(stage_epochs):
            global_epoch += 1
            print(f"\n  Epoch {global_epoch}/{config.num_epochs} ({stage_name})")

            # Schedule GRL lambda (ramp up adversarial strength over training)
            if domain_discriminator is not None:
                # Linear ramp from 0 to 1 over first half of training
                p = min(1.0, global_epoch / (config.num_epochs * 0.5))
                grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Sigmoid schedule
                domain_discriminator.set_lambda(grl_lambda)

            # Train
            train_metrics = train_epoch(
                model, stage_loader, criterion, optimizer, device, config,
                contrastive_criterion, domain_discriminator, domain_adv_loss,
                proto_predictor, scaler
            )

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

            if domain_discriminator is not None:
                log_dict['train/adversarial_loss'] = train_metrics['adversarial_loss']
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
    wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config)
