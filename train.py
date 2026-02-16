#!/usr/bin/env python3
"""
Clean training script for emotion recognition with contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
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
from utils.metrics import calculate_classification_metrics, calculate_vad_metrics
from utils.prototypicality import batch_calculate_difficulty


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, config, contrastive_criterion=None):
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

    Returns:
        dict with loss and metrics
    """
    model.train()

    total_loss = 0
    total_primary_loss = 0
    total_contrastive_loss = 0

    all_predictions = []
    all_labels = []
    all_vad_preds = []
    all_vad_targets = []

    for batch in dataloader:
        # Move to device
        labels = batch['label'].to(device)
        features = batch.get('features')
        if features is not None:
            features = features.to(device)

        # Prepare inputs based on modality
        if config.modality == "audio":
            model_inputs = {'audio_features': features}
        elif config.modality == "text":
            # Tokenize text
            transcripts = batch['transcript']
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                input_ids, attention_mask = model.text_encoder.tokenize_batch(
                    transcripts, max_length=config.text_max_length, device=device
                )
                model_inputs = {'text_input_ids': input_ids, 'text_attention_mask': attention_mask}
        elif config.modality == "both":
            transcripts = batch['transcript']
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                input_ids, attention_mask = model.text_encoder.tokenize_batch(
                    transcripts, max_length=config.text_max_length, device=device
                )
                model_inputs = {
                    'audio_features': features,
                    'text_input_ids': input_ids,
                    'text_attention_mask': attention_mask
                }

        # Forward pass with embedding extraction
        if config.use_contrastive:
            logits, embeddings = model(**model_inputs, return_embeddings=True)
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        else:
            logits = model(**model_inputs)
            embeddings_norm = None

        # Primary loss
        if config.task_type == "regression":
            vad_targets = batch['vad'].to(device)
            loss_primary = criterion(logits, vad_targets)
            all_vad_preds.append(logits.detach().cpu().numpy())
            all_vad_targets.append(vad_targets.cpu().numpy())
        else:
            loss_primary = criterion(logits, labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Contrastive loss
        loss_contrastive = torch.tensor(0.0, device=device)
        if config.use_contrastive and contrastive_criterion is not None and embeddings_norm is not None:
            # Calculate difficulties if needed for prototypical variants
            if config.contrastive_loss_type.startswith('prototypical'):
                difficulties = batch_calculate_difficulty(batch, config.expected_vad).to(device)
                loss_contrastive = contrastive_criterion(embeddings_norm, labels, difficulties)
            else:
                loss_contrastive = contrastive_criterion(embeddings_norm, labels)

        # Combined loss
        loss = loss_primary + config.contrastive_weight * loss_contrastive

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_primary_loss += loss_primary.item()
        if config.use_contrastive:
            total_contrastive_loss += loss_contrastive.item()

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_primary_loss = total_primary_loss / len(dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(dataloader) if config.use_contrastive else 0.0

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
        **metrics
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config):
    """
    Evaluate model

    Args:
        model: EmotionClassifier
        dataloader: DataLoader
        criterion: Loss function
        device: torch device
        config: Config object

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
        features = batch.get('features')
        if features is not None:
            features = features.to(device)

        # Prepare inputs
        if config.modality == "audio":
            model_inputs = {'audio_features': features}
        elif config.modality == "text":
            transcripts = batch['transcript']
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                input_ids, attention_mask = model.text_encoder.tokenize_batch(
                    transcripts, max_length=config.text_max_length, device=device
                )
                model_inputs = {'text_input_ids': input_ids, 'text_attention_mask': attention_mask}
        elif config.modality == "both":
            transcripts = batch['transcript']
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                input_ids, attention_mask = model.text_encoder.tokenize_batch(
                    transcripts, max_length=config.text_max_length, device=device
                )
                model_inputs = {
                    'audio_features': features,
                    'text_input_ids': input_ids,
                    'text_attention_mask': attention_mask
                }

        # Forward pass
        logits = model(**model_inputs)

        # Calculate loss
        if config.task_type == "regression":
            vad_targets = batch['vad'].to(device)
            loss = criterion(logits, vad_targets)
            all_vad_preds.append(logits.cpu().numpy())
            all_vad_targets.append(vad_targets.cpu().numpy())
        else:
            loss = criterion(logits, labels)
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


def train(config):
    """
    Main training function

    Args:
        config: Config object
    """
    # Set seed
    set_seed(config.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Create datasets
    print(f"\n📊 Loading datasets...")
    train_dataset, test_datasets = create_datasets(config)

    # Split train into train/val
    total_samples = len(train_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)

    val_size = int(total_samples * config.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    print(f"📈 Train: {len(train_subset)}, Val: {len(val_subset)}")

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
    print(f"\n🔨 Creating model...")
    model = create_model(config).to(device)
    print(f"   Modality: {config.modality}")
    print(f"   Task: {config.task_type}")
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
        criterion = nn.CrossEntropyLoss(weight=freq_weights_tensor)

    contrastive_criterion = None
    if config.use_contrastive:
        contrastive_criterion = create_contrastive_loss(
            config.contrastive_loss_type,
            temperature=config.contrastive_temperature,
            alpha=config.prototypical_alpha,
            beta=config.prototypical_beta,
            threshold=config.prototypical_threshold
        ).to(device)
        print(f"   Contrastive loss: {config.contrastive_loss_type}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler (CosineAnnealing like old repo)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    print(f"   Scheduler: CosineAnnealingLR (T_max={config.num_epochs})")

    # Initialize WandB
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=config.to_dict()
    )

    # Training loop
    print(f"\n🚀 Starting training for {config.num_epochs} epochs...")
    best_val_metric = 0.0
    best_model_state = None

    for epoch in range(config.num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{config.num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config, contrastive_criterion
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   LR: {current_lr:.2e}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config)

        # Log to WandB
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_metrics['loss'],
            'train/primary_loss': train_metrics['primary_loss'],
            'val/loss': val_metrics['loss'],
        }

        if config.use_contrastive:
            log_dict['train/contrastive_loss'] = train_metrics['contrastive_loss']

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

        # Save best model
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_model_state = model.state_dict().copy()
            print(f"   🌟 New best {metric_name}: {current_metric:.4f}")

    # Load best model and evaluate on test sets
    print(f"\n📊 Evaluating on test datasets...")
    model.load_state_dict(best_model_state)

    for test_loader, test_dataset in zip(test_loaders, test_datasets):
        test_metrics = evaluate(model, test_loader, criterion, device, config)
        dataset_name = test_dataset.dataset_name

        print(f"\n   {config.train_dataset} → {dataset_name}:")
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

    torch.save({
        'model_state_dict': best_model_state,
        'config': config.to_dict(),
        'best_val_metric': best_val_metric,
    }, save_path)

    print(f"\n💾 Model saved to: {save_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config)
