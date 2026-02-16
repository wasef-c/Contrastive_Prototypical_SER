#!/usr/bin/env python3
"""
Evaluation metrics for classification and VAD regression
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def calculate_uar(predictions, labels):
    """
    Calculate Unweighted Average Recall (UAR)

    Args:
        predictions: numpy array of predicted labels
        labels: numpy array of true labels

    Returns:
        float: UAR score
    """
    cm = confusion_matrix(labels, predictions)
    recalls = []
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            recalls.append(cm[i, i] / cm[i].sum())
    return np.mean(recalls) if recalls else 0.0


def calculate_classification_metrics(predictions, labels):
    """
    Calculate classification metrics

    Returns dict with:
        - accuracy
        - uar (unweighted average recall)
        - f1_weighted
        - per_class_recall
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'uar': calculate_uar(predictions, labels),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
    }

    # Per-class recall
    cm = confusion_matrix(labels, predictions)
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            metrics[f'class_{i}_recall'] = cm[i, i] / cm[i].sum()
        else:
            metrics[f'class_{i}_recall'] = 0.0

    return metrics


def calculate_ccc(predictions, targets):
    """
    Calculate Concordance Correlation Coefficient (CCC)

    CCC measures agreement between predictions and targets
    Better than Pearson correlation for regression evaluation

    Args:
        predictions: numpy array
        targets: numpy array

    Returns:
        float: CCC value (-1 to 1, higher is better)
    """
    mean_pred = np.mean(predictions)
    mean_target = np.mean(targets)

    var_pred = np.var(predictions)
    var_target = np.var(targets)

    sd_pred = np.std(predictions)
    sd_target = np.std(targets)

    # Pearson correlation
    if sd_pred > 0 and sd_target > 0:
        pearson_corr = np.corrcoef(predictions, targets)[0, 1]
    else:
        return 0.0

    # CCC formula
    numerator = 2 * pearson_corr * sd_pred * sd_target
    denominator = var_pred + var_target + (mean_pred - mean_target) ** 2

    if denominator > 0:
        ccc = numerator / denominator
    else:
        ccc = 0.0

    return ccc


def calculate_vad_metrics(predictions, targets):
    """
    Calculate VAD regression metrics

    Args:
        predictions: numpy array of shape (n_samples, 3) - V, A, D predictions
        targets: numpy array of shape (n_samples, 3) - V, A, D targets

    Returns:
        dict with MAE, RMSE, correlation, CCC for each dimension and overall
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    vad_names = ['valence', 'arousal', 'dominance']
    metrics = {}

    # Per-dimension metrics
    for i, name in enumerate(vad_names):
        pred_dim = predictions[:, i]
        target_dim = targets[:, i]

        # MAE
        mae = np.mean(np.abs(pred_dim - target_dim))

        # RMSE
        rmse = np.sqrt(np.mean((pred_dim - target_dim) ** 2))

        # Pearson correlation
        if np.std(pred_dim) > 0 and np.std(target_dim) > 0:
            correlation = np.corrcoef(pred_dim, target_dim)[0, 1]
        else:
            correlation = 0.0

        # CCC
        ccc = calculate_ccc(pred_dim, target_dim)

        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_corr'] = correlation
        metrics[f'{name}_ccc'] = ccc

    # Overall metrics (averaged)
    metrics['overall_mae'] = np.mean([metrics[f'{name}_mae'] for name in vad_names])
    metrics['overall_rmse'] = np.mean([metrics[f'{name}_rmse'] for name in vad_names])
    metrics['overall_corr'] = np.mean([metrics[f'{name}_corr'] for name in vad_names])
    metrics['overall_ccc'] = np.mean([metrics[f'{name}_ccc'] for name in vad_names])

    return metrics
