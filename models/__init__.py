#!/usr/bin/env python3
"""
Models package for Emotion2Vec Contrastive Learning
"""

from .classifier import EmotionClassifier, create_model
from .contrastive_loss import (
    SupervisedContrastiveLoss,
    PrototypicalContrastiveLoss_V1,
    PrototypicalContrastiveLoss_V2,
    PrototypicalContrastiveLoss_V3,
    create_contrastive_loss
)
from .encoder import FrozenBERTEncoder
from .fusion import (
    SimpleConcatFusion,
    CrossAttentionFusion,
    GatedFusion,
    AdaptiveFusion,
    get_fusion_module
)

__all__ = [
    'EmotionClassifier',
    'create_model',
    'SupervisedContrastiveLoss',
    'PrototypicalContrastiveLoss_V1',
    'PrototypicalContrastiveLoss_V2',
    'PrototypicalContrastiveLoss_V3',
    'create_contrastive_loss',
    'FrozenBERTEncoder',
    'SimpleConcatFusion',
    'CrossAttentionFusion',
    'GatedFusion',
    'AdaptiveFusion',
    'get_fusion_module',
]
