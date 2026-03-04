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
    PrototypeAnchoredLoss,
    PrototypeAnchoredMultiDSLoss,
    PrototypeDivergenceLoss,
    create_contrastive_loss
)
from .domain_adversarial import (
    GradientReversalLayer,
    DomainDiscriminator,
    PrototypicalDomainAdversarialLoss,
)
from .prototypicality_predictor import PrototypicalityPredictor
from .encoder import FrozenBERTEncoder
from .audio_encoder import Wav2Vec2Encoder, Emotion2VecEncoder
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
    'PrototypeAnchoredLoss',
    'PrototypeAnchoredMultiDSLoss',
    'PrototypeDivergenceLoss',
    'create_contrastive_loss',
    'PrototypicalityPredictor',
    'FrozenBERTEncoder',
    'Wav2Vec2Encoder',
    'Emotion2VecEncoder',
    'SimpleConcatFusion',
    'CrossAttentionFusion',
    'GatedFusion',
    'AdaptiveFusion',
    'get_fusion_module',
]
