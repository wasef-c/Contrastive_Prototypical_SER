#!/usr/bin/env python3
"""
Data package for Emotion2Vec Contrastive Learning
"""

from .dataset import EmotionDataset, MultiCorpusDataset, create_datasets
from .collate import vad_collate_fn

__all__ = [
    'EmotionDataset',
    'MultiCorpusDataset',
    'create_datasets',
    'vad_collate_fn',
]
