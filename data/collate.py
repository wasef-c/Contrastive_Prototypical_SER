#!/usr/bin/env python3
"""
Custom collate function for VAD regression
"""

import torch


def vad_collate_fn(batch):
    """
    Collate function that handles VAD values for regression tasks

    Creates:
        - batch['vad']: [batch_size, 3] tensor combining V, A, D
        - Preserves separate valence/arousal/dominance for compatibility

    Args:
        batch: list of dicts from dataset.__getitem__()

    Returns:
        dict with batched tensors and lists
    """
    collated = {
        'label': [],
        'dataset': [],
    }

    # Check for optional fields
    has_features = 'features' in batch[0] and batch[0]['features'] is not None
    has_transcript = 'transcript' in batch[0]
    has_vad = 'valence' in batch[0]

    if has_features:
        collated['features'] = []
    if has_transcript:
        collated['transcript'] = []
    if has_vad:
        collated['vad'] = []
        collated['valence'] = []
        collated['arousal'] = []
        collated['dominance'] = []

    # Collect items
    for item in batch:
        if has_features:
            collated['features'].append(item['features'])
        if has_transcript:
            collated['transcript'].append(item['transcript'])

        collated['label'].append(item['label'])
        collated['dataset'].append(item['dataset'])

        if has_vad:
            valence = item['valence']
            arousal = item['arousal']
            dominance = item['dominance']

            # Combined VAD tensor
            collated['vad'].append([valence, arousal, dominance])

            # Separate fields
            collated['valence'].append(valence)
            collated['arousal'].append(arousal)
            collated['dominance'].append(dominance)

    # Convert to tensors
    if has_features:
        collated['features'] = torch.stack(collated['features'])
    if has_vad:
        collated['vad'] = torch.tensor(collated['vad'], dtype=torch.float32)
        collated['valence'] = torch.tensor(collated['valence'], dtype=torch.float32)
        collated['arousal'] = torch.tensor(collated['arousal'], dtype=torch.float32)
        collated['dominance'] = torch.tensor(collated['dominance'], dtype=torch.float32)

    collated['label'] = torch.stack(collated['label'])

    return collated
