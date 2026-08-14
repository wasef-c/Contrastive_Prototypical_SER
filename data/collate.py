#!/usr/bin/env python3
"""
Custom collate functions for VAD regression and variable-length waveform padding
"""

import torch


def vad_collate_fn(batch):
    """
    Collate function that handles VAD values, preextracted features,
    and variable-length waveforms.

    Creates:
        - batch['vad']: [batch_size, 3] tensor combining V, A, D
        - batch['waveforms']: [batch_size, max_T] padded waveforms (if present)
        - batch['audio_attention_mask']: [batch_size, max_T] mask (if waveforms present)
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
    has_waveform = 'waveform' in batch[0] and batch[0]['waveform'] is not None
    has_transcript = 'transcript' in batch[0]
    has_vad = 'valence' in batch[0]
    has_difficulty = 'difficulty' in batch[0]
    has_speaker = 'speaker' in batch[0]
    has_label4 = 'label4' in batch[0]

    if has_features:
        collated['features'] = []
    if has_transcript:
        collated['transcript'] = []
    if has_vad:
        collated['vad'] = []
        collated['valence'] = []
        collated['arousal'] = []
        collated['dominance'] = []
    if has_difficulty:
        collated['difficulty'] = []
    if has_speaker:
        collated['speaker'] = []
    if has_label4:
        collated['label4'] = []
    has_sample_index = 'sample_index' in batch[0]
    if has_sample_index:
        collated['sample_index'] = []

    # Collect waveforms separately for padding
    waveform_list = [] if has_waveform else None

    # Collect items
    for item in batch:
        if has_features:
            collated['features'].append(item['features'])
        if has_waveform:
            waveform_list.append(item['waveform'])
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

        if has_difficulty:
            collated['difficulty'].append(item['difficulty'])
        if has_speaker:
            collated['speaker'].append(item['speaker'])
        if has_label4:
            collated['label4'].append(item['label4'])
        if has_sample_index:
            collated['sample_index'].append(item['sample_index'])

    # Convert to tensors
    if has_features:
        collated['features'] = torch.stack(collated['features'])
    if has_vad:
        collated['vad'] = torch.tensor(collated['vad'], dtype=torch.float32)
        collated['valence'] = torch.tensor(collated['valence'], dtype=torch.float32)
        collated['arousal'] = torch.tensor(collated['arousal'], dtype=torch.float32)
        collated['dominance'] = torch.tensor(collated['dominance'], dtype=torch.float32)
    if has_difficulty:
        collated['difficulty'] = torch.tensor(collated['difficulty'], dtype=torch.float32)
    if has_sample_index:
        collated['sample_index'] = torch.tensor(
            collated['sample_index'], dtype=torch.long,
        )

    collated['label'] = torch.stack(collated['label'])
    if has_label4:
        collated['label4'] = torch.stack(collated['label4'])

    # Pad waveforms to max length in batch
    if has_waveform and waveform_list:
        lengths = [w.shape[0] for w in waveform_list]
        max_len = max(lengths)

        padded_waveforms = torch.zeros(len(waveform_list), max_len, dtype=torch.float32)
        attention_mask = torch.zeros(len(waveform_list), max_len, dtype=torch.long)

        for i, (waveform, length) in enumerate(zip(waveform_list, lengths)):
            padded_waveforms[i, :length] = waveform
            attention_mask[i, :length] = 1

        collated['waveforms'] = padded_waveforms
        collated['audio_attention_mask'] = attention_mask

    return collated
