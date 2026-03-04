#!/usr/bin/env python3
"""
Simplified emotion dataset loader for cross-corpus experiments
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
import math


class EmotionDataset(Dataset):
    """
    Simplified dataset for cross-corpus emotion recognition

    Supports:
    - IEMO, MSPI, MSPP (with VAD annotations)
    - Pre-extracted audio features (Emotion2Vec 768-dim)
    - Raw audio waveforms (for Wav2Vec2 encoder)
    - Text transcripts
    - VAD normalization to [0, 1] range
    - Prototypicality (difficulty) calculation
    """

    # Dataset mappings (preextracted Emotion2Vec features)
    DATASET_MAP = {
        "IEMO": "cairocode/IEMO_Audio_Text_Merged",
        "MSPI": "cairocode/MSPI_Audio_Text_Merged",
        "MSPP": "cairocode/MSPP_Audio_Text_Merged",
        "CMUMOSEI": "cairocode/CMUMOSEI_Emotion2Vec_PrecomputedEncodings",
        "SAMSEMO": "cairocode/SAMSEMO_Emotion2Vec_PrecomputedEncodings",
    }

    # Alternative HF paths for wav2vec2 mode (raw audio)
    # CMUMOSEI/SAMSEMO need different paths that contain actual audio
    AUDIO_DATASET_MAP = {
        "CMUMOSEI": "cairocode/cmu_mosei_wav_2",
        "SAMSEMO": "cairocode/samsemo-audio",
    }

    # Datasets that have VAD annotations (for regression and prototypicality)
    DATASETS_WITH_VAD = {"IEMO", "MSPI", "MSPP"}

    def __init__(self, dataset_name, split="train", config=None, task_type="classification"):
        """
        Args:
            dataset_name: str - "IEMO", "MSPI", "MSPP", "CMUMOSEI", or "SAMSEMO"
            split: str - "train" or "test"
            config: Config object with expected_vad and modality
            task_type: str - "classification" or "regression"
        """
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.task_type = task_type
        self.modality = getattr(config, 'modality', 'both') if config else 'both'
        self.audio_encoder_type = getattr(config, 'audio_encoder_type', 'preextracted') if config else 'preextracted'

        self.has_vad = dataset_name in self.DATASETS_WITH_VAD

        # For regression, only datasets with VAD are valid
        if task_type == "regression" and not self.has_vad:
            raise ValueError(f"{dataset_name} has no VAD annotations - cannot use for regression")

        # Determine which HF dataset path to use
        if self.audio_encoder_type in ("wav2vec2", "emotion2vec") and dataset_name in self.AUDIO_DATASET_MAP:
            dataset_path = self.AUDIO_DATASET_MAP[dataset_name]
        elif dataset_name in self.DATASET_MAP:
            dataset_path = self.DATASET_MAP[dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(self.DATASET_MAP.keys())}")

        # Load dataset from HuggingFace
        self.hf_dataset = load_dataset(dataset_path, split=split, trust_remote_code=True)

        print(f"  Loaded {dataset_name}: {len(self.hf_dataset)} samples (encoder: {self.audio_encoder_type})")
        print(f"   Columns: {self.hf_dataset.column_names}")

        # Process data - store only lightweight metadata during init.
        # Raw waveforms are loaded lazily in __getitem__ to avoid loading
        # gigabytes of audio into RAM upfront.
        self.data = []
        self.uses_raw_audio = self.audio_encoder_type in ("wav2vec2", "emotion2vec")
        skipped_vad_count = 0
        skipped_label_count = 0

        for hf_idx, item in enumerate(self.hf_dataset):
            # Extract label
            label = item["label"]

            # Filter out invalid labels (CMUMOSEI/SAMSEMO have labels -1..6, keep only 0-3)
            if label < 0 or label > 3:
                skipped_label_count += 1
                continue

            # For raw audio: only validate the column exists, do NOT load audio yet
            if self.modality in ["audio", "both"]:
                if self.uses_raw_audio:
                    if item.get("audio") is None:
                        continue
                else:
                    # Preextracted: load features eagerly (768 floats, negligible size)
                    if "emotion2vec_features" not in item or item["emotion2vec_features"] is None:
                        continue

            # Extract transcript
            if self.modality in ["text", "both"]:
                transcript = item.get("transcript", item.get("text", "[EMPTY]"))
                if not transcript:
                    transcript = "[EMPTY]"
            else:
                transcript = None

            # Extract VAD values with multiple naming variants
            valence = item.get("valence", item.get("consensus_valence", item.get("EmoVal", None)))
            arousal = item.get("arousal", item.get("consensus_arousal", item.get("EmoAct", None)))
            dominance = item.get("domination", item.get("consensus_dominance", item.get("EmoDom", None)))

            # For regression, skip samples with missing VAD
            if task_type == "regression":
                if valence is None or (isinstance(valence, float) and math.isnan(valence)):
                    skipped_vad_count += 1
                    continue
                if arousal is None or (isinstance(arousal, float) and math.isnan(arousal)):
                    skipped_vad_count += 1
                    continue
                if dominance is None or (isinstance(dominance, float) and math.isnan(dominance)):
                    skipped_vad_count += 1
                    continue

            # Normalize VAD to [0, 1] range
            if valence is not None and not (isinstance(valence, float) and math.isnan(valence)):
                if dataset_name == "MSPP":
                    valence = (valence - 1) / 6  # 1-7 scale -> 0-1
                else:  # IEMO, MSPI use 1-5 scale
                    valence = (valence - 1) / 4  # 1-5 scale -> 0-1

            if arousal is not None and not (isinstance(arousal, float) and math.isnan(arousal)):
                if dataset_name == "MSPP":
                    arousal = (arousal - 1) / 6
                else:
                    arousal = (arousal - 1) / 4

            if dominance is not None and not (isinstance(dominance, float) and math.isnan(dominance)):
                if dataset_name == "MSPP":
                    dominance = (dominance - 1) / 6
                else:
                    dominance = (dominance - 1) / 4

            # For classification, use midpoint if missing
            if valence is None:
                valence = 0.5
            if arousal is None:
                arousal = 0.5
            if dominance is None:
                dominance = 0.5

            # Store lightweight metadata only
            sample = {
                "transcript": transcript,
                "label": label,
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "dataset": dataset_name,
            }

            if self.uses_raw_audio:
                # Store HF index for lazy waveform loading in __getitem__
                sample["hf_idx"] = hf_idx
            else:
                # Preextracted features are small (768 floats), safe to load now
                sample["features"] = item["emotion2vec_features"][0]["feats"]

            self.data.append(sample)

        print(f"  Loaded {len(self.data)} samples from {dataset_name}")
        if skipped_vad_count > 0:
            print(f"   Skipped {skipped_vad_count} samples with missing/NaN VAD values (regression mode)")
        if skipped_label_count > 0:
            print(f"   Skipped {skipped_label_count} samples with label > 3")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample with on-the-fly tensor conversion

        Returns dict with:
            - label: int (0-3)
            - valence, arousal, dominance: float (0-1 normalized)
            - features: tensor [audio_dim] - if preextracted audio
            - waveform: tensor [num_samples] - if wav2vec2 mode
            - transcript: str - if text/both modality
            - dataset: str - dataset name
        """
        item = self.data[idx]

        result = {
            "label": torch.tensor(item["label"], dtype=torch.long),
            "valence": item["valence"],
            "arousal": item["arousal"],
            "dominance": item["dominance"],
            "dataset": item["dataset"],
        }

        # Add audio features (preextracted mode)
        if self.modality in ["audio", "both"] and item.get("features") is not None:
            features = item["features"]
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            result["features"] = features

        # Add waveform (raw audio mode - lazy load from HF dataset on demand)
        if self.modality in ["audio", "both"] and self.uses_raw_audio:
            hf_item = self.hf_dataset[item["hf_idx"]]
            audio = hf_item["audio"]
            waveform = np.array(audio["array"], dtype=np.float32)
            max_samples = int(getattr(self.config, 'max_audio_seconds', 40) * audio["sampling_rate"])
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            result["waveform"] = torch.tensor(waveform, dtype=torch.float32)

        # Add text
        if self.modality in ["text", "both"]:
            result["transcript"] = item["transcript"]

        return result


class MultiCorpusDataset(Dataset):
    """
    Combines multiple EmotionDatasets into one for multi-corpus training.
    Each sample retains its source dataset name for tracking.
    """

    def __init__(self, datasets):
        """
        Args:
            datasets: list of EmotionDataset instances
        """
        self.datasets = datasets
        self.data = []
        self._cumulative = [0]  # cumulative sample counts for __getitem__ routing
        for ds in datasets:
            self.data.extend(ds.data)
            self._cumulative.append(self._cumulative[-1] + len(ds.data))

        self.dataset_name = "+".join(ds.dataset_name for ds in datasets)
        self.modality = datasets[0].modality
        self.audio_encoder_type = datasets[0].audio_encoder_type

        print(f"  Combined {len(datasets)} datasets: {self.dataset_name} ({len(self.data)} total samples)")
        for ds in datasets:
            print(f"   - {ds.dataset_name}: {len(ds.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Route to the parent EmotionDataset so lazy waveform loading works correctly
        import bisect
        ds_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[ds_idx]
        return self.datasets[ds_idx][local_idx]


def create_datasets(config):
    """
    Create train and test datasets based on config.

    Supports single or multi-corpus training:
        train_dataset: "MSPP"                    # single corpus
        train_dataset: ["IEMO", "MSPI", "MSPP"]  # multi-corpus

    Test datasets automatically exclude any datasets used for training.

    Args:
        config: Config object with train_dataset, test_datasets, task_type

    Returns:
        train_dataset, test_datasets (list)
    """
    task_type = getattr(config, 'task_type', 'classification')

    # Support single string or list of training datasets
    train_names = config.train_dataset
    if isinstance(train_names, str):
        train_names = [train_names]

    # Load each training dataset
    train_dataset_list = []
    for name in train_names:
        ds = EmotionDataset(
            name,
            split="train",
            config=config,
            task_type=task_type
        )
        train_dataset_list.append(ds)

    # Combine if multi-corpus, otherwise use single dataset
    if len(train_dataset_list) == 1:
        train_dataset = train_dataset_list[0]
    else:
        train_dataset = MultiCorpusDataset(train_dataset_list)

    # Create test datasets (cross-corpus evaluation)
    test_dataset_names = getattr(config, 'test_datasets', [])

    # If not specified, use all datasets except training ones
    if not test_dataset_names:
        all_datasets = list(EmotionDataset.DATASET_MAP.keys())
        test_dataset_names = [d for d in all_datasets if d not in train_names]

    # Exclude any training datasets from test
    test_dataset_names = [d for d in test_dataset_names if d not in train_names]

    # For regression, filter out datasets without VAD
    if task_type == "regression":
        test_dataset_names = [d for d in test_dataset_names if d in EmotionDataset.DATASETS_WITH_VAD]

    test_datasets = []
    for dataset_name in test_dataset_names:
        test_dataset = EmotionDataset(
            dataset_name,
            split="train",  # Use full dataset for testing
            config=config,
            task_type=task_type
        )
        test_datasets.append(test_dataset)

    print(f"  Training: {train_dataset.dataset_name} -> {test_dataset_names}")

    return train_dataset, test_datasets
