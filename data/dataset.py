#!/usr/bin/env python3
"""
Simplified emotion dataset loader for cross-corpus experiments
"""

import torch
import torchaudio
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
        # "MSPP": "cairocode/MSPP_Audio_Text_Merged",
        # "CMUMOSEI": "cairocode/CMUMOSEI_Emotion2Vec_PrecomputedEncodings",
        # "SAMSEMO": "cairocode/SAMSEMO_Emotion2Vec_PrecomputedEncodings",
        "MSPP": "cairocode/MSPP_WAV",
        "CMUMOSEI": "cairocode/cmu_mosei_wav_2",
        "SAMSEMO": "cairocode/samsemo-audio",
    }

    # Alternative HF paths for raw audio (wav2vec2/emotion2vec online encoder)
    # Used when the main DATASET_MAP entry lacks an 'audio' column
    AUDIO_DATASET_MAP = {
        "MSPP": "cairocode/MSPP_WAV",
        "CMUMOSEI": "cairocode/cmu_mosei_wav_2",
        "SAMSEMO": "cairocode/samsemo-audio",
    }

    # String emotion class → integer label (for datasets like MSPP_WAV without 'label' column)
    EMOCLASS_TO_LABEL = {"N": 0, "H": 1, "S": 2, "A": 3}

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

        # Fetch columns in a single vectorized Arrow access (huge speedup vs
        # per-row dict iteration, especially on Lustre).
        cols = self.hf_dataset.column_names
        n = len(self.hf_dataset)

        def first_col(names: list, default=None) -> list:
            """Return the first column in `names` that exists, else [default]*n."""
            for name in names:
                if name in cols:
                    return self.hf_dataset[name]
            return [default] * n

        all_labels = first_col(["label"])
        all_emoclass = first_col(["EmoClass"])

        # VAD columns: support multiple naming conventions across corpora
        # (e.g. MSPI uses valence/arousal/domination, CMUMOSEI/SAMSEMO use
        # consensus_*, MSPP uses EmoVal/EmoAct/EmoDom).
        all_val = first_col(["valence", "consensus_valence", "EmoVal"])
        all_act = first_col(["arousal", "consensus_arousal", "EmoAct"])
        all_dom = first_col(["dominance", "domination", "consensus_dominance", "EmoDom"])

        all_transcripts = [None] * n
        if self.modality in ["text", "both"]:
            all_transcripts = first_col(["transcript", "text"], default="[EMPTY]")

        # Preextracted feature column is touched per-row (large nested structs);
        # only grab the whole column when actually needed.
        all_features = None
        if not self.uses_raw_audio and self.modality in ["audio", "both"]:
            if "emotion2vec_features" in cols:
                all_features = self.hf_dataset["emotion2vec_features"]

        def is_missing(val) -> bool:
            return val is None or (isinstance(val, float) and math.isnan(val))

        def normalize(val):
            if is_missing(val):
                return None
            return (val - 1) / 6 if dataset_name == "MSPP" else (val - 1) / 4

        for i in range(n):
            # Label: numeric 'label' column or string 'EmoClass' fallback
            label = all_labels[i]
            if label is None and all_emoclass[i] is not None:
                label = self.EMOCLASS_TO_LABEL.get(all_emoclass[i], -1)
            if label is None or label < 0 or label > 3:
                skipped_label_count += 1
                continue

            raw_v, raw_a, raw_d = all_val[i], all_act[i], all_dom[i]

            # Regression requires all three VAD values present
            if task_type == "regression":
                if is_missing(raw_v) or is_missing(raw_a) or is_missing(raw_d):
                    skipped_vad_count += 1
                    continue

            v_norm = normalize(raw_v)
            a_norm = normalize(raw_a)
            d_norm = normalize(raw_d)

            sample = {
                "label": label,
                "valence": 0.5 if v_norm is None else v_norm,
                "arousal": 0.5 if a_norm is None else a_norm,
                "dominance": 0.5 if d_norm is None else d_norm,
                "transcript": all_transcripts[i] or "[EMPTY]",
                "dataset": dataset_name,
            }

            if self.uses_raw_audio:
                # Lazy waveform load in __getitem__
                sample["hf_idx"] = i
            else:
                if all_features is None or all_features[i] is None:
                    continue
                sample["features"] = all_features[i][0]["feats"]

            self.data.append(sample)

        print(f"  Loaded {len(self.data)} samples from {dataset_name}")
        if skipped_vad_count > 0:
            print(f"   Skipped {skipped_vad_count} samples with missing/NaN VAD values (regression mode)")
        if skipped_label_count > 0:
            print(f"   Skipped {skipped_label_count} samples with invalid label (None or > 3)")

    def cache_encoder_features(self, encoder, device, batch_size=16):
        """
        Precompute and cache all audio encoder features, converting this dataset
        from lazy waveform loading to cached feature mode. After caching, each
        epoch is as fast as preextracted features.

        Only useful when the encoder is frozen (features don't change between epochs).

        Args:
            encoder: Audio encoder module (Emotion2VecEncoder or Wav2Vec2Encoder)
            device: torch device
            batch_size: Number of clips to process at once
        """
        if not self.uses_raw_audio:
            return  # Already using preextracted features

        print(f"  Caching {len(self.data)} encoder features for {self.dataset_name}...")
        encoder.eval()

        from data.collate import vad_collate_fn
        from torch.utils.data import DataLoader

        temp_loader = DataLoader(
            self, batch_size=batch_size, shuffle=False,
            collate_fn=vad_collate_fn, num_workers=0,
        )

        all_features = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(temp_loader):
                waveforms = batch['waveforms'].to(device)
                mask = batch['audio_attention_mask'].to(device)
                feats = encoder(waveforms, mask)  # [B, 768]
                all_features.append(feats.cpu())
                if (batch_idx + 1) % 100 == 0:
                    print(f"    {(batch_idx+1)*batch_size}/{len(self.data)} cached...")

        all_features = torch.cat(all_features, dim=0)  # [N, 768]

        # Replace lazy loading with cached features
        for i, sample in enumerate(self.data):
            sample.pop("hf_idx", None)
            sample["features"] = all_features[i]

        self.uses_raw_audio = False
        self.hf_dataset = None  # Free HF dataset memory
        print(f"  Cached {len(self.data)} features ({all_features.shape})")

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
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            src_sr = audio["sampling_rate"]
            # Resample to 16kHz (matches preprocessing used for precomputed features)
            if src_sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=src_sr, new_freq=16000)
            # Truncate to max length (encoder handles per-clip processing, no padding needed)
            max_samples = int(getattr(self.config, 'max_audio_seconds', 40) * 16000)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
            result["waveform"] = waveform

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

    def cache_encoder_features(self, encoder, device, batch_size=16):
        """Cache features for all sub-datasets."""
        for ds in self.datasets:
            ds.cache_encoder_features(encoder, device, batch_size)
        # Rebuild combined data list after caching
        self.data = []
        self._cumulative = [0]
        for ds in self.datasets:
            self.data.extend(ds.data)
            self._cumulative.append(self._cumulative[-1] + len(ds.data))

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
