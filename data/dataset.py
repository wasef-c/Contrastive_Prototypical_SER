#!/usr/bin/env python3
"""
Simplified emotion dataset loader for cross-corpus experiments
"""

import math
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset

from utils.frame_cache import (
    finalize_frame_cache,
    load_frame_cache,
    resample_frames,
    write_frame_cache,
)
from utils.prototypicality import calculate_difficulty


AUDIO_FEATURE_CACHE_DIR = Path(
    os.environ.get("AUDIO_FEATURE_CACHE_DIR", ".flash/audio_feature_cache")
)

# Corpora whose released activation annotations run calm-high rather than
# active-high, and so must be flipped to be comparable with the others.
# See orient_arousal in _load_data for the evidence behind this list.
AROUSAL_REVERSED_CORPORA = {"MSPI"}


def _encoder_cache_path(dataset_name: str, encoder_type: str, model_name: str,
                        pooling: str = "mean") -> Path:
    """Deterministic per-(dataset, encoder) cache filename.

    Args:
        dataset_name: e.g. "MSPI".
        encoder_type: e.g. "emotion2vec".
        model_name: HF or local model id, e.g. "iic/emotion2vec_base".
        pooling: temporal pooling mode. Part of the key because different
            modes produce different feature widths; without it a
            mean_std run would silently load stale 768-dim features.

    Returns:
        Absolute Path under AUDIO_FEATURE_CACHE_DIR.
    """
    model_slug = model_name.replace("/", "__").replace(":", "_")
    if pooling and pooling != "mean":
        model_slug = f"{model_slug}__{pooling}"
    fname = f"{dataset_name}__{encoder_type}__{model_slug}.pt"
    return (AUDIO_FEATURE_CACHE_DIR / fname).resolve()


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

        self.frame_cache = None
        self.frame_durations = None
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

        # Collapse to neutral vs emotional when the arm is a presence detector.
        binary_neutral = bool(getattr(config, "binary_neutral", False))

        # VAD columns: support multiple naming conventions across corpora
        # (e.g. MSPI uses valence/arousal/domination, CMUMOSEI/SAMSEMO use
        # consensus_*, MSPP uses EmoVal/EmoAct/EmoDom).
        all_val = first_col(["valence", "consensus_valence", "EmoVal"])
        all_act = first_col(["arousal", "consensus_arousal", "EmoAct"])
        all_dom = first_col(["dominance", "domination", "consensus_dominance", "EmoDom"])

        # Speaker identity, named differently per corpus: MSP-Podcast uses
        # SpkrID, IEMOCAP and MSP-Improv use speakerID. Read defensively and
        # fall back to a per-row unique id so that a corpus without speaker
        # labels degrades to "every utterance is its own speaker" rather
        # than collapsing every row into one bogus speaker.
        all_speakers = first_col(["SpkrID", "speakerID", "speaker_id", "speaker"],
                                 default=None)

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

        def orient_arousal(val):
            """Put arousal on a shared active-high scale across corpora.

            As distributed to us, MSPI arousal runs calm-high: its
            per-class means order angry (2.76) < happy (3.07) < sad (3.72)
            < neutral (3.86), an exact reversal of IEMOCAP's sad <
            neutral < happy < angry. The reversal is specific to arousal.
            Valence (happy 4.05 high, angry 1.98 low) and
            consensus_dominance (angry 3.56 high, sad 2.76 low) both order
            correctly, which rules out a shuffled or mis-mapped column,
            and arousal, arousal_norm and consensus_arousal all carry the
            same reversal. Listening to samples labelled high-arousal
            confirms they sound calm.

            Whether this originates in the corpus release or in the
            cairocode HF merge is not established. The standard SAM
            convention is 1 = calm, 5 = excited, and no statement of a
            reversed scale was found in the MSP-IMPROV paper, so a merge
            artifact is the more likely cause. The correction is applied
            either way because cross-corpus arousal comparisons are
            meaningless without it.

            Training never touches these values (MSP-Podcast is the only
            training corpus), so no reported UAR depends on this. Any
            cross-corpus analysis that compares arousal does.

            Args:
                val: raw arousal on the corpus's own 1-5 scale.

            Returns:
                Arousal reoriented so higher means more activated.
            """
            if is_missing(val) or dataset_name not in AROUSAL_REVERSED_CORPORA:
                return val
            return 6.0 - val

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
            a_norm = normalize(orient_arousal(raw_a))
            d_norm = normalize(raw_d)

            v_val = 0.5 if v_norm is None else v_norm
            a_val = 0.5 if a_norm is None else a_norm
            d_val = 0.5 if d_norm is None else d_norm

            # Precompute difficulty once. Static expected_vad means this never
            # changes across epochs, so per-batch recomputation is wasted work.
            # Learnable-centroid runs bypass this via `centroid_tracker` in the
            # training loop and recompute on-graph instead.
            if self.has_vad and config is not None and getattr(config, 'expected_vad', None) is not None:
                difficulty = calculate_difficulty(v_val, a_val, d_val, label, config.expected_vad)
            else:
                difficulty = 0.0

            # Binary neutral-vs-emotional target. Collapsing happens after
            # difficulty is computed, since expected_vad is keyed by the
            # original 4-class label. The 4-class label is kept alongside so
            # downstream analysis can still slice by emotion.
            out_label = label
            if binary_neutral:
                out_label = 0 if label == 0 else 1

            sample = {
                "label": out_label,
                "label4": label,
                "valence": v_val,
                "arousal": a_val,
                "dominance": d_val,
                "difficulty": difficulty,
                "transcript": all_transcripts[i] or "[EMPTY]",
                "dataset": dataset_name,
                "speaker": (f"{dataset_name}:{all_speakers[i]}"
                            if all_speakers[i] is not None
                            else f"{dataset_name}:row{i}"),
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

        Persists the features to disk under AUDIO_FEATURE_CACHE_DIR keyed by
        (dataset_name, encoder_type, model_name) so subsequent runs skip the
        forward pass entirely. Cache is invalidated by sample-count mismatch.

        Only useful when the encoder is frozen (features don't change between
        epochs / runs).

        Args:
            encoder: Audio encoder module (Emotion2VecEncoder or Wav2Vec2Encoder)
            device: torch device
            batch_size: Number of clips to process at once
        """
        if not self.uses_raw_audio:
            return  # Already using preextracted features

        # Frame-level mode keeps the time axis. Features go to a memmapped
        # file rather than an in-RAM tensor, because [N, T, 768] does not
        # fit: the pooled dup8 cache at 6144 dims already triggered an OOM.
        if getattr(self.config, "audio_pooling", "mean") == "frames":
            self._cache_frame_features(encoder, device, batch_size)
            return

        cache_path = _encoder_cache_path(
            self.dataset_name,
            getattr(self.config, "audio_encoder_type", "unknown"),
            getattr(self.config, "audio_model_name", "unknown"),
            getattr(self.config, "audio_pooling", "mean"),
        )
        AUDIO_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        all_features = None
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                cached = payload["features"]
                if cached.shape[0] == len(self.data):
                    all_features = cached
                    print(f"  Loaded cached features for {self.dataset_name} "
                          f"from {cache_path} ({tuple(cached.shape)})")
                else:
                    print(f"  Cache size mismatch for {self.dataset_name} "
                          f"(cache {cached.shape[0]} vs data {len(self.data)}), "
                          f"recomputing.")
            except Exception as e:
                print(f"  Failed to load cache {cache_path}: {e}. Recomputing.")

        if all_features is None:
            print(f"  Caching {len(self.data)} encoder features for {self.dataset_name}...")
            encoder.eval()

            from data.collate import vad_collate_fn
            from torch.utils.data import DataLoader

            temp_loader = DataLoader(
                self, batch_size=batch_size, shuffle=False,
                collate_fn=vad_collate_fn, num_workers=0,
            )

            chunks = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(temp_loader):
                    waveforms = batch['waveforms'].to(device)
                    mask = batch['audio_attention_mask'].to(device)
                    feats = encoder(waveforms, mask)  # [B, 768]
                    chunks.append(feats.cpu())
                    if (batch_idx + 1) % 100 == 0:
                        print(f"    {(batch_idx+1)*batch_size}/{len(self.data)} cached...")

            all_features = torch.cat(chunks, dim=0)  # [N, 768]

            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(
                {
                    "features": all_features,
                    "n_samples": all_features.shape[0],
                    "encoder_type": getattr(self.config, "audio_encoder_type", "unknown"),
                    "model_name": getattr(self.config, "audio_model_name", "unknown"),
                    "dataset_name": self.dataset_name,
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
            print(f"  Cached {len(self.data)} features ({all_features.shape}) "
                  f"to {cache_path}")

        # Replace lazy loading with cached features
        for i, sample in enumerate(self.data):
            sample.pop("hf_idx", None)
            sample["features"] = all_features[i]

        self.uses_raw_audio = False
        self.hf_dataset = None  # Free HF dataset memory

    def _cache_frame_features(self, encoder, device, batch_size=16):
        """Cache frame-level features to a memmapped file.

        Each utterance is resampled to a fixed frame count so the array is
        rectangular and can be memory mapped. Only the rows a batch touches
        are ever resident, which is what makes keeping the time axis
        affordable at all.

        Args:
            encoder: audio encoder configured with pooling="frames".
            device: device to run the encoder on.
            batch_size: extraction batch size.
        """
        model_name = getattr(self.config, "audio_model_name", "unknown")
        n_frames = int(getattr(self.config, "num_frames", 32))
        n = len(self.data)

        existing = load_frame_cache(self.dataset_name, model_name, n_frames,
                                    expected_n=n)
        if existing is None:
            print(f"  Extracting {n} frame-level features for "
                  f"{self.dataset_name} (T={n_frames})...")
            arr, tmp_path, meta_path = write_frame_cache(
                self.dataset_name, model_name, n_frames, n,
            )
            durations = np.zeros(n, dtype=np.int32)

            from torch.utils.data import DataLoader
            from data.collate import vad_collate_fn
            loader = DataLoader(self, batch_size=batch_size, shuffle=False,
                                collate_fn=vad_collate_fn, num_workers=0)
            encoder.eval()
            pos = 0
            with torch.no_grad():
                for bi, batch in enumerate(loader):
                    feats = encoder(
                        batch["waveforms"].to(device),
                        batch["audio_attention_mask"].to(device),
                    )                                    # [B, T', D]
                    # Padding was zeroed by the encoder, so a row that is
                    # entirely zero is padding.
                    real = (feats.abs().sum(dim=-1) > 0).sum(dim=1)
                    for j in range(feats.shape[0]):
                        durations[pos] = int(real[j])
                        arr[pos] = resample_frames(
                            feats[j], int(real[j]), n_frames,
                        ).cpu().numpy().astype(np.float16)
                        pos += 1
                    if (bi + 1) % 100 == 0:
                        print(f"    {pos}/{n} extracted...")
            path = finalize_frame_cache(arr, tmp_path, meta_path,
                                        self.dataset_name, model_name,
                                        n_frames, durations)
            print(f"  Cached frame features to {path}")
            existing = load_frame_cache(self.dataset_name, model_name,
                                        n_frames, expected_n=n)

        self.frame_cache, self.frame_durations = existing
        print(f"  Frame cache ready: {self.frame_cache.shape} "
              f"(memmapped, fp16)")
        for i, sample in enumerate(self.data):
            sample.pop("hf_idx", None)
            sample["frame_idx"] = i
        self.uses_raw_audio = False
        self.hf_dataset = None

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
            "difficulty": item.get("difficulty", 0.0),
            "dataset": item["dataset"],
            "speaker": item.get("speaker", "unknown"),
            "label4": torch.tensor(item.get("label4", item["label"]), dtype=torch.long),
            # Position in self.data. Lets training-time trackers accumulate
            # per-sample statistics (e.g. an EMA of model confidence) across
            # epochs. Subset passes through the original index, so this stays
            # valid under a train/val split.
            "sample_index": idx,
        }

        # Frame-level: read one row from the memmap. Only this row is
        # paged in, so RAM stays flat regardless of corpus size.
        if (self.modality in ["audio", "both"]
                and getattr(self, "frame_cache", None) is not None
                and item.get("frame_idx") is not None):
            row = np.asarray(self.frame_cache[item["frame_idx"]])
            result["features"] = torch.from_numpy(row).float()

        # Add audio features (preextracted mode)
        elif self.modality in ["audio", "both"] and item.get("features") is not None:
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
        result = self.datasets[ds_idx][local_idx]
        # Overwrite the sub-dataset's local index with the global one so
        # per-sample trackers key on a unique id across corpora.
        result["sample_index"] = idx
        return result


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
