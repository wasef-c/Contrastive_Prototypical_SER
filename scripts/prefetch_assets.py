#!/usr/bin/env python3
"""
Prefetch models and datasets for offline use on Compute Canada.

Run this on a LOGIN node (with internet). It downloads:
  1. HuggingFace text models (BERT, RoBERTa, DeBERTa)
  2. HuggingFace audio model (Wav2Vec2)
  3. ModelScope audio model (Emotion2Vec, via FunASR)
  4. HuggingFace datasets (cairocode/*)

Before running, point the cache directories at a persistent location
(usually $SCRATCH or $PROJECT on Compute Canada). Example:

    export HF_HOME=$SCRATCH/hf_cache
    export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
    export MODELSCOPE_CACHE=$SCRATCH/modelscope_cache
    python scripts/prefetch_assets.py

On compute nodes, export HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1
and keep the same cache variables pointing at the prefetched location.
"""

import argparse
import os
import sys
from typing import List


TEXT_MODELS: List[str] = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
]

AUDIO_HF_MODELS: List[str] = [
    "facebook/wav2vec2-base-960h",
]

FUNASR_MODELS: List[str] = [
    "iic/emotion2vec_base",
]

HF_DATASETS: List[str] = [
    "cairocode/IEMO_Audio_Text_Merged",
    "cairocode/MSPI_Audio_Text_Merged",
    "cairocode/MSPP_WAV",
    "cairocode/cmu_mosei_wav_2",
    "cairocode/samsemo-audio",
]


def prefetch_text_models(models: List[str]) -> None:
    """Download each HF text model and tokenizer into the HF cache.

    Args:
        models: List of HuggingFace model repo ids to download.
    """
    from transformers import AutoModel, AutoTokenizer

    for name in models:
        print(f"[text] {name}")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)


def prefetch_audio_hf_models(models: List[str]) -> None:
    """Download each HF audio model and processor into the HF cache.

    Args:
        models: List of HuggingFace Wav2Vec2 repo ids to download.
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    for name in models:
        print(f"[audio-hf] {name}")
        Wav2Vec2Processor.from_pretrained(name)
        Wav2Vec2Model.from_pretrained(name)


def prefetch_funasr_models(models: List[str]) -> None:
    """Download each FunASR model into the ModelScope cache.

    Args:
        models: List of ModelScope model ids (e.g. iic/emotion2vec_base).
    """
    from funasr import AutoModel

    for name in models:
        print(f"[funasr] {name}")
        AutoModel(model=name)


def prefetch_datasets(dataset_ids: List[str]) -> None:
    """Download each HuggingFace dataset into the HF datasets cache.

    Args:
        dataset_ids: List of HuggingFace dataset repo ids to download.
    """
    from datasets import load_dataset

    for repo_id in dataset_ids:
        print(f"[dataset] {repo_id}")
        load_dataset(repo_id, split="train", trust_remote_code=True)


def main() -> int:
    """Parse arguments and prefetch the selected asset groups.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-text", action="store_true", help="Skip HF text models")
    parser.add_argument("--skip-audio", action="store_true", help="Skip HF Wav2Vec2 models")
    parser.add_argument("--skip-funasr", action="store_true", help="Skip FunASR/Emotion2Vec models")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip HF datasets")
    args = parser.parse_args()

    print("Cache locations:")
    print(f"  HF_HOME           = {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    print(f"  HF_DATASETS_CACHE = {os.environ.get('HF_DATASETS_CACHE', '(default inside HF_HOME)')}")
    print(f"  MODELSCOPE_CACHE  = {os.environ.get('MODELSCOPE_CACHE', '(default ~/.cache/modelscope)')}")
    print()

    if not args.skip_text:
        prefetch_text_models(TEXT_MODELS)
    if not args.skip_audio:
        prefetch_audio_hf_models(AUDIO_HF_MODELS)
    if not args.skip_funasr:
        prefetch_funasr_models(FUNASR_MODELS)
    if not args.skip_datasets:
        prefetch_datasets(HF_DATASETS)

    print("\nPrefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
