#!/usr/bin/env python3
"""
Memory-mapped frame-level feature cache.

The pooled caches store one vector per utterance and are loaded whole
into RAM. That works up to about 3072 dimensions; at 6144 the dup8 arm
was killed by earlyoom with a dataloader worker holding 13.9 GB. Keeping
the time axis is strictly larger again, so the whole-tensor approach does
not extend to it.

Memory mapping removes the problem. Frames are written once to a .npy
file in fp16 and read back with mmap_mode="r", so only the pages a batch
touches are resident. A 32-frame fp16 cache over all five corpora is
about 4.9 GB on disk and under a megabyte per batch in RAM.

Frames are resampled to a fixed count per utterance rather than padded to
the longest. Emotion2vec emits roughly 50 frames per second, so a 15
second cap is 750 frames and most utterances are far shorter; padding to
the maximum would waste most of the file on zeros and make the mask do
all the work. Uniform resampling to a fixed T keeps every row the same
width, keeps the array rectangular for mmap, and preserves the *shape* of
the trajectory even though it discards absolute duration. Duration is
recorded separately in case it matters later.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


FRAME_CACHE_DIR = Path(
    os.environ.get("FRAME_FEATURE_CACHE_DIR", ".flash/frame_feature_cache")
)


def frame_cache_paths(
    dataset_name: str,
    model_name: str,
    num_frames: int,
) -> Tuple[Path, Path]:
    """Return the (features, metadata) paths for a frame cache.

    Args:
        dataset_name: e.g. "MSPP".
        model_name: encoder id, e.g. "iic/emotion2vec_base".
        num_frames: fixed frames per utterance.

    Returns:
        (path to the .npy feature file, path to the .npz metadata file).
    """
    slug = model_name.replace("/", "__").replace(":", "_")
    stem = f"{dataset_name}__{slug}__frames{num_frames}"
    return (
        (FRAME_CACHE_DIR / f"{stem}.npy").resolve(),
        (FRAME_CACHE_DIR / f"{stem}__meta.npz").resolve(),
    )


def resample_frames(
    x: torch.Tensor,
    valid_len: int,
    num_frames: int,
) -> torch.Tensor:
    """Resample one utterance's real frames to a fixed count.

    Uniform index sampling across the real frames only. Utterances shorter
    than num_frames repeat frames rather than pad with zeros, so every row
    is genuinely populated and the model never has to learn to ignore
    padding inside the sequence.

    Args:
        x: [T, D] frame features for one utterance, may include padding.
        valid_len: number of real frames at the start of x.
        num_frames: target frame count.

    Returns:
        [num_frames, D] resampled features.
    """
    valid_len = max(int(valid_len), 1)
    idx = torch.linspace(0, valid_len - 1, num_frames, device=x.device)
    return x[idx.round().long().clamp(0, valid_len - 1)]


def write_frame_cache(
    dataset_name: str,
    model_name: str,
    num_frames: int,
    num_samples: int,
    feature_dim: int = 768,
) -> Tuple[np.memmap, Path, Path]:
    """Open a writable memmap for a new frame cache.

    Writing through a memmap means the array never has to exist in RAM in
    full, which matters at these sizes.

    Args:
        dataset_name: corpus name.
        model_name: encoder id.
        num_frames: fixed frames per utterance.
        num_samples: number of utterances.
        feature_dim: frame feature width.

    Returns:
        (writable memmap of shape [N, T, D], feature path, meta path).
    """
    feat_path, meta_path = frame_cache_paths(dataset_name, model_name, num_frames)
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = feat_path.with_suffix(".npy.tmp")
    arr = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float16,
        shape=(num_samples, num_frames, feature_dim),
    )
    return arr, tmp, meta_path


def finalize_frame_cache(
    arr: np.memmap,
    tmp_path: Path,
    meta_path: Path,
    dataset_name: str,
    model_name: str,
    num_frames: int,
    durations: np.ndarray,
) -> Path:
    """Flush a written cache and move it into place atomically.

    Args:
        arr: the writable memmap.
        tmp_path: temporary .npy path being written.
        meta_path: destination for the metadata sidecar.
        dataset_name: corpus name.
        model_name: encoder id.
        num_frames: fixed frames per utterance.
        durations: [N] real frame counts before resampling.

    Returns:
        Final feature path.
    """
    arr.flush()
    del arr
    feat_path, _ = frame_cache_paths(dataset_name, model_name, num_frames)
    os.replace(tmp_path, feat_path)
    np.savez(
        meta_path,
        durations=durations.astype(np.int32),
        num_frames=np.array([num_frames]),
        dataset=np.array([dataset_name]),
    )
    return feat_path


def load_frame_cache(
    dataset_name: str,
    model_name: str,
    num_frames: int,
    expected_n: Optional[int] = None,
) -> Optional[Tuple[np.memmap, np.ndarray]]:
    """Open an existing frame cache read-only.

    Args:
        dataset_name: corpus name.
        model_name: encoder id.
        num_frames: fixed frames per utterance.
        expected_n: if given, validates the row count.

    Returns:
        (read-only memmap [N, T, D], durations [N]), or None when the cache
        is absent or the row count disagrees.
    """
    feat_path, meta_path = frame_cache_paths(dataset_name, model_name, num_frames)
    if not feat_path.exists() or not meta_path.exists():
        return None
    arr = np.load(feat_path, mmap_mode="r")
    meta = np.load(meta_path)
    if expected_n is not None and arr.shape[0] != expected_n:
        print(f"  {dataset_name}: frame cache has {arr.shape[0]} rows but "
              f"{expected_n} were expected; ignoring it.")
        return None
    return arr, meta["durations"]
