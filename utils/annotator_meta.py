#!/usr/bin/env python3
"""Per-sample annotator agreement and VAD dispersion for MSP-Podcast.

Two uses, both aimed at the auxiliary prototypicality head.

The head currently regresses a whitened residual computed from VAD values that
are averages over annotators. On a contested utterance that average is a
summary of disagreement rather than a measurement, so the target is noise and
the head is asked to fit it anyway. Weighting the auxiliary loss by agreement
removes that.

Separately, the residual is currently whitened by the CLASS covariance, which
asks "is this sample far relative to how much the class varies". Whitening by
the per-sample annotator dispersion instead asks "is it far relative to how
precisely this particular utterance was measured", which is a different and
arguably more correct question. Measured on MSP-Podcast the two notions of
unusual are nearly orthogonal: our whitened residual correlates with
overall_agreement at only +0.17.

The metadata lives in a different HF dataset revision from the one training
uses, so it is joined by audio filename and cached as an npz keyed by that
filename.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

META_CACHE = pathlib.Path("cache/annotator_meta")
META_SOURCE = "cairocode/MSPP_WAV_Filtered_ordered_v2"


def build_meta_cache(source: str = META_SOURCE,
                     cache_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Read agreement and VAD dispersion from the source set and cache them.

    Args:
        source: HF dataset id carrying overall_agreement and the *_std columns.
        cache_dir: where to write the npz.

    Returns:
        Path to the cached npz.
    """
    from datasets import load_dataset

    cache_dir = cache_dir or META_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{source.replace('/', '__')}.npz"
    if out.exists():
        return out

    ds = load_dataset(source, split="train", trust_remote_code=True)
    ds = ds.select_columns(["FileName", "overall_agreement", "valence_std",
                            "arousal_std", "dominance_std"])
    keys = np.array([os.path.basename(str(f)).rsplit(".", 1)[0]
                     for f in ds["FileName"]])
    np.savez_compressed(
        out,
        keys=keys,
        agreement=np.asarray(ds["overall_agreement"], dtype=np.float32),
        vad_std=np.stack([np.asarray(ds["valence_std"], dtype=np.float32),
                          np.asarray(ds["arousal_std"], dtype=np.float32),
                          np.asarray(ds["dominance_std"], dtype=np.float32)],
                         axis=1),
    )
    return out


def load_meta(cache_path: Optional[pathlib.Path] = None) -> Dict[str, tuple]:
    """Load the cached metadata into a filename -> (agreement, vad_std) map.

    Args:
        cache_path: npz written by build_meta_cache.

    Returns:
        Dict keyed by extension-stripped filename.
    """
    path = cache_path or (META_CACHE /
                          f"{META_SOURCE.replace('/', '__')}.npz")
    if not pathlib.Path(path).exists():
        return {}
    z = np.load(path, allow_pickle=True)
    keys, agr, std = z["keys"], z["agreement"], z["vad_std"]
    return {str(k): (float(a), s) for k, a, s in zip(keys, agr, std)}


def sample_whitened_residual(vad: np.ndarray, labels: np.ndarray,
                             means: np.ndarray, vad_std: np.ndarray,
                             floor: float = 0.25) -> np.ndarray:
    """Residual from the class centre, scaled by per-sample annotator spread.

    Class-covariance whitening asks whether a sample is far relative to how
    much its CLASS varies. This asks whether it is far relative to how
    precisely THIS utterance was measured, so an outlier that every annotator
    agreed on counts as more genuinely atypical than one they argued about.

    Args:
        vad: [N, 3] per-sample VAD.
        labels: [N] class ids.
        means: [C, 3] class centres.
        vad_std: [N, 3] annotator standard deviation per dimension.
        floor: smallest usable std. Some utterances have zero dispersion,
            usually a single annotator, and dividing by that would send the
            target to infinity.

    Returns:
        [N, 3] residuals.
    """
    resid = vad - means[labels]
    scale = np.maximum(np.asarray(vad_std, dtype=np.float64), floor)
    return resid / scale


# Local parquet build of MSP-Podcast carrying the per-annotator statistics.
# Read directly rather than through load_dataset so that no 20 GB copy is
# regenerated into the HuggingFace cache just to reach the metadata columns.
MSPP_BUILD_DIR = pathlib.Path("/mnt/fast/mspp_build/data")
SUBTYPE_CACHE = META_CACHE / "mspp_build_subtypes.npz"

# Ordering of subtype_dist in the build. Stored here so the aux head's output
# dimensions have stable, human-readable names in logs and plots.
SUBTYPE_NAMES: List[str] = [
    "Neutral", "Happy", "Concerned", "Frustrated", "Angry", "Sad", "Amused",
    "Annoyed", "Excited", "Disappointed", "Contempt", "Surprise", "Disgust",
    "Confused", "Depressed", "Fear",
]

# Ordering of label_dist in the build.
PRIMARY_NAMES: List[str] = [
    "Neutral", "Happy", "Sad", "Angry", "Contempt", "Surprise", "Disgust",
    "Fear", "Other",
]


def build_subtype_cache(build_dir: Optional[pathlib.Path] = None,
                        cache_path: Optional[pathlib.Path] = None,
                        force: bool = False) -> pathlib.Path:
    """Cache the per-annotator distributions from the local MSPP build.

    Pulls only the metadata columns out of the parquet shards, so the audio
    payload is never decoded and the working set stays small.

    Args:
        build_dir: directory holding the build's parquet shards.
        cache_path: destination npz. Defaults to SUBTYPE_CACHE.
        force: rebuild even if the cache already exists.

    Returns:
        Path to the cached npz.
    """
    build_dir = pathlib.Path(build_dir or MSPP_BUILD_DIR)
    out = pathlib.Path(cache_path or SUBTYPE_CACHE)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(str(p) for p in build_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {build_dir}")

    table = pq.read_table(shards, columns=[
        "FileName", "EmoClass", "label", "n_annotators", "label_dist",
        "annot_entropy", "annot_vad_mean", "annot_vad_std", "subtype_dist",
    ])
    keys = np.array([os.path.basename(str(f)).rsplit(".", 1)[0]
                     for f in table.column("FileName").to_pylist()])
    np.savez_compressed(
        out,
        keys=keys,
        emo_class=np.array(table.column("EmoClass").to_pylist()),
        label=np.asarray(table.column("label").to_pylist(), dtype=np.int64),
        n_annotators=np.asarray(table.column("n_annotators").to_pylist(),
                                dtype=np.int32),
        label_dist=np.asarray(table.column("label_dist").to_pylist(),
                              dtype=np.float32),
        entropy=np.asarray(table.column("annot_entropy").to_pylist(),
                           dtype=np.float32),
        vad_mean=np.asarray(table.column("annot_vad_mean").to_pylist(),
                            dtype=np.float32),
        vad_std=np.asarray(table.column("annot_vad_std").to_pylist(),
                           dtype=np.float32),
        subtype_dist=np.asarray(table.column("subtype_dist").to_pylist(),
                                dtype=np.float32),
    )
    return out


class SubtypeMeta:
    """Column-oriented view of the cached MSP-Podcast annotator statistics.

    Holds one array per field and a filename -> row index map. Deliberately
    not a dict of per-row dicts: np.load returns a lazy NpzFile whose every
    key access re-inflates the whole array from the zip member, so building
    116k rows by indexing the NpzFile costs hours of decompression rather
    than the seconds the file size suggests. The arrays here are materialised
    exactly once.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Load every field into memory once and index it by filename.

        Args:
            path: npz written by build_subtype_cache.
        """
        with np.load(path, allow_pickle=True) as z:
            keys = z["keys"]
            self.label_dist = np.asarray(z["label_dist"], dtype=np.float32)
            self.subtype_dist = np.asarray(z["subtype_dist"], dtype=np.float32)
            self.vad_mean = np.asarray(z["vad_mean"], dtype=np.float32)
            self.vad_std = np.asarray(z["vad_std"], dtype=np.float32)
            self.entropy = np.asarray(z["entropy"], dtype=np.float32)
            self.n_annotators = np.asarray(z["n_annotators"], dtype=np.int32)
            self.label = np.asarray(z["label"], dtype=np.int64)
        self.index: Dict[str, int] = {str(k): i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        """Number of utterances in the cache."""
        return len(self.index)

    def row(self, key: str) -> Optional[int]:
        """Row index for an extension-stripped filename, or None if absent.

        Args:
            key: filename without directory or extension.

        Returns:
            Integer row index, or None when the utterance is not covered.
        """
        return self.index.get(key)


def load_subtype_meta(cache_path: Optional[pathlib.Path] = None
                      ) -> Optional[SubtypeMeta]:
    """Load the cached build metadata.

    Args:
        cache_path: npz written by build_subtype_cache.

    Returns:
        A SubtypeMeta, or None if the cache is absent.
    """
    path = pathlib.Path(cache_path or SUBTYPE_CACHE)
    if not path.exists():
        return None
    return SubtypeMeta(path)


def agreement_weights(vad_std: np.ndarray, floor: float = 0.25,
                      normalize: bool = True) -> np.ndarray:
    """Per-sample auxiliary loss weights from annotator VAD dispersion.

    An utterance the annotators agreed on carries a VAD target that is a
    measurement; one they argued over carries a target that is a summary of
    the argument. Weighting by 1 / (1 + mean std) lets the auxiliary head
    spend its capacity on the former.

    Args:
        vad_std: [N, 3] per-dimension annotator standard deviation.
        floor: smallest usable std, guarding the 4.4 percent of rows whose
            annotators were unanimous and whose std is therefore exactly zero.
        normalize: rescale so the weights average to one, keeping the
            effective auxiliary loss magnitude comparable to the unweighted
            arm rather than silently shrinking it.

    Returns:
        [N] non-negative weights.
    """
    spread = np.mean(np.maximum(np.asarray(vad_std, dtype=np.float64), floor),
                     axis=1)
    weights = 1.0 / (1.0 + spread)
    if normalize and weights.mean() > 0:
        weights = weights / weights.mean()
    return weights
