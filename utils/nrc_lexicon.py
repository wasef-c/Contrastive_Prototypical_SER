#!/usr/bin/env python3
"""
Lexical prototypicality from the NRC word-emotion association lexicon.

Why this exists. The proto/atyp split in utils/proto_atyp.py measures how
far a sample sits from its class center in VAD space. The prototypicality
analysis (scripts/analyze_prototypicality.py) showed that measure does not
survive a change of corpus: predicting the VAD-defined typical/atypical
split from audio gives a cross-corpus AUC of 0.501 to 0.562, which is
chance to six points above it. It also showed that the theoretical and
class-mean center definitions correlate at rho = 0.88, so they were one
measure rather than two.

The lexical measure here is a different quantity, not a better estimate of
the same one. Its correlation with the VAD measure is rho = -0.02, so the
two name almost entirely different samples as atypical. It transfers
better, though still modestly: cross-corpus AUC 0.561 on average and 0.602
on happy, against 0.521 for the VAD class-mean split.

How it works. Each utterance is scored over the ten NRC categories by
counting affect words in its transcript and dividing by the number of
matched tokens, so the profile describes the balance of affect present
rather than how affect-dense the utterance is. A class center is the mean
profile over that class's training samples, and prototypicality is
Euclidean distance from it, exactly mirroring the VAD path.

Utterances containing no lexicon words have no defined profile. They are
assigned is_atypical = 0 (prototype), matching the convention already used
for samples from corpora without real VAD. Expect this to fire most often
on short neutral utterances.

The lexicon ships with the `nrclex` package (pip install nrclex).
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


NRC_CATEGORIES: List[str] = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]

NRC_DIM = len(NRC_CATEGORIES)

_WORD_RE = re.compile(r"[a-z']+")


@lru_cache(maxsize=1)
def load_lexicon() -> Dict[str, tuple]:
    """Load the NRC word-emotion association lexicon.

    Cached so repeated calls during training are free.

    Returns:
        Mapping from lowercase word to a tuple of category indices into
        NRC_CATEGORIES.

    Raises:
        RuntimeError: if nrclex is not installed or its data file is absent.
    """
    try:
        import nrclex
    except ImportError as exc:
        raise RuntimeError(
            "proto_atyp_center_source='nrc' needs the NRC lexicon. "
            "Install it with: pip install nrclex"
        ) from exc

    data_path = Path(nrclex.__file__).parent / "data" / "nrc_en.json"
    if not data_path.exists():
        raise RuntimeError(f"NRC lexicon not found at {data_path}")

    with open(data_path) as handle:
        raw = json.load(handle)

    index = {cat: i for i, cat in enumerate(NRC_CATEGORIES)}
    lexicon = {}
    for word, cats in raw.items():
        idxs = tuple(index[c] for c in cats if c in index)
        if idxs:
            lexicon[word] = idxs
    return lexicon


def profile_for_text(text: str, lexicon: Dict[str, tuple]) -> Optional[np.ndarray]:
    """Score one transcript over the NRC categories.

    Args:
        text: transcript string.
        lexicon: mapping from load_lexicon().

    Returns:
        [NRC_DIM] float32 profile summing to 1 over matched tokens, or None
        when the transcript contains no lexicon words.
    """
    counts = np.zeros(NRC_DIM, dtype=np.float32)
    hits = 0
    for word in _WORD_RE.findall((text or "").lower()):
        idxs = lexicon.get(word)
        if not idxs:
            continue
        hits += 1
        for i in idxs:
            counts[i] += 1.0
    if hits == 0:
        return None
    return counts / hits


def profiles_for_texts(
    texts: Sequence[str],
    lexicon: Dict[str, tuple],
) -> tuple:
    """Score a batch of transcripts.

    Args:
        texts: transcript strings.
        lexicon: mapping from load_lexicon().

    Returns:
        (profiles [N, NRC_DIM] float32, has_profile [N] bool). Rows without
        lexicon hits are zero and flagged False in has_profile.
    """
    out = np.zeros((len(texts), NRC_DIM), dtype=np.float32)
    has = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        prof = profile_for_text(text, lexicon)
        if prof is not None:
            out[i] = prof
            has[i] = True
    return out, has


def build_nrc_class_centers(
    train_data: Sequence[dict],
    num_classes: int,
) -> np.ndarray:
    """Mean NRC profile per class over the training split.

    Classes with no scoreable samples fall back to a uniform profile so
    downstream distance computations stay finite.

    Args:
        train_data: list of dataset item dicts with 'label' and 'transcript'.
        num_classes: number of primary classes.

    Returns:
        [num_classes, NRC_DIM] float32 centers.
    """
    lexicon = load_lexicon()
    sums = np.zeros((num_classes, NRC_DIM), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    for item in train_data:
        c = int(item.get("label", -1))
        if c < 0 or c >= num_classes:
            continue
        prof = profile_for_text(item.get("transcript", ""), lexicon)
        if prof is None:
            continue
        sums[c] += prof
        counts[c] += 1

    centers = np.full((num_classes, NRC_DIM), 1.0 / NRC_DIM, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            centers[c] = (sums[c] / counts[c]).astype(np.float32)
    return centers


def compute_nrc_thresholds(
    train_data: Sequence[dict],
    centers: np.ndarray,
    num_classes: int,
    criterion: str,
) -> np.ndarray:
    """Distance thresholds separating typical from atypical.

    Args:
        train_data: list of dataset item dicts.
        centers: [num_classes, NRC_DIM] class centers.
        num_classes: number of primary classes.
        criterion: "per_class_median" or "global_median".

    Returns:
        [num_classes] float32 thresholds, broadcast for global_median so
        callers always index by label.

    Raises:
        ValueError: if criterion is not recognized.
    """
    if criterion not in ("per_class_median", "global_median"):
        raise ValueError(f"Unknown proto_atyp_split_criterion: {criterion}")

    lexicon = load_lexicon()
    per_class: List[List[float]] = [[] for _ in range(num_classes)]

    for item in train_data:
        c = int(item.get("label", -1))
        if c < 0 or c >= num_classes:
            continue
        prof = profile_for_text(item.get("transcript", ""), lexicon)
        if prof is None:
            continue
        diff = prof - centers[c]
        per_class[c].append(float(np.sqrt(np.sum(diff * diff))))

    if criterion == "per_class_median":
        thresholds = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            thresholds[c] = float(np.median(per_class[c])) if per_class[c] else 0.0
        return thresholds

    pooled = [d for dists in per_class for d in dists]
    value = float(np.median(pooled)) if pooled else 0.0
    return np.full(num_classes, value, dtype=np.float32)


def batch_nrc_sub_labels(
    batch: dict,
    centers: torch.Tensor,
    thresholds: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute proto/atyp sub-labels for a batch from its transcripts.

    sub = 2 * class + is_atypical, where is_atypical marks a lexical profile
    further from the class center than that class's threshold. Utterances
    with no affect words are treated as prototypical, mirroring how the VAD
    path treats samples from corpora without real VAD.

    Profiles are computed on the fly rather than cached: a batch of 16
    utterances is a few hundred dictionary lookups, which is negligible
    against the forward pass.

    Args:
        batch: collated batch dict with 'label' and 'transcript'.
        centers: [num_classes, NRC_DIM] float tensor on `device`.
        thresholds: [num_classes] float tensor on `device`.
        num_classes: number of primary classes.
        device: torch device for output tensors.

    Returns:
        [B] LongTensor of sub-labels on `device`.
    """
    labels = batch["label"].to(device).long()
    texts = batch.get("transcript", None)
    if texts is None:
        return labels * 2

    lexicon = load_lexicon()
    profiles, has = profiles_for_texts(list(texts), lexicon)

    prof_t = torch.from_numpy(profiles).to(device).float()      # [B, NRC_DIM]
    has_t = torch.from_numpy(has.astype(np.int64)).to(device)   # [B]

    expected = centers[labels]                                   # [B, NRC_DIM]
    distance = torch.sqrt(((prof_t - expected) ** 2).sum(dim=1))
    is_atypical = (distance > thresholds[labels]).long() * has_t

    return labels * 2 + is_atypical
