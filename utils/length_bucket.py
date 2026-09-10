#!/usr/bin/env python3
"""Length-bucketed batching for raw-audio (unfrozen encoder) runs.

The collate function pads every waveform in a batch to the batch's longest, so
the compute for a batch is `batch_size * max_length_in_batch` and the cost of
an epoch is `n_samples * E[max length in batch]`. That expectation grows with
batch size, which is why larger batches were measured to be SLOWER per epoch
here rather than faster: MSP-Podcast durations average 5.5s against a 10s cap,
so a batch of 24 almost always contains a 10s utterance and pays full length
for all 24, while a batch of 8 often does not.

Grouping utterances of similar length into the same batch removes that waste.
Batches are still formed stochastically: indices are shuffled, cut into windows
of several batches, sorted only inside a window, and the resulting batches are
shuffled again. So composition varies across epochs while lengths within a
batch stay close.

Only relevant when the audio encoder is unfrozen. Cached-feature runs have
fixed-width frames and no padding.
"""

from __future__ import annotations

import io
import pathlib
from typing import Iterator, List, Optional, Sequence

import numpy as np


DURATION_CACHE_DIR = pathlib.Path("cache/durations")


def load_or_compute_durations(dataset_name: str, hf_dataset,
                              cache_dir: Optional[pathlib.Path] = None
                              ) -> np.ndarray:
    """Seconds per row, read from audio headers and cached to disk.

    Reading the header with soundfile costs about 8 ms per row and avoids
    decoding the samples, but 80k rows is still ten minutes, so the result is
    cached. The cache is keyed by dataset name and row count, so a corpus whose
    size changes is recomputed rather than silently mismatched.

    Args:
        dataset_name: corpus name, used in the cache filename.
        hf_dataset: the loaded HF dataset, audio column still encoded.
        cache_dir: override for the cache location.

    Returns:
        [n] float32 durations in seconds.
    """
    import soundfile as sf
    from datasets import Audio

    cache_dir = cache_dir or DURATION_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(hf_dataset)
    path = cache_dir / f"{dataset_name}__n{n}.npy"
    if path.exists():
        return np.load(path)

    raw = hf_dataset.cast_column("audio", Audio(decode=False))
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        try:
            info = sf.info(io.BytesIO(raw[i]["audio"]["bytes"]))
            out[i] = info.frames / float(info.samplerate)
        except Exception:
            # A row we cannot read gets the mean so it does not dominate a
            # bucket; it is rare and the batch is still valid.
            out[i] = 0.0
    if (out == 0).any():
        out[out == 0] = float(out[out > 0].mean()) if (out > 0).any() else 1.0
    np.save(path, out)
    return out


class LengthBucketedBatchSampler:
    """Yield batches whose members have similar duration.

    Args:
        durations: [n] duration per index of the underlying dataset.
        indices: the subset of dataset indices to sample from, in dataset
            coordinates. Positions in `durations` must correspond.
        batch_size: samples per batch.
        window_batches: how many batches are sorted together. Larger means
            tighter length grouping and less randomness; 50 keeps padding
            waste low while leaving batch composition varied.
        drop_last: discard a trailing short batch, matching the DataLoader
            setting the training loop uses.
        seed: base RNG seed; the epoch is mixed in via set_epoch.
    """

    def __init__(self, durations: Sequence[float], indices: Sequence[int],
                 batch_size: int, window_batches: int = 50,
                 drop_last: bool = True, seed: int = 42):
        self.durations = np.asarray(durations, dtype=np.float32)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.window = int(window_batches) * int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Vary batch composition across epochs."""
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        order = self.indices.copy()
        rng.shuffle(order)

        batches: List[List[int]] = []
        for start in range(0, len(order), self.window):
            window = order[start:start + self.window]
            # Sort only inside the window, so lengths are close within a batch
            # but the global order stays stochastic.
            window = window[np.argsort(self.durations[window], kind="stable")]
            for b in range(0, len(window), self.batch_size):
                batch = window[b:b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append([int(x) for x in batch])
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


def padding_waste(durations: Sequence[float], batch_size: int,
                  bucketed: bool, seed: int = 0,
                  window_batches: int = 50) -> float:
    """Fraction of padded compute that is padding, for a given batching scheme.

    Args:
        durations: per-sample durations.
        batch_size: samples per batch.
        bucketed: whether to bucket by length or batch at random.
        seed: RNG seed.
        window_batches: window size when bucketing.

    Returns:
        Wasted fraction in [0, 1].
    """
    d = np.asarray(durations, dtype=np.float64)
    rng = np.random.RandomState(seed)
    order = np.arange(len(d))
    rng.shuffle(order)
    if bucketed:
        win = window_batches * batch_size
        chunks = [order[i:i + win] for i in range(0, len(order), win)]
        order = np.concatenate([c[np.argsort(d[c], kind="stable")]
                                for c in chunks])
    real = padded = 0.0
    for b in range(0, len(order) - batch_size + 1, batch_size):
        batch = d[order[b:b + batch_size]]
        real += batch.sum()
        padded += batch.max() * len(batch)
    return 1.0 - real / padded
