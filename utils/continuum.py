#!/usr/bin/env python3
"""Continuous emotion targets: low-rank soft labels and a substitution kernel.

Two target families, both fitted on the official MSP-Podcast Train split
by analysis.build_continuum_targets.

LOW-RANK. The annotator label distribution projected onto the leading
principal components of that distribution and reconstructed. Components
beyond the third are not recoverable from audio on a class-balanced probe,
so the raw distribution spends part of every target on directions the
signal does not carry. Per utterance, so it needs the panel data and
reaches the rows the cache covers.

KERNEL. Given that one annotator of an utterance chose a, how often
another annotator of the same utterance chose b. A measured conditional
with nothing to tune, identical for every utterance of a class, so it
applies to all rows including unanimous ones and transfers to corpora with
no panel data at all.

Both degrade to the hard label when the file is missing or a row is
uncovered, so a run configured for them still trains if the targets have
not been built.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_PATH = pathlib.Path("analysis/continuum/mspp_continuum.npz")
NUM_CLASSES = 4


class ContinuumTargets:
    """Filename-keyed continuum targets, loaded once and shared.

    Flat arrays with an index rather than per-row records, for the reason
    the annotator cache gives: an NpzFile re-inflates a whole array on
    every key access, so building per-row entries by indexing it costs far
    more than the file size suggests.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Load the per-utterance targets and the kernel.

        Args:
            path: npz written by analysis.build_continuum_targets.
        """
        with np.load(path, allow_pickle=True) as z:
            keys = np.asarray(z["keys"]).astype(str)
            self.lowrank = np.asarray(z["lowrank_target"], dtype=np.float32)
            self.raw = np.asarray(z["raw_target"], dtype=np.float32)
            self.smooth = np.asarray(z["smooth_target"], dtype=np.float32)
            self.pc_coords = np.asarray(z["pc_coords"], dtype=np.float32)
            self.kernel = np.asarray(z["kernel"], dtype=np.float32)
            self.ot_cost = np.asarray(z["ot_cost"], dtype=np.float32)
            self.split = np.asarray(z["split"]).astype(str)
        self.index: Dict[str, int] = {k: i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        """Number of utterances covered."""
        return len(self.index)

    def get(self, key: Optional[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Look up one utterance.

        Args:
            key: extension-stripped audio filename, or None.

        Returns:
            Tuple of the [C] low-rank soft target, the [3] PC coordinates,
            and a mask that is one when the utterance is covered. An
            uncovered row returns zeros, which the caller must read as
            "fall back to the hard label" rather than as a measurement.
        """
        if key is None:
            return (np.zeros(NUM_CLASSES, dtype=np.float32),
                    np.zeros(3, dtype=np.float32), 0.0)
        row = self.index.get(key)
        if row is None:
            return (np.zeros(NUM_CLASSES, dtype=np.float32),
                    np.zeros(3, dtype=np.float32), 0.0)
        return self.lowrank[row], self.pc_coords[row], 1.0

    def smooth_row(self, key: Optional[str]) -> Tuple[np.ndarray, float]:
        """The entropy-matched smoothing target for one utterance.

        Args:
            key: extension-stripped audio filename, or None.

        Returns:
            Tuple of the [C] target and a coverage mask.
        """
        if key is None:
            return np.zeros(NUM_CLASSES, dtype=np.float32), 0.0
        row = self.index.get(key)
        if row is None:
            return np.zeros(NUM_CLASSES, dtype=np.float32), 0.0
        return self.smooth[row], 1.0

    def kernel_target(self, label: int) -> np.ndarray:
        """The smoothing target for a hard label.

        Args:
            label: integer class id.

        Returns:
            [C] distribution over classes, summing to one.
        """
        return self.kernel[int(label)]


_CACHE: Dict[str, ContinuumTargets] = {}


def load_continuum(path: Optional[pathlib.Path] = None
                   ) -> Optional[ContinuumTargets]:
    """Load the targets, caching one instance per path.

    Args:
        path: npz to read. Defaults to the official-Train build.

    Returns:
        A ContinuumTargets, or None when the file is absent.
    """
    target = pathlib.Path(path or DEFAULT_PATH)
    if str(target) in _CACHE:
        return _CACHE[str(target)]
    if not target.exists():
        return None
    loaded = ContinuumTargets(target)
    _CACHE[str(target)] = loaded
    return loaded
