#!/usr/bin/env python3
"""Per-utterance trust weights for direction-aware soft labels.

Five annotators splitting three to two tells you the panel disagreed. It
does not tell you whether the utterance genuinely sits between two
emotions or whether the annotators were unreliable about that particular
distinction, because both read the same way. The direction of the split
separates them, given a map of which label confusions carry real structure
in which region of VAD space.

alpha is the share of a panel's own disagreement running along axes that
are real where that utterance sits. It drives two levers:

  target  = (1 - alpha) * onehot(majority) + alpha * annotator distribution
  weight  proportional to alpha

The first changes what the model is taught, the second how much. Both read
the same alpha, so an arm can enable either or both.

The map is fitted by analysis.build_realness_map on the official
MSP-Podcast Train split alone, boundaries included, so no validation row
influences a training target. Rows the map does not cover fall back to the
hard label at full weight, which is exactly how training behaves without
this module.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_PATH = pathlib.Path("analysis/realness_map/mspp_official_train.npz")
NUM_CLASSES = 4


class RealnessLookup:
    """Filename-keyed alpha, loaded once and shared across datasets.

    Holds flat arrays and an index rather than a dict of records, for the
    same reason SubtypeMeta does: an NpzFile re-inflates a whole array on
    every key access, so building per-row entries by indexing it costs
    orders of magnitude more than the file size suggests.
    """

    def __init__(self, path: pathlib.Path, use_placebo: bool = False,
                 gain: float = 1.0, shuffle: bool = False,
                 shuffle_seed: int = 12345) -> None:
        """Load the map's per-utterance columns.

        Args:
            path: npz written by analysis.build_realness_map.
            use_placebo: read the shuffled map's alpha instead of the real
                one. That placebo permutes each label pair's realness
                across VAD cells while holding the pair's corpus average
                fixed, so it removes only the claim that softness belongs
                in particular places. It deliberately preserves which
                label pair a panel split on, which is why it stays
                correlated with the real map and makes a weak control at
                one seed.
            gain: multiplier on alpha before clipping to [0, 1]. Mean alpha
                is 0.128 as fitted, so the targets sit close to the hard
                label and both levers are gentle. Raising the gain tests
                whether a null result is underpowering rather than the
                signal being absent.
            shuffle: permute alpha across covered utterances. This is the
                higher-power control: the distribution of alpha is
                preserved exactly, so the arm is matched on how soft the
                targets are AND on how much the weights vary, while the
                correlation with the real assignment goes to zero. It asks
                whether alpha carries any useful per-utterance information
                at all, rather than the narrower question of whether the
                spatial part specifically does.
            shuffle_seed: fixed so the permutation is one controlled
                assignment rather than noise that differs per run.
        """
        with np.load(path, allow_pickle=True) as z:
            keys = np.asarray(z["keys"]).astype(str)
            column = "alpha_placebo" if use_placebo else "alpha"
            self.alpha = np.asarray(z[column], dtype=np.float32)
            self.covered = np.asarray(z["covered"]).astype(bool)
            self.split = np.asarray(z["split"]).astype(str)
            self.probs = np.asarray(z["probs"], dtype=np.float32)

        if gain != 1.0:
            self.alpha = np.clip(self.alpha * float(gain), 0.0, 1.0
                                 ).astype(np.float32)
        if shuffle:
            # Permute within the covered rows only. Including uncovered
            # rows would move their zeros into covered positions and change
            # the distribution the control is supposed to match.
            idx = np.where(self.covered)[0]
            rng = np.random.RandomState(int(shuffle_seed))
            self.alpha[idx] = self.alpha[idx][rng.permutation(idx.size)]

        self.use_placebo = bool(use_placebo)
        self.gain = float(gain)
        self.shuffled = bool(shuffle)
        self.index: Dict[str, int] = {k: i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        """Number of utterances the map covers at all."""
        return len(self.index)

    def get(self, key: Optional[str]) -> Tuple[float, np.ndarray, float]:
        """Look up one utterance.

        Args:
            key: extension-stripped audio filename, or None.

        Returns:
            Tuple of alpha, the four-class annotator distribution, and a
            mask that is one when the map covers this row. An uncovered row
            returns alpha zero and a zero distribution, which the caller
            must read as "use the hard label at full weight" rather than as
            a measurement.
        """
        if key is None:
            return 0.0, np.zeros(NUM_CLASSES, dtype=np.float32), 0.0
        row = self.index.get(key)
        if row is None or not self.covered[row]:
            return 0.0, np.zeros(NUM_CLASSES, dtype=np.float32), 0.0
        return float(self.alpha[row]), self.probs[row], 1.0


_CACHE: Dict[Tuple[str, bool, float, bool], RealnessLookup] = {}


def load_realness(path: Optional[pathlib.Path] = None,
                  use_placebo: bool = False, gain: float = 1.0,
                  shuffle: bool = False) -> Optional[RealnessLookup]:
    """Load the lookup, caching one instance per configuration.

    Args:
        path: npz to read. Defaults to the official-Train map.
        use_placebo: read the cell-permuted map's alpha.
        gain: multiplier on alpha before clipping.
        shuffle: permute alpha across covered utterances.

    Returns:
        A RealnessLookup, or None when the file is absent, so a run
        configured without the map degrades to ordinary hard-label
        training instead of failing.
    """
    target = pathlib.Path(path or DEFAULT_PATH)
    cache_key = (str(target), bool(use_placebo), float(gain), bool(shuffle))
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if not target.exists():
        return None
    lookup = RealnessLookup(target, use_placebo=use_placebo, gain=gain,
                            shuffle=shuffle)
    _CACHE[cache_key] = lookup
    return lookup


def direction_aware_target(label: int, annot_probs: np.ndarray, alpha: float,
                           mask: float) -> np.ndarray:
    """Interpolate between the hard label and the annotator distribution.

    The hard label is built from the training label rather than from the
    annotator distribution's argmax, so that a row whose stored majority
    disagrees with the pipeline's own label still trains toward the label
    every other arm uses and the contrast stays about softness alone.

    Args:
        label: the integer training label.
        annot_probs: [C] four-class annotator distribution.
        alpha: trust to place in that distribution.
        mask: one when the map covers this row, zero otherwise.

    Returns:
        [C] target distribution summing to one.
    """
    hard = np.zeros(NUM_CLASSES, dtype=np.float32)
    hard[int(label)] = 1.0
    if mask <= 0.0 or alpha <= 0.0:
        return hard
    total = float(annot_probs.sum())
    if total <= 0.0:
        return hard
    soft = annot_probs.astype(np.float32) / total
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - a) * hard + a * soft


def raw_soft_target(label: int, annot_probs: np.ndarray,
                    mask: float) -> np.ndarray:
    """The annotator distribution itself, as the arm that trusts everything.

    This is the comparison the claim actually rests on. Direction-aware
    trust has to beat trusting every panel equally, not merely beat hard
    labels, or the result only says that soft labels help.

    Args:
        label: the integer training label, used as the fallback.
        annot_probs: [C] four-class annotator distribution.
        mask: one when the map covers this row, zero otherwise.

    Returns:
        [C] target distribution summing to one.
    """
    hard = np.zeros(NUM_CLASSES, dtype=np.float32)
    hard[int(label)] = 1.0
    if mask <= 0.0:
        return hard
    total = float(annot_probs.sum())
    if total <= 0.0:
        return hard
    return annot_probs.astype(np.float32) / total


def loss_weight(alpha: float, mask: float, floor: float,
                normalizer: float = 1.0) -> float:
    """Lever two: how much a row counts, given how real its split is.

    A panel whose disagreement runs along noise axes is evidence about the
    annotators rather than about the utterance, so it earns less influence.
    The floor keeps such a row contributing something, because its majority
    label is still the best guess available and dropping it outright would
    also drop whatever its audio teaches.

    Uncovered rows get weight one. They are not measured as noisy, merely
    unmeasured, and down-weighting them would confound missing metadata
    with a finding.

    Args:
        alpha: real-axis share of this panel's disagreement.
        mask: one when the map covers this row, zero otherwise.
        floor: smallest weight any covered row may take.
        normalizer: divisor keeping the mean weight near one, so the
            effective learning rate matches the unweighted arms.

    Returns:
        A non-negative weight.
    """
    if mask <= 0.0:
        return 1.0
    raw = floor + (1.0 - floor) * float(np.clip(alpha, 0.0, 1.0))
    return raw / normalizer if normalizer > 0 else raw
