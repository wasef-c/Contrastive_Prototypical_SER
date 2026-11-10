"""VAD-matched triplets: loading the assignments and batching them.

The assignments are built offline by analysis/build_triplets.py, which
emits two arms keyed by audio filename:

  matched  the negative has nearly the same VAD as the anchor but the
           confusable categorical label, so the only way to separate them
           is a cue VAD does not carry
  random   the same multiset of negative clips, reassigned by a shuffle
           within direction, so the term is present with the matching
           removed

Two implementation notes that matter for interpreting a result.

TRIPLETS ARE SERVED SEPARATELY, NOT PACKED INTO THE MAIN BATCH. Packing
them would make the main batch composition a function of the arm, and
batch composition is already known in this project to move a target by
roughly six times what a mechanism does (see the Global Mixup note in
configs/class_weight.yaml). A separate loader on a duty cycle leaves the
classification stream byte-identical between the two arms.

MEMBERS MISSING FROM THE TRAINING POOL DROP THE WHOLE TRIPLET. The _s9
arms subsample the training pool to nine tenths, and a triplet is only
usable if all three clips survived. The survivor count is returned so it
can be logged rather than silently absorbed.
"""

import pathlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

TRIPLET_PATH = pathlib.Path("analysis/triplets/mspp_triplets.npz")


def load_triplet_indices(
    key_to_pos: Dict[str, int],
    mode: str = "matched",
    path: Optional[pathlib.Path] = None,
) -> Tuple[np.ndarray, int]:
    """Resolve stored filename keys to positions in the training subset.

    Args:
        key_to_pos: audio filename key to its index in the training
            Subset, covering only rows that survived subsampling.
        mode: "matched" or "random".
        path: override for the assignment file.

    Returns:
        (triples, n_stored) where triples is an (N, 3) int64 array of
        subset positions in anchor, positive, negative order, and
        n_stored is how many triplets the file held before resolution.

    Raises:
        FileNotFoundError: if the assignment file is absent.
        KeyError: if mode is not one of the stored arms.
    """
    src = pathlib.Path(path) if path is not None else TRIPLET_PATH
    if not src.exists():
        raise FileNotFoundError(
            "triplet assignments not found at %s; run "
            "python -m analysis.build_triplets first" % src)
    data = np.load(src, allow_pickle=True)
    if mode not in data:
        raise KeyError("triplet mode %r not in %s (have %s)"
                       % (mode, src, sorted(data.files)))
    stored = data[mode]
    out: List[Tuple[int, int, int]] = []
    for row in stored:
        a, p, n = (str(row[0]), str(row[1]), str(row[2]))
        if a in key_to_pos and p in key_to_pos and n in key_to_pos:
            out.append((key_to_pos[a], key_to_pos[p], key_to_pos[n]))
    triples = (np.asarray(out, dtype=np.int64) if out
               else np.zeros((0, 3), dtype=np.int64))
    return triples, int(len(stored))


class TripletBatchSampler:
    """Yields flat index groups of anchor, positive, negative runs.

    Each yielded batch is a list of 3 * batch_triplets indices into the
    training Subset, ordered anchor, positive, negative, anchor, ... so a
    single collate call produces one tensor whose embeddings reshape to
    (batch_triplets, 3, dim).

    The sampler cycles forever: callers pull from it on a duty cycle and
    the epoch boundary of the main loader is what ends training, so the
    triplet set is traversed however many times that duty cycle implies.
    Each traversal is reshuffled.
    """

    def __init__(self, triples: np.ndarray, batch_triplets: int = 6,
                 seed: int = 42) -> None:
        """Store the assignments and the batch width.

        Args:
            triples: (N, 3) array of subset positions.
            batch_triplets: triplets per yielded batch.
            seed: base seed for the per-traversal shuffle.
        """
        self.triples = np.asarray(triples, dtype=np.int64)
        self.batch_triplets = int(batch_triplets)
        self.seed = int(seed)
        self._pass = 0

    def __len__(self) -> int:
        """Batches in one traversal of the triplet set."""
        if self.batch_triplets <= 0:
            return 0
        return len(self.triples) // self.batch_triplets

    def __iter__(self) -> Iterator[List[int]]:
        """Cycle over reshuffled traversals indefinitely."""
        if len(self.triples) == 0 or self.batch_triplets <= 0:
            return
        while True:
            rng = np.random.default_rng(self.seed + self._pass)
            order = rng.permutation(len(self.triples))
            self._pass += 1
            n_full = len(order) // self.batch_triplets
            for b in range(n_full):
                sel = order[b * self.batch_triplets:
                            (b + 1) * self.batch_triplets]
                yield [int(i) for i in self.triples[sel].reshape(-1)]


def build_key_index(dataset: object,
                    indices: Sequence[int]) -> Dict[str, int]:
    """Map audio filename key to position within a training Subset.

    Args:
        dataset: the underlying EmotionDataset, whose .data holds one
            sample dict per row including "utt_key".
        indices: the dataset row indices the Subset wraps, in Subset
            order, so the returned values index the Subset not the
            dataset.

    Returns:
        filename key to Subset position, skipping rows with no key.
    """
    rows = getattr(dataset, "data", None)
    if rows is None:
        return {}
    out: Dict[str, int] = {}
    for pos, idx in enumerate(indices):
        key = str(rows[idx].get("utt_key") or "")
        if key:
            out[key] = pos
    return out
