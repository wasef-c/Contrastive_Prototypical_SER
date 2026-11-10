#!/usr/bin/env python3
"""Which utterances does every arm get wrong, and what do they have in common?

Thirteen arms have been scored on two unseen corpora and the whole session
went into what they were trained toward. Nothing went into what they
actually fail on. Pooling the saved predictions answers a different
question: are the failures spread thinly, so that each arm errs somewhere
else, or is there a hard core that every arm misses regardless of its
target?

A hard core would explain why seven mechanisms all landed within a point
of each other. If the same utterances are wrong in every arm, the target
is not the binding constraint.

Three characterisations of that core, each tied to something already
measured:

VAD CONSISTENCY. For each failure, is it closer to its OWN class centroid
in VAD or to the centroid of the class the models predicted? Cases sitting
nearer the predicted class are ones where the label and the coordinates
disagree, which is the same phenomenon as the 50.4 percent conflict
measured at coincident VAD.

SPEAKER CONCENTRATION. If failures cluster in a few speakers the problem
is adaptation, not emotion.

CONFUSION STRUCTURE. Which ordered label pairs the core consists of,
against the measured annotator substitution rates, to see whether the
models fail where people do.

Usage:
    python -m analysis.failure_modes
"""

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, "/home/rml/Documents/pythontest/Emotion2Vec_Contrastive")

from analysis.arm_table import CORPORA, by_seed

OUT = pathlib.Path("analysis/failure_modes")
NAMES = ("neutral", "happy", "sad", "angry")

# Every arm family with predictions worth pooling. Pooled across seeds too,
# since an utterance missed by one seed and not another is not hard core.
ARMS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("rl_hard", ("rl_hard",)),
    ("rl_raw", ("rl_raw",)),
    ("rl_dir", ("rl_dir",)),
    ("rl_dir_hi", ("rl_dir_hi",)),
    ("rl_lowrank", ("rl_lowrank",)),
    ("rl_kernel", ("rl_kernel",)),
    ("rl_ot", ("rl_ot",)),
    ("rl_ot_hi", ("rl_ot_hi",)),
    ("rl_hard_s9", ("rl_hard_s9",)),
    ("rl_lowrank_s9", ("rl_lowrank_s9",)),
    ("mixup_ours", ("uf4_softmix", "uf4_softmix_more")),
    ("mixup_vanilla", ("uf4_softmix_vanilla", "uf4_softmix_vanilla_more")),
    ("globalmix", ("uf4_globalmix", "uf4_globalmix_more")),
)


def main() -> None:
    """Pool predictions and describe the hard core."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20,
                        help="example failures to list per corpus")
    args = parser.parse_args()

    summary: Dict[str, object] = {}
    for corpus, stem, key in CORPORA:
        meta = np.load("analysis/heldout_meta/%s.npz" % stem,
                       allow_pickle=True)
        lab = meta["label"]
        valid = lab >= 0
        y = lab[valid]
        vad = meta["vad"][valid]
        spk = meta["speaker"][valid].astype(str)

        preds: Dict[str, np.ndarray] = {}
        for name, arms in ARMS:
            d = by_seed(arms, key)
            for seed, logits in sorted(d.items()):
                if logits.shape[0] != len(y):
                    continue
                preds["%s/s%d" % (name, seed)] = logits.argmax(axis=1)
        if not preds:
            continue

        names = sorted(preds)
        stack = np.stack([preds[n] for n in names])
        wrong = stack != y[None, :]
        n_models = len(names)
        miss = wrong.sum(axis=0)

        print("\n" + "#" * 70)
        print("%s: %d utterances, %d model-seed runs pooled"
              % (corpus, len(y), n_models))
        print("  mean accuracy across runs %.3f"
              % float((~wrong).mean()))

        print("\n  how many runs miss each utterance")
        hist = np.bincount(miss, minlength=n_models + 1)
        for k in (0, 1, n_models // 4, n_models // 2,
                  3 * n_models // 4, n_models - 1, n_models):
            if k <= n_models:
                print("    %-3d runs wrong: %6d utterances (%5.1f%%)"
                      % (k, hist[k], 100.0 * hist[k] / len(y)))
        core = miss == n_models
        easy = miss == 0
        print("    ---")
        print("    wrong in EVERY run: %d (%.1f%%)"
              % (int(core.sum()), 100.0 * core.mean()))
        print("    right in every run: %d (%.1f%%)"
              % (int(easy.sum()), 100.0 * easy.mean()))

        print("\n  the hard core by true class")
        print("    %-9s %8s %8s %9s" % ("class", "core", "total", "share"))
        for c in range(4):
            sel = core & (y == c)
            tot = int((y == c).sum())
            print("    %-9s %8d %8d %8.1f%%"
                  % (NAMES[c], int(sel.sum()), tot,
                     100.0 * sel.sum() / max(tot, 1)))

        print("\n  what the core gets called instead (modal prediction)")
        modal = np.zeros(len(y), dtype=np.int64)
        for i in np.where(core)[0]:
            modal[i] = np.bincount(stack[:, i], minlength=4).argmax()
        pairs = collections.Counter(
            (NAMES[y[i]], NAMES[modal[i]]) for i in np.where(core)[0])
        for (t, p), n in pairs.most_common(8):
            print("    %-9s -> %-9s %6d" % (t, p, n))

        # VAD consistency: nearer its own class centroid, or the predicted
        # one? Centroids come from the correctly classified rows, so they
        # are not defined by the failures being described.
        cent = np.stack([vad[easy & (y == c)].mean(axis=0)
                         for c in range(4)])
        d_own = np.linalg.norm(vad - cent[y], axis=1)
        d_pred = np.linalg.norm(vad - cent[modal], axis=1)
        nearer_pred = (d_pred < d_own)
        print("\n  VAD consistency of the hard core")
        print("    closer in VAD to the PREDICTED class than to its own:"
              " %.1f%%" % (100.0 * nearer_pred[core].mean()))
        print("    same figure for utterances every run gets right: %.1f%%"
              % (100.0 * nearer_pred[easy].mean()))
        print("    mean distance to own class centroid: core %.3f, easy %.3f"
              % (d_own[core].mean(), d_own[easy].mean()))

        print("\n  speaker concentration")
        per_spk = collections.Counter(spk[core])
        tot_spk = collections.Counter(spk)
        n_spk = len(tot_spk)
        top = per_spk.most_common(5)
        share = sum(n for _, n in top) / max(core.sum(), 1)
        print("    %d speakers; the 5 worst hold %.1f%% of the core"
              % (n_spk, 100.0 * share))
        for s, n in top:
            print("      %-16s %4d of %4d utterances (%.0f%%)"
                  % (s, n, tot_spk[s], 100.0 * n / tot_spk[s]))

        print("\n  %d example core failures" % args.top)
        print("    %-18s %-9s %-9s %-18s %s"
              % ("speaker", "true", "called", "VAD", "d(own) d(pred)"))
        idx = np.where(core)[0]
        order = idx[np.argsort(-d_own[idx])][:args.top]
        for i in order:
            print("    %-18s %-9s %-9s %-18s %.2f  %.2f"
                  % (spk[i], NAMES[y[i]], NAMES[modal[i]],
                     str(np.round(vad[i], 2)), d_own[i], d_pred[i]))

        summary[corpus] = {
            "n": int(len(y)), "runs": n_models,
            "core": int(core.sum()), "easy": int(easy.sum()),
            "core_by_class": {NAMES[c]: int((core & (y == c)).sum())
                              for c in range(4)},
            "core_pairs": {"%s->%s" % k: v for k, v in pairs.most_common(10)},
            "core_nearer_predicted_vad": float(nearer_pred[core].mean()),
            "easy_nearer_predicted_vad": float(nearer_pred[easy].mean()),
            "top5_speaker_share": float(share),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote %s" % (OUT / "summary.json"))


if __name__ == "__main__":
    main()
