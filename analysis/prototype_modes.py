#!/usr/bin/env python3
"""Is the hard core a secondary mode of its class, or scattered outliers?

Pooling 35 model-seed runs showed 19.0 percent of IEMOCAP and 16.8 percent
of SAMSEMO are wrong in EVERY run, and that this core is atypical of its
own class: 40.9 percent of it sits closer in VAD to the class the models
predicted than to its own, against 6.6 percent of the utterances every run
gets right.

That confirms prototypicality as a description of failure. It does not say
what to do, and it argues against the target-side remedies already tried,
since no target reaches a core that is invariant across 35 runs.

The one prototypicality result that ever helped cross-corpus UAR in this
project used TWO clusters per class. That suggests a sharper question: if
a class has two modes and is modelled with one centroid, the minor mode is
atypical by construction and fails systematically whatever the target. So:

MODE MEMBERSHIP. Cluster each class on the utterances every run gets
RIGHT, so the clusters are not defined by the failures being described.
Then assign the core and ask whether it concentrates in the minor cluster.
Concentration means the failures are a coherent sub-population; an even
spread means they are outliers and prototype structure will not help.

PROTOTYPE COUNT. Distance to the nearest own-class prototype at k = 1, 2,
3. If extra prototypes shorten the core's distance much more than the easy
set's, the core is sitting in a mode the single centroid misses.

ACOUSTIC VERSUS VAD TYPICALITY. Embedding distance to its own class
against the predicted class. Closer to the predicted class means the
signal genuinely resembles the other category and the label is what is in
tension; far from both means an outlier.

Usage:
    python -m analysis.prototype_modes
"""

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/rml/Documents/pythontest/Emotion2Vec_Contrastive")

from analysis.arm_table import CORPORA, by_seed
from analysis.failure_modes import ARMS
from analysis.modality_ablation import load_arm
from data.collate import vad_collate_fn
from data.dataset import create_datasets
from sklearn.cluster import KMeans
from train import _prepare_model_inputs

OUT = pathlib.Path("analysis/prototype_modes")
NAMES = ("neutral", "happy", "sad", "angry")


def embed(model, dataset, config, device, batch_size: int) -> np.ndarray:
    """Post-fusion embeddings for a whole dataset, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=vad_collate_fn)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = _prepare_model_inputs(batch, config, model, device)
            _, _, emb, _ = model(**inputs, return_embeddings=True)
            out.append(emb.detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    """Run the three characterisations per corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="rl_hard", help="checkpoint arm")
    parser.add_argument("--seed", type=int, default=42, help="run seed")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="forward-pass batch size, small for RAM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_arm(args.arm, args.seed, device)
    # create_datasets returns a LIST of datasets, each carrying
    # its own dataset_name, so index it by that rather than by
    # the prediction-file key.
    _, _test_list = create_datasets(config)
    test_sets = {d.dataset_name: d for d in _test_list}
    print("test datasets loaded: %s" % list(test_sets))

    summary: Dict[str, object] = {}
    for corpus, stem, key in CORPORA:
        ds = test_sets.get(key)
        if ds is None:
            print("\n%s: no dataset under key %s, skipping" % (corpus, key))
            continue
        meta = np.load("analysis/heldout_meta/%s.npz" % stem,
                       allow_pickle=True)
        lab = meta["label"]
        valid = lab >= 0
        y = lab[valid]

        preds: List[np.ndarray] = []
        for name, arms in ARMS:
            for seed, logits in sorted(by_seed(arms, key).items()):
                if logits.shape[0] == len(y):
                    preds.append(logits.argmax(axis=1))
        if not preds:
            continue
        stack = np.stack(preds)
        miss = (stack != y[None, :]).sum(axis=0)
        core = miss == len(preds)
        easy = miss == 0

        x_all = embed(model, ds, config, device, args.batch_size)
        if x_all.shape[0] != len(valid):
            print("\n%s: embeddings %d do not match metadata %d, skipping"
                  % (corpus, x_all.shape[0], len(valid)))
            continue
        x = x_all[valid]
        x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        print("\n" + "#" * 70)
        print("%s: %d scored rows, core %d (%.1f%%), easy %d"
              % (corpus, len(y), int(core.sum()), 100 * core.mean(),
                 int(easy.sum())))

        modal = np.zeros(len(y), dtype=np.int64)
        for i in np.where(core)[0]:
            modal[i] = np.bincount(stack[:, i], minlength=4).argmax()

        print("\n  MODE MEMBERSHIP: clusters fitted on the easy rows only")
        print("  %-9s %6s %22s %22s"
              % ("class", "k", "easy split", "core split"))
        rec: Dict[str, object] = {}
        for c in range(4):
            fit_rows = x[easy & (y == c)]
            core_rows = x[core & (y == c)]
            if len(fit_rows) < 60 or len(core_rows) < 20:
                print("  %-9s too few rows (%d easy, %d core)"
                      % (NAMES[c], len(fit_rows), len(core_rows)))
                continue
            for k in (2, 3):
                km = KMeans(n_clusters=k, n_init=10,
                            random_state=0).fit(fit_rows)
                e_share = np.bincount(km.labels_, minlength=k) / len(fit_rows)
                c_share = (np.bincount(km.predict(core_rows), minlength=k)
                           / len(core_rows))
                order = np.argsort(e_share)
                print("  %-9s %6d %22s %22s"
                      % (NAMES[c] if k == 2 else "", k,
                         np.round(e_share[order], 2),
                         np.round(c_share[order], 2)))
                if k == 2:
                    rec[NAMES[c]] = {
                        "easy_minor": float(e_share[order][0]),
                        "core_minor": float(c_share[order][0]),
                    }
        print("  (shares ordered smallest-easy-cluster first; the core")
        print("   concentrating there is the multi-mode signature)")

        print("\n  PROTOTYPE COUNT: mean distance to nearest own-class centre")
        print("  %-9s %8s %8s %8s   %s" % ("class", "k=1", "k=2", "k=3",
                                           "core vs easy gain"))
        for c in range(4):
            fit_rows = x[easy & (y == c)]
            core_rows = x[core & (y == c)]
            if len(fit_rows) < 60 or len(core_rows) < 20:
                continue
            d_core: List[float] = []
            d_easy: List[float] = []
            for k in (1, 2, 3):
                km = KMeans(n_clusters=k, n_init=10,
                            random_state=0).fit(fit_rows)
                cen = km.cluster_centers_
                d_core.append(float(np.linalg.norm(
                    core_rows[:, None, :] - cen[None], axis=2).min(1).mean()))
                d_easy.append(float(np.linalg.norm(
                    fit_rows[:, None, :] - cen[None], axis=2).min(1).mean()))
            gain_core = (d_core[0] - d_core[2]) / max(d_core[0], 1e-9)
            gain_easy = (d_easy[0] - d_easy[2]) / max(d_easy[0], 1e-9)
            print("  %-9s %8.3f %8.3f %8.3f   core %+.1f%% easy %+.1f%%"
                  % (NAMES[c], d_core[0], d_core[1], d_core[2],
                     -100 * gain_core, -100 * gain_easy))

        print("\n  ACOUSTIC TYPICALITY: own class vs the predicted class")
        cen1 = np.stack([x[easy & (y == c)].mean(axis=0) for c in range(4)])
        cen1 = cen1 / np.linalg.norm(cen1, axis=1, keepdims=True).clip(1e-8)
        d_own = np.linalg.norm(x - cen1[y], axis=1)
        d_pred = np.linalg.norm(x - cen1[modal], axis=1)
        nearer = d_pred < d_own
        print("    core closer to the PREDICTED class acoustically: %.1f%%"
              % (100 * nearer[core].mean()))
        print("    mean distance to own class: core %.3f, easy %.3f"
              % (d_own[core].mean(), d_own[easy].mean()))
        print("    core rows further from BOTH than the easy mean: %.1f%%"
              % (100 * ((d_own[core] > d_own[easy].mean())
                        & (d_pred[core] > d_own[easy].mean())).mean()))

        summary[corpus] = {
            "core": int(core.sum()), "easy": int(easy.sum()),
            "two_mode": rec,
            "core_nearer_predicted_acoustic": float(nearer[core].mean()),
            "d_own_core": float(d_own[core].mean()),
            "d_own_easy": float(d_own[easy].mean()),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote %s" % (OUT / "summary.json"))


if __name__ == "__main__":
    main()
