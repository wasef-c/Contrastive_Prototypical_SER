#!/usr/bin/env python3
"""Can a model detect ambiguous utterances, and does that transfer?

Every mechanism tried so far has been target-side, and the failure
analysis says target-side cannot reach the problem: 19.0 percent of
IEMOCAP and 16.8 percent of SAMSEMO are wrong in all 35 model-seed runs.
Detecting ambiguity is a different kind of intervention. It does not try
to classify those rows correctly; it tries to know which rows they are.

MSP-Podcast supplies a measured target for ambiguity that the evaluation
corpora do not have: how much an utterance's own panel disagreed. The
honest version of it is noise-corrected, because a five-person split is
partly luck. The unbiased true agreement is

    agree = (n ||p||^2 - 1) / (n - 1)

and 1 - agree is the chance two annotators drawn from an infinite panel
would differ. That is a property of the utterance, not of the five people
who happened to rate it.

THREE QUESTIONS.

1. Is ambiguity predictable at all? Regress it on embeddings with whole
   speakers held out. A decomposition elsewhere in this work found 89
   percent of the WITHIN-CLASS variance in vote distributions to be
   sampling noise, so this could easily come back near zero, and that
   result would matter: it would mean a five-person panel's split is
   mostly luck and not a detectable property.

2. Is it separable from class identity? Ambiguity correlates with class
   (neutral panels agree more), so the same fit is repeated after removing
   each class's mean. What survives is ambiguity the model could know
   beyond knowing the class.

3. Does it transfer? Fit the detector on MSP-Podcast, apply it to IEMOCAP
   and SAMSEMO where no panel data exists, and ask whether its score
   separates the hard core from the rows every run gets right. If it does,
   an abstention rule follows directly, and risk-coverage curves say what
   it buys.

Usage:
    python -m analysis.ambiguity_detector
"""

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/home/rml/Documents/pythontest/Emotion2Vec_Contrastive")

from analysis.arm_table import CORPORA, by_seed
from analysis.failure_modes import ARMS
from analysis.genlabel_dev_tuning import reproduce_split
from analysis.modality_ablation import load_arm
from data.collate import vad_collate_fn
from data.dataset import EmotionDataset, create_datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from train import _prepare_model_inputs

OUT = pathlib.Path("analysis/ambiguity")
NAMES = ("neutral", "happy", "sad", "angry")


def embed(model, dataset, config, device, batch: int) -> np.ndarray:
    """Post-fusion embeddings in dataset order."""
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=0, collate_fn=vad_collate_fn)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for b in loader:
            inputs = _prepare_model_inputs(b, config, model, device)
            _, _, emb, _ = model(**inputs, return_embeddings=True)
            out.append(emb.detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return 1.0 - float(((y - pred) ** 2).sum()) / float(
        ((y - y.mean()) ** 2).sum())


def main() -> None:
    """Fit the detector on MSP-Podcast and test whether it transfers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="rl_hard", help="checkpoint arm")
    parser.add_argument("--seed", type=int, default=42, help="run seed")
    parser.add_argument("--per-class", type=int, default=3000,
                        help="training rows kept per class, for RAM")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="forward-pass batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_arm(args.arm, args.seed, device)

    raw = config.train_dataset
    corpus = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
    ds = EmotionDataset(corpus, config=config, split="train")
    train_idx, _ = reproduce_split(ds, config)

    # Rows carrying annotator data, balanced by class so the target is not
    # predicted merely by tracking class frequency.
    by_class: Dict[int, List[int]] = collections.defaultdict(list)
    for i in train_idx:
        row = ds.data[i]
        if float(row.get("annot_mask", 0.0)) <= 0.0:
            continue
        by_class[int(row.get("label4", row["label"]))].append(i)
    rng = np.random.RandomState(args.seed)
    sel: List[int] = []
    for c in sorted(by_class):
        pool = by_class[c]
        sel.extend(rng.choice(pool, min(args.per_class, len(pool)),
                              replace=False).tolist())
    sel.sort()
    print("MSP-Podcast rows with panels, class balanced: %d" % len(sel))

    dist = np.stack([np.asarray(ds.data[i]["label_dist"], dtype=np.float64)
                     for i in sel])
    n_ann = np.array([float(np.rint(dist[k].sum() * 0) + 5)
                      for k in range(len(sel))])
    # Panel size is not stored per row, so recover it from the distribution
    # granularity where possible and fall back to five, which 89 percent
    # of the corpus has.
    for k in range(len(sel)):
        frac = dist[k][dist[k] > 0]
        if len(frac):
            inv = 1.0 / frac.min()
            if abs(inv - round(inv)) < 1e-3 and 2 <= round(inv) <= 40:
                n_ann[k] = float(round(inv))
    agree = (n_ann * (dist ** 2).sum(axis=1) - 1.0) / (n_ann - 1.0)
    amb = 1.0 - np.clip(agree, 0.0, 1.0)
    y4 = np.array([int(ds.data[i].get("label4", ds.data[i]["label"]))
                   for i in sel])
    spk = np.array([ds.data[i].get("speaker", "?") for i in sel])
    print("  reliability target: mean %.3f, sd %.3f, range %.3f to %.3f"
          % (amb.mean(), amb.std(), amb.min(), amb.max()))
    print("  by class: %s" % {NAMES[c]: round(float(amb[y4 == c].mean()), 3)
                              for c in range(4)})

    # Second target: does the dimensional annotation agree with the
    # categorical one? Measured as whether the utterance sits closer to
    # another class's VAD centroid than to its own. Centroids are computed
    # from the rows themselves, which is acceptable because the quantity is
    # a property of the annotation and not of any model's errors.
    #
    # Predicting annotator agreement from audio is published work.
    # Predicting the RELATION between two annotation schemes is not, as far
    # as this project knows, and it is the more interesting target: it asks
    # the signal to flag where the two descriptions of the same utterance
    # contradict each other.
    vad = np.stack([[ds.data[i]["valence"], ds.data[i]["arousal"],
                     ds.data[i]["dominance"]] for i in sel])
    cen_vad = np.stack([vad[y4 == c].mean(axis=0) for c in range(4)])
    d_to = np.linalg.norm(vad[:, None, :] - cen_vad[None], axis=2)
    d_own_vad = d_to[np.arange(len(vad)), y4]
    other = d_to.copy()
    other[np.arange(len(vad)), y4] = np.inf
    mismatch = (d_own_vad - other.min(axis=1))
    print("\n  VAD-agreement target: distance to own class minus nearest other")
    print("    mean %.3f, sd %.3f; positive means VAD points elsewhere"
          % (mismatch.mean(), mismatch.std()))
    print("    rows where VAD points at another class: %.1f%%"
          % (100 * (mismatch > 0).mean()))
    print("    by class: %s"
          % {NAMES[c]: round(100 * float((mismatch[y4 == c] > 0).mean()), 1)
             for c in range(4)})

    x = embed(model, Subset(ds, sel), config, device, args.batch_size)
    xs = StandardScaler().fit_transform(x)
    print("  embeddings %s, speakers %d" % (x.shape, len(set(spk))))

    gkf = GroupKFold(n_splits=5)
    targets = (("reliability", amb), ("vad_agreement", mismatch))
    fitted: Dict[str, Ridge] = {}
    for tname, tvec in targets:
        print("\n" + "=" * 68)
        print("%s: predictable from the signal?" % tname)
        pred = cross_val_predict(Ridge(alpha=10.0), xs, tvec, cv=gkf,
                                 groups=spk)
        print("   speaker-held-out R2 = %.3f, correlation %.3f"
              % (r2(tvec, pred), float(np.corrcoef(tvec, pred)[0, 1])))

        # Both targets correlate with class identity, so the same fit is
        # repeated with each class's mean removed. What survives is the
        # part a model could know beyond knowing the class.
        centred = tvec.copy()
        for c in range(4):
            m = y4 == c
            centred[m] -= tvec[m].mean()
        pred_c = cross_val_predict(Ridge(alpha=10.0), xs, centred, cv=gkf,
                                   groups=spk)
        print("   within-class R2 = %.3f  (class means give 0 here)"
              % r2(centred, pred_c))
        fitted[tname] = Ridge(alpha=10.0).fit(xs, tvec)

    scaler = StandardScaler().fit(x)

    print("\n" + "=" * 68)
    print("3. Does the detector transfer to the unseen corpora?")
    # create_datasets returns a LIST of datasets, each carrying
    # its own dataset_name, so index it by that rather than by
    # the prediction-file key.
    _, _test_list = create_datasets(config)
    test_sets = {d.dataset_name: d for d in _test_list}
    summary: Dict[str, object] = {"mspp_r2": r2(amb, pred),
                                 "mspp_within_class_r2": r2(amb_c, pred_c)}
    for name, stem, key in CORPORA:
        tds = test_sets.get(key)
        if tds is None:
            continue
        meta = np.load("analysis/heldout_meta/%s.npz" % stem,
                       allow_pickle=True)
        lab = meta["label"]
        valid = lab >= 0
        yv = lab[valid]
        preds: List[np.ndarray] = []
        for aname, arms in ARMS:
            for s, lg in sorted(by_seed(arms, key).items()):
                if lg.shape[0] == len(yv):
                    preds.append(lg.argmax(axis=1))
        if not preds:
            continue
        stack = np.stack(preds)
        miss = (stack != yv[None, :]).sum(axis=0)
        core = miss == len(preds)
        easy = miss == 0

        xt = embed(model, tds, config, device, args.batch_size)
        if xt.shape[0] != len(valid):
            print("   %s: embedding count mismatch, skipping" % name)
            continue
        xv = scaler.transform(xt[valid])
        acc_all = float((stack == yv[None, :]).mean())
        print("\n   %s  (core %d, easy %d, pooled accuracy %.3f)"
              % (name, int(core.sum()), int(easy.sum()), acc_all))
        summary[name] = {}
        for tname, reg in fitted.items():
            score = reg.predict(xv)
            pooled = np.sqrt(0.5 * (score[core].var(ddof=1)
                                    + score[easy].var(ddof=1)))
            d = ((score[core].mean() - score[easy].mean())
                 / max(pooled, 1e-9))
            print("     %-14s core %.3f  easy %.3f  Cohen's d %+.3f"
                  % (tname, score[core].mean(), score[easy].mean(), d))
            # Risk-coverage: drop the rows the detector flags hardest and
            # see whether accuracy on what remains rises. This is the
            # operational value of the detector, independent of whether it
            # is measuring the same thing the models find hard.
            order = np.argsort(score)
            cov = []
            for frac in (0.9, 0.8, 0.7, 0.5):
                kept = order[:int(frac * len(order))]
                acc = float((stack[:, kept] == yv[None, kept]).mean())
                cov.append((frac, acc, acc - acc_all))
            print("       coverage  %s"
                  % "  ".join("%.0f%%" % (100 * f) for f, _, _ in cov))
            print("       accuracy  %s"
                  % "  ".join("%.3f" % a for _, a, _ in cov))
            print("       gain      %s"
                  % "  ".join("%+.3f" % g for _, _, g in cov))
            summary[name][tname] = {
                "core_score": float(score[core].mean()),
                "easy_score": float(score[easy].mean()),
                "cohens_d": float(d),
                "risk_coverage": [[f, a, g] for f, a, g in cov],
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote %s" % (OUT / "summary.json"))


if __name__ == "__main__":
    main()
