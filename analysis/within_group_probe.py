#!/usr/bin/env python3
"""Holding VAD exactly fixed, can the signal tell the labels apart?

The cleanest existence proof in this work is that 4,412 MSP-Podcast
utterances have a UNANIMOUS four-class label and ZERO annotator dispersion
in VAD, falling into 98 groups that share a VAD coordinate exactly, and
that two members of a group carry different labels 50.4 percent of the
time. Something distinguishes them and VAD does not contain it.

Every attempt so far looked for that something as an extra axis: a
language dimension, an appraisal dimension, finer subtypes, a residual
direction, a low-rank continuum, a transport metric. All failed. This asks
the question directly instead, inside the groups where VAD is constant by
construction, so no amount of VAD could help.

TEST A, the prototype hypothesis. Categories may encode resemblance to
exemplars rather than position on axes: "angry" meaning "like other angry
things" is a similarity judgement, and similarity structure is not
recoverable by adding coordinates, which would explain the shape of the
failures. If so, same-label utterances in a group should sit closer in
embedding space than different-label ones.

Only CROSS-SPEAKER pairs count. A speaker's utterances cluster in
embedding space and speakers repeat within a label, so same-speaker pairs
would show the effect whether or not the labels carry any.

TEST B, direct discrimination. Predict the label from embeddings with
whole speakers held out, after removing each group's mean so only
within-group variation remains. Chance is the within-group label
distribution, which the shuffled baseline reports.

Embeddings come from a model trained on hard labels, which is the
conservative choice: it was optimised to discard within-class variation,
so anything recoverable here is recoverable despite that.

Usage:
    python -m analysis.within_group_probe
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

from analysis.genlabel_dev_tuning import reproduce_split
from analysis.modality_ablation import load_arm
from data.collate import vad_collate_fn
from data.dataset import EmotionDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from train import _prepare_model_inputs
from utils.annotator_meta import load_subtype_meta

OUT = pathlib.Path("analysis/within_group/probe.json")
NAMES = ("neutral", "happy", "sad", "angry")


def coincident_rows(meta) -> Dict[str, Tuple[int, Tuple[float, ...]]]:
    """Filename key to (label, exact VAD) for unanimous, zero-spread rows.

    Args:
        meta: SubtypeMeta carrying label_dist, vad_std and vad_mean.

    Returns:
        Dict from key to its four-class label and VAD coordinate.
    """
    out: Dict[str, Tuple[int, Tuple[float, ...]]] = {}
    for key, row in meta.index.items():
        dist = meta.label_dist[row]
        if dist[:4].sum() < 0.999:
            continue
        if dist.max() < 0.999:
            continue
        if float(meta.vad_std[row].mean()) != 0.0:
            continue
        out[key] = (int(dist.argmax()), tuple(meta.vad_mean[row].tolist()))
    return out


def main() -> None:
    """Run both tests inside the coincident-VAD groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="uf4_softmix", help="checkpoint arm")
    parser.add_argument("--seed", type=int, default=42, help="run seed")
    parser.add_argument("--min-group", type=int, default=12,
                        help="smallest group worth analysing")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="forward-pass batch size, small for RAM")
    args = parser.parse_args()

    meta = load_subtype_meta()
    if meta is None:
        raise FileNotFoundError("annotator cache missing")
    wanted = coincident_rows(meta)
    print("unanimous label and zero VAD spread: %d utterances" % len(wanted))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_arm(args.arm, args.seed, device)
    raw = config.train_dataset
    corpus = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
    dataset = EmotionDataset(corpus, config=config, split="train")
    train_idx, _ = reproduce_split(dataset, config)

    sel: List[int] = []
    keys: List[str] = []
    for idx in train_idx:
        key = dataset.data[idx].get("utt_key") or ""
        if key in wanted:
            sel.append(idx)
            keys.append(key)
    print("matched in the training split: %d" % len(sel))
    if not sel:
        raise RuntimeError("no rows matched; check utt_key is populated")

    loader = DataLoader(Subset(dataset, sel), batch_size=args.batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=vad_collate_fn)
    embs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = _prepare_model_inputs(batch, config, model, device)
            _, _, emb, _ = model(**inputs, return_embeddings=True)
            embs.append(emb.detach().cpu().float().numpy())
    x = np.concatenate(embs, axis=0)
    y = np.array([wanted[k][0] for k in keys])
    grp = np.array([hash(wanted[k][1]) for k in keys])
    spk = np.array([dataset.data[i].get("speaker", "?") for i in sel])
    print("embeddings %s, groups %d, speakers %d"
          % (x.shape, len(set(grp)), len(set(spk))))

    # Keep groups that are big enough AND actually contain two labels.
    usable: List[int] = []
    for g in set(grp):
        m = grp == g
        if m.sum() >= args.min_group and len(set(y[m])) > 1:
            usable.append(g)
    mask = np.isin(grp, usable)
    x, y, grp, spk = x[mask], y[mask], grp[mask], spk[mask]
    print("after filtering: %d rows in %d mixed-label groups"
          % (len(y), len(usable)))
    print("  label mix: %s" % {NAMES[k]: int((y == k).sum())
                               for k in range(4) if (y == k).sum()})

    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)

    print("\n" + "=" * 68)
    print("TEST A. within a group, are same-label pairs closer?")
    print("        cross-speaker pairs only")
    same: List[float] = []
    diff: List[float] = []
    for g in usable:
        m = np.where(grp == g)[0]
        for a_i in range(len(m)):
            for b_i in range(a_i + 1, len(m)):
                a, b = m[a_i], m[b_i]
                if spk[a] == spk[b]:
                    continue
                d = 1.0 - float(xn[a] @ xn[b])
                (same if y[a] == y[b] else diff).append(d)
    same_a, diff_a = np.array(same), np.array(diff)
    pooled = np.sqrt(0.5 * (same_a.var(ddof=1) + diff_a.var(ddof=1)))
    print("  same-label pairs      n=%-7d mean cosine distance %.4f"
          % (len(same_a), same_a.mean()))
    print("  different-label pairs n=%-7d mean cosine distance %.4f"
          % (len(diff_a), diff_a.mean()))
    print("  difference %+.4f, Cohen's d %+.3f"
          % (diff_a.mean() - same_a.mean(),
             (diff_a.mean() - same_a.mean()) / max(pooled, 1e-9)))

    print("\n" + "=" * 68)
    print("TEST B. within-group label prediction, speakers held out")
    # Remove each group's mean so only within-group variation is left.
    xc = x.copy()
    for g in usable:
        m = grp == g
        xc[m] -= xc[m].mean(axis=0)
    xs = StandardScaler().fit_transform(xc)
    n_spk = len(set(spk))
    folds = min(5, n_spk)
    gkf = GroupKFold(n_splits=folds)
    accs: List[float] = []
    for tr, te in gkf.split(xs, y, groups=spk):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(xs[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(xs[te])))
    rng = np.random.default_rng(0)
    null: List[float] = []
    for _ in range(20):
        yp = rng.permutation(y)
        f: List[float] = []
        for tr, te in gkf.split(xs, yp, groups=spk):
            clf = LogisticRegression(max_iter=1000,
                                     class_weight="balanced")
            clf.fit(xs[tr], yp[tr])
            f.append(balanced_accuracy_score(yp[te], clf.predict(xs[te])))
        null.append(float(np.mean(f)))
    print("  balanced accuracy %.3f  (label-shuffled %.3f +- %.3f)"
          % (float(np.mean(accs)), float(np.mean(null)),
             float(np.std(null))))
    print("  per fold: %s" % np.round(accs, 3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_rows": int(len(y)), "n_groups": len(usable),
        "same_mean": float(same_a.mean()), "diff_mean": float(diff_a.mean()),
        "cohens_d": float((diff_a.mean() - same_a.mean()) / max(pooled, 1e-9)),
        "within_group_balanced_acc": float(np.mean(accs)),
        "shuffled_balanced_acc": float(np.mean(null)),
        "args": vars(args),
    }, indent=2))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
