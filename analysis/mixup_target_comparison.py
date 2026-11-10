#!/usr/bin/env python3
"""Compare mixup target constructions under three scoring regimes.

Every arm here shares a backbone, a blend, a lambda distribution, an
optimiser and an evaluation protocol. Only the TARGET differs:

  hard          the dominant endpoint's hard label
  vanilla       lam * y_a + (1 - lam) * y_b, standard mixup
  endpoint      lam * q(v_a) + (1 - lam) * q(v_b), VAD posteriors of the
                endpoints interpolated by lambda
  genlabel      normalised class likelihood at z_mix under class-conditional
                densities fitted on training EMBEDDINGS (Sohn et al. 2022)
  geometry      q(lam * v_a + (1 - lam) * v_b), the posterior AT the blended
                VAD point, which is the arm under test

Three scoring regimes are reported side by side, because they answer
different questions and only the first is a deployable number.

  raw           plain argmax. What the model actually predicts.
  dev-cal       per-class additive logit biases fitted on a DEVELOPMENT set
                and applied unchanged to test. Two development sets are
                reported: the in-domain MSP-Podcast held-out split, which is
                the deployable choice because it needs no target labels, and
                a speaker-disjoint two-fold split of the target corpus,
                which is the strongest calibration a practitioner could get.
  test-oracle   biases fitted on the test labels themselves. Not deployable
                and not a result; an upper bound on what any per-class
                threshold adjustment could buy, reported only so a gain that
                is purely a boundary shift can be recognised as one.

Macro F1 and macro one-vs-rest AUC accompany UAR. UAR is recall-only, so a
favourable recall trade inflates it; F1 is precision-sensitive and AUC is
threshold-free, so an effect present in UAR but absent in both is a
decision-boundary effect rather than a discrimination effect.
"""

import argparse
import glob
import pathlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from analysis.vad_explained_split import uar

CLASS_NAMES = ("neutral", "happy", "sad", "angry")
# IEMOFULL is the standard 4-class IEMOCAP protocol (5,531 utterances, with
# excited merged into happy). The old "IEMO" key is the excited-EXCLUDED
# subset (4,490, happy 595), which is not what the literature reports; every
# result on it is superseded and it is deliberately not listed here.
CORPORA = (("IEMOCAP", "test_IEMOFULL"), ("SAMSEMO", "test_SAMSEMO"))

# (label, arm globs pooled into one arm). Pooling exists because uf4_base is
# seed 42 and uf4_base_more carries the rest, and the _more top-ups are
# config-identical to their parents apart from the seed list.
LADDER: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("no-mixup baseline", ("uf4_base", "uf4_base_more")),
    ("hard label", ("uf4_softmix_ctrl", "uf4_softmix_ctrl_more")),
    ("vanilla mixup", ("uf4_softmix_vanilla", "uf4_softmix_vanilla_more")),
    ("endpoint posterior", ("uf4_softmix_endpt", "uf4_softmix_endpt_more")),
    ("GenLabel (embeddings)", ("uf4_genlabel", "uf4_genlabel_more")),
    ("GEOMETRY (ours)", ("uf4_softmix", "uf4_softmix_more")),
)


def by_seed(arms: Sequence[str], split: str) -> Dict[int, np.ndarray]:
    """Map seed -> logits, pooling several config-identical arm names.

    Args:
        arms: arm names to pool.
        split: prediction file stem, e.g. "IEMO" or "heldout".

    Returns:
        Dict from seed to the [N, C] logit array.
    """
    out: Dict[int, np.ndarray] = {}
    for arm in arms:
        for p in glob.glob(f"checkpoints/{arm}_seed*/predictions/{split}.npz"):
            stem = pathlib.Path(p).parts[1]
            # uf4_softmix_seed42_seed42 must not be matched by uf4_softmix_agg.
            if not re.fullmatch(rf"{re.escape(arm)}_seed\d+(_seed\d+)?", stem):
                continue
            m = re.search(r"_seed(\d+)", stem)
            if m:
                out[int(m.group(1))] = np.load(p, allow_pickle=True)["logits"]
    return out


def dev_labels(arms: Sequence[str]) -> Dict[int, np.ndarray]:
    """Map seed -> development-split labels for the in-domain dev set.

    Args:
        arms: arm names to pool.

    Returns:
        Dict from seed to the [N] label array of that seed's held-out split.
    """
    out: Dict[int, np.ndarray] = {}
    for arm in arms:
        for p in glob.glob(f"checkpoints/{arm}_seed*/predictions/heldout.npz"):
            stem = pathlib.Path(p).parts[1]
            if not re.fullmatch(rf"{re.escape(arm)}_seed\d+(_seed\d+)?", stem):
                continue
            m = re.search(r"_seed(\d+)", stem)
            if m:
                d = np.load(p, allow_pickle=True)
                key = "label4" if "label4" in d.files else "labels"
                out[int(m.group(1))] = d[key]
    return out


def fit_bias(logits: np.ndarray, y: np.ndarray, span: float = 3.0,
             step: float = 0.15, sweeps: int = 5) -> np.ndarray:
    """Per-class additive logit bias maximising UAR by coordinate ascent.

    Args:
        logits: [N, C] logits to fit on.
        y: [N] labels to fit against.
        span: search range for each coordinate.
        step: grid resolution.
        sweeps: number of full passes over the coordinates.

    Returns:
        [C] bias vector.
    """
    n_cls = logits.shape[1]
    b = np.zeros(n_cls)
    best = uar(y, logits.argmax(1))
    for _ in range(sweeps):
        for c in range(n_cls):
            for v in np.arange(-span, span + 1e-9, step):
                t = b.copy()
                t[c] = v
                u = uar(y, (logits + t).argmax(1))
                if u > best:
                    best, b = u, t
    return b


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax.

    Args:
        z: [N, C] logits.

    Returns:
        [N, C] probabilities.
    """
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def scores(logits: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """UAR, macro F1 and macro one-vs-rest AUC, in percent.

    Args:
        logits: [N, C] logits.
        y: [N] labels.

    Returns:
        Dict with keys 'uar', 'acc', 'f1', 'auc'.
    """
    pred = logits.argmax(1)
    prob = softmax(logits)
    aucs = [roc_auc_score((y == c).astype(int), prob[:, c])
            for c in range(logits.shape[1])]
    return {"uar": 100 * uar(y, pred),
            # Overall accuracy, WA in the SER literature. Reported next to
            # UAR because the two can disagree: UAR averages recall and so
            # rewards shrinking a dominant class, which costs accuracy. A
            # change that raises UAR while lowering accuracy has moved the
            # operating point rather than improved the model.
            "acc": 100 * float((pred == y).mean()),
            "f1": 100 * f1_score(y, pred, average="macro"),
            "auc": 100 * float(np.mean(aucs))}


def speaker_folds(speakers: np.ndarray, seed: int = 0) -> List[np.ndarray]:
    """Split row indices into two speaker-disjoint folds of similar size.

    Args:
        speakers: [N] speaker identifiers.
        seed: RNG seed for the speaker shuffle.

    Returns:
        List of two index arrays.
    """
    uniq = np.unique(speakers)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    half = {s: (i % 2) for i, s in enumerate(uniq)}
    side = np.array([half[s] for s in speakers])
    return [np.where(side == 0)[0], np.where(side == 1)[0]]


def crossfit_logits(logits: np.ndarray, y: np.ndarray,
                    speakers: np.ndarray) -> np.ndarray:
    """Apply biases fitted on the opposite speaker-disjoint fold.

    No row is ever scored with a bias fitted on itself, so this is a
    development-calibrated number rather than an oracle.

    Args:
        logits: [N, C] logits.
        y: [N] labels.
        speakers: [N] speaker identifiers.

    Returns:
        [N, C] bias-adjusted logits.
    """
    out = logits.copy()
    folds = speaker_folds(speakers)
    for i, idx in enumerate(folds):
        other = folds[1 - i]
        b = fit_bias(logits[other], y[other])
        out[idx] = logits[idx] + b
    return out


def report(regime: str, table: Dict[str, Dict[int, Dict[str, float]]],
           reference: str) -> None:
    """Print absolute scores and paired deltas against one reference arm.

    Args:
        regime: name of the scoring regime, for the header.
        table: label -> seed -> metric dict.
        reference: label of the arm deltas are taken against.
    """
    print("  --- %s ---" % regime)
    print("    %-24s %4s %9s %9s %9s %9s"
          % ("target", "n", "UAR", "accuracy", "macroF1", "macroAUC"))
    for label in table:
        s = sorted(table[label])
        if not s:
            print("    %-24s %4d  (no runs)" % (label, 0))
            continue
        print("    %-24s %4d %8.2f%% %8.2f%% %8.2f%% %8.2f%%"
              % (label, len(s),
                 np.mean([table[label][i]["uar"] for i in s]),
                 np.mean([table[label][i]["acc"] for i in s]),
                 np.mean([table[label][i]["f1"] for i in s]),
                 np.mean([table[label][i]["auc"] for i in s])))
    ref = table.get(reference, {})
    if not ref:
        print()
        return
    print("    paired deltas against '%s' (unit = training seed)" % reference)
    for label in table:
        if label == reference:
            continue
        s = sorted(set(table[label]) & set(ref))
        if len(s) < 2:
            continue
        bits = []
        for k in ("uar", "acc", "f1", "auc"):
            d = np.array([table[label][i][k] - ref[i][k] for i in s])
            bits.append("%s %+5.2f (p=%.3f)"
                        % ({"uar": "UAR", "acc": "ACC", "f1": "F1",
                            "auc": "AUC"}[k], d.mean(),
                           stats.ttest_1samp(d, 0).pvalue))
        print("      %-24s n=%d  %s" % (label, len(s), "  ".join(bits)))
    print()


def main() -> None:
    """Run every arm through every regime on every corpus."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="vanilla mixup",
                    help="arm label that paired deltas are taken against")
    ap.add_argument("--regimes", nargs="+", default=["raw"],
                    choices=["raw", "dev-in", "dev-target", "oracle"],
                    help="which scoring regimes to print. Default is raw "
                         "alone, which is the only deployable one and the "
                         "only one SER reports. The calibrated regimes are "
                         "diagnostics: they answer whether a gain is a "
                         "boundary shift, and are not results.")
    args = ap.parse_args()

    for corpus, meta_stem in CORPORA:
        # Prediction dumps are named by the dataset key, not the display name.
        corpus = meta_stem[len("test_"):]
        meta = np.load(f"analysis/heldout_meta/{meta_stem}.npz", allow_pickle=True)
        keep = meta["label"] >= 0
        y = meta["label"][keep]
        spk = meta["speaker"][keep]
        print("=" * 82)
        print("%s   N=%d   speakers=%d" % (corpus, len(y), len(np.unique(spk))))
        print()

        raw: Dict[str, Dict[int, Dict[str, float]]] = {}
        dev_in: Dict[str, Dict[int, Dict[str, float]]] = {}
        dev_tg: Dict[str, Dict[int, Dict[str, float]]] = {}
        oracle: Dict[str, Dict[int, Dict[str, float]]] = {}

        for label, arms in LADDER:
            test = by_seed(arms, corpus)
            dev = by_seed(arms, "heldout")
            dlab = dev_labels(arms)
            raw[label], dev_in[label] = {}, {}
            dev_tg[label], oracle[label] = {}, {}
            for seed, lg in sorted(test.items()):
                raw[label][seed] = scores(lg, y)
                if "dev-in" in args.regimes and seed in dev and seed in dlab \
                        and len(dev[seed]) == len(dlab[seed]):
                    b = fit_bias(dev[seed], dlab[seed])
                    dev_in[label][seed] = scores(lg + b, y)
                if "dev-target" in args.regimes:
                    dev_tg[label][seed] = scores(crossfit_logits(lg, y, spk), y)
                if "oracle" in args.regimes:
                    oracle[label][seed] = scores(lg + fit_bias(lg, y), y)

        if "raw" in args.regimes:
            report("raw (no calibration) -- the reported metric",
                   raw, args.reference)
        if "dev-in" in args.regimes:
            report("DIAGNOSTIC dev-cal, in-domain MSP-Podcast held-out split",
                   dev_in, args.reference)
        if "dev-target" in args.regimes:
            report("DIAGNOSTIC dev-cal, target corpus 2-fold",
                   dev_tg, args.reference)
        if "oracle" in args.regimes:
            report("DIAGNOSTIC test-oracle, upper bound only",
                   oracle, args.reference)


if __name__ == "__main__":
    main()
