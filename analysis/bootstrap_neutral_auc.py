"""Speaker-clustered bootstrap confidence intervals on the neutral_auc gain.

Every claim so far rests on n=3 seeds, which yields p values around 0.02 to
0.10 and cannot distinguish a small real effect from seed noise. The saved
per-sample logits allow a far more powerful test: resample SPEAKERS with
replacement, recompute the metric for each arm on the same resample, and take
the paired difference. Speakers rather than utterances, because utterances
from one speaker are not independent and resampling them directly would give
falsely narrow intervals.

Seeds are averaged inside each bootstrap draw, so the interval reflects
sampling variability in the evaluation corpora with seed noise averaged down.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = '/home/rml/Documents/pythontest/Emotion2Vec_Contrastive'
CORPORA = ['IEMO', 'MSPI', 'CMUMOSEI', 'SAMSEMO']
SEEDS = [42, 189, 7]
N_BOOT = 1000
RNG = np.random.default_rng(0)

ARMS: Dict[str, Dict[int, str]] = {
    'res': {s: f'aub_b5e6_res_seed{s}_seed{s}' for s in SEEDS},
    'ctrl': {s: f'aub_b5e6_ctrl_seed{s}_seed{s}' for s in SEEDS},
    'base': {42: 'aub_b5e6_base_seed42',
             189: 'cc_base5e6_seed189_seed189',
             7: 'cc_base5e6_seed7'},
}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, matching utils.metrics.threshold_free_metrics."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=1, keepdims=True)


def load(run_dir: str, corpus: str) -> Optional[Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]]:
    """Return (probs, labels, speakers) for one run and corpus."""
    path = os.path.join(ROOT, 'checkpoints', run_dir, 'predictions',
                        f'{corpus}.npz')
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        return (softmax(np.asarray(z['logits'], dtype=np.float64)),
                np.asarray(z['labels']),
                np.asarray(z['speakers']))


def fast_auc(binary: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), tie-corrected.

    Equivalent to sklearn's roc_auc_score but without its validation
    overhead, which dominates when the metric is recomputed thousands of
    times inside a bootstrap loop.
    """
    n_pos = int(binary.sum())
    n_neg = int(binary.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    # Average ranks within tied groups so the statistic matches sklearn.
    srt = score[order]
    ties = np.flatnonzero(np.diff(srt)) + 1
    for start, stop in zip(np.r_[0, ties], np.r_[ties, score.size]):
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop + 1) / 2.0
    return float((ranks[binary == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def neutral_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Neutral-vs-rest ROC-AUC, identical to the training-time definition."""
    is_emo = (labels != 0).astype(int)
    if not 0 < is_emo.sum() < len(is_emo):
        return float('nan')
    return fast_auc(is_emo, 1.0 - probs[:, 0])


def emo_auc(probs: np.ndarray, labels: np.ndarray,
            num_classes: int = 4) -> float:
    """Macro one-vs-rest AUC over emotional classes only."""
    mask = labels != 0
    if mask.sum() <= 1:
        return float('nan')
    emo_labels = labels[mask] - 1
    if len(np.unique(emo_labels)) <= 1:
        return float('nan')
    ep = probs[mask][:, 1:num_classes]
    ep = ep / np.clip(ep.sum(axis=1, keepdims=True), 1e-12, None)
    try:
        return float(roc_auc_score(emo_labels, ep, multi_class='ovr',
                                   average='macro'))
    except ValueError:
        return float('nan')


METRIC_FNS = {'neutral_auc': neutral_auc, 'emo_auc': emo_auc}


def compare(arm_a: str, arm_b: str, corpus: str, metric: str) -> Optional[dict]:
    """Speaker-clustered paired bootstrap of arm_a minus arm_b."""
    fn = METRIC_FNS[metric]
    data_a, data_b = [], []
    speakers = None
    labels = None
    for seed in SEEDS:
        da = load(ARMS[arm_a][seed], corpus)
        db = load(ARMS[arm_b][seed], corpus)
        if da is None or db is None:
            return None
        pa, la, sa = da
        pb, lb, _ = db
        # The evaluation set is fixed, so labels must align across arms; if
        # they do not, the runs are not comparable per-sample and pairing
        # would be meaningless.
        if not np.array_equal(la, lb):
            return None
        data_a.append(pa)
        data_b.append(pb)
        speakers, labels = sa, la

    uniq = np.unique(speakers)
    idx_by_speaker = {s: np.flatnonzero(speakers == s) for s in uniq}

    point = float(np.mean([fn(a, labels) - fn(b, labels)
                           for a, b in zip(data_a, data_b)]))

    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        drawn = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_speaker[s] for s in drawn])
        lb_ = labels[idx]
        if len(np.unique(lb_)) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = np.mean([fn(a[idx], lb_) - fn(b[idx], lb_)
                            for a, b in zip(data_a, data_b)])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'point': point, 'lo': float(lo), 'hi': float(hi),
            'p_gt0': float((diffs > 0).mean()), 'n_speakers': len(uniq),
            'n_samples': len(labels)}


def main() -> None:
    """Print bootstrap intervals for the key contrasts."""
    for metric in ['neutral_auc', 'emo_auc']:
        for a, b in [('res', 'base'), ('res', 'ctrl'), ('ctrl', 'base')]:
            print(f'\n=== {metric}: {a} minus {b} '
                  f'(speaker-clustered bootstrap, {N_BOOT} draws) ===')
            print(f"{'corpus':10s} {'diff':>9s} {'95% CI':>21s} "
                  f"{'P(>0)':>7s} {'spk':>5s} {'n':>6s}")
            for corpus in CORPORA:
                r = compare(a, b, corpus, metric)
                if r is None:
                    print(f'{corpus:10s}   (predictions missing)')
                    continue
                ci = f'[{r["lo"]:+.4f}, {r["hi"]:+.4f}]'
                star = ' *' if (r['lo'] > 0 or r['hi'] < 0) else ''
                print(f'{corpus:10s} {r["point"]:+9.4f} {ci:>21s} '
                      f'{r["p_gt0"]:7.3f} {r["n_speakers"]:5d} '
                      f'{r["n_samples"]:6d}{star}')
    print('\n* = 95% interval excludes zero.')


if __name__ == '__main__':
    main()
