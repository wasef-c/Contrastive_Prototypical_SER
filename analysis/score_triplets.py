#!/usr/bin/env python3
"""Score the VAD-matched triplet arms against their shuffled control.

Three numbers decide this experiment, and they are reported in this order
so the contrast cannot be read selectively.

1 THE CONTROL CONTRAST. rl_trip_vad minus rl_trip_rand. Both arms carry
  the triplet term, the same duty cycle and the same multiset of negative
  clips; only the anchor-to-negative matching differs. A gain that shows
  up in both arms is the triplet term and is NOT the claim here. This
  difference is the claim.

2 THE SUBGROUP ENDPOINT. Recall on low-arousal originally-happy IEMOCAP
  clips, which is where the failure analysis locates the confusion the
  matching targets. Stated at all three arousal thresholds because the
  definition has been written inconsistently in this project before.

3 THE BOUNDARY GUARD. The share of neutral clips called happy. A
  one-directional version of this term could raise quiet-happy recall by
  sliding the happy/neutral boundary, which would show up here as a
  matching rise in false alarms rather than as a learned cue. The arms
  are built bidirectional precisely to block that, so this number is the
  check on whether the block held.

Headline UAR, accuracy and macro F1 per corpus come last, because a
mechanism that moves the subgroup without moving the corpus is still
interesting and one that moves the corpus without the subgroup probably
is not this mechanism.

Usage:
    python -m analysis.score_triplets
    python -m analysis.score_triplets --arms rl_trip_vad rl_trip_rand
"""

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from analysis.mixup_target_comparison import by_seed

META = pathlib.Path("analysis/heldout_meta")
OUT = pathlib.Path("analysis/triplets/scores.json")
CORPORA = (("IEMOCAP", "test_IEMOFULL", "IEMOFULL"),
           ("SAMSEMO", "test_SAMSEMO", "SAMSEMO"))
NAMES = ("neutral", "happy", "sad", "angry")


def uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted average recall over the classes present in y_true."""
    recalls = []
    for c in np.unique(y_true):
        sel = y_true == c
        if sel.sum():
            recalls.append(float((y_pred[sel] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def load_meta(stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, VAD and the validity mask for one evaluation corpus.

    Args:
        stem: meta file stem, e.g. "test_IEMOFULL".

    Returns:
        (labels, vad, valid) with labels and vad already restricted to
        valid rows.
    """
    m = np.load(META / ("%s.npz" % stem), allow_pickle=True)
    lab = m["label"]
    valid = lab >= 0
    return lab[valid], m["vad"][valid], valid


def corpus_rows(arm: str, pred_stem: str,
                n_expect: int) -> Dict[int, np.ndarray]:
    """Per-seed argmax predictions for one arm and corpus.

    Args:
        arm: arm name.
        pred_stem: prediction file stem.
        n_expect: expected row count, used to reject stale dumps.

    Returns:
        seed to predicted class array.
    """
    out: Dict[int, np.ndarray] = {}
    for seed, logits in by_seed((arm,), pred_stem).items():
        if logits.shape[0] != n_expect:
            print("   skipping %s seed %d: %d rows, expected %d (stale dump)"
                  % (arm, seed, logits.shape[0], n_expect))
            continue
        out[seed] = logits.argmax(axis=1)
    return out


def paired(a: Dict[int, float], b: Dict[int, float]) -> Optional[Tuple]:
    """Mean difference over the shared seeds, with the per-seed signs.

    Args:
        a: seed to metric for the treatment arm.
        b: seed to metric for the reference arm.

    Returns:
        (n, mean_diff, per_seed_list) or None if no seeds are shared.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = [a[s] - b[s] for s in shared]
    return len(shared), float(np.mean(diffs)), list(zip(shared, diffs))


def main() -> None:
    """Report the control contrast, the subgroup and the headline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+",
                        default=["rl_trip_vad", "rl_trip_rand",
                                 "rl_hard_s9"])
    parser.add_argument("--treatment", default="rl_trip_vad")
    parser.add_argument("--control", default="rl_trip_rand")
    args = parser.parse_args()

    report: Dict[str, object] = {"arms": args.arms, "corpora": {}}
    subgroup: Dict[str, Dict[int, float]] = {a: {} for a in args.arms}
    falsealarm: Dict[str, Dict[int, float]] = {a: {} for a in args.arms}
    headline: Dict[str, Dict[str, Dict[int, float]]] = {}

    for disp, meta_stem, pred_stem in CORPORA:
        if not (META / ("%s.npz" % meta_stem)).exists():
            print("no meta for %s, skipping" % disp)
            continue
        y, vad, _ = load_meta(meta_stem)
        preds = {a: corpus_rows(a, pred_stem, len(y)) for a in args.arms}
        headline[disp] = {"uar": {}, "acc": {}, "f1": {}}
        per_arm: Dict[str, object] = {}

        print("\n" + "=" * 70)
        print("%s  (%d rows)" % (disp, len(y)))
        print("%-16s %5s %7s %7s %7s" % ("arm", "seeds", "UAR", "Acc", "F1"))
        for arm in args.arms:
            rows = preds[arm]
            if not rows:
                print("%-16s %5s %7s" % (arm, "-", "no runs"))
                continue
            u = {s: 100 * uar(y, p) for s, p in rows.items()}
            a_ = {s: 100 * float((p == y).mean()) for s, p in rows.items()}
            f_ = {s: 100 * f1_score(y, p, average="macro")
                  for s, p in rows.items()}
            headline[disp]["uar"][arm] = u
            headline[disp]["acc"][arm] = a_
            headline[disp]["f1"][arm] = f_
            print("%-16s %5d %7.2f %7.2f %7.2f"
                  % (arm, len(u), np.mean(list(u.values())),
                     np.mean(list(a_.values())), np.mean(list(f_.values()))))
            per_arm[arm] = {"seeds": sorted(u),
                            "uar": {str(k): v for k, v in u.items()},
                            "acc": {str(k): v for k, v in a_.items()},
                            "f1": {str(k): v for k, v in f_.items()}}

            # Subgroup and boundary guard need VAD, so IEMOCAP only:
            # every SAMSEMO row carries (0.5, 0.5, 0.5), which makes any
            # arousal split on it vacuous.
            if disp == "IEMOCAP":
                happy = y == 1
                ar = vad[:, 1]
                sel = happy & (ar <= 0.5)
                neutral = y == 0
                for s, p in rows.items():
                    if sel.sum():
                        subgroup[arm][s] = 100 * float((p[sel] == 1).mean())
                    if neutral.sum():
                        falsealarm[arm][s] = 100 * float(
                            (p[neutral] == 1).mean())
        report["corpora"][disp] = per_arm

    # 1. The control contrast.
    print("\n" + "=" * 70)
    print("1. CONTROL CONTRAST: %s minus %s" % (args.treatment, args.control))
    print("   Both arms carry the term; only the VAD matching differs.")
    contrast: Dict[str, object] = {}
    for disp in headline:
        for metric in ("uar", "acc", "f1"):
            t = headline[disp][metric].get(args.treatment, {})
            c = headline[disp][metric].get(args.control, {})
            res = paired(t, c)
            if res is None:
                continue
            n, mean, per = res
            signs = "".join("+" if d > 0 else "-" for _, d in per)
            print("   %-9s %-4s n=%d  mean %+.2f  per-seed %s"
                  % (disp, metric.upper(), n, mean, signs))
            contrast["%s_%s" % (disp, metric)] = {"n": n, "mean": mean,
                                                  "per_seed": per}
    report["control_contrast"] = contrast

    # 2. The subgroup endpoint.
    print("\n" + "=" * 70)
    print("2. SUBGROUP: IEMOCAP low-arousal happy recall (arousal <= 0.5)")
    for arm in args.arms:
        if subgroup[arm]:
            vals = list(subgroup[arm].values())
            print("   %-16s n=%d  %6.2f%%  %s"
                  % (arm, len(vals), np.mean(vals),
                     " ".join("%.1f" % v for v in vals)))
    res = paired(subgroup[args.treatment], subgroup[args.control])
    if res is not None:
        n, mean, per = res
        print("   %s minus %s: n=%d  mean %+.2f pp  per-seed %s"
              % (args.treatment, args.control, n, mean,
                 "".join("+" if d > 0 else "-" for _, d in per)))
        report["subgroup_contrast"] = {"n": n, "mean": mean, "per_seed": per}

    # 3. The boundary guard.
    print("\n" + "=" * 70)
    print("3. BOUNDARY GUARD: IEMOCAP neutral clips called happy")
    print("   A gain in 2 with a matching rise here is the decision")
    print("   boundary moving, not a cue being learned.")
    for arm in args.arms:
        if falsealarm[arm]:
            vals = list(falsealarm[arm].values())
            print("   %-16s n=%d  %6.2f%%  %s"
                  % (arm, len(vals), np.mean(vals),
                     " ".join("%.1f" % v for v in vals)))
    res = paired(falsealarm[args.treatment], falsealarm[args.control])
    if res is not None:
        n, mean, per = res
        print("   %s minus %s: n=%d  mean %+.2f pp"
              % (args.treatment, args.control, n, mean))
        report["falsealarm_contrast"] = {"n": n, "mean": mean,
                                         "per_seed": per}
        sub = report.get("subgroup_contrast")
        if sub is not None and sub["mean"] > 0 and mean > 0:
            ratio = mean / sub["mean"]
            print("   false-alarm rise is %.2f times the subgroup gain%s"
                  % (ratio, "; that is a boundary shift, not a cue"
                     if ratio > 0.5 else ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
