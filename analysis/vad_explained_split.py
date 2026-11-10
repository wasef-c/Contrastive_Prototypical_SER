"""Split an evaluation corpus by whether VAD alone explains its label.

A Gaussian class-conditional classifier is fitted on the MSP-Podcast training
VAD and transferred to an evaluation corpus. Rows it classifies correctly are
the ones a VAD-shaped model can already reach; rows it misclassifies are the
population the categorical label disagrees with VAD about, which is where the
loud-sad and muted-happy utterances live.

Measured on IEMOCAP over 16 arm/control pairs, 13 arms LOSE accuracy on the
unexplained half while gaining on the explained half, so a headline UAR gain
can be entirely an artefact of the split. This script reports both halves.

Usage:
    python -m analysis.vad_explained_split --corpus IEMO
    python -m analysis.vad_explained_split --corpus IEMO --arm uf4_whitevad \
        --control uf4_ctrl_pool
"""

import argparse
import glob
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

CLASS_NAMES = ("neutral", "happy", "sad", "angry")
META_DIR = pathlib.Path("analysis/heldout_meta")
CKPT_DIR = pathlib.Path("checkpoints")
NUM_CLASSES = 4


def fit_class_gaussians(vad: np.ndarray, label: np.ndarray,
                        ridge: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a mean and covariance per class.

    Args:
        vad: [N, 3] VAD coordinates.
        label: [N] class labels in [0, NUM_CLASSES).
        ridge: Added to the covariance diagonal for conditioning.

    Returns:
        Tuple of means [C, 3] and covariances [C, 3, 3].
    """
    means = np.stack([vad[label == c].mean(axis=0) for c in range(NUM_CLASSES)])
    covs = np.stack([np.cov(vad[label == c].T) + ridge * np.eye(vad.shape[1])
                     for c in range(NUM_CLASSES)])
    return means, covs


def gaussian_log_scores(vad: np.ndarray, means: np.ndarray,
                        covs: np.ndarray) -> np.ndarray:
    """Per-class Gaussian log scores under uniform class priors.

    Uniform priors are deliberate: fitted source priors would let the
    classifier profit from source-to-target prior shift, which is not the
    property being measured here.

    Args:
        vad: [N, 3] VAD coordinates.
        means: [C, 3] class means.
        covs: [C, 3, 3] class covariances.

    Returns:
        [N, C] log scores.
    """
    out = []
    for c in range(len(means)):
        inv = np.linalg.inv(covs[c])
        _, logdet = np.linalg.slogdet(covs[c])
        d = vad - means[c]
        out.append(-0.5 * (np.einsum("ni,ij,nj->n", d, inv, d) + logdet))
    return np.stack(out, axis=1)


def uar(true: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted average recall over the classes present in true.

    Args:
        true: [N] reference labels.
        pred: [N] predicted labels.

    Returns:
        Mean per-class recall.
    """
    return float(np.mean([(pred[true == c] == c).mean()
                          for c in range(NUM_CLASSES) if (true == c).sum() > 0]))


def load_split(corpus: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an evaluation corpus and mark which rows VAD explains.

    Args:
        corpus: Corpus tag, for example "IEMO" or "MSPI".

    Returns:
        Tuple of labels [N], the VAD-explained boolean mask [N], and the
        corpus VAD [N, 3].

    Raises:
        ValueError: If the corpus carries no usable VAD annotations.
    """
    train = np.load(META_DIR / "train_all.npz", allow_pickle=True)
    keep = train["label"] >= 0
    means, covs = fit_class_gaussians(train["vad"][keep], train["label"][keep])

    data = np.load(META_DIR / f"test_{corpus}.npz", allow_pickle=True)
    mask = data["label"] >= 0
    vad, label = data["vad"][mask], data["label"][mask]
    if float(np.var(vad)) < 1e-9:
        raise ValueError(
            f"{corpus} carries no VAD annotations (zero variance), so the "
            f"VAD-explained split is undefined for it. SAMSEMO and CMUMOSEI "
            f"are in this position.")
    explained = gaussian_log_scores(vad, means, covs).argmax(axis=1) == label
    return label, explained, vad


def describe(corpus: str, label: np.ndarray, explained: np.ndarray,
             vad: np.ndarray) -> None:
    """Print the composition of the unexplained population.

    Args:
        corpus: Corpus tag.
        label: [N] reference labels.
        explained: [N] VAD-explained mask.
        vad: [N, 3] corpus VAD.
    """
    train = np.load(META_DIR / "train_all.npz", allow_pickle=True)
    keep = train["label"] >= 0
    means, covs = fit_class_gaussians(train["vad"][keep], train["label"][keep])
    pred = gaussian_log_scores(vad, means, covs).argmax(axis=1)

    print(f"{corpus}: N={len(label)}, VAD explains "
          f"{100.0 * explained.mean():.1f} percent")
    print(f"  {'true':<9}{'n':>7}{'VAD-wrong':>12}   VAD says instead")
    for c in range(NUM_CLASSES):
        sel = label == c
        wrong = sel & ~explained
        counts = np.bincount(pred[wrong], minlength=NUM_CLASSES).astype(float)
        top = np.argsort(-counts)[:2]
        desc = ", ".join(f"{CLASS_NAMES[t]} {100.0 * counts[t] / max(counts.sum(), 1):.0f}%"
                         for t in top if counts[t] > 0)
        print(f"  {CLASS_NAMES[c]:<9}{sel.sum():>7}"
              f"{100.0 * wrong.sum() / max(sel.sum(), 1):>11.1f}%   {desc}")


def logits_by_seed(arm: str, corpus: str) -> Dict[int, np.ndarray]:
    """Collect one arm's prediction logits keyed by seed.

    Args:
        arm: Arm name without the seed suffix.
        corpus: Corpus tag.

    Returns:
        Mapping of seed to [N, C] logits.
    """
    found: Dict[int, np.ndarray] = {}
    for path in glob.glob(str(CKPT_DIR / f"{arm}_seed*" / "predictions"
                              / f"{corpus}.npz")):
        match = re.search(r"_seed(\d+)", pathlib.Path(path).parts[1])
        if match:
            found[int(match.group(1))] = np.load(path, allow_pickle=True)["logits"]
    return found


def compare(arm: str, control: str, corpus: str, label: np.ndarray,
            explained: np.ndarray) -> Optional[Tuple[float, float, float, float, int]]:
    """Score one arm against its control on both halves of the split.

    Args:
        arm: Arm name.
        control: Matched control arm name.
        corpus: Corpus tag.
        label: [N] reference labels.
        explained: [N] VAD-explained mask.

    Returns:
        Tuple of (delta explained, p, delta unexplained, p, seeds), or None
        when fewer than two seeds are shared.
    """
    arm_logits, ctrl_logits = logits_by_seed(arm, corpus), logits_by_seed(control, corpus)
    seeds = sorted(set(arm_logits) & set(ctrl_logits))
    if len(seeds) < 2:
        return None
    d_exp, d_unexp = [], []
    for seed in seeds:
        pa = arm_logits[seed].argmax(axis=1)
        pc = ctrl_logits[seed].argmax(axis=1)
        d_exp.append((pa[explained] == label[explained]).mean()
                     - (pc[explained] == label[explained]).mean())
        d_unexp.append((pa[~explained] == label[~explained]).mean()
                       - (pc[~explained] == label[~explained]).mean())
    d_exp, d_unexp = 100.0 * np.array(d_exp), 100.0 * np.array(d_unexp)
    p_exp = float(stats.ttest_1samp(d_exp, 0.0).pvalue) if len(seeds) > 1 else float("nan")
    p_unexp = float(stats.ttest_1samp(d_unexp, 0.0).pvalue) if len(seeds) > 1 else float("nan")
    return float(d_exp.mean()), p_exp, float(d_unexp.mean()), p_unexp, len(seeds)


def discover_pairs() -> List[Tuple[str, str]]:
    """Default arm/control pairs used when none are given on the command line.

    Returns:
        List of (arm, control) name pairs.
    """
    return [
        ("uf4_whitevad", "uf4_ctrl_pool"),
        ("uf4_scalematch", "uf4_ctrl_pool"),
        ("uf4_res_more", "uf4_ctrl_pool"),
        ("uf4_clsbal", "uf4_ctrl_pool"),
        ("uf4_maskneu", "uf4_ctrl_pool"),
        ("uf4_agree_lam", "uf4_ctrl_pool"),
        ("uf4_rawvad", "uf4_ctrl_pool"),
        ("uf4_offlabel", "uf4_ctrl_pool"),
        ("uf4_disp", "uf4_ctrl_pool"),
        ("uf4_euclid", "uf4_ctrl_pool"),
        ("uf4_symw", "uf4_ctrl_pool"),
        ("uf4_auxvad", "uf4_ctrl_pool"),
        ("uf4_vadtrust", "uf4_vadtrust_ctrl"),
        ("uf4_vadtrust", "uf4_ctrl_pool"),
        ("uf4_ctrl_pool", "uf4_base_more"),
        ("pl_res", "pl_ctrl"),
        ("cw_sqrt_res_more", "cw_sqrt_ctrl"),
    ]


def main() -> None:
    """Report the VAD-explained split and per-arm deltas on both halves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="IEMO",
                        help="corpus tag with VAD annotations (IEMO or MSPI)")
    parser.add_argument("--arm", default=None, help="single arm to score")
    parser.add_argument("--control", default="uf4_ctrl_pool",
                        help="control arm for --arm")
    args = parser.parse_args()

    label, explained, vad = load_split(args.corpus)
    describe(args.corpus, label, explained, vad)
    print()

    pairs = [(args.arm, args.control)] if args.arm else discover_pairs()
    print(f"  {'arm':<22}{'vs':<18}{'n':>3}{'d EXPLAINED':>18}{'d UNEXPLAINED':>18}")
    print("  " + "-" * 77)
    negative = 0
    total = 0
    for arm, control in pairs:
        got = compare(arm, control, args.corpus, label, explained)
        if got is None:
            continue
        d_exp, p_exp, d_unexp, p_unexp, n_seeds = got
        star_e = "*" if p_exp < 0.05 else " "
        star_u = "*" if p_unexp < 0.05 else " "
        print(f"  {arm:<22}{control:<18}{n_seeds:>3}"
              f"{d_exp:>+11.2f} {star_e} p={p_exp:.3f}"
              f"{d_unexp:>+11.2f} {star_u} p={p_unexp:.3f}")
        total += 1
        if d_unexp < 0:
            negative += 1
    if total:
        print()
        print(f"  {negative} of {total} arms lose accuracy on the VAD-unexplained "
              f"half (chance would be {total / 2:.0f}).")


if __name__ == "__main__":
    main()
