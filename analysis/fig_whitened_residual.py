"""Figure: what the class-whitened VAD residual measures.

Draws the auxiliary target on real MSP-Podcast statistics rather than a
schematic. The point the figure has to make is why centroid subtraction alone
is not enough, which is the result the ablation shows: the unwhitened residual
moves neutral-vs-rest AUC by +0.07 while the whitened one moves it by +1.33.

Left panel, raw valence-arousal space. Each class has its own centroid and its
own scatter, and those scatters differ in both size and orientation: on the
training split, happy has a valence-arousal correlation of +0.43 while angry
has -0.33. Two utterances can sit the same Euclidean distance from their
respective centroids and yet be differently unusual for their own class.

Right panel, after whitening by the class covariance. Every class becomes an
isotropic unit ball, so a residual of a given length means the same thing
whatever the class produced it: this many within-class standard deviations,
in this direction. That comparability is what the head can learn from.

Usage:
    python analysis/fig_whitened_residual.py [--out paper/figures]
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

CLASS_NAMES: Tuple[str, ...] = ("neutral", "happy", "sad", "angry")
# Colourblind-safe, print-legible.
CLASS_COLOURS: Tuple[str, ...] = ("#4C72B0", "#DD8452", "#55A868", "#C44E52")
SCRATCH = ("/tmp/claude-1000/-home-rml-Documents-pythontest-Emotion2Vec-"
           "Contrastive/f291c2af-b101-4064-aa1e-73a12ab2a49b/scratchpad")


def whitening(cov: np.ndarray) -> np.ndarray:
    """Inverse matrix square root of a covariance.

    Args:
        cov: [d, d] symmetric positive definite covariance.

    Returns:
        [d, d] matrix W with W cov W^T equal to the identity.
    """
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def ellipse_from_cov(mean: np.ndarray, cov: np.ndarray, n_sd: float,
                     **kwargs) -> Ellipse:
    """Build a covariance ellipse at n_sd standard deviations."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_sd * np.sqrt(vals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def main() -> None:
    """Render the two-panel figure to PDF and PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    labels = np.load(f"{SCRATCH}/fig_lab.npy")
    vad = np.load(f"{SCRATCH}/fig_vad.npy")[:, :2]   # valence, arousal

    means = [vad[labels == c].mean(axis=0) for c in range(4)]
    covs = [np.cov(vad[labels == c].T) for c in range(4)]
    whit = [whitening(cov) for cov in covs]

    # Two utterances chosen to sit at nearly the same Euclidean distance from
    # their own centroid, in classes whose scatter differs most in
    # orientation. They are the figure's argument: equally far in raw space,
    # differently unusual once the class scatter is accounted for.
    picks = []
    for cls, direction in ((1, np.array([0.0, 1.0])),      # happy, high arousal
                           (3, np.array([0.0, 1.0]))):     # angry, high arousal
        point = means[cls] + 1.35 * direction
        picks.append((cls, point))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    ax = axes[0]
    for c in range(4):
        subset = vad[labels == c]
        idx = np.random.RandomState(0).choice(len(subset),
                                              size=min(400, len(subset)),
                                              replace=False)
        ax.scatter(subset[idx, 0], subset[idx, 1], s=3, alpha=0.16,
                   color=CLASS_COLOURS[c], linewidths=0)
        for n_sd in (1.0, 2.0):
            ax.add_patch(ellipse_from_cov(
                means[c], covs[c], n_sd, facecolor="none",
                edgecolor=CLASS_COLOURS[c], lw=1.2,
                alpha=0.9 if n_sd == 1.0 else 0.45))
        ax.plot(*means[c], marker="+", ms=9, mew=2, color=CLASS_COLOURS[c])
        ax.annotate(CLASS_NAMES[c], means[c], textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=CLASS_COLOURS[c],
                    fontweight="bold")
    for cls, point in picks:
        ax.annotate("", xy=point, xytext=means[cls],
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
        ax.plot(*point, marker="o", ms=6, color="black", zorder=5)
    ax.set_xlabel("valence")
    ax.set_ylabel("arousal")
    ax.set_title(r"raw space:  $v - \mu_c$", fontsize=10)

    ax = axes[1]
    for c in range(4):
        subset = vad[labels == c]
        idx = np.random.RandomState(0).choice(len(subset),
                                              size=min(400, len(subset)),
                                              replace=False)
        z = (subset[idx] - means[c]) @ whit[c].T
        ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.16,
                   color=CLASS_COLOURS[c], linewidths=0)
    for n_sd in (1.0, 2.0):
        ax.add_patch(plt.Circle((0, 0), n_sd, facecolor="none",
                                edgecolor="0.35", lw=1.2, ls="--",
                                alpha=0.9 if n_sd == 1.0 else 0.5))
    ax.plot(0, 0, marker="+", ms=9, mew=2, color="0.2")
    for cls, point in picks:
        z = whit[cls] @ (point - means[cls])
        ax.annotate("", xy=z, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.8,
                                    color=CLASS_COLOURS[cls]))
        ax.plot(*z, marker="o", ms=6, color=CLASS_COLOURS[cls], zorder=5)
        ax.annotate(f"{CLASS_NAMES[cls]}: {np.linalg.norm(z):.2f} sd",
                    z, textcoords="offset points", xytext=(8, -2),
                    fontsize=9, color=CLASS_COLOURS[cls])
    lim = 3.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("whitened valence")
    ax.set_ylabel("whitened arousal")
    ax.set_title(r"whitened:  $W_c\,(v - \mu_c)$", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"whitened_residual.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")

    # The numbers the caption should quote, printed so they are not retyped.
    print("\nfor the caption:")
    for c in range(4):
        sd = np.sqrt(np.diag(covs[c]))
        corr = covs[c][0, 1] / (sd[0] * sd[1])
        print(f"  {CLASS_NAMES[c]:8s} mu=({means[c][0]:.2f}, {means[c][1]:.2f})  "
              f"sd=({sd[0]:.2f}, {sd[1]:.2f})  corr={corr:+.2f}")
    for cls, point in picks:
        raw = np.linalg.norm(point - means[cls])
        z = np.linalg.norm(whit[cls] @ (point - means[cls]))
        print(f"  marked {CLASS_NAMES[cls]:8s} raw distance {raw:.2f}, "
              f"whitened {z:.2f} sd")


if __name__ == "__main__":
    main()
