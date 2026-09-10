"""Figure: the class-whitened VAD residual, drawn in the full three dimensions.

The auxiliary target is three dimensional, so a two-axis projection understates
what the whitening does. Here each class is drawn as its one-standard-deviation
covariance ellipsoid in valence-arousal-dominance space, then again after the
per-class whitening, where every ellipsoid becomes the same unit sphere.

The transform matches training exactly: the Cholesky factor L of the inverse
class covariance, so that L L^T equals inv(cov) and the whitened residual has
identity covariance within each class. Cholesky rather than the symmetric
matrix square root matters for the components the head sees, though not for
their norm, which is the Mahalanobis distance under either.

Usage:
    python analysis/fig_whitened_3d.py [--out paper/figures]
"""

from __future__ import annotations

import argparse
import os

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prototypicality import whitening_matrices

CLASS_NAMES = ("neutral", "happy", "sad", "angry")
CLASS_COLOURS = ("#3E6E8E", "#C97B3C", "#4F8A63", "#B5484C")
AXIS_NAMES = ("valence", "arousal", "dominance")
SCRATCH = ("/tmp/claude-1000/-home-rml-Documents-pythontest-Emotion2Vec-"
           "Contrastive/f291c2af-b101-4064-aa1e-73a12ab2a49b/scratchpad")


def ellipsoid(mean: np.ndarray, cov: np.ndarray, n_sd: float = 1.0,
              res: int = 40):
    """Surface coordinates of a covariance ellipsoid at n_sd.

    Args:
        mean: [3] centre.
        cov: [3, 3] covariance.
        n_sd: radius in standard deviations.
        res: angular resolution.

    Returns:
        Tuple of [res, res] x, y, z coordinate grids.
    """
    vals, vecs = np.linalg.eigh(cov)
    radii = n_sd * np.sqrt(vals)
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    sphere = np.stack([np.outer(np.cos(u), np.sin(v)),
                       np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v))])
    pts = (vecs @ (radii[:, None] * sphere.reshape(3, -1))) + mean[:, None]
    return [c.reshape(res, res) for c in pts]


def main() -> None:
    """Render the two-panel three-dimensional figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    labels = np.load(f"{SCRATCH}/fig_lab.npy")
    vad = np.load(f"{SCRATCH}/fig_vad.npy")

    means = [vad[labels == c].mean(axis=0) for c in range(4)]
    covs = [np.cov(vad[labels == c].T) for c in range(4)]
    invs = np.stack([np.linalg.inv(cov) for cov in covs])
    chol = whitening_matrices(invs)          # exactly what training uses

    fig = plt.figure(figsize=(11.0, 4.7))
    rng = np.random.RandomState(0)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for c in range(4):
        pts = vad[labels == c]
        idx = rng.choice(len(pts), size=min(150, len(pts)), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=2.5, alpha=0.14,
                   color=CLASS_COLOURS[c], linewidths=0, depthshade=False)
        x, y, z = ellipsoid(means[c], covs[c])
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, linewidth=0.55,
                          color=CLASS_COLOURS[c], alpha=0.55)
        ax.scatter(*means[c], s=45, marker="+", color=CLASS_COLOURS[c],
                   linewidths=2.2, depthshade=False, label=CLASS_NAMES[c])
    ax.set_title("original VAD\n" + r"$v$", fontsize=11, pad=0,
                 linespacing=1.7)
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(-0.06, 0.98))

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for c in range(4):
        pts = vad[labels == c]
        idx = rng.choice(len(pts), size=min(150, len(pts)), replace=False)
        z_pts = (pts[idx] - means[c]) @ chol[c]
        ax2.scatter(z_pts[:, 0], z_pts[:, 1], z_pts[:, 2], s=2.5, alpha=0.10,
                    color=CLASS_COLOURS[c], linewidths=0, depthshade=False)
    # One sphere, not four. All four classes map onto the same unit ball, and
    # drawing four coincident wireframes renders as an indistinct blob that
    # reads as a single muddy shape rather than as the point being made.
    x, y, z = ellipsoid(np.zeros(3), np.eye(3))
    ax2.plot_wireframe(x, y, z, rstride=4, cstride=4, linewidth=0.6,
                       color="0.35", alpha=0.5)
    ax2.scatter(0, 0, 0, s=45, marker="+", color="0.25", linewidths=2.2,
                depthshade=False)
    # Spell out that four separate transforms produced this, not one. Four
    # ellipsoids collapsing to one sphere otherwise reads as a single global
    # mapping, which would be the wrong idea entirely.
    ax2.text2D(0.02, 0.93, "per class $L_c$, $\\mu_c$",
               transform=ax2.transAxes, fontsize=9, color="0.4")
    ax2.set_title("whitened residual\n" + r"$L_c^{\top}(v-\mu_c)$",
                  fontsize=11, pad=0, linespacing=1.7)
    # Frame each panel on its own content. The ellipsoid extents, not the
    # scatter tails, decide the raw panel: a handful of far outliers otherwise
    # set the limits and leave the shapes tiny.
    lo = np.min([m - 2.4 * np.sqrt(np.diag(c))
                 for m, c in zip(means, covs)], axis=0)
    hi = np.max([m + 2.4 * np.sqrt(np.diag(c))
                 for m, c in zip(means, covs)], axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    for a in (ax2,):
        a.set_xlim(-2.3, 2.3)
        a.set_ylim(-2.3, 2.3)
        a.set_zlim(-2.3, 2.3)

    for a, prefix in ((ax, ""), (ax2, "whitened ")):
        a.set_xlabel(prefix + AXIS_NAMES[0], fontsize=8.5, labelpad=-2)
        a.set_ylabel(prefix + AXIS_NAMES[1], fontsize=8.5, labelpad=-2)
        a.set_zlabel(prefix + AXIS_NAMES[2], fontsize=8.5, labelpad=-2)
        a.tick_params(labelsize=7, pad=0)
        a.view_init(elev=18, azim=-58)
        a.set_box_aspect((1, 1, 0.92), zoom=1.28)
        a.grid(True, alpha=0.18)
        for pane in (a.xaxis, a.yaxis, a.zaxis):
            pane.pane.set_alpha(0.04)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"whitened_residual_3dsurf.{ext}")
        fig.savefig(path, dpi=190, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
