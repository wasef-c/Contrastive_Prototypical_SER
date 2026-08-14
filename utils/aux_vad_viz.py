#!/usr/bin/env python3
"""
End-of-training visualization for the aux VAD cluster head.

Produces one PNG with six panels:
  1-3. Valence vs Arousal, V vs D, A vs D scatter, colored by cluster ID,
       with centroids marked as black X.
  4.   Cluster x class contingency heatmap (which VAD clusters correspond to
       which primary emotion classes).
  5.   Per-cluster boxplot of prototypicality (VAD distance to the sample's
       expected class centroid). Prototypicality is the difficulty concept
       from utils.prototypicality: lower = closer to prototype, higher =
       atypical. If clusters are semantically meaningful this plot separates
       "typical" clusters (low median) from "atypical" clusters (high median).
  6.   Cluster x prototypicality-tercile heatmap. Terciles are computed over
       the whole VAD-annotated training subset (bottom 33% = proto, middle,
       top 33% = atyp). Directly shows whether each cluster overrepresents
       proto or atyp samples relative to the base rate.

Also prints a per-cluster summary to console: mean difficulty, dominant
emotion class, and prototypicality tercile distribution.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.prototypicality import DATASETS_WITH_VAD, calculate_difficulty


def _collect_points(
    train_data: Sequence[dict],
    num_classes: int,
    expected_vad: Optional[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (points [M,3], labels [M], difficulty [M]) for VAD samples only.

    Difficulty is VAD Euclidean distance to expected_vad[label]. If
    expected_vad is None or missing a label, difficulty defaults to NaN so
    plotting can skip it cleanly.
    """
    points = []
    labels = []
    diffs = []
    for item in train_data:
        if item.get("dataset") not in DATASETS_WITH_VAD:
            continue
        c = int(item["label"])
        if c < 0 or c >= num_classes:
            continue
        points.append([
            float(item["valence"]),
            float(item["arousal"]),
            float(item["dominance"]),
        ])
        labels.append(c)
        if expected_vad is not None and c in expected_vad:
            d = calculate_difficulty(
                item["valence"], item["arousal"], item["dominance"],
                c, expected_vad,
            )
        else:
            d = float("nan")
        diffs.append(d)
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(diffs, dtype=np.float32),
    )


def _assign_clusters(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Argmin distance from each point to each centroid; returns [M] int."""
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    diff = points[:, None, :] - centroids[None, :, :]
    dist_sq = (diff * diff).sum(axis=-1)
    return dist_sq.argmin(axis=-1).astype(np.int64)


def _assign_per_class_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Assign each point to the nearest centroid within its own class.

    Args:
        points: [M, 3] VAD points.
        labels: [M] class labels aligned with points.
        centroids: [num_classes, clusters_per_class, 3] centroids.

    Returns:
        [M] int array of flat subtype IDs
        (label * clusters_per_class + local_cluster).
    """
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    clusters_per_class = centroids.shape[1]
    own = centroids[labels]  # [M, clusters_per_class, 3]
    diff = points[:, None, :] - own
    local = (diff * diff).sum(axis=-1).argmin(axis=-1)
    return (labels * clusters_per_class + local).astype(np.int64)


def _print_cluster_summary(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    difficulty: np.ndarray,
    tercile_edges: Optional[Tuple[float, float]],
    num_classes: int,
    k: int,
) -> None:
    """Print a compact per-cluster summary to stdout."""
    print("  Per-cluster summary (aux VAD cluster head):")
    header = (
        f"    {'cluster':>7}  {'n':>5}  {'mean_diff':>9}  {'median_diff':>11}  "
        f"{'dom_class':>9}  {'proto%':>7}  {'atyp%':>6}"
    )
    print(header)
    for c in range(k):
        m = cluster_ids == c
        n = int(m.sum())
        if n == 0:
            print(f"    {c:>7}  {0:>5}")
            continue
        d = difficulty[m]
        finite = d[np.isfinite(d)]
        mean_d = float(finite.mean()) if finite.size > 0 else float("nan")
        med_d = float(np.median(finite)) if finite.size > 0 else float("nan")
        cls_counts = np.bincount(labels[m], minlength=num_classes)
        dom = int(cls_counts.argmax())
        dom_pct = float(cls_counts[dom]) / n * 100.0

        proto_pct = atyp_pct = float("nan")
        if tercile_edges is not None and finite.size > 0:
            lo, hi = tercile_edges
            proto_pct = float((finite <= lo).sum()) / finite.size * 100.0
            atyp_pct = float((finite > hi).sum()) / finite.size * 100.0

        print(
            f"    {c:>7}  {n:>5}  {mean_d:>9.3f}  {med_d:>11.3f}  "
            f"{dom:>3} ({dom_pct:>3.0f}%)  {proto_pct:>6.1f}  {atyp_pct:>5.1f}"
        )


def save_cluster_visualization(
    train_data: Sequence[dict],
    centroids: np.ndarray,
    num_classes: int,
    out_path: Path,
    title: str = "",
    expected_vad: Optional[dict] = None,
    per_class_scope: bool = False,
) -> None:
    """Render and save the 6-panel figure and print per-cluster stats.

    Args:
        train_data: list of dataset item dicts (train_dataset.data).
        centroids: [k, 3] VAD cluster centroids, or
            [num_classes, clusters_per_class, 3] when per_class_scope.
        num_classes: number of primary classes.
        out_path: destination PNG path. Parent directory is created.
        title: figure suptitle (usually the experiment name).
        expected_vad: dict mapping label -> [V, A, D] used to compute
            per-sample prototypicality. If None, the prototypicality
            panels degrade gracefully.
        per_class_scope: True when centroids were fit inside each class.
            Points are then assigned within their own class and cluster
            IDs are flat subtype labels.
    """
    points, labels, difficulty = _collect_points(train_data, num_classes, expected_vad)

    if points.shape[0] == 0:
        print("  aux_vad_viz: no VAD-annotated samples, skipping figure.")
        return

    if per_class_scope:
        cluster_ids = _assign_per_class_clusters(points, labels, centroids)
        k = int(centroids.shape[0] * centroids.shape[1])
        # Flatten for plotting: subtype j of class c sits at row
        # c * clusters_per_class + j, matching the flat cluster IDs.
        centroids = centroids.reshape(k, centroids.shape[-1])
    else:
        k = int(centroids.shape[0])
        cluster_ids = _assign_clusters(points, centroids)

    finite_diff = difficulty[np.isfinite(difficulty)]
    tercile_edges = None
    if finite_diff.size >= 3:
        lo = float(np.percentile(finite_diff, 33.3))
        hi = float(np.percentile(finite_diff, 66.7))
        tercile_edges = (lo, hi)

    _print_cluster_summary(
        cluster_ids, labels, difficulty, tercile_edges, num_classes, k,
    )

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    dim_names = ["Valence", "Arousal", "Dominance"]
    pair_axes = [(0, 1), (0, 2), (1, 2)]
    cmap = plt.get_cmap("tab20" if k > 10 else "tab10")

    for panel_i, (ax_pos, (dx, dy)) in enumerate(zip(
            [axes[0, 0], axes[0, 1], axes[0, 2]], pair_axes)):
        for c in range(k):
            m = cluster_ids == c
            if not m.any():
                continue
            ax_pos.scatter(
                points[m, dx], points[m, dy],
                s=8, alpha=0.35, color=cmap(c % cmap.N),
                label=f"c{c}" if panel_i == 0 else None,
            )
        ax_pos.scatter(
            centroids[:, dx], centroids[:, dy],
            s=100, marker="X", color="black", edgecolors="white", linewidths=1.2,
            zorder=5,
        )
        ax_pos.set_xlabel(dim_names[dx])
        ax_pos.set_ylabel(dim_names[dy])
        ax_pos.set_title(f"{dim_names[dx]} vs {dim_names[dy]}")
        ax_pos.grid(True, alpha=0.2)

    axes[0, 0].legend(
        loc="upper left", bbox_to_anchor=(-0.35, 1.0),
        fontsize=8, frameon=False,
    )

    contingency = np.zeros((k, num_classes), dtype=np.int64)
    for c_id, y in zip(cluster_ids, labels):
        contingency[int(c_id), int(y)] += 1

    ax = axes[1, 0]
    im = ax.imshow(contingency, aspect="auto", cmap="viridis")
    ax.set_xlabel("Emotion class")
    ax.set_ylabel("VAD cluster")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(k))
    ax.set_title("Cluster x class counts")
    for i in range(k):
        for j in range(num_classes):
            val = int(contingency[i, j])
            if val == 0:
                continue
            color = "white" if val < contingency.max() * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    if finite_diff.size == 0:
        ax.text(0.5, 0.5, "no expected_vad -> no prototypicality",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        box_data = []
        positions = []
        for c in range(k):
            m = (cluster_ids == c) & np.isfinite(difficulty)
            if m.any():
                box_data.append(difficulty[m])
                positions.append(c)
        bp = ax.boxplot(
            box_data, positions=positions, widths=0.6, patch_artist=True,
            showfliers=False,
        )
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap(positions[j] % cmap.N))
            patch.set_alpha(0.6)
        ax.axhline(np.median(finite_diff), color="black",
                   linestyle="--", linewidth=1.0, label="overall median")
        if tercile_edges is not None:
            ax.axhline(tercile_edges[0], color="gray",
                       linestyle=":", linewidth=0.8, label="proto/mid edge")
            ax.axhline(tercile_edges[1], color="gray",
                       linestyle=":", linewidth=0.8, label="mid/atyp edge")
        ax.set_xlabel("VAD cluster")
        ax.set_ylabel("Prototypicality distance")
        ax.set_title("Per-cluster prototypicality")
        ax.set_xticks(range(k))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1, 2]
    if tercile_edges is None:
        ax.text(0.5, 0.5, "no expected_vad -> no tercile heatmap",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        lo, hi = tercile_edges
        tercile_ids = np.full(difficulty.shape, -1, dtype=np.int64)
        tercile_ids[np.isfinite(difficulty) & (difficulty <= lo)] = 0
        tercile_ids[np.isfinite(difficulty) & (difficulty > lo) & (difficulty <= hi)] = 1
        tercile_ids[np.isfinite(difficulty) & (difficulty > hi)] = 2

        tercile_counts = np.zeros((k, 3), dtype=np.int64)
        for c_id, t_id in zip(cluster_ids, tercile_ids):
            if t_id < 0:
                continue
            tercile_counts[int(c_id), int(t_id)] += 1

        row_totals = tercile_counts.sum(axis=1, keepdims=True).clip(min=1)
        tercile_frac = tercile_counts / row_totals

        im2 = ax.imshow(tercile_frac, aspect="auto", cmap="magma",
                        vmin=0.0, vmax=1.0)
        ax.set_xlabel("Prototypicality tercile")
        ax.set_ylabel("VAD cluster")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["proto", "mid", "atyp"])
        ax.set_yticks(range(k))
        ax.set_title("Cluster x prototypicality tercile (row-normalized)")
        for i in range(k):
            for j in range(3):
                val = tercile_frac[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1.0, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved aux VAD cluster viz to {out_path}")
