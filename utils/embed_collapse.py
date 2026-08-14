#!/usr/bin/env python3
"""
Within-class embedding collapse metrics for the aux VAD mechanism study.

Hypothesis: auxiliary VAD supervision improves cross-corpus transfer by
preventing the encoder from collapsing each emotion class toward a point.
These metrics quantify collapse on a labeled evaluation set. Both are
scale-invariant so arms with differently scaled embeddings stay comparable.

    within_var_ratio  mean within-class variance divided by total variance.
                      Lower = classes collapsed to points relative to the
                      overall spread of the embedding cloud.
    effective_rank    participation ratio of the within-class covariance
                      eigenvalue spectrum, (sum lambda)^2 / sum lambda^2,
                      averaged over classes. Ranges 1..D. Lower = the
                      within-class variation that remains is confined to
                      fewer directions.

Prediction under the collapse-prevention mechanism: baseline and
permuted-label arms show the lowest values, partition arms the highest,
and the values correlate with cross-corpus UAR across arms.
"""

from typing import Dict

import numpy as np


def collapse_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Compute within-class collapse metrics over an embedding set.

    Args:
        embeddings: [N, D] float array of raw fused embeddings.
        labels: [N] int array of class labels aligned with embeddings.
        num_classes: number of primary classes.

    Returns:
        Dict with keys:
            within_var_ratio: mean within-class variance / total variance.
            effective_rank: mean per-class participation ratio of the
                within-class covariance eigenvalues.
        Classes with fewer than 2 samples are skipped. If nothing is
        computable both values are NaN.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)

    if emb.shape[0] < 2:
        return {"within_var_ratio": float("nan"),
                "effective_rank": float("nan")}

    total_var = float(emb.var(axis=0).sum())
    if total_var <= 0.0:
        return {"within_var_ratio": float("nan"),
                "effective_rank": float("nan")}

    within_vars = []
    eff_ranks = []
    for c in range(num_classes):
        m = lab == c
        if int(m.sum()) < 2:
            continue
        class_emb = emb[m]
        centered = class_emb - class_emb.mean(axis=0, keepdims=True)
        # Per-dim variance sums to the trace of the covariance matrix.
        within_vars.append(float(centered.var(axis=0).sum()))

        # Eigenvalues of the covariance via singular values of the
        # centered matrix: lambda_i = s_i^2 / (n - 1). The participation
        # ratio only needs relative magnitudes.
        s = np.linalg.svd(centered, compute_uv=False)
        lam = s ** 2
        lam_sum = float(lam.sum())
        if lam_sum <= 0.0:
            eff_ranks.append(1.0)
        else:
            eff_ranks.append(float(lam_sum ** 2 / (lam ** 2).sum()))

    if len(within_vars) == 0:
        return {"within_var_ratio": float("nan"),
                "effective_rank": float("nan")}

    return {
        "within_var_ratio": float(np.mean(within_vars)) / total_var,
        "effective_rank": float(np.mean(eff_ranks)),
    }
