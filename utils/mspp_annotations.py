#!/usr/bin/env python3
"""Per-utterance annotator distributions from the official MSP-Podcast release.

The consensus CSV in the official release is row-for-row identical to the
cairocode/MSPP_WAV dataset used for training (116,221 rows, same order, same
FileName and same A/V/D), so the detailed per-annotator labels can be attached
to the existing pipeline without changing the training set or invalidating any
baseline.

What this provides that the pipeline did not have:

    label_dist   fraction of annotators choosing each of the four classes.
                 The real target distribution, rather than the arg-max
                 consensus the model currently trains on.
    entropy      Shannon entropy of that distribution: how contested the
                 utterance is. Measured against the fitted-Gaussian residual
                 the pipeline has been using, human disagreement correlates
                 at only about 0.17, so this is genuinely different
                 information rather than a re-parameterisation.
    vad_mean     per-annotator A/V/D averaged, and
    vad_std      their dispersion, computed from the raw ratings rather than
                 taken from a precomputed column.
    subtype_dist fraction of annotators applying each secondary tag. These
                 are annotator-supplied subtypes within a coarse class
                 (Amused/Excited inside happy, Depressed/Disappointed inside
                 sad, Frustrated/Annoyed inside angry, Concerned alongside
                 neutral), which is the structure the VAD k-means arms were
                 trying to recover indirectly.
    n_annotators every utterance has at least five, but the count ranges up
                 to 32, so any statistic derived here should be weighted or
                 filtered by it.
"""

from __future__ import annotations

import csv
import pathlib
import re
from typing import Dict, List, Optional

import numpy as np

LABELS_DIR = pathlib.Path("/home/rml/Documents/pythontest/mspp/Labels")
CACHE_DIR = pathlib.Path("cache/mspp_annotations")

# Primary annotator emotions that map onto the four training classes. Every
# other primary label (Contempt, Surprise, Disgust, Fear, Other-*) is counted
# in `other_frac` rather than forced into one of the four.
PRIMARY_TO_CLASS = {"Neutral": 0, "Happy": 1, "Sad": 2, "Angry": 3}

# Secondary tags kept as subtype targets, grouped by the coarse class they sit
# under. Chosen by frequency from the release: each appears at least 26k times.
SUBTYPES = ["Concerned", "Frustrated", "Angry", "Sad", "Amused", "Annoyed",
            "Excited", "Disappointed", "Contempt", "Surprise", "Disgust",
            "Confused", "Depressed", "Fear", "Happy", "Neutral"]

_AVD = re.compile(r"^([AVD]):([\d.]+)$")


def parse_detailed(labels_dir: Optional[pathlib.Path] = None,
                   cache_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Parse labels_detailed.csv into per-utterance arrays and cache them.

    Args:
        labels_dir: directory holding labels_detailed.csv and
            labels_consensus.csv.
        cache_dir: where to write the npz.

    Returns:
        Path to the cached npz, keyed by FileName.
    """
    labels_dir = labels_dir or LABELS_DIR
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "mspp_annotator_stats.npz"
    if out.exists():
        return out

    sub_index = {s: i for i, s in enumerate(SUBTYPES)}
    per_file: Dict[str, dict] = {}

    with open(labels_dir / "labels_detailed.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            fn, detail = row[0], row[1]
            parts = [p.strip() for p in detail.split(";") if p.strip()]
            if len(parts) < 2:
                continue
            primary = parts[1]
            # The secondary field is absent on some rows; A/V/D always carry a
            # colon, which distinguishes them from a tag list.
            secondary = parts[2] if len(parts) > 2 and ":" not in parts[2] else ""
            avd = {}
            for p in parts:
                m = _AVD.match(p)
                if m:
                    avd[m.group(1)] = float(m.group(2))

            e = per_file.setdefault(fn, {
                "counts": np.zeros(4, dtype=np.float64),
                "other": 0.0,
                "n": 0,
                "avd": [],
                "sub": np.zeros(len(SUBTYPES), dtype=np.float64),
            })
            cls = PRIMARY_TO_CLASS.get(primary)
            if cls is None:
                e["other"] += 1.0
            else:
                e["counts"][cls] += 1.0
            e["n"] += 1
            if {"A", "V", "D"} <= set(avd):
                e["avd"].append((avd["V"], avd["A"], avd["D"]))
            for s in secondary.split(","):
                s = s.strip()
                if s in sub_index:
                    e["sub"][sub_index[s]] += 1.0

    keys = sorted(per_file)
    n = len(keys)
    label_dist = np.zeros((n, 4), dtype=np.float32)
    entropy = np.zeros(n, dtype=np.float32)
    other_frac = np.zeros(n, dtype=np.float32)
    vad_mean = np.zeros((n, 3), dtype=np.float32)
    vad_std = np.zeros((n, 3), dtype=np.float32)
    subtype = np.zeros((n, len(SUBTYPES)), dtype=np.float32)
    n_ann = np.zeros(n, dtype=np.int32)

    for i, k in enumerate(keys):
        e = per_file[k]
        total = max(e["n"], 1)
        n_ann[i] = e["n"]
        other_frac[i] = e["other"] / total
        four = e["counts"].sum()
        # Distribution over the four training classes only; utterances whose
        # annotators mostly chose something else are flagged by other_frac.
        label_dist[i] = (e["counts"] / four) if four > 0 else 0.25
        p = label_dist[i][label_dist[i] > 0]
        entropy[i] = float(-(p * np.log(p)).sum())
        subtype[i] = e["sub"] / total
        if e["avd"]:
            a = np.asarray(e["avd"], dtype=np.float64)
            vad_mean[i] = a.mean(axis=0)
            # Population std: with five raters the sample correction is noise.
            vad_std[i] = a.std(axis=0)

    np.savez_compressed(
        out, keys=np.array(keys), label_dist=label_dist, entropy=entropy,
        other_frac=other_frac, vad_mean=vad_mean, vad_std=vad_std,
        subtype=subtype, n_annotators=n_ann,
        subtype_names=np.array(SUBTYPES),
    )
    return out


def load_stats(cache_path: Optional[pathlib.Path] = None) -> dict:
    """Load the cached per-utterance annotator statistics.

    Args:
        cache_path: npz written by parse_detailed.

    Returns:
        Dict of arrays plus `index`, a FileName -> row mapping.
    """
    path = cache_path or (CACHE_DIR / "mspp_annotator_stats.npz")
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["index"] = {str(k): i for i, k in enumerate(z["keys"])}
    return out
