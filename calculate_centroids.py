#!/usr/bin/env python3
"""
Calculate emotion centroids (mean VAD values per class) from the datasets.
Saves per-dataset and overall centroids to CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from data.dataset import EmotionDataset


def calculate_dataset_centroids(dataset_name):
    """
    Calculate emotion centroids for a single dataset

    Args:
        dataset_name: "IEMO", "MSPI", or "MSPP"

    Returns:
        dict: {emotion_label: [mean_valence, mean_arousal, mean_dominance]}
    """
    print(f"\nProcessing {dataset_name}...")

    # Load dataset from HuggingFace
    dataset = EmotionDataset(
        dataset_name=dataset_name,
        split="train",
        config=None,
        task_type="classification"  # Just to load the data
    )

    # Group by emotion label
    emotion_vads = defaultdict(lambda: {'valence': [], 'arousal': [], 'dominance': []})

    skipped_count = 0
    for sample in dataset.data:
        label = sample['label']
        valence = sample.get('valence')
        arousal = sample.get('arousal')
        dominance = sample.get('dominance')

        # Skip samples with missing VAD
        if valence is None or arousal is None or dominance is None:
            skipped_count += 1
            continue

        # Skip NaN values
        if any(isinstance(v, float) and np.isnan(v) for v in [valence, arousal, dominance]):
            skipped_count += 1
            continue

        emotion_vads[label]['valence'].append(valence)
        emotion_vads[label]['arousal'].append(arousal)
        emotion_vads[label]['dominance'].append(dominance)

    # Calculate means
    centroids = {}
    for label, vads in emotion_vads.items():
        if len(vads['valence']) > 0:
            mean_v = np.mean(vads['valence'])
            mean_a = np.mean(vads['arousal'])
            mean_d = np.mean(vads['dominance'])
            centroids[label] = [mean_v, mean_a, mean_d]

            print(f"  Emotion {label}: n={len(vads['valence'])}, "
                  f"VAD=[{mean_v:.3f}, {mean_a:.3f}, {mean_d:.3f}]")

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} samples with missing VAD")

    return centroids


def calculate_overall_centroids(all_centroids):
    """
    Calculate overall centroids by averaging across datasets

    Args:
        all_centroids: dict of {dataset_name: {label: [v, a, d]}}

    Returns:
        dict: {label: [mean_v, mean_a, mean_d]}
    """
    # Collect all labels
    all_labels = set()
    for dataset_centroids in all_centroids.values():
        all_labels.update(dataset_centroids.keys())

    # Average across datasets
    overall = {}
    for label in sorted(all_labels):
        vads = []
        for dataset_name, centroids in all_centroids.items():
            if label in centroids:
                vads.append(centroids[label])

        if vads:
            overall[label] = np.mean(vads, axis=0).tolist()

    return overall


def save_centroids_to_csv(all_centroids, overall_centroids, output_path):
    """
    Save centroids to CSV file

    Args:
        all_centroids: dict of {dataset_name: {label: [v, a, d]}}
        overall_centroids: dict of {label: [v, a, d]}
        output_path: Path to save CSV
    """
    rows = []

    # Per-dataset centroids
    for dataset_name in sorted(all_centroids.keys()):
        centroids = all_centroids[dataset_name]
        for label in sorted(centroids.keys()):
            v, a, d = centroids[label]
            rows.append({
                'dataset': dataset_name,
                'emotion_label': label,
                'valence': v,
                'arousal': a,
                'dominance': d
            })

    # Overall centroids
    for label in sorted(overall_centroids.keys()):
        v, a, d = overall_centroids[label]
        rows.append({
            'dataset': 'OVERALL',
            'emotion_label': label,
            'valence': v,
            'arousal': a,
            'dominance': d
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✅ Centroids saved to: {output_path}")


def print_yaml_format(centroids, name="expected_vad"):
    """
    Print centroids in YAML config format

    Args:
        centroids: dict of {label: [v, a, d]}
        name: Name for the config section
    """
    print(f"\n{name}:")
    for label in sorted(centroids.keys()):
        v, a, d = centroids[label]
        # Map label to emotion name (assuming standard 4-class setup)
        emotion_names = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'anger'}
        emotion = emotion_names.get(label, f'class_{label}')
        print(f"  {label}: [{v:.4f}, {a:.4f}, {a:.4f}]  # {emotion}")


if __name__ == "__main__":
    # Datasets with VAD annotations
    datasets = ["IEMO", "MSPI", "MSPP"]

    print("=" * 60)
    print("Calculating Emotion Centroids from VAD Data")
    print("=" * 60)

    # Calculate centroids for each dataset
    all_centroids = {}
    for dataset_name in datasets:
        try:
            centroids = calculate_dataset_centroids(dataset_name)
            all_centroids[dataset_name] = centroids
        except Exception as e:
            print(f"  ⚠️  Error processing {dataset_name}: {e}")

    # Calculate overall centroids
    print("\n" + "=" * 60)
    print("Overall Centroids (averaged across datasets)")
    print("=" * 60)
    overall_centroids = calculate_overall_centroids(all_centroids)
    for label in sorted(overall_centroids.keys()):
        v, a, d = overall_centroids[label]
        emotion_names = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'anger'}
        emotion = emotion_names.get(label, f'class_{label}')
        print(f"  Emotion {label} ({emotion}): VAD=[{v:.4f}, {a:.4f}, {d:.4f}]")

    # Save to CSV
    output_path = Path("emotion_centroids.csv")
    save_centroids_to_csv(all_centroids, overall_centroids, output_path)

    # Print in YAML format for easy copy-paste to config files
    print("\n" + "=" * 60)
    print("YAML Format (normalized values, copy to config):")
    print("=" * 60)
    print_yaml_format(overall_centroids, "expected_vad")

    print("\n✅ Done!")
