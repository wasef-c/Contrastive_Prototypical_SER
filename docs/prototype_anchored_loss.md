# Prototype-Anchored Loss with VAD-Guided Initialization

## Overview

The Prototype-Anchored Loss is a novel contrastive learning approach for cross-corpus emotion recognition. It replaces traditional pairwise contrastive learning (which failed to beat baselines in our experiments) with a **prototype-centered** approach that leverages prototypicality — a continuous measure of how "canonical" an emotion sample is, derived from its distance to expected VAD (Valence-Arousal-Dominance) centroids.

## Core Concept: Prototypicality

Every emotion sample has a **prototypicality score** (called "difficulty") computed as the Euclidean distance between its actual VAD values and the expected VAD centroid for its emotion class:

```
difficulty_i = ||VAD_actual - VAD_expected[class_i]||
```

| Emotion | Expected VAD (V, A, D) |
|---------|----------------------|
| Neutral | 0.49, 0.55, 0.55 |
| Happy   | 0.71, 0.55, 0.55 |
| Sad     | 0.31, 0.49, 0.49 |
| Anger   | 0.24, 0.61, 0.61 |

- **Low difficulty** = prototypical (a clearly happy utterance with high valence)
- **High difficulty** = atypical (an ambiguous or boundary sample)

These scores are computed from **fixed VAD centroids** and never change during training.

## Architecture

```
                                    ┌─────────────────────┐
Audio (768d) ──┐                    │   Classification    │
               ├─► Fusion ─► Embedding (1024d) ─► Head ─► logits
Text  (768d) ──┘       │                           │
                        │                          │
                        │    ┌─────────────────────┘
                        │    │
                        ▼    ▼
                   Projection Head (1024 → 512 → 128d)
                        │
                        ▼
                   Normalized Embeddings (128d, unit sphere)
                        │
                        ▼
               ┌────────────────────┐
               │ Prototype-Anchored │
               │       Loss         │
               └────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      L_anchor    L_separation    Learnable
    (per-sample)  (inter-class)   Prototypes
                                  (128d x 4)
```

The classification head and the prototype-anchored loss operate on **separate representations** — the classifier uses the raw 1024d embeddings while the contrastive loss uses projected 128d embeddings. This prevents conflicting gradient signals.

## Two Spaces, Two Roles

### 1. Fixed VAD Space (3D) — Scoring

The original VAD centroids live in 3D psychological space and are **never modified**. They serve one purpose: computing the prototypicality score for each sample via Euclidean distance. This score determines how "canonical" an emotion expression is.

### 2. Learnable Embedding Space (128D) — Optimization

Four learnable class prototypes live in the 128D projected embedding space. These are `nn.Parameter` tensors updated by gradient descent. They serve as **class anchors** that samples are pulled toward (or pushed away from).

### Connection Between the Two

The VAD centroids **initialize** the learnable prototypes — the first 3 dimensions of each 128D prototype are set from scaled VAD values, giving them a meaningful starting configuration (e.g., happy and sad start far apart). After initialization, the prototypes move freely via gradient descent. The VAD scores then **weight** how strongly each sample interacts with these prototypes.

## Loss Components

### L_anchor — Pull Samples Toward Their Class Prototype

```
L_anchor = mean_i [ w_i * max(0, ||embed_i - proto_{y_i}||^2 - margin_i) ]
```

Each sample is pulled toward its class prototype, but with two prototypicality-dependent modulations:

**Weight** (how hard to pull):
```
w_i = exp(-alpha * difficulty_i)
```
- Prototypical sample (difficulty ~ 0) → weight ~ 1.0 → strong pull
- Atypical sample (difficulty ~ 1.5) → weight ~ 0.05 → weak pull

**Margin** (how close it must get):
```
margin_i = margin_base + beta * difficulty_i
```
- Prototypical sample → tight margin (must be very close to prototype)
- Atypical sample → relaxed margin (allowed to be farther away)

This creates a structure where **canonical samples define tight class cores** while ambiguous samples aren't forced into positions that would distort the learned representation.

### L_separation — Push Class Prototypes Apart

```
L_separation = sum_{c != c'} max(0, delta - ||proto_c - proto_{c'}||^2)
```

A hinge loss that penalizes prototypes that are closer than `delta` (separation_margin). This ensures the four emotion classes maintain well-separated centers in embedding space, preventing collapse.

### Combined Loss

```
L_total = L_classification + lambda * (L_anchor + w_sep * L_separation)
```

Where `lambda` is `contrastive_weight` and `w_sep` is `separation_weight`.

## Why This Works for Cross-Corpus Transfer

Standard contrastive learning (SupCon) pulls all same-class samples together equally. This doesn't help with domain shift because corpus-specific artifacts get encoded alongside emotion information.

The prototype-anchored approach addresses this by:

1. **Canonical samples anchor the space**: Prototypical samples (consistent across corpora — a clearly happy utterance sounds happy regardless of recording conditions) are weighted heavily and pulled tightly to prototypes. These form domain-invariant class cores.

2. **Atypical samples don't distort**: Corpus-specific or ambiguous samples get low weight and relaxed margins. They can exist at the periphery without pulling the class center toward corpus-specific artifacts.

3. **VAD grounding provides universal structure**: The VAD-based initialization and scoring are grounded in psychological theory, not corpus statistics. This gives the model a starting structure that generalizes across datasets.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prototypical_alpha` | 2.0 | Weight decay rate. Higher = more focus on prototypical samples |
| `prototypical_beta` | 0.5 | Margin scaling. Higher = more slack for atypical samples |
| `margin_base` | 0.1 | Minimum margin for the most prototypical samples |
| `separation_margin` | 2.0 | Minimum squared distance between class prototypes |
| `separation_weight` | 0.5 | Weight for L_separation relative to L_anchor |
| `contrastive_weight` | 0.5 | Weight for combined prototype loss relative to classification loss |
| `projection_dim` | 128 | Dimensionality of the contrastive embedding space |

## Sweep Configuration

The sweep in `configs/prototype_anchored_sweep.yaml` tests:

- **Alpha sweep** (1.0, 2.0, 5.0) — How aggressively to focus on prototypical samples
- **Contrastive weight sweep** (0.1, 0.3, 1.0, 2.0) — Balance between classification and prototype anchoring
- **Margin base sweep** (0.01, 0.05, 0.3) — How tight the prototypical cluster must be
- **Beta sweep** (0.1, 1.0, 2.0) — How much slack atypical samples get
- **Separation margin sweep** (1.0, 3.0) — How far apart prototypes are pushed
- **Projection dim sweep** (64, 256) — Size of the contrastive embedding space

Run with:
```bash
python runner.py --config configs/prototype_anchored_sweep.yaml --all
```

## File Locations

- Loss implementation: `models/contrastive_loss.py` (`PrototypeAnchoredLoss`, `PrototypeDivergenceLoss`)
- Prototypicality scoring: `utils/prototypicality.py`
- Config defaults: `utils/config.py`
- Sweep config: `configs/prototype_anchored_sweep.yaml`
