# Emotion2Vec Contrastive Learning

Domain adaptation for cross-corpus emotion recognition using prototypicality-guided contrastive learning.

## Overview

This repository implements **prototypicality-weighted supervised contrastive learning** for improving cross-corpus generalization in emotion recognition. The key innovation is using VAD (Valence-Arousal-Dominance) distance as a measure of sample prototypicality to weight contrastive learning.

**Core Idea**: Prototypical samples (close to expected VAD values) are pulled harder toward their class centers in embedding space, while atypical samples receive less weight. This creates more robust, domain-invariant representations.

## Project Structure

```
Emotion2Vec_Contrastive/
├── models/
│   ├── classifier.py          # Main emotion classifier with embedding extraction
│   ├── encoder.py             # Frozen BERT text encoder
│   ├── fusion.py              # Multimodal fusion modules (cross-attention, concat, etc.)
│   └── contrastive_loss.py    # Contrastive loss variants
├── data/
│   ├── dataset.py             # Dataset loader with VAD normalization
│   └── collate.py             # Batch collation for VAD values
├── utils/
│   ├── config.py              # Configuration system
│   ├── metrics.py             # Evaluation metrics (UAR, CCC, etc.)
│   └── prototypicality.py     # VAD-based difficulty calculation
├── configs/                   # YAML configuration files
├── train.py                   # Main training script
├── calculate_centroids.py     # Calculate empirical VAD centroids
└── README.md                  # This file
```

## Key Components

### 1. Emotion Classifier (`models/classifier.py`)

**Purpose**: Main model for emotion recognition with built-in embedding extraction.

**Key Features**:
- Supports three modalities: `audio`, `text`, or `both`
- Extracts embeddings at `[batch, 1024]` layer before final classification
- Returns `(logits, embeddings)` when `return_embeddings=True`

**Architecture Paths**:
- **Audio-only**: Audio features [768] → Linear → [1024] → Classifier → [4/3]
- **Text-only**: BERT [768] → Linear → [1024] → Classifier → [4/3]
- **Multimodal**: Audio [768] + Text [768] → Fusion → [512] → Linear → [1024] → Classifier → [4/3]

**Embedding Extraction**:
```python
logits, embeddings = model(audio_features=x, return_embeddings=True)
# embeddings: [batch_size, 1024] - used for contrastive learning
```

### 2. Contrastive Loss (`models/contrastive_loss.py`)

**Base Implementation**: `SupervisedContrastiveLoss`
- Standard supervised contrastive learning (SupCon)
- Pulls same-class samples together, pushes different classes apart
- Temperature τ = 0.07 by default

**Prototypical Variants**:

1. **V1 - Sample Weighting** (`PrototypicalContrastiveLoss_V1`)
   - Formula: `weight_i = exp(-α * difficulty_i)`
   - Prototypical samples (low difficulty) → high weight
   - Atypical samples (high difficulty) → low weight
   - Hyperparameter: `alpha` (default: 1.0)

2. **V2 - Pair Weighting** (`PrototypicalContrastiveLoss_V2`)
   - Formula: `weight_ij = exp(-β * (difficulty_i + difficulty_j))`
   - Weights each positive pair by combined difficulty
   - Both prototypical → strongest pull
   - Hyperparameter: `beta` (default: 0.5)

3. **V3 - Threshold Separation** (`PrototypicalContrastiveLoss_V3`)
   - Formula: `weight_i = 1.0 if difficulty_i < threshold else 0.1`
   - Binary separation: prototypical vs atypical
   - Hyperparameter: `threshold` (default: 1.0)

### 3. Prototypicality Calculation (`utils/prototypicality.py`)

**Definition**: Euclidean distance in 3D VAD space between actual and expected VAD values.

```python
difficulty = sqrt((v_actual - v_expected)² + (a_actual - a_expected)² + (d_actual - d_expected)²)
```

**Expected VAD Values** (normalized 0-1, empirically calculated):
- Neutral: [0.4896, 0.5458, 0.5458]
- Happy: [0.7127, 0.5518, 0.5518]
- Sad: [0.3060, 0.4895, 0.4895]
- Anger: [0.2397, 0.6102, 0.6102]

**Usage**:
- Low difficulty → prototypical sample (close to class center)
- High difficulty → atypical sample (far from class center)

### 4. Dataset Loader (`data/dataset.py`)

**Supported Datasets**: IEMO, MSPI, MSPP (all have VAD annotations)
- Loaded from HuggingFace: `cairocode/<DATASET>_Audio_Text_Merged`
- Pre-extracted Emotion2Vec features (768-dim)
- Text transcripts included

**VAD Normalization**:
- **MSPP**: 1-7 scale → `(val - 1) / 6` → [0, 1]
- **IEMO/MSPI**: 1-5 scale → `(val - 1) / 4` → [0, 1]

**Performance Optimization**:
- Defers tensor conversion to `__getitem__()` for 10x faster loading
- Skips samples with NaN VAD for regression tasks

### 5. Training Loop (`train.py`)

**Combined Loss**:
```python
loss = loss_primary + λ_con * loss_contrastive
```

Where:
- `loss_primary`: CrossEntropyLoss (classification) or MSELoss (regression)
- `loss_contrastive`: Prototypical contrastive loss
- `λ_con`: Contrastive weight (default: 0.5)

**Training Flow**:
1. Forward pass with `return_embeddings=True`
2. L2-normalize embeddings
3. Calculate primary loss (classification/regression)
4. Calculate difficulty scores from VAD distance
5. Calculate contrastive loss weighted by difficulty
6. Backprop combined loss

**Evaluation**:
- Cross-corpus: Train on one dataset, test on others
- Metrics: UAR (classification), CCC (regression)

## Configuration

Configs are stored in `configs/` as YAML files. Key parameters:

```yaml
# Task
task_type: "classification"  # or "regression"
num_classes: 4  # neutral, happy, sad, anger
modality: "both"  # "audio", "text", or "both"

# Contrastive learning
use_contrastive: true
contrastive_loss_type: "prototypical_v1"  # v1, v2, v3, or supervised
contrastive_weight: 0.5
contrastive_temperature: 0.07

# Prototypicality
prototypical_alpha: 1.0  # For V1
prototypical_beta: 0.5   # For V2
prototypical_threshold: 1.0  # For V3

# Expected VAD centroids (empirically calculated)
expected_vad:
  0: [0.4896, 0.5458, 0.5458]  # neutral
  1: [0.7127, 0.5518, 0.5518]  # happy
  2: [0.3060, 0.4895, 0.4895]  # sad
  3: [0.2397, 0.6102, 0.6102]  # anger
```

## Usage

### 1. Calculate Empirical Centroids (Optional)

```bash
python calculate_centroids.py
```

This calculates mean VAD values for each emotion class from the actual data and saves to `emotion_centroids.csv`.

### 2. Train Baseline (No Contrastive Learning)

```bash
python train.py --config configs/baseline_classification.yaml
```

### 3. Train with Prototypical Contrastive Learning

```bash
# Sample weighting (V1)
python train.py --config configs/prototypical_v1.yaml

# Pair weighting (V2)
python train.py --config configs/prototypical_v2.yaml

# Threshold separation (V3)
python train.py --config configs/prototypical_v3.yaml
```

### 4. VAD Regression

```bash
python train.py --config configs/baseline_regression.yaml
```

## Implementation Details

### How Prototypicality Guides Contrastive Learning

**Step 1**: Calculate difficulty for each sample
```python
difficulties = batch_calculate_difficulty(batch, expected_vad)
# difficulties: [batch_size] - VAD distance for each sample
```

**Step 2**: Weight contrastive loss by prototypicality
```python
# V1: Sample weighting
weights = exp(-alpha * difficulties)
loss = base_contrastive_loss(embeddings, labels, weights)

# V2: Pair weighting
pair_weights = exp(-beta * (diff_i + diff_j))
loss = weighted_pair_contrastive_loss(embeddings, labels, pair_weights)

# V3: Threshold
weights = 1.0 if difficulty < threshold else 0.1
loss = base_contrastive_loss(embeddings, labels, weights)
```

**Result**: Prototypical samples are pulled harder toward class centers, creating more robust embeddings.

### Cross-Corpus Evaluation

The system trains on one dataset and evaluates on others to test domain generalization:

- **Training**: MSPP (train split)
- **Validation**: MSPP (10% held-out)
- **Testing**: IEMO + MSPI (full datasets)

Expected improvements:
- Baseline cross-corpus UAR: ~40-50%
- With prototypical contrastive: +3-10% UAR improvement

### Multimodal Fusion

Four fusion strategies available (`models/fusion.py`):

1. **CrossAttentionFusion**: Audio and text attend to each other
2. **SimpleConcatFusion**: Concatenate and project
3. **GatedFusion**: Learned gating mechanism
4. **AdaptiveFusion**: Handles missing modalities

Default: `cross_attention` for best performance.

## Key Design Decisions

### Why VAD Distance for Prototypicality?

- **Theory**: Emotions map to continuous VAD space
- **Observation**: Samples near expected VAD values are more "typical" of their class
- **Application**: Use this to weight learning - focus on prototypical patterns, downweight noise

### Why Deferred Tensor Conversion?

Original implementation called `torch.tensor()` on 10k+ samples during dataset `__init__()`, causing slow startup. Solution:

```python
# Before: Slow
self.data.append({"features": torch.tensor(features)})  # In __init__

# After: Fast
self.data.append({"features": features})  # Store raw list in __init__
# Convert in __getitem__:
features = torch.tensor(item["features"]) if not isinstance(...) else item["features"]
```

Result: 10x faster dataset loading.

### Why Normalize VAD to [0, 1]?

Datasets use different scales:
- MSPP: 1-7 scale
- IEMO/MSPI: 1-5 scale

Normalization ensures:
- Consistent prototypicality calculation
- Fair comparison across datasets
- Expected VAD values are portable

## Metrics

### Classification
- **UAR (Unweighted Average Recall)**: Mean of per-class recalls
- **Accuracy**: Overall accuracy
- **F1 (weighted)**: Weighted F1 score

### Regression (VAD)
- **CCC (Concordance Correlation Coefficient)**: Agreement metric
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Pearson Correlation**

All metrics reported per-dimension (V, A, D) and overall average.

## WandB Logging

Training metrics logged to Weights & Biases:

```python
log_dict = {
    'epoch': epoch,
    'train/loss': total_loss,
    'train/primary_loss': classification_loss,
    'train/contrastive_loss': contrastive_loss,
    'val/accuracy': val_acc,
    'val/uar': val_uar,
    'test/IEMO_uar': test_uar,
    'test/MSPI_uar': test_uar,
}
```

## Extending the System

### Adding a New Contrastive Loss Variant

1. Create new class in `models/contrastive_loss.py`:
```python
class PrototypicalContrastiveLoss_V4(nn.Module):
    def forward(self, embeddings, labels, difficulties):
        # Your implementation
        pass
```

2. Add to factory function:
```python
def create_contrastive_loss(loss_type, **kwargs):
    if loss_type == "prototypical_v4":
        return PrototypicalContrastiveLoss_V4(**kwargs)
```

3. Create config file in `configs/prototypical_v4.yaml`

### Adding a New Dataset

1. Add to `DATASET_MAP` in `data/dataset.py`:
```python
DATASET_MAP = {
    "IEMO": "cairocode/IEMO_Audio_Text_Merged",
    "MSPI": "cairocode/MSPI_Audio_Text_Merged",
    "MSPP": "cairocode/MSPP_Audio_Text_Merged",
    "NEWDATASET": "cairocode/NEWDATASET_Audio_Text_Merged",  # Add here
}
```

2. Add VAD normalization if needed:
```python
if dataset_name == "NEWDATASET":
    valence = (valence - min_val) / (max_val - min_val)
```

3. Calculate new centroids:
```python
python calculate_centroids.py  # Will include new dataset
```

## Troubleshooting

### Slow Dataset Loading
- Ensure deferred tensor conversion is working
- Check if VAD normalization is outside the loop

### Poor Cross-Corpus Performance
- Verify VAD normalization is correct
- Try different contrastive weights (0.1-1.0)
- Adjust alpha/beta/threshold hyperparameters

### NaN Loss
- Check for samples with missing VAD values
- Ensure embeddings are L2-normalized
- Verify batch has at least 2 samples per class

### CUDA Errors
- Driver/library mismatch: Reboot system
- Out of memory: Reduce batch_size or text_max_length

## Future Directions

**Alternative Approaches** (not yet implemented):
- Domain adversarial learning with gradient reversal
- Self-supervised pre-training (MoCo/SimCLR)
- Multi-task learning (classification + regression + contrastive)
- Learned class prototypes (vs. fixed expected VAD)
- Domain-specific prototypes

**Improvements**:
- Automatic hyperparameter tuning
- Data augmentation strategies
- Dynamic contrastive weight scheduling
- Embedding visualization (t-SNE/UMAP)

## Citation

If you use this code, please cite the original Emotion2Vec work and the supervised contrastive learning paper:

```
@inproceedings{supcon2020,
  title={Supervised Contrastive Learning},
  author={Khosla, Prannay and Teterwak, Piotr and others},
  booktitle={NeurIPS},
  year={2020}
}
```
