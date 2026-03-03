# Wav2Vec2 Audio Encoder + Two-Stage Prototypicality Training

## Context

BERT unfreezing gave the biggest improvement so far (+6.1% CMUMOSEI). Emotion2Vec weights are closed-source so we can't unfreeze it. Switching to Wav2Vec2 (`facebook/wav2vec2-base-960h`) as the audio encoder lets us unfreeze audio layers too — giving trainable representations on both modalities. Combined with two-stage prototypicality training (clean data first, then full data).

## Two major changes

### A. Wav2Vec2 Audio Encoder (with unfreezing support)
### B. Two-Stage Prototypicality Training

---

## A. Wav2Vec2 Audio Encoder

### 1. New module: `models/audio_encoder.py`

Wav2Vec2 encoder mirroring the BERT encoder pattern (see `models/encoder.py`):

```python
class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", unfreeze_layers=0):
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        # Freeze all, then unfreeze top N encoder layers (same pattern as BERT)
        # wav2vec2.encoder.layers is the transformer stack
        # Mean-pool over time: [B, T, 768] -> [B, 768]

    def forward(self, waveforms, attention_mask=None):
        # Process raw waveforms through wav2vec2
        # Mean pool last_hidden_state over time dimension
        return pooled_features  # [B, 768]

    def process_waveforms(self, waveforms, sampling_rate=16000):
        # Use self.processor to prepare inputs (normalization)
        return input_values

    def get_audio_params(self):
        return [p for p in self.wav2vec2.parameters() if p.requires_grad]
```

### 2. Modify `data/dataset.py` — load raw audio

All 5 HF datasets have both `audio` column (raw waveforms) and `emotion2vec_features`. Same dataset paths work for both modes — just read a different column.

For CMUMOSEI and SAMSEMO, use different HF dataset paths that contain audio:

```python
DATASET_MAP = {
    "IEMO": "cairocode/IEMO_Audio_Text_Merged",
    "MSPI": "cairocode/MSPI_Audio_Text_Merged",
    "MSPP": "cairocode/MSPP_Audio_Text_Merged",
    "CMUMOSEI": "cairocode/CMUMOSEI_Emotion2Vec_PrecomputedEncodings",
    "SAMSEMO": "cairocode/SAMSEMO_Emotion2Vec_PrecomputedEncodings",
}

# When audio_encoder_type == "wav2vec2", use these for CMUMOSEI/SAMSEMO
AUDIO_DATASET_MAP = {
    "CMUMOSEI": "cairocode/cmu_mosei_wav_2",
    "SAMSEMO": "cairocode/samsemo-audio",
}
```

When `audio_encoder_type == "wav2vec2"`:
- Load `item["audio"]["array"]` and `item["audio"]["sampling_rate"]`
- Store waveform as numpy array in sample dict (convert to tensor in `__getitem__`)
- **Filter out labels > 3** (CMUMOSEI/SAMSEMO have extra emotion classes)
- Skip `emotion2vec_features` column

When `audio_encoder_type == "preextracted"` (default):
- Keep current behavior exactly

### 3. Modify `data/collate.py` — variable-length audio padding

Add waveform collation when `waveform` key is present in samples:
- Pad all waveforms to max length in batch with zeros
- Create attention mask tensor (1=real audio, 0=padding)
- Return `batch['waveforms']` [B, max_T] and `batch['audio_attention_mask']` [B, max_T]

### 4. Modify `models/classifier.py`

Add `audio_encoder_type` and `audio_model_name` params:
- When `"wav2vec2"`: create `Wav2Vec2Encoder`, forward passes waveforms through it first
- Forward signature gets `audio_waveforms` and `audio_attention_mask` params
- When `"preextracted"`: keep current behavior (audio_features -> embedding_layer)
- `create_model()` passes new config params

### 5. Modify `train.py`

- Detect audio_encoder_type and pass correct inputs (waveforms vs features)
- Add Wav2Vec2 encoder params to optimizer with differential LR (like BERT)
- Access waveforms from batch dict

### 6. Config additions (`utils/config.py`)

```python
self.audio_encoder_type = "preextracted"  # "preextracted" or "wav2vec2"
self.audio_model_name = "facebook/wav2vec2-base-960h"
self.unfreeze_audio_layers = 0
self.audio_learning_rate = 5e-7  # differential LR for wav2vec2
```

---

## B. Two-Stage Prototypicality Training

### Concept
- **Stage 1**: Train on prototypical samples only (difficulty below percentile threshold) — learn clean decision boundaries
- **Stage 2**: Fine-tune on ALL data with lower LR — adjust for edge cases

### Implementation in `train.py`

Modify `train()` function. When `config.use_two_stage_training`:

1. Before training loop, compute per-sample difficulty using `calculate_difficulty()` from `utils/prototypicality.py`
2. Compute threshold as `np.percentile(difficulties, stage1_percentile)`
3. Split train indices into prototypical subset (difficulty <= threshold)
4. **Stage 1**: Create dataloader from proto subset, train for `stage1_epochs`
5. **Stage 2**: Switch to full dataset dataloader, reduce LR by `stage2_lr_factor`, train for remaining epochs
6. Early stopping resets between stages (stage 2 gets fresh patience)
7. Best model can come from either stage

### Config additions
```python
self.use_two_stage_training = False
self.stage1_percentile = 50     # bottom 50% most prototypical for stage 1
self.stage1_epochs = 30         # epochs for stage 1
self.stage2_lr_factor = 0.1     # reduce LR by 10x for stage 2
```

---

## Files to modify

1. **`models/audio_encoder.py`** — NEW: Wav2Vec2Encoder with unfreezing + processing
2. **`models/classifier.py`** — Support wav2vec2 audio path, new forward params
3. **`models/__init__.py`** — Export Wav2Vec2Encoder
4. **`data/dataset.py`** — Raw audio loading, label>3 filtering, AUDIO_DATASET_MAP
5. **`data/collate.py`** — Variable-length waveform padding + attention masks
6. **`train.py`** — Wav2Vec2 differential LR, two-stage training, waveform passing
7. **`utils/config.py`** — New config params + numeric fields
8. **`configs/wav2vec2_sweep.yaml`** — NEW: Sweep config

## Sweep config structure

- Baselines: preextracted (Emotion2Vec) frozen vs wav2vec2 frozen
- Wav2Vec2 unfreeze: 0, 2, 4 layers
- Wav2Vec2 + BERT unfreeze combined
- Two-stage training: percentile 50%, 70%
- Two-stage + wav2vec2 unfreeze + BERT unfreeze + wCE (full stack)
- Single-corpus (MSPP) variants

## Verification

```bash
python runner.py --config configs/wav2vec2_sweep.yaml --all
```
