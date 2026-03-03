# Prototypicality Research Roadmap

## Current State (Feb 2026)
## The novel idea:

Measure prortptypicality as the deviation or distance from expected VAD values for each class
e.g. expect happy to be high valnece, high arousal middle domination, a sample with low arousal is not prortotypical

I aim to use this idea to reduce the effects of subjectivity in SER, non prototypical samples may be mislabeled or
highly open to interpretation

### What we've tried
- Standard SupCon (pairwise contrastive) — did not beat baseline
- Prototypical weighting variants (V1/V2/V3) — marginal gains
- Prototype-Anchored Loss (novel, VAD-initialized learnable prototypes) — small gains on single-corpus
- Multi-corpus training (IEMO+MSPI+MSPP) — helps, especially with MSPP included
- Prototype-Anchored MultiDS with cross-domain alignment — best CMUMOSEI result (0.553 vs 0.510 baseline), PA stable at high weights where SupCon collapses
- Curriculum learning with prototypicality — tried, unsatisfying results
- Domain adversarial with prototypicality weighting — proto adds ~1.6% UAR on CMUMOSEI, SAMSEMO flat. Same marginal pattern as other auxiliary losses.
- Prototypicality-weighted primary CE loss + label smoothing — implemented, pending results. Weights the main loss by `exp(-α * difficulty)` and scales label smoothing by difficulty for atypical samples.

### Key findings
- PA regularizes effectively (lowest train-val gap, stable at high contrastive weights)
- MSPP is critical for generalization — any combo without it collapses
- IEMO+MSPI without MSPP is catastrophically bad (-10pts on SAMSEMO)
- Gains are real but small (+1.7% CMUMOSEI, ~0% SAMSEMO over well-tuned baseline)
- The bottleneck may be feature quality / model capacity, not loss function
- Curriculum learning only improved perfomance when model hyperpramateres weren't fully optimized, and doesn't fundamentally solve the problem of subjectivity
- **All auxiliary losses (contrastive, adversarial) show marginal gains.** The primary CE loss is where 90%+ of gradient comes from — that's where prototypicality should act.
- VAD regression CCC too low cross-corpus (~0.22 overall) for reliable pseudo-VAD generation

### Core limitation
Prototypicality requires VAD annotations. CMUMOSEI and SAMSEMO lack VAD, so they can't benefit from prototypicality weighting during training. This limits the approach to 3 of 5 datasets.

---

## Priority 1: Pseudo-VAD Generation

**Impact: High | Effort: Medium | Novelty: High**
**Status: DEPRIORITIZED — cross-corpus VAD regression CCC too low (~0.22)**

Train a VAD regression model on IEMO+MSPI+MSPP (all have VAD), then predict pseudo-VAD values for CMUMOSEI and SAMSEMO. This unlocks prototypicality scoring for ALL datasets.

### Why this was deprioritized
- Cross-corpus VAD regression achieves only ~0.22 CCC overall
- Arousal CCC is negative for MSPI, meaning predictions are worse than random
- Pseudo-VAD quality insufficient for reliable prototypicality scoring
- May revisit if a better VAD model is found or if prototypicality only needs coarse ranking

---

## Priority 2: Domain Adversarial Training with Prototypicality

**Impact: High | Effort: Medium | Novelty: Medium-High**
**Status: DONE — marginal gains (~1.6% UAR on CMUMOSEI with proto weighting, SAMSEMO flat)**

Implemented GRL + domain discriminator with prototypicality-weighted adversarial loss. Prototypical samples forced domain-invariant, atypical allowed to retain domain info.

### Result
- Proto weighting adds ~1.6% UAR on CMUMOSEI over standard adversarial
- SAMSEMO essentially unchanged across all adversarial variants
- Same pattern as contrastive: auxiliary losses provide marginal gains
- Requires multi-corpus training (single-corpus has no domains to discriminate)

---

## Priority 2.5: Prototypicality-Weighted Primary Loss & Label Smoothing

**Impact: High | Effort: Low | Novelty: Medium**
**Status: IMPLEMENTED — pending results**

Directly modify the primary classification CE loss using prototypicality, rather than adding auxiliary losses.

### Two mechanisms
1. **Weighted CE**: `loss = (CE_per_sample * exp(-α * difficulty)).mean()` — prototypical samples dominate gradient
2. **Label smoothing**: `smoothing_i = β * difficulty_i` — atypical samples get softer targets

### Why this matters
- Acts on the primary loss (90%+ of gradient), not an auxiliary signal
- Works with single-corpus training (no multi-corpus needed)
- Directly addresses the subjectivity thesis: ambiguous samples treated as ambiguous
- Stacks with adversarial/contrastive if needed

### Sweep
`configs/prototypicality_primary_sweep.yaml` — 21 experiments across MSPP, IEMO, ALL3

---

## Priority 3: Auxiliary Prototypicality Prediction (NEXT)

**Impact: Medium-High | Effort: Low | Novelty: High**

Add a small MLP head that predicts prototypicality score from the shared embedding. Multi-task learning forces the backbone to encode WHERE a sample sits relative to its class prototype.

### Why this is novel
- Uses prototypicality as a **learning signal**, not just a loss weight
- The model learns an internal representation of annotation confidence/subjectivity
- This structure directly helps cross-corpus transfer — VAD-distance relationships are corpus-invariant even when raw VAD values aren't
- Distinct from prior work: no existing SER work uses VAD-derived prototypicality as an auxiliary target

### Implementation
1. Add MLP head: shared embedding (1024) → hidden (256) → prototypicality score (1)
2. Target: `difficulty = euclidean_dist(sample_VAD, class_centroid_VAD)`
3. Loss: `L_total = L_CE + λ_proto * MSE(predicted_proto, actual_proto)`
4. Only applied to samples with real VAD (datasets with VAD annotations)
5. At test time, the prediction head is discarded — its value is in regularizing the shared backbone

### Sweep plan
- λ_proto sweep: 0.1, 0.5, 1.0, 2.0
- Single-corpus (MSPP, IEMO) and multi-corpus (ALL3)
- Combined with prototypicality-weighted CE from Priority 2.5

---

## Priority 3.5: BERT Unfreezing (BIGGEST GAIN SO FAR)

**Impact: HIGH | Effort: Low | Novelty: Low**
**Status: DONE — single biggest improvement in the project**

### Results
| Setup | CMUMOSEI | SAMSEMO |
|---|---|---|
| ALL3 frozen BERT | 0.511 | 0.581 |
| ALL3 unfreeze4 | 0.572 (+6.1%) | 0.598 (+1.7%) |
| ALL3 unfreeze4 + wCE | **0.573** | **0.603** |
| MSPP frozen | 0.520 | 0.580 |
| MSPP unfreeze4 | 0.544 (+2.4%) | 0.591 (+1.1%) |

### Key findings
- 4 layers is the sweet spot (6 overfits, 2 underfits)
- Differential LR critical: BERT at 5e-7, rest at 5e-6
- Train UAR dropped (0.773→0.711) = less overfitting
- wCE adds small but consistent SAMSEMO boost on top

### Next: Text model ablation
`configs/text_model_ablation.yaml` — RoBERTa, DeBERTa-v3 vs BERT with unfreeze levels + prototypicality methods

---

## Priority 4: Partial BERT Unfreezing

**Impact: Medium | Effort: Low | Novelty: Low**

Currently BERT is fully frozen. Unfreezing the last 2-4 transformer layers with a lower learning rate allows the text encoder to learn emotion-specific representations.

### Implementation
1. Add config param: `unfreeze_bert_layers: 2` (number of layers from the top)
2. Use differential learning rate: 1e-6 for BERT layers, 5e-6 for rest
3. Test on best-performing config from current sweeps

### Risk
- Overfitting on small datasets (mitigate with lower LR + weight decay)
- Slower training (more parameters to update)

---

## Priority 5: Cross-Modal Contrastive with Prototypicality

**Impact: Medium | Effort: Medium | Novelty: High**

Pull audio and text embeddings of the same sample together in a shared space, weighted by prototypicality. A prototypical sample's audio and text should strongly agree; an atypical sample may have conflicting signals.

### Implementation
1. Extract separate audio and text embeddings before fusion
2. Contrastive loss pulling same-sample audio/text together
3. Weight by prototypicality: `w = exp(-alpha * difficulty)`
4. Prototypical samples → strong cross-modal agreement enforced
5. Atypical samples → less forced alignment (modalities may legitimately disagree)

---

## Priority 6: Prototypicality-Aware Attention

**Impact: Low-Medium | Effort: Medium | Novelty: High**

Feed prototypicality information into the cross-attention fusion module as a conditioning signal. Let the model know whether to trust both modalities equally (prototypical) or be cautious (atypical).

### Implementation
1. Compute prototypicality score per sample
2. Use as attention bias or gating signal in cross-attention fusion
3. Prototypical → equal attention to both modalities
4. Atypical → model learns which modality to trust more

---

## Deprioritized

### Wav2Vec2 / HuBERT instead of Emotion2Vec
- Emotion2Vec is already specialized for emotion — general speech models likely worse
- Only worth trying as an ablation for a paper comparison

### Curriculum learning with prototypicality
- Already tried, results unsatisfying
- May revisit if combined with other approaches (e.g., pseudo-VAD + curriculum)

---

## Experiment Tracking

| Priority | Approach | Status | Key Result |
|----------|----------|--------|------------|
| — | SupCon contrastive | Done | Did not beat baseline |
| — | Prototypical weighting V1/V2/V3 | Done | Marginal gains |
| — | Prototype-Anchored Loss | Done | Small gains single-corpus, +4.3% CMUMOSEI multi-corpus |
| — | Curriculum learning | Done | Only helps with untuned hyperparams |
| 1 | Pseudo-VAD generation | Deprioritized | Cross-corpus CCC ~0.22, too low |
| 2 | Domain adversarial + prototypicality | Done | +1.6% CMUMOSEI, SAMSEMO flat |
| 2.5 | Prototypical-weighted primary CE + label smoothing | Done | Marginal gains (~0.5% SAMSEMO) |
| 3 | Auxiliary prototypicality prediction | Done | Marginal gains |
| 3.5 | **BERT unfreezing** | **Done** | **+6.1% CMUMOSEI, +1.7% SAMSEMO** |
| — | Text model ablation (BERT/RoBERTa/DeBERTa) | **Running** | — |
| 4 | Partial BERT unfreeze | Superseded by 3.5 | — |
| 5 | Cross-modal contrastive | Not started | — |
| 6 | Prototypicality-aware attention | Not started | — |
