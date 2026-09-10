# Cross-corpus SER: running results and investigations log

Training corpus MSP-Podcast, zero-shot evaluation on IEMOCAP, MSP-Improv,
CMU-MOSEI, SAMSEMO. Architecture: emotion2vec + BERT + cross-attention fusion.

Every mechanism claim here is stated against a **capacity-matched placebo**
(same head, same parameter count, same gradient path, targets permuted), not
against a no-auxiliary baseline, because a scrambled auxiliary head is itself
worth real UAR and comparing to no-aux conflates the two.

---

## 0. Measurement floors (read before interpreting anything)

Per-class sample counts in each evaluation corpus, and the resulting binomial
standard error on UAR (macro over four classes, at recall ~0.6):

| corpus | neutral | happy | sad | angry | UAR noise floor |
|---|---|---|---|---|---|
| IEMO | 1708 | 595 | 1084 | 1103 | **0.0078** |
| MSPI | 3501 | 2629 | 879 | 789 | **0.0068** |
| SAMSEMO | 2213 | 1936 | 337 | 557 | **0.0093** |
| CMUMOSEI | 948 | 871 | **46** | **73** | **0.0238** |

**Correction (2026-08-24): the binomial floor is the wrong statistic for
arm-vs-arm comparisons.** It answers "how precisely do we know this model's
absolute UAR", but every mechanism claim here is a PAIRED comparison on an
identical test set, where most sampling noise cancels. The right measure is
the seed-to-seed sd of the difference, measured across three independent
3-seed experiments:

| corpus | paired-difference sd | binomial floor | over-conservative by |
|---|---|---|---|
| IEMO | **0.0029** | 0.0078 | 2.7x |
| SAMSEMO | **0.0032** | 0.0093 | 2.9x |
| MSPI | 0.0037 | 0.0068 | 1.8x |
| CMUMOSEI | 0.0098 | 0.0238 | 2.4x |

**SAMSEMO is as precise as IEMOCAP for paired comparisons** and should not be
excluded; earlier statements treating it as unresolvable were wrong. In the
frozen 8-epoch recipe it gave +0.0091 +/- 0.0005, the tightest effect measured
anywhere in this project.

**CMU-MOSEI remains excluded**: 0.0098 is three times the other corpora and
exceeds most effects measured here. Its 46 sad and 73 angry utterances are the
cause.

---

## 1. The largest effects are data and encoder, not mechanism

| change | IEMO | MSPI | CMUMOSEI | SAMSEMO |
|---|---|---|---|---|
| merge MSPP Train+Development | +0.0156 | +0.0129 | **+0.1280** | +0.0584 |
| fine-tune emotion2vec (vs frozen) | +0.0146 | +0.0230 | -0.0208 | +0.0306 |
| best auxiliary mechanism (vs placebo) | +0.0065 | ~0 | ~0 | +0.0035 |

### 1.1 The MSP-Podcast split finding
The official Development split is **angry-heavy**: angry is 26.6% of
Development but 11.4% of Train, so 5,836 of the corpus's 12,567 angry
utterances sit in Development. Training on Train alone, or on a pool that
under-represents Development, strips the minority class hardest.

The previously used corpus (`MSPP_WAV_Filtered_ordered_v2`, 88,887 rows) had
53.4% neutral and only 10.0% angry. The official Train+Development 4-class
subset (80,941 rows) has 45.3% neutral and 15.5% angry. Switching recovered
**+0.128 UAR on CMU-MOSEI**, roughly 18x any mechanism effect.

### 1.2 Encoder fine-tuning
emotion2vec at LR 1e-5 on the full pool beats frozen on 3/4 corpora, all above
their noise floors. This is the single largest controllable gain available.

---

## 2. Bugs found (each invalidated prior conclusions)

| bug | effect | status |
|---|---|---|
| `load_subtype_meta` used lazy `NpzFile` indexing | 813,547 array re-inflations, 3.4 TB, ~2.3 h; stalled the machine | fixed, 0.31 s |
| BERT `gradient_checkpointing_enable()` defaulted to `use_reentrant=True` | **0 of 34 unfrozen tensors received gradient**; every "unfrozen BERT" run trained nothing, and three LRs spanning 100x gave bit-identical results | fixed (`use_reentrant=False`), verified 32/34 |
| runner auto-resumed from a stale `latest.pt` | one run silently mixed 12-epoch full-data and 8-epoch subsampled schedules | launcher now deletes checkpoints per arm |
| config generation dropped `unfreeze_*` flags | a full overnight "both encoders" sweep ran fully frozen | fixed |
| 30% training subsample to fit compute | model underfit (train UAR 0.6441 vs 0.7065 frozen, gap 0.016 vs 0.064); "unfrozen is worse than frozen" was an artifact | abandoned, full pool restored |

MSP-Improv arousal is inverted relative to convention (angry lowest, neutral
highest; corr(A,D) = -0.664 against +0.804 on MSPP). This was **already known
and corrected** in `data/dataset.py:292` (`orient_arousal`); rediscovered here
independently, which at least confirms the correction is warranted.

---

## 3. What the auxiliary target actually contains

A target that is a deterministic invertible function of VAD carries exactly
the information of VAD regression, however it is dressed up. Measured by
R^2 of a linear probe predicting VAD from (target, one-hot label):

| target | dims | VAD-recoverability R^2 | verdict |
|---|---|---|---|
| `allclass` (residual to every centroid) | 12 | 1.000 | VAD in disguise; rank 3 |
| signed residual, shared whitening | 3 | 1.000 | exactly linear in (VAD, label) |
| signed residual, per-class whitening | 3 | 0.983 | VAD in disguise |
| `|residual|` (any whitening) | 3 | **0.406** | genuinely lossy |
| `||residual||` scalar | 1 | 0.391 | genuinely lossy |

Every target above ~0.9 has tied its placebo in this project: global k-means,
per-class k-means, shared residual modes, `allclass`, prototype-distance
logits. Only three available quantities escape the VAD-function trap: the
**label**, **annotator distribution**, and **annotator dispersion**.

Note the whitening does not add information (it is invertible per class); the
0.983 rather than 1.000 reflects only that a linear probe cannot express a
transform that changes per class.

---

## 4. What the whitening does, and a problem with it

Whitening maps `r = vad - mu_c` through `L_c` where `L_c L_c^T = Sigma_c^-1`,
turning the class's elliptical scatter into a sphere so that `||r_w||` is the
Mahalanobis distance. It accounts for **correlations**, not just per-axis
variance: an identical 0.10 offset from the angry centroid scores 1.101 on
valence but **1.399 on dominance**, because dominance normally co-varies with
valence and arousal and moving it alone breaks that pattern.

### 4.1 The axis-reliability problem
On the native 1-7 scale, MSP-Podcast:

| axis | inter-annotator sd | spread across clips | range used (p5-p95) | reliability |
|---|---|---|---|---|
| valence | 0.762 | 0.984 | 3.25 | 0.893 |
| arousal | 0.863 | 1.010 | 3.20 | 0.869 |
| dominance | **0.734** | **0.853** | **2.80** | **0.867** |

Dominance has the lowest absolute disagreement but the narrowest range in use,
so per unit of real signal it is the least reliable axis. **Whitening weights
by 1/sqrt(variance), so the narrowest axis gets the largest weight** - the
current target leans hardest on the weakest axis.

Two corrections were proposed and **rejected on measurement**:
- Subtracting annotator-error variance before inverting raises dominance's
  relative weight from 1.20x to **3.49x** (removing noise from a narrow axis
  narrows it further).
- Weighting by 1/noise fails identically, since dominance has the lowest
  absolute disagreement.

Roughly **29% of within-class VAD spread is measurement noise** (annotator
variance / n over consensus variance), reaching **51% for sad valence**.

---

## 5. Centroid definitions

All corpora normalised to 0-1 by the pipeline's own formula, arousal
active-high everywhere (MSP-Improv corrected).

### 5.1 Are MSP-Podcast centroids robust across corpora?

| class | MSPP | IEMO | MSPI |
|---|---|---|---|
| neutral | 0.481 0.471 0.522 | 0.492 0.431 0.459 | 0.489 0.279 0.464 |
| happy | 0.639 0.590 0.600 | 0.737 0.529 0.482 | 0.765 0.485 0.603 |
| sad | 0.350 0.393 0.458 | 0.313 0.391 0.457 | 0.246 0.315 0.439 |
| angry | 0.273 0.686 0.711 | 0.226 0.659 0.737 | 0.244 0.564 0.642 |

Mean euclidean deviation from MSPP: **IEMO 0.084, MSPI 0.160**, against a mean
within-class sd of **0.120**. So the corpus-to-corpus centroid offset is
comparable to (IEMO) or larger than (MSPI) the within-class spread the
mechanism is trying to model. **Absolute VAD is corpus-biased**, which is the
strongest available argument for a class-conditional residual: the residual
cancels the corpus-level offset, the raw value does not.

Happy is the least stable class (deviation 0.165 IEMO, 0.164 MSPI): MSP-Podcast
raters are markedly less positive about happy (V 0.639) than IEMOCAP (0.737) or
MSP-Improv (0.765).

### 5.2 MSPP-only vs pooled centroids

| pooling | neutral | happy | sad | angry | shift vs MSPP |
|---|---|---|---|---|---|
| n-weighted | 0.482 0.454 0.514 | 0.654 0.579 0.594 | 0.340 0.387 0.456 | 0.268 0.675 0.707 | 0.012-0.019 |
| equal per corpus | 0.487 0.394 0.482 | 0.714 0.535 0.562 | 0.303 0.366 0.451 | 0.248 0.636 0.697 | **0.054-0.101** |

MSP-Podcast is 87.2% of the pooled sample, so n-weighted pooling is
effectively MSPP-only. Equal-weight pooling shifts centroids by up to 0.101,
which is comparable to the IEMO corpus offset itself.

**Caveat for any use of pooled centroids:** the evaluation corpora are the
targets. Fitting centroids on them leaks target statistics and voids the
zero-shot claim. Centroids must come from MSP-Podcast only.

### 5.3 Two notions of consensus give different centroids

Top-quartile within class, MSP-Podcast:

| class | all samples | top-q **VAD** agreement | top-q **CLASS** agreement |
|---|---|---|---|
| neutral | 0.481 0.471 0.522 | 0.449 0.461 0.501 | 0.488 0.454 0.501 |
| happy | 0.639 0.590 0.600 | **0.586 0.540 0.540** | **0.652 0.582 0.583** |
| sad | 0.350 0.393 0.458 | 0.345 0.369 0.437 | 0.311 0.349 0.410 |
| angry | 0.273 0.686 0.711 | 0.227 0.713 0.738 | 0.228 0.714 0.741 |

- Shift from all-sample centroid: VAD-agreement **0.057**, class-agreement **0.047**
- Distance between the two definitions: **0.045** mean
- Per-sample correlation of the two agreement measures: **r = +0.433**

They are related but far from redundant, and they disagree most on **happy**
(0.090). The direction is interpretable and opposite:
- **High VAD-agreement happy is LESS extreme** (V 0.586 vs 0.639 overall) -
  raters agree on the numbers for mild, central clips.
- **High class-agreement happy is MORE extreme** (V 0.652) - raters agree on
  the label for prototypically strong clips.

On **angry the two agree almost exactly** (distance 0.003), so the divergence
is specific to the classes where categorical and dimensional judgements come
apart.

Class-agreement centroids widen prototype separation (sad/angry +26.5%,
neutral/sad +36.1%) but **did not improve results** when tested frozen.

What relocating a centroid does and does not change is worth stating precisely,
because it is easy to overclaim. It does **not** change the information content
of the target: the shift is a per-class constant, and since the label is
already supervised, `(target, label)` still determines VAD exactly, so nothing
new becomes learnable. It **does** change the loss surface. The head is linear
(1024 -> 3, 3,075 parameters), so to fit a target shifted by a class-dependent
constant it must decode class identity from the embedding and subtract the
right offset. It can largely do that, since class is linearly decodable from
the same embedding, but not perfectly, and the attempt sends different
gradients into the trunk. The learned representation therefore **can** differ,
and in the frozen test it did (`pl_hc` != `pl_res`), just not favourably.

So the expectation is a small effect of uncertain sign, not a provable null.
Under `|residual|` the centre enters through a nonlinearity and changes the
target itself rather than only its frame, which is the stronger test.

---

### 5.4 How the centroids were computed

All figures in section 5 come from the same procedure, so they can be
reproduced exactly.

**Sources.** MSP-Podcast from the local `mspp_build` parquet (consensus VAD,
which is verified to be the plain mean of the per-annotator ratings, exact to
float precision on all 116,221 rows). IEMOCAP and MSP-Improv from their HF
arrow caches, columns `valence`/`arousal`/`domination` and
`consensus_valence`/`consensus_arousal`/`consensus_dominance` respectively.

**Normalisation.** The pipeline's own formula, so the numbers match what the
model sees: `(x - 1) / 6` for MSP-Podcast (1-7 scale) and `(x - 1) / 4` for
IEMOCAP and MSP-Improv (1-5 scale). MSP-Improv arousal is reversed before
normalising, `a -> (1 + 5) - a`, matching `orient_arousal` in
`data/dataset.py`.

**Centroid.** Unweighted mean of the normalised VAD of every 4-class sample
with that label. Rows with any non-finite VAD are dropped, which costs 6 rows
on IEMOCAP and 381 on MSP-Improv.

**Agreement measures**, both computed per utterance and then used only to
select rows, never to weight them:

    CLASS agreement = label_dist.max()
                    = fraction of annotators choosing the majority label,
                      from labels_detailed.csv (mean 5.59 annotators per clip)

    VAD agreement   = 1 / (1 + mean(annot_vad_std))
                    = inverse mean per-dimension annotator dispersion,
                      floored at 1e-6

For IEMOCAP and MSP-Improv the equivalents are `overall_agreement` and the
`valence_std`/`arousal_std`/`dominance_std` triplet.

**Consensus-rooted centroid.** Within each class independently, take the
samples at or above the `q`-th quantile of the chosen agreement measure and
average only those. Filtering within class leaves class balance untouched;
only which rows define `mu_c` changes. A class left with fewer than 50 rows
falls back to all its samples. `q = 0.75` throughout section 5.3.

**Pooling.** "n-weighted" averages the three corpus centroids weighted by
sample count; "equal-weight" averages them equally. MSP-Podcast is 87.2% of
the pooled sample, so n-weighting is nearly MSP-Podcast alone. Neither is
usable in a model: the other two corpora are evaluation targets, so fitting
centroids on them leaks target statistics.

**Code paths.** `class_vad_stats(train_data, num_classes, consensus_q,
dims, consensus_source)` in `utils/prototypicality.py` implements the
centroid and covariance fit, with `consensus_source` selecting `"class"` or
`"vad"`. Config keys: `proto_centroid_consensus_q`,
`proto_consensus_source`, `proto_vad_dims`.

---

## 6. Mechanism results

All against capacity-matched placebos.

### 6.1 Closed
subtype vocabulary (0/4; `subtype_dist` recovers the label at 89.8%, so it
restates the main task), shared residual deviation modes (0/4 UAR and emo_auc),
per-class residual clustering, k-means clustering (2/4, all deltas < 0.003 on
the corrected corpus), `allclass`, prototype-distance logits (learned alpha
0.003 when free; non-monotonic and gone by floor 0.30), consensus-defined
centroids (frozen), LDAM, muted mixup, all six ensembling variants.

### 6.2 Live: own-class whitened residual
IEMOCAP **and SAMSEMO** are positive across regimes, 3/3 seeds wherever seeds
were run. SAMSEMO across the three 3-seed experiments: **+0.0091 +/- 0.0005**
(frozen 8ep), +0.0035 +/- 0.0048 (both encoders), +0.0027 +/- 0.0043 (frozen
20ep) - positive in all three, significant in one.

IEMOCAP:

| regime | IEMO UAR gain | seeds |
|---|---|---|
| frozen, 20ep/64fr | +0.0109 | 3/3, p=0.073 |
| frozen, 8ep/32fr | +0.0079 | 3/3, p=0.017 |
| emotion2vec fine-tuned | +0.0071 | 1 seed |
| both encoders training | +0.0065 | 3/3, p=0.015 |

The threshold-free decomposition identifies what kind of gain it is:

```
IEMO neutral_auc  +0.0140   3/3   p=0.032    <- real, threshold-free
IEMO emo_auc      +0.0001   1/3   p=0.911    <- nothing
```

**It is a neutral-vs-emotional discrimination gain, not an emotion
discrimination gain.** Accuracy and weighted F1 stay flat because neutral is
the largest true class, so the recall trade is weight-neutral. This also
explains the persistent MSP-Improv failure: MSPI's errors are angry collapsing
into sad, an emotional-vs-emotional confusion, which is exactly where this
mechanism does nothing.

### 6.3 The unresolved problem
Against **no auxiliary head at all**, the mechanism is -0.0011 (IEMO) and
-0.0065 (MSPI). It beats its placebo because, with encoders training, a
scrambled target actively *damages* training (-0.0077 on IEMO) and the real
target merely repairs that damage. So:

- "Does VAD prototypicality carry information?" **Yes** (placebo comparison).
- "Should this head ship?" **Not on current evidence** (baseline comparison).

The no-aux baseline is currently single-seed and is carrying that verdict.

---

## 6.4 Effect by annotator-agreement quartile

Test-set utterances split into quartiles by `overall_agreement`, residual arm
against placebo, 3 seeds, both encoders training.

**IEMOCAP**

| quartile | n | res UAR | ctrl UAR | diff | sd | seeds |
|---|---|---|---|---|---|---|
| Q1 (lowest) | 1130 | 0.6233 | 0.6189 | +0.0043 | 0.0024 | 3/3 |
| Q2 | 1118 | 0.6410 | 0.6283 | **+0.0128** | 0.0026 | 3/3 |
| Q3 | 1246 | 0.6153 | 0.6163 | -0.0010 | 0.0105 | 2/3 |
| Q4 (highest) | 996 | 0.5960 | 0.5847 | +0.0113 | 0.0062 | 3/3 |

**MSP-Improv**

| quartile | n | res UAR | ctrl UAR | diff | sd | seeds |
|---|---|---|---|---|---|---|
| Q1 | 1950 | 0.5089 | 0.5050 | +0.0039 | 0.0082 | 2/3 |
| Q2 | 1949 | 0.5249 | 0.5170 | **+0.0079** | 0.0072 | 3/3 |
| Q3 | 1949 | 0.5447 | 0.5453 | -0.0006 | 0.0054 | 2/3 |
| Q4 | 1950 | 0.6141 | 0.6141 | **+0.0001** | 0.0071 | 2/3 |

Two findings:

**Label noise does not explain MSP-Improv's null.** Its cleanest quartile shows
+0.0001, with the two arms tying to four decimals. The hypothesis that MSPI's
consistent null was noise masking a real effect is **dead**.

**The effect is largest in Q2 in both corpora**, the second-lowest agreement
quartile, not the highest: +0.0128 on IEMOCAP (3/3, sd 0.0026, the largest and
tightest effect measured in this project) and +0.0079 on MSP-Improv (3/3). A
plausible story is that prototypicality matters most on ambiguous utterances,
where "how far from this class's norm" carries information the categorical
label does not, while clean samples are already right and hopeless ones are
beyond help. But IEMOCAP Q4 also shows +0.0113, Q3 is null in both, and these
are eight numbers with sds of 0.002-0.011. **Found post hoc; would need to be
pre-registered and retested before being claimed.**

### Label quality of the evaluation corpora

| corpus | mean agreement | model UAR Q1 -> Q4 |
|---|---|---|
| IEMOCAP | 0.421 | 0.6154 -> 0.5922 (no gradient) |
| MSP-Improv | **0.292** | 0.5142 -> **0.6335** (+0.12) |

MSP-Improv is **acted, like IEMOCAP, yet far less consistently labelled**. Its
model accuracy tracks label quality almost perfectly, and on its highest-
agreement quartile the model scores 0.6335, better than its overall IEMOCAP
score. So MSP-Improv is thinly labelled rather than mislabelled: the earlier
suspicion that its arousal inversion implied broken class labels is not
supported (label matches the `emotion` field exactly, valence and dominance
order correctly by class, transcripts are plausible).

On IEMOCAP the relationship runs the other way (higher agreement, slightly
lower UAR), for the no-aux baseline as well, so `overall_agreement` is not
measuring the same thing on both corpora.

---

## 7. Novelty position (literature review, 2026-08-24)

- **Class-conditional VAD residual as an auxiliary target: no direct prior work found.**
  Closest is EmoSphere-TTS/SER (Interspeech 2024/25), which takes a residual
  from a *single global neutral* centroid, not per-class, and without covariance.
- **Mahalanobis distance in the external VAD label space: novel in SER.**
  It appears only as an OOD score in embedding space.
- **The name is taken:** Schuller et al. 2011, "Prototypicality vs.
  Generalization", uses acoustic-feature distance for training-set selection.
- **The expected reviewer objection is the same one identified internally:**
  predicting `vad - mu_y` given the label is information-equivalent to
  predicting `vad`. It cannot be answered with novelty, only with the ablation
  grid (absolute VAD < centered < whitened, plus the permuted placebo).
- **Strongest reframing available:** absolute VAD is corpus-biased, so the
  class-conditional residual cancels the corpus-level offset and is the
  transferable part of the label. Section 5.1 measures exactly this and
  supports it (offsets of 0.084-0.160 against a within-class sd of 0.120).

---

## 8. Open / running

- `va_res` / `va_ctrl`: drop dominance, valence+arousal only. Direct test of
  whether the least reliable axis contributes or costs.
- `am_gate`: salience gate converting predicted-VAD intensity into a neutral
  logit shift, the missing conversion step for a gain that lives in
  `neutral_auc`.
- No-aux baseline at 3 seeds: required to settle section 6.3.
- `|residual|` with and without consensus centroids: the only untested target
  form with low VAD-recoverability.
- `it_base` / `it_res`: train on IEMOCAP, test on MSP-Podcast plus the other
  three. Tests whether the training corpus's label quality matters more than
  its acted/natural status. Exploratory: IEMOCAP is 4,490 rows against
  MSP-Podcast's 80,941, and the centroids are fitted on IEMOCAP.
- `cc_class` / `cc_vad`: consensus-rooted centroids at q=0.75, selecting on
  categorical and dimensional agreement respectively, both against
  `aub_b5e6_res` (all-sample centroids) on seed 42. Honest expectation is a
  null under the signed residual, since any per-class constant leaves the
  target affine in (VAD, label); the informative version is these paired with
  a nonlinear target.

## 9. Consensus centroids are inert (attribution test, 3 seeds)

`cc_ref` was added to separate "the residual head helps" from "consensus
centroids help". It is the plain residual target with ordinary all-sample
class-mean centroids; `cc_class3` is the same head with centroids estimated
from the top-quartile-consensus rows (q=0.75). Both unfrozen 2/2, bert_lr
2e-5, seeds 42/189/7.

    arm         IEMO      MSPI      CMUMOSEI  SAMSEMO
    base        0.6035    0.5387    0.5436    0.6854
    cc_ref      0.6162    0.5419    0.5439    0.6952
    cc_class3   0.6155    0.5428    0.5401    0.6971

    cc_ref    vs base   UAR  +0.0127 (3/3, p=0.103) IEMO
                             +0.0098 (3/3, p=0.078) SAMSEMO
                        neutral_auc +0.0160 (3/3, p=0.020) IEMO
                                    +0.0053 (3/3, p=0.053) SAMSEMO
    cc_class3 vs base   UAR  +0.0120 (3/3, p=0.029) IEMO
                             +0.0117 (3/3, p=0.146) SAMSEMO
                        neutral_auc +0.0144 (3/3, p=0.006) IEMO
                                    +0.0056 (3/3, p=0.037) SAMSEMO

    cc_class3 minus cc_ref  UAR  -0.0008  +0.0009  -0.0038  +0.0019

Conclusion. The consensus centroids contribute nothing: the two arms are
indistinguishable on every corpus. The +1.2 percent IEMOCAP and +1.0 percent
SAMSEMO gain over the no-aux baseline is real and survives at 3/3 seeds, but
it attributes to the residual auxiliary head, not to how the centroid is
defined. This restores the original information argument recorded in section
5: a per-class constant leaves the target affine in (VAD, label), so moving
mu_c cannot change what the head can learn. The seed-42 result that appeared
to refute that argument was single-seed noise.

Practical effect: the method simplifies. No consensus filtering, no q to
tune, no dependence on per-annotator metadata. The q ablation (cc_q50,
cc_q90) is now pointless and is dropped from the queue.

Caveat carried forward: both arms are negative on MSPI weighted F1 (-0.0154
and -0.0140). That cost belongs to the residual head itself.

Open question. `cc_ref` and `cc_class3` have nearly equal means but very
different p-values on IEMOCAP (0.103 against 0.029), which means `cc_class3`
has the tighter seed-to-seed spread. Equal-mean-lower-variance would still be
worth having, but at three seeds the difference between two variance
estimates is not itself resolvable, so this is not yet a claim.

## 10. Infrastructure note: SWA state is not checkpointed

A power outage on 2026-08-26 killed `cc_base5e6` seed 7 at epoch 8 of 8, with
training complete and the final evaluation still to run. Resuming would have
saved about two hours and would have been wrong.

`swa_window` (train.py:2990) is an in-memory list that is never written into
the checkpoint. On resume, the per-epoch loop skips every completed epoch
(train.py:3093), so the window stays empty, the `model_selection ==
'swa_last_n' and swa_window` branch at train.py:3320 evaluates false, and
control falls through to `model.load_state_dict(best_model_state)`. That is
best_val selection, which this project deliberately does not use because
validation UAR on the training corpus does not track cross-corpus test UAR.

The failure is silent: the run completes, writes results.json, and reports a
number produced under a different model-selection rule than its sibling
seeds. Any interrupted swa_last_n run must be rerun from scratch, not
resumed. Checkpoints for incomplete runs are now cleared by the launcher
before relaunch, and only for directories with no results.json, so a finished
seed is never destroyed.

## 11. The neutral_auc gain, with confidence intervals

Every earlier statement about this effect rested on three seeds, giving p
values between 0.02 and 0.10. That is not enough resolution to separate a
small real effect from seed noise, and it is not what the claim should rest
on. The saved per-sample logits allow a far more powerful test.

Method. Resample SPEAKERS with replacement, recompute the metric for both
arms on the same resample, take the paired difference, and repeat 1000 times.
Speakers rather than utterances, because utterances from one speaker are not
independent and resampling them directly gives falsely narrow intervals. The
three seeds are averaged inside each draw, so the interval reflects sampling
variability in the evaluation corpus with seed noise averaged down. AUC is
computed by a tie-corrected rank statistic verified to match
sklearn.roc_auc_score exactly. Arms are aub_b5e6_res, aub_b5e6_ctrl and the
no-aux baseline, all bert_lr 5e-6, unfrozen 2/2, seeds 42/189/7.

IEMOCAP:

    contrast                   diff       95% CI            P(>0)
    neutral_auc  res - base  +0.0109  [+0.0072, +0.0148]    1.000
    neutral_auc  res - ctrl  +0.0140  [+0.0093, +0.0184]    1.000
    emo_auc      res - base  +0.0007  [-0.0015, +0.0032]    0.735

The two intervals do not overlap. The auxiliary head improves neutral against
emotional ranking and leaves discrimination among emotions untouched, and both
halves of that statement now carry intervals rather than 3-seed p values. This
is the specificity argument in its strongest form: generic regularisation
would lift both metrics, and it does not.

Remaining corpora, neutral_auc, res - base:

    MSPI       -0.0011  [-0.0068, +0.0048]
    CMUMOSEI   -0.0033  [-0.0085, +0.0021]
    SAMSEMO    +0.0026  [-0.0000, +0.0052]   P(>0) = 0.974

So the claim is IEMOCAP-solid, SAMSEMO-suggestive, null on the other two.

CORRECTION to an earlier reading. The selectivity was described in this log
and in discussion as being flattered by a harmful placebo. That was wrong.
ctrl - base on IEMOCAP is -0.0031 with CI [-0.0080, +0.0018] and P(>0) = 0.100,
so the placebo is statistically indistinguishable from the no-aux baseline.
The +0.0140 selectivity is the mechanism gaining, not the control sinking.

LIMITATION. IEMOCAP contributes only 10 test speakers and MSP-Improv 12, so
the cluster bootstrap resamples very few units on exactly the two corpora
where the acted/naturalistic distinction matters; cluster bootstrap intervals
are known to be optimistic below roughly 30 clusters, so both intervals are
probably narrower than the truth. CMU-MOSEI and SAMSEMO carry one speaker per
utterance (1938 and 5043), so clustering is vacuous there and those intervals
are on firmer ground.

Script: analysis/bootstrap_neutral_auc.py
