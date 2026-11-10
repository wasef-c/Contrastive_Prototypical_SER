# Handoff: VAD, categorical labels, and eight failed mechanisms

Branch `levers`. Everything below is reproducible from `analysis/` with the
caches already on disk. Written 2026-11-10.

## One-paragraph summary

The measurement side of this work holds up and is worth a paper. The method
side does not: eight mechanisms were tried and the only effect any of them
produced is that **softer targets help cross-corpus SER by about half a
point of UAR**, with no evidence that *which* soft target matters. A
failure analysis pooling 37 model-seed runs then explains why — a fifth of
each evaluation corpus is wrong in **every** run, so the target was never
the binding constraint.

---

## 1. What survived, with the numbers

### VAD underdetermines categorical emotion

| finding | value | where |
|---|---|---|
| Unanimous panels at **identical** VAD carry different labels | **50.4%** (baseline 67.7%) | `vad_label_conflict.py` |
| Regional conflict map, validated against estimator-free counting | **r = +0.826** | `vad_label_conflict.py` |
| VAD explains this share of the **true** between-utterance variance in annotator labelling | **37.1%** | `critique.py` (tmp) |
| Of the observed variance, sampling noise accounts for | **37.1%** | same |

The 50.4% figure needs no estimator: 4,412 MSP-Podcast utterances have a
unanimous four-class label **and** zero annotator dispersion in VAD, and
within the 98 groups sharing a VAD coordinate exactly, pairs disagree on
the label half the time.

### Model failures are VAD-atypical

Pooling 37 model-seed runs (`failure_report.py`):

| | IEMOCAP | SAMSEMO |
|---|---|---|
| right in **every** run | 36.2% | 49.3% |
| **wrong in every run** | **19.0%** | **16.8%** |
| pooled accuracy | 0.631 | 0.681 |

The hard core is atypical of its own label in VAD: **47.4%** of it sits
closer to another class's VAD centroid than its own, against **18.8%** of
the rows every run gets right. Over half the core is an emotional
utterance being called neutral, while neutral is the *easiest* class.

Arm families fail on the **same rows** — always-wrong sets overlap at
77–95% — and mean pairwise agreement on error-vs-correct is **0.898**
against **0.534** for independent errors at the same rate. No ensembling
or routing across these families can help.

---

## 2. What was retracted, and why

These were all claimed in this project and then falsified by a later
check. They are listed so nobody re-derives them.

**"The label space is a three-dimensional continuum."** The basis fitted
to annotator vote distributions and the basis fitted to one-hot majority
labels span the same subspace (principal angles **2.59, 2.83, 6.91
degrees**, against ~80 for unrelated 3-d subspaces). Four classes span
three dimensions by arithmetic. Inside that basis the variance is **66.5%
class identity**, and of the 33.5% within-class part the exactly projected
multinomial sampling noise accounts for **89%** — leaving **3.7%** for
anything else. See `vs_hardlabels.py` (tmp).

**"PC1 is a direction VAD cannot see."** Fitted jointly and nonlinearly
with speakers held out across 2,112 speakers, VAD recovers **32.4%** of
PC1, 59.4% of PC2, 42.2% of PC3. Low pairwise correlations concealed this.
What survives is only that PC1 is simultaneously the largest component and
the least VAD-predictable.

**"A missing fourth axis."** A residual axis fitted from annotator
substitution gave **2%** out-of-sample improvement under
leave-one-label-out — overfitting 7 free positions to 21 distances.

**"The missing axis is communicative/linguistic."** VAD beat transcripts
on **all 21** label pairs, and adding transcripts to VAD changed mean
balanced accuracy by **−0.0004**. No pair improved by even 0.02.

**"The affect-geometry mixup gain is the geometry."** `uf4_softmix_prior`
(identical config, empirical priors instead of uniform) takes the +1.14
IEMOCAP UAR gain to **−0.04**. Documented already at
`configs/class_weight.yaml:3409`; confirmed here.

---

## 3. The eight mechanisms

All at 8 epochs on MSP-Podcast's **official** Train/Development split
(`val_split_official: true`), which costs 46% of the angry four-class rows
and makes these arms incomparable to the mixup table.

| arm | target | IEMOCAP UAR | verdict |
|---|---|---|---|
| `rl_hard` | majority one-hot | 62.40 | baseline |
| `rl_raw` | annotator distribution | 63.41 | works |
| `rl_dir` | distribution weighted by VAD-local "realness" | 63.03 | failed |
| `rl_dir_hi` | same, gain 4, weight floor 0 | 62.23 | failed, dose-response wrong way |
| `rl_lowrank` | distribution projected to rank 3 | 63.70 | works |
| `rl_kernel` | measured substitution kernel | **49.10** | collapsed, sad recall 2.7% |
| `rl_ot` | transport loss, lambda 0.5 | −0.23 vs base | inert |
| `rl_ot_hi` | transport loss, lambda 2.0 | −1.23 vs base | worse |

At `train_subsample_frac: 0.9`, three seeds (`score_s9.py`):

| | IEMOCAP UAR | Acc | F1 | SAMSEMO UAR | Acc | F1 |
|---|---|---|---|---|---|---|
| `rl_lowrank_s9` − `rl_hard_s9` | **+0.62** | +0.78 | +0.70 | +0.04 | **+1.34** | **+1.22** |

Positive on all three seeds, paired sd 0.26, mean/se 4.2.

### The entropy control settles the mechanism

Four arms at matched data, seed 42 only:

| arm | target entropy | IEMOCAP UAR gain |
|---|---|---|
| `rl_hard_s9` | 0.000 | — |
| `rl_raw_s9` | 0.486 | +0.15 |
| `rl_smooth_s9` | **0.708** | **+0.67** |
| `rl_lowrank_s9` | **0.708** | **+0.76** |

`rl_smooth_s9` is the raw distribution mixed toward uniform by a single
scalar (epsilon 0.0921, solved so mean entropy matches to four decimals)
with **no structure at all**. `rl_lowrank − rl_smooth` is **+0.09 UAR,
−0.18 Acc, −0.12 F1** on IEMOCAP and **+0.07 / −0.18 / −0.33** on
SAMSEMO. Indistinguishable.

**So the arm works and the explanation is mundane.** The gain is real and
replicated; the principal-component basis contributes nothing a scalar
could not. Keep `rl_lowrank` as a practical choice, drop the continuum
story. Caveat: `rl_lowrank_s9` is n=3 and `rl_smooth_s9` is n=1, so the
equivalence rests on one seed of the control — two more seeds would settle
it.

---

## 4. Null results from the detector and probe

Both were attempts to leave the target alone, since the failure analysis
says the target cannot reach the hard core.

**Predicting annotator reliability from audio: not possible.**
Speaker-held-out R² = **−0.041**. Consistent with 89% of within-class vote
variance being sampling noise — a five-person split is largely luck.

**Predicting VAD-categorical disagreement from audio: only through the
class.** Overall R² = **0.187**, but with class means removed, R² =
**0.000**. The model learns "neutral is often VAD-atypical (55.1%) and
angry rarely is (12.0%)", not anything per-utterance.

**The prototype hypothesis: no signal at fixed VAD.** Inside the 23
mixed-label coincident-VAD groups (2,515 rows), same-label pairs sit at
cosine distance 0.3624 and different-label pairs at 0.3660 — Cohen's d
**+0.033**, cross-speaker pairs only. Within-group label prediction with
speakers held out gives balanced accuracy **0.332** against a shuffled
baseline of **0.334 ± 0.013**. Exactly chance.

**Caveat that matters:** these embeddings come from `rl_hard`, a model
trained to predict the four-class label, so it was optimised to discard
within-class variation. The test is biased toward the null. Repeating it on
raw self-supervised features (emotion2vec before fine-tuning) has **not**
been done and is the obvious next step before concluding the information
is absent from the signal.

---

## 5. Infrastructure findings worth knowing

**Seeds do not vary training on the official split.** Runs come out
bitwise identical — same logits, same final loss to five decimals. The
generators, the length-bucketed sampler and the data order all vary
correctly; **model initialisation does not** (initial trainable weight sum
−136517.507113 for every seed). With a fixed start and 89.8M of 235.8M
trainable parameters pretrained, batch order washes out over a cosine
schedule.

Consequence: **variance in this project comes from the train/val split
draw, not from training stochasticity**, so any seed error bar quoted
elsewhere measures which rows were held out. The `_s9` arms work around it
via `train_subsample_frac`, drawn with the run seed. **Seeding the head
initialisation is the real fix and has not been done.**

**SAMSEMO has no usable VAD** in `analysis/heldout_meta/test_SAMSEMO.npz` —
every row is (0.5, 0.5, 0.5). Any VAD analysis on that corpus is vacuous.

**Speaker fields are unusable for failure analysis.** IEMOCAP has 10
speakers, so concentration statistics are near-proportional; SAMSEMO
assigns one synthetic speaker per row.

---

## 6. If I were continuing

1. **Repeat the within-group probe on raw emotion2vec features.** The
   strongest claim available — "the 50.4% conflict is not recoverable from
   the signal" — currently rests on embeddings that were trained to throw
   that information away.
2. **Seed the head initialisation**, then re-measure anything that quotes
   seed variance.
3. **Two more seeds of `rl_smooth_s9`** if you want to state the mechanism
   rather than just the arm.
4. **Write the measurement paper.** VAD explains 37.1% of true annotator
   variance; unanimous panels at identical VAD disagree 50.4% of the time;
   47.4% of universally-misclassified utterances are VAD-atypical of their
   label. The eight mechanisms are evidence for the bound, not the
   contribution.

What I would **not** do: another target-side mechanism. Eight is enough,
and the 19% core is invariant across all 37 runs.

---

## 7. Files

### Analysis, in dependency order
- `analysis/vad_label_conflict.py` — the matched-VAD conflict estimator,
  both corpora. Analytic and split-half estimators, shuffle control.
- `analysis/disagreement_direction.py` — splits panel disagreement into
  real-axis and noise-axis parts.
- `analysis/build_realness_map.py` — fits the regional map on official
  Train, emits per-utterance alpha plus a placebo.
- `analysis/build_continuum_targets.py` — PC basis, low-rank targets,
  substitution kernel, transport cost, entropy-matched smoothing control.
- `analysis/failure_report.py` — the 37-run failure analysis. **Use this
  one**, not `failure_modes.py`, which has the biased centroid.
- `analysis/prototype_modes.py` — are the failures a secondary mode?
- `analysis/ambiguity_detector.py` — the two detector targets.
- `analysis/within_group_probe.py` — prototype hypothesis at fixed VAD.

### Outputs
- `analysis/failure_report/{iemocap,samsemo}_core.csv` — 400 worst rows
  each, with speaker, true class, modal prediction, runs-wrong, VAD,
  atypicality flag, distances.
- `analysis/failure_report/summary.json`
- `analysis/continuum/mspp_continuum.npz`,
  `analysis/realness_map/mspp_official_train.npz` — the fitted targets.
- `logs_realness/*.log` — every run and analysis.

### Known-bad
- `analysis/failure_modes.py` — superseded. Its "6.6% baseline" compares
  easy rows against the **neutral** centroid because the modal prediction
  array was only filled for core rows. Kept for provenance only.

### Published pages
- Arm reference: https://claude.ai/artifact/DePH79bLmtv6mjSruJk38d
- Confusion matrices: https://claude.ai/artifact/9JwxzyDbj5MmTv8Y44u7AK
- Low-rank basis: https://claude.ai/artifact/JgDmYFYzwsHX75EXQsGtZr
- Matched-VAD explainer: https://claude.ai/artifact/DMExvpJ4wwVDWSRhScgWZH

The first three predate some retractions in section 2; the low-rank page
was corrected, the other two were not.

---

## 8. The VAD-matched triplet experiment (queued 2026-11-10)

The ninth mechanism, and the first that is not target-side. Sections 1 and
4 are the reason: the target was never the binding constraint, and 47.4%
of the universally-misclassified core is VAD-atypical of its own label. So
this arm leaves the target as hard labels and changes what the
representation is asked to separate.

A margin triplet on the **raw fused embedding** (not the contrastive
projection: a null in projection space is uninterpretable, because the
margin can be satisfied there while the classifier, which reads the raw
embedding, never changes a prediction). The negative is a clip of the
confusable class whose VAD is within 0.05 of the anchor's, from a
different speaker. VAD selects the negative offline and is never a model
input.

### Why both directions

The first design used happy anchors against neutral negatives only. That
is satisfiable by sliding the happy/neutral boundary toward happy:
quiet-happy recall rises, neutral-called-happy rises with it, and no cue
has been learned. Anchoring both directions makes moving the boundary cost
as much as it gains.

### Why this pair and no others

Coverage for every ordered pair, anchors restricted to VAD-atypical clips,
official Train, agreement >= 0.8, radius 0.05, cross-speaker:

| anchor -> negative | anchors | covered | cov% | gini | cap 8 |
|---|---|---|---|---|---|
| neutral -> happy | 5669 | 4322 | 76.2% | 0.57 | 3086 |
| neutral -> sad | 5669 | 3899 | 68.8% | 0.57 | 2809 |
| happy -> neutral | 2199 | 2158 | 98.1% | 0.70 | 1063 |
| happy -> sad | 2199 | 1689 | 76.8% | 0.73 | 715 |
| neutral -> angry | 5669 | 1169 | 20.6% | 0.44 | 1057 |
| sad -> neutral | 370 | 336 | 90.8% | 0.34 | 325 |
| happy -> angry | 2199 | 300 | 13.6% | 0.53 | 245 |
| sad -> angry | 370 | 251 | 67.8% | 0.27 | 247 |
| sad -> happy | 370 | 220 | 59.5% | 0.31 | 214 |
| angry -> sad | 184 | 168 | 91.3% | 0.30 | 167 |
| angry -> neutral | 184 | 160 | 87.0% | 0.28 | 160 |
| angry -> happy | 184 | 72 | 39.1% | 0.16 | 72 |

**Angry is unavailable rather than untried.** Only 184 angry clips are
VAD-atypical at all and happy-to-angry covers 13.6%. That is the same
fact as angry being VAD-atypical only 12.0% of the time in the detector
probe (section 4): angry occupies a VAD region nothing else reaches, so
there is no matched-negative experiment to run there. **Sad** has coverage
but 214-325 anchors after capping, too few to move a 53k-row pool.
Happy/neutral and neutral/sad are the only viable pairs; the first arm
takes happy/neutral, where errors run both ways.

### Build

Anchor rule is symmetric: a clip closer to the OTHER class's VAD centroid
than to its own. This replaced the happy-side `arousal < 0.5` filter,
which does not transfer to neutral; the centroid rule also yields more
happy anchors (2,142 usable against 1,810) and covers 1,416 of those
1,810, so the low-arousal IEMOCAP endpoint stays relevant.

```
happy_to_neutral: pool 2176, usable 2142, kept 1048 / 402 spk (cap 8, gini 0.44)
   matched median 0.0431 within radius 100.0%   random 0.2134 / 4.3%
neutral_to_happy: pool 2827, usable 2789, kept 1191 / 748 spk (cap 2, gini 0.15)
   matched median 0.0471 within radius 100.0%   random 0.1417 / 4.1%
matched 2239 triples, random 2235; distinct negatives 1571 vs 1570
```

Caps chosen for parity (1.14:1), so neither direction dominates;
uncapped, neutral would outweigh happy close to 3:1.

### The control is the whole point

`rl_trip_rand` carries the same term, the same duty cycle and the **same
multiset of negative clips**, reassigned by a shuffle within direction.
100% of matched negatives sit inside the radius against ~4% of shuffled
ones. A gain in both arms is the triplet term, not the VAD matching, and
is not the claim. Stated confound: the global shuffle changes
anchor-to-negative speaker pairings. A within-speaker shuffle preserves
them but leaves 53.4% of negatives inside the radius, which would make the
control most of the way to the treatment.

### Mechanics

Triplets are served by a **separate loader on a duty cycle**, not packed
into the main batch: packing would make main batch composition a function
of the arm, and batch composition is already known here to move a target
roughly 6x more than a mechanism does. The classification stream is
byte-identical between arms.

The triplet forward and backward run **before** the main forward, so the
two graphs never coexist; peak VRAM is the larger of the two rather than
their sum (measured 12.9 of 24.5 GB against ~11 for the baseline arm).
6 triplets every 3 main batches, ~3 traversals of the triplet set per
epoch, ~37% extra forward cost. 1,633 of 2,239 triplets resolve against
the 0.9 subsample, which is 0.9 cubed as expected.

### Endpoints, fixed before the runs

1. **Control contrast**, `rl_trip_vad` minus `rl_trip_rand`, per corpus.
2. **Subgroup**: IEMOCAP low-arousal happy recall. Baseline `rl_hard_s9`
   is **46.39%** (n=3: 46.0, 46.2, 47.0).
3. **Boundary guard**: neutral called happy. Baseline **14.97%** (12.9,
   16.3, 15.7). A subgroup gain with a proportionate rise here is the
   boundary moving, not a cue.

Also watch the **margin-active fraction** printed each epoch. If it
collapses to zero the term stopped contributing and a null means nothing.

### Files
- `analysis/vad_matched_pairs.py` - coverage, any ordered pair via
  `--anchor`/`--negative`.
- `analysis/triplet_design.py` - sampling weight, caps, control contrast.
- `analysis/build_triplets.py` - the bidirectional builder.
- `utils/triplets.py` - key resolution and the cycling batch sampler.
- `analysis/score_triplets.py` - the three endpoints above.
- `run_triplets.sh` - interleaved by seed, so stopping early leaves
  complete paired seeds.
