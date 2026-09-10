# Literature audit

Every citation in `refs.bib` was retrieved from a primary source during the
sweep. Two carry known gaps and must not be cited further without action:

- **xia2017multitask** - bibliographic details verified only from the
  reference list of Parthasarathy and Busso 2017. The full text was NOT
  retrieved. This is the closest architectural prior art (auxiliary VAD
  regression under a categorical emotion classifier), so the paper must be
  read before we write the sentence distinguishing our work from it.
- **paralbench2024** - author list not extracted. Its cross-corpus results,
  which include the MSP-Podcast to IEMOCAP direction we want, exist only as a
  bar chart (Figure 3) with no numeric table.

## Recent work (added 2026-09-08)

The four prior-art proxies we run come from 2012 to 2017 papers, which a
reviewer will notice. These two 2026 papers close that gap and should be cited
in preference to the older ones as the statement of CURRENT practice.

### Omidi and Hansen, Interspeech 2026 - the modern comparison

"Learning from Annotation Uncertainty: Entropy-Aware Curriculum for Speech
Emotion Recognition", arXiv:2606.27536 (June 2026). Omidi and Hansen, UT
Dallas. VERIFIED FROM THE FULL PDF on 2026-09-08, not from the abstract.

Setup. WavLM-Base multitask model predicting categorical emotion AND
dimensional VAD jointly, on the full 9-class MSP-Podcast 2.0. Hard consensus
training is compared against distribution-based supervision from primary
annotator votes and from MERGED primary-secondary distributions, mixed at
0.9P/0.1S and 0.8P/0.2S. AdamW, backbone lr 1e-5, head lr 1e-4, NewBob
scheduler, batch 32, up to 18 epochs, early stopping on development Macro-F1
with patience 4. Metrics are Macro-F1 and UAR for decisions, JSD and KLD for
distributional alignment. An entropy-aware curriculum is applied to the
categorical branch only, in both standard and reverse directions, by
filtering or by weighting.

Three things matter for us.

FIRST, they use a SINGLE FIXED RANDOM SEED, with bootstrap confidence
intervals over utterances rather than seed variance. Our five seeds with
paired per-seed statistics is the stronger design, and that is worth one
sentence in the paper.

SECOND, they run NO control or placebo arm. Verified by searching the full
text for shuffle, permutation, placebo, control condition and random label:
zero matches. The entropy-stratified evaluation is a subgroup breakdown, not
a control. A 2026 Interspeech paper using distributional supervision still
validates against a hard-label baseline only.

THIRD, their categorical gains are also small: "Macro-F1 differences are
smaller than the divergence differences". The distributional objectives
clearly improve JSD and KLD, but hard-decision performance barely moves, and
they note hard-label systems stay competitive on Macro-F1 partly because they
predict the catch-all Other class well (Hard-CBCE reaches 36.0/33.0 Other-class
F1 against 0.9/0.9 for Prim-KLD). That is useful context when defending our
own sub-point effects: a current Interspeech paper reports the same pattern.

They also use curriculum learning, a direction this project has ruled out, so
that is a clean point of difference rather than an overlap.

### Wong, Talat, Aldarmaki and Field, 2026 - NOT about controls

"Unrequited Emotions: Investigating the Gaps in Motivation and Practice in
Speech Emotion Recognition Research", arXiv:2604.25776 (April 2026). Wong
(JHU), Talat (Edinburgh), Aldarmaki (MBZUAI), Field (JHU).

CORRECTION, 2026-09-08. An earlier version of this file claimed this paper
criticises SER for inadequate control conditions, weak baselines and missing
ablations. That is WRONG. It came from an automated page summary that
fabricated the content, and the error was caught by the researcher rather than
by verification. The full text contains no mention of control conditions,
placebos, ablations or baselines.

What it actually argues: a systematic survey of stated motivations in SER
research against the datasets actually used. It finds that papers cite
appealing goals such as voice-activated systems or healthcare applications
while the common datasets do not reflect those deployment contexts, and argues
this misalignment raises ethical concerns about misuse and downstream harm.
It is a critical and ethics survey, not a methodological one.

Whether to cite it. Not as support for the control-task argument, which it
does not make. It could support a sentence in limitations about corpus choice
and deployment validity, since MSP-Podcast, IEMOCAP and SAMSEMO are podcast,
acted-dyadic and film data respectively rather than any stated application
domain. That is honest but tangential; leaving it out is also fine.

LESSON, worth keeping: two automated page summaries produced fabricated
methodological content on 2026-09-08, this one and a phantom shuffled-label
control in Omidi and Hansen. Extract the PDF text and grep it before writing
any claim about a paper into these notes.

### How to frame the comparison, given these

Do not present the table as "we beat Kim and Provost 2015". Present it as:
these are the auxiliary signals the field uses, from the 2012-2017 papers that
introduced them through to Omidi and Hansen 2026, and none of the papers
reporting gains from them run a capacity-matched control. Here is what each
signal is worth in an identical setup when one is applied.

## Closest prior art, in order of risk

1. **Kim and Provost, ACII 2015.** Defines "prototypicality" as annotator
   agreement, uses it as a per-instance SVM weight on IEMOCAP four-class, and
   finds the gain concentrated on neutral (43.64 to 46.75). Also reports that
   the information gain of emotion given prototypicality is nearly identical
   for the four-class and neutral-vs-rest cases (0.058 against 0.052 bits),
   meaning almost all of it concerns neutral. Distinctions to state: their
   score is label-side (needs rater counts at test time), ours is a geometric
   VAD residual; theirs is a loss weight, ours an auxiliary target; theirs is
   within-corpus with per-speaker oracle weights, ours zero-shot; they run no
   control arm.
2. **Xia and Liu, IEEE TAC 2017.** The canonical auxiliary-VAD-under-
   categorical architecture. Our differentiation is the target, not the
   architecture: the within-class whitened residual is orthogonal to the class
   label by construction and so cannot be a re-encoding of the main task.
3. **Schuller et al. 2011.** Objective prototypicality as distance in acoustic
   feature space, used for cross-corpus training-data selection. Nearest
   ancestor on the cross-corpus axis. Their gain was on arousal; valence was
   not significant.
4. **Eyben et al. 2012.** Adds inter-rater standard deviation as an extra MTL
   target, structurally the same move as ours but with a label-side target.

## The control-arm gap

Across every verified SER auxiliary-task paper (Xia and Liu 2017,
Parthasarathy and Busso 2017, Eyben et al. 2012, Latif et al. 2022), none runs
a capacity-matched control. The nearest is the "random curriculum" row in
Lotfian and Busso 2019, which shuffles an ordering rather than a
sample-to-target pairing and does not hold head capacity fixed. The claim to
be first in applying a Hewitt-Liang control task to an auxiliary head in SER
is supported by this search, but must be phrased "to our knowledge".

Note one design difference a probing-literate reviewer will catch: Hewitt and
Liang randomise at the type level with a fixed assignment, so their control is
learnable by memorisation. Our within-batch permutation is instance level and
resampled every step, which destroys learnability instead of relocating it. A
regression target has no word-type analogue, so this is a necessary
adaptation; it preserves the target marginal exactly, which is the property
capacity matching needs. Justify it explicitly in the method section.

## Alternative explanation we must rule out

Zhang et al. 2026 ("Cosine Misleads", arXiv:2606.05753) find that auxiliary
objectives in vision-language models reshape the model through shared
parameters rather than through the latent they nominally optimise, and that
corrupting the supervised latent shifts accuracy by at most four points. That
is exactly the null hypothesis our permuted control addresses: a
shared-parameter mechanism survives permutation, an information mechanism does
not. Cite as motivation for the control design rather than as an established
result (recent preprint, different modality).

## Verified gaps in the literature

1. No MSP-Podcast to IEMOCAP zero-shot four-class categorical UAR number
   exists in any retrievable source. ParaLBench ran the experiment but
   published it only as a figure.
2. No control task or capacity-matched placebo has been applied to speech
   representations or to auxiliary-task learning in SER.
3. No data-cartography-style training-dynamics analysis of speech or emotion.
4. No prior use of a covariance-whitened within-class VAD residual as a
   target, weight, or score.

Item 4 is the novelty claim and it survived the search.

## Unverified, do not cite without checking

- Ando et al., soft-target training with ambiguous emotional utterances,
  IEEE Xplore 8461299, probably ICASSP 2018. Author list from search snippets
  is garbled. Highly relevant; worth chasing.
- arXiv:2606.27536, entropy-aware curriculum for SER, stated as Interspeech
  2026. Very close to our topic and very recent.
- arXiv:2306.06232, probing SSL speech models for aspiration.
- Chou and Lee, Interspeech 2020, co-rater training with soft and hard labels.
