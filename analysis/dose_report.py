"""Regenerate docs/DOSE_RESPONSE.md from whatever arms are on disk.

Safe to run mid-sweep: unfinished arms are listed as pending rather than
omitted, so the table always shows the full design.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

ROOT = '/home/rml/Documents/pythontest/Emotion2Vec_Contrastive'
OUT = os.path.join(ROOT, 'docs', 'DOSE_RESPONSE.md')
OUT_UNFROZEN = os.path.join(ROOT, 'docs', 'DOSE_RESPONSE_UNFROZEN.md')
CORPORA = ['IEMO', 'MSPI', 'CMUMOSEI', 'SAMSEMO']
METRICS = ['neutral_auc', 'emo_auc', 'uar', 'f1_weighted']
S3 = [42, 189, 7]

Row = Tuple[str, str, Dict[int, str]]

# Frozen screen: bert_lr 2e-5, unfreeze 0/0. New arms are 1 seed; the two
# anchors already existed at 3 seeds.
FROZEN: List[Row] = [
    ('`0` no aux head', 'dose', {42: 'fr_base_seed42'}),
    ('`0.25` real target', 'dose', {42: 'fr_w025_seed42'}),
    ('`0.5` real target', 'dose', {42: 'fr_w050_seed42'}),
    ('`1.0` real target (anchor)', 'dose',
     {s: f'aub_b2e5_res_seed{s}_seed{s}' for s in S3}),
    ('`2.0` real target', 'dose', {42: 'fr_w200_seed42'}),
    ('`4.0` real target', 'dose', {42: 'fr_w400_seed42'}),
    ('`1.0` shuffled placebo (anchor)', 'placebo',
     {s: f'aub_b2e5_ctrl_seed{s}_seed{s}' for s in S3}),
    ('form: `1 - residual` (audit)', 'form', {42: 'fr_inv_seed42'}),
    ('form: `exp(-d)` protoscore', 'form', {42: 'fr_score_seed42'}),
    ('form: `exp(-d)` placebo', 'placebo', {42: 'fr_score_ctrl_seed42'}),
]

# Unfrozen sweep: bert_lr 5e-6, unfreeze 2/2. The regime the claim lives in.
UNFROZEN: List[Row] = [
    ('`0` no aux head', 'dose', {42: 'aub_b5e6_base_seed42',
                                 189: 'cc_base5e6_seed189_seed189',
                                 7: 'cc_base5e6_seed7'}),
    ('`0.25` real target', 'dose', {42: 'us_w025_seed42'}),
    ('`0.5` real target', 'dose', {42: 'us_w050_seed42'}),
    ('`1.0` real target (anchor)', 'dose',
     {s: f'aub_b5e6_res_seed{s}_seed{s}' for s in S3}),
    ('`2.0` real target', 'dose', {42: 'us_w200_seed42'}),
    ('`1.0` shuffled placebo (anchor)', 'placebo',
     {s: f'aub_b5e6_ctrl_seed{s}_seed{s}' for s in S3}),
    ('form: `1 - residual` (audit)', 'form', {42: 'us_inv_seed42'}),
    ('form: `exp(-d)` protoscore', 'form', {42: 'us_score_seed42'}),
]

# Headline pair extended to 5 seeds. Listed separately so the added seeds are
# visible rather than silently folded into the weight 1.0 rows above.
SEEDS5: List[Row] = [
    ('no aux head, 5 seeds', 'dose', {
        42: 'aub_b5e6_base_seed42',
        189: 'cc_base5e6_seed189_seed189',
        7: 'cc_base5e6_seed7',
        1234: 'us_base_more_seed1234_seed1234',
        2024: 'us_base_more_seed2024_seed2024'}),
    ('real target w=1.0, 5 seeds', 'dose', {
        42: 'aub_b5e6_res_seed42_seed42',
        189: 'aub_b5e6_res_seed189_seed189',
        7: 'aub_b5e6_res_seed7_seed7',
        1234: 'us_res_more_seed1234_seed1234',
        2024: 'us_res_more_seed2024_seed2024'}),
]


def load(run_dir: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Return {corpus: {metric: value}} for a finished run, else None."""
    path = os.path.join(ROOT, 'checkpoints', run_dir, 'results.json')
    if not os.path.exists(path):
        return None
    return {e['dataset']: e['results']
            for e in json.load(open(path))['test_results']}


def cell(dirs: Dict[int, str], corpus: str,
         metric: str) -> Tuple[Optional[float], int]:
    """Mean of one metric over whichever seeds have finished."""
    vals = []
    for run_dir in dirs.values():
        res = load(run_dir)
        if res and corpus in res and res[corpus].get(metric) is not None:
            vals.append(res[corpus][metric])
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def table(rows: List[Row], metric: str) -> str:
    """Render one metric's table for a set of arms."""
    out = [f'### {metric}', '',
           '| arm | seeds | ' + ' | '.join(CORPORA) + ' |',
           '|---|---|' + '---|' * len(CORPORA)]
    for label, _kind, dirs in rows:
        cells, n_max = [], 0
        for corpus in CORPORA:
            v, n = cell(dirs, corpus, metric)
            n_max = max(n_max, n)
            cells.append('_pending_' if v is None
                         else f'{v:.4f}' + ('' if n > 1 else ' *'))
        out.append(f'| {label} | {n_max or "-"} | ' + ' | '.join(cells) + ' |')
    out.append('')
    return '\n'.join(out)


FROZEN_HEADER = """# Frozen screen: dose response and target form

Two blocks. The frozen screen is fast and is for RANKING ideas. The unfrozen
reference is slow and is what any claim must ultimately rest on.

`*` marks a single-seed value. Read those as trend points, never as results:
within-arm seed spread is about 0.008 IEMOCAP UAR and 0.004 `neutral_auc`,
which is the same size as the effects being looked for.

## What is being tested

Single-dose comparisons cannot separate "this target carries information"
from "an extra regression head perturbs training in a way that happens to
help". A dose response can. Two predictions:

1. `neutral_auc` varies systematically with the auxiliary loss weight, rather
   than jumping at any weight above zero.
2. `emo_auc` stays flat across the whole range.

The second is the specificity control and is load-bearing. The target is the
whitened residual to the sample's OWN class centroid, so it is conditioned on
the label and carries within-class deviation only, with between-class
geometry subtracted out by construction. It should sharpen the boundary
between low and high deviation, which is the neutral against emotional
boundary, and do nothing for angry against sad. A curve lifting both metrics
together would instead indicate generic regularisation, the main rival
explanation.

## Target forms

- `residual` is the incumbent: whitened residual to the own-class centroid.
- `1 - residual` is an AUDIT, predicted null. The transform is affine and the
  head ends in a bare Linear, so its final layer absorbs the negation and the
  shift; MSE is symmetric, so the optimum and the gradient reaching the shared
  trunk are unchanged. The same holds through a sigmoid, since
  sigmoid(-z) = 1 - sigmoid(z). This is the argument that retired the
  consensus centroids. A difference here beyond seed noise falsifies it.
- `exp(-d)` protoscore is a real candidate. Monotone but NOT affine, so it
  changes the geometry of the loss rather than relabelling it. Whitened
  distances have a long right tail, so MSE on the raw distance is dominated by
  the most atypical samples; exp(-d) compresses that tail and reallocates the
  head's resolution to samples near the class centre, which is where the
  neutral boundary sits. It has its own matched placebo because its head is
  scalar where the residual head is 3-dimensional.

"""

FROZEN_NOTE = """## Frozen screen (bert_lr 2e-5, unfreeze 0/0, ~0.5 h per arm)

Validated as a RANKING tool only, by comparing the arms that exist in both
regimes (res minus ctrl):

|            | IEMO `neutral_auc` | IEMO `emo_auc` | CMUMOSEI `neutral_auc` |
|---|---|---|---|
| frozen 2e-5   | +0.0093 (3/3) | +0.0024 (3/3) | +0.0044 (3/3) |
| unfrozen 5e-6 | +0.0140 (3/3) | +0.0001 (1/3) | -0.0033 (0/3) |

The IEMOCAP `neutral_auc` effect keeps its sign and its 3/3 consistency, so
frozen ranks target forms correctly on that metric. Two things do not carry
over: frozen also lifts `emo_auc` 3/3, so the specificity dissociation is
invisible here, and CMU-MOSEI reverses sign outright.

Score this block on IEMOCAP `neutral_auc` against the matched placebo. Do not
read effect sizes, specificity, or CMU-MOSEI from it.

"""

UNFROZEN_HEADER = """# Unfrozen dose response and target form

bert_lr 5e-6, emotion2vec and BERT both unfrozen at the top 2 layers, trained
on MSP-Podcast, 8 epochs, SWA over the last 3. This is the regime the claim
lives in; the frozen screen is a separate document and its numbers do not
belong in the same table.

`*` marks a single-seed value. Read those as trend points, never as results:
within-arm seed spread here is about 0.008 IEMOCAP UAR and 0.004
`neutral_auc`, the same size as the effects being looked for. Rows without a
`*` are seed means over 3 or 5 seeds.

## What this tests

A single on/off comparison cannot separate "this target carries information"
from "an extra regression head perturbs training in a way that happens to
help". A dose response can. Two predictions:

1. `neutral_auc` varies systematically with the auxiliary loss weight, rather
   than jumping at any weight above zero.
2. `emo_auc` stays flat across the whole range.

The second is the specificity control and is load-bearing. The target is the
whitened residual to the sample's OWN class centroid, so it is conditioned on
the label and carries within-class deviation only, with between-class geometry
subtracted out by construction. It should sharpen the boundary between low and
high deviation, which is the neutral against emotional boundary, and do
nothing for angry against sad. A curve lifting both metrics together would
instead indicate generic regularisation, the main rival explanation.

## Established at weight 1.0, 3 seeds, speaker-clustered bootstrap

    IEMOCAP  neutral_auc  res - base  +0.0109  [+0.0072, +0.0148]
    IEMOCAP  neutral_auc  res - ctrl  +0.0140  [+0.0093, +0.0184]
    IEMOCAP  emo_auc      res - base  +0.0007  [-0.0015, +0.0032]

The two intervals do not overlap. What the dose curve below adds is whether
that gain scales with the loss weight, which is the difference between an
effect and a causal account of it.

## Target forms

- `residual` is the incumbent: whitened residual to the own-class centroid.
- `1 - residual` is an AUDIT, predicted null. The transform is affine and the
  head ends in a bare Linear, so its final layer absorbs the negation and the
  shift; MSE is symmetric, so the optimum and the gradient reaching the shared
  trunk are unchanged. The same holds through a sigmoid, since
  sigmoid(-z) = 1 - sigmoid(z). In the frozen screen it matched the residual
  to within 0.0001 UAR on CMU-MOSEI, the corpus with the largest effect.
- `exp(-d)` protoscore is monotone but NOT affine, so it changes the geometry
  of the loss rather than relabelling it. It was null on every corpus in the
  frozen screen; it is retested here only because frozen was shown not to rank
  this mechanism reliably.

"""

FOOTER = """## Reading guide

What matters is the SHAPE of `neutral_auc` against the FLATNESS of `emo_auc`.
Adjacent single-seed doses are not separable from seed noise; a monotone trend
across five doses is worth far more than any individual point.

Known cost carried by the real target so far: `f1_weighted` runs below the
no-aux baseline, most consistently on MSP-Improv. Watch whether that cost also
scales with the dose. If it does, the head trades calibration for neutral
discrimination and the trade is tunable. If it does not, the two effects are
independent.
"""


def main() -> None:
    """Write the report to docs/DOSE_RESPONSE.md."""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    frozen = [FROZEN_HEADER, FROZEN_NOTE]
    for metric in METRICS:
        frozen.append(table(FROZEN, metric))
    frozen.append(FOOTER)
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(frozen))
    print(f'wrote {OUT}')

    unfrozen = [UNFROZEN_HEADER]
    for metric in METRICS:
        unfrozen.append(table(UNFROZEN, metric))
    unfrozen.append('## Headline pair at 5 seeds\n')
    for metric in ['neutral_auc', 'emo_auc']:
        unfrozen.append(table(SEEDS5, metric))
    unfrozen.append(FOOTER)
    with open(OUT_UNFROZEN, 'w') as fh:
        fh.write('\n'.join(unfrozen))
    print(f'wrote {OUT_UNFROZEN}')


if __name__ == '__main__':
    main()
