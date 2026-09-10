"""Per-corpus, per-seed comparison of the bert_lr 5e-6 arms."""
import json
import os
from itertools import combinations
from typing import Dict, Optional

import numpy as np

CORPORA = ['IEMO', 'MSPI', 'CMUMOSEI', 'SAMSEMO']
METRICS = ['uar', 'neutral_auc', 'emo_auc', 'f1_weighted']
ROOT = '/home/rml/Documents/pythontest/Emotion2Vec_Contrastive'


def load(d: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Load per-corpus metrics for one finished run directory."""
    p = os.path.join(ROOT, 'checkpoints', d, 'results.json')
    if not os.path.exists(p):
        return None
    out = {}
    for e in json.load(open(p))['test_results']:
        r = e['results']
        out[e['dataset']] = {m: r.get(m) for m in METRICS}
    return out


ARMS = {
    'base': {42: 'aub_b5e6_base_seed42',
             189: 'cc_base5e6_seed189_seed189',
             7: 'cc_base5e6_seed7'},
    'res': {42: 'aub_b5e6_res_seed42_seed42',
            189: 'aub_b5e6_res_seed189_seed189',
            7: 'aub_b5e6_res_seed7_seed7'},
    'ctrl': {42: 'aub_b5e6_ctrl_seed42_seed42',
             189: 'aub_b5e6_ctrl_seed189_seed189',
             7: 'aub_b5e6_ctrl_seed7_seed7'},
}
SEEDS = [42, 189, 7]

data = {}
for a, seeds in ARMS.items():
    data[a] = {}
    for s, d in seeds.items():
        r = load(d)
        if r is None:
            print(f'MISSING {a} seed{s} -> {d}')
        else:
            data[a][s] = r

for metric in METRICS:
    print(f'\n===== {metric} =====')
    hdr = ' '.join(f'{c:>9s}' for c in CORPORA)
    print(f"{'arm':6s} {'seed':>5s} {hdr}")
    for a in ARMS:
        for s in SEEDS:
            if s in data[a]:
                row = ' '.join(f'{data[a][s][c][metric]:9.4f}' for c in CORPORA)
                print(f'{a:6s} {s:>5d} {row}')
        if len(data[a]) == len(SEEDS):
            mean = ' '.join(
                f'{np.mean([data[a][s][c][metric] for s in SEEDS]):9.4f}'
                for c in CORPORA)
            print(f'{a:6s} {"MEAN":>5s} {mean}')
        print()

print('\n===== paired differences (per seed, 3 seeds) =====')
for x, y in [('res', 'base'), ('ctrl', 'base'), ('res', 'ctrl')]:
    if len(data[x]) < 3 or len(data[y]) < 3:
        continue
    print(f'\n--- {x} minus {y} ---')
    for metric in ['uar', 'neutral_auc', 'f1_weighted']:
        parts = []
        for c in CORPORA:
            d = np.array([data[x][s][c][metric] - data[y][s][c][metric]
                          for s in SEEDS])
            wins = int((d > 0).sum())
            sd = d.std(ddof=1)
            # Paired t against zero; with n=3 this is weak by construction and
            # is reported as a spread indicator, not as a significance claim.
            t = d.mean() / (sd / np.sqrt(3)) if sd > 0 else float('inf')
            parts.append(f'{c}: {d.mean():+.4f} ({wins}/3, sd={sd:.4f}, t={t:+.2f})')
        print(f'  {metric:12s} ' + '  '.join(parts))
