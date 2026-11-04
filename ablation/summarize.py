"""Collect <part_dir>/<variant>/metrics.csv ([overall] / [live] / [object] rows) into <part_dir>/summary.{csv,md}."""
import csv
import json
import os
import os.path as osp
import sys

METRICS = ['dino', 'clip_i', 'clip_t', 'clip_t_cls', 'div']

part = sys.argv[1]
rows = []
for name in sorted(os.listdir(part)):
    path = osp.join(part, name, 'metrics.csv')
    if not osp.isfile(path):
        continue
    with open(osp.join(part, name, 'config.json')) as f:
        v = json.load(f)['variant']
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['subject'].startswith('['):
                group = r['subject'][1:r['subject'].index(']')]
                n_sub = r['subject'].split('(')[1].split()[0]
                rows.append(dict(variant=name, group=group, subjects=n_sub, alpha=v['alpha'], blend=v['blend'], ca=v['ca'],
                                 offset=v['offset'], offset_scales=v['offset_scales'], cfg=v['cfg'],
                                 **{m: f'{float(r[m]):.4f}' if r[m] else '' for m in METRICS}))
if not rows:
    sys.exit(f'no metrics.csv under {part}')
with open(osp.join(part, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
cols = ['variant', 'subjects', 'alpha', 'blend', 'ca', 'offset', 'offset_scales', 'cfg'] + METRICS
lines = []
for group in ('overall', 'live', 'object'):
    g = [r for r in rows if r['group'] == group]
    if g:
        lines += [f'### {group}', '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
        lines += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in g] + ['']
open(osp.join(part, 'summary.md'), 'w').write('\n'.join(lines))
print('\n'.join(lines))
