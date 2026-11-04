"""
Rebuild <variant>/metrics.csv from the per-subject <subject>/<subject>_metrics.csv files (same averaging as
CSD-VAR/evaluate_personalization.py: prompts -> subject -> overall / live / object). Used after re-evaluating
only some subjects.

    python ablation/reaggregate.py <variant_dir> [<variant_dir> ...]
"""
import csv
import os.path as osp
import sys
from statistics import mean

VICO_DIR = '/workspace/CSD-VAR/benchmark_data/vico_csd'
METRICS = ['dino', 'clip_i', 'clip_t', 'clip_t_cls', 'div', 'pres', 'csd_content', 'csd_style']


def reaggregate(vdir, data_dir=VICO_DIR):
    with open(osp.join(data_dir, 'subjects.csv')) as f:
        subjects = list(csv.DictReader(f))
    rows = []
    for s in subjects:
        path = osp.join(vdir, s['subject'], f"{s['subject']}_metrics.csv")
        if not osp.exists(path):
            continue
        with open(path) as f:
            per_prompt = list(csv.DictReader(f))
        row = dict(subject=s['subject'], class_name=s['class_name'], prompts=s['prompts'])
        for k in METRICS:
            vals = [float(r[k]) for r in per_prompt if r.get(k, '') != '']
            row[k] = mean(vals) if vals else ''
        rows.append(row)
    groups = dict(overall=rows, live=[r for r in rows if r['prompts'].startswith('prompts_live')],
                  object=[r for r in rows if r['prompts'].startswith('prompts_object')])
    summary = []
    for name, g in groups.items():
        if not g:
            continue
        avg = dict(subject=f'[{name}] ({len(g)} subjects)', class_name='', prompts='')
        for k in METRICS:
            vals = [r[k] for r in g if r[k] != '']
            avg[k] = mean(vals) if vals else ''
        summary.append(avg)
    with open(osp.join(vdir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject', 'class_name', 'prompts'] + METRICS)
        w.writeheader()
        w.writerows(rows + summary)
    print(vdir, summary[0])


if __name__ == '__main__':
    for v in sys.argv[1:]:
        reaggregate(v)
