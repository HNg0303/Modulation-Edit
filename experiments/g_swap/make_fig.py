"""Assemble Figure 11 (main) and a controls figure from g_swap.py outputs."""
import os, sys, json
from PIL import Image, ImageDraw, ImageFont
D = sys.argv[1] if len(sys.argv) > 1 else 'output/g_swap/e0'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
meta = json.load(open(os.path.join(D, 'meta.json')))
seeds = [int(s) for s in meta['seeds'].split(',')]
ts = [float(t) for t in meta['ts'].split(',')]
ks = [int(k) for k in meta['early'].split(',')]
c1, c2 = meta['c1'], meta['c2']
S = 256
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
except Exception: font = ImageFont.load_default()

def grid(rows, cols, cell, out):
    H = 28
    W = Image.new('RGB', (S * len(cols), H + S * len(rows)), 'white'); d = ImageDraw.Draw(W)
    for j, c in enumerate(cols): d.text((S * j + 6, 6), c, fill='black', font=font)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            f = os.path.join(D, cell(r, c))
            if os.path.exists(f): W.paste(Image.open(f).resize((S, S), Image.LANCZOS), (S * j, H + S * i))
    W.save(os.path.join(D, out), quality=92); print('wrote', out)

# Fig 11: neutral prompt everywhere; only G changes
cols = ['prompt only (G_n)', f'G_{c1} all scales', f'G_{c2} all scales'] + [f'early{K} t={t:g}' for t in ts]
def cell(s, c):
    if c.startswith('prompt only'): return f's{s}_ref_neutral.jpg'
    if c.startswith(f'G_{c1}'): return f's{s}_Gall_c1.jpg'
    if c.startswith(f'G_{c2}'): return f's{s}_Gall_c2.jpg'
    return f's{s}_early{K}_t{float(c.split("=")[1]):.2f}.jpg'
grid(seeds, cols, cell, f'fig11_early{K}.jpg')

# |S| sweep for one seed
cols = [f't={t:g}' for t in ts]
grid(ks, cols, lambda k, c: f's{seeds[0]}_early{k}_t{float(c[2:]):.2f}.jpg', f'fig_S_sweep_seed{seeds[0]}.jpg')

# controls
alphas = [float(a) for a in meta['alphas'].split(',') if a]
cols = [f'full "{c1}"', f'full "{c2}"', f'"{c1}" w/ G_n', f'"{c2}" w/ G_n'] + \
       [f'G amp x{a:g} {c}' for a in alphas for c in (c1, c2)]
files = [f'ref_c1', 'ref_c2', 'promptc1_Gn', 'promptc2_Gn'] + [f'Gamp{a:g}_{t}' for a in alphas for t in ('c1', 'c2')]
m = dict(zip(cols, files))
grid(seeds, cols, lambda s, c: f's{s}_{m[c]}.jpg', 'fig_controls.jpg')
