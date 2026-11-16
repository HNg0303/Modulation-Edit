"""E1: image grid + CLIP attribute score P(attr text | {base text, attr text})."""
import os, sys, json, torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor
D = sys.argv[1]
CLIP = os.environ.get('CLIP_PATH', 'weights/clip-vit-large-patch14')
meta = json.load(open(f'{D}/meta.json'))
attrs, base = meta['attrs'].split('|'), meta['base']
seeds = [int(s) for s in meta['seeds'].split(',')]
alphas = [float(a) for a in meta['alphas'].split(',')]
conds = ['ref', 'textroute'] + [f'G{a:g}' for a in alphas]
labels = {'ref': 'full attr prompt', 'textroute': 'attr via SOS+CA (G=base)', **{f'G{a:g}': f'attr via G only a={a:g}' for a in alphas}}

m = CLIPModel.from_pretrained(CLIP).cuda().eval(); proc = CLIPProcessor.from_pretrained(CLIP)
@torch.no_grad()
def p_attr(img_path, i):
    x = proc(text=[base, attrs[i]], images=Image.open(img_path), return_tensors='pt', padding=True).to('cuda')
    return m(**x).logits_per_image.softmax(-1)[0, 1].item()

res = {}
for i, A in enumerate(attrs):
    res[A] = {'base (no attr)': sum(p_attr(f'{D}/s{s}_base.jpg', i) for s in seeds) / len(seeds)}
    for c in conds:
        res[A][labels[c]] = sum(p_attr(f'{D}/s{s}_a{i}_{c}.jpg', i) for s in seeds) / len(seeds)
json.dump(res, open(f'{D}/clip_scores.json', 'w'), indent=1)
hdr = ['base'] + [labels[c] for c in conds]
print('| attribute | ' + ' | '.join(hdr) + ' |\n|' + '---|' * (len(hdr) + 1))
for A, r in res.items(): print(f'| {A} | ' + ' | '.join(f'{v:.2f}' for v in r.values()) + ' |')

S = 224
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12) if os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') else ImageFont.load_default()
for s in seeds:
    cols = ['base'] + conds
    W = Image.new('RGB', (S * len(cols) + 190, 24 + S * len(attrs)), 'white'); d = ImageDraw.Draw(W)
    for j, c in enumerate(cols): d.text((190 + S * j + 4, 5), 'base "cat"' if c == 'base' else labels[c], fill='black', font=font)
    for i, A in enumerate(attrs):
        d.text((4, 24 + S * i + S // 2), A.replace('a photo of ', ''), fill='black', font=font)
        for j, c in enumerate(cols):
            f = f'{D}/s{s}_base.jpg' if c == 'base' else f'{D}/s{s}_a{i}_{c}.jpg'
            W.paste(Image.open(f).resize((S, S), Image.LANCZOS), (190 + S * j, 24 + S * i))
    W.save(f'{D}/e1_grid_seed{s}.jpg', quality=92)
