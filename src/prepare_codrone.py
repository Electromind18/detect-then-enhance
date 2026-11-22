"""Real low-light test on CODrone (Ye et al., 2025, arXiv:2504.20032): UAV images with oriented-box labels,
captured under various lighting conditions including night. CODrone does not tag night images in its file
layout, so night images are selected by low mean luminance and checked visually (contact sheet saved);
daytime images from the same split are the 'clean' reference.

Questions are presence questions answered from the REAL box labels ("Is there a <class> in this image?"),
so no reference image is needed.

  python prepare_codrone.py --root /kaggle/tmp/codrone_raw --out codrone_v --n_night 150 --n_day 100
"""
import argparse, glob, json, os, random
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ACTIONS, enhance, image_stats, drift_stats

EXT = (".jpg", ".jpeg", ".png", ".tif", ".bmp")


def read_dota(p):
    cls = set()
    for line in open(p, errors="ignore"):
        t = line.split()
        if len(t) >= 9 and not line.startswith(("imagesource", "gsd")):
            try:
                float(t[0])
            except ValueError:
                continue
            cls.add(t[8].lower())
    return cls


def load_small(p, side):
    im = Image.open(p).convert("RGB")
    im.thumbnail((side, side), Image.BICUBIC)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="codrone_v")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_night", type=int, default=150)
    ap.add_argument("--n_day", type=int, default=100)
    ap.add_argument("--night_lum", type=float, default=0.20, help="mean luminance below this = night candidate")
    ap.add_argument("--day_lum", type=float, default=0.35)
    ap.add_argument("--side", type=int, default=768, help="longest side of saved images")
    ap.add_argument("--q_per_img", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    ann = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(a.root, "**", a.split, "annfile", "*.txt"), recursive=True)}
    imgs = [p for p in glob.glob(os.path.join(a.root, "**", a.split, "images", "*"), recursive=True)
            if p.lower().endswith(EXT) and os.path.splitext(os.path.basename(p))[0] in ann]
    print(f"split {a.split}: {len(imgs)} images with annotations")
    if not imgs:
        raise SystemExit("no images found: send the folder listing")

    rows = []
    for p in tqdm(sorted(imgs), desc="luminance"):
        im = Image.open(p).convert("L")
        im.thumbnail((256, 256))
        rows.append((p, float(np.asarray(im).mean() / 255), read_dota(ann[os.path.splitext(os.path.basename(p))[0]])))
    lum = np.array([r[1] for r in rows])
    print("luminance quantiles 1/5/25/50/75%:", np.round(np.quantile(lum, [.01, .05, .25, .5, .75]), 3))
    allcls = Counter(c for r in rows for c in r[2])
    print("classes (images containing):", dict(allcls.most_common()))
    classes = [c for c, n in allcls.items() if n >= 10 and c != "ignored"]  # "ignored" is a mask label, not an object
    print("question classes:", sorted(classes))

    tag = lambda r: os.path.basename(r[0]).lower().replace(' ', '')
    night = [r for r in rows if tag(r).startswith('night')]   # CODrone file names carry day_/night_ tags
    day = [r for r in rows if tag(r).startswith('day')]
    print('by file name -> night:', len(night), '| day:', len(day), '| night lum median:',
          round(float(np.median([r[1] for r in night])), 3) if night else None)
    print(f"night candidates (lum<{a.night_lum}): {len(night)} | day candidates (lum>{a.day_lum}): {len(day)}")
    night = rng.sample(night, min(a.n_night, len(night)))
    day = rng.sample(day, min(a.n_day, len(day)))

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    qs, feats, sheet = [], [], []
    for i, ((p, l, have), cond) in enumerate(tqdm([(r, "lowlight") for r in night] + [(r, "clean") for r in day],
                                                 desc="versions")):
        key = f"cd{i:04d}"
        img = load_small(p, a.side)
        base = image_stats(img)
        for act in ACTIONS:
            out = enhance(img, act)
            fn = f"{key}_{cond}_{act}.png"
            Image.fromarray(out).save(os.path.join(a.out, "images", fn))
            feats.append(dict(img_key=key, cond=cond, action=act, file=fn, level=int(cond == "lowlight"),
                              source=os.path.basename(p), lum_raw=l,
                              **{f"in_{k}": v for k, v in base.items()},
                              **{f"drift_{k}": v for k, v in drift_stats(img, out).items()}))
        Image.fromarray(np.full_like(img, 128)).save(os.path.join(a.out, "images", f"{key}_blank.png"))
        if cond == "lowlight" and len(sheet) < 24:
            sheet.append(Image.fromarray(img).resize((256, 144)))
        pos, neg = sorted(have & set(classes)), sorted(set(classes) - have)
        rng.shuffle(pos); rng.shuffle(neg)
        k = a.q_per_img // 2
        pick = [(c, "yes") for c in pos[:k]] + [(c, "no") for c in neg[:a.q_per_img - min(k, len(pos))]]
        for j, (c, ans) in enumerate(pick):
            noun = {"motor": "motorcycle", "people": "person"}.get(c, c.replace("-", " ").replace("_", " "))
            qs.append(dict(qid=f"{key}_q{j}", img_key=key, question=f"Is there a {noun} in this image?",
                           answer=ans, qtype="presence_object", qcat="object", target=c))
    if sheet:  # visual check that the selected images really are night scenes
        W = Image.new("RGB", (256 * 6, 144 * ((len(sheet) + 5) // 6)))
        for n, s in enumerate(sheet):
            W.paste(s, ((n % 6) * 256, (n // 6) * 144))
        W.save(os.path.join(a.out, "night_contact_sheet.jpg"), quality=85)
    with open(os.path.join(a.out, "questions.jsonl"), "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in qs)
    with open(os.path.join(a.out, "features.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in feats)
    yes = sum(q["answer"] == "yes" for q in qs)
    print(f"{len(night)} night + {len(day)} day images, {len(qs)} questions ({yes} yes) -> {len(qs) * 5} VLM runs")


if __name__ == "__main__":
    main()
