"""Real-haze test on RRSHID (Zhu et al., IEEE TGRS 2025): real hazy / haze-free satellite image pairs,
thin / moderate / thick subsets.

There are no VQA labels for RRSHID, so we ask presence questions and use the VLM's own answer on the
HAZE-FREE image as the reference (answer consistency). Only confident reference answers are kept at
evaluation time (transfer_eval.py --pseudo).

Writes a folder in the same format as make_versions.py:
  {key}_clean_{act}.png  <- haze-free image (+ enhancers)
  {key}_haze_{act}.png   <- REAL hazy image (+ enhancers)
  level = 1 thin, 2 moderate, 3 thick

  python prepare_rrshid.py --root /kaggle/working/rrshid_raw --out rrshid_v --per_density 50
"""
import argparse, json, os, random, re
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ACTIONS, enhance, image_stats, drift_stats

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
HAZY = re.compile(r"(^|[/_\-])(hazy|haze|input|degraded|lq|low)([/_\-.]|$)", re.I)
CLEAR = re.compile(r"(^|[/_\-])(gt|clear|clean|target|haze[_\-]?free|dehazed|hq|label|groundtruth)([/_\-.]|$)", re.I)
DENS = [("thin", 1), ("moderate", 2), ("thick", 3)]
QUESTIONS = [  # (noun phrase, qtype)
    ("buildings", "presence_object"), ("trees", "presence_surface"), ("water", "presence_surface"),
    ("a road", "presence_object"), ("farmland", "presence_surface")]


def density(path):
    p = path.lower()
    for name, lv in DENS:
        if name in p:
            return lv
    return 0


def find_pairs(root):
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(EXT)]
    print(f"{len(files)} image files under {root}")
    groups = defaultdict(dict)
    for f in files:
        rel = os.path.relpath(f, root)
        role = "clear" if CLEAR.search(rel) else ("hazy" if HAZY.search(rel) else None)
        if role is None:
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        stem = re.sub(r"[_\-]?(hazy|haze|gt|clear|clean)$", "", stem, flags=re.I)
        lv = density(rel)
        split = "test" if "test" in rel.lower() else ("train" if "train" in rel.lower() else "other")
        groups[(lv, split, stem)][role] = f
    pairs = [(k, v["hazy"], v["clear"]) for k, v in groups.items() if "hazy" in v and "clear" in v]
    print(f"{len(pairs)} hazy/clear pairs found")
    for lv in (1, 2, 3, 0):
        for sp in ("test", "train", "other"):
            n = sum(1 for (l, s, _), _, _ in pairs if l == lv and s == sp)
            if n:
                print(f"  level {lv} split {sp}: {n}")
    if pairs:
        print("example pair:", pairs[0][1], "|", pairs[0][2])
    return pairs


def load_rgb(p, side):
    im = Image.open(p).convert("RGB")
    return np.array(im.resize((side, side), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="rrshid_v")
    ap.add_argument("--per_density", type=int, default=50)
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pairs = find_pairs(a.root)
    if not pairs:
        raise SystemExit("No pairs found: send the printed folder listing so the matcher can be adjusted.")

    chosen = []
    for lv in (1, 2, 3):
        pool = [p for p in pairs if p[0][0] == lv]
        pref = [p for p in pool if p[0][1] == "test"] or pool  # prefer the official test split
        chosen += rng.sample(sorted(pref), min(a.per_density, len(pref)))
    if not chosen:  # no density in the paths: sample from everything
        chosen = rng.sample(sorted(pairs), min(3 * a.per_density, len(pairs)))
    print(f"using {len(chosen)} pairs")

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    qs, feats = [], []
    for i, ((lv, sp, stem), hz, cl) in enumerate(tqdm(chosen, desc="versions")):
        key = f"rr{i:04d}"
        clear, hazy = load_rgb(cl, a.side), load_rgb(hz, a.side)
        for cond, img, level in (("clean", clear, 0), ("haze", hazy, lv)):
            base = image_stats(img)
            for act in ACTIONS:
                out = enhance(img, act)
                fn = f"{key}_{cond}_{act}.png"
                Image.fromarray(out).save(os.path.join(a.out, "images", fn))
                feats.append(dict(img_key=key, cond=cond, action=act, file=fn, level=level, source=stem,
                                  **{f"in_{k}": v for k, v in base.items()},
                                  **{f"drift_{k}": v for k, v in drift_stats(img, out).items()},
                                  **{f"vsclean_{k}": v for k, v in drift_stats(clear, out).items()}))
        Image.fromarray(np.full_like(clear, 128)).save(os.path.join(a.out, "images", f"{key}_blank.png"))
        for j, (noun, qt) in enumerate(QUESTIONS):
            qs.append(dict(qid=f"{key}_q{j}", img_key=key, question=f"Is there {noun} in this image?",
                           answer="yes",  # placeholder: the reference is the VLM answer on the haze-free image
                           qtype=qt, qcat=qt.split("_")[1], level=lv))
    with open(os.path.join(a.out, "questions.jsonl"), "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in qs)
    with open(os.path.join(a.out, "features.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in feats)
    print(f"{len(chosen)} images, {len(qs)} questions -> {len(qs) * 9} VLM runs needed")


if __name__ == "__main__":
    main()
