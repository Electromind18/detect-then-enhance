"""Phase 0: sample a pilot subset, create 3 conditions x 4 actions = 12 versions per image ONCE,
and save cheap gate features. No GPU needed.

Input, either:
  --rsvqa_dir  folder with RSVQA-LR (Zenodo): LR_split_{split}_{questions,answers}.json + Images_LR/
  --csv        generic CSV with columns: image_path,question,answer[,qtype]   (for HR / DisasterM3 later)
"""
import argparse, json, os, random, zlib
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import CONDITIONS, ACTIONS, degrade, enhance, image_stats, drift_stats, question_category

YESNO = {"yes", "no"}
MCQ = {"a", "b", "c", "d"}


def load_rsvqa(d, split):
    def rows(name):
        with open(os.path.join(d, f"LR_split_{split}_{name}.json")) as f:
            obj = json.load(f)
        return [r for r in obj[name] if r.get("active", True)]
    ans = {a["question_id"]: a["answer"] for a in rows("answers")}
    out = []
    for q in rows("questions"):
        a = str(ans.get(q["id"], "")).strip().lower()
        out.append(dict(qid=f"lr{q['id']}", img_key=f"lr{q['img_id']}",
                        image_path=os.path.join(d, "Images_LR", f"{q['img_id']}.tif"),
                        question=q["question"], answer=a, qtype=q["type"]))
    return out


def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    out = []
    for i, r in df.iterrows():
        out.append(dict(qid=f"c{i}", img_key=os.path.splitext(os.path.basename(r.image_path))[0],
                        image_path=r.image_path, question=r.question,
                        answer=str(r.answer).strip().lower(),
                        qtype=r.qtype if "qtype" in df.columns else question_category(r.question),
                        **({"qcat": r.qcat} if "qcat" in df.columns else {})))
    return out


def sample(qs, n_images, max_q_per_img, seed, skip=0):
    rng = random.Random(seed)
    by_img = defaultdict(list)
    for q in qs:
        if q["answer"] in YESNO or q["answer"] in MCQ:   # 1 forward pass, no generation
            by_img[q["img_key"]].append(q)
    keys = sorted(by_img)
    rng.shuffle(keys)
    picked = []
    for k in keys[skip:skip + n_images]:   # skip>0 -> images never used before (prefix is seed-stable)
        qq = by_img[k]
        rng.shuffle(qq)
        # round-robin over question types so each image contributes diverse types
        by_t = defaultdict(list)
        for q in qq:
            by_t[q["qtype"]].append(q)
        sel = []
        while len(sel) < max_q_per_img and any(by_t.values()):
            for t in list(by_t):
                if by_t[t] and len(sel) < max_q_per_img:
                    sel.append(by_t[t].pop())
        picked += sel
    return picked


def load_rgb(p, max_side):
    im = Image.open(p).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.BICUBIC)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsvqa_dir"); ap.add_argument("--csv"); ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="pilot")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--q_per_img", type=int, default=3)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", action="store_true", help="random severity 1-3 per image/condition")
    ap.add_argument("--fixed_level", type=int, default=0, help="force one severity for all degraded images (4 = unseen)")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N sampled images (use fresh images)")
    a = ap.parse_args()

    qs = load_rsvqa(a.rsvqa_dir, a.split) if a.rsvqa_dir else load_csv(a.csv)
    qs = sample(qs, a.n_images, a.q_per_img, a.seed, a.skip)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    imgs = sorted({q["img_key"]: q["image_path"] for q in qs}.items())
    feats = []
    for key, path in tqdm(imgs, desc="versions"):
        src = load_rgb(path, a.max_side)
        for ci, cond in enumerate(CONDITIONS):
            rng = np.random.default_rng(zlib.crc32(f"{a.seed}|{key}|{cond}".encode()))  # reproducible
            level = int(rng.integers(1, 4)) if (a.graded and cond != "clean") else 0
            if a.fixed_level and cond != "clean":
                level = a.fixed_level
            inp = degrade(src, cond, rng, level or None)
            base = image_stats(inp)
            for act in ACTIONS:
                out = enhance(inp, act)
                fn = f"{key}_{cond}_{act}.png"
                Image.fromarray(out).save(os.path.join(a.out, "images", fn))
                # fidelity to the ORIGINAL clean image (analysis only, NOT a gate feature)
                fid = drift_stats(src, out)
                feats.append(dict(img_key=key, cond=cond, action=act, file=fn, level=level,
                                  **{f"in_{k}": v for k, v in base.items()},
                                  **{f"drift_{k}": v for k, v in drift_stats(inp, out).items()},
                                  **{f"vsclean_{k}": v for k, v in fid.items()}))
        # text-only control: mid-grey image of same size
        Image.fromarray(np.full_like(src, 128)).save(os.path.join(a.out, "images", f"{key}_blank.png"))

    for q in qs:
        q.setdefault("qcat", question_category(q["question"]))
    with open(os.path.join(a.out, "questions.jsonl"), "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    with open(os.path.join(a.out, "features.jsonl"), "w") as f:
        for r in feats:
            f.write(json.dumps(r) + "\n")
    n_runs = len(qs) * (len(CONDITIONS) * len(ACTIONS) + 1)
    print(f"{len(imgs)} images, {len(qs)} questions -> {n_runs} VLM runs needed")


if __name__ == "__main__":
    main()
