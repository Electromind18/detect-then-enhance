"""Cross-dataset test: AID (Xia et al., 2017), same three question types as UC Merced.
  scene_mcq        : from blanchon/AID (30 classes, 600x600), N images per class
  presence_surface / presence_object : from jonathan-roberts1/AID_MultiLabel (same 17 labels as UCM-ML)
Writes aid.csv for make_versions.py --csv.  Images are saved at 512x512 (VLM input is capped at 512^2 anyway).

  python prepare_aid.py --out /kaggle/working/aid --per_class 20 --n_ml 400
"""
import argparse, os, random
from collections import defaultdict
import pandas as pd
from datasets import load_dataset
from prepare_ucm_ml import ML, SURFACE, OBJECT, NOUN

LETTERS = "ABCD"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def fetch(fn, tries=8):
    import time
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" not in str(e) or k == tries - 1:
                raise
            wait = 30 * (k + 1)
            print(f"rate limited, waiting {wait}s ...", flush=True)
            time.sleep(wait)


def pretty(name):
    n = name.replace("_", " ").replace("-", " ").strip().lower()
    fixes = {"bareland": "bare land", "baseballfield": "baseball field", "denseresidential": "dense residential",
             "mediumresidential": "medium residential", "sparseresidential": "sparse residential",
             "playground": "playground", "railwaystation": "railway station", "storagetanks": "storage tanks",
             "parking": "parking lot"}
    return fixes.get(n.replace(" ", ""), n)


def first_key(feats, kinds):
    for k, v in feats.items():
        if type(v).__name__ in kinds:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="aid")
    ap.add_argument("--per_class", type=int, default=20)
    ap.add_argument("--n_ml", type=int, default=400, help="multi-label images for presence questions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rows = []

    # ---------- scene MCQ from full AID ----------
    # Download only the files we need (per_class x 30), one by one with retry,
    # instead of all 10,000 files (which triggers HF 429 rate limits).
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    files = [f for f in HfApi().list_repo_files("blanchon/AID", repo_type="dataset")
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]
    by_c = defaultdict(list)
    for f in files:
        if f.count("/") >= 2: by_c[f.split("/")[-2]].append(f)
    names = {c: pretty(c) for c in by_c}
    print(len(by_c), "scene classes:", sorted(names.values()))
    allnames = sorted(names.values())
    for c in sorted(by_c):
        for f in rng.sample(sorted(by_c[c]), min(a.per_class, len(by_c[c]))):
            local = fetch(lambda: hf_hub_download("blanchon/AID", f, repo_type="dataset"))
            path = os.path.join(a.out, "images", "aid_" + os.path.basename(f).rsplit(".", 1)[0] + ".jpg")
            Image.open(local).convert("RGB").resize((512, 512)).save(path, quality=95)
            true = names[c]
            opts = rng.sample([n for n in allnames if n != true], 3) + [true]
            rng.shuffle(opts)
            q = ("Which land-use class best describes this aerial image?\n"
                 + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, opts)))
            rows.append(dict(image_path=path, question=q, answer=LETTERS[opts.index(true)],
                             qtype="scene_mcq", qcat="scene", target=true))
    print("scene questions:", len(rows))

    # ---------- presence questions from AID multi-label ----------
    ml = fetch(lambda: load_dataset("jonathan-roberts1/AID_MultiLabel", split="train"))
    print("AID-ML features:", ml.features)
    ik = first_key(ml.features, {"Image"})
    lk = [k for k in ml.features if k != ik][0]
    try:
        lnames = [n.lower().replace(" ", "-") for n in ml.features[lk].feature.names]
    except Exception:
        lnames = ML
    print("labels:", lnames)
    order = list(range(len(ml)))
    rng.shuffle(order)
    for i in order[:a.n_ml]:
        ex = ml[i]
        have = {lnames[j] for j in ex[lk]}
        path = os.path.join(a.out, "images", f"aidml{i}.jpg")
        ex[ik].convert("RGB").resize((512, 512)).save(path, quality=95)
        for kind, pool in [("surface", SURFACE), ("object", OBJECT)]:
            pos, neg = sorted(have & pool), sorted(pool - have)
            if rng.random() < 0.5 and pos:
                t, ans = rng.choice(pos), "yes"
            elif neg:
                t, ans = rng.choice(neg), "no"
            else:
                continue
            rows.append(dict(image_path=path, question=f"Is there {NOUN[t]} in this image?", answer=ans,
                             qtype=f"presence_{kind}", qcat=kind, target=t))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "aid.csv"), index=False)
    print(df.groupby(["qtype", "answer"]).size())
    print(f"{len(df)} questions on {df.image_path.nunique()} images -> {a.out}/aid.csv")


if __name__ == "__main__":
    main()
