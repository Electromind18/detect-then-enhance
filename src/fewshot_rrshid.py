"""Few-shot detector recalibration for real haze: image statistics of k RRSHID TRAIN pairs per density
(hazy -> 'haze', haze-free -> 'clean'). Only the degradation detector sees them; the VLM, the lookup table and
the TEST images used for evaluation are untouched. No GPU needed.

  python fewshot_rrshid.py --root /kaggle/working/rrshid_raw --k 10 --out fewshot_k10.jsonl
"""
import argparse, json, random
from common import image_stats
from prepare_rrshid import find_pairs, load_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--k", type=int, default=10, help="pairs per haze density")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
pairs = [p for p in find_pairs(a.root) if p[0][1] == "train"]
rows = []
for lv in (1, 2, 3):
    for (_, _, stem), hz, cl in rng.sample(sorted(p for p in pairs if p[0][0] == lv), a.k):
        for cond, path in (("haze", hz), ("clean", cl)):
            rows.append(dict(cond=cond, level=lv, source=stem,
                             **{f"in_{k}": v for k, v in image_stats(load_rgb(path, 512)).items()}))
with open(a.out, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(f"{len(rows)} labelled inputs ({3 * a.k} train pairs) -> {a.out}")
