"""Phase 1 data: UC Merced multi-label -> 3 question types per image (needed for the question-aware claim)
  1. scene_mcq         : 4-option land-use class      (evidence: layout / texture)
  2. presence_surface  : "Is there water/grass/...?"  (evidence: colour / surface)
  3. presence_object   : "Is there a car/ship/...?"   (evidence: small objects / edges)
Presence questions are balanced yes/no. Writes ucm_ml.csv for make_versions.py --csv.

  python prepare_ucm_ml.py --out /kaggle/working/ucm_ml
"""
import argparse, os, random
import pandas as pd
from datasets import load_dataset

# 17 multi-labels (alphabetical, BigEarth release)
ML = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock", "field", "grass",
      "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"]
SURFACE = {"bare-soil", "chaparral", "field", "grass", "pavement", "sand", "sea", "trees", "water"}
OBJECT = {"airplane", "buildings", "cars", "court", "dock", "mobile-home", "ship", "tanks"}
NOUN = {"airplane": "an airplane", "bare-soil": "bare soil", "buildings": "a building", "cars": "a car",
        "chaparral": "chaparral (shrubland)", "court": "a sports court", "dock": "a dock", "field": "a crop field",
        "grass": "grass", "mobile-home": "a mobile home", "pavement": "pavement", "sand": "sand",
        "sea": "sea", "ship": "a ship", "tanks": "a storage tank", "trees": "trees", "water": "water"}
# 21 scene classes in the original alphabetical file order (100 images each)
SCENES = ["agricultural", "airplane", "baseball diamond", "beach", "buildings", "chaparral",
          "dense residential", "forest", "freeway", "golf course", "harbor", "intersection",
          "medium residential", "mobile home park", "overpass", "parking lot", "river", "runway",
          "sparse residential", "storage tanks", "tennis court"]
LETTERS = "ABCD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ucm_ml")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = load_dataset("jonathan-roberts1/UC_Merced_LandUse_MultiLabel", split="train")
    try:
        names = [n.lower().replace(" ", "-") for n in ds.features["label"].feature.names]
    except Exception:
        names = ML
    print("label names:", names)
    labels = [[names[i] for i in ex] for ex in ds["label"]]

    # Sanity check: images are in scene-class order (100 per class) -> scene label = index // 100
    def frac(block, lab):
        return sum(lab in labels[i] for i in range(block * 100, block * 100 + 100)) / 100
    checks = {"airplane block has airplane": frac(1, "airplane"),
              "harbor block has ship/dock": max(frac(10, "ship"), frac(10, "dock")),
              "forest block has trees": frac(7, "trees")}
    print("order checks (should all be > 0.8):", checks)
    scene_ok = len(ds) == 2100 and min(checks.values()) > 0.8
    if not scene_ok:
        print("WARNING: order check failed -> scene MCQ questions skipped, presence questions only")

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rng = random.Random(a.seed)
    rows = []
    for i, ex in enumerate(ds):
        path = os.path.join(a.out, "images", f"ucm{i}.png")
        ex["image"].convert("RGB").save(path)
        have = set(labels[i])
        base = dict(image_path=path, labels="|".join(sorted(have)))
        if scene_ok:
            true = SCENES[i // 100]
            opts = rng.sample([s for s in SCENES if s != true], 3) + [true]
            rng.shuffle(opts)
            q = ("Which land-use class best describes this aerial image?\n"
                 + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, opts)))
            rows.append(dict(base, question=q, answer=LETTERS[opts.index(true)], qtype="scene_mcq",
                             qcat="scene", target=true))
        for kind, pool in [("surface", SURFACE), ("object", OBJECT)]:
            pos, neg = sorted(have & pool), sorted(pool - have)
            want_yes = rng.random() < 0.5
            if want_yes and pos:
                t, ans = rng.choice(pos), "yes"
            elif neg:
                t, ans = rng.choice(neg), "no"
            else:
                continue
            rows.append(dict(base, question=f"Is there {NOUN[t]} in this image?", answer=ans,
                             qtype=f"presence_{kind}", qcat=kind, target=t))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "ucm_ml.csv"), index=False)
    print(df.groupby(["qtype", "answer"]).size())
    print(f"{len(df)} questions on {len(ds)} images -> {a.out}/ucm_ml.csv")


if __name__ == "__main__":
    main()
