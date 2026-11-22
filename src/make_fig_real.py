"""Real-data figure: two RRSHID real-haze pairs and two CODrone real night images, shown exactly as the VLM
saw them (same resizing as prepare_rrshid.py / prepare_codrone.py), with the real Qwen2.5-VL-3B answers.
RRSHID rows: haze-free twin (reference answer), real hazy input, dehazed (the router's choice after the
few-shot detector update). CODrone rows: real night input, gamma (the router's choice), CLAHE (for comparison).
Examples were chosen to illustrate one success per dataset and one failure on night imagery; not a random sample.

Kaggle, CPU, internet on:
  python make_fig_real.py --rrshid /kaggle/working/rrshid_raw --codrone /tmp/codrone_raw --out fig_real
"""
import argparse, glob, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import enhance, image_stats

# (tile title, condition/action key, P(yes) copied from results_Qwen2.5-VL-3B-Instruct_*.jsonl, stored lum_mean)
ROWS = [
    dict(ds="rrshid", density="moderate", src="360", q="Is there buildings in this image?", ref="Yes", tiles=[
        ("Haze-free twin (reference)", "clean|none", .88, .1549),
        ("Real haze (moderate)", "haze|none", .13, .5829),
        ("Dehazed (router)", "haze|dehaze", .87, .5829)]),
    dict(ds="rrshid", density="thick", src="435", q="Is there farmland in this image?", ref="Yes", tiles=[
        ("Haze-free twin (reference)", "clean|none", .87, .2664),
        ("Real haze (thick)", "haze|none", .12, .8132),
        ("Dehazed (router)", "haze|dehaze", .82, .8132)]),
    dict(ds="codrone", src="night_hangzhou_100m_30c_2_frame_3500.jpg", q="Is there a bus in this image?", ref="Yes", tiles=[
        ("Real night", "lowlight|none", .38, .2723),
        ("Gamma (router)", "lowlight|gamma", .78, .2723),
        ("CLAHE", "lowlight|clahe", .25, .2723)]),
    dict(ds="codrone", src="night _yangfanpark_60m_90c_3_frame_1800.jpg", q="Is there a person in this image?", ref="Yes",
         tiles=[("Real night", "lowlight|none", .53, .1910),
                ("Gamma (router)", "lowlight|gamma", .44, .1910),
                ("CLAHE", "lowlight|clahe", .53, .1910)]),
]
EDGE = {"none": "#4d4d4d", "dehaze": "#2ca02c", "gamma": "#ff7f0e", "clahe": "#7f7f7f"}


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    test = [h for h in hits if "/test/" in h]
    if not (test or hits):
        raise SystemExit(f"not found: {pattern}")
    return (test or hits)[0]


def load_rrshid(root, density, stem, which):
    sub = "hazy" if which == "haze" else "*"
    hits = [h for h in glob.glob(f"{root}/**/{density}_fog/*/{sub}/{stem}.png", recursive=True)
            if ("/hazy/" in h) == (which == "haze")]
    p = sorted(hits, key=lambda h: "/test/" not in h)[0]
    return np.array(Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC))


def load_codrone(root, name):
    p = find(f"{root}/**/images/{name}")
    im = Image.open(p).convert("RGB")
    im.thumbnail((768, 768), Image.BICUBIC)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrshid", required=True)
    ap.add_argument("--codrone", required=True)
    ap.add_argument("--out", default="fig_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8})
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(7.16, 7.6), gridspec_kw=dict(height_ratios=[1, 1, .62, .62]))
    for r, row in enumerate(ROWS):
        imgs = {}
        for t, (title, key, py, stored) in enumerate(row["tiles"]):
            cond, act = key.split("|")
            if cond not in imgs:
                imgs[cond] = (load_rrshid(a.rrshid, row["density"], row["src"], cond) if row["ds"] == "rrshid"
                              else load_codrone(a.codrone, row["src"]))
            inp = imgs[cond]
            lum = image_stats(inp)["lum_mean"]
            print(f"{row['src']} {key}: lum {lum:.4f} vs stored {stored:.4f} -> {'OK' if abs(lum - stored) < 2e-3 else 'MISMATCH'}")
            out = enhance(inp, act)
            Image.fromarray(out).save(os.path.join(a.out, f"{row['ds']}_{r}_{cond}_{act}.png"))
            ax = axes[r, t]
            ax.imshow(out); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(EDGE[act]); s.set_linewidth(1.6)
            ax.set_title(title, fontsize=8, fontweight="bold", color=EDGE[act] if act != "none" else "black", pad=3)
            pred = "Yes" if py > 0.5 else "No"
            ok = pred == row["ref"]
            ax.set_xlabel(f"{pred} {'✓' if ok else '✗'}   P(yes)={py:.2f}", fontsize=7.5, labelpad=2,
                          color="#1a7f1a" if ok else "#c0392b", fontfamily="DejaVu Sans")
        lab = "RRSHID (satellite)" if row["ds"] == "rrshid" else "CODrone (UAV, night)"
        ref = "answer on haze-free twin" if row["ds"] == "rrshid" else "label"
        axes[r, 0].set_ylabel(f"{lab}\n{row['q']}\n({ref}: {row['ref']})", fontsize=7, labelpad=4)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.7)
    fig.savefig(os.path.join(a.out, "real_examples.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, "real_examples.png"), dpi=220, bbox_inches="tight")
    print("saved", os.path.join(a.out, "real_examples.pdf"))


if __name__ == "__main__":
    main()
