"""Fig. 2 (qualitative examples): three UC Merced questions from the main study, shown exactly as the VLM saw
them (same seeds and severities as make_versions.py --graded), with the real Qwen2.5-VL-3B answers.
The examples were chosen as cases where the router's operator corrected a wrong answer under BOTH haze and
low light; they illustrate the mechanism and are not a random sample.

Run on Kaggle (CPU is enough, internet on):
  python make_fig_examples.py --out /kaggle/working/fig_examples
"""
import argparse, os, zlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from common import degrade, enhance, image_stats

# Answers, probabilities and severities copied from results_Qwen2.5-VL-3B-Instruct_*.jsonl (Phase 1B main study).
# 'lum' is the stored mean luminance of the un-enhanced input; it is used to check the regenerated image.
EXAMPLES = [
    dict(idx=890, scene="Freeway", q="Is there a car?", gt="Yes", tiles={
        "clean|none": ("Yes", 1, .80, .6839), "haze|none": ("No", 0, .32, .8889), "haze|dehaze": ("Yes", 1, .82, .8889),
        "lowlight|none": ("No", 0, .09, .0971), "lowlight|gamma": ("Yes", 1, .73, .0971)}),
    dict(idx=1442, scene="Overpass", q="Is there a car?", gt="Yes", tiles={
        "clean|none": ("Yes", 1, .88, .4842), "haze|none": ("No", 0, .47, .8064), "haze|dehaze": ("Yes", 1, .87, .8064),
        "lowlight|none": ("No", 0, .38, .1154), "lowlight|gamma": ("Yes", 1, .65, .1154)}),
    dict(idx=439, scene="Buildings", q="Land-use class? (4 options)", gt="buildings", tiles={
        "clean|none": ("buildings", 1, .64, .5882), "haze|none": ("parking lot", 0, .17, .8257),
        "haze|dehaze": ("buildings", 1, .56, .8257), "lowlight|none": ("dense resid.", 0, .21, .1726),
        "lowlight|gamma": ("buildings", 1, .36, .1726)}),
]
COLS = [("clean|none", "Clean"), ("haze|none", "Hazy"), ("haze|dehaze", "Hazy + dehazing"),
        ("lowlight|none", "Low light"), ("lowlight|gamma", "Low light + gamma")]
EDGE = {"clean|none": "#4d4d4d", "haze|none": "#7f7f7f", "haze|dehaze": "#2ca02c",
        "lowlight|none": "#1f1f1f", "lowlight|gamma": "#ff7f0e"}


def version(src, key, cond, seed=0):
    """Exactly as make_versions.py --graded."""
    rng = np.random.default_rng(zlib.crc32(f"{seed}|{key}|{cond}".encode()))
    level = int(rng.integers(1, 4)) if cond != "clean" else 0
    return degrade(src, cond, rng, level or None), level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_examples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = load_dataset("jonathan-roberts1/UC_Merced_LandUse_MultiLabel", split="train")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8})
    fig, axes = plt.subplots(len(EXAMPLES), len(COLS), figsize=(7.16, 5.3))
    for r, ex in enumerate(EXAMPLES):
        key = f"ucm{ex['idx']}"
        src = np.array(ds[ex["idx"]]["image"].convert("RGB"))
        inp = {"clean": (src, 0)}
        for cond in ("haze", "lowlight"):
            inp[cond] = version(src, key, cond)
        for c, (tk, title) in enumerate(COLS):
            cond, act = tk.split("|")
            img, level = inp[cond]
            lum = image_stats(img)["lum_mean"]
            stored = ex["tiles"][tk][3]
            ok_img = abs(lum - stored) < 2e-3
            print(f"{key} {tk}: level {level}, lum {lum:.4f} vs stored {stored:.4f} -> {'OK' if ok_img else 'MISMATCH'}")
            out = enhance(img, act)
            Image.fromarray(out).save(os.path.join(a.out, f"{key}_{cond}_{act}.png"))
            ax = axes[r, c]
            ax.imshow(out)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(EDGE[tk]); s.set_linewidth(1.6)
            pred, correct, p, _ = ex["tiles"][tk]
            sev = f" · sev. {level}" if level else ""
            if r == 0:
                ax.set_title(title, fontsize=8.5, fontweight="bold", color=EDGE[tk] if act != "none" else "black")
            ax.set_xlabel(f"{pred} {'✓' if correct else '✗'}\np={p:.2f}{sev}", fontsize=7,
                          color="#1a7f1a" if correct else "#c0392b", labelpad=2, fontfamily="DejaVu Sans")
        axes[r, 0].set_ylabel(f"{ex['scene']}\n{ex['q']}\n(true: {ex['gt']})", fontsize=7.5, labelpad=4)
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.6)
    fig.savefig(os.path.join(a.out, "qual_examples.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, "qual_examples.png"), dpi=220, bbox_inches="tight")
    print("saved", os.path.join(a.out, "qual_examples.pdf"))


if __name__ == "__main__":
    main()
