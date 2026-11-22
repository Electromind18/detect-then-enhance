"""Paper figures + numbers from a results folder (re-run after Phase 1B to refresh everything).

  python make_figures.py --data phase1 --model_tag Qwen2.5-VL-3B-Instruct --out ../paper/graphs
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_pilot import load, cv_gate, cv_router, boot_diff
from common import ACTIONS

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
GREY, BLUE = "#8c8c8c", "#1f77b4"
ACOL = {"none": "#4d4d4d", "clahe": "#9467bd", "gamma": "#ff7f0e", "dehaze": "#2ca02c"}
COND = {"clean": "Clean", "haze": "Haze", "lowlight": "Low light"}
QT = {"presence_object": "Object", "presence_surface": "Surface", "scene_mcq": "Scene"}


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="phase1")
    ap.add_argument("--model_tag", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="graphs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, blank = load(a.data, a.model_tag)
    router, det = cv_router(df)
    pol = {"Never": df[df.action == "none"], "CLAHE": df[df.action == "clahe"],
           "Gamma": df[df.action == "gamma"], "Dehaze": df[df.action == "dehaze"],
           "Gate": cv_gate(df, 0.02), "Router": router}
    best_c = df.groupby(["cond", "action"]).correct.mean().unstack().idxmax(axis=1)
    pol["Cond. oracle"] = df[df.action == df.cond.map(best_c)]
    numbers = {"detector_acc": det, "blank_acc": blank.correct.mean()}

    # (a) accuracy heat map, condition x action
    m = df.pivot_table(index="cond", columns="action", values="correct", aggfunc="mean").reindex(
        index=list(COND), columns=ACTIONS) * 100
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    im = ax.imshow(m.values, cmap="Blues", vmin=m.values.min() - 5, vmax=m.values.max() + 2)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            best = m.values[i, j] == m.values[i].max()
            ax.text(j, i, f"{m.values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if best else "normal",
                    color="white" if m.values[i, j] > m.values.mean() + 3 else "black")
    ax.set_xticks(range(4), ["None", "CLAHE", "Gamma", "Dehaze"])
    ax.set_yticks(range(3), [COND[c] for c in m.index])
    ax.set_xlabel("Enhancement action"); ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Accuracy (%)")
    fig.savefig(f"{a.out}/heatmap_cond_action.pdf"); fig.savefig(f"{a.out}/heatmap_cond_action.png"); plt.close(fig)

    # (b) overall accuracy of each policy with 95% bootstrap CI
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    names = list(pol)
    acc = [pol[k].correct.mean() * 100 for k in names]
    cis = [boot_ci(pol[k].correct.values) for k in names]
    err = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(acc, cis)]).T
    cols = [BLUE if k == "Router" else ("#c7c7c7" if "oracle" in k else GREY) for k in names]
    ax.bar(range(len(names)), acc, color=cols, yerr=err, capsize=2, error_kw={"lw": 0.8})
    for i, v in enumerate(acc):
        ax.text(i, v + err[1][i] + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(names)), names, fontsize=7, rotation=30, ha="right")
    ax.set_ylim(min(acc) - 6, max(acc) + 3); ax.set_ylabel("Overall accuracy (%)")
    fig.savefig(f"{a.out}/policy_accuracy.pdf"); fig.savefig(f"{a.out}/policy_accuracy.png"); plt.close(fig)
    numbers["policy_acc"] = dict(zip([n.replace("\n", " ") for n in names], acc))

    # (c,d) accuracy by severity, haze and low light
    sev_rows = {}
    for cond in ["haze", "lowlight"]:
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        for k, st in [("Never", dict(color="#4d4d4d", ls="-")), ("Gamma", dict(color=ACOL["gamma"], ls=":")),
                      ("Dehaze", dict(color=ACOL["dehaze"], ls="--")), ("Router", dict(color=BLUE, ls="-", lw=2.2))]:
            p = pol[k][pol[k].cond == cond]
            s = p.groupby("level").correct.mean() * 100
            sev_rows[(cond, k)] = s.round(1).to_dict()
            ax.plot(s.index, s.values, marker="o", ms=4, label=k, **st)
        ax.set_xticks([1, 2, 3], ["Mild", "Medium", "Strong"])
        ax.set_ylabel("Accuracy (%)"); ax.set_xlabel(f"{COND[cond]} severity")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.savefig(f"{a.out}/severity_{cond}.pdf"); fig.savefig(f"{a.out}/severity_{cond}.png"); plt.close(fig)
    numbers["severity"] = {f"{c}|{k}": v for (c, k), v in sev_rows.items()}

    # (e) hurt rate by question type x action (degraded inputs)
    base = df[df.action == "none"][["qid", "cond", "correct"]]
    mm = df[(df.action != "none") & (df.cond != "clean")].merge(base, on=["qid", "cond"], suffixes=("", "_n"))
    mm["hurt"] = (mm.correct < mm.correct_n).astype(float) * 100
    mm["helped"] = (mm.correct > mm.correct_n).astype(float) * 100
    h = mm.pivot_table(index="qtype", columns="action", values="hurt", aggfunc="mean").reindex(
        index=list(QT), columns=["clahe", "gamma", "dehaze"])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    w = 0.26
    for j, act in enumerate(h.columns):
        ax.bar(np.arange(3) + (j - 1) * w, h[act].values, w, label=act.upper() if act == "clahe" else act.capitalize(),
               color=ACOL[act])
    ax.set_xticks(range(3), [QT[q] for q in h.index]); ax.set_ylabel("Hurt rate (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(f"{a.out}/hurt_by_qtype.pdf"); fig.savefig(f"{a.out}/hurt_by_qtype.png"); plt.close(fig)
    numbers["hurt_qtype"] = h.round(1).to_dict()
    numbers["help_qtype"] = mm.pivot_table(index="qtype", columns="action", values="helped", aggfunc="mean").round(1).to_dict()

    # numbers for the text
    for k in ["Gamma", "Dehaze", "Gate"]:
        d, lo, hi = boot_diff(pol["Router"], pol[k]); numbers[f"router_vs_{k}"] = [d, lo, hi]
    d, lo, hi = boot_diff(pol["Router"], pol["Never"]); numbers["router_vs_Never"] = [d, lo, hi]
    numbers["by_cond"] = {k.replace("\n", " "): (pol[k].groupby("cond").correct.mean() * 100).round(1).to_dict() for k in pol}
    numbers["by_qtype"] = {k.replace("\n", " "): (pol[k].groupby("qtype").correct.mean() * 100).round(1).to_dict() for k in pol}
    json.dump(numbers, open(f"{a.out}/numbers_{a.model_tag}.json", "w"), indent=1, default=float)
    print(json.dumps(numbers, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
