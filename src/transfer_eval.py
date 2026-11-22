"""Transfer test: fit the router on a SOURCE folder (UC Merced, Phase 1B), apply it UNCHANGED to a TARGET
folder (AID, unseen severity, ...). Nothing is fitted on the target. Image-level paired bootstrap.

  python transfer_eval.py --src phase1 --tgt aid_v --model_tag Qwen2.5-VL-3B-Instruct
  python transfer_eval.py --src phase1 --tgt aid_v --model_tag Qwen2-VL-2B-Instruct --table_from Qwen2.5-VL-3B-Instruct
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from analyze_pilot import load
from common import ACTIONS

CONDS = ["clean", "haze", "lowlight"]


def img_boot(a, b, n=2000, seed=0):
    """Paired difference in accuracy (a - b), resampling whole images."""
    k = ["qid", "cond"]
    x = a.set_index(k)[["correct", "img_key"]].join(b.set_index(k)[["correct"]], rsuffix="_b")
    x["d"] = x.correct - x.correct_b
    g = x.groupby("img_key").d.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    bs = [(lambda i: s[i].sum() / c[i].sum())(rng.integers(0, len(g), len(g))) for _ in range(n)]
    return s.sum() / c.sum() * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100


def inputs(df):
    """One row per (question, condition): the un-enhanced input and its image statistics."""
    return df[df.action == "none"].reset_index(drop=True)


def fit_threshold(inp):
    """Two thresholds: mean luminance -> low light; mean dark channel -> haze; else clean."""
    L = np.quantile(inp.in_lum_mean, np.linspace(.02, .6, 40))
    H = np.quantile(inp.in_haze_dark, np.linspace(.3, .98, 40))
    best = (-1, None)
    for l in L:
        for h in H:
            p = np.where(inp.in_lum_mean < l, "lowlight", np.where(inp.in_haze_dark > h, "haze", "clean"))
            acc = (p == inp.cond.values).mean()
            if acc > best[0]:
                best = (acc, (float(l), float(h)))
    return best[1]


def apply_policy(df, pred_cond, table):
    t = inputs(df)[["qid", "cond"]].assign(act=[table[c] for c in pred_cond])
    s = df.merge(t, on=["qid", "cond"])
    return s[s.action == s.act].drop(columns="act")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="phase1")
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--model_tag", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--pseudo", type=float, default=0,
                    help="real data without labels: reference = VLM answer on the clean image if P(yes) is outside "
                         "[1-x, x] (e.g. 0.8); 'correct' becomes agreement with that reference")
    ap.add_argument("--extra_det", default=None,
                    help="few-shot detector recalibration: jsonl of extra labelled inputs (in_* stats + cond) added "
                         "to the DETECTOR training set only (lookup table and VLM untouched)")
    ap.add_argument("--table_from", default=None, help="take the lookup table from another VLM's source results")
    a = ap.parse_args()
    L = []
    P = lambda *x: (print(*x), L.append(" ".join(map(str, x))))

    src, _ = load(a.src, a.table_from or a.model_tag)
    src_m, _ = load(a.src, a.model_tag)
    tgt, blank = load(a.tgt, a.model_tag)
    if a.pseudo:
        ref = tgt[(tgt.cond == "clean") & (tgt.action == "none")].set_index("qid").p_yes
        ref = ref[(ref > a.pseudo) | (ref < 1 - a.pseudo)]
        print(f"pseudo-labels: kept {len(ref)} of {tgt.qid.nunique()} questions (clean-image P(yes) confident), "
              f"{(ref > 0.5).mean() * 100:.1f}% yes")
        tgt = tgt[tgt.qid.isin(ref.index)].copy()
        tgt["correct"] = ((tgt.p_yes > 0.5) == (tgt.qid.map(ref) > 0.5)).astype(int)
        blank = blank[blank.qid.isin(ref.index)].copy()
        blank["correct"] = ((blank.p_yes > 0.5) == (blank.qid.map(ref) > 0.5)).astype(int)
    si, ti = inputs(src_m), inputs(tgt)
    feats = [c for c in si.columns if c.startswith("in_")]
    if a.extra_det:
        ex = pd.read_json(a.extra_det, lines=True)
        rep = max(1, len(si) // (5 * len(ex)))  # extra rows get ~1/6 of the detector's training weight
        P(f"few-shot detector: +{len(ex)} labelled inputs ({ex.cond.value_counts().to_dict()}), repeated x{rep}")
        si = pd.concat([si] + [ex[feats + ["cond"]]] * rep, ignore_index=True)

    table = src.groupby(["cond", "action"]).correct.mean().unstack().reindex(columns=ACTIONS).idxmax(axis=1).to_dict()
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(si[feats], si.cond)
    pc_gbdt = clf.predict(ti[feats])
    l, h = fit_threshold(si)
    pc_thr = np.where(ti.in_lum_mean < l, "lowlight", np.where(ti.in_haze_dark > h, "haze", "clean"))

    tgt_best = tgt.groupby(["cond", "action"]).correct.mean().unstack().reindex(columns=ACTIONS).idxmax(axis=1).to_dict()
    pol = {"never-enhance": tgt[tgt.action == "none"]}
    for act in ACTIONS[1:]:
        pol[f"always-{act}"] = tgt[tgt.action == act]
    pol["router: GBDT (fit on source)"] = apply_policy(tgt, pc_gbdt, table)
    pol["router: thresholds (fit on source)"] = apply_policy(tgt, pc_thr, table)
    pol["oracle: true condition + source table"] = apply_policy(tgt, ti.cond.values, table)
    pol["oracle: true condition + TARGET table"] = apply_policy(tgt, ti.cond.values, tgt_best)

    P(f"# Transfer report: {a.src} -> {a.tgt} | VLM {a.model_tag}"
      + (f" | lookup table from {a.table_from}" if a.table_from else ""))
    P(f"target: questions={tgt.qid.nunique()} images={tgt.img_key.nunique()} | "
      f"blank-image acc {blank.correct.mean() * 100:.1f}% | levels {sorted(tgt.level.unique())}")
    P(f"lookup table (source): {table} | target-best: {tgt_best}")
    P(f"thresholds (source): lum<{l:.3f} -> lowlight, dark>{h:.3f} -> haze")
    P(f"detector accuracy on target: GBDT {(pc_gbdt == ti.cond.values).mean() * 100:.1f}% | "
      f"thresholds {(pc_thr == ti.cond.values).mean() * 100:.1f}%")
    P("GBDT confusion (rows true, cols predicted):\n" + pd.crosstab(ti.cond, pc_gbdt).to_markdown())

    rows = []
    for k, p in pol.items():
        acc = p.groupby("cond").correct.mean() * 100
        r = dict(policy=k, clean=acc.get("clean"), haze=acc.get("haze"), lowlight=acc.get("lowlight"),
                 overall=p.correct.mean() * 100)
        if k != "never-enhance":
            d, lo, hi = img_boot(p, pol["never-enhance"])
            r["gain_vs_never"] = f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]"
        rows.append(r)
    P("\n## Accuracy % on target (image-level 95% CI)\n" + pd.DataFrame(rows).round(1).to_markdown(index=False))
    P("\n## Key paired comparisons (router fit on source)")
    best_fixed = max([f"always-{x}" for x in ACTIONS[1:]], key=lambda k: pol[k].correct.mean())
    for r in ["router: GBDT (fit on source)", "router: thresholds (fit on source)"]:
        for ref in dict.fromkeys([best_fixed, "always-gamma", "always-dehaze"]):
            d, lo, hi = img_boot(pol[r], pol[ref])
            P(f"- {r} vs {ref}: {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    P("\n## By question type\n" + pd.DataFrame({k: pol[k].groupby("qtype").correct.mean() * 100 for k in
        ["never-enhance", "always-gamma", "always-dehaze", "router: GBDT (fit on source)"]}).round(1).to_markdown())
    if tgt.level.max() > 0:
        P("\n## By condition and severity\n" + pd.DataFrame({k: pol[k][pol[k].cond != "clean"].groupby(["cond", "level"])
            .correct.mean() * 100 for k in ["never-enhance", "always-gamma", "always-dehaze",
                                             "router: GBDT (fit on source)"]}).round(1).to_markdown())
    tag = a.model_tag + (f"_table-{a.table_from}" if a.table_from else "")
    open(os.path.join(a.tgt, f"transfer_{tag}{'_fewshot' if a.extra_det else ''}.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
