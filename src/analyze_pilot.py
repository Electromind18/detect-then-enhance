"""Phase 0 analysis + go/no-go. CPU only, seconds.

  python analyze_pilot.py --data pilot --model_tag Qwen2.5-VL-3B-Instruct
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from common import ACTIONS

META = {"key", "qid", "img_key", "cond", "action", "qtype", "qcat", "answer", "file",
        "p_yes", "yn_mass", "p_correct", "correct"}


def load(data, tag):
    files = sorted(glob.glob(os.path.join(data, f"results_{tag}*.jsonl")))  # all GPU shards
    r = pd.concat([pd.read_json(f, lines=True) for f in files]).drop_duplicates("key")
    f = pd.read_json(os.path.join(data, "features.jsonl"), lines=True)
    blank = r[r.cond == "blank"]
    r = r[r.cond != "blank"].merge(f, on=["img_key", "cond", "action"], how="left")
    return r, blank


def gate_features(df, use_q=True, use_img=True):
    cols = []
    if use_img:
        cols += [c for c in df.columns if c.startswith("in_") or c.startswith("drift_")]
    X = df[cols].copy()
    for a in ACTIONS:
        X[f"act_{a}"] = (df.action == a).astype(int)
    if use_q:
        for c in sorted(df.qcat.unique()):
            X[f"q_{c}"] = (df.qcat == c).astype(int)
        for c in sorted(df.qtype.unique()):
            X[f"t_{c}"] = (df.qtype == c).astype(int)
    return X


def pick(df, score_col, margin):
    """Per (qid, cond): best-scored action; fall back to 'none' unless gain > margin."""
    out = []
    for (_, _), g in df.groupby(["qid", "cond"], sort=False):
        g = g.set_index("action")
        best = g[score_col].idxmax()
        if best != "none" and g.loc[best, score_col] - g.loc["none", score_col] <= margin:
            best = "none"
        row = g.loc[best].copy()
        row["action"] = best
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def cv_gate(df, margin, seed=0, **kw):
    df = df.copy()
    df["pred"] = np.nan
    for tr, te in GroupKFold(n_splits=5).split(df, groups=df.img_key):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(gate_features(df.iloc[tr], **kw), df.iloc[tr].p_correct)
        df.iloc[te, df.columns.get_loc("pred")] = m.predict(gate_features(df.iloc[te], **kw))
    return pick(df, "pred", margin)


def cv_qtype_rule(df):
    """Best fixed action per question type, chosen on train folds (condition unknown at test)."""
    sel = []
    for tr, te in GroupKFold(n_splits=5).split(df, groups=df.img_key):
        best = df.iloc[tr].groupby(["qcat", "action"]).p_correct.mean().unstack().idxmax(axis=1)
        t = df.iloc[te]
        sel.append(t[t.action == t.qcat.map(best).fillna("none")])
    return pd.concat(sel)


def cv_router(df, use_q=False, seed=0):
    """Two-stage router (no VLM): (1) detect degradation type from input image stats,
    (2) apply the action that was best for that type [and question type] on the training folds."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    inp = df[df.action == "none"].reset_index(drop=True)
    X = inp[[c for c in inp.columns if c.startswith("in_")]]
    out, accs = [], []
    for tr, te in GroupKFold(n_splits=5).split(inp, groups=inp.img_key):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X.iloc[tr], inp.cond.iloc[tr])
        pred = clf.predict(X.iloc[te])
        accs.append((pred == inp.cond.iloc[te].values).mean())
        dtr = df[df.qid.isin(set(inp.qid.iloc[tr]))]
        keys = ["cond", "qtype"] if use_q else ["cond"]
        best = dtr.groupby(keys + ["action"]).correct.mean().unstack().idxmax(axis=1)
        t = inp.iloc[te][["qid", "cond", "qtype"]].copy()
        t["act"] = [best.get((c, q), "none") if use_q else best.get(c, "none") for c, q in zip(pred, t.qtype)]
        s = df.merge(t[["qid", "cond", "act"]], on=["qid", "cond"])
        out.append(s[s.action == s.act].drop(columns="act"))
    return pd.concat(out), float(np.mean(accs))


def boot_diff(a, b, n=2000, seed=0):
    """Paired bootstrap over (qid,cond) units of acc(a)-acc(b)."""
    a = a.set_index(["qid", "cond"]).correct
    b = b.set_index(["qid", "cond"]).correct.reindex(a.index)
    d = (a - b).values
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean() * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--model_tag", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args()
    df, blank = load(a.data, a.model_tag)
    L = []
    P = lambda *x: (print(*x), L.append(" ".join(map(str, x))))

    P(f"# Pilot report: {a.model_tag}\n")
    P(f"questions={df.qid.nunique()}  images={df.img_key.nunique()}  runs={len(df) + len(blank)}")
    P(f"text-only (grey image) accuracy: {blank.correct.mean() * 100:.1f}%   "
      f"mean answer-token prob. mass: {df.yn_mass.mean():.3f}\n")

    P("## Accuracy % (rows=condition, cols=action)")
    P(df.pivot_table(index="cond", columns="action", values="correct", aggfunc="mean")
      .reindex(columns=ACTIONS).mul(100).round(1).to_markdown())
    P("\n## Mean P(correct) (soft score, lower variance)")
    P(df.pivot_table(index="cond", columns="action", values="p_correct", aggfunc="mean")
      .reindex(columns=ACTIONS).round(3).to_markdown())

    base = df[df.action == "none"][["qid", "cond", "correct", "p_correct"]]
    m = df[df.action != "none"].merge(base, on=["qid", "cond"], suffixes=("", "_none"))
    m["helped"] = (m.correct > m.correct_none).astype(int)
    m["hurt"] = (m.correct < m.correct_none).astype(int)
    P("\n## Help / hurt rate % vs 'none' (by condition x action)")
    P(m.groupby(["cond", "action"])[["helped", "hurt"]].mean().mul(100).round(1).unstack().to_markdown())
    P("\n## Hurt rate % by question type x action")
    P(m.pivot_table(index="qtype", columns="action", values="hurt", aggfunc="mean").mul(100).round(1).to_markdown())

    P("\n## Evidence drift: mean drift when enhancement HURT vs did not")
    for col in ["drift_dE", "drift_hue_drift", "drift_hf_ratio"]:
        g = m.groupby("hurt")[col].mean()
        P(f"- {col}: not-hurt={g.get(0, np.nan):.3f}  hurt={g.get(1, np.nan):.3f}")

    # ---- policies ----
    never = df[df.action == "none"]
    oracle = df.loc[df.groupby(["qid", "cond"]).p_correct.idxmax()]
    oracle_hard = df.groupby(["qid", "cond"]).correct.max().reset_index()
    pol = {"never-enhance": never}
    for act in ACTIONS[1:]:
        pol[f"always-{act}"] = df[df.action == act]
    pol["qtype-rule (CV)"] = cv_qtype_rule(df)
    pol["gate: image-only (CV)"] = cv_gate(df, a.margin, use_q=False)
    pol["gate: question-only (CV)"] = cv_gate(df, a.margin, use_img=False)
    pol["gate: full (CV)"] = cv_gate(df, a.margin)
    pol["router: degradation (CV)"], det_acc = cv_router(df)
    pol["router: degradation+qtype (CV)"], _ = cv_router(df, use_q=True)
    best_c = df.groupby(["cond", "action"]).correct.mean().unstack().idxmax(axis=1)
    pol["condition-oracle (knows degradation)"] = df[df.action == df.cond.map(best_c)]

    P(f"\n## Policies (accuracy %, margin={a.margin}); gain vs never with 95% paired-bootstrap CI")
    rows = []
    for name, p in pol.items():
        acc = p.groupby("cond").correct.mean().mul(100)
        d, lo, hi = boot_diff(p, never) if name != "never-enhance" else (0, 0, 0)
        enh = (p.action != "none").mean() * 100
        rows.append(dict(policy=name, clean=acc.get("clean"), haze=acc.get("haze"),
                         lowlight=acc.get("lowlight"), overall=p.correct.mean() * 100,
                         gain=d, ci=f"[{lo:+.1f},{hi:+.1f}]", pct_enhanced=enh))
    rows.append(dict(policy="ORACLE (any action correct; optimistic)",
                     clean=oracle_hard[oracle_hard.cond == "clean"].correct.mean() * 100,
                     haze=oracle_hard[oracle_hard.cond == "haze"].correct.mean() * 100,
                     lowlight=oracle_hard[oracle_hard.cond == "lowlight"].correct.mean() * 100,
                     overall=oracle_hard.correct.mean() * 100,
                     gain=(oracle_hard.correct.mean() - never.correct.mean()) * 100))
    tab = pd.DataFrame(rows).round(1)
    P(tab.to_markdown(index=False))
    P(f"\nDegradation detector accuracy (image stats only, CV): {det_acc * 100:.1f}%")
    for ref in ["always-gamma", "always-dehaze", "gate: full (CV)"]:
        d, lo, hi = boot_diff(pol["router: degradation (CV)"], pol[ref])
        P(f"- router vs {ref}: {d:+.2f} [{lo:+.2f},{hi:+.2f}]")

    # ---- question-awareness: does the best policy differ by question type? ----
    if df.qtype.nunique() > 1:
        P("\n## Accuracy % by question type (key evidence for 'question-aware')")
        keep = ["never-enhance", "always-gamma", "always-clahe", "always-dehaze",
                "gate: full (CV)", "router: degradation (CV)", "router: degradation+qtype (CV)"]
        t = pd.DataFrame({k: pol[k].groupby("qtype").correct.mean() * 100 for k in keep}).round(1)
        P(t.to_markdown())
        P("\n## Hurt rate % by question type x action, degraded inputs only")
        P(m[m.cond != "clean"].pivot_table(index="qtype", columns="action", values="hurt",
                                           aggfunc="mean").mul(100).round(1).to_markdown())
    if "level" in df.columns and df.level.max() > 0:
        P("\n## Accuracy % by degradation severity (1=mild, 3=strong)")
        rows = {}
        for k in ["never-enhance", "always-gamma", "always-dehaze", "router: degradation (CV)"]:
            p = pol[k].merge(df[["qid", "cond", "level"]].drop_duplicates(), on=["qid", "cond"],
                             suffixes=("_x", "")) if "level" not in pol[k] else pol[k]
            p = p[p.cond != "clean"]
            rows[k] = p.groupby(["cond", "level"]).correct.mean() * 100
        P(pd.DataFrame(rows).round(1).to_markdown())

    # ---- go / no-go ----
    og = (oracle_hard.correct.mean() - never.correct.mean()) * 100
    gd, glo, _ = boot_diff(pol["router: degradation (CV)"], never)
    clean_harm = (pol["router: degradation (CV)"].query("cond=='clean'").correct.mean()
                  - never.query("cond=='clean'").correct.mean()) * 100
    P("\n## GO / NO-GO")
    P(f"- oracle headroom: {og:+.1f} pts (need >= 3)")
    P(f"- router vs never: {gd:+.1f} pts, CI lower bound {glo:+.1f} (want > 0)")
    P(f"- router effect on CLEAN images: {clean_harm:+.1f} pts (want >= -0.5: gate must not hurt clean)")
    clean_acc = never[never.cond == "clean"].correct.mean() * 100
    P(f"- image dependence: never-enhance clean {clean_acc:.1f}% vs text-only {blank.correct.mean()*100:.1f}%")
    if og >= 3 and glo > 0:
        P("=> GO: gating story holds. Scale to Phase 1.")
    elif og >= 3:
        P("=> PARTIAL: headroom exists but gate not yet significant. Scale data (Phase 1) before deciding.")
    else:
        P("=> NO-GO for gating: switch to diagnostic framing ('when/why enhancement hurts RS-VLMs').")

    with open(os.path.join(a.data, f"report_{a.model_tag}.md"), "w") as f:
        f.write("\n".join(L))
    m.to_csv(os.path.join(a.data, f"help_hurt_{a.model_tag}.csv"), index=False)


if __name__ == "__main__":
    main()
