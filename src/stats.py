"""stats.py (script 6/8) - statistical analysis.
v0.1: preregistered v0.7 criteria C1-C4 (docs/theory.md Sec. 6, 11, 13). The same functions are reused
unchanged for the single Test evaluation."""
import argparse, json, pathlib, subprocess
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MENU = ["adv_burst", "adv_sustained", "adv_early", "adv_suppress_top"]
BASELINES = ["fifo", "recency", "severity", "random", "aoi", "voi_myopic"]
FULL = "robust_v1+probe"
ABLATIONS = ["robust_v1", "robust_v1+probe_rnd", "fifo_rb+probe", "no_transfer_rb+probe", "severity_rb",
             "fifo_rb", "no_transfer_rb", "no_transfer", "ref_full_clean", "ref_full_clean_rb"]
SCEN = ["InTAS_highway_2", "InTAS_urban_2", "InTAS_highway_7", "InTAS_urban_7"]
METRICS = ["nominal_mean", "menu_mean", "menu_worst", "td_worst"]
EXTENDED = ["beta_rep", "central_ma", "f2md", "lstm_sev"]   # A9: secondary on Test, not part of C1-C4
SENS = ["f2md_tuned"]                                        # sensitivity only, never counted
ABL_LSTM = ["lstm_v1+probe", "lstm_v1"]                    # A11 trace_lstm ablation


def load(res_dir, seed, version):
    dd = []
    for sc in SCEN:
        man = json.loads((res_dir / f"manifest_{sc}_s{seed}.json").read_text())
        if man.get("version") != version:
            raise SystemExit(f"ABORT: {sc} results are version {man.get('version')}, expected {version}; "
                             f"re-run: python src/model.py --smoke --scenario {sc}")
        d = pd.read_parquet(res_dir / f"episodes_{sc}_s{seed}.parquet",
                            columns=["session_uid", "x", "method", "profile", "variant", "loss", "action"])
        d["scen"] = sc
        d["cl"] = sc + "|" + d["session_uid"].str.split("|").str[-1]
        dd.append(d)
    return pd.concat(dd, ignore_index=True)


def metrics(cm, idx, m):
    is_m = (idx["method"] == m).to_numpy()
    menu = idx["profile"].isin(MENU).to_numpy()
    nom = np.flatnonzero(is_m & (idx["profile"] == "nominal").to_numpy())
    men = np.flatnonzero(is_m & menu)
    td = np.flatnonzero(is_m & menu & (idx["variant"] == "timeDelayAttack").to_numpy())
    assert len(nom) == 4 and len(men) == 16 and len(td) == 4, f"{m}: cells nom={len(nom)} menu={len(men)} td={len(td)}"
    return {"nominal_mean": cm[..., nom].mean(-1), "menu_mean": cm[..., men].mean(-1),
            "menu_worst": cm[..., men].max(-1), "td_worst": cm[..., td].max(-1)}


def worst_cell(pt, idx, m):
    k = np.flatnonzero((idx["method"] == m).to_numpy() & idx["profile"].isin(MENU).to_numpy())
    j = k[np.argmax(pt[k])]
    return f"{idx['profile'].iloc[j]}/{idx['variant'].iloc[j]} {pt[j]:.4f}"


def criteria(d, B, seed):
    methods = [FULL] + BASELINES + ABLATIONS + EXTENDED + SENS + ABL_LSTM
    d = d[d["method"].isin(methods)]
    g = d.groupby(["method", "profile", "variant", "cl"])["loss"].agg(["sum", "count"])
    S = g["sum"].unstack("cl", fill_value=0.0)
    N = g["count"].unstack("cl", fill_value=0).reindex(index=S.index, columns=S.columns)
    idx, S, N = S.index.to_frame(index=False), S.to_numpy(float), N.to_numpy(float)
    pt, C = S.sum(1) / N.sum(1), S.shape[1]
    W = np.random.default_rng(seed).multinomial(C, np.full(C, 1.0 / C), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ S.T) / (W @ N.T)
    P = {m: metrics(pt, idx, m) for m in methods}
    BS = {m: metrics(bs, idx, m) for m in methods}
    rows = []
    for m in methods:
        for k in METRICS:
            lo, hi = np.nanpercentile(BS[m][k] - BS[FULL][k], [2.5, 97.5])
            rows.append(dict(method=m, metric=k, level=float(P[m][k]), diff=float(P[m][k] - P[FULL][k]),
                             lo=float(lo), hi=float(hi), sig="*" if (lo > 0 or hi < 0) else ""))
    T = pd.DataFrame(rows)
    r = lambda m, k: T[(T["method"] == m) & (T["metric"] == k)].iloc[0]
    best = lambda k: min(BASELINES, key=lambda m: P[m][k])
    b2, b3 = best("menu_mean"), best("nominal_mean")
    res = {
        "C1": bool(P[FULL]["menu_worst"] < min(P[m]["menu_worst"] for m in BASELINES)
                   and r("fifo", "menu_worst")["lo"] > 0 and r("severity", "menu_worst")["lo"] > 0),
        "C2": bool(not r(b2, "menu_mean")["hi"] < 0), "C2_best": b2,
        "C3": bool(not r(b3, "nominal_mean")["hi"] < 0), "C3_best": b3,
        "C4": bool(r("no_transfer_rb", "td_worst")["lo"] > 0)}
    wc = {m: worst_cell(pt, idx, m) for m in [FULL, "robust_v1", "no_transfer_rb", "fifo", "severity", b2, b3]}
    wc.update({m: worst_cell(pt, idx, m) for m in EXTENDED})
    for k in METRICS:
        bm = min(BASELINES + EXTENDED, key=lambda m: P[m][k]); rr = r(bm, k)
        res[f"best10_{k}"] = {"method": bm, "level": float(P[bm][k]), "full": float(P[FULL][k]),
                              "lo": float(rr["lo"]), "hi": float(rr["hi"])}
    for k in METRICS:
        lo, hi = np.nanpercentile(BS["lstm_sev"][k] - BS["lstm_v1+probe"][k], [2.5, 97.5])
        res[f"trace_lstm_{k}"] = {"abl": float(P["lstm_v1+probe"][k]), "noprobe": float(P["lstm_v1"][k]),
                                 "base": float(P["lstm_sev"][k]), "full": float(P[FULL][k]), "lo": float(lo), "hi": float(hi)}
    return T, C, res, wc


def per_scen(d):
    methods = [FULL] + BASELINES + ABLATIONS + EXTENDED + SENS + ABL_LSTM
    out = []
    for sc, g in d[d["method"].isin(methods)].groupby("scen"):
        cm = g.groupby(["method", "profile", "variant"])["loss"].mean()
        idx = cm.index.to_frame(index=False)
        for m in methods:
            out.append(dict(scen=sc, method=m, **{k: float(v) for k, v in metrics(cm.to_numpy(), idx, m).items()}))
    return pd.DataFrame(out)


def secondary(res_dir, seed):
    dd = [pd.read_csv(res_dir / f"eval_alpha_{sc}_s{seed}.csv").assign(scen=sc) for sc in SCEN]
    e = pd.concat(dd, ignore_index=True)
    mn = e[e["profile"].isin(MENU)].groupby(["scen", "method"])["miss_at_alpha"]
    return pd.concat([e[e["profile"] == "nominal"].groupby(["scen", "method"])["miss_at_alpha"].mean().rename("nominal_mean"),
                      mn.mean().rename("menu_mean"), mn.max().rename("menu_worst")], axis=1).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--version", default="v0.7.1")
    ap.add_argument("--out", default="docs/criteria_v07_validation.txt")
    a = ap.parse_args()
    if not a.criteria:
        raise SystemExit("usage: python src/stats.py --criteria")
    res_dir = ROOT / "results" / "model_smoke"
    d = load(res_dir, a.seed, a.version)
    T, C, res, wc = criteria(d, a.B, a.seed)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    L = [f"# v0.7 preregistered criteria, pooled Validation (auto-generated {datetime.now():%Y-%m-%d %H:%M})",
         f"- git {git}; seed {a.seed}; B={a.B}; clusters={C} (scenario, vehicle); version {a.version}",
         f"- diff = method - {FULL} (positive = {FULL} better); * = 95% CI excludes 0",
         "", "## Primary: loss at the matched operating point (Train-calibrated thresholds)"]
    for k in METRICS:
        L += ["", f"### {k}", T[T["metric"] == k].drop(columns="metric").sort_values("level").round(4).to_string(index=False)]
    L += ["", "## Worst menu cell (pooled)"] + [f"- {m}: {v}" for m, v in wc.items()]
    L += ["", "## Criteria (theory.md Sec. 13)",
          f"C1 (menu_worst lowest vs 6 baselines + sig. better than fifo & severity): {'PASS' if res['C1'] else 'FAIL'}",
          f"C2 (menu_mean not sig. worse than best baseline = {res['C2_best']}): {'PASS' if res['C2'] else 'FAIL'}",
          f"C3 (nominal_mean not sig. worse than best baseline = {res['C3_best']}): {'PASS' if res['C3'] else 'FAIL'}",
          f"C4 (td_worst sig. better than no_transfer_rb): {'PASS' if res['C4'] else 'FAIL'}"]
    hs = d[(d["profile"] == "nominal") & (d["x"] == 0)].groupby(["method", "scen"])["action"].apply(lambda s: (s >= 1).mean())
    L += ["", "## Realised honest restrict-or-quarantine share (nominal; target alpha = 0.10)", hs.unstack().round(3).to_string()]
    L += ["", "## SECONDARY (A9): TRACE-MEC vs best of all 10 performance baselines (not C1-C4)",
          "- pool: " + ", ".join(BASELINES + EXTENDED) + "; diff = best - " + FULL]
    for k in METRICS:
        b = res[f"best10_{k}"]
        verdict = f"{FULL} sig. better" if b["lo"] > 0 else (f"{b['method']} sig. better" if b["hi"] < 0 else "n.s.")
        L.append(f"- {k}: best = {b['method']} {b['level']:.4f} vs {FULL} {b['full']:.4f}; "
                 f"95% CI of diff [{b['lo']:.4f}, {b['hi']:.4f}] -> {verdict}")
    L += ["", "## ABLATION (A11) trace_lstm: diff = lstm_sev - lstm_v1+probe (positive = TRACE pipeline helps the LSTM)"]
    for k in METRICS:
        b = res[f"trace_lstm_{k}"]
        v = "pipeline sig. better" if b["lo"] > 0 else ("lstm_sev sig. better" if b["hi"] < 0 else "n.s.")
        L.append(f"- {k}: lstm_v1+probe {b['abl']:.4f} | lstm_v1 {b['noprobe']:.4f} | lstm_sev {b['base']:.4f} | "
                 f"robust_v1+probe {b['full']:.4f}; 95% CI [{b['lo']:.4f}, {b['hi']:.4f}] -> {v}")
    sup = (res["trace_lstm_menu_mean"]["lo"] > 0 and res["trace_lstm_menu_worst"]["lo"] > 0
           and not res["trace_lstm_nominal_mean"]["hi"] < 0)
    res["trace_lstm_supported"] = bool(sup)
    L.append(f"- A11 rule: {'SUPPORTED' if sup else 'NOT SUPPORTED'}")
    ps = per_scen(d)
    for k in ["menu_worst", "menu_mean", "nominal_mean"]:
        L += ["", f"## Per-scenario primary {k} (point estimates)", ps.pivot(index="method", columns="scen", values=k).round(4).to_string()]
    se = secondary(res_dir, a.seed)
    for k in ["menu_worst", "menu_mean", "nominal_mean"]:
        L += ["", f"## SECONDARY miss at exactly alpha, {k} (per scenario)", se.pivot(index="method", columns="scen", values=k).round(3).to_string()]
    (ROOT / a.out).write_text("\n".join(L) + "\n")
    (ROOT / a.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
# END-OF-FILE
