from common import *
print("[fig07] sensitivity (Test ablations)"); style()
D = "results/ablation_test"
LA = pd.read_csv(track(ROOT / f"{D}/levels_s0.csv")); L1 = LA.set_index(["setting", "method"])
META = {sc: json.load(open(track(ROOT / f"{D}/meta_{sc}_s0.json"))) for sc in SCEN}
ST = META[SCEN[0]]["settings"]
for sc in SCEN:
    if META[sc]["settings"] != ST: stop(f"settings differ in meta_{sc}_s0.json")
P0 = ST["primary"]; miss = set([FULL] + BASE10) - set(LA.method)
if miss: stop(f"{D}/levels_s0.csv: methods missing {miss}")
def sweep(par):
    s = sorted([k for k, v in ST.items() if all(v[q] == P0[q] for q in P0 if q != par)], key=lambda k: ST[k][par])
    if len(s) < 3: stop(f"sweep {par}: settings {s}")
    return s
SW = {p: sweep(p) for p in ["sigma", "kappa", "alpha"]}; CHECKS["sweeps"] = SW
BT = {s: boot(D, s, FULL, BASE10)[0] for s in dict.fromkeys(s for v in SW.values() for s in v)}
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained"); res = {}
for ax, par, letter in [(axs[0, 0], "sigma", "a"), (axs[0, 1], "kappa", "b"), (axs[1, 0], "alpha", "c")]:
    S = SW[par]; x = np.arange(len(S))
    for k, (mm, cc, of) in DMK.items():
        dvs = []
        for xi_, s in zip(x, S):
            bb = min(BASE10, key=lambda m: float(L1.loc[(s, m), k])); dv, lo, hi = BT[s][bb][k]; sig = lo > 0 or hi < 0; dvs.append(dv)
            ax.errorbar(xi_ - 0.5 * of, dv, yerr=[[max(dv - lo, 0)], [max(hi - dv, 0)]], fmt=mm, color=cc, mfc=cc if sig else "white", ms=6, lw=1.1, capsize=2.5, zorder=3)
            res.setdefault(par, {}).setdefault(k, {})[s] = {"best_baseline": bb, "full": float(L1.loc[(s, FULL), k]), "baseline": float(L1.loc[(s, bb), k]), "diff_ci": [dv, lo, hi]}
        ax.plot(x - 0.5 * np.array([of] * len(x)), dvs, color=cc, lw=0.9, alpha=0.6, zorder=2)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([f"{ST[s][par]:g}" for s in S]); ax.set_xlim(-0.5, len(S) - 0.5)
    ax.get_xticklabels()[S.index("primary")].set_fontweight("bold"); ax.grid(axis="x", visible=False)
    ax.set_xlabel(nm(par, "parameters")); ax.set_ylabel(f"Best baseline {MINUS} {nm(FULL)}"); panel(ax, letter, nm(f"t_{par}", "fig_labels"))
KS = SW["kappa"]; cl = []
for sc in SCEN:
    cl.append(pq.read_table(track(ROOT / f"{D}/clusters_{sc}_s0.parquet"), columns=["setting", "method", "profile", "variant", "pr1", "pp1"],
              filters=[("method", "=", FULL), ("profile", "=", "nominal"), ("variant", "=", TD), ("setting", "in", KS)]).to_pandas().assign(sc=sc))
g = pd.concat(cl).groupby(["sc", "setting"])[["pr1", "pp1"]].sum()
for sc in SCEN:
    ep = pq.read_table(track(ROOT / f"results/final/item256/episodes_{sc}_s0.parquet"), columns=["x", "probed", "probe_pos"],
                       filters=[("method", "=", FULL), ("profile", "=", "nominal"), ("variant", "=", TD)]).to_pandas()
    ep = ep[ep.x.astype(int) == 1]; ref = [int(ep.probed.sum()), int(ep.probe_pos.sum())]; got = [int(g.loc[(sc, "primary"), "pr1"]), int(g.loc[(sc, "primary"), "pp1"])]
    CHECKS[f"probe_counts_primary_{sc}"] = {"episodes_s0": ref, "clusters_pr1_pp1": got}
    if ref != got: stop(f"{sc}: clusters pr1/pp1 {got} != episodes probed/probe_pos {ref} - column meaning not confirmed")
ax = axs[1, 1]; KM = dict(zip(KS, ["^", "s", "o"])); tr = {}
for sc in SCEN:
    lag = float(META[sc]["lag_kappa1"][TD]["median_m"])
    xs = [ST[s]["kappa"] * lag for s in KS]; ys = [100 * g.loc[(sc, s), "pp1"] / g.loc[(sc, s), "pr1"] for s in KS]
    ax.plot(xs, ys, color=SCOL[sc], lw=1.4, zorder=2)
    for s, x_, y_ in zip(KS, xs, ys): ax.scatter(x_, y_, marker=KM[s], color=SCOL[sc], s=60, edgecolor="k", lw=0.4, zorder=3)
    tr[sc] = {"median_lag_kappa1_m": lag, "points": {s: [x_, y_] for s, x_, y_ in zip(KS, xs, ys)}}
ax.set_ylim(0, 100); ax.margins(x=0.06); ax.set_xlabel(nm("poserr", "fig_labels")); ax.set_ylabel(nm("detect", "fig_labels"))
panel(ax, "d", nm("t_trade", "fig_labels"))
rows_legend(fig, [diff_handles(), [Line2D([], [], color=SCOL[s], lw=2, label=nm(s, "scenarios")) for s in SCEN] +
                  [Line2D([], [], marker=KM[s], color="0.5", mec="k", mew=0.4, ls="", ms=7, label=f"$\\kappa$ = {ST[s]['kappa']:g}") for s in KS]])
CHECKS["sensitivity"] = res; CHECKS["tradeoff"] = tr
save(fig, "fig07_sensitivity")
