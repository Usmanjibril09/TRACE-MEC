from common import *
print("[fig02] scenarios"); style()
fig, ax = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained"); xi = np.arange(len(SCEN))
a = ax[0, 0]; tps = [float(v) for v in find_dict_with(CFG, "t_p_s")["t_p_s"]]; tmin = {}
for sc in SCEN:
    t = pd.read_parquet(track(ROOT / f"data/heldout/{sc}/Test/handovers.parquet"), columns=["t_ho_s"])["t_ho_s"].to_numpy()
    tmin[sc] = float(t.min()); x = np.sort(t)
    a.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=SCOL[sc], lw=1.4)
    o = json.load(open(track(ROOT / f"results/final/item256/operating_{sc}_s0.json")))["probe"]
    CHECKS[f"T_HO_min_{sc}"] = {"handovers": tmin[sc], "operating_json": o["t_ho_min"]}
a.axvspan(min(tps), max(tps), color="0.6", alpha=0.3, lw=0)
a.text(np.sqrt(min(tps) * max(tps)), 0.5, "Probe RTT", rotation=90, ha="center", va="center", fontsize=FS["annot"])
a.set_xscale("log"); fmt_g(a); a.set_xlim(left=min(tps) * 0.7); a.set_ylim(0, 1.02)
a.set_xlabel("Handover window $T_{HO}$ (s)"); a.set_ylabel("ECDF"); panel(a, "a", "Handover window")

b = ax[0, 1]; zero = {}
sizes = sorted(int(p.name[4:]) for p in (ROOT / "results/final").glob("item*") if p.name[4:].isdigit())
LS = dict(zip(sizes, ["-", "--", ":", "-."]))
for sz in sizes:
    for sc in SCEN:
        f = track(ROOT / f"results/final/item{sz}/episodes_{sc}_s0.parquet")
        v = pq.read_table(f, columns=["budget"], filters=[("method", "=", FULL), ("profile", "=", "nominal")])["budget"].to_numpy()
        x = np.sort(v); b.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=SCOL[sc], ls=LS[sz], lw=1.3)
        zero[(sz, sc)] = float((v == 0).mean())
CHECKS["zero_budget_share"] = {f"{k[0]}|{k[1]}": v for k, v in zero.items()}
b.set_xscale("symlog", linthresh=1); fmt_g(b); b.set_xlim(left=-0.2); b.set_ylim(0, 1.02)
b.set_xlabel("Transfer budget (items)"); b.set_ylabel("ECDF"); panel(b, "b", "Transfer budget")

c = ax[1, 0]; seeds = seeds_in("results/final/item256"); rows = []
for sc in SCEN:
    for s in seeds:
        d = pd.read_csv(track(ROOT / f"results/final/item256/summary_{sc}_s{s}.csv"))
        rows.append(d[(d.profile == "nominal") & d.method.isin(SHOW) & (d.variant != "baseline")][["method", "variant", "belief_auc"]])
au = pd.concat(rows).groupby(["method", "variant"]).belief_auc.mean().unstack()
atk = [v for v in VARS if v != "baseline"]
VM = dict(zip(atk, ["o", "s", "^"])); VC = dict(zip(atk, ["#0072B2", "#009E73", "#CC79A7"]))
yy = np.arange(len(SHOW))[::-1]; iF = SHOW.index(FULL)
c.axhspan(yy[iF] - 0.5, yy[iF] + 0.5, color=COL[FULL], alpha=0.10, lw=0)
for v in atk:
    c.scatter(au.loc[SHOW, v], yy, marker=VM[v], color=VC[v], s=34, edgecolor="k", lw=0.3, zorder=3)
c.axvline(0.5, color="0.4", ls=":", lw=0.9)
c.set_yticks(yy); c.set_yticklabels([nm(m) for m in SHOW]); bold_label(c)
c.set_xlim(min(0.45, float(au.loc[SHOW, atk].min().min()) - 0.03), 1.02); c.set_ylim(-0.6, len(SHOW) - 0.4)
c.set_xlabel(nm("belief_auc", "metrics")); panel(c, "c", nm("belief_auc", "metrics"))
CHECKS["belief_auc_mean_over_scenarios_seeds"] = au.loc[SHOW, atk].round(4).to_dict()

dd = ax[1, 1]; pp, hv = {}, {}
splits = [("Train", "data/processed"), ("Validation", "data/processed"), ("Test", "data/heldout")]
for sp, root in splits:
    for sc in SCEN:
        h = pd.read_parquet(track(ROOT / f"{root}/{sc}/{sp}/handovers.parquet"), columns=["sender_id", "pingpong"])
        pp[(sp, sc)] = 100 * float(h.pingpong.mean()); hv[(sp, sc)] = len(h) / h.sender_id.nunique()
SPC = dict(zip([s for s, _ in splits], ["0.85", "0.6", "0.3"])); w = 0.26; d2 = dd.twinx()
for j, (sp, _) in enumerate(splits):
    dd.bar(xi + (j - 1) * w, [pp[(sp, s)] for s in SCEN], w, color=SPC[sp], edgecolor="k", lw=0.4)
    d2.plot(xi + (j - 1) * w, [hv[(sp, s)] for s in SCEN], "D", color="w", mec="k", ms=5, mew=0.7)
dd.set_xticks(xi); dd.set_xticklabels([two_line(nm(s, "scenarios")) for s in SCEN]); dd.set_xlim(-0.55, len(SCEN) - 0.45)
dd.set_ylabel("Ping-pong (%)"); d2.set_ylabel("Handovers per vehicle"); d2.grid(False); panel(dd, "d", "Handovers")
CHECKS["pingpong_pct"] = {f"{k[0]}|{k[1]}": v for k, v in pp.items()}
rows_legend(fig, [[Line2D([], [], color=SCOL[s], lw=2, label=nm(s, "scenarios")) for s in SCEN] +
                  [Line2D([], [], color="k", ls=LS[s], lw=1.3, label=nm(s, "item_sizes")) for s in sizes],
                  [Line2D([], [], marker=VM[v], color=VC[v], mec="k", mew=0.3, ls="", ms=6, label=nm(v, "variants")) for v in atk] +
                  [Patch(fc=SPC[s], ec="k", label=s) for s, _ in splits] +
                  [Line2D([], [], marker="D", ls="", color="w", mec="k", ms=5, label="Handovers per vehicle")]])
save(fig, "fig02_scenarios")
