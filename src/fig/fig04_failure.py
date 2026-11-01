from common import *
print("[fig04] where methods fail"); style()
a, how, sm, L, per, J0, seeds = load_main()
cells = cells5(a, how, sm)
R = np.array([[cells[(m, p, v)] for p, v in CELLS] for m in SHOW])
ev = []
for sc in SCEN:
    for s in seeds:
        ev.append(pd.read_csv(track(ROOT / f"results/final/item256/eval_alpha_{sc}_s{s}.csv")).assign(scenario=sc, seed=s))
ev = pd.concat(ev); mi = ev.groupby(["method", "profile", "variant"]).miss_at_alpha.mean()
na = sorted({v for _, v in CELLS} - set(mi.index.unique("variant")))
for v in na:
    nx = 0
    for sc in SCEN:
        for s in seeds:
            e = pd.read_parquet(track(ROOT / f"results/final/item256/episodes_{sc}_s{s}.parquet"), columns=["variant", "x"], filters=[("variant", "==", v)])
            if e.empty: stop(f"variant {v}: no episodes in {sc} s{s}")
            nx += int(e.x.sum())
    CHECKS[f"attacker_episodes_in_{v}"] = nx
    if nx: stop(f"variant {v} has {nx} attacker episodes but no miss-at-alpha rows - eval_alpha bug, not undefined")
    print(f"  {v}: 0 attacker episodes (all scenarios, seeds) -> miss rate undefined, columns omitted in (b)")
CB = [(p, v) for p, v in CELLS if v not in na]; CHECKS["miss_columns_omitted"] = na
Mx = np.array([[mi[(m, p, v)] for p, v in CB] for m in SHOW])
abl = pd.concat([pd.read_csv(track(ROOT / f"results/ablation_test/eval_alpha_{sc}_s0.csv")) for sc in SCEN])
f0 = lambda d: d[(d.profile == "nominal") & (d.method == FULL)].groupby("scenario").miss_at_alpha.mean().mean()
c1, c2 = f0(ev[ev.seed == 0]), f0(abl[abl.setting == "primary"])
CHECKS["miss_alpha_seed0_final_vs_ablation_test"] = [c1, c2]
if abs(c1 - c2) > 1e-9: stop(f"miss-at-alpha seed 0 differs between results/final and results/ablation_test: {c1} vs {c2}")
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained")
def heat(ax, Z, cl, label, letter, title, ylab):
    profs = list(dict.fromkeys(p for p, _ in cl))
    if len(cl) % len(profs): stop(f"unequal cells per profile in panel {letter}")
    nv = len(cl) // len(profs)
    im = ax.imshow(Z, aspect="auto", cmap="magma_r", vmin=0, vmax=float(np.nanmax(Z)))
    for j in range(Z.shape[1]): ax.plot(j, np.nanargmin(Z[:, j]), "o", ms=3.5, color="w", mec="k", mew=0.5)
    iF = SHOW.index(FULL); ax.add_patch(plt.Rectangle((-0.5, iF - 0.5), Z.shape[1], 1, fill=False, ec=COL[FULL], lw=1.8))
    for x in range(nv, Z.shape[1], nv): ax.axvline(x - 0.5, color="w", lw=2)
    ax.set_yticks(range(len(SHOW))); ax.set_yticklabels([nm(m) for m in SHOW] if ylab else [])
    ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels([nm(v, "variants") for _, v in cl], rotation=90, fontsize=FS["tick"] - 1)
    sx = ax.secondary_xaxis("top"); sx.set_xticks([g * nv + (nv - 1) / 2 for g in range(len(profs))])
    sx.set_xticklabels([wrap(nm(p, "profiles"), 10) for p in profs], fontsize=FS["annot"]); sx.tick_params(length=0)
    ax.grid(False); cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label(label)
    if ylab: bold_label(ax)
    panel(ax, letter, title)
heat(axs[0, 0], R, CELLS, nm("loss", "metrics"), "a", "Risk per cell", True)
heat(axs[0, 1], Mx, CB, nm("miss_at_alpha", "metrics"), "b", nm("miss_at_alpha", "metrics"), False)
ax = axs[1, 0]; seg = []
for sc in SCEN:
    x = a[a.scenario == sc]; parts = [x] if sm == "pool" else [x[x.seed == s] for s in seeds]
    vals = []
    for g in parts:
        c = cells5(g, "pooled", "pool"); vals.append(c[c.index.get_level_values(1).isin(MENU)].groupby(level=0).max())
    seg.append(pd.concat(vals, axis=1).mean(1).rename(sc))
W = pd.concat(seg, axis=1); off = np.linspace(-0.36, 0.36, len(SHOW)); xi = np.arange(len(SCEN))
for i, m in enumerate(SHOW):
    hl = m == FULL
    ax.scatter(xi + off[i], W.loc[m, SCEN], marker=MARK[m], color=COL[m], s=90 if hl else 28, edgecolor="k" if hl else "none", lw=0.6, zorder=4 if hl else 3)
for x in xi[1:]: ax.axvline(x - 0.5, color="0.8", lw=0.8)
ax.set_xticks(xi); ax.set_xticklabels([two_line(nm(s, "scenarios")) for s in SCEN]); ax.set_xlim(-0.5, len(SCEN) - 0.5); ax.grid(axis="x", visible=False)
ax.set_ylabel(nm("menu_worst", "metrics")); panel(ax, "c", "Worst case per scenario")
ax = axs[1, 1]; A6 = pd.read_csv(track(ROOT / "results/final/item256_seeds0-4/a6_metrics_pooled_seeds0-4.csv"))
groups = sorted(A6.group.astype(str).unique()); CHECKS["a6_groups"] = groups
if "nominal" not in groups: stop(f"a6 groups are {groups}: no 'nominal' group - paste this line")
g = A6[A6.group == "nominal"].set_index("method"); fp = [c for c in A6.columns if c.startswith("TPR@FPR")]
xs = [int(c.replace("TPR@FPR", "")) for c in fp]; sc_ = 100 if g[fp].max().max() <= 1 else 1
for m in SHOW:
    if m not in g.index: stop(f"{m} missing in a6_metrics_pooled_seeds0-4.csv")
    hl = m == FULL
    ax.plot(xs, sc_ * g.loc[m, fp].astype(float).to_numpy(), marker=MARK[m], color=COL[m], lw=2.2 if hl else 1.0,
            ms=9 if hl else 5, mec="k" if hl else COL[m], mew=0.6, zorder=4 if hl else 3)
ax.set_xticks(xs); ax.set_xlabel("False-positive rate (%)"); ax.set_ylabel("True-positive rate (%)")
panel(ax, "d", "TPR at fixed FPR (nominal)")
h = method_handles(SHOW); rows_legend(fig, [h[:6], h[6:]])
CHECKS["worst_per_scenario"] = W.round(4).rename(index=nm).to_dict()
save(fig, "fig04_failure")
