from common import *
print("[fig03] main comparison"); style()
a, how, sm, L, per, J0, seeds = load_main()
B, PT = boot("results/ablation_test", "primary", FULL, BASE10)
for k in METRICS:
    b = J0[f"best10_{k}"]; mine = B[b["method"]][k]
    CHECKS[f"ci_vs_stats_{k}"] = {"method": b["method"], "stats_py": [b["lo"], b["hi"]], "figure": [mine[1], mine[2]]}
    if abs(mine[1] - b["lo"]) > 0.006 or abs(mine[2] - b["hi"]) > 0.006: stop(f"bootstrap CI for {b['method']} {k} {mine} vs stats.py {b['lo']},{b['hi']}")
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained")
y = np.arange(len(SHOW))[::-1]; iF = SHOW.index(FULL)
for ax, k, letter in zip([axs[0, 0], axs[0, 1], axs[1, 0]], METRICS, "abc"):
    pts = {m: np.array([per[s][m][k] for s in seeds]) for m in SHOW}
    lo = min(min(p.min(), L[m][k]) for m, p in pts.items()); hi = max(max(p.max(), L[m][k]) for m, p in pts.items()); pad = 0.06 * (hi - lo)
    ax.axhspan(y[iF] - 0.5, y[iF] + 0.5, color=COL[FULL], alpha=0.10, lw=0)
    ax.axvline(L[FULL][k], color=COL[FULL], ls="--", lw=0.9)
    for yi, m in zip(y, SHOW):
        hl = m == FULL
        ax.hlines(yi, pts[m].min(), pts[m].max(), color=COL[m], lw=1.4, alpha=0.8)
        ax.scatter(pts[m], np.full(len(seeds), yi), s=7, color="0.3", zorder=4, lw=0)
        ax.scatter([L[m][k]], [yi], marker=MARK[m], s=130 if hl else 50, color=COL[m], edgecolor="k", lw=0.6, zorder=5)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.6, len(SHOW) - 0.4)
    ax.set_yticks(y); ax.set_yticklabels([nm(m) for m in SHOW]); bold_label(ax)
    value_column(ax, y, [L[m][k] for m in SHOW], iF)
    ax.set_xlabel(nm("loss", "metrics")); panel(ax, letter, nm(k, "metrics"))
bb = min(BASE10, key=lambda m: L[m]["menu_worst"])
CHECKS["gap_worst_best_baseline"] = {"baseline": nm(bb), "gap": L[bb]["menu_worst"] - L[FULL]["menu_worst"]}
ax = axs[1, 1]; yy = np.arange(len(BASE10))[::-1]
for yi, m in zip(yy, BASE10):
    for k, (mm, cc, of) in DMK.items():
        dv, lo_, hi_ = B[m][k]; sig = lo_ > 0 or hi_ < 0
        ax.errorbar(dv, yi + of, xerr=[[dv - lo_], [hi_ - dv]], fmt=mm, color=cc, mfc=cc if sig else "white", ms=5, lw=1, capsize=2)
ax.axvline(0, color="k", lw=0.8); ax.set_ylim(-0.6, len(BASE10) - 0.4)
ax.set_yticks(yy); ax.set_yticklabels([nm(m) for m in BASE10])
ax.set_xlabel(f"Risk difference (baseline {MINUS} {nm(FULL)})\n> 0: {nm(FULL)} better"); panel(ax, "d", "Paired difference, 95% CI")
rows_legend(fig, [diff_handles() + [Line2D([], [], marker="o", color="0.3", ls="", ms=3, label="Single seed"),
                                    Line2D([], [], color="0.3", lw=1.4, label="Range over seeds")]])
CHECKS["levels_5seed"] = {nm(m): L[m] for m in SHOW}; CHECKS["paired_diff_seed0"] = {nm(m): B[m] for m in BASE10}
save(fig, "fig03_main")
