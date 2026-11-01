from common import *
import matplotlib.ticker as mt
print("[fig09] overhead and post-quantum signatures"); style()
SZ = [256, 2676]
def lv0(sz, crit):
    a = agg_episodes(f"results/final/item{sz}", [0], ["method", "profile", "variant"], ["loss"], SHOW)
    if set(SHOW) - set(a.method): stop(f"item{sz}: methods missing {set(SHOW) - set(a.method)}")
    A, _ = anchors(crit); tried = {}
    for how in ("pooled", "scenmean"):
        L = levels5(a, how, "pool"); tried[how] = max(abs(L[m][k] - v) for m, k, v in A)
        if tried[how] < 5e-4: CHECKS[f"levels_item{sz}"] = {"definition": how, "anchor_err": tried[how]}; return L
    stop(f"item{sz}: no level definition reproduces {crit}: {tried}")
LV = {256: lv0(256, "docs/criteria_test_item256.json"), 2676: lv0(2676, "docs/criteria_test_item2676.json")}
A6 = {}
need = {"group", "method", "bytes_per_decision", "deadline_met", "probes_per_HO", "TPR@FPR10"}
for sz in SZ:
    f = ROOT / f"results/final/item{sz}/a6_metrics_pooled.csv"
    if not f.exists(): stop(f"{f.relative_to(ROOT)} not found")
    d = pd.read_csv(track(f))
    if need - set(d.columns): stop(f"{f.name}: missing {sorted(need - set(d.columns))}; has {list(d.columns)}")
    d["group"] = d.group.astype(str)
    if not {"nominal", "menu_mean"} <= set(d.group): stop(f"{f.name}: groups {sorted(set(d.group))}")
    if d.duplicated(["group", "method"]).any(): stop(f"{f.name}: duplicate (group, method) rows")
    d = d.set_index(["group", "method"])
    for g in ("nominal", "menu_mean"):
        miss = set(SHOW) - set(d.loc[g].index)
        if miss: stop(f"{f.name}: group {g} methods missing {miss}")
    A6[sz] = d
def val(sz, g, m, col): return float(A6[sz].loc[(g, m), col])
y = np.arange(len(SHOW))[::-1]; iF = SHOW.index(FULL); h = 0.38; out = {}
afs = plt.rcParams["xtick.labelsize"]; afs = (afs if isinstance(afs, (int, float)) else 10) * 0.9
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained")
# (a) bytes per decision, nominal, log axis; zero-byte methods labelled
ax = axs[0, 0]; band(ax, y, iF)
V = {sz: [val(sz, "nominal", m, "bytes_per_decision") for m in SHOW] for sz in SZ}
pos = [x for sz in SZ for x in V[sz] if x > 0]
if not pos: stop("bytes_per_decision: no positive values")
x0 = 10 ** np.floor(np.log10(min(pos))); x1 = 2.5 * max(pos)
ax.set_xscale("log"); ax.set_xlim(x0, x1)
for j, sz in enumerate(SZ):
    yy = y + (0.5 - j) * h
    ax.barh(yy, [max(x - x0, 0) for x in V[sz]], h, left=x0, color=[COL[m] for m in SHOW], edgecolor="k", lw=0.4,
            hatch=None if j == 0 else "////", alpha=1 if j == 0 else 0.55)
    for yi, x in zip(y, V[sz]):
        if x <= 0 and j == 0: ax.text(x0 * 1.15, yi, nm("zero_bytes", "fig_labels"), va="center", ha="left", fontsize=afs, color="0.25")
    out.setdefault("bytes_per_decision_nominal", {})[sz] = dict(zip(SHOW, V[sz]))
ax.xaxis.set_major_locator(mt.LogLocator(base=10)); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_minor_formatter(mt.NullFormatter())
rows_axis(ax, y, True); ax.set_xlabel(nm("bytes", "fig_labels")); ax.grid(axis="y", visible=False); panel(ax, "a", nm("t_bytes", "fig_labels"))
# (b) TPR at 10% FPR under attack (menu_mean group), percent
ax = axs[0, 1]; band(ax, y, iF)
T = {sz: [100 * val(sz, "menu_mean", m, "TPR@FPR10") for m in SHOW] for sz in SZ}
for j, sz in enumerate(SZ):
    ax.barh(y + (0.5 - j) * h, T[sz], h, color=[COL[m] for m in SHOW], edgecolor="k", lw=0.4,
            hatch=None if j == 0 else "////", alpha=1 if j == 0 else 0.55)
    out.setdefault("tpr_at_fpr10_menu_mean_pct", {})[sz] = dict(zip(SHOW, T[sz]))
ax.set_xlim(0, 1.08 * max(max(T[s]) for s in SZ))
rows_axis(ax, y, False); ax.set_xlabel(nm("tpr10", "fig_labels")); ax.grid(axis="y", visible=False); panel(ax, "b", nm("t_tpr10", "fig_labels"))
# table-only quantities kept for Table VI
for col in ("deadline_met", "probes_per_HO"):
    out[f"{col}_nominal"] = {sz: {m: val(sz, "nominal", m, col) for m in SHOW} for sz in SZ}
for ax, k, letter, lab in [(axs[1, 0], "menu_worst", "c", True), (axs[1, 1], "menu_mean", "d", False)]:
    band(ax, y, iF); v0 = [LV[256][m][k] for m in SHOW]; v1 = [LV[2676][m][k] for m in SHOW]
    dumbbell(ax, y, v0, v1); rows_axis(ax, y, lab); ax.set_xlabel(nm("loss", "metrics")); panel(ax, letter, nm(k, "metrics"))
    out[k] = {m: [a, b] for m, a, b in zip(SHOW, v0, v1)}
from matplotlib.legend_handler import HandlerTuple
H0 = (Patch(fc="0.6", ec="k"), Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", mew=1.1, ms=7))
H1 = (Patch(fc="0.6", ec="k", hatch="////", alpha=0.55), Line2D([], [], marker="o", ls="", color="0.3", mec="k", mew=0.4, ms=7))
fig.legend(handles=[H0, H1], labels=[nm(256, "item_sizes"), nm(2676, "item_sizes")], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)},
           loc="outside upper center", ncol=2, frameon=False, handlelength=3.2, columnspacing=3.0)
for sz in SZ:
    print(f"  item{sz}: {FULL} bytes {val(sz,'nominal',FULL,'bytes_per_decision'):,.0f} | TPR@FPR10 {100*val(sz,'menu_mean',FULL,'TPR@FPR10'):.1f}% | "
          f"worst {LV[sz][FULL]['menu_worst']:.4f} mean {LV[sz][FULL]['menu_mean']:.4f}")
CHECKS["overhead_and_pqc_seed0"] = out
save(fig, "fig09_overhead")
