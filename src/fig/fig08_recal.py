from common import *
from matplotlib.colors import TwoSlopeNorm
print("[fig08] honest-share control"); style()
D = "results/ablation_test_a18"; SETS = ["primary", "recal=on"]
LA = pd.read_csv(track(ROOT / f"{D}/levels_s0.csv")); L1 = LA.set_index(["setting", "method"])
if not set(SETS) <= set(LA.setting): stop(f"{D}/levels_s0.csv settings {sorted(LA.setting.unique())}")
miss = [(s, m) for s in SETS for m in SHOW if (s, m) not in L1.index]
if miss: stop(f"{D}/levels_s0.csv missing {miss}")
alpha = float(json.load(open(track(ROOT / f"{D}/meta_{SCEN[0]}_s0.json")))["settings"]["primary"]["alpha"])
y = np.arange(len(SHOW))[::-1]; iF = SHOW.index(FULL); out = {}
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained")
for ax, col, sc_, letter, title, xl, lab in [(axs[0, 0], "honest_restrict", 100, "a", nm("t_honest", "fig_labels"), nm("honest", "fig_labels"), True),
                                             (axs[0, 1], "menu_worst", 1, "b", nm("menu_worst", "metrics"), nm("loss", "metrics"), False),
                                             (axs[1, 0], "menu_mean", 1, "c", nm("menu_mean", "metrics"), nm("loss", "metrics"), True)]:
    v0 = [sc_ * float(L1.loc[("primary", m), col]) for m in SHOW]; v1 = [sc_ * float(L1.loc[("recal=on", m), col]) for m in SHOW]
    band(ax, y, iF)
    if col == "honest_restrict": ax.axvline(100 * alpha, color="k", ls=":", lw=1.1)
    dumbbell(ax, y, v0, v1); rows_axis(ax, y, lab); ax.set_xlabel(xl); panel(ax, letter, title)
    out[col] = {m: [a, b] for m, a, b in zip(SHOW, v0, v1)}
HS = []
for k in range(4):
    h = pd.read_csv(track(ROOT / f"results/pathfold/fold{k}/honest_share.csv")).set_index("method")
    if set(SHOW) - set(h.index) or set(SCEN) - set(h.columns): stop(f"fold{k}/honest_share.csv lacks {set(SHOW) - set(h.index)} / {set(SCEN) - set(h.columns)}")
    HS.append(h)
Z = np.array([[100 * float(HS[k].loc[m, sc]) for sc in SCEN for k in range(4)] for m in SHOW])
ax = axs[1, 1]; a100 = 100 * alpha
im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=a100, vmin=min(Z.min(), a100 - 1), vmax=max(Z.max(), a100 + 1)))
for x in range(4, Z.shape[1], 4): ax.axvline(x - 0.5, color="w", lw=2)
ax.add_patch(plt.Rectangle((-0.5, iF - 0.5), Z.shape[1], 1, fill=False, ec=COL[FULL], lw=1.8))
ax.set_yticks(range(len(SHOW))); ax.set_yticklabels([]); ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels([str(k) for _ in SCEN for k in range(4)])
ax.set_xlabel("Path fold"); sx = ax.secondary_xaxis("top"); sx.set_xticks([4 * i + 1.5 for i in range(len(SCEN))])
sx.set_xticklabels([two_line(nm(s, "scenarios")) for s in SCEN], fontsize=FS["annot"]); sx.tick_params(length=0)
ax.grid(False); cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label(nm("honest", "fig_labels")); panel(ax, "d", nm("t_folds", "fig_labels"))
rows_legend(fig, [[Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", mew=1.1, ms=7, label=nm("primary_thr", "fig_labels")),
                   Line2D([], [], marker="o", ls="", color="0.3", mec="k", mew=0.4, ms=7, label=nm("recal", "fig_labels")),
                   Line2D([], [], color="k", ls=":", lw=1.1, label=nm("target", "fig_labels"))]])
CHECKS["primary_vs_recal"] = out; CHECKS["alpha"] = alpha; CHECKS["honest_share_folds"] = {nm(m): r.tolist() for m, r in zip(SHOW, Z.round(2))}
save(fig, "fig08_recal")
