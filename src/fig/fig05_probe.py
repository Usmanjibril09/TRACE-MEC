from common import *
print("[fig05] deception probe"); style()
def lv(d):
    Lr = pd.read_csv(track(ROOT / f"results/{d}/levels_s0.csv"))
    st = json.load(open(track(ROOT / f"results/{d}/meta_{SCEN[0]}_s0.json")))["settings"]
    rate = {s: float(v["rate"]) for s, v in st.items() if "rate" in v}
    if len(rate) < 3: stop(f"{d}: probe rates in meta settings: {rate}")
    return Lr, rate
LT, rT = lv("ablation_test_a18"); LV, rV = lv("ablation_a18")
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained")
pct = mticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
for ax, k, letter in [(axs[0, 0], "menu_worst", "a"), (axs[0, 1], "menu_mean", "b")]:
    allv = []
    for m in [FULL, RND, LSTMV]:
        for Lr, rate, test in [(LT, rT, True), (LV, rV, False)]:
            s_ = sorted(rate, key=rate.get); x = [rate[s] for s in s_]
            yv = [float(Lr[(Lr.setting == s) & (Lr.method == m)][k].iloc[0]) for s in s_]; allv += yv
            ax.plot(x, yv, marker=MARK[m], color=COL[m], mec=COL[m], ls="-" if test else ":", lw=1.8 if test else 1.0,
                    mfc=COL[m] if test else "white", ms=8 if m == FULL else 6)
    P0 = LT[LT.setting == "primary"].set_index("method"); bb = min(BASE10, key=lambda m: P0.loc[m, k]); yb = float(P0.loc[bb, k])
    rs = sorted(rT.values()); span = rs[-1] - rs[0]
    lo, hi = min(allv + [yb]), max(allv + [yb])
    ax.axhline(yb, color=COL[bb], ls="--", lw=1.1)
    ax.set_xlim(rs[0] - 0.08 * span, rs[-1] + 0.08 * span); ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.14 * (hi - lo))
    ax.text(rs[-1] + 0.06 * span, yb, f"Best baseline ({nm(bb)})", ha="right", va="bottom", fontsize=FS["annot"], color=COL[bb])
    ax.set_xticks(rs); ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Probe rate"); ax.set_ylabel(nm("loss", "metrics")); panel(ax, letter, nm(k, "metrics"))
    CHECKS[f"best_baseline_{k}_test"] = [bb, yb]
seeds = seeds_in("results/final/item256")
e = agg_episodes("results/final/item256", seeds, ["method", "profile", "variant", "x"], ["probed", "probe_pos"], [FULL, RND])
e = e[e.profile == "nominal"].copy(); e["x"] = e.x.astype(int)
grp = [("honest", e.x == 0, nm("honest", "classes"))] + [(v, (e.x == 1) & (e.variant == v), nm(v, "variants")) for v in VARS if v != "baseline"]
ax = axs[1, 0]; xi = np.arange(len(grp)); stats = {}; w = 0.38
for j, m in enumerate([FULL, RND]):
    em = e[e.method == m]; ps, det = [], []
    for g, mask, _ in grp:
        s = em[mask.loc[em.index]].sum(numeric_only=True); ps.append(100 * s.probed / s.one); det.append(100 * s.probe_pos / s.probed)
    stats[nm(m)] = {"probed_pct": dict(zip([g for g, _, _ in grp], ps)), "detect_pct": dict(zip([g for g, _, _ in grp], det)),
                    "overall_probe_rate": float(em.probed.sum() / em.one.sum())}
    xb = xi + (j - 0.5) * w
    ax.bar(xb, det, w, color=COL[m], edgecolor="k", lw=0.4, alpha=1.0 if j == 0 else 0.75)
    for xx, d_, p_ in zip(xb, det, ps): ax.text(xx, d_ + 1.5, f"{p_:.0f}%", ha="center", va="bottom", fontsize=FS["annot"] - 1)
ax.set_xticks(xi); ax.set_xticklabels([wrap(t, 9) for _, _, t in grp]); ax.set_ylim(0, 120); ax.grid(axis="x", visible=False)
ax.text(0.01, 0.98, "Bar labels: share probed", transform=ax.transAxes, va="top", fontsize=FS["annot"])
ax.set_ylabel("Detected among probed (%)"); panel(ax, "c", "Probe outcome")
CHECKS["probe_allocation_test_5seed_nominal"] = stats
ax = axs[1, 1]; sets = sorted(rT, key=rT.get) + [s for s in LT.setting.unique() if s not in rT]
yy = np.arange(len(sets))[::-1]; out = {}
for yi, s in zip(yy, sets):
    res, _ = boot("results/ablation_test_a18", s, FULL, [RND]); out[s] = res[RND]
    for k, (mm, cc, of) in DMK.items():
        dv, lo, hi = res[RND][k]; sig = lo > 0 or hi < 0
        ax.errorbar(dv, yi + of, xerr=[[dv - lo], [hi - dv]], fmt=mm, color=cc, mfc=cc if sig else "white", ms=5, lw=1, capsize=2)
ax.axvline(0, color="k", lw=0.8); ax.set_ylim(-0.6, len(sets) - 0.4)
ax.set_yticks(yy); ax.set_yticklabels([nm(s, "settings") for s in sets])
ax.set_xlabel(f"Risk difference (random {MINUS} VOI)\n> 0: VOI trigger better"); panel(ax, "d", "VOI vs random trigger")
CHECKS["voi_vs_random_test"] = out
rows_legend(fig, [method_handles([FULL, RND, LSTMV], line=True) +
                  [Line2D([], [], color="k", ls="-", lw=1.8, label="Test"), Line2D([], [], color="k", ls=":", marker="o", mfc="white", label="Validation")],
                  diff_handles()])
save(fig, "fig05_probe")
