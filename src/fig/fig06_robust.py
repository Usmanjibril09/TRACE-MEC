from common import *
print("[fig06] robustness of the main result"); style()
f4 = sorted((ROOT / "docs/pathfold").glob("criteria_fold4*_s0.json"))
if len(f4) != 1: stop(f"expected one fold-4 criteria file, found {[p.name for p in f4]}")
EV = {"test": "docs/criteria_test_item256_seeds0-4.json", **{f"fold{k}": f"docs/pathfold/criteria_fold{k}_s0.json" for k in range(4)},
      "fold4": str(f4[0].relative_to(ROOT)), "gate_fast": "docs/gate/criteria_gate_fast_seeds0-4.json",
      "gate_typical": "docs/gate/criteria_gate_typical_seeds0-4.json", "pqc": "docs/criteria_test_item2676.json"}
GR = [["test"], ["fold0", "fold1", "fold2", "fold3", "fold4"], ["gate_fast", "gate_typical"], ["pqc"]]
J = {}
for e, rel in EV.items():
    J[e] = json.load(open(track(ROOT / rel)))
    miss = [k for k in ["C1", "C2", "C3", "C4"] + [f"best10_{m}" for m in METRICS] + [f"trace_lstm_{m}" for m in METRICS] if k not in J[e]]
    if miss: stop(f"{rel}: keys missing {miss}")
CHECKS["criteria_pass_C1_C4"] = {e: [bool(J[e][c]) for c in ["C1", "C2", "C3", "C4"]] for e in EV}
Y, pos = {}, 0.0
for g in GR:
    for e in g: Y[e] = pos; pos += 1
    pos += 0.7
top = pos - 1.7; Y = {e: top - v for e, v in Y.items()}
seps = [(Y[g[-1]] + Y[GR[i + 1][0]]) / 2 for i, g in enumerate(GR[:-1])]
fig, axs = plt.subplots(2, 2, figsize=SIZE["double"], layout="constrained"); used, out = [], {}
for ax, k, letter, lab in [(axs[0, 0], "nominal_mean", "a", True), (axs[0, 1], "menu_mean", "b", False), (axs[1, 0], "menu_worst", "c", True)]:
    for e in EV:
        b = J[e][f"best10_{k}"]; t = J[e][f"trace_lstm_{k}"]; m = b["method"]; y = Y[e]
        if m not in COL: stop(f"{EV[e]}: best baseline {m} has no colour (not a shown method)")
        if m not in used: used.append(m)
        ax.hlines(y, min(b["full"], b["level"]), max(b["full"], b["level"]), color="0.6", lw=1.2, zorder=2)
        ax.scatter(b["level"], y, marker=MARK[m], color=COL[m], s=55, edgecolor="k", lw=0.4, zorder=3)
        ax.scatter(t["abl"], y, marker=MARK[LSTMV], color=COL[LSTMV], s=55, edgecolor="k", lw=0.4, zorder=4)
        ax.scatter(b["full"], y, marker=MARK[FULL], color=COL[FULL], s=150, edgecolor="k", lw=0.6, zorder=5)
        out.setdefault(e, {})[k] = {"full": b["full"], "best": [m, b["level"]], "lstm_variant": t["abl"]}
    for s_ in seps: ax.axhline(s_, color="0.75", lw=0.8, ls="--")
    ax.set_yticks(list(Y.values())); ax.set_yticklabels([nm(e, "evaluations") for e in Y] if lab else [])
    ax.set_ylim(-0.6, top + 0.6); ax.margins(x=0.08); ax.set_xlabel(nm("loss", "metrics")); panel(ax, letter, nm(k, "metrics"))
ax = axs[1, 1]
for e in EV:
    for k, (mm, cc, of) in DMK.items():
        b = J[e][f"best10_{k}"]; dv = b["level"] - b["full"]; lo, hi = b["lo"], b["hi"]; sig = lo > 0 or hi < 0
        ax.errorbar(dv, Y[e] + of, xerr=[[max(dv - lo, 0)], [max(hi - dv, 0)]], fmt=mm, color=cc, mfc=cc if sig else "white", ms=5, lw=1, capsize=2)
        out[e][k]["ci"] = [lo, hi]
for s_ in seps: ax.axhline(s_, color="0.75", lw=0.8, ls="--")
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(list(Y.values())); ax.set_yticklabels([]); ax.set_ylim(-0.6, top + 0.6)
ax.set_xlabel(f"Best baseline {MINUS} {nm(FULL)} (risk)\n> 0: {nm(FULL)} better"); panel(ax, "d", "Paired difference, 95% CI")
rows_legend(fig, [method_handles([FULL, LSTMV] + used), diff_handles()])
CHECKS["per_evaluation"] = out
save(fig, "fig06_robust")
