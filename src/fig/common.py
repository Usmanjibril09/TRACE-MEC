"""Shared figure helpers. Every number comes from result files; every label from config/names.yaml."""
import ast, hashlib, json, re, subprocess, sys, textwrap, pathlib as P
import numpy as np, pandas as pd, yaml
import pyarrow as pa, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = P.Path(__file__).resolve().parents[2]
def stop(msg): print(f"STOP: {msg}", flush=True); sys.exit(2)
_d = subprocess.run(["git", "status", "--porcelain", "--", "src", "config"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
if _d: stop(f"uncommitted changes in src/ or config/ - commit before drawing:\n{_d}")
GIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
REG = yaml.safe_load(open(ROOT / "config/names.yaml"))
CFG = yaml.safe_load(open(ROOT / "config/config.yaml"))
SRC, CHECKS = set(), {}
def track(f): f = P.Path(f); SRC.add(f.resolve()); return f

SCEN = list(REG["scenarios"]); VARS = list(REG["variants"]); TD = "timeDelayAttack"
MENU = [p for p, d in REG["profiles"].items() if d["report"] == "main" and p != "nominal"]
CELLS = [("nominal", v) for v in VARS] + [(p, v) for p in MENU for v in VARS]
METRICS = ["nominal_mean", "menu_mean", "menu_worst"]
FULL, LSTMV, RND = "robust_v1+probe", "lstm_v1+probe", "robust_v1+probe_rnd"
SHOW = [m for m, d in REG["methods"].items() if d["report"] in ("main", "variant", "secondary")]
BASE6 = ast.literal_eval(re.search(r"^BASELINES\s*=\s*(\[.*?\])", (ROOT / "src/stats.py").read_text(), re.M).group(1))
BASE10 = BASE6 + [m for m in SHOW if REG["methods"][m]["report"] == "secondary"]
if not set(BASE6) <= set(SHOW): stop(f"stats.py BASELINES {BASE6} not all in registry")

def nm(code, kind="methods", form="short"):
    try: return REG[kind][code][form]
    except KeyError: stop(f"'{code}' missing from names.yaml [{kind}] - add it, no fallback to codes")

PAL = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#332288", "#000000", "#882255", "#44AA99", "#999933", "#888888"]
COL = {FULL: "#D55E00", LSTMV: "#F28E2B", RND: "#7F3C8D"}
for m, c in zip([m for m in SHOW if m not in COL], PAL): COL[m] = c
MARK = dict(zip(SHOW, ["*", "P", "o", "s", "^", "v", "D", "<", ">", "p", "h", "X"])); MARK[RND] = "d"
SCOL = dict(zip(SCEN, ["#4477AA", "#66CCEE", "#228833", "#AA3377"]))
FS = dict(title=15, label=12, tick=10, legend=10, annot=9)   # canvas pt; printed = canvas x SCALE
SCALE = 0.7
SIZE = {"single": (3.5 / SCALE, 3.3 / SCALE), "double": (7.16 / SCALE, 5.2 / SCALE)}

def style():
    try: fm.findfont("Times New Roman", fallback_to_default=False)
    except Exception as e: stop(f"Times New Roman not found: {e}")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
        "axes.titlesize": FS["title"], "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False, "legend.handlelength": 1.5, "xtick.major.size": 3, "ytick.major.size": 3, "axes.titlepad": 5})
def panel(ax, letter, text, pad=4): ax.set_title(f"({letter}) {text}", loc="center", fontsize=FS["title"], pad=pad)
def bold_label(ax, axis="y"):
    for t in (ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()):
        if t.get_text() == nm(FULL): t.set_fontweight("bold"); t.set_color(COL[FULL])
def find_dict_with(d, key):
    if isinstance(d, dict):
        if key in d: return d
        for v in d.values():
            r = find_dict_with(v, key)
            if r is not None: return r
    return None

def seeds_in(d):
    s = sorted({int(re.search(r"_s(\d+)\.parquet$", f.name).group(1)) for f in (ROOT / d).glob(f"episodes_{SCEN[0]}_s*.parquet")})
    miss = [f"{sc}_s{k}" for sc in SCEN for k in s if not (ROOT / d / f"episodes_{sc}_s{k}.parquet").exists()]
    if not s or miss: stop(f"{d}: seeds {s}, missing {miss}")
    return s

def agg_episodes(d, seeds, keys, vals, methods):
    out = []
    for sc in SCEN:
        for s in seeds:
            f = track(ROOT / d / f"episodes_{sc}_s{s}.parquet")
            t = pq.read_table(f, columns=list(dict.fromkeys(keys + vals)), filters=[("method", "in", list(methods))])
            for v in vals:
                if pa.types.is_boolean(t.schema.field(v).type):
                    t = t.set_column(t.schema.get_field_index(v), v, t[v].cast(pa.int64()))
            t = t.append_column("one", pa.array(np.ones(t.num_rows, dtype=np.int64)))
            g = t.group_by(keys).aggregate([(v, "sum") for v in vals + ["one"]]).to_pandas()
            g = g.rename(columns={f"{v}_sum": v for v in vals + ["one"]}); g["scenario"] = sc; g["seed"] = s
            out.append(g); print(f"  read {f.relative_to(ROOT)} ({t.num_rows:,} rows kept)", flush=True)
    return pd.concat(out, ignore_index=True)

def _metrics(cell, cols):
    """cell: (R, K) cell risks; cols: K rows of method/profile/variant(/scenario). Scenario columns are averaged."""
    cols = cols.reset_index(drop=True)
    if "scenario" in cols.columns:
        key = list(zip(cols.method, cols.profile, cols.variant)); uk = list(dict.fromkeys(key)); ix = {k: i for i, k in enumerate(uk)}
        M = np.zeros((len(key), len(uk))); M[np.arange(len(key)), [ix[k] for k in key]] = 1.0; M /= M.sum(0, keepdims=True)
        cell = cell @ M; cols = pd.DataFrame(uk, columns=["method", "profile", "variant"])
    out = {}
    for m in cols.method.unique():
        a = np.where((cols.method == m) & (cols.profile == "nominal"))[0]
        b = np.where((cols.method == m) & cols.profile.isin(MENU))[0]
        t = np.where((cols.method == m) & cols.profile.isin(MENU) & (cols.variant == TD))[0]
        if len(a) != len(VARS) or len(b) != len(MENU) * len(VARS): stop(f"{m}: {len(a)} nominal / {len(b)} menu cells")
        out[m] = {"nominal_mean": cell[:, a].mean(1), "menu_mean": cell[:, b].mean(1), "menu_worst": cell[:, b].max(1), "td_worst": cell[:, t].max(1)}
    return out

def cells5(a, how, sm):
    def one(x):
        keys = ["method", "profile", "variant"] + (["scenario"] if how == "scenmean" else [])
        g = x.groupby(keys)[["loss", "one"]].sum(); c = g["loss"] / g["one"]
        return c.groupby(level=[0, 1, 2]).mean() if how == "scenmean" else c
    if sm == "pool": return one(a)
    return pd.concat([one(a[a.seed == s]) for s in sorted(a.seed.unique())], axis=1).mean(1)
def _cellvec(c):
    c = c[c.index.get_level_values(1).isin(["nominal"] + MENU)]
    cols = c.index.to_frame(index=False); cols.columns = ["method", "profile", "variant"]
    return _metrics(c.to_numpy()[None, :], cols)
def levels5(a, how, sm):
    if sm == "pool":
        M = _cellvec(cells5(a, how, "pool")); return {m: {k: float(M[m][k][0]) for k in METRICS} for m in M}
    per = [_cellvec(cells5(a[a.seed == s], how, "pool")) for s in sorted(a.seed.unique())]
    return {m: {k: float(np.mean([q[m][k][0] for q in per])) for k in METRICS} for m in per[0]}

def anchors(rel):
    J = json.load(open(track(ROOT / rel))); A = []
    for k in METRICS:
        b = J[f"best10_{k}"]; A += [(FULL, k, b["full"]), (b["method"], k, b["level"])]
        if f"trace_lstm_{k}" in J: A.append((LSTMV, k, J[f"trace_lstm_{k}"]["abl"]))
    return A, J
def load_main():
    """Test item 256 B, all seeds, shown methods; level definition detected against committed criteria JSONs."""
    seeds = seeds_in("results/final/item256")
    a = agg_episodes("results/final/item256", seeds, ["method", "profile", "variant"], ["loss"], SHOW)
    A5, _ = anchors("docs/criteria_test_item256_seeds0-4.json"); tried = {}
    for how in ("pooled", "scenmean"):
        for sm in ("avg", "pool"):
            L = levels5(a, how, sm); tried[f"{how}/{sm}"] = max(abs(L[m][k] - v) for m, k, v in A5)
            if tried[f"{how}/{sm}"] < 5e-4: break
        else: continue
        break
    else: stop(f"no level definition reproduces criteria_test_item256_seeds0-4.json; max errors {tried}")
    A0, J0 = anchors("docs/criteria_test_item256.json"); L0 = levels5(a[a.seed == 0], how, "pool")
    e0 = max(abs(L0[m][k] - v) for m, k, v in A0)
    if e0 >= 5e-4: stop(f"seed-0 levels differ from criteria_test_item256.json by {e0:.5f}")
    per = {s: levels5(a[a.seed == s], how, "pool") for s in seeds}
    CHECKS.update(level_definition=f"{how}/{sm}", anchor_err_5seed=tried[f"{how}/{sm}"], anchor_err_seed0=e0, anchors_pass=True, seeds=seeds)
    print(f"  level definition {how}/{sm}: reproduces criteria JSONs (max err {tried[f'{how}/{sm}']:.1e} / {e0:.1e})", flush=True)
    return a, how, sm, L, per, J0, seeds

def boot(cdir, setting, ref, comps, B=2000, seed=0):
    """Paired cluster bootstrap by (scenario, vehicle) on clusters files; levels must reproduce levels_s0.csv."""
    meths = [ref] + comps; fr = []
    for sc in SCEN:
        f = track(ROOT / cdir / f"clusters_{sc}_s0.parquet")
        fr.append(pq.read_table(f, columns=["setting", "scenario", "cluster", "method", "profile", "variant", "loss", "n"],
                  filters=[("setting", "=", setting), ("method", "in", meths), ("profile", "in", ["nominal"] + MENU)]).to_pandas())
    d = pd.concat(fr, ignore_index=True)
    miss = set(meths) - set(d.method)
    if miss: stop(f"{cdir} setting {setting}: methods missing {miss}")
    d["cid"] = d.scenario.astype(str) + "|" + d.cluster.astype(str)
    L = pd.read_csv(track(ROOT / cdir / "levels_s0.csv")); L = L[L.setting == setting].set_index("method")
    found, cand = None, {}
    for lossdef in ("mean", "sum"):
        for how in ("pooled", "scenmean"):
            keys = ["method", "profile", "variant"] + (["scenario"] if how == "scenmean" else [])
            ps = d.assign(S=d.loss * d.n if lossdef == "mean" else d.loss).pivot_table(index="cid", columns=keys, values="S", aggfunc="sum", fill_value=0.0)
            pn = d.pivot_table(index="cid", columns=keys, values="n", aggfunc="sum", fill_value=0.0).reindex(columns=ps.columns)
            cols = ps.columns.to_frame(index=False)
            pt = _metrics((ps.values.sum(0) / pn.values.sum(0))[None, :], cols)
            err = max(abs(pt[m][k][0] - L.loc[m, k]) for m in meths if m in L.index for k in METRICS)
            cand[f"{lossdef}/{how}"] = err
            if err < 5e-4: found = (lossdef, how, ps, pn, cols, pt); break
        if found: break
    if not found: stop(f"{cdir} {setting}: cluster levels do not reproduce levels_s0.csv; max errors {cand}")
    lossdef, how, ps, pn, cols, pt = found
    rng = np.random.default_rng(seed); C = len(ps)
    W = rng.multinomial(C, np.full(C, 1.0 / C), size=B).astype(float)
    WS, WN = W @ ps.values, W @ pn.values
    bt = _metrics(np.divide(WS, WN, out=np.full_like(WS, np.nan), where=WN > 0), cols)
    res = {m: {k: (float(pt[m][k][0] - pt[ref][k][0]), *map(float, np.nanpercentile(bt[m][k] - bt[ref][k], [2.5, 97.5]))) for k in METRICS} for m in comps}
    CHECKS[f"boot[{cdir}|{setting}]"] = dict(definition=f"{lossdef}/{how}", clusters=C, B=B, rng_seed=seed, levels_err=cand[f"{lossdef}/{how}"])
    print(f"  bootstrap {cdir}/{setting}: {C} clusters, B={B}, levels reproduce levels_s0.csv", flush=True)
    return res, {m: {k: float(pt[m][k][0]) for k in METRICS} for m in meths}

def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def save(fig, stem):
    out = ROOT / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None}); fig.savefig(out / f"{stem}.png", dpi=600)
    side = {"figure": stem, "git": GIT, "python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
            "checks": CHECKS, "canvas_in": [float(v) for v in fig.get_size_inches()], "printed_pt": {k: round(v * SCALE, 2) for k, v in FS.items()}, "sources": {str(p.relative_to(ROOT)): _md5(p) for p in sorted(SRC)}}
    (out / f"{stem}.json").write_text(json.dumps(side, indent=1, default=str))
    print(f"SAVED figures/{stem}.pdf/.png/.json ({len(SRC)} source files hashed)", flush=True)

import matplotlib.ticker as mticker
MINUS = "\u2212"
def two_line(s): return s.replace("-", "\n", 1)
def wrap(s, w=10): return "\n".join(textwrap.wrap(s, w, break_long_words=False)) or s
def fmt_g(ax, axis="x"): (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
def method_handles(ms, line=False):
    return [Line2D([], [], marker=MARK[m], color=COL[m], ls="-" if line else "", lw=1.8, mec="k" if m == FULL else COL[m],
                   mew=0.6, ms=9 if m == FULL else 6, label=nm(m)) for m in ms]
DMK = {"nominal_mean": ("o", "#0072B2", 0.25), "menu_mean": ("s", "#009E73", 0.0), "menu_worst": ("D", COL[FULL], -0.25)}
def diff_handles():
    return [Line2D([], [], marker=v[0], color=v[1], ls="", ms=6, label=nm(k, "metrics")) for k, v in DMK.items()] + \
           [Line2D([], [], marker="o", color="0.3", mfc="white", ls="", ms=6, label="Open marker: 95% CI includes 0")]
def value_column(ax, y, vals, bold=None):
    t = ax.twinx(); t.set_ylim(ax.get_ylim()); t.set_yticks(y); t.set_yticklabels([f"{v:.3f}" for v in vals])
    t.tick_params(axis="y", length=0, labelsize=FS["tick"]); t.grid(False)
    if bold is not None:
        lab = t.get_yticklabels()[bold]; lab.set_fontweight("bold"); lab.set_color(COL[FULL])
    return t
def rows_legend(fig, rows):
    n = max(map(len, rows)); blank = Line2D([], [], ls="", marker="", label=" ")
    rows = [list(r) + [blank] * (n - len(r)) for r in rows]
    fig.legend(handles=[r[j] for j in range(n) for r in rows], ncol=n, loc="outside upper center",
               fontsize=FS["legend"], handletextpad=0.4, columnspacing=1.4)

def band(ax, y, i): ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=COL[FULL], alpha=0.10, lw=0)
def rows_axis(ax, y, lab):
    ax.set_yticks(y); ax.set_yticklabels([nm(m) for m in SHOW] if lab else []); ax.set_ylim(-0.6, len(y) - 0.4)
    if lab: bold_label(ax)
def dumbbell(ax, y, v0, v1):
    for yi, m, a0, a1 in zip(y, SHOW, v0, v1):
        ax.annotate("", xy=(a1, yi), xytext=(a0, yi), arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=1.2, shrinkA=4, shrinkB=4, mutation_scale=9))
        ax.scatter(a0, yi, marker=MARK[m], s=60, facecolor="white", edgecolor=COL[m], lw=1.1, zorder=3)
        ax.scatter(a1, yi, marker=MARK[m], s=60, color=COL[m], edgecolor="k", lw=0.4, zorder=4)
    ax.margins(x=0.08)
