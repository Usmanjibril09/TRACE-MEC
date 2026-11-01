#!/usr/bin/env python
"""ablation.py (script 5/8): preregistered sensitivity sweeps on Validation (docs/theory.md A12).

model.py is NOT modified. model.smoke() is driven with read-only hooks that capture the Train fit set and
every Validation episode table passed to apply_operating(); the post-simulation layer (operating point +
deception probe) is replayed under each setting. Only the central_ma delay D needs re-simulation.
Abort guards: replay at primary reproduces smoke bit-for-bit; D runs leave all other methods identical;
results/model_smoke + docs/model_smoke.md restored byte-identically; results/frozen_v071 untouched.
  python src/ablation.py --run --report   |   --report (re-tabulate)   |   --restore (after a hard kill)
  python src/ablation.py --final [--resume]   A17: the same A12 sweeps on held-out Test, run once
  python src/ablation.py --a18 | --a18 --final [--resume]   A18: probe rate + online recalibration (Validation, then Test once)
"""
import argparse, copy, hashlib, json, shutil, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if "if __name__" not in (SRC / "model.py").read_text():
    raise SystemExit("ABORT: model.py has no __main__ guard; importing it would run main()")
sys.path.insert(0, str(SRC))
import model as Mo  # noqa: E402

ROOT = Path(Mo.ROOT)
PROBE_E = Mo.probe_e
SCEN = ["InTAS_highway_2", "InTAS_urban_2", "InTAS_highway_7", "InTAS_urban_7"]
MENU = ["adv_burst", "adv_sustained", "adv_early", "adv_suppress_top"]
FULL = "robust_v1+probe"
BASE10 = ["fifo", "recency", "severity", "random", "aoi", "voi_myopic", "beta_rep", "f2md", "central_ma", "lstm_sev"]
TRACE = ["robust_v1+probe", "robust_v1", "robust_v1+probe_rnd", "fifo_rb+probe", "no_transfer_rb+probe",
         "lstm_v1+probe", "lstm_v1"]
KEEP = TRACE + BASE10 + ["f2md_tuned", "ref_full_clean", "ref_full_clean_rb"]
SHOW = ["robust_v1+probe", "robust_v1", "robust_v1+probe_rnd", "lstm_v1+probe", "lstm_sev", "severity", "fifo",
        "voi_myopic", "beta_rep", "f2md", "central_ma"]
COMPARE = [(FULL, b) for b in ["robust_v1", "robust_v1+probe_rnd", "severity", "fifo", "lstm_sev"]] + [("lstm_v1+probe", "lstm_sev")]
ALPHAS, D_LIST, PRIMARY_ALPHA = [0.05, 0.10, 0.20], [1.0, 5.0, 30.0], 0.10
ITEM_SIZES = {256: "unsigned", 320: "ECDSA-P256", 922: "Falcon-512", 2676: "ML-DSA-44"}   # A13
INV = ["no_transfer", "no_transfer_rb"]
ITEM_KEEP = KEEP + ["severity_rb"]
ITEM_COLS = ["session_uid", "ho_idx", "variant", "profile", "method", "x", "p", "action", "loss", "probed", "probe_pos",
             "budget", "n_sent", "n_arrived"]
EXPECTED = {"nominal_mean": 0.2483, "menu_mean": 0.2578, "menu_worst": 0.7279}   # stats.py pooled, criteria_v08
FENCE = "`" * 3
TEST = None   # set only by --final (A17): {"sim": callable, "ref": callable}
SPLIT_LABEL, REPORT_DOC = "Validation", "docs/ablation_validation.md"
EXPECTED_TEST = {"nominal_mean": 0.2548, "menu_mean": 0.2605, "menu_worst": 0.7717}   # A14 seed-0 Test, stats.py
TEST_MIN = {"InTAS_highway_2": 1, "InTAS_urban_2": 7, "InTAS_highway_7": 20, "InTAS_urban_7": 24}   # Test sim minutes (evaluate_seeds.log)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def md5_tree(d):
    d = Path(d)
    return {str(f.relative_to(d)): md5(f) for f in sorted(d.rglob("*")) if f.is_file()} if d.exists() else {}


def fp(out, exclude=None):
    o = out if exclude is None else out[out["method"] != exclude]
    cols = [c for c in ("method", "x", "p", "action", "loss", "probed", "probe_pos") if c in o.columns]
    h = pd.util.hash_pandas_object(o[cols].reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha1(h.tobytes()).hexdigest()


def same_frames(f1, f2):
    a, b = pd.read_parquet(f1), pd.read_parquet(f2)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" and y.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all():
            return False
    return True


class Snapshot:
    def __init__(self, smoke_dir, doc, store):
        self.sd, self.doc, self.store = Path(smoke_dir), Path(doc), Path(store)
        if self.store.exists():
            raise SystemExit(f"ABORT: stale snapshot {self.store} (hard kill?). Run: python src/ablation.py --restore")
        shutil.copytree(self.sd, self.store / "model_smoke")
        shutil.copy2(self.doc, self.store / "model_smoke.md")
        self.h_dir, self.h_doc = md5_tree(self.sd), md5(self.doc)

    def restore(self):
        for f in list(self.sd.rglob("*")):
            if f.is_file() and str(f.relative_to(self.sd)) not in self.h_dir:
                f.unlink()
        base = self.store / "model_smoke"
        for f in base.rglob("*"):
            if f.is_file():
                t = self.sd / f.relative_to(base)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
        shutil.copy2(self.store / "model_smoke.md", self.doc)
        if md5_tree(self.sd) != self.h_dir or md5(self.doc) != self.h_doc:
            raise SystemExit(f"ABORT: snapshot restore mismatch; snapshot kept at {self.store}")

    def drop(self):
        shutil.rmtree(self.store)


class Hooks:
    NAMES = ("fit_operating", "apply_operating", "probe_info", "simulate")

    def __init__(self, faults):
        self.faults, self.orig = faults, {n: getattr(Mo, n) for n in self.NAMES}
        self.reset("base")

    def reset(self, mode):
        self.mode, self.fit, self.items, self.var, self.prof, self.op = mode, None, [], None, None, None

    def _pname(self, prof):
        ex = [n for n, v in self.faults.items() if v == prof]
        if len(ex) == 1:
            return ex[0]
        if isinstance(prof, dict):
            sub = [n for n, v in self.faults.items() if all(prof.get(k) == w for k, w in v.items())]
            if sub:
                return max(sub, key=lambda n: len(self.faults[n]))
        return None

    def install(self):
        o = self.orig

        def fit_operating(fit, PI, M, L, seed):
            self.fit = (fit.copy(), PI.copy(), copy.deepcopy(M), copy.deepcopy(L), seed)
            return o["fit_operating"](fit, PI, M, L, seed)

        def probe_info(*a, **k):
            if len(a) > 3 and a[2] in ("Validation", "Test"):
                self.var = a[3]
            return o["probe_info"](*a, **k)

        def simulate(*a, **k):
            self.prof = self._pname(a[3] if len(a) > 3 else k.get("prof"))
            return o["simulate"](*a, **k)

        def apply_operating(df, PI, OP, L, seed):
            out = o["apply_operating"](df, PI, OP, L, seed)
            if self.prof is None and "profile" not in out.columns:
                raise RuntimeError("ablation hook: cannot identify the fault profile of captured episodes")
            if self.op is None:
                self.op = copy.deepcopy(OP)
            it = {"var": self.var, "prof": self.prof, "fp_all": fp(out), "fp_rest": fp(out, exclude="central_ma")}
            it["fp_inv"] = fp(out[out["method"].isin(INV)])
            if self.mode == "base":
                it.update(df=df, PI=PI)
            elif self.mode == "items":
                it["d"] = out.loc[out["method"].isin(ITEM_KEEP), [c for c in ITEM_COLS if c in out.columns]].copy()
            else:
                it["cma"] = out[out["method"] == "central_ma"].copy()
            self.items.append(it)
            return out

        for n, f in (("fit_operating", fit_operating), ("apply_operating", apply_operating),
                     ("probe_info", probe_info), ("simulate", simulate)):
            setattr(Mo, n, f)

    def uninstall(self):
        for n, f in self.orig.items():
            setattr(Mo, n, f)
        Mo.probe_e = PROBE_E


def settings(M):
    P, O = M["probe"], M["operating"]
    prim = {"alpha": float(O["alpha"]), "sigma": float(P["sigma_primary"]), "t_p": float(P["t_p_primary"]), "kappa": 1.0}
    exp = {"alpha": 0.10, "sigma": 5.0, "t_p": 0.05, "kappa": 1.0}
    if prim != exp or float(Mo.EXT_CFG["ma_delay_s"]) != 5.0:
        raise SystemExit(f"ABORT: config primary {prim}, D={Mo.EXT_CFG['ma_delay_s']} != preregistered {exp}, D=5")
    S = {"primary": dict(prim)}
    for k, vals in (("alpha", ALPHAS), ("sigma", P["sigma_s"]), ("t_p", P["t_p_s"]), ("kappa", P["kappa"])):
        for v in vals:
            if float(v) != prim[k]:
                S[f"{k}={float(v):g}"] = dict(prim, **{k: float(v)})
    return S


def replay(H, s):
    fit, PI, M, L, seed = H.fit
    M2 = copy.deepcopy(M)
    M2["operating"]["alpha"], M2["probe"]["sigma_primary"], M2["probe"]["t_p_primary"] = s["alpha"], s["sigma"], s["t_p"]
    M2["probe"]["rate"] = float(s.get("rate", M["probe"]["rate"]))   # A18 (no-op for A12/A17 settings)
    OP = H.orig["fit_operating"](fit, PI, M2, L, seed)
    borrowed = {}
    for key in ("thr", "cut"):
        miss = sorted(set(H.op.get(key, {})) - set(OP.get(key, {})))
        if miss:
            if key == "thr" and s["alpha"] != PRIMARY_ALPHA:
                raise SystemExit(f"ABORT: thresholds for {miss} are set outside fit_operating; cannot sweep alpha")
            for m in miss:
                OP[key][m] = H.op[key][m]
            borrowed[key] = miss
    k = s["kappa"]
    if k != 1.0:
        Mo.probe_e = lambda q, sigma, kappa, sd: PROBE_E(q, sigma, kappa * k, sd)
    try:
        outs = [(H.orig["apply_operating"](it["df"], it["PI"], OP, L, seed), it["var"], it["prof"]) for it in H.items]
    finally:
        Mo.probe_e = PROBE_E
    return OP, outs, borrowed


def frame(outs):
    parts = []
    for out, var, prof in outs:
        d = out[out["method"].isin(KEEP)]
        add = {}
        if "variant" not in d.columns:
            add["variant"] = var
        if "profile" not in d.columns:
            add["profile"] = prof
        parts.append(d.assign(**add) if add else d)
    return pd.concat(parts, ignore_index=True)


def clusters(sc, uid):
    u = pd.Series(pd.unique(uid))
    t = u.astype(str).str.split("|")
    return uid.map(dict(zip(u, sc + "|" + t.apply(lambda z: "|".join(z[:1] + z[2:])))))


def tabulate(d, sc, setting):
    x, a = d["x"].to_numpy().astype(int), d["action"].to_numpy()
    pr, pp = d["probed"].to_numpy().astype(bool), d["probe_pos"].to_numpy().astype(bool)
    t = pd.DataFrame({"cluster": clusters(sc, d["session_uid"]).to_numpy(), "method": d["method"].to_numpy(),
                      "profile": d["profile"].to_numpy(), "variant": d["variant"].to_numpy(),
                      "loss": d["loss"].to_numpy(dtype=float), "n": 1, "n_x1": x == 1, "miss": (x == 1) & (a == 0),
                      "hr": (x == 0) & (a >= 1), "pr1": pr & (x == 1), "pp1": pp & (x == 1),
                      "pr0": pr & (x == 0), "pp0": pp & (x == 0)})
    g = t.groupby(["cluster", "method", "profile", "variant"], sort=False).sum().reset_index()
    g.insert(0, "setting", setting)
    g.insert(1, "scenario", sc)
    return g


def eval_tab(d, alpha, seed, sc, name):
    e = Mo.eval_alpha(d, alpha, seed)
    e.insert(0, "setting", name)
    e.insert(1, "scenario", sc)
    return e


def lag_stats(items):
    res = {}
    for it in items:
        v = it["var"]
        if v in res or v in (None, "baseline"):
            continue
        g = it["df"][it["df"]["method"] == "robust_v1"]
        if g.empty:
            continue
        q = Mo._probe_q(g[["session_uid", "ho_idx", "x"]], it["PI"])
        k = (q["x"].to_numpy() == 1) & q["has"].to_numpy()
        lag = np.hypot(q["lx"].to_numpy(), q["ly"].to_numpy())[k]
        if lag.size:
            res[v] = {"n": int(lag.size), "median_m": float(np.median(lag)), "p90_m": float(np.quantile(lag, 0.9))}
    return res


def run_scenario(cfg, sc, seed, S, snap, H, out):
    t0, checks = time.time(), {}
    ep = snap.sd / f"episodes_{sc}_s{seed}.parquet"
    say(f"[{sc}] primary smoke with capture hooks")
    H.reset("base")
    H.install()
    try:
        if TEST is None:
            Mo.smoke(cfg, sc, seed)
            old = snap.store / "model_smoke" / ep.name
            checks["fresh smoke reproduces stored v0.9 episodes file"] = bool(same_frames(old, ep)) if old.exists() and ep.exists() else "n/a"
        else:
            TEST["sim"](cfg, sc, seed)
            ok_ref, msg_ref = TEST["ref"](sc, seed)
            checks["fresh Test simulation reproduces A14 seed-0 Test episodes bit-for-bit"] = bool(ok_ref)
            if not ok_ref:
                raise SystemExit(f"ABORT [{sc}]: Test simulation differs from A14 seed-0 episodes ({msg_ref})")
    finally:
        H.uninstall()
        snap.restore()
    if H.fit is None or not H.items:
        raise SystemExit(f"ABORT [{sc}]: hooks captured nothing (smoke internals changed)")
    meta = {"scenario": sc, "seed": seed, "settings": S, "probe": {}, "lag_kappa1": lag_stats(H.items), "op_keys_borrowed": {}}
    tabs, eas = [], []
    for name, s in S.items():
        OP, outs, borrowed = replay(H, s)
        if borrowed:
            meta["op_keys_borrowed"][name] = borrowed
        if name == "primary":
            ok = len(outs) == len(H.items) and all(fp(o) == it["fp_all"] for (o, _, _), it in zip(outs, H.items))
            checks["replay at primary reproduces smoke bit-for-bit"] = ok
            if not ok:
                raise SystemExit(f"ABORT [{sc}]: replay at primary differs from smoke")
        d = frame(outs)
        del outs
        tabs.append(tabulate(d, sc, name))
        eas.append(eval_tab(d, s["alpha"], seed, sc, name))
        P = OP["probe"]
        meta["probe"][name] = {"sigma": P["sigma"], "theta": P["theta"], "d": P["d"], "f": P["f"], "t_p": P["t_p"],
                               "rate_train": OP.get("probe_rate_train", {})}
        say(f"[{sc}] replayed {name}: theta={P['theta']:.1f} m d={P['d']:.3f} f={P['f']:.3f}")
        del d
    keys, rest = [(it["var"], it["prof"]) for it in H.items], [it["fp_rest"] for it in H.items]
    for D in D_LIST:
        if D == 5.0:
            continue
        say(f"[{sc}] re-simulation for central_ma D={D:g} s")
        H.reset("D")
        Mo.EXT_CFG["ma_delay_s"] = D
        H.install()
        try:
            (Mo.smoke if TEST is None else TEST["sim"])(cfg, sc, seed)
        finally:
            H.uninstall()
            Mo.EXT_CFG["ma_delay_s"] = 5.0
            snap.restore()
        ok = [(it["var"], it["prof"]) for it in H.items] == keys and [it["fp_rest"] for it in H.items] == rest
        checks[f"D={D:g}: every other method identical"] = ok
        if not ok:
            raise SystemExit(f"ABORT [{sc}]: D={D:g} changed methods other than central_ma")
        d = frame([(it["cma"], it["var"], it["prof"]) for it in H.items])
        if d.empty:
            raise SystemExit(f"ABORT [{sc}]: no central_ma rows captured")
        tabs.append(tabulate(d, sc, f"D={D:g}"))
        eas.append(eval_tab(d, PRIMARY_ALPHA, seed, sc, f"D={D:g}"))
    H.reset("base")
    pd.concat(tabs, ignore_index=True).to_parquet(out / f"clusters_{sc}_s{seed}.parquet", index=False)
    pd.concat(eas, ignore_index=True).to_csv(out / f"eval_alpha_{sc}_s{seed}.csv", index=False)
    meta.update(checks=checks, runtime_s=round(time.time() - t0), git=git_head())
    (out / f"meta_{sc}_s{seed}.json").write_text(json.dumps(meta, indent=2, default=str))
    for k, v in checks.items():
        say(f"CHECK [{sc}] {k}: {'PASS' if v is True else 'FAIL' if v is False else v}")


def run_items(cfg, sc, seed, snap, H, out):
    """A13: evidence-item size sweep (signature overhead). Full re-simulation + Train refit per size."""
    t0, checks, tabs, eas, trs, thm = time.time(), {}, [], [], [], {}
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit(f"ABORT: evidence_item_bytes={cfg['traces']['evidence_item_bytes']}, preregistered primary 256")
    ep = snap.sd / f"episodes_{sc}_s{seed}.parquet"
    keys_ref = inv_ref = None
    K = ["session_uid", "ho_idx", "variant", "profile"]
    for nb, lab in ITEM_SIZES.items():
        name = f"item={nb}"
        say(f"[{sc}] smoke with evidence_item_bytes={nb} ({lab})")
        c2 = copy.deepcopy(cfg)
        c2["traces"]["evidence_item_bytes"] = nb
        H.reset("items")
        H.install()
        try:
            Mo.smoke(c2, sc, seed)
            if nb == 256:
                old = snap.store / "model_smoke" / ep.name
                checks["item=256 reproduces stored v0.9 episodes file"] = bool(same_frames(old, ep)) if old.exists() and ep.exists() else "n/a"
        finally:
            H.uninstall()
            snap.restore()
        if not H.items:
            raise SystemExit(f"ABORT [{sc}]: hooks captured nothing at {name}")
        keys, inv = [(it["var"], it["prof"]) for it in H.items], [it["fp_inv"] for it in H.items]
        if nb == 256:
            keys_ref, inv_ref = keys, inv
        else:
            ok = keys == keys_ref and inv == inv_ref
            checks[f"{name}: no_transfer and no_transfer_rb identical to item=256"] = ok
            if not ok:
                raise SystemExit(f"ABORT [{sc}]: {name} changed methods that do not transfer evidence")
        parts = []
        for it in H.items:
            add = {k: v for k, v in (("variant", it["var"]), ("profile", it["prof"])) if k not in it["d"].columns}
            parts.append(it["d"].assign(**add) if add else it["d"])
        d = pd.concat(parts, ignore_index=True)
        H.reset("base")
        del parts
        j = d[d.method == "robust_v1"][K + ["action"]].merge(d[d.method == "severity_rb"][K + ["action"]], on=K, suffixes=("", "_sev"))
        thm[name] = float((j["action"] == j["action_sev"]).mean()) if len(j) else float("nan")
        nm = d[d.profile == "nominal"]
        for m in [FULL, "robust_v1", "severity", "fifo", "voi_myopic", "lstm_sev", "central_ma"]:
            g = nm[nm.method == m]
            if len(g) and "budget" in g.columns:
                b, s_ = g["budget"].to_numpy(float), g["n_sent"].to_numpy(float)
                trs.append({"scenario": sc, "setting": name, "method": m, "n": len(g), "budget_mean": b.mean(),
                            "budget_zero": float((b <= 0).mean()), "sent_mean": s_.mean(),
                            "arrived_mean": g["n_arrived"].to_numpy(float).mean(),
                            "binds": float(((b > 0) & (s_ == b)).mean())})
        tabs.append(tabulate(d, sc, name))
        eas.append(eval_tab(d, PRIMARY_ALPHA, seed, sc, name))
        say(f"[{sc}] {name}: robust_v1 == severity_rb actions {thm[name]:.4f}")
        del d, nm, j
    TRd = pd.DataFrame(trs)
    bm = [float(TRd[(TRd.setting == f"item={n}") & (TRd.method == "severity")]["budget_mean"].mean()) for n in ITEM_SIZES]
    ok = all(a > b for a, b in zip(bm, bm[1:]))
    checks["mean budget strictly decreases with item size"] = ok
    if not ok:
        raise SystemExit(f"ABORT [{sc}]: budget does not shrink with item size {bm}; simulator ignores item_bytes")
    checks["Thm 2(b): robust_v1 == severity_rb actions at every size"] = all(v == 1.0 for v in thm.values())
    pd.concat(tabs, ignore_index=True).to_parquet(out / f"items_clusters_{sc}_s{seed}.parquet", index=False)
    pd.concat(eas, ignore_index=True).to_csv(out / f"items_eval_alpha_{sc}_s{seed}.csv", index=False)
    TRd.to_csv(out / f"items_transfer_{sc}_s{seed}.csv", index=False)
    meta = {"scenario": sc, "seed": seed, "sizes": ITEM_SIZES, "thm2b": thm, "budget_mean_severity": bm,
            "checks": checks, "runtime_s": round(time.time() - t0), "git": git_head()}
    (out / f"items_meta_{sc}_s{seed}.json").write_text(json.dumps(meta, indent=2, default=str))
    for k, v in checks.items():
        say(f"CHECK [{sc}] {k}: {'PASS' if v is True else 'FAIL' if v is False else v}")


def _mats(T, s, m):
    t = T[(T.setting == s) & (T.method == m)]
    if t.empty:
        return None
    kw = dict(index="cluster", columns=["profile", "variant"], aggfunc="sum", fill_value=0)
    return t.pivot_table(values="loss", **kw), t.pivot_table(values="n", **kw)


def boot_did(T, s0, s1, A, Bm, B, seed):
    """Difference-in-differences: (B@s1 - B@s0) - (A@s1 - A@s0); positive = B degrades more than A."""
    ms = [_mats(T, s, m) for s, m in ((s0, A), (s1, A), (s0, Bm), (s1, Bm))]
    if any(x is None for x in ms):
        return None
    idx, cols = ms[0][0].index, ms[0][0].columns
    for S_, _ in ms[1:]:
        idx, cols = idx.union(S_.index), cols.intersection(S_.columns)
    prof = np.array([c[0] for c in cols])
    nom, men = prof == "nominal", np.isin(prof, MENU)
    n = len(idx)
    W = np.vstack([np.ones(n), np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=B)]).astype(float)
    mets = []
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for S_, N_ in ms:
            Sa = S_.reindex(index=idx, columns=cols, fill_value=0).to_numpy(float)
            Na = N_.reindex(index=idx, columns=cols, fill_value=0).to_numpy(float)
            c = (W @ Sa) / (W @ Na)
            mets.append({"nominal_mean": np.nanmean(c[:, nom], 1), "menu_mean": np.nanmean(c[:, men], 1),
                         "menu_worst": np.nanmax(c[:, men], 1)})
        res = {}
        for k in mets[0]:
            dA, dB = mets[1][k] - mets[0][k], mets[3][k] - mets[2][k]
            dd = dB - dA
            lo, hi = np.nanpercentile(dd[1:], [2.5, 97.5])
            res[k] = (dA[0], dB[0], dd[0], lo, hi, bool(lo > 0 or hi < 0))
    return res


def report_items(out, seed, B):
    files = sorted(out.glob(f"items_clusters_*_s{seed}.parquet"))
    if not files:
        raise SystemExit("no A13 results; run: python src/ablation.py --run --report --sweep items")
    T = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    E = pd.concat([pd.read_csv(f) for f in sorted(out.glob(f"items_eval_alpha_*_s{seed}.csv"))], ignore_index=True)
    TR = pd.concat([pd.read_csv(f) for f in sorted(out.glob(f"items_transfer_*_s{seed}.csv"))], ignore_index=True)
    metas = [json.loads(f.read_text()) for f in sorted(out.glob(f"items_meta_*_s{seed}.json"))]
    S = [f"item={n}" for n in ITEM_SIZES]
    SH = [FULL, "robust_v1", "lstm_v1+probe"] + BASE10
    C = T.groupby(["setting", "method", "profile", "variant"], sort=False)[["loss", "n", "n_x1", "miss", "hr"]].sum().reset_index()
    C["cell"] = C["loss"] / C["n"]
    rows = []
    for (s, m), c in C[C.method.isin(SH)].groupby(["setting", "method"], sort=False):
        nom, men = c[c.profile == "nominal"], c[c.profile.isin(MENU)]
        rows.append({"setting": s, "method": m, "nominal_mean": nom.cell.mean(), "menu_mean": men.cell.mean(),
                     "menu_worst": men.cell.max(),
                     "honest_restrict": nom.hr.sum() / max(float((nom.n - nom.n_x1).sum()), 1.0),
                     "miss_nominal": nom.miss.sum() / max(float(nom.n_x1.sum()), 1.0)})
    LV = pd.DataFrame(rows)
    LV["setting"] = pd.Categorical(LV.setting, S, ordered=True)
    LV = LV.sort_values(["setting", "method"])
    LV.to_csv(out / f"items_levels_s{seed}.csv", index=False)
    star = lambda lo, hi: "*" if (lo > 0 or hi < 0) else ""
    raw, PR = {}, []
    for s in S:
        for b in ["robust_v1"] + BASE10:
            r = boot(T, s, FULL, s, b, B, seed)
            if r is None:
                continue
            raw[(s, b)] = r
            PR.append({"setting": s, "B": b, **{k: f"{va:.4f} vs {vb:.4f}: {dl:+.4f} [{lo:+.4f},{hi:+.4f}]{star(lo, hi)}"
                                                 for k, (va, vb, dl, lo, hi, _) in r.items()}})
    DD, rule = [], {}
    for s1 in S[1:]:
        for b in BASE10 + ["lstm_v1+probe"]:
            r = boot_did(T, S[0], s1, FULL, b, B, seed)
            if r is None:
                continue
            if s1 == S[-1] and b in ("severity", "lstm_sev"):
                rule[b] = r["menu_mean"]
            DD.append({"change": f"{S[0]}->{s1}", "B": b,
                       **{k: f"TRACE {dA:+.4f} | B {dB:+.4f} | DiD {dd:+.4f} [{lo:+.4f},{hi:+.4f}]{star(lo, hi)}"
                          for k, (dA, dB, dd, lo, hi, _) in r.items()}})
    h13 = len(rule) == 2 and all(v[3] > 0 for v in rule.values())
    top = [raw[(S[-1], b)]["menu_worst"] for b in BASE10 if (S[-1], b) in raw]
    h13c = bool(top) and all(v[3] > 0 for v in top)
    TP = []
    for (s, m), g in TR.groupby(["setting", "method"], sort=False):
        w = g["n"].to_numpy(float)
        TP.append({"setting": s, "method": m, **{c: float(np.average(g[c].to_numpy(float), weights=w))
                                                  for c in ["budget_mean", "budget_zero", "sent_mean", "arrived_mean", "binds"]}})
    TP = pd.DataFrame(TP)
    TP["setting"] = pd.Categorical(TP.setting, S, ordered=True)
    TP = TP.sort_values(["setting", "method"])
    E2 = []
    for (s, m, sc), e in E.groupby(["setting", "method", "scenario"], sort=False):
        E2.append({"setting": s, "method": m, "nominal_mean": e[e.profile == "nominal"].miss_at_alpha.mean(),
                   "menu_worst": e[e.profile.isin(MENU)].miss_at_alpha.max()})
    E2 = pd.DataFrame(E2).groupby(["setting", "method"], sort=False)[["nominal_mean", "menu_worst"]].mean().reset_index()
    E2 = E2[E2.method.isin(SH)]
    rs = ", ".join(f"{b}: DiD {v[2]:+.4f} [{v[3]:+.4f},{v[4]:+.4f}]" for b, v in rule.items())
    R = [f"# A13 item-size sweep, Validation (auto-generated {time.strftime('%Y-%m-%d %H:%M')}; seed {seed}; B={B}; git {git_head()})",
         "", "Sizes: " + ", ".join(f"{n} B = {l}" for n, l in ITEM_SIZES.items()), "", "## Guards and Thm 2(b)"]
    for m in metas:
        R.append(f"- {m['scenario']}: " + "; ".join(f"{k}: {'PASS' if v is True else 'FAIL' if v is False else v}"
                                                   for k, v in m["checks"].items()) + f"; Thm 2(b) {m['thm2b']}")
    R += ["", f"## H13 (A13 rule, menu_mean DiD {S[0]} -> {S[-1]} vs severity AND lstm_sev): {'SUPPORTED' if h13 else 'NOT SUPPORTED'}",
          f"- {rs}",
          f"- TRACE-MEC sig. better on menu_worst than every baseline at {S[-1]}: {'YES' if h13c else 'NO'}",
          "", "## 1. Transfer budget (Validation nominal, pooled over scenarios, weighted by episodes)",
          FENCE, TP.round(4).to_string(index=False), FENCE,
          "", "## 2. Difference-in-differences (positive = baseline degrades MORE than TRACE-MEC; * = 95% CI excludes 0)",
          FENCE, pd.DataFrame(DD).to_string(index=False), FENCE,
          "", "## 3. TRACE-MEC vs each baseline at each size (diff = B - A; positive = TRACE-MEC better)",
          FENCE, pd.DataFrame(PR).to_string(index=False), FENCE,
          "", "## 4. Levels (pooled cells; matched operating point refitted per size)",
          FENCE, LV.round(4).to_string(index=False), FENCE,
          "", "## 5. SECONDARY: miss at exactly alpha (mean over scenarios)",
          FENCE, E2.round(4).to_string(index=False), FENCE]
    (ROOT / "docs" / "ablation_items_validation.md").write_text("\n".join(R) + "\n")
    say("Report: docs/ablation_items_validation.md")


def boot(T, sA, mA, sB, mB, B, seed):
    a, b = T[(T.setting == sA) & (T.method == mA)], T[(T.setting == sB) & (T.method == mB)]
    if a.empty or b.empty:
        return None

    def mats(t):
        kw = dict(index="cluster", columns=["profile", "variant"], aggfunc="sum", fill_value=0)
        return t.pivot_table(values="loss", **kw), t.pivot_table(values="n", **kw)
    (Sa, Na), (Sb, Nb) = mats(a), mats(b)
    idx, cols = Sa.index.union(Sb.index), Sa.columns.intersection(Sb.columns)
    Sa, Na, Sb, Nb = (m.reindex(index=idx, columns=cols, fill_value=0).to_numpy(float) for m in (Sa, Na, Sb, Nb))
    prof = np.array([c[0] for c in cols])
    nom, men = prof == "nominal", np.isin(prof, MENU)
    n = len(idx)
    W = np.vstack([np.ones(n), np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=B)]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca, cb = (W @ Sa) / (W @ Na), (W @ Sb) / (W @ Nb)
        met = lambda c: {"nominal_mean": np.nanmean(c[:, nom], 1), "menu_mean": np.nanmean(c[:, men], 1),
                         "menu_worst": np.nanmax(c[:, men], 1)}
        ma, mb = met(ca), met(cb)
        res = {}
        for k in ma:
            dl = mb[k] - ma[k]
            lo, hi = np.nanpercentile(dl[1:], [2.5, 97.5])
            res[k] = (ma[k][0], mb[k][0], dl[0], lo, hi, bool(lo > 0 or hi < 0))
    return res


def report(out, seed, B):
    files = sorted(out.glob(f"clusters_*_s{seed}.parquet"))
    if not files:
        raise SystemExit("no ablation results; run --run first")
    T = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    E = pd.concat([pd.read_csv(f) for f in sorted(out.glob(f"eval_alpha_*_s{seed}.csv"))], ignore_index=True)
    metas = {}
    for f in sorted(out.glob(f"meta_*_s{seed}.json")):
        m = json.loads(f.read_text())
        metas[m["scenario"]] = m
    m0 = next(iter(metas.values()))
    order = list(dict.fromkeys(list(m0["settings"]) + list(T.setting.unique())))
    V = ["loss", "n", "n_x1", "miss", "hr", "pr1", "pp1", "pr0", "pp0"]
    C = T.groupby(["setting", "method", "profile", "variant"], sort=False)[V].sum().reset_index()
    C["cell"] = C["loss"] / C["n"]
    rows = []
    for (s, m), c in C.groupby(["setting", "method"], sort=False):
        nom, men = c[c.profile == "nominal"], c[c.profile.isin(MENU)]
        w = men.loc[men.cell.idxmax()] if len(men) else None
        rows.append({"setting": s, "method": m, "nominal_mean": nom.cell.mean(), "menu_mean": men.cell.mean(),
                     "menu_worst": men.cell.max(), "worst_cell": f"{w.profile}/{w.variant}" if w is not None else "",
                     "honest_restrict": nom.hr.sum() / max(float((nom.n - nom.n_x1).sum()), 1.0),
                     "miss_nominal": nom.miss.sum() / max(float(nom.n_x1.sum()), 1.0)})
    LV = pd.DataFrame(rows)
    LV["setting"] = pd.Categorical(LV.setting, order, ordered=True)
    LV = LV.sort_values(["setting", "method"])
    LV.to_csv(out / f"levels_s{seed}.csv", index=False)
    pairs = []
    for s in order:
        pairs += [("primary", FULL, s, "central_ma")] if s.startswith("D=") else [(s, a, s, b) for a, b in COMPARE]
    pairs.append(("primary", FULL, "primary", "central_ma"))
    PR = []
    for sA, mA, sB, mB in pairs:
        r = boot(T, sA, mA, sB, mB, B, seed)
        if r is None:
            continue
        row = {"setting": sB, "A": mA, "B": mB}
        for k, (va, vb, dl, lo, hi, sig) in r.items():
            row[k] = f"{va:.4f} vs {vb:.4f}: {dl:+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if sig else ''}"
        PR.append(row)
    PR = pd.DataFrame(PR)
    PB = []
    for s in [s for s in order if not s.startswith("D=")]:
        c = C[(C.setting == s) & (C.method == FULL) & (C.profile == "nominal")]
        row = {"setting": s, "theta_m": " ".join(f"{metas[sc]['probe'][s]['theta']:.1f}" for sc in metas),
               "probe_rate": (c.pr1.sum() + c.pr0.sum()) / max(c.n.sum(), 1),
               "false_alarm_probed_honest": c.pp0.sum() / max(c.pr0.sum(), 1)}
        for v in ("timeDelayAttack", "dataReplay", "dosAttack"):
            cv = c[c.variant == v]
            row[f"det_{v}"] = cv.pp1.sum() / max(cv.pr1.sum(), 1)
        PB.append(row)
    PB = pd.DataFrame(PB)
    KP = []
    for s in [s for s in order if s == "primary" or s.startswith("kappa=")]:
        k = m0["settings"][s]["kappa"]
        c = C[(C.setting == s) & (C.method == FULL) & (C.profile == "nominal")]
        for v in ("timeDelayAttack", "dataReplay"):
            cv = c[c.variant == v]
            KP.append({"setting": s, "variant": v, "probe_detection": cv.pp1.sum() / max(cv.pr1.sum(), 1),
                       "miss_nominal": cv.miss.sum() / max(cv.n_x1.sum(), 1),
                       "median_induced_error_m": " ".join(f"{sc.replace('InTAS_', '')}:{k * metas[sc]['lag_kappa1'][v]['median_m']:.1f}"
                                                          for sc in metas if v in metas[sc]["lag_kappa1"])})
    KP = pd.DataFrame(KP)
    E2 = []
    for (s, m, sc), e in E.groupby(["setting", "method", "scenario"], sort=False):
        E2.append({"setting": s, "method": m, "nominal_mean": e[e.profile == "nominal"].miss_at_alpha.mean(),
                   "menu_worst": e[e.profile.isin(MENU)].miss_at_alpha.max()})
    E2 = pd.DataFrame(E2).groupby(["setting", "method"], sort=False)[["nominal_mean", "menu_worst"]].mean().reset_index()
    E2 = E2[E2.method.isin(SHOW)]
    pv = LV[(LV.setting == "primary") & (LV.method == FULL)].iloc[0]
    rep = all(abs(pv[k] - v) < 5e-4 for k, v in EXPECTED.items())
    R = [f"# Ablation, {SPLIT_LABEL} (auto-generated {time.strftime('%Y-%m-%d %H:%M')}; A12; seed {seed}; B={B}; git {git_head()})",
         "", "## Guards"]
    for sc, m in metas.items():
        R += [f"- {sc}: " + "; ".join(f"{k}: {'PASS' if v is True else 'FAIL' if v is False else v}" for k, v in m["checks"].items())
              + (f"; borrowed OP keys {m['op_keys_borrowed']}" if m["op_keys_borrowed"] else "")]
    R += [f"- pooled clusters (scenario, vehicle): {T[T.setting == 'primary'].cluster.nunique()}",
          f"- primary TRACE-MEC here: nominal {pv.nominal_mean:.4f}, menu mean {pv.menu_mean:.4f}, menu worst {pv.menu_worst:.4f}; "
          f"stats.py pooled {EXPECTED}: {'MATCH' if rep else 'DIFFERENT cell weighting (tables internally consistent, not identical to stats.py)'}",
          "", "## 1. Paired cluster bootstrap (diff = B - A; positive = A better; * = 95% CI excludes 0)",
          FENCE, PR.to_string(index=False), FENCE,
          "", "## 2. Levels (pooled cells; matched operating point per setting)",
          FENCE, LV[LV.method.isin(SHOW)].round(4).to_string(index=False), FENCE,
          "", f"## 3. Probe ({FULL}, {SPLIT_LABEL} nominal; theta per scenario {list(metas)})",
          FENCE, PB.round(4).to_string(index=False), FENCE,
          "", "## 4. Adaptive attacker: detectability vs damage (defender not recalibrated)",
          FENCE, KP.round(4).to_string(index=False), FENCE,
          "", "## 5. SECONDARY: miss at exactly alpha (mean over scenarios)",
          FENCE, E2.round(4).to_string(index=False), FENCE]
    (ROOT / REPORT_DOC).write_text("\n".join(R) + "\n")
    say(f"Report: {REPORT_DOC}")


# ---------------- A17: the A12 sweeps on held-out Test (same run_scenario / report code) ----------------
def _rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS reports bytes


class _NoSnap:
    """Test mode: eval_split never writes results/model_smoke; protection is an md5 check instead."""
    def __init__(self, d):
        self.sd = self.store = Path(d)

    def restore(self):
        pass


class _TestAudit:
    """Records every pandas parquet read; aborts on Validation, path-fold or gate traces."""
    BAD = ("/Validation/", "/data/pathfold/", "/data/gate")

    def __init__(self):
        self.seen = set()

    def __enter__(self):
        self._o = pd.read_parquet

        def rp(path, *a, **k):
            try:
                rel = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
            except (TypeError, ValueError):
                rel = None
            if rel is not None:
                self.seen.add(rel)
                if any(b in "/" + rel for b in self.BAD):
                    raise SystemExit(f"ABORT (read audit): Test ablation tried to read {rel}")
            return self._o(path, *a, **k)
        pd.read_parquet = rp
        return self

    def __exit__(self, *exc):
        pd.read_parquet = self._o
        return False


def _scen_done(out, sc, seed):
    mf = out / f"meta_{sc}_s{seed}.json"
    if not mf.exists():
        return False
    m = json.loads(mf.read_text())
    return bool(m.get("checks")) and all(v is True for v in m["checks"].values())


def run_test(seed, B, resume):
    global TEST, EXPECTED, SPLIT_LABEL, REPORT_DOC
    import evaluate as E
    import pyarrow.parquet as pq
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/ablation.py", "src/evaluate.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted changes (commit first):\n{dirty}")
    E.pf_frozen_check()
    say("CHECK frozen code (model.py, stats.py, train.py; config = frozen + gate/pathfold blocks only): PASS")
    cfg, how = E.load_cfg()
    say(f"config loaded via {how}")
    M = cfg.get("model", cfg)
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit("ABORT: evidence_item_bytes must be 256 (A17)")
    if seed != 0:
        raise SystemExit("ABORT: A17 fixes seed 0")
    S = settings(M)
    res = ROOT / cfg["paths"]["results"]
    out, ref_dir = res / "ablation_test", res / "final" / "item256"
    scratch = out / "_scratch"
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ROOT).free / 1e9
    say(f"free disk {free:.1f} GB")
    if free < 10:
        raise SystemExit(f"ABORT: only {free:.1f} GB free; need >= 10 GB (scratch episodes + macOS swap)")
    for sc in SCEN:
        rf = ref_dir / f"episodes_{sc}_s{seed}.parquet"
        if not rf.exists():
            raise SystemExit(f"ABORT: A14 reference missing {rf}")
        say(f"A14 reference {sc}: {pq.ParquetFile(rf).metadata.num_rows:,} episode rows ({rf.stat().st_size / 1e6:.0f} MB)")
    lock = out / "LOCK.json"
    att = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_head(), "mode": "resume" if resume else "first"}
    if lock.exists():
        L = json.loads(lock.read_text())
        if L.get("status") == "complete":
            raise SystemExit("REFUSED: Test ablation already complete (run-once lock). See docs/ablation_test.md")
        if not resume:
            raise SystemExit(f"REFUSED: an earlier attempt stopped at '{L.get('status')}'; rerun with --resume (disclosed)")
        L["attempts"].append(att)
    else:
        if resume:
            raise SystemExit("REFUSED: nothing to resume")
        L = {"created": att["start"], "attempts": [att]}

    def setst(s, **kw):
        L["status"] = s
        L["attempts"][-1].update(status=s, **kw)
        lock.write_text(json.dumps(L, indent=2, default=str))

    prot0 = {d: md5_tree(res / d) for d in ("final", "frozen_v071", "model_smoke")}
    say("protected trees hashed: results/final, results/frozen_v071, results/model_smoke")
    f = E.build_eval_split()
    todo = [sc for sc in SCEN if not _scen_done(out, sc, seed)]
    total = 3.0 * sum(TEST_MIN[sc] for sc in todo)
    st = {"done": 0.0, "t0": time.time()}
    say(f"=== {len(todo)} scenario(s) to run {todo}; 3 Test simulations each (primary D=5 s, central_ma D=1 s, D=30 s); "
        f"planned ~{total / 60:.1f} h + replays + bootstrap")

    def sim(c_, sc, s_):
        D = float(Mo.EXT_CFG["ma_delay_s"])
        lab = f"central_ma D={D:g} s" + (" (primary)" if D == 5.0 else "")
        say(f"START Test simulation {sc} | {lab}")
        t = time.time()
        shutil.rmtree(scratch, ignore_errors=True)
        f(c_, sc, s_, "Test", scratch)
        st["done"] += TEST_MIN[sc]
        el = time.time() - st["t0"]
        eta = (total - st["done"]) * el / st["done"] if st["done"] else float("nan")
        say(f"DONE  Test simulation {sc} | {lab} | {(time.time() - t) / 60:.1f} min | planned {st['done']:.0f}/{total:.0f} min "
            f"| elapsed {el / 3600:.2f} h | ETA {eta / 3600:.2f} h | peak RSS {_rss_gb():.1f} GB")

    def ref(sc, s_):
        ok, msg = E.compare(scratch / f"episodes_{sc}_s{s_}.parquet", ref_dir / f"episodes_{sc}_s{s_}.parquet")
        say(f"REFCHECK {sc} vs A14 seed-0 Test episodes: {'PASS' if ok else 'FAIL'} - {msg}")
        return ok, msg

    TEST, EXPECTED = {"sim": sim, "ref": ref}, dict(EXPECTED_TEST)
    SPLIT_LABEL, REPORT_DOC = "Test (held-out, A17)", "docs/ablation_test.md"
    H, snap = Hooks(M["faults"]), _NoSnap(scratch)
    E._TEST_OPEN = True
    setst("running")
    try:
        with _TestAudit() as AU:
            for sc in SCEN:
                if sc not in todo:
                    say(f"[{sc}] completed in an earlier attempt (reused; deterministic)")
                    continue
                run_scenario(cfg, sc, seed, S, snap, H, out)
                setst("running", **{f"done_{sc}": time.strftime("%Y-%m-%dT%H:%M:%S")})
    except BaseException as e:
        setst("aborted", error=repr(e)[:800])
        raise
    finally:
        H.uninstall()
        E._TEST_OPEN = False
        shutil.rmtree(scratch, ignore_errors=True)
    prot_ok = {d: md5_tree(res / d) == h for d, h in prot0.items()}
    for d, v in prot_ok.items():
        say(f"CHECK results/{d} untouched: {'PASS' if v else 'FAIL'}")
    setst("report")
    report(out, seed, B)
    seen = sorted(AU.seen)
    tops = pd.Series(["/".join(p.split("/")[:2]) for p in seen]).value_counts().to_dict() if seen else {}
    X = ["", "## A17 run record", f"- attempts: {len(L['attempts'])} (disclose if > 1)"]
    X += [f"  - {a['start']} mode={a['mode']} status={a.get('status')} {a.get('error', '')}" for a in L["attempts"]]
    X += [f"- read audit: {len(seen)} distinct parquet files read via pandas, by top dir {tops}; none under Validation, pathfold or gate",
          "- primary Test simulation reproduced A14 seed-0 episodes bit-for-bit in every scenario (see Guards)",
          "- protected trees untouched: " + ", ".join(f"results/{d} {'yes' if v else 'NO'}" for d, v in prot_ok.items())]
    with open(ROOT / REPORT_DOC, "a") as fh:
        fh.write("\n".join(X) + "\n")
    setst("complete", end=time.strftime("%Y-%m-%dT%H:%M:%S"), runtime_s=round(time.time() - st["t0"]))
    print("=== RESULT: TEST ABLATION COMPLETE (A17; run-once lock closed)")
    for sc in SCEN:
        m = json.loads((out / f"meta_{sc}_s{seed}.json").read_text())
        print(f"- {sc}: " + "; ".join(f"{k}: {'PASS' if v is True else 'FAIL'}" for k, v in m["checks"].items()))
    print("\n".join(l for l in (ROOT / REPORT_DOC).read_text().splitlines() if "primary TRACE-MEC here" in l))
    print(f"- protected trees untouched: {all(prot_ok.values())}")
    print(f"Full report: {REPORT_DOC}")


# ---------------- A18: probe rate {0.10, 0.30} + online recalibration (docs/theory.md A18) ----------------
RATES = [0.10, 0.30]
A18_RECAL_SHOW = [FULL, "lstm_v1+probe", "lstm_sev", "severity", "central_ma", "voi_myopic"]


def settings_a18(M):
    prim = dict(settings(M)["primary"])
    r0 = float(M["probe"]["rate"])
    if r0 != 0.20:
        raise SystemExit(f"ABORT: config probe rate {r0} != preregistered 0.20")
    prim["rate"] = r0
    S = {"primary": prim}
    for r in RATES:
        S[f"rate={r:g}"] = dict(prim, rate=float(r))
    return S


def recal_shares(H, OP):
    """Label-free target shares per method: fraction of ALL Train nominal episodes above the Train threshold.
    Mirrors fit_operating exactly (same scores, same tie-break). Also returns the Train honest share (must = alpha)."""
    fit, PI, M, L, seed = H.fit
    lam, hon = {}, {}
    u, p, x = Mo.ep_u(fit, seed, 9), fit["p"].to_numpy(), fit["x"].to_numpy().astype(int)
    for m, idx in fit.groupby("method").indices.items():
        if m in OP["thr"]:
            s, t = p[idx] + 1e-9 * u[idx], OP["thr"][m]
            lam[m] = (float((s > t[0]).mean()), float((s > t[1]).mean()))
            hon[m] = float((s[x[idx] == 0] > t[0]).mean())
    for pm, (bm, _) in Mo.PROBES.items():
        g2, _ = Mo.probe_apply(fit[fit["method"] == bm], PI, OP, pm, L, seed)
        s, t = g2["p"].to_numpy() + 1e-9 * Mo.ep_u(g2, seed, 9), OP["thr"][pm]
        xg = g2["x"].to_numpy().astype(int)
        lam[pm] = (float((s > t[0]).mean()), float((s > t[1]).mean()))
        hon[pm] = float((s[xg == 0] > t[0]).mean())
    return lam, hon


def recal_apply(d, lam, OP, L, seed):
    """Per (profile, method), pooled over variants: thresholds = quantiles of ALL scores at the Train shares."""
    p, u = d["p"].to_numpy(), Mo.ep_u(d, seed, 9)
    s, x = p + 1e-9 * u, d["x"].to_numpy().astype(int)
    a_off, a_on = np.zeros(len(d), np.int64), np.full(len(d), -1, np.int64)
    for (pf, m), idx in d.groupby(["profile", "method"]).indices.items():
        a_off[idx] = Mo.op_act(p[idx], u[idx], OP["thr"][m])
        if m in lam:
            tr, tq = np.quantile(s[idx], 1 - lam[m][0]), np.quantile(s[idx], 1 - lam[m][1])
            a_on[idx] = (s[idx] > tr).astype(np.int64) + (s[idx] > tq).astype(np.int64)
    same = bool(np.array_equal(a_off, d["action"].to_numpy().astype(np.int64)))
    keep = a_on >= 0
    skip = sorted(set(d.loc[~keep, "method"]))
    d_on = d.loc[keep].assign(action=a_on[keep], loss=L[a_on[keep], x[keep]])
    return d_on, same, skip


def run_scenario_a18(cfg, sc, seed, S, snap, H, out):
    t0, checks = time.time(), {}
    ep = snap.sd / f"episodes_{sc}_s{seed}.parquet"
    H.reset("base")
    H.install()
    try:
        if TEST is None:
            say(f"[{sc}] primary Validation smoke with capture hooks")
            Mo.smoke(cfg, sc, seed)
            old = snap.store / "model_smoke" / ep.name
            checks["fresh smoke reproduces stored v0.9 episodes file"] = bool(same_frames(old, ep)) if old.exists() and ep.exists() else "n/a"
        else:
            TEST["sim"](cfg, sc, seed)
            ok_ref, msg_ref = TEST["ref"](sc, seed)
            checks["fresh Test simulation reproduces A14 seed-0 Test episodes bit-for-bit"] = bool(ok_ref)
            if not ok_ref:
                raise SystemExit(f"ABORT [{sc}]: Test simulation differs from A14 seed-0 episodes ({msg_ref})")
    finally:
        H.uninstall()
        snap.restore()
    if H.fit is None or not H.items:
        raise SystemExit(f"ABORT [{sc}]: hooks captured nothing (smoke internals changed)")
    alpha = float(H.fit[2]["operating"]["alpha"])
    meta = {"scenario": sc, "seed": seed, "settings": S, "probe": {}, "lag_kappa1": lag_stats(H.items),
            "op_keys_borrowed": {}, "amendment": "A18"}
    tabs, eas, nfp0 = [], [], None
    for name, s in S.items():
        OP, outs, borrowed = replay(H, s)
        if borrowed:
            meta["op_keys_borrowed"][name] = borrowed
        nfp = [fp(o[~o["method"].isin(list(Mo.PROBES))]) for o, _, _ in outs]
        if name == "primary":
            ok = len(outs) == len(H.items) and all(fp(o) == it["fp_all"] for (o, _, _), it in zip(outs, H.items))
            checks["replay at primary reproduces smoke bit-for-bit"] = ok
            if not ok:
                raise SystemExit(f"ABORT [{sc}]: replay at primary differs from smoke")
            nfp0 = nfp
        else:
            ok = nfp == nfp0
            checks[f"{name}: every non-probe method identical to primary"] = ok
            if not ok:
                raise SystemExit(f"ABORT [{sc}]: {name} changed a method without probe")
        d = frame(outs)
        del outs
        tabs.append(tabulate(d, sc, name))
        eas.append(eval_tab(d, s["alpha"], seed, sc, name))
        P = OP["probe"]
        meta["probe"][name] = {"sigma": P["sigma"], "theta": P["theta"], "d": P["d"], "f": P["f"], "t_p": P["t_p"],
                               "rate": P["rate"], "rate_train": OP.get("probe_rate_train", {})}
        say(f"[{sc}] replayed {name}: rate={P['rate']:.2f} theta={P['theta']:.1f} m d={P['d']:.3f} f={P['f']:.3f}")
        if name == "primary":
            lam, hon = recal_shares(H, OP)
            dev = max(abs(v - alpha) for v in hon.values())
            checks["recalibration mirrors fit_operating on Train (honest share = alpha, max dev < 0.005)"] = bool(dev < 0.005)
            if dev >= 0.005:
                raise SystemExit(f"ABORT [{sc}]: Train honest share deviates from alpha by {dev:.4f}")
            d_on, same, skip = recal_apply(d, lam, OP, H.fit[3], seed)
            checks["recalibration code with Train thresholds reproduces primary actions exactly"] = same
            if not same:
                raise SystemExit(f"ABORT [{sc}]: recalibration path cannot reproduce primary actions")
            tabs.append(tabulate(d_on, sc, "recal=on"))
            eas.append(eval_tab(d_on, PRIMARY_ALPHA, seed, sc, "recal=on"))
            meta["probe"]["recal=on"] = dict(meta["probe"]["primary"])
            meta["recal"] = {"lambda": {k: [round(a, 5), round(b, 5)] for k, (a, b) in lam.items()},
                             "train_honest_max_dev": dev, "not_recalibrated": skip}
            say(f"[{sc}] recal=on applied (label-free); not recalibrated: {skip or 'none'}")
            del d_on
        del d
    H.reset("base")
    pd.concat(tabs, ignore_index=True).to_parquet(out / f"clusters_{sc}_s{seed}.parquet", index=False)
    pd.concat(eas, ignore_index=True).to_csv(out / f"eval_alpha_{sc}_s{seed}.csv", index=False)
    meta.update(checks=checks, runtime_s=round(time.time() - t0), git=git_head())
    (out / f"meta_{sc}_s{seed}.json").write_text(json.dumps(meta, indent=2, default=str))
    for k, v in checks.items():
        say(f"CHECK [{sc}] {k}: {'PASS' if v is True else 'FAIL' if v is False else v}")


def report_a18(out, seed, B):
    report(out, seed, B)
    doc = ROOT / REPORT_DOC
    txt = doc.read_text().replace("; A12;", "; A18 probe rate + online recalibration;", 1)
    T = pd.concat([pd.read_parquet(f) for f in sorted(out.glob(f"clusters_*_s{seed}.parquet"))], ignore_index=True)
    metas = {}
    for f in sorted(out.glob(f"meta_*_s{seed}.json")):
        m = json.loads(f.read_text())
        metas[m["scenario"]] = m
    pairs = [("primary", FULL, s, FULL) for s in ("rate=0.1", "rate=0.3")]
    pairs += [("primary", m, "recal=on", m) for m in A18_RECAL_SHOW]
    PR = []
    for sA, mA, sB, mB in pairs:
        r = boot(T, sA, mA, sB, mB, B, seed)
        row = {"comparison": f"{mA}: {sB} vs {sA}"}
        if r is None:
            row["nominal_mean"] = "n/a (not recalibrated or absent)"
        else:
            for k, (va, vb, dl, lo, hi, sig) in r.items():
                row[k] = f"{va:.4f} -> {vb:.4f}: {dl:+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if sig else ''}"
        PR.append(row)
    nom = T[T.profile == "nominal"]
    g = nom.groupby(["setting", "method", "scenario"])[["hr", "n", "n_x1"]].sum()
    hs = (g["hr"] / (g["n"] - g["n_x1"]).clip(lower=1)).rename("honest").reset_index()
    HS = hs[hs.method.isin(SHOW)].pivot_table(index=["setting", "method"], columns="scenario", values="honest").round(3)
    skips = {sc: m.get("recal", {}).get("not_recalibrated", []) for sc, m in metas.items()}
    devs = {sc: round(m.get("recal", {}).get("train_honest_max_dev", float("nan")), 5) for sc, m in metas.items()}
    lam = {sc: {k: v for k, v in m.get("recal", {}).get("lambda", {}).items() if k in SHOW} for sc, m in metas.items()}
    X = ["", "## 6. A18 within-method effects (diff = setting - primary; positive = primary better; * = 95% CI excludes 0)",
         FENCE, pd.DataFrame(PR).to_string(index=False), FENCE,
         "", "## 7. Realised honest restrict-or-quarantine share, nominal, per scenario (target alpha = 0.10)",
         FENCE, HS.to_string(), FENCE,
         "", "## 8. Online recalibration record (label-free; sensitivity analysis, not a design change)",
         f"- methods not recalibrated (threshold not fitted in fit_operating): {skips}",
         f"- Train honest-share max deviation from alpha (mirror check): {devs}",
         f"- Train label-free target shares (restrict, quarantine), shown methods: {json.dumps(lam)}"]
    doc.write_text(txt + "\n".join(X) + "\n")
    say(f"Report: {REPORT_DOC}")


def run_a18_val(seed, B, scen):
    global SPLIT_LABEL, REPORT_DOC
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/ablation.py", "src/model.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted changes (commit first):\n{dirty}")
    cfg = Mo.load_cfg()
    M = cfg.get("model", cfg)
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit("ABORT: evidence_item_bytes must be 256 (A18)")
    S = settings_a18(M)
    res = ROOT / cfg["paths"]["results"]
    out = res / "ablation_a18"
    out.mkdir(parents=True, exist_ok=True)
    prot = [d for d in ("frozen_v071", "final", "ablation", "ablation_test") if (res / d).exists()]
    say(f"hashing protected trees {prot} ...")
    prot0 = {d: md5_tree(res / d) for d in prot}
    say(f"=== A18 Validation: {len(scen)} scenario(s); settings {list(S)} + recal=on; expected ~20 min")
    snap = Snapshot(res / "model_smoke", ROOT / "docs" / "model_smoke.md", out / "_snapshot")
    H = Hooks(M["faults"])
    t0 = time.time()
    try:
        for i, sc in enumerate(scen):
            say(f"START [{sc}] ({i + 1}/{len(scen)}) | elapsed {(time.time() - t0) / 60:.1f} min | peak RSS {_rss_gb():.1f} GB")
            run_scenario_a18(cfg, sc, seed, S, snap, H, out)
            say(f"DONE  [{sc}] ({i + 1}/{len(scen)}) | elapsed {(time.time() - t0) / 60:.1f} min | peak RSS {_rss_gb():.1f} GB")
    finally:
        H.uninstall()
        snap.restore()
        snap.drop()
    for d, h in prot0.items():
        ok = md5_tree(res / d) == h
        say(f"CHECK results/{d} untouched: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"ABORT: results/{d} changed")
    SPLIT_LABEL, REPORT_DOC = "Validation (A18)", "docs/ablation_validation_a18.md"
    report_a18(out, seed, B)
    print("=== RESULT: A18 VALIDATION COMPLETE")
    for sc in scen:
        m = json.loads((out / f"meta_{sc}_s{seed}.json").read_text())
        print(f"- {sc}: " + "; ".join(f"{k}: {'PASS' if v is True else 'FAIL' if v is False else v}" for k, v in m["checks"].items()))
    print("\n".join(l for l in (ROOT / REPORT_DOC).read_text().splitlines() if "primary TRACE-MEC here" in l))
    print(f"Full report: {REPORT_DOC}")


def run_test_a18(seed, B, resume):
    global TEST, EXPECTED, SPLIT_LABEL, REPORT_DOC
    import evaluate as E
    import pyarrow.parquet as pq
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/ablation.py", "src/evaluate.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted changes (commit first):\n{dirty}")
    E.pf_frozen_check()
    say("CHECK frozen code (model.py, stats.py, train.py; config = frozen + gate/pathfold blocks only): PASS")
    cfg, how = E.load_cfg()
    say(f"config loaded via {how}")
    M = cfg.get("model", cfg)
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit("ABORT: evidence_item_bytes must be 256 (A18)")
    if seed != 0:
        raise SystemExit("ABORT: A18 fixes seed 0")
    S = settings_a18(M)
    res = ROOT / cfg["paths"]["results"]
    vout = res / "ablation_a18"
    if not (ROOT / "docs" / "ablation_validation_a18.md").exists() or not all(_scen_done(vout, sc, seed) for sc in SCEN):
        raise SystemExit("REFUSED: A18 order is Validation first (python src/ablation.py --a18) with all checks PASS")
    out, ref_dir = res / "ablation_test_a18", res / "final" / "item256"
    scratch = out / "_scratch"
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ROOT).free / 1e9
    say(f"free disk {free:.1f} GB")
    if free < 10:
        raise SystemExit(f"ABORT: only {free:.1f} GB free; need >= 10 GB")
    for sc in SCEN:
        rf = ref_dir / f"episodes_{sc}_s{seed}.parquet"
        if not rf.exists():
            raise SystemExit(f"ABORT: A14 reference missing {rf}")
        say(f"A14 reference {sc}: {pq.ParquetFile(rf).metadata.num_rows:,} episode rows")
    lock = out / "LOCK.json"
    att = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_head(), "mode": "resume" if resume else "first"}
    if lock.exists():
        L = json.loads(lock.read_text())
        if L.get("status") == "complete":
            raise SystemExit("REFUSED: A18 Test ablation already complete (run-once lock). See docs/ablation_test_a18.md")
        if not resume:
            raise SystemExit(f"REFUSED: an earlier attempt stopped at '{L.get('status')}'; rerun with --resume (disclosed)")
        L["attempts"].append(att)
    else:
        if resume:
            raise SystemExit("REFUSED: nothing to resume")
        L = {"created": att["start"], "attempts": [att]}

    def setst(s, **kw):
        L["status"] = s
        L["attempts"][-1].update(status=s, **kw)
        lock.write_text(json.dumps(L, indent=2, default=str))

    prot = [d for d in ("final", "frozen_v071", "model_smoke", "ablation_test", "ablation_a18") if (res / d).exists()]
    say(f"hashing protected trees {prot} ...")
    prot0 = {d: md5_tree(res / d) for d in prot}
    f = E.build_eval_split()
    todo = [sc for sc in SCEN if not _scen_done(out, sc, seed)]
    total = float(sum(TEST_MIN[sc] for sc in todo))
    st = {"done": 0.0, "t0": time.time()}
    say(f"=== {len(todo)} scenario(s) to run {todo}; 1 Test simulation each + 3 replays + recalibration; "
        f"planned ~{total / 60:.1f} h simulation + replays + bootstrap (expect ~1-1.5 h)")

    def sim(c_, sc, s_):
        say(f"START Test simulation {sc} | primary (A18)")
        t = time.time()
        shutil.rmtree(scratch, ignore_errors=True)
        f(c_, sc, s_, "Test", scratch)
        st["done"] += TEST_MIN[sc]
        el = time.time() - st["t0"]
        eta = (total - st["done"]) * el / st["done"] if st["done"] else float("nan")
        say(f"DONE  Test simulation {sc} | {(time.time() - t) / 60:.1f} min | planned {st['done']:.0f}/{total:.0f} min "
            f"| elapsed {el / 3600:.2f} h | ETA {eta / 3600:.2f} h | peak RSS {_rss_gb():.1f} GB")

    def ref(sc, s_):
        ok, msg = E.compare(scratch / f"episodes_{sc}_s{s_}.parquet", ref_dir / f"episodes_{sc}_s{s_}.parquet")
        say(f"REFCHECK {sc} vs A14 seed-0 Test episodes: {'PASS' if ok else 'FAIL'} - {msg}")
        return ok, msg

    TEST, EXPECTED = {"sim": sim, "ref": ref}, dict(EXPECTED_TEST)
    SPLIT_LABEL, REPORT_DOC = "Test (held-out, A18)", "docs/ablation_test_a18.md"
    H, snap = Hooks(M["faults"]), _NoSnap(scratch)
    E._TEST_OPEN = True
    setst("running")
    try:
        with _TestAudit() as AU:
            for sc in SCEN:
                if sc not in todo:
                    say(f"[{sc}] completed in an earlier attempt (reused; deterministic)")
                    continue
                run_scenario_a18(cfg, sc, seed, S, snap, H, out)
                setst("running", **{f"done_{sc}": time.strftime("%Y-%m-%dT%H:%M:%S")})
    except BaseException as e:
        setst("aborted", error=repr(e)[:800])
        raise
    finally:
        H.uninstall()
        E._TEST_OPEN = False
        shutil.rmtree(scratch, ignore_errors=True)
    prot_ok = {d: md5_tree(res / d) == h for d, h in prot0.items()}
    for d, v in prot_ok.items():
        say(f"CHECK results/{d} untouched: {'PASS' if v else 'FAIL'}")
    setst("report")
    report_a18(out, seed, B)
    seen = sorted(AU.seen)
    tops = pd.Series(["/".join(p.split("/")[:2]) for p in seen]).value_counts().to_dict() if seen else {}
    X = ["", "## A18 run record", f"- attempts: {len(L['attempts'])} (disclose if > 1)"]
    X += [f"  - {a['start']} mode={a['mode']} status={a.get('status')} {a.get('error', '')}" for a in L["attempts"]]
    X += [f"- read audit: {len(seen)} distinct parquet files read via pandas, by top dir {tops}; none under Validation, pathfold or gate",
          "- primary Test simulation reproduced A14 seed-0 episodes bit-for-bit in every scenario (see Guards)",
          "- protected trees untouched: " + ", ".join(f"results/{d} {'yes' if v else 'NO'}" for d, v in prot_ok.items())]
    with open(ROOT / REPORT_DOC, "a") as fh:
        fh.write("\n".join(X) + "\n")
    setst("complete", end=time.strftime("%Y-%m-%dT%H:%M:%S"), runtime_s=round(time.time() - st["t0"]))
    print("=== RESULT: A18 TEST ABLATION COMPLETE (run-once lock closed)")
    for sc in SCEN:
        m = json.loads((out / f"meta_{sc}_s{seed}.json").read_text())
        print(f"- {sc}: " + "; ".join(f"{k}: {'PASS' if v is True else 'FAIL'}" for k, v in m["checks"].items()))
    print("\n".join(l for l in (ROOT / REPORT_DOC).read_text().splitlines() if "primary TRACE-MEC here" in l))
    print(f"- protected trees untouched: {all(prot_ok.values())}")
    print(f"Full report: {REPORT_DOC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=SCEN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--sweep", choices=["a12", "items"], default="a12")
    ap.add_argument("--final", action="store_true", help="A17: the A12 sweeps on held-out Test, run once")
    ap.add_argument("--resume", action="store_true", help="A17: continue an interrupted --final (disclosed)")
    ap.add_argument("--a18", action="store_true", help="A18: probe rate + online recalibration (with --final: Test once)")
    a = ap.parse_args()
    if a.a18:
        return run_test_a18(a.seed, a.B, a.resume) if a.final else run_a18_val(a.seed, a.B, a.scenarios)
    if a.final:
        return run_test(a.seed, a.B, a.resume)
    cfg = Mo.load_cfg()
    M = cfg.get("model", cfg)
    res = ROOT / cfg["paths"]["results"]
    out = res / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    store = out / "_snapshot"
    if a.restore:
        shutil.copytree(store / "model_smoke", res / "model_smoke", dirs_exist_ok=True)
        shutil.copy2(store / "model_smoke.md", ROOT / "docs" / "model_smoke.md")
        shutil.rmtree(store)
        say("snapshot restored and removed")
        return
    if a.run:
        S = settings(M)
        frozen = res / "frozen_v071"
        h_fz = md5_tree(frozen)
        snap = Snapshot(res / "model_smoke", ROOT / "docs" / "model_smoke.md", store)
        H = Hooks(M["faults"])
        try:
            for sc in a.scenarios:
                if a.sweep == "items":
                    run_items(cfg, sc, a.seed, snap, H, out)
                else:
                    run_scenario(cfg, sc, a.seed, S, snap, H, out)
        finally:
            H.uninstall()
            snap.restore()
            snap.drop()
        say(f"CHECK frozen_v071 untouched: {'PASS' if md5_tree(frozen) == h_fz else 'FAIL'}")
        say("CHECK model_smoke outputs restored byte-identically: PASS")
    if a.report:
        (report_items if a.sweep == "items" else report)(out, a.seed, a.B)


if __name__ == "__main__":
    main()
