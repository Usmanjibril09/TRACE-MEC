"""a20.py - A20 extension on the held-out Test split: official VeReMi NextGen position attacks
(constantPositionOffset, randomPositionOffset) on InTAS_highway_2 and InTAS_urban_2, seeds 0-4, 256 B items.
Frozen pipeline: model.smoke() compiled with evaluate.SUBS plus one variant-list substitution; Train fit unchanged."""
import hashlib, inspect, json, pathlib, re, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate as EV    # imports model (MD) and stats (ST); installs the Test gate on trace_dir
import preprocess as PP

MD, ST = EV.MD, EV.ST
SCN = ["InTAS_highway_2", "InTAS_urban_2"]
ATT = ["constantPositionOffset", "randomPositionOffset"]
EXT = ["baseline"] + ATT
SEEDS = [0, 1, 2, 3, 4]
ZIPS = ROOT / "data" / "a20_check"
ZMD5 = {"InTAS_highway_2.zip": "d5fefb873c5e766b702bd036f18a326d",
        "InTAS_highway_2_constantPositionOffset.zip": "4b8aa6b684a093de218256bae74c1a6e",
        "InTAS_highway_2_randomPositionOffset.zip": "0b72ffd770351817deb18fa3ec630a61",
        "InTAS_urban_2.zip": "a8e55948b2f90c554a76ef619d852e43",
        "InTAS_urban_2_constantPositionOffset.zip": "c4340ca0d1d4a517cff248bf015bcacd",
        "InTAS_urban_2_randomPositionOffset.zip": "3064cf6c427487a5a9ffd70e914f40ef"}
HELD, PRIM = ROOT / "data" / "heldout", ROOT / "results" / "final" / "item256"
OUT = ROOT / "results" / "a20"
RUN, AVG, TMP, LOCKF = OUT / "item256", OUT / "item256_seeds0-4", OUT / "_tmp", OUT / "LOCK_a20.json"
PROTECT_MD5 = ["results/final", "results/model_smoke", "results/frozen_v071", "results/gate", "results/pathfold",
               "results/ablation_test", "results/ablation_test_a18", "results/lstm"]
PROTECT_STAT = ["data/heldout", "data/interim", "data/processed"]
K = ["variant", "profile", "method", "session_uid", "ho_idx"]
JK = PP.JOIN_KEY
METRICS = ["nominal_mean", "menu_mean", "menu_worst", "pos_worst"]
FULL, POOL10 = ST.FULL, ST.BASELINES + ST.EXTENDED
T0 = time.time()
DESC, LAGS, GUARDS = [], [], []
L = {}


def say(m):
    print(f"[{datetime.now():%H:%M:%S} | +{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def stop(m):
    say("STOP: " + m); sys.exit(2)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return str(pathlib.Path(p).relative_to(ROOT))


class _Log:
    def info(self, m):
        say(m)


def setst(s, **kw):
    L["status"] = s
    L["attempts"][-1].update(status=s, **kw)
    LOCKF.write_text(json.dumps(L, indent=2, default=str))


def whitelist():
    w = set()
    for sc in SCN:
        for v in ATT:
            w |= {f"data/heldout/veremi/{sc}/{v}_Test.parquet", f"data/heldout/{sc}/Test/evidence_{v}.parquet",
                  f"data/heldout/{sc}/Test/sessions_{v}.parquet", f"results/lstm/{sc}/{v}_Test.parquet"}
    return w


def snapshot():
    s = {}
    for d, use_md5 in [(x, True) for x in PROTECT_MD5] + [(x, False) for x in PROTECT_STAT]:
        if (ROOT / d).exists():
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file():
                    st = p.stat()
                    s[rel(p)] = md5f(p) if use_md5 else f"{st.st_size}:{st.st_mtime_ns}"
    return s


def same(a, b, what):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        stop(f"{what}: shape/columns differ {a.shape} vs {b.shape}; {list(a.columns)} vs {list(b.columns)}")
    bad = [c for c in a.columns if not a[c].equals(b[c])]
    if bad:
        stop(f"{what}: columns differ {[(c, str(a[c].dtype), str(b[c].dtype)) for c in bad]}")


def a20_text():
    lines = (ROOT / "docs" / "theory.md").read_text().splitlines()
    hd = [i for i, l in enumerate(lines) if re.match(r"^## A20(?![0-9A-Za-z])", l)]
    if len(hd) != 1:
        stop(f"expected exactly one '## A20' heading in docs/theory.md, found {len(hd)}")
    j = next((k for k in range(hd[0] + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return lines[hd[0]:j]


def source_guard():
    pat = re.compile(r'\bVARIANTS\b|timeDelayAttack|dataReplay|dosAttack|"docs"')
    hits = []
    for fn in ["report_v07", "loss_sweep", "summarize", "apply_operating", "probe_apply", "simulate", "_simulate_core",
               "decide", "ext_apply", "ext_item_scores", "f2md_table", "probe_info", "load_variant", "build_episodes"]:
        f = getattr(MD, fn, None)
        if f is None:
            stop(f"model.{fn} not found")
        src, start = inspect.getsourcelines(f)
        hits += [f"  model.py:{start + i} ({fn}): {l.rstrip()}" for i, l in enumerate(src) if pat.search(l)]
    if hits:
        say("\n".join(hits))
        stop("functions applied to the evaluation set reference fixed variant names or docs/ (lines above); nothing was read")


def build_split():
    src = inspect.getsource(MD.smoke)
    for a, b in EV.SUBS:
        if src.count(a) != 1:
            stop(f"expected 1 occurrence in model.smoke() of {a!r}, found {src.count(a)}")
        src = src.replace(a, b)
    extra = [("def eval_split(cfg, sc, seed, split, out_dir):", "def a20_split(cfg, sc, seed, split, out_dir):"),
             ("for v in VARIANTS:\n        ev, ho, se = load_variant(cfg, sc, split, v)",
              "for v in A20_VARIANTS:\n        ev, ho, se = load_variant(cfg, sc, split, v)")]
    for a, b in extra:
        if src.count(a) != 1:
            stop(f"A20 substitution anchor found {src.count(a)} times (expected 1): {a!r}")
        src = src.replace(a, b)
    old = '(ROOT / "docs" / "model_smoke.md")'
    if src.count(old) == 1:
        src = src.replace(old, '(out_dir / f"model_report_{sc}_s{seed}.md")')
    for bad in ['"Validation"', '"docs"', '"model_smoke"']:
        if bad in src:
            stop(f"compiled evaluation function still contains {bad}")
    MD.A20_VARIANTS = list(EXT)
    exec(compile(src, "<a20: model.smoke with split/output/variant substitutions>", "exec"), MD.__dict__)
    return MD.__dict__["a20_split"]


def stage_clean(sc, rec):
    tz = pathlib.Path("/tmp/a20_zip"); tz.mkdir(exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    b = pd.read_parquet(HELD / "veremi" / sc / "baseline_Test.parquet")
    t = TMP / f"{sc}_baseline_Test.parquet"
    PP.clean_veremi_split(ZIPS, sc, "baseline", "Test", t, tz, 500_000, _Log())
    same(pd.read_parquet(t), b, f"{sc}: official attack-free Test vs held baseline_Test"); t.unlink()
    GUARDS.append(f"- {sc}: frozen cleaner on the official attack-free Test archive reproduces the held file exactly ({len(b):,} rows)")
    say(GUARDS[-1][2:])
    ids = b["receiver_id"].unique()
    bare = {str(h).split("\\")[-1]: h for h in ids}
    if len(bare) != len(ids):
        stop(f"{sc}: held receiver ids are not unique after removing the prefix")
    kb = b[JK].sort_values(JK, kind="stable").reset_index(drop=True)
    for v in ATT:
        dest = HELD / "veremi" / sc / f"{v}_Test.parquet"
        if dest.exists():
            if rec.get(rel(dest)) != md5f(dest):
                stop(f"{rel(dest)} exists but was not written by a recorded A20 attempt")
            say(f"{sc}/{v}: cleaned file from earlier attempt (md5 verified)"); continue
        t = TMP / f"{sc}_{v}_Test.parquet"
        PP.clean_veremi_split(ZIPS, sc, v, "Test", t, tz, 500_000, _Log())
        d = pd.read_parquet(t); t.unlink()
        r = d["receiver_id"].astype(str).str.split("\\", regex=False).str[-1]
        miss = sorted(set(r) - set(bare))
        if miss:
            stop(f"{sc}/{v}: {len(miss)} receivers not in held attack-free Test, e.g. {miss[:3]}")
        d["receiver_id"] = r.map(bare).astype(b["receiver_id"].dtype)
        if set(d["receiver_id"].unique()) != set(ids):
            stop(f"{sc}/{v}: receiver set differs from held attack-free Test")
        ka = d[JK].sort_values(JK, kind="stable").reset_index(drop=True)
        if len(d) != len(b) or not ka.equals(kb):
            stop(f"{sc}/{v}: message keys differ from attack-free Test ({len(d):,} vs {len(b):,} rows)")
        mm = d.drop_duplicates(JK).merge(b.drop_duplicates(JK), on=JK, suffixes=("", "_b"))
        chg = {c: float((mm[c] != mm[c + "_b"]).mean()) if c.endswith("_ns") else
               float((~np.isclose(mm[c].astype(float), mm[c + "_b"].astype(float), equal_nan=True)).mean())
               for c in ["s_x", "s_y", "s_spd", "s_hed", "send_ns", "rcv_ns", "r_x", "r_y"]}
        pq.write_table(pa.Table.from_pandas(d, schema=PP.CLEAN_SCHEMA, preserve_index=False), dest, compression="zstd")
        rec[rel(dest)] = md5f(dest); setst("cleaning")
        DESC.append(f"- {sc}/{v}: {len(d):,} rows, {len(ids):,} receivers (ids remapped to held form); "
                    "share of receptions changed vs attack-free: " + ", ".join(f"{c} {x:.3f}" for c, x in chg.items()))
        say(DESC[-1][2:])


def evidence_block(cfg, sc, var):
    """Per-variant block of preprocess.run_traces (held-out branch), copied verbatim; inputs = existing mapping/handovers."""
    T, sp = cfg["traces"], "Test"
    out = HELD / sc / sp
    mp = pd.read_parquet(out / "mapping.parquet")
    ho = pd.read_parquet(out / "handovers.parquet")
    lookup = {r["sender_id"]: (np.array([], np.int64), [r["init_edge"]]) for r in mp.to_dict("records")}
    for sid, g in ho.sort_values(["sender_id", "ho_idx"]).groupby("sender_id"):
        lookup[sid] = (g.t_exec_ns.to_numpy(np.int64), [lookup[sid][1][0]] + g.dst.astype(int).tolist())
    delay_ns = int(round(T["report_delay_s"] * PP.NS_PER_S))
    df = pd.read_parquet(HELD / "veremi" / sc / f"{var}_{sp}.parquet")
    df = df[df.sender_id.isin(mp.sender_id)]
    ev = PP.evidence_features(df)
    uu = df.drop_duplicates(["sender_id", "message_id"]).sort_values(["sender_id", "send_ns"])
    gg = uu.groupby("sender_id")
    first_lab, any_att = gg.attacker.first(), gg.attacker.max()
    first_att = uu[uu.attacker == 1].groupby("sender_id").send_ns.min()
    sess = mp.set_index("sender_id").copy()
    sess["compromised"] = any_att.reindex(sess.index).fillna(0).astype("int8")
    fl = (first_lab.reindex(sess.index).fillna(0) == 1).to_numpy()
    onset = np.full(len(sess), np.nan)
    onset[fl] = sess.first_ns.to_numpy(float)[fl]
    rest = (sess.compromised.to_numpy() == 1) & ~fl
    onset[rest] = first_att.reindex(sess.index).to_numpy(float)[rest]
    sess["onset_ns"] = pd.array(np.where(np.isnan(onset), np.nan, onset), dtype="Float64").astype("Int64")
    rep = ev.rcv_ns.to_numpy(np.int64) + delay_ns
    edge = np.empty(len(ev), np.int16)
    for sid, idx in ev.groupby("sender_id").indices.items():
        tx, E = lookup[sid]
        edge[idx] = np.asarray(E, dtype=np.int16)[np.searchsorted(tx, rep[idx], side="right")]
    ev["report_ns"], ev["edge"] = rep, edge
    on = ev.sender_id.map(pd.Series(onset, index=sess.index)).to_numpy(float)
    with np.errstate(invalid="ignore"):
        ev["gt_x"] = (ev.send_ns.to_numpy(float) >= on).astype("int8")
    ev["gt_msg_attack"] = ev.attacker.astype("int8")
    ev = ev[PP.EV_COLS]
    sess["n_evidence"] = ev.groupby("sender_id").size().reindex(sess.index).fillna(0).astype(int)
    sess = sess.reset_index()
    sess.insert(0, "session_uid", f"{sc}|{var}|{sp}|" + sess.sender_id)
    sess.insert(1, "scenario", sc)
    sess.insert(2, "variant", var)
    sess.insert(3, "split", sp)
    return ev, sess


def stage_traces(cfg, sc, rec):
    d = HELD / sc / "Test"
    eb, sb = evidence_block(cfg, sc, "baseline")
    same(eb, pd.read_parquet(d / "evidence_baseline.parquet"), f"{sc}: rebuilt evidence_baseline")
    same(sb, pd.read_parquet(d / "sessions_baseline.parquet"), f"{sc}: rebuilt sessions_baseline")
    GUARDS.append(f"- {sc}: copied run_traces block rebuilds evidence_baseline ({len(eb):,} rows) and sessions_baseline exactly")
    say(GUARDS[-1][2:])
    for v in ATT:
        fe, fs = d / f"evidence_{v}.parquet", d / f"sessions_{v}.parquet"
        if fe.exists() or fs.exists():
            if rec.get(rel(fe)) != md5f(fe) or rec.get(rel(fs)) != md5f(fs):
                stop(f"{rel(fe)} / {rel(fs)} exist but were not written by a recorded A20 attempt")
            say(f"{sc}/{v}: evidence/sessions from earlier attempt (md5 verified)"); continue
        ev, se = evidence_block(cfg, sc, v)
        ev.to_parquet(fe, index=False); se.to_parquet(fs, index=False)
        rec[rel(fe)], rec[rel(fs)] = md5f(fe), md5f(fs); setst("traces")
        say(f"{sc}/{v}: evidence {len(ev):,} items, {len(se):,} sessions written")


def stage_lstm(sc, rec):
    r = subprocess.run([str(EV.TORCH_PY), "src/a20_lstm.py", sc, "baseline", str(TMP / "lstm_check" / sc)], cwd=ROOT)
    if r.returncode:
        stop(f"{sc}: LSTM re-scoring of baseline did not reproduce stored Test scores (exit {r.returncode})")
    GUARDS.append(f"- {sc}: frozen LSTM re-scores baseline Test within 1e-4 of stored scores")
    for v in ATT:
        dest = ROOT / "results" / "lstm" / sc / f"{v}_Test.parquet"
        if dest.exists():
            if rec.get(rel(dest)) != md5f(dest):
                stop(f"{rel(dest)} exists but was not written by a recorded A20 attempt")
            say(f"{sc}/{v}: LSTM scores from earlier attempt (md5 verified)"); continue
        r = subprocess.run([str(EV.TORCH_PY), "src/a20_lstm.py", sc, v], cwd=ROOT)
        if r.returncode:
            stop(f"{sc}/{v}: LSTM scoring failed (exit {r.returncode})")
        rec[rel(dest)] = md5f(dest); setst("lstm")


def preflight(cfg):
    for sc in SCN:
        for v in ATT:
            ev, ho, se = MD.load_variant(cfg, sc, "Test", v)
            h = MD.build_episodes(ho, se)
            PI = MD.probe_info(cfg, sc, "Test", v, h)
            raw = MD.ext_item_scores(cfg, sc, "Test", v, ev.assign(variant_=v, llr=0.0))
            bad = [k for k, a in raw.items() if not np.isfinite(np.asarray(a, dtype=float)).all()]
            if "llr_lstm" not in raw or bad:
                stop(f"{sc}/{v}: external scores {sorted(raw)} non-finite {bad}")
            lag = np.hypot(PI["lx"], PI["ly"]).to_numpy()[PI["has"].to_numpy()]
            nz = lag[lag > 1e-6]
            LAGS.append(f"- {sc}/{v}: {len(ev):,} items, {len(h):,} episodes, CAM matched {PI['has'].mean():.3f}; "
                        f"episodes with non-zero position lag {len(nz) / max(len(lag), 1):.3f}, "
                        f"median non-zero lag {(np.median(nz) if len(nz) else float('nan')):.1f} m")
            say("preflight " + LAGS[-1][2:])


def guard_vs_primary(sc, sd):
    a = json.loads((RUN / f"manifest_{sc}_s{sd}.json").read_text())
    b = json.loads((PRIM / f"manifest_{sc}_s{sd}.json").read_text())
    for k in ["operating", "aggregator_coefs_std", "aggregator_train_auc"]:
        if json.dumps(a[k], sort_keys=True, default=str) != json.dumps(b[k], sort_keys=True, default=str):
            stop(f"seed {sd} {sc}: Train fit differs from primary Test run ({k})")
    flt = [("variant", "==", "baseline")]
    na = pd.read_parquet(RUN / f"episodes_{sc}_s{sd}.parquet", filters=flt).sort_values(K, kind="stable")
    nb = pd.read_parquet(PRIM / f"episodes_{sc}_s{sd}.parquet", filters=flt).sort_values(K, kind="stable")
    same(na, nb, f"seed {sd} {sc}: baseline-variant episodes vs primary")
    GUARDS.append(f"- seed {sd} {sc}: Train fit identical to primary; baseline episodes identical ({len(na):,} rows)")
    say(GUARDS[-1][2:])


def average():
    AVG.mkdir(parents=True, exist_ok=True)
    for sc in SCN:
        base = acc = None
        for sd in SEEDS:
            d = pd.read_parquet(RUN / f"episodes_{sc}_s{sd}.parquet", columns=K + ["x", "loss", "action"])
            d = d.sort_values(K, kind="stable").reset_index(drop=True)
            if base is None:
                base, acc = d, d["loss"].to_numpy(dtype=float).copy()
            else:
                for c in K + ["x"]:
                    if len(d) != len(base) or not np.array_equal(base[c].to_numpy(), d[c].to_numpy()):
                        stop(f"episode keys differ, seed 0 vs {sd} ({sc}, {c})")
                acc += d["loss"].to_numpy(dtype=float)
        base["loss"] = acc / len(SEEDS)
        base.to_parquet(AVG / f"episodes_{sc}_s0.parquet", index=False)
        say(f"5-seed average written: {sc} ({len(base):,} episodes)")


def xmetrics(cm, idx, m):
    is_m = (idx["method"] == m).to_numpy()
    menu = idx["profile"].isin(ST.MENU).to_numpy()
    nom = np.flatnonzero(is_m & (idx["profile"] == "nominal").to_numpy())
    men = np.flatnonzero(is_m & menu)
    pos = np.flatnonzero(is_m & menu & idx["variant"].isin(ATT).to_numpy())
    if (len(nom), len(men), len(pos)) != (3, 12, 8):
        stop(f"{m}: cells nom={len(nom)} menu={len(men)} pos={len(pos)} (expected 3/12/8)")
    return {"nominal_mean": cm[..., nom].mean(-1), "menu_mean": cm[..., men].mean(-1),
            "menu_worst": cm[..., men].max(-1), "pos_worst": cm[..., pos].max(-1)}


def xcriteria(d, B=2000, seed=0):
    methods = [FULL] + ST.BASELINES + ST.ABLATIONS + ST.EXTENDED + ST.SENS + ST.ABL_LSTM
    miss = sorted(set(methods) - set(d["method"].unique()))
    if miss:
        stop(f"methods missing from episodes: {miss}")
    d = d[d["method"].isin(methods)]
    g = d.groupby(["method", "profile", "variant", "cl"])["loss"].agg(["sum", "count"])
    S = g["sum"].unstack("cl", fill_value=0.0)
    N = g["count"].unstack("cl", fill_value=0).reindex(index=S.index, columns=S.columns)
    idx, S, N = S.index.to_frame(index=False), S.to_numpy(float), N.to_numpy(float)
    pt, C = S.sum(1) / N.sum(1), S.shape[1]
    W = np.random.default_rng(seed).multinomial(C, np.full(C, 1.0 / C), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ S.T) / (W @ N.T)
    P = {m: xmetrics(pt, idx, m) for m in methods}
    BS = {m: xmetrics(bs, idx, m) for m in methods}
    rows = []
    for m in methods:
        for k in METRICS:
            lo, hi = np.nanpercentile(BS[m][k] - BS[FULL][k], [2.5, 97.5])
            rows.append(dict(method=m, metric=k, level=float(P[m][k]), diff=float(P[m][k] - P[FULL][k]),
                             lo=float(lo), hi=float(hi), sig="*" if (lo > 0 or hi < 0) else ""))
    return pd.DataFrame(rows), C, P, BS


def per_seed_tables():
    hs, pr = [], []
    for sd in SEEDS:
        for sc in SCN:
            d = pd.read_parquet(RUN / f"episodes_{sc}_s{sd}.parquet",
                                columns=["method", "profile", "variant", "x", "action", "probed", "probe_pos"])
            n = d[d["profile"] == "nominal"].copy()
            n["probed"], n["probe_pos"] = n["probed"].astype(bool), n["probe_pos"].astype(bool)
            hs.append(n[n["x"] == 0].groupby("method")["action"].apply(lambda a: float((a >= 1).mean())).rename(f"{sc}|{sd}"))
            for (m, v), g2 in n[n["method"].isin(["robust_v1+probe", "lstm_v1+probe"])].groupby(["method", "variant"]):
                at, hon = g2[g2["x"] == 1], g2[g2["x"] == 0]
                pr.append(dict(scen=sc, seed=sd, method=m, variant=v,
                               probed_share_attacked=float(at["probed"].mean()) if len(at) else np.nan,
                               detection_probed=float(at.loc[at["probed"], "probe_pos"].mean()) if at["probed"].any() else np.nan,
                               false_alarm_probed=float(hon.loc[hon["probed"], "probe_pos"].mean()) if hon["probed"].any() else np.nan))
    H = pd.concat(hs, axis=1)
    H5 = pd.DataFrame({sc: H[[c for c in H.columns if c.startswith(sc + "|")]].mean(axis=1) for sc in SCN}).round(3)
    PR = pd.DataFrame(pr).groupby(["method", "variant", "scen"])[["probed_share_attacked", "detection_probed", "false_alarm_probed"]].mean().round(3)
    return H5, PR


def item_aucs():
    out = []
    for sc in SCN:
        man = json.loads((RUN / f"manifest_{sc}_s0.json").read_text())
        out.append(f"- {sc}: item model (key 'item_auc_validation' is hard-coded in the frozen manifest; here = Test): "
                   + json.dumps({k: round(float(v), 3) for k, v in man["item_auc_validation"].items()}))
        for v in ATT:
            s = pd.read_parquet(ROOT / "results" / "lstm" / sc / f"{v}_Test.parquet")
            e = pd.read_parquet(HELD / sc / "Test" / f"evidence_{v}.parquet", columns=MD.EXT_KEYS + ["gt_msg_attack"])
            m = e.merge(s, on=MD.EXT_KEYS, how="left")
            out.append(f"- {sc}/{v}: LSTM item AUC {roc_auc_score(m['gt_msg_attack'], m['llr_lstm']):.3f} ({len(m):,} items)")
    return out


CONFIG_ADDED = [("hetnet", "gate"), ("pathfold",)]   # A15 and A16 additions to config.yaml after the pin


def frozen_check_a20():
    """Code byte-identical to 126e05a; config.yaml identical except the A15/A16 blocks (absent at the pin)."""
    pin, code = "126e05a", ["src/model.py", "src/stats.py", "src/train.py"]
    if sorted(code + ["config/config.yaml"]) != sorted(EV.FROZEN):
        stop(f"evaluate.FROZEN is {EV.FROZEN}, expected {code + ['config/config.yaml']}")
    if EV.git("rev-parse", EV.FREEZE_REF).stdout.strip() != EV.git("rev-parse", pin).stdout.strip():
        stop(f"evaluate.FREEZE_REF {EV.FREEZE_REF} is not {pin}")
    dirty = EV.git("status", "--porcelain", "--", *EV.FROZEN).stdout.strip()
    if dirty:
        stop(f"uncommitted changes in frozen files: {dirty}")
    if EV.git("diff", "--quiet", pin, "HEAD", "--", *code).returncode != 0:
        stop(f"model.py / stats.py / train.py changed since {pin}")
    r = EV.git("show", f"{pin}:config/config.yaml")
    if r.returncode != 0:
        stop(f"cannot read config.yaml at {pin}")
    old, new = yaml.safe_load(r.stdout), yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    for path in CONFIG_ADDED:
        o, n = old, new
        for k in path[:-1]:
            o, n = o.get(k, {}), n[k]
        if path[-1] in o:
            stop(f"config key {'.'.join(path)} already present at {pin}")
        if path[-1] not in n:
            stop(f"config key {'.'.join(path)} not present at HEAD")
        del n[path[-1]]
    if old != new:
        stop(f"config.yaml differs from {pin} beyond the A15 hetnet.gate and A16 pathfold blocks")
    GUARDS.append(f"- model.py, stats.py, train.py byte-identical to {pin}; config.yaml identical to {pin} "
                  "except the A15 (hetnet.gate) and A16 (pathfold) blocks")
    return {f: EV.sha(f) for f in EV.FROZEN}


def main():
    global L
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    say("=== [0/7] guards before any Test read")
    hashes = frozen_check_a20()
    if EV.git("status", "--porcelain", "--", "src/a20.py", "src/a20_lstm.py").stdout.strip():
        stop("commit src/a20.py and src/a20_lstm.py first")
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        stop("evidence_item_bytes must be 256")
    if not EV.TORCH_PY.exists():
        stop(f"{EV.TORCH_PY} missing")
    a20 = a20_text()
    for z, h in ZMD5.items():
        if not (ZIPS / z).exists() or md5f(ZIPS / z) != h:
            stop(f"archive missing or MD5 mismatch: {z}")
    source_guard()
    f = build_split()
    say("frozen code unchanged since 126e05a, config differs only by A15/A16 blocks; A20 section found; 6 archives MD5 OK; evaluation function compiled")
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(exist_ok=True)
    L = json.loads(LOCKF.read_text()) if LOCKF.exists() else {"created": datetime.now().isoformat(timespec="seconds"), "attempts": []}
    if L.get("status") == "complete":
        stop("A20 already complete (run-once lock). See docs/a20_report.md")
    L["attempts"].append({"start": datetime.now().isoformat(timespec="seconds"),
                          "git": EV.git("rev-parse", "--short", "HEAD").stdout.strip(), "sha256": hashes})
    rec = L.setdefault("files", {})
    setst("snapshot")
    say("=== [1/7] fingerprint of protected results (md5) and data (size+time); a few minutes")
    s0 = snapshot(); say(f"{len(s0):,} protected files recorded")
    EV._TEST_OPEN = True
    try:
        say("=== [2/7] clean official Test files"); setst("cleaning")
        for sc in SCN:
            stage_clean(sc, rec)
        say("=== [3/7] evidence + sessions (frozen run_traces block)"); setst("traces")
        for sc in SCN:
            stage_traces(cfg, sc, rec)
        say("=== [4/7] LSTM-Det Test scores (frozen model.pt)"); setst("lstm")
        for sc in SCN:
            stage_lstm(sc, rec)
        say("=== [5/7] preflight"); setst("preflight")
        preflight(cfg)
        say("=== [6/7] Test simulation: 2 scenarios x 5 seeds"); setst("simulating")
        jobs = [(sd, sc) for sd in SEEDS for sc in SCN]
        done = lambda j: (RUN / f"episodes_{j[1]}_s{j[0]}.parquet").exists() and (RUN / f"manifest_{j[1]}_s{j[0]}.json").exists()
        dur = {}
        for i, (sd, sc) in enumerate(jobs, 1):
            if done((sd, sc)):
                say(f"=== [{i}/{len(jobs)}] seed {sd} {sc}: done in an earlier attempt (reused, guard re-run)")
            else:
                known = [x for v in dur.values() for x in v]
                rem = [np.mean(dur[s]) if s in dur else np.mean(known) for (_, s) in jobs[i - 1:] if not done((_, s))] if known else []
                eta = f"~{sum(rem) / 60:.0f} min left incl. this job" if known else "ETA after the first job"
                say(f"=== [{i}/{len(jobs)}] seed {sd} {sc}: START | {eta}")
                setst(f"simulating seed {sd} {sc}")
                ts = time.time()
                f(cfg, sc, sd, "Test", RUN)
                dur.setdefault(sc, []).append(time.time() - ts)
                say(f"=== [{i}/{len(jobs)}] seed {sd} {sc}: DONE in {(time.time() - ts) / 60:.1f} min")
            guard_vs_primary(sc, sd)
        say("=== [7/7] 5-seed average, bootstrap, report"); setst("stats")
        average()
        dd = []
        for sc in SCN:
            d = pd.read_parquet(AVG / f"episodes_{sc}_s0.parquet", columns=["session_uid", "x", "method", "profile", "variant", "loss", "action"])
            d["scen"], d["cl"] = sc, sc + "|" + d["session_uid"].str.split("|").str[-1]
            dd.append(d)
        D = pd.concat(dd, ignore_index=True)
        T, C, P, BS = xcriteria(D)
        ci = lambda a, b, k: np.nanpercentile(BS[a][k] - BS[b][k], [2.5, 97.5])
        best, pair, res = [], [], {"clusters": C, "levels": {m: {k: float(P[m][k]) for k in METRICS} for m in P}}
        for k in METRICS:
            bm = min(POOL10, key=lambda m: P[m][k]); lo, hi = ci(bm, FULL, k)
            vd = f"{FULL} sig. better" if lo > 0 else (f"{bm} sig. better" if hi < 0 else "n.s.")
            best.append(f"- {k}: best of 10 = {bm} {P[bm][k]:.4f} vs {FULL} {P[FULL][k]:.4f}; 95% CI of diff [{lo:.4f}, {hi:.4f}] -> {vd}")
            res[f"best10_{k}"] = {"method": bm, "level": float(P[bm][k]), "full": float(P[FULL][k]), "lo": float(lo), "hi": float(hi), "verdict": vd}
            lo2, hi2 = ci("lstm_sev", "lstm_v1+probe", k)
            vd2 = "TRACE pipeline sig. better" if lo2 > 0 else ("lstm_sev sig. better" if hi2 < 0 else "n.s.")
            pair.append(f"- {k}: lstm_v1+probe {P['lstm_v1+probe'][k]:.4f} vs lstm_sev {P['lstm_sev'][k]:.4f}; 95% CI [{lo2:.4f}, {hi2:.4f}] -> {vd2}")
            res[f"paired_lstm_{k}"] = {"abl": float(P["lstm_v1+probe"][k]), "base": float(P["lstm_sev"][k]), "lo": float(lo2), "hi": float(hi2), "verdict": vd2}
        H5, PR = per_seed_tables()
        aucs = item_aucs()
        ps = []
        for sc, g in D.groupby("scen"):
            cm = g.groupby(["method", "profile", "variant"])["loss"].mean()
            ix = cm.index.to_frame(index=False)
            for m in [FULL] + POOL10 + ["lstm_v1+probe", "robust_v1"]:
                ps.append(dict(scen=sc, method=m, **{k: float(v) for k, v in xmetrics(cm.to_numpy(), ix, m).items()}))
        PS = pd.DataFrame(ps)
        s1 = snapshot()
        changed = [k for k in s0 if s1.get(k) != s0[k]]
        extra = sorted(set(s1) - set(s0) - whitelist())
        prot_ok = not changed and not extra
        R = [f"# A20 extension: official position attacks, held-out Test ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"- scenarios {SCN}; attacks {ATT}; seeds {SEEDS} (5-seed averaged episode loss); 256 B; B=2000; clusters {C} (scenario, vehicle)",
             f"- git {L['attempts'][-1]['git']}; attempts {len(L['attempts'])} (disclose if > 1); runtime {(time.time() - T0) / 60:.0f} min",
             f"- protected results/data untouched: {prot_ok}" + (f" | changed {changed[:5]} | unexpected new {extra[:5]}" if not prot_ok else ""),
             "- diff = method - " + FULL + " (positive = " + FULL + " better); pos_worst = worst menu cell over the two position attacks",
             "", "## A20 protocol (committed text, verbatim)", ""] + a20 + \
            ["", "## Guards"] + GUARDS + ["", "## Inputs (label-free description)"] + DESC + \
            ["", "## Probe inputs per attack"] + LAGS + ["", "## Item AUC"] + aucs
        for k in METRICS:
            R += ["", f"## {k}", "```", T[T["metric"] == k].drop(columns="metric").sort_values("level").round(4).to_string(index=False), "```"]
        R += ["", "## TRACE-MEC vs best of the 10 performance baselines"] + best
        R += ["", "## Paired: TRACE-MEC-LSTM (lstm_v1+probe) vs LSTM-Det (lstm_sev), same scores"] + pair
        R += ["", "## Honest restrict-or-quarantine share, nominal, mean over seeds (target 0.10)", "```", H5.to_string(), "```"]
        R += ["", "## Probe outcome, nominal, mean over seeds", "```", PR.to_string(), "```"]
        for k in METRICS:
            R += ["", f"## Per-scenario {k} (point estimates)", "```", PS.pivot(index="method", columns="scen", values=k).round(4).to_string(), "```"]
        (ROOT / "docs" / "a20_report.md").write_text("\n".join(R) + "\n")
        res.update(honest_share=H5.to_dict(), probe=PR.reset_index().to_dict("records"), protected_ok=prot_ok)
        (ROOT / "docs" / "a20_results.json").write_text(json.dumps(res, indent=2, default=str))
        if not prot_ok:
            setst("aborted_protect", changed=changed[:20], extra=extra[:20])
            stop("protected files changed or unexpected new files (see docs/a20_report.md); report written but run NOT closed")
        shutil.rmtree(TMP, ignore_errors=True)
        setst("complete", end=datetime.now().isoformat(timespec="seconds"))
        print("=== RESULT: A20 COMPLETE (lock closed)")
        print("\n".join(best)); print("--- paired LSTM"); print("\n".join(pair))
        print("--- probe (nominal)\n" + PR.to_string())
        print("--- honest share (target 0.10)\n" + H5.loc[[m for m in [FULL, "lstm_sev", "central_ma", "severity"] if m in H5.index]].to_string())
        print(f"- protected files untouched: {prot_ok}\nFull report: docs/a20_report.md")
    except SystemExit as e:
        if e.code not in (0, None) and L.get("status") not in ("aborted_protect",):
            setst("aborted", error=f"exit {e.code}")
        raise
    except Exception as e:
        setst("aborted", error=repr(e)[:800])
        raise


if __name__ == "__main__":
    main()
