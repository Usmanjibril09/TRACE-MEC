"""evaluate.py (script 5/8) - the single held-out Test evaluation.
Protocol: docs/theory.md A5 (Test), A6 (reported metrics), A9 (extended baselines secondary), A11 (trace_lstm),
A13 outcome (item=256 B primary; item=2676 B ML-DSA-44 one secondary point).
  --dryrun [--scenarios ...]  same code path on Validation; must reproduce results/model_smoke episodes bit-for-bit.
  --final [--resume | --rerun-after-abort]   Test, ONCE: frozen-code guard -> LSTM Test scoring (label-free)
      -> label-free preflight -> item=256 and item=2676 on 4 scenarios -> stats.py main() unchanged -> A6 -> report.
The simulation function IS model.smoke(), re-compiled at run time with asserted split/output substitutions."""
import argparse, contextlib, copy, hashlib, inspect, io, json, pathlib, re, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model as MD  # noqa: E402
import stats as ST  # noqa: E402

SCEN, MENU = list(ST.SCEN), list(ST.MENU)
SIZES = [(256, "primary, unsigned"), (2676, "secondary, ML-DSA-44 (A13 outcome)")]
FROZEN = ["src/model.py", "src/stats.py", "src/train.py", "config/config.yaml"]
FREEZE_REF = "126e05a"
PROTECT = ["results/model_smoke", "results/frozen_v071"]
FINAL, DRY = ROOT / "results" / "final", ROOT / "results" / "final_dryrun"
LOCK = FINAL / "LOCK.json"
TORCH_PY = ROOT / ".venv_torch" / "bin" / "python"
A6_COLS = ["TPR", "FPR", "precision", "F1", "ROC_AUC", "PR_AUC", "TPR@FPR5", "TPR@FPR10", "TPR@FPR20",
           "bytes_per_decision", "deadline_met", "probes_per_HO"]
_TEST_OPEN = False


def say(m):
    MD.log(m)


def now():
    return datetime.now().isoformat(timespec="seconds")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


_orig_trace_dir = MD.trace_dir


def _trace_dir(cfg, sc, split):
    if split != "Test":
        return _orig_trace_dir(cfg, sc, split)
    if not _TEST_OPEN:
        raise SystemExit("REFUSED: Test is opened only inside evaluate.py --final after all guards.")
    d = ROOT / (cfg.get("paths") or {}).get("heldout", "data/heldout") / sc / "Test"
    if not d.exists():
        raise SystemExit(f"Missing held-out trace dir {d}")
    return d


MD.trace_dir = _trace_dir

SUBS = [
    ("def smoke(cfg, sc, seed):", "def eval_split(cfg, sc, seed, split, out_dir):"),
    ('load_variant(cfg, sc, "Validation", v)', "load_variant(cfg, sc, split, v)"),
    ('ext_apply(cfg, sc, "Validation", v, ev, ctx)', "ext_apply(cfg, sc, split, v, ev, ctx)"),
    ('probe_info(cfg, sc, "Validation", v, build_episodes(ho, se))', "probe_info(cfg, sc, split, v, build_episodes(ho, se))"),
    ('log(f"simulated {sc}/Validation/{v}")', 'log(f"simulated {sc}/{split}/{v}")'),
    ('out = ROOT / cfg["paths"]["results"] / "model_smoke"', "out = out_dir"),
    ("evaluated on Validation; seed", "evaluated on {split}; seed"),
    ('(ROOT / "docs" / "model_smoke.md").write_text', '(out / f"report_{sc}_s{seed}.md").write_text'),
    ('"script": "model.py --smoke"', '"script": "evaluate.py", "split": split'),
]


def build_eval_split():
    src = inspect.getsource(MD.smoke)
    for a, b in SUBS:
        n = src.count(a)
        if n != 1:
            raise SystemExit(f"ABORT: expected exactly 1 occurrence in model.smoke() of {a!r}, found {n}")
        src = src.replace(a, b)
    if '"Validation"' in src:
        raise SystemExit("ABORT: model.smoke() still references \"Validation\" after substitution")
    exec(compile(src, "<evaluate: model.smoke with split/output substitutions>", "exec"), MD.__dict__)
    return MD.__dict__["eval_split"]


def load_cfg():
    m = re.search(r"^\s*cfg\s*=\s*(.+?)\s*$", inspect.getsource(MD.main), re.M)
    if m:
        try:
            c = eval(m.group(1), dict(MD.__dict__))
            if isinstance(c, dict):
                return c, f"model.main expression: {m.group(1)}"
        except Exception:
            pass
    return yaml.safe_load((ROOT / "config/config.yaml").read_text()), "yaml.safe_load(config/config.yaml)"


def frozen_check(self_committed):
    files = FROZEN + (["src/evaluate.py"] if self_committed else [])
    dirty = git("status", "--porcelain", "--", *files).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted changes in frozen files (commit first):\n{dirty}")
    if git("diff", "--quiet", FREEZE_REF, "HEAD", "--", *FROZEN).returncode != 0:
        raise SystemExit(f"ABORT: frozen files changed since {FREEZE_REF}")
    return {f: sha(f) for f in files}


def snapshot():
    h = {}
    for d in PROTECT:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file():
                h[str(p.relative_to(ROOT))] = hashlib.md5(p.read_bytes()).hexdigest()
    return h


def compare(new, ref):
    a, b = pd.read_parquet(new), pd.read_parquet(ref)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, (f"shape/columns differ {a.shape} vs {b.shape}; only-new {sorted(set(a.columns) - set(b.columns))} "
                       f"only-ref {sorted(set(b.columns) - set(a.columns))}")
    bad = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float, na_value=np.nan), y.to_numpy(dtype=float, na_value=np.nan)
            if not np.array_equal(xv, yv, equal_nan=True):
                bad.append(f"{c} (max|diff| {np.nanmax(np.abs(xv - yv)):.3g})")
        elif not (x.astype(str).to_numpy() == y.astype(str).to_numpy()).all():
            bad.append(c)
    return (not bad), ("identical, all columns" if not bad else "differing: " + ", ".join(bad))


def dryrun(cfg, scs, seed, how):
    frozen_check(False)
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit("ABORT: evidence_item_bytes must be 256")
    s0, f = snapshot(), build_eval_split()
    out, res = DRY / "item256", {}
    for sc in scs:
        f(cfg, sc, seed, "Validation", out)
        ok, msg = compare(out / f"episodes_{sc}_s{seed}.parquet", ROOT / "results/model_smoke" / f"episodes_{sc}_s{seed}.parquet")
        res[sc] = msg
        say(f"DRYRUN {sc}: {'PASS' if ok else 'FAIL'} - {msg}")
        if not ok:
            raise SystemExit(f"=== RESULT: DRYRUN FAIL on {sc} ({msg}). Do NOT run --final; paste this.")
    if snapshot() != s0:
        raise SystemExit("=== RESULT: DRYRUN FAIL - protected results/model_smoke or frozen_v071 changed")
    (DRY / "dryrun_ok.json").write_text(json.dumps({"time": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(),
        "evaluate_sha": sha("src/evaluate.py"), "seed": seed, "scenarios": scs, "cfg_loader": how, "result": res}, indent=2))
    print(f"=== RESULT: DRYRUN PASS ({', '.join(scs)}): Validation episodes reproduced bit-for-bit; protected results untouched.")


def run_stats(res_dir, seed, out_doc, label, scratch=None):
    fake = (scratch or FINAL) / f"_stats_root_{res_dir.name}"
    if fake.exists():
        shutil.rmtree(fake)
    (fake / "results").mkdir(parents=True)
    (fake / "docs").mkdir()
    (fake / "results" / "model_smoke").symlink_to(res_dir.resolve(), target_is_directory=True)
    old = (ST.ROOT, sys.argv)
    ST.ROOT = fake
    sys.argv = ["stats.py", "--criteria", "--seed", str(seed), "--B", "2000", "--version", "v0.7.1", "--out", "docs/criteria.txt"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ST.main()
    finally:
        ST.ROOT, sys.argv = old
    txt = (fake / "docs" / "criteria.txt").read_text()
    hdr = (f"# HELD-OUT TEST, {label}\n# produced by stats.py main() UNCHANGED (frozen code) on {res_dir.relative_to(ROOT)}; "
           "the frozen title line below says 'Validation' because it is hard-coded.\n\n")
    (ROOT / out_doc).write_text(hdr + txt)
    shutil.copy(fake / "docs" / "criteria.json", (ROOT / out_doc).with_suffix(".json"))
    return txt


def a6(res_dir, seed, nb):
    rows = []
    for sc in SCEN:
        d = pd.read_parquet(res_dir / f"episodes_{sc}_s{seed}.parquet",
                            columns=["x", "p", "action", "method", "profile", "n_sent", "n_arrived", "probed"])
        d = d[d["profile"].isin(["nominal"] + MENU)]
        for (m, pr), g in d.groupby(["method", "profile"], sort=False):
            x, p, pos = g["x"].to_numpy().astype(int), g["p"].to_numpy(dtype=float), g["action"].to_numpy() >= 1
            n1, n0 = int((x == 1).sum()), int((x == 0).sum())
            tp, fp, fn = int((pos & (x == 1)).sum()), int((pos & (x == 0)).sum()), int((~pos & (x == 1)).sum())
            r = {"scen": sc, "method": m, "profile": pr, "TPR": tp / n1 if n1 else np.nan, "FPR": fp / n0 if n0 else np.nan,
                 "precision": tp / (tp + fp) if tp + fp else np.nan, "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}
            if n1 and n0:
                fpr, tpr, _ = roc_curve(x, p)
                r.update(ROC_AUC=roc_auc_score(x, p), PR_AUC=average_precision_score(x, p),
                         **{f"TPR@FPR{int(round(100 * f))}": float(np.interp(f, fpr, tpr)) for f in (0.05, 0.10, 0.20)})
            ns, na = g["n_sent"].to_numpy(dtype=float), g["n_arrived"].to_numpy(dtype=float)
            r.update(bytes_per_decision=float(np.nanmean(ns)) * nb,
                     deadline_met=float(np.nansum(na) / np.nansum(ns)) if np.nansum(ns) > 0 else np.nan,
                     probes_per_HO=float(g["probed"].astype(float).mean()))
            rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv(res_dir / "a6_metrics_per_scenario_profile.csv", index=False)
    per["group"] = np.where(per["profile"] == "nominal", "nominal", "menu_mean")
    pooled = per.groupby(["group", "method"])[[c for c in A6_COLS if c in per.columns]].mean()
    pooled.to_csv(res_dir / "a6_metrics_pooled.csv")
    return pooled


def size_guards(seed):
    out = []
    for sc in SCEN:
        e = {nb: pd.read_parquet(FINAL / f"item{nb}" / f"episodes_{sc}_s{seed}.parquet", columns=["method", "p", "action"]) for nb, _ in SIZES}
        for m in ["no_transfer", "no_transfer_rb"]:
            x, y = e[256][e[256].method == m], e[2676][e[2676].method == m]
            same = len(x) == len(y) and np.array_equal(x.p.to_numpy(), y.p.to_numpy()) and np.array_equal(x.action.to_numpy(), y.action.to_numpy())
            out.append(f"- {sc} {m}: {'identical' if same else 'DIFFERENT'} at 256 vs 2676 B")
        b = [float(pd.read_csv(FINAL / f"item{nb}" / f"summary_{sc}_s{seed}.csv").query("method == 'severity'")["budget"].mean()) for nb, _ in SIZES]
        out.append(f"- {sc}: mean budget (severity) {b[0]:.2f} -> {b[1]:.2f} items ({'shrinks' if b[1] < b[0] else 'DOES NOT SHRINK'})")
    return out


def final(cfg, seed, resume, rerun):
    global _TEST_OPEN
    hashes = frozen_check(True)
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit("ABORT: evidence_item_bytes must be 256 (primary)")
    okf = DRY / "dryrun_ok.json"
    if not okf.exists():
        raise SystemExit("ABORT: run --dryrun first (it must PASS)")
    ok = json.loads(okf.read_text())
    if ok.get("evaluate_sha") != sha("src/evaluate.py") or "InTAS_highway_2" not in ok.get("scenarios", []) or ok.get("seed") != seed:
        raise SystemExit("ABORT: dry run was made with a different evaluate.py / seed; re-run --dryrun")
    if not TORCH_PY.exists():
        raise SystemExit(f"ABORT: {TORCH_PY} missing")
    FINAL.mkdir(parents=True, exist_ok=True)
    att = {"start": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(), "sha256": hashes,
           "mode": "resume" if resume else "rerun-after-abort" if rerun else "first"}
    if LOCK.exists():
        L = json.loads(LOCK.read_text())
        st = L.get("status", "")
        if st == "complete":
            raise SystemExit("REFUSED: Test evaluation already complete (run-once lock). See docs/final_test_report.md")
        no_res = st in ("lstm_scoring", "aborted_lstm", "preflight", "aborted_preflight")
        if no_res and not rerun:
            raise SystemExit(f"REFUSED: last attempt stopped at '{st}' before any result; use --rerun-after-abort (disclosed)")
        if not no_res and not resume:
            raise SystemExit(f"REFUSED: last attempt stopped at '{st}' after simulation began; use --resume (disclosed)")
        L["attempts"].append(att)
    else:
        if resume or rerun:
            raise SystemExit("REFUSED: nothing to resume")
        L = {"created": att["start"], "attempts": [att]}

    def setst(s, **kw):
        L["status"] = s
        L["attempts"][-1].update(status=s, **kw)
        LOCK.write_text(json.dumps(L, indent=2, default=str))

    setst("lstm_scoring")
    t0, s0, f = time.time(), snapshot(), build_eval_split()
    for sc in SCEN:
        if all((ROOT / "results/lstm" / sc / f"{v}_Test.parquet").exists() for v in MD.VARIANTS) and (resume or rerun):
            say(f"[1/5] LSTM Test scores exist for {sc} (reused)")
            continue
        say(f"[1/5] LSTM Test scoring, label-free: {sc}")
        r = subprocess.run([str(TORCH_PY), "src/train.py", "--stage", "lstm", "--scenario", sc, "--score-split", "Test"], cwd=ROOT)
        if r.returncode != 0:
            setst("aborted_lstm", error=f"train.py exit {r.returncode} on {sc}")
            raise SystemExit(f"=== RESULT: ABORTED at LSTM scoring ({sc}); no result computed. Paste the traceback.")
    setst("preflight")
    _TEST_OPEN, pre = True, []
    try:
        for sc in SCEN:
            for v in MD.VARIANTS:
                ev, ho, se = MD.load_variant(cfg, sc, "Test", v)
                h = MD.build_episodes(ho, se)
                PI = MD.probe_info(cfg, sc, "Test", v, h)
                raw = MD.ext_item_scores(cfg, sc, "Test", v, ev.assign(variant_=v, llr=0.0))
                bad = [k for k, a in raw.items() if not np.isfinite(np.asarray(a, dtype=float)).all()]
                if "llr_lstm" not in raw or bad:
                    raise RuntimeError(f"ext scores {sorted(raw)} non-finite {bad} for {sc}/{v}")
                pre.append(f"- {sc}/{v}: {len(ev):,} items, {len(h):,} episodes, CAM matched {PI['has'].mean():.3f}, ext {sorted(raw)} OK")
                say("preflight " + pre[-1][2:])
                del ev, ho, se, h, PI, raw
    except (Exception, SystemExit) as e:
        setst("aborted_preflight", error=repr(e)[:800])
        raise SystemExit(f"=== RESULT: PREFLIGHT FAILED, no result computed: {e!r}")
    setst("simulating", preflight=pre)
    for nb, lab in SIZES:
        out = FINAL / f"item{nb}"
        c2 = copy.deepcopy(cfg)
        c2["traces"]["evidence_item_bytes"] = nb
        for sc in SCEN:
            if (out / f"report_{sc}_s{seed}.md").exists():
                if not resume:
                    raise SystemExit(f"ABORT: unexpected existing output {out}/report_{sc}_s{seed}.md")
                say(f"[3/5] item={nb} {sc}: completed in an earlier attempt (reused, deterministic)")
                continue
            say(f"[3/5] TEST item={nb} B ({lab}): {sc}")
            f(c2, sc, seed, "Test", out)
    prot_ok = snapshot() == s0
    setst("stats")
    crit = {nb: run_stats(FINAL / f"item{nb}", seed, f"docs/criteria_test_item{nb}.txt", f"item={nb} B, {lab}") for nb, lab in SIZES}
    setst("a6")
    tabs = {nb: a6(FINAL / f"item{nb}", seed, nb) for nb, _ in SIZES}
    sg = size_guards(seed)
    PAT = re.compile(r"Prop\. 1|Thm 2\(b\)|injected-item pool share|probe rate|probe \(sigma|realized honest")
    R = [f"# TRACE-MEC held-out Test evaluation (evaluate.py --final, {now()})", "",
         f"- seed {seed}; git {att['git']}; runtime {time.time() - t0:.0f} s; attempts {len(L['attempts'])} (disclose if > 1)",
         "- protocol: theory.md A5, A6 (reporting metrics), A9, A11, A13 outcome (256 B primary, 2676 B secondary)",
         "- simulation = model.smoke() re-compiled with split/output substitutions only; dry run reproduced Validation bit-for-bit",
         f"- protected results untouched: {prot_ok}", "- frozen sha256: " + json.dumps(hashes), "", "## Attempts"]
    R += [f"- {a['start']} mode={a['mode']} status={a.get('status')} {a.get('error', '')}" for a in L["attempts"]]
    R += ["", "## Preflight (label-free)"] + pre + ["", "## Item-size guards"] + sg
    for nb, _ in SIZES:
        R += ["", f"## Per-scenario checks, item={nb} B (report text says 'Validation nominal' = this evaluation set)"]
        for sc in SCEN:
            R += [f"- {sc}: " + l.strip("- ") for l in (FINAL / f"item{nb}" / f"report_{sc}_s{seed}.md").read_text().splitlines() if PAT.search(l)]
    for nb, lab in SIZES:
        R += ["", f"## {'PRIMARY' if nb == 256 else 'SECONDARY'}: stats.py criteria, item={nb} B ({lab})", "```", crit[nb], "```"]
    for nb, _ in SIZES:
        for gname in ["nominal", "menu_mean"]:
            R += ["", f"## A6 reported metrics (reporting only), item={nb} B, {gname}; unweighted mean over scenarios",
                  "```", tabs[nb].loc[gname].round(4).to_string(), "```"]
    (ROOT / "docs" / "final_test_report.md").write_text("\n".join(R) + "\n")
    setst("complete", end=now(), runtime_s=round(time.time() - t0))
    KEY = re.compile(r"^(C[1-4] |- (nominal_mean|menu_mean|menu_worst|td_worst): best|- A11 rule)")
    print("=== RESULT: TEST EVALUATION COMPLETE (lock closed)")
    for nb, lab in SIZES:
        print(f"--- item={nb} B ({lab})")
        print("\n".join(l for l in crit[nb].splitlines() if KEY.search(l)))
    print("--- guards\n" + "\n".join(sg) + f"\n- protected results untouched: {prot_ok}")
    print("Full report: docs/final_test_report.md")


def seeds_run(cfg, seed_list):
    """A5 completion (A14): Test seeds 1-4 at item=256 B. Live progress, own lock, frozen stats.py."""
    global _TEST_OPEN
    hashes = frozen_check(True)
    if float(cfg["traces"]["evidence_item_bytes"]) != 256.0:
        raise SystemExit("ABORT: evidence_item_bytes must be 256")
    if not LOCK.exists() or json.loads(LOCK.read_text()).get("status") != "complete":
        raise SystemExit("ABORT: the seed-0 Test run must be complete first")
    okf = DRY / "dryrun_ok.json"
    if not okf.exists() or json.loads(okf.read_text()).get("evaluate_sha") != sha("src/evaluate.py"):
        raise SystemExit("ABORT: run --dryrun with this evaluate.py first (must PASS)")
    if sorted(set(seed_list)) != [1, 2, 3, 4]:
        raise SystemExit("ABORT: A5/A14 require exactly seeds 1 2 3 4 (seed 0 done)")
    SL = FINAL / "LOCK_seeds.json"
    L = json.loads(SL.read_text()) if SL.exists() else {"created": now(), "attempts": []}
    if L.get("status") == "complete":
        raise SystemExit("REFUSED: seeds 1-4 already complete. See docs/final_test_report_seeds.md")
    L["attempts"].append({"start": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(), "sha256": hashes})

    def setst(st, **kw):
        L["status"] = st
        L["attempts"][-1].update(status=st, **kw)
        SL.write_text(json.dumps(L, indent=2, default=str))

    out, t0, s0 = FINAL / "item256", time.time(), snapshot()
    mt = {(nb, sc): (FINAL / f"item{nb}" / f"report_{sc}_s0.md").stat().st_mtime for nb, _ in SIZES for sc in SCEN}
    seq = [mt[(256, SCEN[-1])]] + [mt[(2676, sc)] for sc in SCEN]
    est = {sc: max(seq[i + 1] - seq[i], 60.0) for i, sc in enumerate(SCEN)}
    jobs = [(sd, sc) for sd in sorted(set(seed_list)) for sc in SCEN]
    todo = [j for j in jobs if not (out / f"report_{j[1]}_s{j[0]}.md").exists()]
    left = sum(est[sc] for _, sc in todo)
    print(f"=== {len(jobs)} jobs, {len(jobs) - len(todo)} already done; expected ~{left / 3600:.1f} h + stats "
          "(seed-0 timings: " + ", ".join(f"{sc} {est[sc] / 60:.0f} min" for sc in SCEN) + ")", flush=True)
    _TEST_OPEN, f, done_est, done_act = True, build_eval_split(), 0.0, 0.0
    for i, (sd, sc) in enumerate(jobs, 1):
        if (out / f"report_{sc}_s{sd}.md").exists():
            print(f"=== [{i}/{len(jobs)}] seed {sd} {sc}: done in an earlier attempt (reused)", flush=True)
            continue
        setst(f"simulating seed {sd} {sc}")
        ts, r = time.time(), (done_act / done_est if done_est else 1.0)
        print(f"=== [{i}/{len(jobs)}] seed {sd} {sc}: START {now()} | elapsed {(ts - t0) / 60:.0f} min | "
              f"est. remaining ~{left * r / 60:.0f} min", flush=True)
        f(cfg, sc, sd, "Test", out)
        done_act += time.time() - ts; done_est += est[sc]; left -= est[sc]
        print(f"=== [{i}/{len(jobs)}] seed {sd} {sc}: DONE in {(time.time() - ts) / 60:.1f} min", flush=True)
    setst("stats")
    allS = [0, 1, 2, 3, 4]
    crit = {}
    for sd in allS[1:]:
        print(f"=== stats.py (unchanged), seed {sd} only ...", flush=True)
        crit[sd] = run_stats(out, sd, f"docs/criteria_test_item256_s{sd}.txt", f"item=256 B, seed {sd} only")
    pool = FINAL / "item256_seeds0-4"
    pool.mkdir(exist_ok=True)
    K, hs = ["variant", "profile", "method", "session_uid", "ho_idx"], []
    for sc in SCEN:
        print(f"=== averaging episode loss over seeds 0-4: {sc}", flush=True)
        base = acc = None
        for sd in allS:
            d = pd.read_parquet(out / f"episodes_{sc}_s{sd}.parquet", columns=K + ["x", "loss", "action"])
            d = d.sort_values(K, kind="stable").reset_index(drop=True)
            hs.append(d[(d["profile"] == "nominal") & (d["x"] == 0)].groupby("method")["action"]
                      .apply(lambda a: float((a >= 1).mean())).rename(f"{sc}|{sd}"))
            if base is None:
                base, acc = d, d["loss"].to_numpy(dtype=float).copy()
            else:
                for c in K + ["x"]:
                    if len(d) != len(base) or not np.array_equal(base[c].to_numpy(), d[c].to_numpy()):
                        raise SystemExit(f"ABORT: episode keys differ, seed 0 vs {sd} ({sc}, {c})")
                acc += d["loss"].to_numpy(dtype=float)
            del d
        base["loss"] = acc / len(allS)
        base.to_parquet(pool / f"episodes_{sc}_s0.parquet", index=False)
        shutil.copy(out / f"manifest_{sc}_s0.json", pool / f"manifest_{sc}_s0.json")
        ea = pd.concat([pd.read_csv(out / f"eval_alpha_{sc}_s{sd}.csv") for sd in allS], ignore_index=True)
        ks = [c for c in ea.columns if not pd.api.types.is_numeric_dtype(ea[c])]
        ea.groupby(ks, sort=False).mean(numeric_only=True).reset_index().to_csv(pool / f"eval_alpha_{sc}_s0.csv", index=False)
        del base, acc
    print("=== stats.py (unchanged), PRIMARY 5-seed average ...", flush=True)
    cp = run_stats(pool, 0, "docs/criteria_test_item256_seeds0-4.txt",
                   "item=256 B, PRIMARY per A5: episode loss averaged over seeds 0-4 "
                   "(honest-share section here = seed-0 actions; 5-seed honest share in final_test_report_seeds.md)")
    H = pd.concat(hs, axis=1)
    H5 = pd.DataFrame({sc: H[[c for c in H.columns if c.startswith(sc + "|")]].mean(axis=1) for sc in SCEN}).round(3)
    print("=== A6 metrics per seed ...", flush=True)
    tabs, tmp = [], FINAL / "_a6_tmp"
    for sd in allS:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        for sc in SCEN:
            (tmp / f"episodes_{sc}_s{sd}.parquet").symlink_to((out / f"episodes_{sc}_s{sd}.parquet").resolve())
        tabs.append(a6(tmp, sd, 256))
    shutil.rmtree(tmp)
    A6m = pd.concat(tabs).groupby(level=[0, 1]).mean()
    A6m.to_csv(pool / "a6_metrics_pooled_seeds0-4.csv")
    prot_ok = snapshot() == s0
    KEY = re.compile(r"^(C[1-4] |- (nominal_mean|menu_mean|menu_worst|td_worst): best|- A11 rule)")
    R = [f"# TRACE-MEC Test, seeds 0-4 (A5 via A14), item=256 B ({now()})", "",
         f"- runtime {time.time() - t0:.0f} s; attempts {len(L['attempts'])}; protected results untouched: {prot_ok}",
         "- frozen sha256: " + json.dumps(hashes), "", "## PRIMARY (A5): 5-seed averaged episode loss", "```", cp, "```",
         "", "## Honest restrict-or-quarantine share, nominal, mean over seeds 0-4 (target 0.10)", "```", H5.to_string(), "```"]
    for sd in allS[1:]:
        R += ["", f"## Seed {sd} only (key lines)"] + [l for l in crit[sd].splitlines() if KEY.search(l)]
    for gname in ["nominal", "menu_mean"]:
        R += ["", f"## A6 reported metrics, {gname}, mean over seeds 0-4 and scenarios", "```", A6m.loc[gname].round(4).to_string(), "```"]
    (ROOT / "docs" / "final_test_report_seeds.md").write_text("\n".join(R) + "\n")
    setst("complete", end=now(), runtime_s=round(time.time() - t0))
    print("=== RESULT: SEEDS 0-4 COMPLETE")
    print("--- PRIMARY (5-seed average)\n" + "\n".join(l for l in cp.splitlines() if KEY.search(l)))
    for sd in allS[1:]:
        print(f"--- seed {sd}: " + " | ".join(l.split(":")[0] + (" PASS" if "PASS" in l else " FAIL") for l in crit[sd].splitlines() if re.match(r"^C[1-4] ", l)))
    show = [m for m in ["robust_v1+probe", "robust_v1", "lstm_sev", "central_ma", "voi_myopic", "severity"] if m in H5.index]
    print("--- honest share, mean over 5 seeds (target 0.10)\n" + H5.loc[show].to_string())
    print(f"- protected results untouched: {prot_ok}\nFull report: docs/final_test_report_seeds.md")


# ===================== A15 gate evaluation (v0.17) =====================
import pyarrow.parquet as _pq  # noqa: E402

GATE = ROOT / "results" / "gate"
GATE_LOCK = GATE / "LOCK_gate.json"
GATE_DOCS = ROOT / "docs" / "gate"
GATE_SEEDS = [0, 1, 2, 3, 4]
GATE_NB = 256
GATE_VARIANTS = ["gate_typical", "gate_fast"]
GATE_MD5 = ["results/final", "results/lstm", "results/model_smoke", "results/frozen_v071"]
GATE_STAT = ["data/processed", "data/heldout", "data/interim"]
GATE_KEY = re.compile(r"^(C[1-4] |- (nominal_mean|menu_mean|menu_worst|td_worst): best|- A11 rule)")


def _tree_sig(dirs, deep):
    h = {}
    for d in dirs:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            k = str(p.relative_to(ROOT))
            if deep:
                m = hashlib.md5()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 24), b""):
                        m.update(b)
                h[k] = m.hexdigest()
            else:
                s = p.stat()
                h[k] = (s.st_size, s.st_mtime_ns)
    return h


def gate_snapshot():
    return {"md5": _tree_sig(GATE_MD5, True), "stat": _tree_sig(GATE_STAT, False)}


def _snap_diff(a, b):
    out = []
    for kind in ("md5", "stat"):
        x, y = a[kind], b[kind]
        out += [f"{kind}: added {k}" for k in sorted(set(y) - set(x))]
        out += [f"{kind}: removed {k}" for k in sorted(set(x) - set(y))]
        out += [f"{kind}: changed {k}" for k in sorted(set(x) & set(y)) if x[k] != y[k]]
    return out


def _ydiff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append(("added", p))
            elif k not in b:
                out.append(("removed", p))
            else:
                out += _ydiff(a[k], b[k], p)
        return out
    return [] if a == b else [("changed", path)]


def gate_frozen_check():
    code = ["src/model.py", "src/stats.py", "src/train.py"]
    dirty = git("status", "--porcelain", "--", *code, "config/config.yaml", "src/evaluate.py", "src/preprocess.py").stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted changes (commit first):\n{dirty}")
    if git("diff", "--quiet", FREEZE_REF, "HEAD", "--", *code).returncode != 0:
        raise SystemExit(f"ABORT: model.py/stats.py/train.py changed since {FREEZE_REF}")
    r = git("show", f"{FREEZE_REF}:config/config.yaml")
    if r.returncode != 0:
        raise SystemExit(f"ABORT: cannot read config.yaml at {FREEZE_REF}: {r.stderr}")
    d = _ydiff(yaml.safe_load(r.stdout), yaml.safe_load((ROOT / "config/config.yaml").read_text()))
    if not d or any(not (kind == "added" and p.split(".")[-1] == "gate") for kind, p in d):
        raise SystemExit(f"ABORT: config.yaml vs {FREEZE_REF} must differ ONLY by the added A15 gate block; diff = {d}")
    return {f: sha(f) for f in code + ["config/config.yaml", "src/evaluate.py"]}, d


def _primary_complete():
    for p in (LOCK, FINAL / "LOCK_seeds.json"):
        if not p.exists() or json.loads(p.read_text()).get("status") != "complete":
            raise SystemExit(f"ABORT: {p.relative_to(ROOT)} must be complete before the gate")


def gate_cfg(cfg, variant):
    c = copy.deepcopy(cfg)
    if float(c["traces"]["evidence_item_bytes"]) != float(GATE_NB):
        raise SystemExit("ABORT: evidence_item_bytes must be 256")
    if variant == "dataset":
        return c
    r = ROOT / "data" / "gate" / variant
    for k in ("processed", "heldout"):
        if not (r / k).is_dir():
            raise SystemExit(f"ABORT: missing {r / k}")
        c["paths"][k] = str((r / k).relative_to(ROOT))
    return c


class ReadAudit:
    """Fails fast if a gate run reads any file under the frozen trace roots (VeReMi clean messages excepted)."""

    def __init__(self, variant):
        self.variant, self.seen = variant, set()

    def __enter__(self):
        self._orig = pd.read_parquet

        def rp(path, *a, **k):
            self.check(path)
            return self._orig(path, *a, **k)
        pd.read_parquet = rp
        return self

    def __exit__(self, *exc):
        pd.read_parquet = self._orig
        return False

    def check(self, path):
        try:
            rel = pathlib.Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
        except (TypeError, ValueError):
            return
        self.seen.add(rel)
        parts = rel.split("/")
        if (self.variant != "dataset" and len(parts) > 2 and parts[0] == "data"
                and parts[1] in ("processed", "heldout") and parts[2] != "veremi"):
            raise SystemExit(f"ABORT (read audit): gate run '{self.variant}' tried to read frozen trace file {rel}")


def _frames_equal_except(a, b, skip):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, f"shape/columns differ {a.shape} vs {b.shape}"
    bad = [c for c in a.columns if c not in skip
           and not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True))]
    return (not bad), ("identical except " + ",".join(sorted(skip))) if not bad else "differing: " + ",".join(bad)


def gate_evidence_guard(cfg, variant):
    lines = []
    for sc in SCEN:
        for sp in ("Train", "Validation", "Test"):
            sub = "heldout" if sp == "Test" else "processed"
            fr, gt = ROOT / "data" / sub / sc / sp, ROOT / "data" / "gate" / variant / sub / sc / sp
            for v in MD.VARIANTS:
                a = pd.read_parquet(fr / f"evidence_{v}.parquet")
                b = pd.read_parquet(gt / f"evidence_{v}.parquet")
                ok, msg = _frames_equal_except(a, b, {"edge"})
                share = float((a["edge"].to_numpy() != b["edge"].to_numpy()).mean()) if ok else float("nan")
                lf = ROOT / cfg["paths"]["results"] / "lstm" / sc / f"{v}_{sp}.parquet"
                line = (f"- {variant} {sc}/{sp}/{v}: {len(a):,} items, {msg}; edge differs {share:.3f}; "
                        f"LSTM scores {'present' if lf.exists() else 'MISSING'}")
                lines.append(line)
                print(line, flush=True)
                if not ok or not lf.exists():
                    raise SystemExit("=== RESULT: GATE GUARD FAIL (LSTM-reuse guard, A15): " + line)
                del a, b
    return lines


def gate_guard(cfg):
    global _TEST_OPEN
    hashes, yd = gate_frozen_check()
    _primary_complete()
    GATE.mkdir(parents=True, exist_ok=True)
    GATE_DOCS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== [1/3] freeze check OK: model/stats/train unchanged since {FREEZE_REF}; config diff = {yd}", flush=True)
    s0 = gate_snapshot()
    lines = []
    for variant in GATE_VARIANTS:
        print(f"=== [2/3] LSTM-reuse guard (A15): {variant} evidence vs frozen, all columns except edge", flush=True)
        lines += gate_evidence_guard(cfg, variant)
    print("=== [3/3] dataset-path reproduction: gate code path on FROZEN traces, InTAS_highway_2 Test seed 0", flush=True)
    out = GATE / "_guard_dataset"
    if out.exists():
        shutil.rmtree(out)
    f, _TEST_OPEN = build_eval_split(), True
    ts = time.time()
    with ReadAudit("dataset"):
        f(gate_cfg(cfg, "dataset"), "InTAS_highway_2", 0, "Test", out)
    ok, msg = compare(out / "episodes_InTAS_highway_2_s0.parquet", FINAL / "item256" / "episodes_InTAS_highway_2_s0.parquet")
    print(f"=== [3/3] {'PASS' if ok else 'FAIL'}: {msg} ({(time.time() - ts) / 60:.1f} min)", flush=True)
    diff = _snap_diff(s0, gate_snapshot())
    if not ok or diff:
        why = msg if not ok else "protected files changed: " + "; ".join(diff[:20])
        raise SystemExit(f"=== RESULT: GATE GUARD FAIL - {why}. Do NOT run --gate; paste this.")
    (GATE / "guard_ok.json").write_text(json.dumps({
        "time": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(), "evaluate_sha": sha("src/evaluate.py"),
        "frozen_sha256": hashes, "config_diff": yd, "evidence_guard": lines,
        "reproduction": msg}, indent=2, default=str))
    print(f"=== RESULT: GATE GUARD PASS ({(time.time() - t0) / 60:.1f} min): freeze OK, gate evidence = frozen except edge "
          "(LSTM reuse valid), dataset path reproduces Test highway_2 seed 0 bit-for-bit, protected files untouched.")


def gate_ho_stats(variant):
    rows = []
    for tag in ("dataset", variant):
        for sc in SCEN:
            for sp in ("Train", "Validation", "Test"):
                sub = "heldout" if sp == "Test" else "processed"
                d = (ROOT / "data" / sub if tag == "dataset" else ROOT / "data" / "gate" / tag / sub) / sc / sp
                h = pd.read_parquet(d / "handovers.parquet", columns=["t_ho_s", "pingpong", "dwell_s"])
                ns = _pq.ParquetFile(str(d / "sessions_baseline.parquet")).metadata.num_rows
                t = h["t_ho_s"]
                rows.append({"traces": tag, "scenario": sc, "split": sp, "session_rows": ns, "HO": len(h),
                             "HO_per_session": len(h) / ns if ns else np.nan, "T_HO_min": t.min(),
                             "T_HO_q50": t.quantile(.5), "T_HO_q90": t.quantile(.9),
                             "pingpong": float(h["pingpong"].astype(float).mean()) if len(h) else np.nan,
                             "dwell_q50": h["dwell_s"].quantile(.5)})
    return pd.DataFrame(rows)


def gate_run(cfg, variant):
    global _TEST_OPEN
    hashes, yd = gate_frozen_check()
    _primary_complete()
    okf = GATE / "guard_ok.json"
    if not okf.exists() or json.loads(okf.read_text()).get("evaluate_sha") != sha("src/evaluate.py"):
        raise SystemExit("ABORT: run --gate-guard with this evaluate.py first (must PASS)")
    vdir = GATE / variant
    out, pool = vdir / f"item{GATE_NB}", vdir / f"item{GATE_NB}_seeds0-4"
    out.mkdir(parents=True, exist_ok=True)
    GATE_DOCS.mkdir(parents=True, exist_ok=True)
    L = json.loads(GATE_LOCK.read_text()) if GATE_LOCK.exists() else {"created": now(), "variants": {}}
    V = L["variants"].setdefault(variant, {"attempts": []})
    if V.get("status") == "complete":
        raise SystemExit(f"REFUSED: gate {variant} already complete (run-once). See docs/gate/gate_report_{variant}.md")
    V["attempts"].append({"start": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(), "sha256": hashes})

    def setst(st, **kw):
        V["status"] = st
        V["attempts"][-1].update(status=st, **kw)
        GATE_LOCK.write_text(json.dumps(L, indent=2, default=str))

    c = gate_cfg(cfg, variant)
    t0, s0, f = time.time(), gate_snapshot(), build_eval_split()
    _TEST_OPEN = True
    order = [(sd, sc) for sd in (1, 2, 3, 4) for sc in SCEN]
    mt = [(FINAL / "item256" / f"report_{sc}_s{sd}.md").stat().st_mtime for sd, sc in order]
    durs = {sc: [] for sc in SCEN}
    for i in range(1, len(order)):
        durs[order[i][1]].append(mt[i] - mt[i - 1])
    est = {sc: float(np.median(durs[sc])) if durs[sc] else 600.0 for sc in SCEN}
    jobs = [(sd, sc) for sd in GATE_SEEDS for sc in SCEN]
    todo = [j for j in jobs if not (out / f"report_{j[1]}_s{j[0]}.md").exists()]
    left = sum(est[sc] for _, sc in todo)
    print(f"=== GATE {variant}: {len(jobs)} jobs (5 seeds x 4 scenarios), {len(jobs) - len(todo)} already done; first estimate "
          f"~{left / 3600:.1f} h from your dataset Test timings (" + ", ".join(f"{sc} {est[sc] / 60:.0f} min" for sc in SCEN)
          + "); gate traces have different handover counts, so the estimate is rescaled after every finished job", flush=True)
    done_est = done_act = 0.0
    for i, (sd, sc) in enumerate(jobs, 1):
        if (out / f"report_{sc}_s{sd}.md").exists():
            print(f"=== [{i}/{len(jobs)}] {variant} seed {sd} {sc}: done in an earlier attempt (reused, deterministic)", flush=True)
            continue
        setst(f"simulating seed {sd} {sc}")
        ts, r = time.time(), (done_act / done_est if done_est else 1.0)
        print(f"=== [{i}/{len(jobs)}] {variant} seed {sd} {sc}: START {now()} | elapsed {(ts - t0) / 60:.0f} min | "
              f"est. remaining ~{left * r / 60:.0f} min", flush=True)
        with ReadAudit(variant):
            f(c, sc, sd, "Test", out)
        done_act += time.time() - ts
        done_est += est[sc]
        left -= est[sc]
        msg = ""
        if sd == 0:
            g = pd.read_parquet(out / f"episodes_{sc}_s0.parquet", columns=["budget"])["budget"].to_numpy()
            d = pd.read_parquet(FINAL / "item256" / f"episodes_{sc}_s0.parquet", columns=["budget"])["budget"].to_numpy()
            if len(g) == len(d) and np.array_equal(g, d):
                setst("aborted_identical_to_dataset")
                raise SystemExit(f"=== ABORT: {sc} gate episodes identical to dataset episodes (gate traces not used). Paste this.")
            msg = f" | episodes gate {len(g):,} vs dataset {len(d):,} (gate traces in use)"
        print(f"=== [{i}/{len(jobs)}] {variant} seed {sd} {sc}: DONE in {(time.time() - ts) / 60:.1f} min{msg}", flush=True)
    setst("stats")
    crit = {}
    for sd in GATE_SEEDS:
        print(f"=== stats.py (unchanged), {variant} seed {sd} only ...", flush=True)
        crit[sd] = run_stats(out, sd, f"docs/gate/criteria_{variant}_s{sd}.txt",
                             f"A15 gate {variant}, item=256 B, seed {sd} only", scratch=vdir)
    pool.mkdir(exist_ok=True)
    K, hs = ["variant", "profile", "method", "session_uid", "ho_idx"], []
    for sc in SCEN:
        print(f"=== averaging episode loss over seeds 0-4: {variant} {sc}", flush=True)
        base = acc = None
        for sd in GATE_SEEDS:
            d = pd.read_parquet(out / f"episodes_{sc}_s{sd}.parquet", columns=K + ["x", "loss", "action"])
            d = d.sort_values(K, kind="stable").reset_index(drop=True)
            hs.append(d[(d["profile"] == "nominal") & (d["x"] == 0)].groupby("method")["action"]
                      .apply(lambda a: float((a >= 1).mean())).rename(f"{sc}|{sd}"))
            if base is None:
                base, acc = d, d["loss"].to_numpy(dtype=float).copy()
            else:
                for col in K + ["x"]:
                    if len(d) != len(base) or not np.array_equal(base[col].to_numpy(), d[col].to_numpy()):
                        raise SystemExit(f"ABORT: episode keys differ, seed 0 vs {sd} ({sc}, {col})")
                acc += d["loss"].to_numpy(dtype=float)
            del d
        base["loss"] = acc / len(GATE_SEEDS)
        base.to_parquet(pool / f"episodes_{sc}_s0.parquet", index=False)
        shutil.copy(out / f"manifest_{sc}_s0.json", pool / f"manifest_{sc}_s0.json")
        ea = pd.concat([pd.read_csv(out / f"eval_alpha_{sc}_s{sd}.csv") for sd in GATE_SEEDS], ignore_index=True)
        ks = [col for col in ea.columns if not pd.api.types.is_numeric_dtype(ea[col])]
        ea.groupby(ks, sort=False).mean(numeric_only=True).reset_index().to_csv(pool / f"eval_alpha_{sc}_s0.csv", index=False)
        del base, acc
    print(f"=== stats.py (unchanged), {variant} 5-seed average ...", flush=True)
    cp = run_stats(pool, 0, f"docs/gate/criteria_{variant}_seeds0-4.txt",
                   f"A15 gate {variant}, item=256 B: episode loss averaged over seeds 0-4 (as A5/A14)", scratch=vdir)
    H = pd.concat(hs, axis=1)
    H5 = pd.DataFrame({sc: H[[col for col in H.columns if col.startswith(sc + "|")]].mean(axis=1) for sc in SCEN}).round(3)
    print("=== handover statistics, dataset vs gate (label-free) ...", flush=True)
    ho = gate_ho_stats(variant)
    ho.to_csv(vdir / "ho_stats.csv", index=False)
    diff = _snap_diff(s0, gate_snapshot())
    prim = (ROOT / "docs" / "criteria_test_item256_seeds0-4.txt").read_text()
    cl = [l for l in cp.splitlines() if re.match(r"^C[1-4] ", l)]
    passed = len(cl) == 4 and all("PASS" in l for l in cl)
    R = [f"# TRACE-MEC A15 gate: {variant} ({now()})", "",
         "- frozen model.smoke() via evaluate.py substitutions; Train refit on gate traces; Test seeds 0-4; item 256 B; "
         "stats.py unchanged (B=2000); LSTM scores reused (guard: evidence identical except edge)",
         f"- runtime {time.time() - t0:.0f} s; attempts {len(V['attempts'])} (disclose if > 1); git {V['attempts'][-1]['git']}",
         f"- config diff vs {FREEZE_REF}: {yd}", "- frozen sha256: " + json.dumps(hashes),
         f"- protected files untouched (md5: {', '.join(GATE_MD5)}; size+mtime: {', '.join(GATE_STAT)}): {not diff}"]
    R += ["  - " + x for x in diff[:50]]
    R += ["", f"## A15 decision rule: level passes iff C1-C4 all pass (5-seed average) -> {'PASS' if passed else 'NOT PASSED (report as speed dependence)'}",
          "", "## Gate, 5-seed average (key lines)"] + [l for l in cp.splitlines() if GATE_KEY.search(l)]
    R += ["", "## Primary result on dataset traces (docs/criteria_test_item256_seeds0-4.txt), key lines"]
    R += [l for l in prim.splitlines() if GATE_KEY.search(l)]
    for sd in GATE_SEEDS:
        R += ["", f"## Gate seed {sd} only (key lines)"] + [l for l in crit[sd].splitlines() if GATE_KEY.search(l)]
    R += ["", "## Honest restrict-or-quarantine share, nominal, mean over seeds 0-4 (target 0.10)", "```", H5.to_string(), "```"]
    R += ["", "## Handover statistics, dataset vs gate (label-free, descriptive; HO_per_session = HO / rows of sessions_baseline)",
          "```", ho.round(3).to_string(index=False), "```", "", "## Gate 5-seed full stats.py output", "```", cp, "```"]
    (GATE_DOCS / f"gate_report_{variant}.md").write_text("\n".join(R) + "\n")
    setst("complete", end=now(), runtime_s=round(time.time() - t0))
    print(f"=== RESULT: GATE {variant} COMPLETE -> A15 level {'PASS' if passed else 'NOT PASSED'}")
    print("--- gate, 5-seed average\n" + "\n".join(l for l in cp.splitlines() if GATE_KEY.search(l)))
    print("--- per seed: " + " || ".join(f"s{sd}: " + " ".join(l.split(" ")[0] + ("+" if "PASS" in l else "-")
          for l in crit[sd].splitlines() if re.match(r"^C[1-4] ", l)) for sd in GATE_SEEDS))
    show = [m for m in ["robust_v1+probe", "robust_v1", "lstm_sev", "central_ma", "voi_myopic", "severity"] if m in H5.index]
    print("--- honest share, mean over 5 seeds (target 0.10)\n" + H5.loc[show].to_string())
    print(f"- protected files untouched: {not diff}\nFull report: docs/gate/gate_report_{variant}.md")


# ===================== A16 path-fold rotation (v0.18) =====================
# Protocol: docs/theory.md A16 + A16a. Rotated folds 0-3, seed 0 only, item 256 B. Fold 4 = primary layout (A14 seed 0, not rerun).
PF = ROOT / "results" / "pathfold"
PF_LOCK = PF / "LOCK_pathfold.json"
PF_DOCS = ROOT / "docs" / "pathfold"
PF_FOLDS = [0, 1, 2, 3]
PF_SEED = 0
PF_MD5 = GATE_MD5 + ["results/gate"]
PF_STAT = GATE_STAT + ["data/gate", "data/pathfold"]
PF_SHOW = ["robust_v1+probe", "robust_v1", "lstm_sev", "central_ma", "voi_myopic", "severity"]
PF_REF = "docs/pathfold/criteria_fold4_primary_s0.txt"


def pf_conf(cfg):
    return cfg.get("pathfold") or yaml.safe_load((ROOT / "config/config.yaml").read_text())["pathfold"]


def pf_groups(cfg):
    return [[int(u) for u in g] for g in pf_conf(cfg)["groups"]]


def pf_root(cfg, k):
    return ROOT / pf_conf(cfg)["root"] / f"fold{k}"


def pf_pools(cfg, k):
    import preprocess as PP
    return PP.pathfold_pools({"pathfold": pf_conf(cfg), "traces": cfg["traces"]}, k)


def pf_snapshot():
    return {"md5": _tree_sig(PF_MD5, True), "stat": _tree_sig(PF_STAT, False)}


def pf_crit_ok(txt):
    cl = [l for l in txt.splitlines() if re.match(r"^C[1-4] ", l)]
    return (len(cl) == 4 and all("PASS" in l for l in cl)), cl


def pf_key(txt):
    return [l for l in txt.splitlines() if GATE_KEY.search(l)]


def pf_frozen_check():
    code = ["src/model.py", "src/stats.py", "src/train.py"]
    dirty = git("status", "--porcelain", "--", *code, "config/config.yaml", "src/evaluate.py", "src/preprocess.py").stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted changes (commit first):\n{dirty}")
    if git("diff", "--quiet", FREEZE_REF, "HEAD", "--", *code).returncode != 0:
        raise SystemExit(f"ABORT: model.py/stats.py/train.py changed since {FREEZE_REF}")
    r = git("show", f"{FREEZE_REF}:config/config.yaml")
    if r.returncode != 0:
        raise SystemExit(f"ABORT: cannot read config.yaml at {FREEZE_REF}: {r.stderr}")
    d = _ydiff(yaml.safe_load(r.stdout), yaml.safe_load((ROOT / "config/config.yaml").read_text()))
    allowed = lambda kind, p: kind == "added" and (p.split(".")[-1] == "gate" or p == "pathfold")
    if not any(p == "pathfold" for _, p in d) or any(not allowed(kind, p) for kind, p in d):
        raise SystemExit(f"ABORT: config.yaml vs {FREEZE_REF} may differ ONLY by the added gate (A15) and pathfold (A16) blocks; diff = {d}")
    return {f: sha(f) for f in code + ["config/config.yaml", "src/evaluate.py", "src/preprocess.py"]}, d


def pf_cfg(cfg, k):
    c = copy.deepcopy(cfg)
    if float(c["traces"]["evidence_item_bytes"]) != float(GATE_NB):
        raise SystemExit("ABORT: evidence_item_bytes must be 256 (A16)")
    if k is None:
        return c
    r = pf_root(cfg, k)
    for key in ("processed", "heldout"):
        if not (r / key).is_dir():
            raise SystemExit(f"ABORT: missing {r / key}")
        c["paths"][key] = str((r / key).relative_to(ROOT))
    return c


def pf_evidence_guard(cfg, k):
    pools, lines = pf_pools(cfg, k), []
    for sc in SCEN:
        for sp in ("Train", "Validation", "Test"):
            sub = "heldout" if sp == "Test" else "processed"
            fr, pd_ = ROOT / cfg["paths"][sub] / sc / sp, pf_root(cfg, k) / sub / sc / sp
            used = sorted(set(pd.read_parquet(pd_ / "mapping.parquet", columns=["path_ue"])["path_ue"].astype(int)))
            if not set(used) <= set(pools[sp]):
                raise SystemExit(f"=== RESULT: PATHFOLD GUARD FAIL - fold {k} {sc}/{sp} uses paths {used} outside pool {pools[sp]}")
            for v in MD.VARIANTS:
                a = pd.read_parquet(fr / f"evidence_{v}.parquet")
                b = pd.read_parquet(pd_ / f"evidence_{v}.parquet")
                ok, msg = _frames_equal_except(a, b, {"edge"})
                share = float((a["edge"].to_numpy() != b["edge"].to_numpy()).mean()) if ok else float("nan")
                lf = ROOT / cfg["paths"]["results"] / "lstm" / sc / f"{v}_{sp}.parquet"
                line = (f"- fold {k} {sc}/{sp}/{v}: {len(a):,} items, {msg}; edge differs {share:.3f}; paths {used}; "
                        f"LSTM scores {'present' if lf.exists() else 'MISSING'}")
                lines.append(line)
                print(line, flush=True)
                if not ok or not lf.exists():
                    raise SystemExit("=== RESULT: PATHFOLD GUARD FAIL (LSTM-reuse guard, A16): " + line)
                del a, b
    return lines


def pf_honest(out, sc, seed):
    d = pd.read_parquet(out / f"episodes_{sc}_s{seed}.parquet", columns=["profile", "x", "method", "action"])
    return d[(d["profile"] == "nominal") & (d["x"] == 0)].groupby("method")["action"].apply(lambda a: float((a >= 1).mean()))


def pf_ho_stats(cfg, k):
    rows = []
    for tag in ("primary", k):
        for sc in SCEN:
            for sp in ("Train", "Validation", "Test"):
                sub = "heldout" if sp == "Test" else "processed"
                d = (ROOT / cfg["paths"][sub] if tag == "primary" else pf_root(cfg, k) / sub) / sc / sp
                h = pd.read_parquet(d / "handovers.parquet", columns=["t_ho_s", "pingpong"])
                ns = _pq.ParquetFile(str(d / "mapping.parquet")).metadata.num_rows
                rows.append({"traces": "fold 4 (primary)" if tag == "primary" else f"fold {k}", "scenario": sc, "split": sp,
                             "sessions": ns, "HO": len(h), "HO_per_session": len(h) / ns if ns else np.nan,
                             "T_HO_min": h["t_ho_s"].min(), "T_HO_q50": h["t_ho_s"].quantile(.5),
                             "pingpong": float(h["pingpong"].astype(float).mean()) if len(h) else np.nan})
    return pd.DataFrame(rows)


def pathfold_guard(cfg):
    global _TEST_OPEN
    okf = PF / "guard_ok.json"
    if okf.exists():
        okf.unlink()
    hashes, yd = pf_frozen_check()
    _primary_complete()
    PF.mkdir(parents=True, exist_ok=True)
    PF_DOCS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== [1/5] freeze check OK: model/stats/train unchanged since {FREEZE_REF}; config diff = {yd}", flush=True)
    print(f"=== [2/5] snapshot of protected files (md5: {', '.join(PF_MD5)}; size+mtime: {', '.join(PF_STAT)}) {now()}", flush=True)
    s0 = pf_snapshot()
    lines = []
    for k in PF_FOLDS:
        print(f"=== [3/5] fold {k}: evidence vs frozen (all columns except edge), path pools, LSTM scores | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        lines += pf_evidence_guard(cfg, k)
    print(f"=== [4/5] reproduction: path-fold runner, primary traces, InTAS_highway_2 Test seed 0 vs A14 | START {now()}", flush=True)
    out = PF / "_guard_primary"
    if out.exists():
        shutil.rmtree(out)
    f, _TEST_OPEN = build_eval_split(), True
    ts = time.time()
    with ReadAudit("dataset"):
        f(pf_cfg(cfg, None), "InTAS_highway_2", PF_SEED, "Test", out)
    ok, msg = compare(out / "episodes_InTAS_highway_2_s0.parquet", FINAL / "item256" / "episodes_InTAS_highway_2_s0.parquet")
    print(f"=== [4/5] {'PASS' if ok else 'FAIL'}: {msg} ({(time.time() - ts) / 60:.1f} min)", flush=True)
    if not ok:
        raise SystemExit(f"=== RESULT: PATHFOLD GUARD FAIL - reproduction: {msg}. Do NOT run --pathfold; paste this.")
    print("=== [5/5] fold-4 reference: stats.py (unchanged) on copies of the A14 Test seed-0 episodes (no simulation)", flush=True)
    ref = PF / "fold4_ref" / f"item{GATE_NB}"
    if ref.exists():
        shutil.rmtree(ref)
    ref.mkdir(parents=True)
    for sc in SCEN:
        for fn in (f"episodes_{sc}_s0.parquet", f"manifest_{sc}_s0.json", f"eval_alpha_{sc}_s0.csv"):
            shutil.copy(FINAL / "item256" / fn, ref / fn)
    rtxt = run_stats(ref, 0, PF_REF, "A16 fold 4 = primary layout (Test = HetNet paths 16-19), A14 seed-0 episodes, item=256 B",
                     scratch=PF / "fold4_ref")
    rok, _ = pf_crit_ok(rtxt)
    diff = _snap_diff(s0, pf_snapshot())
    if diff:
        raise SystemExit("=== RESULT: PATHFOLD GUARD FAIL - protected files changed: " + "; ".join(diff[:20]))
    okf.write_text(json.dumps({"time": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(),
                               "evaluate_sha": sha("src/evaluate.py"), "frozen_sha256": hashes, "config_diff": yd,
                               "evidence_guard": lines, "reproduction": msg, "fold4_reference_pass": rok}, indent=2, default=str))
    print(f"=== RESULT: PATHFOLD GUARD PASS ({(time.time() - t0) / 60:.1f} min): freeze OK; folds 0-3 evidence = frozen except edge "
          "(LSTM reuse valid), pools respected; path-fold runner reproduces A14 highway_2 seed 0 bit-for-bit; protected files untouched.")
    print(f"--- fold 4 reference (primary layout, seed 0): {'PASS' if rok else 'NOT PASSED'}\n" + "\n".join(pf_key(rtxt)), flush=True)


def pf_finalize(cfg, k, out, vdir):
    G = pf_groups(cfg)
    print(f"=== fold {k}: stats.py (unchanged), seed 0 ...", flush=True)
    crit = run_stats(out, PF_SEED, f"docs/pathfold/criteria_fold{k}_s0.txt",
                     f"A16 path-fold {k} (Test = HetNet paths {G[k]}), item=256 B, seed 0", scratch=vdir)
    passed, _ = pf_crit_ok(crit)
    H = pd.DataFrame({sc: pf_honest(out, sc, PF_SEED) for sc in SCEN}).round(3)
    H.to_csv(vdir / "honest_share.csv")
    ho = pf_ho_stats(cfg, k)
    ho.to_csv(vdir / "ho_stats.csv", index=False)
    R = [f"# TRACE-MEC A16 path-fold {k} ({now()})", "",
         f"- Test = HetNet paths {G[k]}; pools {pf_pools(cfg, k)}",
         "- frozen model.smoke() via evaluate.py substitutions; Train refit on fold traces; thresholds on Train nominal; "
         "Test seed 0; item 256 B; stats.py unchanged (B=2000); LSTM scores reused (guard: evidence identical except edge)", "",
         f"## A16a rule for this fold: C1-C4 all pass at seed 0 -> {'PASS' if passed else 'NOT PASSED'}", "",
         "## Key lines"] + pf_key(crit)
    R += ["", "## Honest restrict-or-quarantine share, nominal, seed 0 (target 0.10)", "```", H.to_string(), "```",
          "", "## Handover statistics, primary vs this fold (label-free, descriptive)", "```", ho.round(3).to_string(index=False), "```",
          "", "## Full stats.py output", "```", crit, "```"]
    (PF_DOCS / f"pathfold_report_fold{k}.md").write_text("\n".join(R) + "\n")
    return passed, crit, H


def pf_summary(cfg, L, diff=None):
    G = pf_groups(cfg)
    ref = (ROOT / PF_REF).read_text()
    R = [f"# TRACE-MEC A16/A16a path-fold rotation summary ({now()})", "",
         f"## Fold 4 = primary layout (Test = paths {G[4]}), A14 seed 0", *pf_key(ref)]
    fails, done, Hs = [], [], {}
    for k in PF_FOLDS:
        st = L["folds"].get(str(k), {}).get("status")
        p = ROOT / f"docs/pathfold/criteria_fold{k}_s0.txt"
        if st != "complete" or not p.exists():
            R += ["", f"## Fold {k} (Test = paths {G[k]}): not complete (status {st})"]
            continue
        txt = p.read_text()
        ok, _ = pf_crit_ok(txt)
        done.append(k)
        if not ok:
            fails.append(k)
        R += ["", f"## Fold {k} (Test = paths {G[k]}, Validation = paths {G[(k - 1) % 5]}): {'PASS' if ok else 'NOT PASSED'}", *pf_key(txt)]
        Hs[f"fold{k}"] = pd.read_csv(PF / f"fold{k}" / "honest_share.csv", index_col=0)
    if len(done) == 4:
        dec = ("PATH-ROBUST: C1-C4 PASS at seed 0 in all 4 rotated folds (A16a)" if not fails else
               "NOT path-robust: C1-C4 failed in fold(s) " + ", ".join(f"{k} (Test paths {G[k]})" for k in fails)
               + "; reported as such, no path-robustness claim (A16a)")
    else:
        dec = f"INCOMPLETE: folds complete {done}; rerun the same command to finish"
    R += ["", f"## A16a decision: {dec}"]
    if Hs:
        for k, h in Hs.items():
            show = [m for m in PF_SHOW if m in h.index]
            R += ["", f"## Honest share {k} (target 0.10)", "```", h.loc[show].to_string(), "```"]
    if diff is not None:
        R += ["", f"## Protected files untouched: {not diff}"] + ["  - " + x for x in diff[:50]]
    (PF_DOCS / "pathfold_summary.md").write_text("\n".join(R) + "\n")
    return dec, fails, done, Hs


def pathfold_run(cfg, folds):
    global _TEST_OPEN
    hashes, yd = pf_frozen_check()
    _primary_complete()
    okf = PF / "guard_ok.json"
    if not okf.exists() or json.loads(okf.read_text()).get("evaluate_sha") != sha("src/evaluate.py"):
        raise SystemExit("ABORT: run --pathfold-guard with this evaluate.py first (must PASS)")
    if not (ROOT / PF_REF).exists():
        raise SystemExit(f"ABORT: {PF_REF} missing (made by --pathfold-guard)")
    folds, G = sorted(set(folds)), pf_groups(cfg)
    PF_DOCS.mkdir(parents=True, exist_ok=True)
    L = json.loads(PF_LOCK.read_text()) if PF_LOCK.exists() else {"created": now(), "folds": {}}

    def save():
        PF_LOCK.write_text(json.dumps(L, indent=2, default=str))

    def setst(F, st, **kw):
        F["status"] = st
        F["attempts"][-1].update(status=st, **kw)
        save()

    for k in folds:
        if L["folds"].get(str(k), {}).get("status") == "complete":
            print(f"=== fold {k}: already complete (run-once) - skipped; see docs/pathfold/pathfold_report_fold{k}.md", flush=True)
    pending = [k for k in folds if L["folds"].get(str(k), {}).get("status") != "complete"]
    if not pending:
        dec, *_ = pf_summary(cfg, L)
        print(f"=== nothing to do. A16a decision: {dec}")
        return
    jobs = [(k, sc) for k in pending for sc in SCEN
            if not (PF / f"fold{k}" / f"item{GATE_NB}" / f"report_{sc}_s{PF_SEED}.md").exists()]
    need = sum(sum(p.stat().st_size for p in (FINAL / "item256").glob(f"*_{sc}_s0.*")) for _, sc in jobs) * 1.3 + 3e9
    free = shutil.disk_usage(ROOT).free
    print(f"=== disk: free {free / 1e9:.1f} GB; this run needs ~{need / 1e9:.1f} GB incl. margin", flush=True)
    if free < need:
        raise SystemExit("ABORT: not enough free disk space - nothing run")
    order = [(sd, sc) for sd in (1, 2, 3, 4) for sc in SCEN]
    mt = [(FINAL / "item256" / f"report_{sc}_s{sd}.md").stat().st_mtime for sd, sc in order]
    durs = {sc: [] for sc in SCEN}
    for i in range(1, len(order)):
        durs[order[i][1]].append(mt[i] - mt[i - 1])
    est = {sc: float(np.median(durs[sc])) if durs[sc] else 600.0 for sc in SCEN}
    left = sum(est[sc] for _, sc in jobs)
    print(f"=== PATH-FOLD (A16a): folds {pending}, {len(jobs)} jobs to run (seed 0); first estimate ~{left / 3600:.1f} h from your "
          f"dataset Test timings (" + ", ".join(f"{sc} {est[sc] / 60:.0f} min" for sc in SCEN) + "); rescaled after every job", flush=True)
    print(f"=== snapshot of protected files {now()} ...", flush=True)
    t0, s0 = time.time(), pf_snapshot()
    f = build_eval_split()
    _TEST_OPEN = True
    done_est = done_act = 0.0
    j = 0
    for k in pending:
        F = L["folds"].setdefault(str(k), {"attempts": []})
        F["attempts"].append({"start": now(), "git": git("rev-parse", "--short", "HEAD").stdout.strip(), "sha256": hashes,
                              "test_paths": G[k]})
        save()
        c, vdir = pf_cfg(cfg, k), PF / f"fold{k}"
        out = vdir / f"item{GATE_NB}"
        out.mkdir(parents=True, exist_ok=True)
        tf = time.time()
        for sc in SCEN:
            if (out / f"report_{sc}_s{PF_SEED}.md").exists():
                print(f"=== fold {k} {sc}: done in an earlier attempt (reused, deterministic)", flush=True)
                continue
            j += 1
            setst(F, f"simulating {sc}")
            ts, r = time.time(), (done_act / done_est if done_est else 1.0)
            print(f"=== [{j}/{len(jobs)}] fold {k} (Test paths {G[k]}) {sc} seed 0: START {now()} | elapsed {(ts - t0) / 60:.0f} min | "
                  f"est. remaining ~{left * r / 60:.0f} min (~{left * r / 3600:.1f} h)", flush=True)
            with ReadAudit(f"pathfold{k}"):
                f(c, sc, PF_SEED, "Test", out)
            done_act += time.time() - ts
            done_est += est[sc]
            left -= est[sc]
            g = pd.read_parquet(out / f"episodes_{sc}_s0.parquet", columns=["budget"])["budget"].to_numpy()
            d = pd.read_parquet(FINAL / "item256" / f"episodes_{sc}_s0.parquet", columns=["budget"])["budget"].to_numpy()
            if len(g) == len(d) and np.array_equal(g, d):
                setst(F, "aborted_identical_to_primary")
                raise SystemExit(f"=== ABORT: fold {k} {sc} episodes identical to primary (fold traces not used). Paste this.")
            print(f"=== [{j}/{len(jobs)}] fold {k} {sc}: DONE in {(time.time() - ts) / 60:.1f} min | episodes fold {len(g):,} "
                  f"vs primary {len(d):,} (fold traces in use)", flush=True)
        setst(F, "stats")
        passed, crit, H = pf_finalize(cfg, k, out, vdir)
        setst(F, "complete", end=now(), runtime_s=round(time.time() - tf), passed=passed)
        print(f"=== FOLD {k} COMPLETE (Test paths {G[k]}) -> {'PASS' if passed else 'NOT PASSED'}\n" + "\n".join(pf_key(crit)), flush=True)
        pf_summary(cfg, L)
    print(f"=== end-of-run integrity check {now()} ...", flush=True)
    diff = _snap_diff(s0, pf_snapshot())
    L["integrity"] = {"time": now(), "protected_untouched": not diff, "diff": diff[:50]}
    save()
    dec, fails, done, Hs = pf_summary(cfg, L, diff)
    print(f"=== RESULT: PATH-FOLD RUN FINISHED ({(time.time() - t0) / 3600:.2f} h)")
    print("--- fold 4 (primary, A14 seed 0): " + " ".join(l.split(" ")[0] + ("+" if "PASS" in l else "-")
          for l in pf_crit_ok((ROOT / PF_REF).read_text())[1]))
    for k in done:
        _, cl = pf_crit_ok((ROOT / f"docs/pathfold/criteria_fold{k}_s0.txt").read_text())
        print(f"--- fold {k} (Test paths {G[k]}): " + " ".join(l.split(" ")[0] + ("+" if "PASS" in l else "-") for l in cl))
    for k, h in Hs.items():
        if "robust_v1+probe" in h.index:
            print(f"--- honest share {k}, robust_v1+probe (target 0.10): " + h.loc["robust_v1+probe"].to_string().replace("\n", " | "))
    print(f"--- A16a decision: {dec}\n- protected files untouched: {not diff}\nFull summary: docs/pathfold/pathfold_summary.md")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dryrun", action="store_true")
    g.add_argument("--final", action="store_true")
    g.add_argument("--seeds", nargs="+", type=int)
    g.add_argument("--gate-guard", action="store_true")
    g.add_argument("--gate", choices=["gate_typical", "gate_fast"])
    g.add_argument("--pathfold-guard", action="store_true")
    g.add_argument("--pathfold", nargs="+", type=int, choices=[0, 1, 2, 3])
    ap.add_argument("--scenarios", nargs="+", default=["InTAS_highway_2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rerun-after-abort", action="store_true")
    a = ap.parse_args()
    cfg, how = load_cfg()
    say(f"config loaded via {how}")
    if a.dryrun:
        dryrun(cfg, SCEN if a.scenarios == ["all"] else a.scenarios, a.seed, how)
    elif a.gate_guard:
        gate_guard(cfg)
    elif a.gate:
        gate_run(cfg, a.gate)
    elif a.pathfold_guard:
        pathfold_guard(cfg)
    elif a.pathfold:
        pathfold_run(cfg, a.pathfold)
    elif a.seeds:
        seeds_run(cfg, a.seeds)
    else:
        final(cfg, a.seed, a.resume, a.rerun_after_abort)


if __name__ == "__main__":
    main()
# END-OF-FILE
