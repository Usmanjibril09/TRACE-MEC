#!/usr/bin/env python3
"""TRACE-MEC model.py (v0.2)

  ItemModel     per-evidence-item likelihood ratio (HistGradientBoosting on gt_msg_attack)
  oof_llr       out-of-fold item LLRs on Train (GroupKFold by sender_id) for fitting the aggregator
  Aggregator    session-level calibrated belief: logistic regression on aggregate evidence features
                (count, mean, max, min, #strong items). Replaces v0.1 tempered log-odds sum, which
                treated "looks normal" as proof of innocence and diluted decisive items.
  simulate      source snapshot at prep start -> method send order under byte budget/deadline
                -> faults (loss/dup/late) -> adversary (suppress/inject) -> aggregate features
  decide        belief p -> Bayes action allow/restrict/quarantine
  --smoke       fit on <scenario>/Train, evaluate on <scenario>/Validation, one seed

Rules: gt_* never features (asserted); Test refused (only evaluate.py --final reads it);
aggregator fitted on NOMINAL profile only (adversarial = unseen robustness test);
common random numbers: fault/adversary draws made once per chunk, shared by all methods.
References: ref_full_clean = all genuine items held at source, no budget/faults/adversary (information bound);
            full_no_deadline = all items after adversary manipulation, no budget/faults.
v0.2 simplifications: source holds only its own reports (no multi-hop carry-over); reconciliation = dedup.
"""
import argparse, json, subprocess, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["is_first", "dt_send_s", "dt_rcv_s", "dup", "msgid_back", "send_back",
            "implied_spd", "spd_resid", "dist_sr_m", "rate_1s"]
VARIANTS = ["baseline", "timeDelayAttack", "dataReplay", "dosAttack"]
ACTIONS = ["allow", "restrict", "quarantine"]
METHODS = ["no_transfer", "fifo", "recency", "severity", "random", "aoi", "voi_myopic", "robust_mix",
           "full_no_deadline", "ref_full_clean"]
BUDGETED = ["fifo", "recency", "severity", "random", "aoi", "voi_myopic", "robust_mix"]
BASE = {"robust_v1": "severity", "severity_rb": "severity", "fifo_rb": "fifo", "no_transfer_rb": "no_transfer"}
RB = set(BASE)                 # robust belief: incriminating items only
POS_ONLY = {"robust_v1"}       # send only incriminating items
METHODS += list(BASE)
BUDGETED += ["robust_v1", "severity_rb", "fifo_rb"]
AGG_COLS = ["f_logn", "f_mean", "f_max", "f_min", "f_loghi1", "f_loghi2", "f_none"]

# ===== v0.7 (docs/theory.md Sec. 8-11): shared monotone RB aggregator, matched operating point,
# ===== common random numbers keyed by item identity, deception probe. Sweeps live in ablation.py.
from scipy.optimize import minimize
from scipy.special import expit, logit
MONO_COLS = ["m_logn", "m_sum", "m_max", "m_loghi1", "m_loghi2"]
BASE["ref_full_clean_rb"] = "ref_full_clean"
RB.add("ref_full_clean_rb")
METHODS = list(METHODS) + [m for m in ["ref_full_clean_rb"] if m not in METHODS]
PROBES = {"robust_v1+probe": ("robust_v1", "voi"), "robust_v1+probe_rnd": ("robust_v1", "rnd"),
          "fifo_rb+probe": ("fifo_rb", "voi"), "no_transfer_rb+probe": ("no_transfer_rb", "voi")}
BUDGETED += ["robust_v1+probe", "robust_v1+probe_rnd", "fifo_rb+probe"]
# v0.8 extended baselines (docs/baselines.md): added after Validation closed (A7); Train-tuned only
EXT_CFG = {"beta_lambda_grid": [0.5, 0.8, 0.9, 0.95, 0.99, 1.0], "ma_delay_s": 5.0}
METHODS = list(METHODS) + [m for m in ["beta_rep", "central_ma"] if m not in METHODS]
BUDGETED += ["beta_rep"]
# v0.8b: #8 F2MD checks and #10 LSTM = own item score, then severity transfer + own aggregator
EXT_DET = {"f2md": "llr_f2md", "f2md_tuned": "llr_f2md_tuned", "lstm_sev": "llr_lstm"}
METHODS = list(METHODS) + [m for m in EXT_DET if m not in METHODS]
BUDGETED += list(EXT_DET)
# v0.9 trace_lstm ablation (A9/A11): TRACE-MEC pipeline on LSTM item scores
EXT_ALIAS = {"lstm_v1": "lstm_sev", "lstm_ref_rb": "lstm_sev"}
BASE.update({"lstm_v1": "severity", "lstm_ref_rb": "ref_full_clean"})
RB.update(["lstm_v1", "lstm_ref_rb"]); POS_ONLY.add("lstm_v1")
LSTM_RB = {"lstm_v1", "lstm_ref_rb"}
PROBES["lstm_v1+probe"] = ("lstm_v1", "voi")
METHODS = list(METHODS) + [m for m in ["lstm_v1", "lstm_ref_rb"] if m not in METHODS]
BUDGETED += ["lstm_v1", "lstm_v1+probe"]
EXT_KEYS = ["receiver_id", "sender_id", "message_id", "rcv_ns"]
F2MD_P = dict(range_m=420.0, vmax=40.0, acc=3.0, dec=4.5, mgt=4.0, mgt_dn=6.2, mgt_up=2.1, beacon_s=0.9, delta_s=3.1)
_M64 = (1 << 64) - 1
_C1, _C2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_PB = {}


def _salt(s):
    return np.uint64((int(s) * 0x9E3779B97F4A7C15) & _M64)


def _mix(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
        return z ^ (z >> np.uint64(31))


def _unif(z):
    return (z >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


def ep_hash(h, seed):
    k = pd.util.hash_pandas_object(h[["session_uid", "ho_idx"]], index=False).to_numpy().astype(np.uint64)
    return _mix(k ^ _salt(seed + 1))


def ep_u(h, seed, salt):
    return _unif(_mix(ep_hash(h, seed) ^ _salt(salt + 100)))


def item_u(eh, c, salt):
    """Uniform keyed by item identity (episode, report time, receiver code): same draw for the same item across
    methods and profiles, unaffected by injected items (common random numbers, theory.md A1-D)."""
    e = eh[c["ep"].to_numpy(np.int64)]
    r = c["report_ns"].to_numpy().astype(np.int64).view(np.uint64)
    q = c["rcv_code"].to_numpy().astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        return _unif(_mix(e ^ _mix(r ^ _salt(salt)) ^ _mix(q + _salt(salt + 50))))


def mono_feats(n, ep, llr, w, tau, thr):
    """Monotone features of the incriminating multiset {l > tau} (M1/M2 hold with non-negative weights)."""
    k = llr > tau
    ep, l, w = ep.astype(np.int64)[k], llr[k] - tau, w[k]
    cnt = np.bincount(ep, weights=w, minlength=n)
    s = np.bincount(ep, weights=l * w, minlength=n)
    mx = np.zeros(n)
    if len(ep):
        np.maximum.at(mx, ep, l)
    h1 = np.bincount(ep, weights=w * (l + tau > thr[0]), minlength=n)
    h2 = np.bincount(ep, weights=w * (l + tau > thr[1]), minlength=n)
    return np.column_stack([np.log1p(cnt), s, mx, np.log1p(h1), np.log1p(h2)])


class MonoAgg:
    """Shared robust aggregator: logistic regression with NON-NEGATIVE weights on MONO_COLS (L-BFGS-B bounds);
    the empty incriminating set scores lowest. Same L2 scaling as sklearn's C."""
    def __init__(self, C):
        self.C = C

    def fit(self, df):
        X, y = df[MONO_COLS].to_numpy(dtype="float64"), df["x"].to_numpy(dtype="float64")
        self.sc = np.maximum(X.std(0), 1e-9)
        Z, nn = X / self.sc, len(y)

        def f(t):
            z = t[0] + Z @ t[1:]
            g = expit(z) - y
            val = (np.logaddexp(0.0, z) - y * z).mean() + 0.5 / (self.C * nn) * (t[1:] @ t[1:])
            return val, np.r_[g.mean(), Z.T @ g / nn + t[1:] / (self.C * nn)]
        t0 = np.r_[float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6))), np.zeros(Z.shape[1])]
        self.t = minimize(f, t0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0.0, None)] * Z.shape[1]).x
        return self

    def p(self, df):
        return expit(self.t[0] + (df[MONO_COLS].to_numpy(dtype="float64") / self.sc) @ self.t[1:])

    def coefs(self):
        return {"intercept": round(float(self.t[0]), 3), **{c: round(float(v), 3) for c, v in zip(MONO_COLS, self.t[1:])}}


def op_thr(p, u, x, alpha, alpha_q):
    s = (p + 1e-9 * u)[x == 0]
    return [float(np.quantile(s, 1 - alpha)), float(np.quantile(s, 1 - alpha_q))]


def op_act(p, u, t):
    s = p + 1e-9 * u
    return (s > t[0]).astype(np.int64) + (s > t[1]).astype(np.int64)


def _eloss(b, a, L):
    return (1 - b) * L[a, 0] + b * L[a, 1]


def _sid(s):
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)$", expand=False), errors="coerce").fillna(-1).astype(np.int64)


def _msgs(sc, split, var):
    key = (sc, split, var)
    if key in _PB:
        return _PB[key]
    f = ROOT / "data" / ("heldout" if split == "Test" else "interim") / "veremi" / sc / f"{var}_{split}.parquet"
    d = pd.read_parquet(f, columns=["sender_id", "message_id", "send_ns", "s_x", "s_y", "s_pos_nx", "s_pos_ny"])
    d = d.drop_duplicates(["sender_id", "message_id", "send_ns"]).copy()
    for col in ["s_x", "s_y", "s_pos_nx", "s_pos_ny"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["sid"] = _sid(d["sender_id"])
    d["send_ns"] = d["send_ns"].astype(np.int64)
    d = d.sort_values("send_ns").reset_index(drop=True)
    if var == "baseline":
        _PB[key] = d
    return d


def probe_info(cfg, sc, split, var, h):
    """Per-episode probe inputs at t_exec: the sender's latest CAM claimed by then, its offset from the true position
    (same sender's baseline CAM nearest the claimed send time; 0 for unmodified CAMs), and its GPS noise scale."""
    b = _msgs(sc, split, "baseline")
    if var == "baseline":
        a = b.assign(lx=0.0, ly=0.0)
    else:
        a = pd.merge_asof(_msgs(sc, split, var), b[["sid", "send_ns", "s_x", "s_y"]].rename(
            columns={"send_ns": "t_b", "s_x": "xb", "s_y": "yb"}), left_on="send_ns", right_on="t_b", by="sid",
            direction="nearest", tolerance=int(1.5e9))
        a["lx"], a["ly"] = (a["s_x"] - a["xb"]).fillna(0.0), (a["s_y"] - a["yb"]).fillna(0.0)
    e = h[["session_uid", "ho_idx", "sender_id", "t_exec_ns", "t_ho_s"]].copy()
    e["sid"], e["t_exec_ns"] = _sid(e["sender_id"]), e["t_exec_ns"].astype(np.int64)
    j = pd.merge_asof(e.sort_values("t_exec_ns"), a[["sid", "send_ns", "lx", "ly", "s_pos_nx", "s_pos_ny"]],
                      left_on="t_exec_ns", right_on="send_ns", by="sid", direction="backward")
    has = j["send_ns"].notna()
    if has.mean() < 0.5:
        raise RuntimeError(f"probe_info: only {has.mean():.2%} of episodes matched a CAM ({sc}/{split}/{var}); "
                           f"episode ids {e['sender_id'].head(3).tolist()} vs CAM ids {a['sender_id'].head(3).tolist()}")
    return pd.DataFrame({"session_uid": j["session_uid"].to_numpy(), "ho_idx": j["ho_idx"].to_numpy(),
                         "t_ho_s": j["t_ho_s"].to_numpy(), "lx": j["lx"].fillna(0.0).to_numpy(),
                         "ly": j["ly"].fillna(0.0).to_numpy(), "gx": j["s_pos_nx"].abs().to_numpy(),
                         "gy": j["s_pos_ny"].abs().to_numpy(), "has": has.to_numpy()})


def _probe_q(g, PI):
    q = g[["session_uid", "ho_idx"] + (["x"] if "x" in g else [])].merge(PI, on=["session_uid", "ho_idx"], how="left")
    gm = float(np.nanmedian(PI["gx"]))
    q[["lx", "ly"]] = q[["lx", "ly"]].fillna(0.0)
    q[["gx", "gy"]] = q[["gx", "gy"]].fillna(gm)
    q["t_ho_s"] = q["t_ho_s"].fillna(np.inf)
    q["has"] = q["has"].fillna(False).astype(bool)
    return q


def probe_e(q, sigma, kappa, seed):
    u = [ep_u(q, seed, s) for s in (11, 12, 13, 14)]
    r1, r2 = np.sqrt(-2 * np.log1p(-u[0])), np.sqrt(-2 * np.log1p(-u[2]))
    z1, z2 = r1 * np.cos(2 * np.pi * u[1]), r1 * np.sin(2 * np.pi * u[1])
    z3, z4 = r2 * np.cos(2 * np.pi * u[3]), r2 * np.sin(2 * np.pi * u[3])
    return np.hypot(kappa * q["lx"].to_numpy() + q["gx"].to_numpy() * z1 + sigma * z3,
                    kappa * q["ly"].to_numpy() + q["gy"].to_numpy() * z2 + sigma * z4)


def probe_apply(g, PI, OP, pm, L, seed, cut=None):
    """Deception probe on base method bm: VOI under the deployed decision rule (standard VOI, cited); budget =
    probe rate; Bayes update with Train-estimated (d, f)."""
    bm, trig = PROBES[pm]
    pr, q, u = OP["probe"], _probe_q(g, PI), ep_u(g, seed, 9)
    p0 = g["p"].to_numpy()
    b = np.clip(p0, 1e-9, 1 - 1e-9)
    kk = (g["m_logn"].to_numpy() > 0).astype(int)   # A3: likelihoods conditional on incriminating evidence arrived
    d, f, t = np.asarray(pr["cond"][bm]["d"])[kk], np.asarray(pr["cond"][bm]["f"])[kk], OP["thr"][bm]
    Pp = b * d + (1 - b) * f
    bp, bn = b * d / Pp, b * (1 - d) / (1 - Pp)
    voi = (_eloss(b, op_act(b, u, t), L) - Pp * _eloss(bp, op_act(bp, u, t), L)
           - (1 - Pp) * _eloss(bn, op_act(bn, u, t), L))
    if trig == "voi":
        s = voi + 1e-12 * ep_u(g, seed, 16)
        cut = float(np.quantile(s, 1 - pr["rate"]))   # A3: budget enforced per set (label-free)
        want = s > cut
    else:
        u15 = ep_u(g, seed, 15)
        cut = float(np.quantile(u15, pr["rate"]))
        want = u15 < cut
    probed = want & (q["t_ho_s"].to_numpy() >= pr["t_p"])
    pos = probe_e(q, pr["sigma"], 1.0, seed) > pr["theta"]
    p2 = np.where(probed, np.where(pos, bp, bn), p0)
    return g.assign(method=pm, p=p2, probed=probed, probe_pos=probed & pos, voi=voi), cut


def fit_operating(fit, PI, M, L, seed):
    """PRIMARY decision rule (theory.md Sec. 8B/11): per-method thresholds on Train nominal so that the honest
    restrict-or-quarantine share = alpha (quarantine = alpha_q); probe calibration and VOI cutoffs on Train."""
    O, P = M["operating"], M["probe"]
    OP = {"alpha": float(O["alpha"]), "alpha_q": float(O["alpha_q"]), "thr": {}, "cut": {}, "probe_rate_train": {}}
    u, x, p = ep_u(fit, seed, 9), fit["x"].to_numpy().astype(int), fit["p"].to_numpy()
    for m, idx in fit.groupby("method").indices.items():
        OP["thr"][m] = op_thr(p[idx], u[idx], x[idx], OP["alpha"], OP["alpha_q"])
    q = _probe_q(fit.loc[fit["method"] == "no_transfer_rb", ["session_uid", "ho_idx", "x"]], PI)
    sg = float(P["sigma_primary"])
    e, xq = probe_e(q, sg, 1.0, seed), q["x"].to_numpy().astype(int)
    th = float(np.quantile(e[xq == 0], 1 - float(P["false_alarm"])))
    vq = q["session_uid"].str.split("|").str[1].to_numpy()
    OP["probe"] = {"sigma": sg, "theta": th, "d": float((e[xq == 1] > th).mean()), "f": float((e[xq == 0] > th).mean()),
                   "rate": float(P["rate"]), "t_p": float(P["t_p_primary"]), "matched": float(q["has"].mean()),
                   "d_by_variant": {v: round(float((e[(xq == 1) & (vq == v)] > th).mean()), 3) for v in sorted(set(vq[xq == 1]))},
                   "t_ho_min": float(PI["t_ho_s"].min()),
                   "frac_tho_lt_tp": {str(t): float((PI["t_ho_s"] < float(t)).mean()) for t in P["t_p_s"]}}
    OP["probe"]["cond"] = {}
    for bm in sorted({b for b, _ in PROBES.values()}):
        gb = fit[fit["method"] == bm]
        qb = _probe_q(gb[["session_uid", "ho_idx", "x"]], PI)
        eb, xb = probe_e(qb, sg, 1.0, seed) > th, qb["x"].to_numpy().astype(int)
        kb = gb["m_logn"].to_numpy() > 0
        c = {"d": [], "f": [], "n_x1": [], "n_x0": []}
        for kk in (False, True):
            s1, s0 = (xb == 1) & (kb == kk), (xb == 0) & (kb == kk)
            c["d"].append(float(np.clip(eb[s1].mean(), 1e-3, 1 - 1e-3)) if s1.sum() >= 20 else OP["probe"]["d"])
            c["f"].append(float(np.clip(eb[s0].mean(), 1e-3, 1 - 1e-3)) if s0.sum() >= 20 else OP["probe"]["f"])
            c["n_x1"].append(int(s1.sum())); c["n_x0"].append(int(s0.sum()))
        OP["probe"]["cond"][bm] = c
    for pm, (bm, _) in PROBES.items():
        g2, OP["cut"][pm] = probe_apply(fit[fit["method"] == bm], PI, OP, pm, L, seed)
        OP["thr"][pm] = op_thr(g2["p"].to_numpy(), ep_u(g2, seed, 9), g2["x"].to_numpy().astype(int), OP["alpha"], OP["alpha_q"])
        OP["probe_rate_train"][pm] = float(g2["probed"].mean())
    return OP


def apply_operating(df, PI, OP, L, seed):
    df = df.assign(action_bayes=df["action"], loss_bayes=df["loss"], probed=False, probe_pos=False)
    parts = [df]
    for pm in PROBES:
        g2, _ = probe_apply(df[df["method"] == PROBES[pm][0]], PI, OP, pm, L, seed, cut=OP["cut"][pm])
        ab, lb = bayes(g2["p"].to_numpy(), g2["x"].to_numpy().astype(int), L)
        parts.append(g2.assign(action_bayes=ab, loss_bayes=lb))
    out = pd.concat(parts, ignore_index=True)
    u, p, x = ep_u(out, seed, 9), out["p"].to_numpy(), out["x"].to_numpy().astype(int)
    a = np.zeros(len(out), np.int64)
    for m, idx in out.groupby("method").indices.items():
        a[idx] = op_act(p[idx], u[idx], OP["thr"][m])
    return out.assign(action=a, loss=L[a, x])


def eval_alpha(df, alpha, seed):
    """SECONDARY (A3): miss rate at exactly alpha honest restrict share on THIS set (ROC operating point);
    thresholds per (profile, method), pooled over variants (the defender does not know the variant)."""
    s = df["p"].to_numpy() + 1e-9 * ep_u(df, seed, 9)
    x, v = df["x"].to_numpy().astype(int), df["variant"].to_numpy()
    rows = []
    for (pf, m), idx in df.groupby(["profile", "method"]).indices.items():
        t = np.quantile(s[idx][x[idx] == 0], 1 - alpha)
        for var in np.unique(v[idx][x[idx] == 1]):
            k = (x[idx] == 1) & (v[idx] == var)
            rows.append((pf, m, var, float((s[idx][k] <= t).mean())))
    return pd.DataFrame(rows, columns=["profile", "method", "variant", "miss_at_alpha"])


def report_v07(df, OP, ctx, sc, seed, out):
    (out / f"operating_{sc}_s{seed}.json").write_text(json.dumps(OP, indent=2, default=str))
    nom = df[df["profile"] == "nominal"]
    hr = nom[nom["x"] == 0].groupby("method")["action"].apply(lambda s: float((s >= 1).mean()))
    K = ["session_uid", "ho_idx", "variant"]
    r = df[(df["method"] == "robust_v1") & df["profile"].isin(["adv_burst", "adv_sustained", "adv_early"])].pivot_table(
        index=K, columns="profile", values="p")
    same = float((np.isclose(r["adv_burst"], r["adv_sustained"]) & np.isclose(r["adv_burst"], r["adv_early"])).mean())
    a = df[df["method"] == "robust_v1"][K + ["profile", "action", "n_sent", "budget"]]
    b = df[df["method"] == "severity_rb"][K + ["profile", "action", "n_sent"]]
    j = a.merge(b, on=K + ["profile"], suffixes=("", "_sev"))
    P = OP["probe"]
    pr = nom[nom["method"].isin(list(PROBES))].groupby("method")["probed"].mean()
    pool = np.asarray(ctx["inject_pool"])
    ea = eval_alpha(df, OP["alpha"], seed)
    ea.to_csv(out / f"eval_alpha_{sc}_s{seed}.csv", index=False)
    MENU = ["adv_burst", "adv_sustained", "adv_early", "adv_suppress_top"]
    em = ea[ea["profile"].isin(MENU)].groupby("method")["miss_at_alpha"]
    tb = pd.concat([ea[ea["profile"] == "nominal"].groupby("method")["miss_at_alpha"].mean().rename("nominal_mean"),
                    em.mean().rename("menu_mean"), em.max().rename("menu_worst")], axis=1).round(3)
    EXTRA = [f"- probe likelihoods conditional on k=1[incriminating arrived], index [k=0, k=1], Train nominal: {P['cond']}",
             "- probe budget enforced per evaluation set (label-free, A3)", "",
             f"## SECONDARY (A3): miss rate at exactly alpha={OP['alpha']} on this set (ROC point; per profile, pooled over variants)",
             '```', tb.to_string(), '```']
    return ["", "## v0.7: operating point, probe, checks (theory.md Sec. 8-11)",
            "- PRIMARY: the summary table above (action/loss) uses the matched operating point; loss_bayes in the episodes parquet is secondary.",
            f"- matched operating point: alpha={OP['alpha']} honest restrict-or-quarantine, alpha_q={OP['alpha_q']}; thresholds per method on Train nominal",
            "- realized honest restrict-or-quarantine share, Validation nominal: " + ", ".join(f"{m} {v:.3f}" for m, v in hr.items()),
            f"- shared monotone RB aggregator (non-negative weights, fitted on ref_full_clean_rb): {OP['mono']}",
            f"- injected-item pool share with llr > rb_tau (Prop. 1 assumption violated if > 0): {float((pool > ctx['rb_tau']).mean()):.4f}",
            f"- Prop. 1 check (common random numbers): robust_v1 belief identical across burst/sustained/early: {same:.4f} (expected 1.0000)",
            f"- Thm 2(b): robust_v1 vs severity_rb action identical {float((j['action'] == j['action_sev']).mean()):.4f}; items sent {j['n_sent'].mean():.2f} vs {j['n_sent_sev'].mean():.2f}; budget binds {float(((j['budget'] > 0) & (j['n_sent'] == j['budget'])).mean()):.3f}",
            f"- probe (sigma_s={P['sigma']} m, 1% false alarm on Train benign): theta={P['theta']:.1f} m, d={P['d']:.3f}, f={P['f']:.3f}; d by variant {P['d_by_variant']}; CAM matched {P['matched']:.3f}",
            f"- probe timing: min T_HO {P['t_ho_min']:.2f} s; share T_HO < t_p {P['frac_tho_lt_tp']}",
            "- probe rate, Validation nominal: " + ", ".join(f"{m} {v:.3f}" for m, v in pr.items()) + f" (target {P['rate']})"] + EXTRA

assert not any(f.startswith("gt_") for f in FEATURES), "label column in FEATURES"


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | model | {msg}", flush=True)


def load_cfg():
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def trace_dir(cfg, sc, split):
    if split == "Test":
        raise SystemExit("REFUSED: Test is held-out; only evaluate.py --final may read it.")
    d = ROOT / (cfg.get("paths") or {}).get("processed", "data/processed") / sc / split
    if not d.exists():
        raise SystemExit(f"Missing trace dir {d}")
    return d


def load_variant(cfg, sc, split, var):
    d = trace_dir(cfg, sc, split)
    ev = pd.read_parquet(d / f"evidence_{var}.parquet")
    ev["edge"] = ev["edge"].astype("int64")
    ev["rcv_code"] = pd.factorize(ev["receiver_id"])[0].astype("int64")
    return ev, pd.read_parquet(d / "handovers.parquet"), pd.read_parquet(d / f"sessions_{var}.parquet")


class ItemModel:
    def __init__(self, M, seed):
        p = M["item_model"]
        self.clf = HistGradientBoostingClassifier(
            max_iter=p["max_iter"], learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
            early_stopping=True, validation_fraction=0.1, random_state=seed)
        self.clip, self.max_rows, self.seed = float(M["llr_clip"]), int(p["train_rows_max"]), seed

    @staticmethod
    def X(ev):
        return ev[FEATURES].to_numpy(dtype="float64")

    def fit(self, ev):
        X, y = self.X(ev), ev["gt_msg_attack"].to_numpy().astype(int)
        if len(y) > self.max_rows:
            idx = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.clf.fit(X, y)
        self.base_lo = float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6)))
        return self

    def llr(self, ev):
        p = np.clip(self.clf.predict_proba(self.X(ev))[:, 1], 1e-6, 1 - 1e-6)
        return np.clip(logit(p) - self.base_lo, -self.clip, self.clip)


def oof_llr(M, ev, seed, k):
    out = np.zeros(len(ev))
    for f, (tr_i, te_i) in enumerate(GroupKFold(n_splits=k).split(ev, groups=ev["sender_id"].to_numpy())):
        out[te_i] = ItemModel(M, seed + f).fit(ev.iloc[tr_i]).llr(ev.iloc[te_i])
    return out


class BetaAgg:
    """#7 Josang & Ismail (2002): p = (s+1)/(r+s+2); r = benign-looking, s = incriminating feedback."""
    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def fit(self, df):
        return self

    def p(self, df):
        r, s = df["beta_r"].to_numpy(dtype="float64"), df["beta_s"].to_numpy(dtype="float64")
        return (s + 1.0) / (r + s + 2.0)

    def coefs(self):
        return {"lambda": self.lam}


class Aggregator:
    def __init__(self, C):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))

    def fit(self, df):
        self.m.fit(df[AGG_COLS].to_numpy(dtype="float64"), df["x"].to_numpy().astype(int))
        return self

    def p(self, df):
        return self.m.predict_proba(df[AGG_COLS].to_numpy(dtype="float64"))[:, 1]

    def coefs(self):
        lr = self.m[-1]
        return {"intercept": round(float(lr.intercept_[0]), 3),
                **{c: round(float(v), 3) for c, v in zip(AGG_COLS, lr.coef_[0])}}


def agg_feats(n, ep, llr, w, thr):
    ep = ep.astype(np.int64)
    cnt = np.bincount(ep, weights=w, minlength=n)
    s = np.bincount(ep, weights=llr * w, minlength=n)
    mx, mn = np.full(n, -np.inf), np.full(n, np.inf)
    if len(ep):
        np.maximum.at(mx, ep, llr)
        np.minimum.at(mn, ep, llr)
    has = cnt > 0
    hi1 = np.bincount(ep, weights=w * (llr > thr[0]), minlength=n)
    hi2 = np.bincount(ep, weights=w * (llr > thr[1]), minlength=n)
    return np.column_stack([np.log1p(cnt), np.where(has, s / np.maximum(cnt, 1e-9), 0.0),
                            np.where(has, mx, 0.0), np.where(has, mn, 0.0),
                            np.log1p(hi1), np.log1p(hi2), (~has).astype(float)])


def _find_key(d, k):
    if isinstance(d, dict):
        if k in d:
            return d[k]
        for v in d.values():
            r = _find_key(v, k)
            if r is not None:
                return r
    return None


def _clean_path(sc, split, var):
    for base in ("data/interim/veremi", "data/heldout/veremi"):
        p = ROOT / base / sc / f"{var}_{split}.parquet"
        if p.exists():
            return p
    raise FileNotFoundError(f"raw messages for {sc}/{split}/{var}")


def f2md_table(sc, split, var):
    """Receiver-side F2MD inputs per (receiver, sender) sequence (veins-f2md @340e3cb, LegacyChecks)."""
    num = ["s_x", "s_y", "s_spd", "r_x", "r_y"]
    r = pd.read_parquet(_clean_path(sc, split, var), columns=EXT_KEYS + ["send_ns"] + num)
    for c0 in num:
        r[c0] = pd.to_numeric(r[c0], errors="coerce").astype("float64")
    r = r.sort_values(["receiver_id", "sender_id", "rcv_ns"], kind="stable").drop_duplicates(EXT_KEYS).reset_index(drop=True)
    g = r.groupby(["receiver_id", "sender_id"], sort=False)
    f = lambda c0: r[c0].to_numpy(dtype="float64")
    return r[EXT_KEYS].assign(
        dt=g["send_ns"].diff().to_numpy(dtype="float64") / 1e9,
        dd=np.hypot(g["s_x"].diff().to_numpy(dtype="float64"), g["s_y"].diff().to_numpy(dtype="float64")),
        v=np.abs(f("s_spd")), vp=np.abs(g["s_spd"].shift().to_numpy(dtype="float64")),
        rng_m=np.hypot(f("s_x") - f("r_x"), f("s_y") - f("r_y")),
        fresh=(f("rcv_ns") - f("send_ns")) / 1e9)


def f2md_flag(T, vmax):
    """ThresholdApp rule: flagged if ANY check fails. Sudden-appearance omitted; position-speed uses mean speed."""
    Q = F2MD_P
    dt, dd, v, vp = (T[c0].to_numpy(dtype="float64") for c0 in ["dt", "dd", "v", "vp"])
    with np.errstate(invalid="ignore"):
        pos = dt > 0
        fl = (T["rng_m"].to_numpy(dtype="float64") > Q["range_m"]) | (v > vmax) | (T["fresh"].to_numpy(dtype="float64") > Q["delta_s"])
        fl |= np.isfinite(dt) & (dt < Q["beacon_s"])
        fl |= pos & (dd > vmax * dt + Q["mgt"])
        dv = v - vp
        fl |= pos & ((dv > Q["acc"] * dt) | (-dv > Q["dec"] * dt))
        err = dd - 0.5 * (v + vp) * dt
        fl |= pos & ((err > Q["mgt"] + Q["mgt_up"]) | (err < -(Q["mgt"] + Q["mgt_dn"])))
    return fl.astype(np.int8)


def ext_item_scores(cfg, sc, split, var, ev):
    m = ev[EXT_KEYS].merge(f2md_table(sc, split, var), on=EXT_KEYS, how="left", indicator=True)
    if len(m) != len(ev):
        raise RuntimeError("f2md merge changed row count")
    hit = float((m["_merge"] == "both").mean())
    if hit < 0.99:
        raise RuntimeError(f"f2md: only {hit:.3f} of {sc}/{split}/{var} items matched raw messages")
    out = {"flag_f2md": f2md_flag(m, F2MD_P["vmax"]),
           "flag_f2md_tuned": f2md_flag(m, float(_find_key(cfg, "max_speed_mps") or 60.0))}
    p = ROOT / cfg["paths"]["results"] / "lstm" / sc / f"{var}_{split}.parquet"
    if p.exists():
        m2 = ev[EXT_KEYS].merge(pd.read_parquet(p), on=EXT_KEYS, how="left")
        if len(m2) == len(ev):
            out["llr_lstm"] = m2["llr_lstm"].to_numpy(dtype="float64")
    return out


def ext_fit(cfg, sc, ev_all, ctx, M):
    """Train only: F2MD flag -> LLR calibration; injected fakes = benign-looking under each detector."""
    raw = {}
    for v in VARIANTS:
        mk = (ev_all["variant_"] == v).to_numpy()
        for k2, a2 in ext_item_scores(cfg, sc, "Train", v, ev_all.loc[mk]).items():
            raw.setdefault(k2, np.full(len(ev_all), np.nan))[mk] = a2
    y = ev_all["gt_msg_attack"].to_numpy().astype(int)
    clip, q = float(M["llr_clip"]), float(M["adversary_inject_quantile"])
    ctx["ext_par"], ctx["ext_ctx"] = {}, {}
    for m_, col in EXT_DET.items():
        src = "llr_lstm" if m_ == "lstm_sev" else "flag_" + m_
        a = raw.get(src)
        if a is None or not np.isfinite(a).all():
            log(f"{m_}: scores unavailable for {sc}/Train -> method skipped")
            continue
        note = ""
        if src.startswith("flag_"):
            fl = a.astype(int)
            n1, n0 = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
            a1, b1 = (fl[y == 1].sum() + 1) / (n1 + 2), (fl[y == 0].sum() + 1) / (n0 + 2)
            l1 = float(np.clip(np.log(a1 / b1), -clip, clip))
            l0 = float(np.clip(np.log((1 - a1) / (1 - b1)), -clip, clip))
            ctx["ext_par"][m_] = [l1, l0]
            a = np.where(fl == 1, l1, l0)
            note = f"; flag rate attack {a1:.3f} benign {b1:.3f}; llr(flag) {l1:.2f}, llr(no flag) {l0:.2f}"
        ev_all[col] = a
        ben = np.sort(a[y == 0])
        ctx["ext_ctx"][m_] = {"inject_llr": float(np.quantile(ben, q)), "inject_pool": ben[:max(1, int(0.25 * len(ben)))]}
        auc = {}
        for v in VARIANTS:
            mk = (ev_all["variant_"] == v).to_numpy()
            if 0 < y[mk].sum() < mk.sum():
                auc[v] = round(float(roc_auc_score(y[mk], a[mk])), 3)
        log(f"{m_}: item AUC Train {auc}{note}")
    return ev_all


def ext_apply(cfg, sc, split, var, ev, ctx):
    if not ctx.get("ext_ctx"):
        return ev
    raw = ext_item_scores(cfg, sc, split, var, ev)
    for m_, col in EXT_DET.items():
        if m_ not in ctx["ext_ctx"]:
            continue
        if m_ in ctx["ext_par"]:
            l1, l0 = ctx["ext_par"][m_]
            ev[col] = np.where(raw["flag_" + m_] == 1, l1, l0)
        elif "llr_lstm" in raw and np.isfinite(raw["llr_lstm"]).all():
            ev[col] = raw["llr_lstm"]
        else:
            raise RuntimeError(f"{m_}: scores missing for {sc}/{split}/{var} (run train.py --stage lstm)")
    return ev


def build_episodes(ho, se):
    h = ho.merge(se[["sender_id", "session_uid", "compromised", "onset_ns"]], on="sender_id", how="inner")
    onset = h["onset_ns"].to_numpy(dtype="float64", na_value=np.nan)
    h["x"] = ((h["compromised"].to_numpy() == 1) & (onset <= h["t_exec_ns"].to_numpy())).astype(np.int8)
    return h.sort_values(["sender_id", "ho_idx"]).reset_index(drop=True)


def candidates(e, h, edge_col, t_col):
    """Items of each episode's session held at edge h[edge_col] with report_ns <= h[t_col]."""
    c = h[["ep", "sender_id", edge_col, t_col]].merge(
        e[["sender_id", "edge", "report_ns", "llr", "gt_msg_attack", "rcv_code"]],
        left_on=["sender_id", edge_col], right_on=["sender_id", "edge"], how="inner")
    c = c[c["report_ns"].to_numpy() <= c[t_col].to_numpy()]
    return c[["ep", "report_ns", "llr", "gt_msg_attack", "rcv_code"]].reset_index(drop=True)


def rank_within(ep, key):
    if len(ep) == 0:
        return np.zeros(0, np.int64)
    o = np.lexsort((key, ep))
    eps = ep[o]
    start = np.r_[0, np.flatnonzero(eps[1:] != eps[:-1]) + 1]
    r_sorted = np.arange(len(eps)) - np.repeat(start, np.diff(np.r_[start, len(eps)]))
    r = np.empty(len(eps), np.int64)
    r[o] = r_sorted
    return r


def aoi_key(ep, rcv, rep):
    """AoI-style scheduling: round-robin over evidence sources (receivers), freshest item first per source."""
    if len(ep) == 0:
        return np.zeros(0)
    g = pd.factorize(ep.astype(np.int64) * (1 << 32) + (rcv + (1 << 31)))[0]
    r = rank_within(g, -rep).astype(float)
    span = float(rep.max() - rep.min()) + 1.0
    return r + 0.999 * (rep.max() - rep).astype(float) / span


def voi_myopic(llr, ctx):
    """Myopic single-item VoI surrogate (independence assumption): Bayes-risk reduction when the item
    alone updates the Train prior. Replaced by faithful Holm et al. greedy VoI after full reading."""
    L, p0 = ctx["L"], ctx["prior"]
    def risk(p):
        return ((1 - p)[:, None] * L[:, 0][None, :] + p[:, None] * L[:, 1][None, :]).min(1)
    return risk(np.array([p0]))[0] - risk(expit(logit(p0) + llr))


def robust_key(ep, rep, llr, budget, ctx, rng):
    """TRACE-MEC robust selection v0 (heuristic): reserve q-share of each episode budget for the highest-LLR
    (decision-flipping) items; fill the rest by a randomized golden-ratio sweep over the session's time span
    (temporal coverage resists recency-targeted injection; offset unknown to the adversary)."""
    if len(ep) == 0:
        return np.zeros(0)
    k = np.ceil(ctx["robust_q"] * budget).astype(np.int64)[ep]
    rs = rank_within(ep, -llr)
    tr = rank_within(ep, rep).astype(float)
    off = rng.random(int(ep.max()) + 1)[ep]
    sweep = np.mod(tr * 0.6180339887498949 + off, 1.0)
    return np.where(rs < k, rs.astype(float) - 1e6, sweep)


def bayes(p, x, L):
    E = (1 - p)[:, None] * L[:, 0][None, :] + p[:, None] * L[:, 1][None, :]
    a = E.argmin(1)
    return a, L[a, x]


def simulate_chunk(e, h, prof, methods, ctx, seed, ci, dedup):
    n = len(h)
    h = h.assign(ep=np.arange(n))
    x, t_ho = h["x"].to_numpy(), h["t_ho_s"].to_numpy(dtype="float64")
    own = candidates(e, h, "dst", "t_exec_ns")
    oep, ollr = own["ep"].to_numpy(np.int64), own["llr"].to_numpy(dtype="float64")
    c0 = candidates(e, h, "src", "prep_start_ns")
    c = c0
    rng = np.random.default_rng([seed, ci, 101])   # now only for injected-item content/timing
    eh = ep_hash(h, seed)
    if prof.get("p_suppress", 0) > 0 and len(c):           # adversary hides incriminating items
        elig = ((x[c["ep"].to_numpy()] == 1) & (c["gt_msg_attack"].to_numpy() == 1) & (c["llr"].to_numpy() > 0))
        if prof.get("suppress_mode", "random") == "top":     # hide the most incriminating items first
            ce, cl = c["ep"].to_numpy(np.int64), c["llr"].to_numpy(dtype="float64")
            r = rank_within(ce, np.where(elig, -cl, np.inf))
            ne = np.bincount(ce, weights=elig, minlength=n)
            sup = elig & (r < np.floor(prof["p_suppress"] * ne[ce]))
        else:
            sup = elig & (item_u(eh, c, 1) < prof["p_suppress"])
        c = c[~sup].reset_index(drop=True)
    k, att = int(prof.get("n_inject", 0)), np.flatnonzero(x == 1)
    if k > 0 and len(att):                                    # adversary injects fresh benign-looking items
        ep_f = np.repeat(att, k)
        prep = h["prep_start_ns"].to_numpy().astype(np.int64)
        mode = prof.get("inject_timing", "burst")
        if mode == "burst":
            rep = prep[ep_f] - np.tile(np.arange(1, k + 1, dtype=np.int64), len(att)) * 1_000_000
        else:
            lo = prep - 1_000_000_000
            if len(c0):
                mn = np.full(n, np.iinfo(np.int64).max)
                np.minimum.at(mn, c0["ep"].to_numpy(np.int64), c0["report_ns"].to_numpy(np.int64))
                lo = np.where(mn < prep, mn, lo)
            if mode == "sustained":
                u = rng.random(len(ep_f))
            elif mode == "early":
                u = np.tile((np.arange(k) + 0.5) / k * 0.10, len(att))
            else:
                raise KeyError(f"unknown inject_timing {mode}")
            rep = (lo[ep_f] + u * (prep[ep_f] - lo[ep_f])).astype(np.int64)
        c = pd.concat([c, pd.DataFrame({"ep": ep_f, "report_ns": rep, "llr": (rng.choice(ctx["inject_pool"], len(ep_f)) if prof.get("inject_sampled") else ctx["inject_llr"]),
                                        "gt_msg_attack": 0,
                                        "rcv_code": -1 - np.tile(np.arange(k, dtype=np.int64), len(att))})],
                      ignore_index=True)
    m = len(c)
    cep, crep, cllr = c["ep"].to_numpy(np.int64), c["report_ns"].to_numpy(), c["llr"].to_numpy(dtype="float64")
    crcv = c["rcv_code"].to_numpy(np.int64)
    lost = item_u(eh, c, 2) < prof["p_loss"]
    dupl = item_u(eh, c, 3) < prof["p_dup"]
    extra = np.where(item_u(eh, c, 4) < prof["p_delay"], -prof["delay_mean_s"] * np.log1p(-item_u(eh, c, 5)), 0.0)
    w_fault = 1.0 + dupl * (0.0 if dedup else 1.0)
    item_s = 8.0 * ctx["item_bytes"] / ctx["rate_bps"]
    budget = np.floor(np.maximum(t_ho - ctx["setup_s"], 0.0) / item_s + 1e-9).astype(np.int64)
    n_cand = np.bincount(cep, minlength=n)
    texe = h["t_exec_ns"].to_numpy(dtype="float64")
    oage = np.maximum(texe[oep] - own["report_ns"].to_numpy(dtype="float64"), 0.0) / 1e9
    out = []
    for mname in methods:
        name = BASE.get(mname, "severity" if mname in EXT_DET else mname)
        br = bs = np.zeros(n)
        if name == "no_transfer":
            s_ep, s_llr, s_w = np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
            b = n_sent = n_arr = np.zeros(n, np.int64)
        elif name == "ref_full_clean":
            s_ep, s_llr = c0["ep"].to_numpy(np.int64), c0["llr"].to_numpy(dtype="float64")
            s_w = np.ones(len(s_ep))
            b = n_sent = n_arr = np.bincount(s_ep, minlength=n)
        elif name == "beta_rep":                          # #7: (r, s) summary in one item slot
            lam, tau = float(ctx.get("beta_lambda", 1.0)), ctx["rb_tau"]
            wc = lam ** (np.maximum(texe[cep] - crep.astype("float64"), 0.0) / 1e9)
            wo = lam ** oage
            got = ((budget >= 1) & (np.random.default_rng([seed, ci, 404]).random(n) >= prof["p_loss"])
                   & (ctx["setup_s"] + item_s <= t_ho + 1e-9))
            br = (np.bincount(oep, weights=wo * (ollr <= tau), minlength=n)
                  + got * np.bincount(cep, weights=wc * (cllr <= tau), minlength=n))
            bs = (np.bincount(oep, weights=wo * (ollr > tau), minlength=n)
                  + got * np.bincount(cep, weights=wc * (cllr > tau), minlength=n))
            s_ep, s_llr, s_w = np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
            b, n_sent, n_arr = budget, (budget >= 1).astype(np.int64), got.astype(np.int64)
        elif name == "central_ma":                        # #9: flagged reports from other edges, delay D
            fl = e.loc[e["llr"].to_numpy() > ctx["rb_tau"], ["sender_id", "edge", "report_ns", "llr"]]
            mm = h[["ep", "sender_id", "dst", "t_exec_ns"]].merge(fl, on="sender_id", how="inner")
            kma = ((mm["report_ns"].to_numpy() <= mm["t_exec_ns"].to_numpy() - int(ctx["ma_delay_s"] * 1e9))
                   & (mm["edge"].to_numpy() != mm["dst"].to_numpy()))
            s_ep, s_llr = mm["ep"].to_numpy(np.int64)[kma], mm["llr"].to_numpy(dtype="float64")[kma]
            s_w = np.ones(len(s_ep))
            b, n_sent = np.zeros(n, np.int64), np.zeros(n, np.int64)
            n_arr = np.bincount(s_ep, minlength=n)
        elif name == "full_no_deadline":
            s_ep, s_llr, s_w = cep, cllr, np.ones(m)
            b = n_sent = n_arr = n_cand
        else:
            if name == "fifo": key = crep
            elif name == "recency": key = -crep
            elif name == "severity": key = -cllr
            elif name == "random": key = item_u(eh, c, 6)
            elif name == "aoi": key = aoi_key(cep, crcv, crep)
            elif name == "voi_myopic": key = -voi_myopic(cllr, ctx)
            elif name == "robust_mix": key = robust_key(cep, crep, cllr, budget, ctx, np.random.default_rng([seed, ci, 303]))
            else: raise KeyError(f"unknown method {name}")
            r = rank_within(cep, key)
            sent = (r < budget[cep]) & ((cllr > ctx["rb_tau"]) if mname in POS_ONLY else True)
            arr = sent & ~lost & (ctx["setup_s"] + (r + 1) * item_s + extra <= t_ho[cep] + 1e-9)
            s_ep, s_llr, s_w = cep[arr], cllr[arr], w_fault[arr]
            b, n_sent, n_arr = budget, np.bincount(cep[sent], minlength=n), np.bincount(cep[arr], minlength=n)
        a_ep, a_llr, a_w = np.r_[oep, s_ep].astype(np.int64), np.r_[ollr, s_llr], np.r_[np.ones(len(oep)), s_w]
        if mname in RB:                                   # robust belief: ignore forgeable reassuring items
            keep = a_llr > ctx["rb_tau"]
            a_ep, a_llr, a_w = a_ep[keep], a_llr[keep], a_w[keep]
        F = agg_feats(n, a_ep, a_llr, a_w, ctx["thr"])
        FM = mono_feats(n, a_ep, a_llr, a_w, ctx["rb_tau"], ctx["thr"])
        d = pd.DataFrame({"session_uid": h["session_uid"].to_numpy(), "ho_idx": h["ho_idx"].to_numpy(),
                          "x": x, "method": mname, "n_cand": n_cand, "budget": b,
                          "n_sent": n_sent, "n_arrived": n_arr})
        d[AGG_COLS] = F
        d[MONO_COLS] = FM
        d["beta_r"], d["beta_s"] = br, bs
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _simulate_core(ev, ho, se, prof, methods, ctx, seed, chunk, dedup=True):
    h_all = build_episodes(ho, se)
    sids = h_all["sender_id"].unique()
    out = []
    for ci in range(0, len(sids), chunk):
        part = list(sids[ci:ci + chunk])
        out.append(simulate_chunk(ev[ev["sender_id"].isin(part)], h_all[h_all["sender_id"].isin(part)],
                                  prof, methods, ctx, seed, ci, dedup))
    return pd.concat(out, ignore_index=True)


def simulate(ev, ho, se, prof, methods, ctx, seed, chunk, dedup=True):
    """v0.8b wrapper: core methods run exactly as before; #8/#10 rerun the same machinery on their own score."""
    core = [m for m in methods if m not in EXT_DET and m not in EXT_ALIAS]
    out = [_simulate_core(ev, ho, se, prof, core, ctx, seed, chunk, dedup)] if core else []
    for m in methods:
        k_ = EXT_ALIAS.get(m, m)
        cx = (ctx.get("ext_ctx") or {}).get(k_)
        if k_ in EXT_DET and cx is not None and EXT_DET[k_] in ev.columns:
            out.append(_simulate_core(ev.assign(llr=ev[EXT_DET[k_]].to_numpy()), ho, se, prof, [m],
                                      dict(ctx, **cx), seed, chunk, dedup))
    return pd.concat(out, ignore_index=True)


def decide(df, agg, L):
    p = np.zeros(len(df))
    for mth, idx in df.groupby("method").indices.items():
        p[idx] = agg[mth].p(df.iloc[idx])
    a, loss = bayes(p, df["x"].to_numpy().astype(int), L)
    return df.assign(p=p, action=a, loss=loss)


def summarize(df):
    keys = ["profile", "variant", "method"]
    d = df.assign(x1=df.x == 1, x0=df.x == 0, miss=(df.action == 0) & (df.x == 1),
                  fq=(df.action == 2) & (df.x == 0), q1=(df.action == 2) & (df.x == 1), rs=df.action == 1)
    s = d.groupby(keys, sort=False).agg(
        n_ep=("x", "size"), n_x1=("x1", "sum"), n_x0=("x0", "sum"), loss=("loss", "mean"),
        miss=("miss", "sum"), fq=("fq", "sum"), q1=("q1", "sum"), restrict=("rs", "mean"),
        cand=("n_cand", "mean"), budget=("budget", "mean"), arrived=("n_arrived", "mean"))
    aucs = {k: (roc_auc_score(g["x"], g["p"]) if 0 < g["x"].sum() < len(g) else np.nan)
            for k, g in df.groupby(keys, sort=False)}
    s["belief_auc"] = [aucs.get(k, np.nan) for k in s.index]
    s["miss_rate"] = s.miss / s.n_x1.clip(lower=1)
    s["false_q_rate"] = s.fq / s.n_x0.clip(lower=1)
    s["detect_q_rate"] = s.q1 / s.n_x1.clip(lower=1)
    cols = ["n_ep", "n_x1", "loss", "belief_auc", "miss_rate", "false_q_rate", "detect_q_rate",
            "restrict", "cand", "budget", "arrived"]
    return s[cols].round(4).reset_index()


def loss_matrix(miss, rb, rho):
    return np.array([[0.0, miss], [rb, rho * miss], [1.0, 0.0]], dtype=float)


def loss_sweep(df, S):
    """Re-decide from stored beliefs p under each loss setting; two-action = restrict disabled."""
    x, p = df["x"].to_numpy().astype(int), df["p"].to_numpy()
    sets = [(f"miss={m:g},rb={rb:g},rho={rho:g}", loss_matrix(m, rb, rho))
            for m in S["miss"] for rb in S["restrict_benign"] for rho in S["restrict_residual"]]
    sets += [(f"miss={m:g},two-action", loss_matrix(m, 1e9, 1.0)) for m in S["miss"]]
    rows = []
    for name, Lm in sets:
        a, l = bayes(p, x, Lm)
        g = (df[["profile", "variant", "method"]].assign(loss=l, rs=(a == 1))
             .groupby(["profile", "variant", "method"], sort=False).agg(loss=("loss", "mean"), rs=("rs", "mean"))
             .reset_index())
        rows.append(g.groupby(["profile", "method"], sort=False)
                    .agg(mean_loss=("loss", "mean"), worst_loss=("loss", "max"), restrict=("rs", "mean"))
                    .reset_index().assign(setting=name))
    return pd.concat(rows, ignore_index=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def smoke(cfg, sc, seed):
    t0, M = time.time(), cfg["model"]
    A, chunk = M["aggregator"], int(M["chunk_senders"])
    L = np.array([M["loss"][a] for a in ACTIONS], dtype=float)
    tr = {v: load_variant(cfg, sc, "Train", v) for v in VARIANTS}
    ev_all = pd.concat([tr[v][0].assign(variant_=v) for v in VARIANTS], ignore_index=True)
    ev_all["llr"] = oof_llr(M, ev_all, seed, int(A["oof_folds"]))
    y = ev_all["gt_msg_attack"].to_numpy()
    auc_tr = {}
    for v in VARIANTS:
        mk = (ev_all["variant_"] == v).to_numpy()
        if 0 < y[mk].sum() < mk.sum():
            auc_tr[v] = roc_auc_score(y[mk], ev_all["llr"].to_numpy()[mk])
    model = ItemModel(M, seed).fit(ev_all)
    log(f"item model: OOF LLRs + final fit on {sc}/Train ({len(ev_all):,} rows)")
    ctx = dict(L=L, rb_tau=float((M.get("robust") or {}).get("rb_tau", 0.0)), prior=float(np.mean(np.concatenate([build_episodes(tr[v][1], tr[v][2])["x"].to_numpy() for v in VARIANTS]))), robust_q=float((M.get("robust") or {}).get("q_severity", 0.5)), rate_bps=float(M["transfer_rate_bps"]), item_bytes=float(cfg["traces"]["evidence_item_bytes"]),
               setup_s=float(M["setup_s"]), thr=[float(t) for t in A["hi_thresholds"]],
               inject_llr=float(np.quantile(ev_all.loc[y == 0, "llr"], M["adversary_inject_quantile"])),
               inject_pool=np.sort(ev_all.loc[y == 0, "llr"].to_numpy())[:max(1, int(0.25 * int((y == 0).sum())))])
    ev_all = ext_fit(cfg, sc, ev_all, ctx, M)
    ctx["ma_delay_s"] = float(EXT_CFG["ma_delay_s"])
    lam_auc = {}
    for lam in EXT_CFG["beta_lambda_grid"]:
        ctx["beta_lambda"] = float(lam)
        g = pd.concat([simulate(ev_all[ev_all["variant_"] == v], tr[v][1], tr[v][2], M["faults"]["nominal"],
                                ["beta_rep"], ctx, seed, chunk) for v in VARIANTS], ignore_index=True)
        lam_auc[float(lam)] = round(float(roc_auc_score(g["x"], BetaAgg(lam).p(g))), 4)
    ctx["beta_lambda"] = max(lam_auc, key=lam_auc.get)
    beta_agg = BetaAgg(ctx["beta_lambda"])
    log(f"beta_rep: lambda -> Train-nominal AUC {lam_auc}; chosen {ctx['beta_lambda']}")
    fit = pd.concat([simulate(ev_all[ev_all["variant_"] == v], tr[v][1], tr[v][2], M["faults"]["nominal"],
                              METHODS, ctx, seed, chunk) for v in VARIANTS], ignore_index=True)
    mono = MonoAgg(float(A["C"])).fit(fit[fit["method"] == "ref_full_clean_rb"])
    mono_l = MonoAgg(float(A["C"])).fit(fit[fit["method"] == "lstm_ref_rb"]) if (fit["method"] == "lstm_ref_rb").any() else None
    if mono_l is not None:
        pl = np.asarray(ctx["ext_ctx"]["lstm_sev"]["inject_pool"])
        log(f"trace_lstm: monotone RB aggregator on LSTM scores {mono_l.coefs()}; "
            f"injected LSTM pool share > rb_tau (Prop. 1 violated if > 0): {float((pl > ctx['rb_tau']).mean()):.4f}")
    agg = {mth: (mono_l if mth in LSTM_RB else mono if mth in RB else beta_agg if mth == "beta_rep" else Aggregator(float(A["C"])).fit(g)) for mth, g in fit.groupby("method")}
    auc_fit = {mth: round(float(roc_auc_score(g["x"], agg[mth].p(g))), 3) for mth, g in fit.groupby("method")}
    prior = float(fit["x"].mean())
    PI_tr = pd.concat([probe_info(cfg, sc, "Train", v, build_episodes(tr[v][1], tr[v][2])) for v in VARIANTS], ignore_index=True)
    fit = decide(fit, agg, L)
    OP = fit_operating(fit, PI_tr, M, L, seed)
    OP["mono"] = mono.coefs()
    log(f"v0.7 operating point on Train nominal: probe theta={OP['probe']['theta']:.1f} m d={OP['probe']['d']:.3f} f={OP['probe']['f']:.3f}")
    del tr, ev_all, PI_tr
    log(f"per-method aggregators fitted on Train nominal episodes (P(x=1)={prior:.3f}, AUC={auc_fit})")
    res, auc_va = [], {}
    for v in VARIANTS:
        ev, ho, se = load_variant(cfg, sc, "Validation", v)
        ev["llr"] = model.llr(ev)
        ev = ext_apply(cfg, sc, "Validation", v, ev, ctx)
        PI_va = probe_info(cfg, sc, "Validation", v, build_episodes(ho, se))
        yv = ev["gt_msg_attack"].to_numpy()
        if 0 < yv.sum() < len(yv):
            auc_va[v] = roc_auc_score(yv, ev["llr"])
        for pname, prof in M["faults"].items():
            d = apply_operating(decide(simulate(ev, ho, se, prof, METHODS, ctx, seed, chunk), agg, L), PI_va, OP, L, seed)
            res.append(d.assign(variant=v, profile=pname))
        log(f"simulated {sc}/Validation/{v}")
    df = pd.concat(res, ignore_index=True)
    tab = summarize(df)
    ps = np.linspace(0, 1, 100001)
    a_ps = bayes(ps, np.zeros(len(ps), int), L)[0]
    t1, t2 = ps[np.argmax(a_ps >= 1)], ps[np.argmax(a_ps >= 2)]
    out = ROOT / cfg["paths"]["results"] / "model_smoke"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"episodes_{sc}_s{seed}.parquet", index=False)
    tab.to_csv(out / f"summary_{sc}_s{seed}.csv", index=False)
    man = {"script": "model.py --smoke", "version": "v0.7.1", "ext_version": "v0.9", "git": git_commit(), "time": datetime.now().isoformat(),
           "scenario": sc, "seed": seed, "model_cfg": M,
           "ctx": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ctx.items() if k != "inject_pool"},
           "item_auc_train_oof": auc_tr, "item_auc_validation": auc_va,
           "aggregator_coefs_std": {m: a.coefs() for m, a in agg.items()}, "aggregator_train_auc": auc_fit, "operating": OP}
    (out / f"manifest_{sc}_s{seed}.json").write_text(json.dumps(man, indent=2, default=str))
    item_ms = 8 * ctx["item_bytes"] / ctx["rate_bps"] * 1000
    R = [f"# model.py v0.7 smoke test (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
         f"- {sc}; fit on Train (variants pooled), evaluated on Validation; seed {seed}; git {git_commit()}",
         "- item AUC, Train out-of-fold: " + ", ".join(f"{k} {v:.3f}" for k, v in auc_tr.items()),
         "- item AUC, Validation: " + ", ".join(f"{k} {v:.3f}" for k, v in auc_va.items()),
         f"- aggregators: one per method, each fitted on its own nominal Train episodes (equal-tuning rule); P(x=1)={prior:.3f}; train AUC {auc_fit}",
         f"- ref_full_clean aggregator coefficients: {agg['ref_full_clean'].coefs()}",
         f"- injected-item LLR = {ctx['inject_llr']:.3f}; item tx time = {item_ms:.2f} ms",
         f"- Bayes thresholds from loss matrix: allow if p < {t1:.4f}; quarantine if p >= {t2:.4f}",
         f"- runtime {time.time() - t0:.0f} s", "", "```", tab.to_string(index=False), "```"]
    sw = loss_sweep(df, M["loss_sweep"])
    sw.to_csv(out / f"loss_sweep_{sc}_s{seed}.csv", index=False)
    n_set = sw["setting"].nunique()
    bud = sw[sw["method"].isin(BUDGETED)]
    wins = pd.concat([bud.loc[bud.groupby(["profile", "setting"])[c].idxmin()].groupby(["profile", "method"]).size()
                      .rename(f"wins_{c}") for c in ["mean_loss", "worst_loss"]], axis=1).fillna(0).astype(int)
    nom = sw[sw["profile"] == "nominal"].set_index(["setting", "method"])["mean_loss"].unstack()
    ref_ok = int((nom["ref_full_clean"] <= nom[BUDGETED].min(axis=1) + 1e-9).sum())
    rs = sw[(sw["profile"] == "nominal") & (sw["method"] == "ref_full_clean")]["restrict"]
    R += ["", f"## Loss-matrix sensitivity ({n_set} settings; full table in loss_sweep CSV)",
          "wins = settings where the method has the lowest loss among budgeted methods (all budgeted methods)",
          "```", wins.reset_index().to_string(index=False), "```",
          f"- nominal: ref_full_clean <= best budgeted method (mean loss) in {ref_ok}/{n_set} settings",
          f"- restrict share (nominal, ref_full_clean) across settings: {rs.min():.2f}-{rs.max():.2f}"]
    R += report_v07(df, OP, ctx, sc, seed, out)
    (ROOT / "docs" / "model_smoke.md").write_text("\n".join(R) + "\n")
    log("Done. Report: docs/model_smoke.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scenario", default="InTAS_highway_2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.smoke:
        raise SystemExit("Only --smoke is implemented (training lives in train.py).")
    smoke(load_cfg(), a.scenario, a.seed)


if __name__ == "__main__":
    main()
