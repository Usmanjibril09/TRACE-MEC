#!/usr/bin/env python3
"""TRACE-MEC preprocessing (single staged script).

Stages
  profile : read raw archives in place (no extraction); write schema/statistics reports to
            results/profile/ and a readable summary to docs/profile_summary.md.
  clean   : type-normalised VeReMi parquet (Train/Validation -> data/interim, Test -> data/heldout),
            session label tables, integrity/leakage diagnostics; HetNet long-format parquet.
  traces  : sessions, speed-driven handover timelines, causal evidence features + gt_ labels
            -> data/processed (Train/Validation), data/heldout (Test). Faults applied later in model.py.

Usage
  python src/preprocess.py --stage profile                         # dev scenario + HetNet
  python src/preprocess.py --stage profile --scenarios InTAS_urban_2 --skip-hetnet
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import ROOT, get_logger, load_config, save_json, set_seed, write_manifest

NS_PER_S = 1e9
KEY = ["sender_id", "messageID"]
COLS = ["receiver_id", "rcvTime", "sendTime", "sender_id", "sender_alias", "messageID", "attacker",
        "s_x", "s_y", "s_spd", "s_acl", "s_hed", "r_x", "r_y"]
NUMERIC = ["rcvTime", "sendTime", "sender_alias", "messageID", "attacker",
           "s_x", "s_y", "s_spd", "s_acl", "s_hed", "r_x", "r_y"]
FMT = lambda x: f"{x:.4g}"


# ============================================================================ VeReMi
def archive_stem(scenario: str, variant: str) -> str:
    return scenario if variant == "baseline" else f"{scenario}_{variant}"


def iter_receiver_files(vdir: Path, scenario: str, variant: str, split: str):
    """Yield (receiver_id, messages) from outer zip -> {split}/inner zip, fully in memory."""
    stem = archive_stem(scenario, variant)
    with zipfile.ZipFile(vdir / f"{stem}.zip") as outer:
        inner_bytes = outer.read(f"{stem}/{split}/{stem}.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for name in inner.namelist():
            if name.endswith(".json"):
                yield Path(name).stem, json.loads(inner.read(name))


def _xy(pos):
    if pos is None:
        return None, None
    parts = pos.split(",") if isinstance(pos, str) else list(pos)
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError, TypeError):
        return None, None


def flatten(msgs: list, receiver_id) -> list:
    rows = []
    for m in msgs:
        s, r = m.get("sender") or {}, m.get("receiver") or {}
        sx, sy = _xy(s.get("pos"))
        rx, ry = _xy(r.get("pos"))
        rows.append((receiver_id, m.get("rcvTime"), m.get("sendTime"), m.get("sender_id"),
                     m.get("sender_alias"), m.get("messageID"), m.get("attacker"),
                     sx, sy, s.get("spd"), s.get("acl"), s.get("hed"), rx, ry))
    return rows


def to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def describe_schema(m):
    if m is None:
        return None
    return {k: describe_schema(v) if isinstance(v, dict) else type(v).__name__ for k, v in m.items()}


def load_split(vdir, scenario, variant, split):
    rows, first = [], None
    for rid, msgs in iter_receiver_files(vdir, scenario, variant, split):
        if first is None and msgs:
            first = msgs[0]
        rows.extend(flatten(msgs, rid))
    return to_frame(rows), first


def load_ground_truth(vdir, cfg, scenario, split):
    fname = cfg["veremi"]["ground_truth_files"][split].format(scenario=scenario)
    msgs = json.loads((vdir / fname).read_bytes())
    return to_frame(flatten(msgs, None)), (msgs[0] if msgs else None), fname


def split_stats(df, scenario, variant, split) -> dict:
    lat_ms = (df.rcvTime - df.sendTime) / NS_PER_S * 1e3
    has_att = float(df.attacker.notna().mean()) if len(df) else 0.0
    row = dict(scenario=scenario, variant=variant, split=split,
               n_receiver_files=int(df.receiver_id.nunique()), n_msgs=len(df),
               n_senders=int(df.sender_id.nunique()),
               n_unique_messageID=int(df.messageID.nunique()),
               n_dup_receiver_messageID=int(df.duplicated(["receiver_id", "messageID"]).sum()),
               t_rcv_min_s=df.rcvTime.min() / NS_PER_S, t_rcv_max_s=df.rcvTime.max() / NS_PER_S,
               lat_median_ms=lat_ms.median(), lat_p99_ms=lat_ms.quantile(0.99),
               n_negative_latency=int((lat_ms < 0).sum()),
               n_nan_time=int(df.rcvTime.isna().sum() + df.sendTime.isna().sum()),
               frac_has_attacker_field=has_att,
               aliases_per_sender_mean=df.groupby("sender_id").sender_alias.nunique().mean(),
               msgs_per_sender_median=df.groupby("sender_id").size().median(),
               sender_spd_median=df.s_spd.median(), sender_spd_p95=df.s_spd.quantile(0.95))
    row["duration_s"] = row["t_rcv_max_s"] - row["t_rcv_min_s"]
    if has_att > 0:
        g = df.dropna(subset=["attacker"]).groupby("sender_id").attacker.agg(["min", "max"])
        att = (df.attacker == 1).to_numpy()
        row.update(frac_attack_msgs=float(att.mean()),
                   n_attacker_senders=int((g["max"] == 1).sum()),
                   frac_senders_label_constant=float((g["min"] == g["max"]).mean()),
                   lat_median_ms_attack=float(lat_ms[att].median()) if att.any() else np.nan,
                   lat_median_ms_benign=float(lat_ms[~att].median()))
    return row


def sender_labels(df, scenario, variant, split) -> pd.DataFrame:
    d = df.dropna(subset=["attacker"])
    g = d.groupby("sender_id").agg(n_msgs=("attacker", "size"), n_attack=("attacker", "sum")).reset_index()
    g["frac_attack"] = g.n_attack / g.n_msgs
    g.insert(0, "split", split)
    g.insert(0, "variant", variant)
    g.insert(0, "scenario", scenario)
    return g


def gt_overlap(gt, v, scenario, variant, split, gt_file) -> dict:
    g = gt.drop_duplicates(KEY)
    m = g.merge(v, on=KEY, suffixes=("_gt", "_v"))
    row = dict(scenario=scenario, variant=variant, split=split, gt_file=gt_file,
               n_gt_rows=len(gt), n_gt_unique_keys=len(g), n_variant_unique_keys=len(v),
               n_common_keys=len(m),
               n_common_messageID_only=len(set(g.messageID.dropna()) & set(v.messageID.dropna())),
               frac_gt_in_variant=len(m) / max(len(g), 1),
               frac_variant_in_gt=len(m) / max(len(v), 1))
    if len(m):
        def close(a, b):
            return np.isclose(m[a].to_numpy(float), m[b].to_numpy(float), atol=1e-6, equal_nan=True)
        row.update(frac_rcvTime_equal=float(close("rcvTime_gt", "rcvTime_v").mean()),
                   frac_spd_differs=float((~close("s_spd_gt", "s_spd_v")).mean()),
                   frac_pos_differs=float((~(close("s_x_gt", "s_x_v") & close("s_y_gt", "s_y_v"))).mean()))
    return row


def profile_veremi(cfg, scenarios, out, log):
    vdir = ROOT / cfg["paths"]["veremi_raw"]
    split_rows, gt_rows, label_tables, schema = [], [], [], {}
    for sc in scenarios:
        keys = {}
        for var in cfg["veremi"]["variants"]:
            for sp in cfg["veremi"]["splits"]:
                df, first = load_split(vdir, sc, var, sp)
                schema.setdefault(var, describe_schema(first))
                split_rows.append(split_stats(df, sc, var, sp))
                if df.attacker.notna().any():
                    label_tables.append(sender_labels(df, sc, var, sp))
                keys[(var, sp)] = df[KEY + ["rcvTime", "s_spd", "s_x", "s_y"]].drop_duplicates(KEY)
                log.info(f"{sc} | {var:15s} | {sp:10s} | {len(df):>10,} msgs | {df.sender_id.nunique():>6,} senders")
                del df
        for sp in cfg["veremi"]["splits"]:
            gt, first, fname = load_ground_truth(vdir, cfg, sc, sp)
            schema.setdefault("ground_truth", describe_schema(first))
            for var in cfg["veremi"]["variants"]:
                gt_rows.append(gt_overlap(gt, keys[(var, sp)], sc, var, sp, fname))
            log.info(f"{sc} | ground truth {sp:10s} | {len(gt):>10,} rows")
    vprof, gtov = pd.DataFrame(split_rows), pd.DataFrame(gt_rows)
    vprof.to_csv(out / "veremi_split_profile.csv", index=False)
    gtov.to_csv(out / "veremi_gt_overlap.csv", index=False)
    if label_tables:
        pd.concat(label_tables).to_csv(out / "veremi_sender_labels.csv", index=False)
    save_json(schema, out / "veremi_schema.json")
    return vprof, gtov, schema


# ============================================================================ HetNet
def a3_cho(rsrp: np.ndarray, t: np.ndarray, dt: float, hys: float, ttt: float, prep_off: float) -> list:
    """Profiling heuristic: A3 execution (neighbour > serving + hys for TTT) with a CHO-like
    preparation marker (neighbour > serving + prep_off). prep_window_s = execution - preparation start."""
    k = max(1, int(round(ttt / dt)))
    r = np.where(np.isfinite(rsrp), rsrp, -np.inf)
    T, G = r.shape
    serving = int(np.argmax(r[0]))
    cnt = np.zeros(G, dtype=int)
    prep = np.full(G, -1)
    events = []
    with np.errstate(invalid="ignore"):
        for i in range(1, T):
            diff = r[i] - r[i, serving]
            p_ok = diff > prep_off
            p_ok[serving] = False
            prep = np.where(p_ok, np.where(prep < 0, i, prep), -1)
            e_ok = diff > hys
            e_ok[serving] = False
            cnt = np.where(e_ok, cnt + 1, 0)
            ready = np.flatnonzero(cnt >= k)
            if ready.size:
                tgt = int(ready[np.argmax(r[i, ready])])
                pw = float(t[i] - t[prep[tgt]]) if prep[tgt] >= 0 else np.nan
                events.append({"t_s": float(t[i]), "src": serving, "dst": tgt, "prep_window_s": pw})
                serving = tgt
                cnt[:] = 0
                prep[:] = -1
    return events


def profile_hetnet(cfg, out, log):
    H, P = cfg["hetnet"], cfg["profiling"]
    dt = H["dt_s"]
    frames = {}
    with zipfile.ZipFile(ROOT / cfg["paths"]["hetnet_raw"] / H["archive"]) as z:
        for n in sorted(z.namelist()):
            if n.endswith("traces.csv"):
                ue, gnb = (int(x) for x in n.split("/")[:2])
                frames[(ue, gnb)] = pd.read_csv(io.BytesIO(z.read(n)))
    log.info(f"HetNet: loaded {len(frames)} trace files")

    counters = ["TxBytes", "TxPackets", "RxBytes", "RxPackets", "LatencySum", "JitterSum", "LostPackets"]
    file_rows = []
    for (ue, gnb), df in sorted(frames.items()):
        mono = {c: bool((df[c].diff().dropna() >= 0).all()) for c in counters}
        dts = np.unique(np.round(df.Time.diff().dropna().to_numpy(float), 6))
        file_rows.append(dict(
            ue=ue, gnb=gnb, rows=len(df), t_min=float(df.Time.min()), t_max=float(df.Time.max()),
            dt_values=",".join(f"{x:g}" for x in dts[:5]), n_nan=int(df.isna().sum().sum()),
            all_counters_monotonic=all(mono.values()),
            nonmonotonic=",".join(k for k, ok in mono.items() if not ok),
            rsrp_min=float(df.Rsrp.min()), rsrp_median=float(df.Rsrp.median()), rsrp_max=float(df.Rsrp.max()),
            n_rsrp_nonfinite=int((~np.isfinite(df.Rsrp.to_numpy(float))).sum()),
            dist_min=float(df.Distance.min()), dist_max=float(df.Distance.max()),
            final_loss_frac=float(df.LostPackets.iloc[-1] / max(df.TxPackets.iloc[-1], 1)),
            latsum_median=float(df.LatencySum.median()), latlast_median=float(df.LatencyLast.median())))
    hfile = pd.DataFrame(file_rows)

    ue_rows, ev_rows = [], []
    for ue in range(H["n_ues"]):
        dfs = [frames[(ue, g)] for g in range(H["n_gnbs"])]
        T = min(len(d) for d in dfs)
        t = dfs[0].Time.to_numpy(float)[:T]
        time_dev = max(float(np.abs(d.Time.to_numpy(float)[:T] - t).max()) for d in dfs)
        pos = [d["UE Position"].iloc[:T].str.split(":", expand=True).astype(float).to_numpy() for d in dfs]
        pos_dev = max(float(np.abs(p - pos[0]).max()) for p in pos)
        xy = pos[0][:, :2]
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        spd = step / np.diff(t)
        rsrp = np.column_stack([d.Rsrp.to_numpy(float)[:T] for d in dfs])
        best = np.argmax(np.where(np.isfinite(rsrp), rsrp, -np.inf), axis=1)
        evs = a3_cho(rsrp, t, dt, P["a3_hysteresis_db"], P["a3_ttt_s"], P["cho_prep_offset_db"])
        for i, e in enumerate(evs):
            prev = evs[i - 1] if i else None
            e["pingpong"] = bool(prev is not None and e["dst"] == prev["src"]
                                 and e["t_s"] - prev["t_s"] <= P["pingpong_window_s"])
            e["ue"] = ue
        ev_rows += evs
        iho = np.diff([e["t_s"] for e in evs]) if len(evs) > 1 else np.array([])
        pw = np.array([e["prep_window_s"] for e in evs], dtype=float)
        ue_rows.append(dict(
            ue=ue, samples=T, time_dev_s=time_dev, pos_dev_m=pos_dev,
            x_min=float(xy[:, 0].min()), x_max=float(xy[:, 0].max()),
            y_min=float(xy[:, 1].min()), y_max=float(xy[:, 1].max()),
            path_len_m=float(step.sum()), spd_mean=float(np.nanmean(spd)),
            spd_p05=float(np.nanpercentile(spd, 5)), spd_p95=float(np.nanpercentile(spd, 95)),
            n_best_cell_changes=int((np.diff(best) != 0).sum()), n_a3_ho=len(evs),
            n_pingpong=int(sum(e["pingpong"] for e in evs)),
            inter_ho_median_s=float(np.median(iho)) if iho.size else np.nan,
            prep_window_median_s=float(np.nanmedian(pw)) if np.isfinite(pw).any() else np.nan))
    hue, hev = pd.DataFrame(ue_rows), pd.DataFrame(ev_rows)
    hfile.to_csv(out / "hetnet_file_profile.csv", index=False)
    hue.to_csv(out / "hetnet_ue_profile.csv", index=False)
    hev.to_csv(out / "hetnet_handover_events_profiling.csv", index=False)
    log.info(f"HetNet: {len(hev)} A3 handovers across {H['n_ues']} UEs (profiling heuristic)")
    return hfile, hue, hev


# ============================================================================ clean
F64 = pa.float64()
CLEAN_SCHEMA = pa.schema([
    ("receiver_id", pa.string()), ("sender_id", pa.string()),
    ("rcv_ns", pa.int64()), ("send_ns", pa.int64()),
    ("sender_alias", pa.int64()), ("message_id", pa.int64()),
    ("attacker", pa.int8()), ("label_from_field", pa.bool_()),
    ("s_x", F64), ("s_y", F64), ("s_pos_nx", F64), ("s_pos_ny", F64),
    ("s_spd", F64), ("s_spd_noise", F64), ("s_acl", F64), ("s_acl_noise", F64),
    ("s_hed", F64), ("s_hed_noise", F64), ("s_profile", pa.string()),
    ("r_x", F64), ("r_y", F64), ("r_spd", F64), ("r_hed", F64),
])
RAW_COLS = ["receiver_id", "sender_id", "rcv_ns", "send_ns", "sender_alias", "message_id", "attacker",
            "s_x", "s_y", "s_pos_nx", "s_pos_ny", "s_spd", "s_spd_noise", "s_acl", "s_acl_noise",
            "s_hed", "s_hed_noise", "s_profile", "r_x", "r_y", "r_spd", "r_hed"]
INT_COLS = ["rcv_ns", "send_ns", "sender_alias", "message_id"]
FLOAT_COLS = [f.name for f in CLEAN_SCHEMA if f.type == F64]
JOIN_KEY = ["receiver_id", "sender_id", "message_id"]


def _q(s, qs=(0.0, 0.5, 0.9, 0.99, 1.0)) -> str:
    s = pd.Series(s, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "n=0"
    return "  ".join(f"q{int(q * 100)}={s.quantile(q):.4g}" for q in qs) + f"  (n={len(s)})"


def _pair(v):
    if v is None:
        return None, None
    p = v.split(",") if isinstance(v, str) else list(v)
    return (p[0], p[1]) if len(p) >= 2 else (None, None)


def flatten_clean(msgs: list, receiver_id) -> list:
    # distance_to_road_edge is deliberately NOT extracted: it exists only in attack variants (format leakage).
    out = []
    for m in msgs:
        s, r = m.get("sender") or {}, m.get("receiver") or {}
        sx, sy = _pair(s.get("pos"))
        nx, ny = _pair(s.get("pos_noise"))
        rx, ry = _pair(r.get("pos"))
        out.append((receiver_id, m.get("sender_id"), m.get("rcvTime"), m.get("sendTime"),
                    m.get("sender_alias"), m.get("messageID"), m.get("attacker"),
                    sx, sy, nx, ny, s.get("spd"), s.get("spd_noise"), s.get("acl"), s.get("acl_noise"),
                    s.get("hed"), s.get("hed_noise"), s.get("driversProfile"),
                    rx, ry, r.get("spd"), r.get("hed")))
    return out


def normalize_chunk(rows: list, is_baseline: bool, ctx: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RAW_COLS)
    for c in INT_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            raise SystemExit(f"[{ctx}] {int(v.isna().sum())} unparsable values in integer field '{c}'")
        df[c] = v.astype("int64")
    if is_baseline:
        if df.attacker.notna().any():
            raise SystemExit(f"[{ctx}] baseline file unexpectedly contains an 'attacker' field")
        df["attacker"] = 0
        df["label_from_field"] = False
    else:
        a = pd.to_numeric(df.attacker, errors="coerce")
        if a.isna().any() or not a.isin([0, 1]).all():
            raise SystemExit(f"[{ctx}] missing/invalid 'attacker' labels: {int(a.isna().sum())} NaN")
        df["attacker"] = a
        df["label_from_field"] = True
    df["attacker"] = df.attacker.astype("int8")
    df["label_from_field"] = df.label_from_field.astype(bool)
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("receiver_id", "sender_id"):
        df[c] = df[c].astype(object)
    df["s_profile"] = df.s_profile.astype(str).str.upper().where(df.s_profile.notna(), None)
    return df[CLEAN_SCHEMA.names]


def clean_veremi_split(vdir, scenario, variant, split, dest, tmpdir, chunk_rows, log) -> dict:
    stem = archive_stem(scenario, variant)
    member = f"{stem}/{split}/{stem}.zip"
    tmp = tmpdir / f"{stem}_{split}.zip"
    with zipfile.ZipFile(vdir / f"{stem}.zip") as outer, outer.open(member) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    ctx = f"{scenario}/{variant}/{split}"
    n_rows = n_files = 0
    nan = Counter()
    buf = []

    def flush(writer):
        nonlocal n_rows, buf
        df = normalize_chunk(buf, variant == "baseline", ctx)
        buf = []
        nan.update({k: int(v) for k, v in df.isna().sum().items() if v})
        n_rows += len(df)
        writer.write_table(pa.Table.from_pandas(df, schema=CLEAN_SCHEMA, preserve_index=False))

    try:
        with zipfile.ZipFile(tmp) as inner, pq.ParquetWriter(part, CLEAN_SCHEMA, compression="zstd") as writer:
            for name in inner.namelist():
                if not name.endswith(".json"):
                    continue
                buf.extend(flatten_clean(json.loads(inner.read(name)), Path(name).stem))
                n_files += 1
                if len(buf) >= chunk_rows:
                    flush(writer)
            if buf:
                flush(writer)
        part.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    log.info(f"clean | {ctx:45s} | {n_files:>6,} receiver files | {n_rows:>11,} rows -> {dest.relative_to(ROOT)}")
    return dict(scenario=scenario, variant=variant, split=split, receiver_files=n_files, rows=n_rows,
                nan_counts=json.dumps(dict(nan)), output=str(dest.relative_to(ROOT)))


def session_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sender_id")
    t = g.agg(n_rcv=("message_id", "size"), n_unique_msgs=("message_id", "nunique"),
              n_attack_rcv=("attacker", "sum"), first_ns=("send_ns", "min"), last_ns=("send_ns", "max"))
    t["compromised"] = (t.n_attack_rcv > 0).astype("int8")
    t["frac_attack"] = t.n_attack_rcv / t.n_rcv
    t["lifetime_s"] = (t.last_ns - t.first_ns) / NS_PER_S
    u = df.drop_duplicates(["sender_id", "message_id"])[["sender_id", "send_ns"]].sort_values(["sender_id", "send_ns"])
    u["gap_s"] = u.groupby("sender_id").send_ns.diff() / NS_PER_S
    t["cam_gap_median_s"] = u.groupby("sender_id").gap_s.median()
    return t.reset_index()


def veremi_diagnostics(cfg, sc, paths, vdir, log) -> list:
    V, P = cfg["veremi"], cfg["paths"]
    dev = [s for s in V["splits"] if s not in V["heldout_splits"]]
    cols = ["receiver_id", "sender_id", "message_id", "send_ns", "rcv_ns", "attacker", "s_x", "s_y", "s_spd"]

    def load(var, sp):
        d = pd.read_parquet(paths[(var, sp)], columns=cols)
        d["attacker"] = d.attacker.astype("int32")
        return d

    L = [f"## VeReMi {sc}", "", "### Sender-ID overlap across splits (leakage check)"]
    ids = {sp: set(pd.read_parquet(paths[("baseline", sp)], columns=["sender_id"]).sender_id) for sp in V["splits"]}
    for x, y in [("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")]:
        if x in ids and y in ids:
            L.append(f"- {x} ∩ {y}: {len(ids[x] & ids[y])} shared (|{x}|={len(ids[x])}, |{y}|={len(ids[y])})")

    sess = {}
    for var in V["variants"]:
        for sp in V["splits"]:
            t = session_table(load(var, sp))
            t.insert(0, "split", sp)
            t.insert(0, "variant", var)
            sess[(var, sp)] = t
    for root, sps in [(P["interim"], dev), (P["heldout"], V["heldout_splits"])]:
        if sps:
            tab = pd.concat([sess[(v, s)] for v in V["variants"] for s in sps], ignore_index=True)
            p = ROOT / root / "veremi" / sc / "session_labels.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            tab.to_parquet(p, index=False)

    L += ["", "### Session (sender) statistics: baseline, all splits (no labels)"]
    for sp in V["splits"]:
        t = sess[("baseline", sp)]
        L += [f"- {sp}: lifetime_s {_q(t.lifetime_s)}",
              f"  - unique msgs/sender {_q(t.n_unique_msgs)}",
              f"  - receptions/sender {_q(t.n_rcv)}",
              f"  - median CAM gap per sender (s) {_q(t.cam_gap_median_s)}"]

    atk = [v for v in V["variants"] if v != "baseline"]
    L += ["", "### Compromised-sender sets across attack variants (dev splits only)"]
    for sp in dev:
        S = {v: set(sess[(v, sp)].loc[sess[(v, sp)].compromised == 1, "sender_id"]) for v in atk}
        for i, x in enumerate(atk):
            for y in atk[i + 1:]:
                L.append(f"- {sp}: {x} vs {y}: |A|={len(S[x])} |B|={len(S[y])} "
                         f"Jaccard={len(S[x] & S[y]) / max(len(S[x] | S[y]), 1):.3f}")

    for sp in dev:
        base = load("baseline", sp)
        if "timeDelayAttack" in atk:
            m = load("timeDelayAttack", sp).merge(base, on=JOIN_KEY, suffixes=("", "_b"))
            a, ben = m[m.attacker == 1], m[m.attacker == 0]
            same = np.isclose(a.s_x, a.s_x_b) & np.isclose(a.s_y, a.s_y_b) & np.isclose(a.s_spd, a.s_spd_b)
            L += ["", f"### timeDelayAttack vs baseline ({sp}): {len(m)} matched receptions, {len(a)} attack",
                  f"- Δrcv (s) {_q((a.rcv_ns - a.rcv_ns_b) / NS_PER_S)}",
                  f"- Δsend (s) {_q((a.send_ns - a.send_ns_b) / NS_PER_S)}",
                  f"- Δlatency (ms) {_q(((a.rcv_ns - a.send_ns) - (a.rcv_ns_b - a.send_ns_b)) / 1e6)}",
                  f"- attack receptions with unchanged content (pos+spd): "
                  f"{(float(np.mean(same)) if len(a) else float('nan')):.3f}",
                  f"- benign receptions with any timestamp change: "
                  f"{float(((ben.rcv_ns != ben.rcv_ns_b) | (ben.send_ns != ben.send_ns_b)).mean()):.4f}"]
        if "dataReplay" in atk:
            m = load("dataReplay", sp).merge(base, on=JOIN_KEY, suffixes=("", "_b"))
            chg = ~(np.isclose(m.s_x, m.s_x_b) & np.isclose(m.s_y, m.s_y_b))
            att = (m.attacker == 1).to_numpy()
            t = sess[("dataReplay", sp)]
            mixed = t[(t.n_attack_rcv > 0) & (t.n_attack_rcv < t.n_rcv)]
            L += ["", f"### dataReplay vs baseline ({sp})",
                  f"- attack receptions with changed position: {(float(chg[att].mean()) if att.any() else float('nan')):.3f}",
                  f"- benign receptions with changed position: {float(chg[~att].mean()):.4f}",
                  f"- compromised senders with mixed labels: {len(mixed)} of {int(t.compromised.sum())}; "
                  f"their attack fraction {_q(mixed.frac_attack)}"]
        if "dosAttack" in atk:
            d = load("dosAttack", sp)
            j = sess[("dosAttack", sp)].merge(sess[("baseline", sp)][["sender_id", "n_unique_msgs", "n_rcv"]], on="sender_id", how="left", suffixes=("", "_b")).set_index("sender_id")
            c, b = j[j.compromised == 1], j[j.compromised == 0]
            dup = d.duplicated(JOIN_KEY, keep=False)
            L += ["", f"### dosAttack vs baseline ({sp})",
                  f"- unique-message ratio dos/baseline, compromised senders {_q(c.n_unique_msgs / c.n_unique_msgs_b)}",
                  f"- unique-message ratio dos/baseline, benign senders {_q(b.n_unique_msgs / b.n_unique_msgs_b)}",
                  f"- receptions sharing (receiver, sender, message_id): {int(dup.sum())}; attack-labelled among them: "
                  f"{(float(d.attacker[dup].mean()) if dup.any() else float('nan')):.3f}"]

    for sp in V["heldout_splits"]:
        fname = V["ground_truth_files"][sp].format(scenario=sc)
        g = normalize_chunk(flatten_clean(json.loads((vdir / fname).read_bytes()), None), True, f"GT {fname}")
        g = g.drop_duplicates(["sender_id", "message_id"])
        b = (pd.read_parquet(paths[("baseline", sp)],
                             columns=["sender_id", "message_id", "s_x", "s_y", "s_spd", "s_pos_nx", "s_pos_ny"])
             .drop_duplicates(["sender_id", "message_id"]))
        m = g.merge(b, on=["sender_id", "message_id"], suffixes=("_gt", ""))
        L += ["", f"### Ground-truth file vs baseline ({sp}; integrity check only, no labels)",
              f"- matched messages: {len(m)} of GT {len(g)}",
              f"- |Δx| (m) {_q((m.s_x_gt - m.s_x).abs())}",
              f"- |Δy| (m) {_q((m.s_y_gt - m.s_y).abs())}",
              f"- |Δspd| (m/s) {_q((m.s_spd_gt - m.s_spd).abs())}",
              f"- |gt_x - (x - noise_x)| {_q((m.s_x_gt - (m.s_x - m.s_pos_nx)).abs())}",
              f"- |gt_x - (x + noise_x)| {_q((m.s_x_gt - (m.s_x + m.s_pos_nx)).abs())}",
              f"- GT noise_x equals baseline noise_x: {float(np.isclose(m.s_pos_nx_gt, m.s_pos_nx).mean()):.3f}"]
    log.info(f"clean | diagnostics done for {sc}")
    return L + [""]


HETNET_VARIANTS = ("dataset", "gate_typical", "gate_fast")


def hetnet_variant_roots(cfg, variant="dataset", trace_root=None) -> dict:
    """A15: locations of a HetNet variant's long table and traces. 'dataset' without trace_root = frozen layout."""
    from pathlib import Path
    if variant not in HETNET_VARIANTS:
        raise SystemExit(f"unknown hetnet variant {variant}")
    P = cfg["paths"]
    name = "hetnet_long.parquet" if variant == "dataset" else f"hetnet_long_{variant}.parquet"
    long = ROOT / P["interim"] / "hetnet" / name
    if variant == "dataset" and trace_root is None:
        return dict(name=variant, long=long, processed=ROOT / P["processed"], heldout=ROOT / P["heldout"],
                    report=ROOT / P["docs"] / "traces_report.md", manifest=ROOT / P["results"] / "traces")
    r = Path(trace_root) if trace_root else ROOT / cfg["hetnet"]["gate"]["trace_root"] / variant
    report = r / "traces_report.md" if trace_root else ROOT / P["docs"] / f"traces_report_{variant}.md"
    return dict(name=variant, long=long, processed=r / "processed", heldout=r / "heldout",
                report=report, manifest=r / "manifest")


def _hetnet_frames(cfg, variant="dataset"):
    """Yield (ue, gnb, raw trace) in the same string-sorted 'ue/gnb' order for every variant."""
    from pathlib import Path
    H = cfg["hetnet"]
    if variant == "dataset":
        with zipfile.ZipFile(ROOT / cfg["paths"]["hetnet_raw"] / H["archive"]) as z:
            for n in sorted(z.namelist()):
                if not n.endswith("traces.csv"):
                    continue
                ue, gnb = (int(x) for x in n.split("/")[:2])
                yield ue, gnb, pd.read_csv(io.BytesIO(z.read(n)))
        return
    G = H["gate"]
    src = Path(G["source_root"]) / variant.split("_", 1)[1]
    nrow = int(G["rows"][variant])
    want = {f"{u}/{g}" for u in range(H["n_ues"]) for g in range(H["n_gnbs"])}
    have = {f"{p.parent.parent.name}/{p.parent.name}" for p in src.glob("*/*/traces.csv")}
    if have != want:
        raise SystemExit(f"{src}: trace set mismatch (missing {sorted(want - have)[:5]}, extra {sorted(have - want)[:5]})")
    for n in sorted(f"{k}/traces.csv" for k in want):
        d = pd.read_csv(src / n)
        if len(d) != nrow:
            raise SystemExit(f"{src / n}: {len(d)} rows, expected {nrow}")
        ue, gnb = (int(x) for x in n.split("/")[:2])
        yield ue, gnb, d


def clean_hetnet(cfg, log, variant="dataset", out_path=None) -> list:
    H = cfg["hetnet"]
    dt = H["dt_s"]
    parts = []
    for ue, gnb, d in _hetnet_frames(cfg, variant):
        xyz = d["UE Position"].str.split(":", expand=True).astype(float)
        r = d.Rsrp.to_numpy(float)
        ooc = ~np.isfinite(r)
        inc = lambda c: d[c].diff().fillna(d[c].iloc[0])
        parts.append(pd.DataFrame({
            "ue": ue, "gnb": gnb, "t_s": d.Time.round(3),
            "rsrp_dbm": np.where(ooc, np.nan, r), "out_of_coverage": ooc,
            "x": xyz[0], "y": xyz[1], "dist_m": d.Distance,
            "thr_rx_mbps": inc("RxBytes") * 8 / dt / 1e6,
            "loss_frac_interval": inc("LostPackets") / inc("TxPackets").clip(lower=1),
            "lat_interval_s": d.LatencySum, "lat_last_s": d.LatencyLast, "jitter_interval_s": d.JitterSum}))
    df = pd.concat(parts, ignore_index=True)
    df["ue"] = df.ue.astype("int16")
    df["gnb"] = df.gnb.astype("int8")
    p = out_path if out_path is not None else hetnet_variant_roots(cfg, variant)["long"]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
    macro = df.gnb.isin(H["macro_gnbs"]).to_numpy()
    log.info(f"clean | HetNet ({variant}): {len(df):,} rows -> {rel}")
    return ["## HetNet clean" + ("" if variant == "dataset" else f" ({variant})"), f"- rows={len(df):,} -> {rel}",
            f"- out-of-coverage samples (RSRP=-inf -> NaN, flagged): {int(df.out_of_coverage.sum())}",
            f"- loss_frac_interval > 1: {int((df.loss_frac_interval > 1).sum())}; < 0: {int((df.loss_frac_interval < 0).sum())}",
            f"- RX throughput (Mbps), micro gNBs {_q(df.thr_rx_mbps[~macro])}",
            f"- RX throughput (Mbps), macro gNBs {_q(df.thr_rx_mbps[macro])}",
            "- NOTE: LatencySum/JitterSum are NOT cumulative (README discrepancy, all 160 files); "
            "stored as per-interval values lat_interval_s / jitter_interval_s.",
            "- NOTE: each file is a saturated 380 Mbps flow to ONE gNB (counterfactual link), not a real attachment.", ""]


def run_clean(cfg, scenarios, a, log):
    V, P = cfg["veremi"], cfg["paths"]
    out = ROOT / P["results"] / "clean"
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# Clean-stage report (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
         "Train/Validation -> data/interim; Test -> data/heldout (to be read only by evaluate.py --final).", ""]
    files = []
    if not a.skip_veremi:
        vdir = ROOT / P["veremi_raw"]
        (ROOT / P["interim"]).mkdir(parents=True, exist_ok=True)
        tmpdir = Path(tempfile.mkdtemp(prefix="veremi_tmp_", dir=ROOT / P["interim"]))
        try:
            for sc in scenarios:
                paths = {}
                for var in V["variants"]:
                    for sp in V["splits"]:
                        root = P["heldout"] if sp in V["heldout_splits"] else P["interim"]
                        dest = ROOT / root / "veremi" / sc / f"{var}_{sp}.parquet"
                        paths[(var, sp)] = dest
                        if dest.exists() and not a.force:
                            log.info(f"clean | skip existing {dest.relative_to(ROOT)} (use --force to rebuild)")
                            continue
                        files.append(clean_veremi_split(vdir, sc, var, sp, dest, tmpdir,
                                                        cfg["clean"]["chunk_rows"], log))
                L += veremi_diagnostics(cfg, sc, paths, vdir, log)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if files:
            ft = pd.DataFrame(files)
            ft.to_csv(out / "veremi_clean_files.csv", index=False)
            L += ["## VeReMi files written", "```", ft.drop(columns=["output"]).to_string(index=False), "```", ""]
    if not a.skip_hetnet:
        L += clean_hetnet(cfg, log)
    report = ROOT / P["docs"] / "clean_report.md"
    report.write_text("\n".join(L))
    write_manifest(out, cfg, "preprocess.py", {"stage": "clean", "scenarios": scenarios, "force": a.force})
    log.info(f"Done. Report: {report.relative_to(ROOT)}")


# ============================================================================ traces
EVID_FEATURES = ["is_first", "dt_send_s", "dt_rcv_s", "dup", "msgid_back", "send_back",
                 "implied_spd", "spd_resid", "dist_sr_m", "rate_1s"]
EV_COLS = (["sender_id", "receiver_id", "message_id", "send_ns", "rcv_ns", "report_ns", "edge"]
           + EVID_FEATURES + ["gt_msg_attack", "gt_x"])
HO_COLS = ["sender_id", "ho_idx", "t_exec_ns", "prep_start_ns", "t_ho_s", "src", "dst", "pingpong", "dwell_s"]


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def load_hetnet_paths(cfg, long_path=None) -> dict:
    """Per UE path: arc length s (m), per-gNB RSRP along s (floored), total length L."""
    floor = float(cfg["traces"]["rsrp_floor_dbm"])
    n_g = cfg["hetnet"]["n_gnbs"]
    df = pd.read_parquet(long_path if long_path is not None else ROOT / cfg["paths"]["interim"] / "hetnet" / "hetnet_long.parquet",
                         columns=["ue", "gnb", "t_s", "rsrp_dbm", "x", "y"])
    out = {}
    for ue, d in df.groupby("ue"):
        piv = d.pivot(index="t_s", columns="gnb", values="rsrp_dbm").sort_index().reindex(columns=range(n_g))
        xy = d[d.gnb == 0].set_index("t_s").sort_index().loc[piv.index, ["x", "y"]].to_numpy(float)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        rsrp = np.maximum(np.nan_to_num(piv.to_numpy(float), nan=floor), floor)
        out[int(ue)] = {"s": s[keep], "rsrp": rsrp[keep], "L": float(s[keep][-1])}
    return out


def session_mobility(t_msg_s, spd, lifetime_s, path, offset_m, dt, vmax):
    """Speed-driven spatial resampling with reflecting traversal (continuous position and RSRP)."""
    grid = np.arange(0.0, lifetime_s + dt / 2, dt)
    v = np.clip(np.interp(grid, t_msg_s, spd), 0.0, vmax)
    d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    L = path["L"]
    m = np.mod(offset_m + d, 2 * L)
    p = np.where(m <= L, m, 2 * L - m)
    rsrp = np.column_stack([np.interp(p, path["s"], path["rsrp"][:, g]) for g in range(path["rsrp"].shape[1])])
    return grid, rsrp, float(d[-1]), int((offset_m + d[-1]) // L)


def session_handovers(rsrp, grid, first_ns, last_ns, dt, H) -> list:
    evs = a3_cho(rsrp, grid, dt, H["hysteresis_db"], H["ttt_s"], H["prep_offset_db"])
    rows = []
    for i, e in enumerate(evs):
        t_exec = int(first_ns + round(e["t_s"] * NS_PER_S))
        tho = e["prep_window_s"] if np.isfinite(e["prep_window_s"]) else H["ttt_s"]
        prev = evs[i - 1] if i else None
        rows.append(dict(ho_idx=i, t_exec_ns=t_exec, prep_start_ns=int(t_exec - round(tho * NS_PER_S)),
                         t_ho_s=float(tho), src=int(e["src"]), dst=int(e["dst"]),
                         pingpong=bool(prev is not None and e["dst"] == prev["src"]
                                       and e["t_s"] - prev["t_s"] <= H["pingpong_window_s"])))
    for i, r in enumerate(rows):
        end = rows[i + 1]["t_exec_ns"] if i + 1 < len(rows) else last_ns
        r["dwell_s"] = max(0.0, (end - r["t_exec_ns"]) / NS_PER_S)
    return rows


def evidence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal per-reception features: only the current and earlier receptions of the same (receiver, sender)."""
    df = df.sort_values(["receiver_id", "sender_id", "rcv_ns", "message_id"], kind="mergesort").reset_index(drop=True)
    code = df.groupby(["receiver_id", "sender_id"], sort=False).ngroup().to_numpy(np.int64)
    g = df.groupby(code, sort=False)
    prev = {c: g[c].shift(1) for c in ("send_ns", "rcv_ns", "s_x", "s_y", "s_spd")}
    dts = (df.send_ns - prev["send_ns"]) / NS_PER_S
    dist = np.hypot(df.s_x - prev["s_x"], df.s_y - prev["s_y"])
    implied = dist / dts.where(dts > 0)
    prev_max_id = g.message_id.cummax().groupby(code).shift(1)
    prev_max_send = g.send_ns.cummax().groupby(code).shift(1)
    out = df[["sender_id", "receiver_id", "message_id", "send_ns", "rcv_ns", "attacker"]].copy()
    out["is_first"] = prev["send_ns"].isna().astype("int8")
    out["dt_send_s"] = dts
    out["dt_rcv_s"] = (df.rcv_ns - prev["rcv_ns"]) / NS_PER_S
    out["dup"] = (df.groupby(["receiver_id", "sender_id", "message_id"], sort=False).cumcount() > 0).astype("int8")
    out["msgid_back"] = (df.message_id < prev_max_id).astype("int8")
    out["send_back"] = (df.send_ns < prev_max_send).astype("int8")
    out["implied_spd"] = implied
    out["spd_resid"] = (implied - 0.5 * (df.s_spd + prev["s_spd"])).abs()
    out["dist_sr_m"] = np.hypot(df.s_x - df.r_x, df.s_y - df.r_y)
    if len(df):
        # millisecond key: safe for <9e8 receiver-sender pairs and pair spans <115 days
        local_ms = (df.rcv_ns.to_numpy(np.int64)
                    - df.groupby(code).rcv_ns.transform("min").to_numpy(np.int64)) // 10**6
        assert code.max() < 9 * 10**8 and local_ms.max() < 10**10, "rate key overflow"
        key = code * 10**10 + local_ms
        out["rate_1s"] = np.arange(len(key)) - np.searchsorted(key, key - 1000, side="right") + 1
    else:
        out["rate_1s"] = pd.Series(dtype="int64")
    return out


def pathfold_pools(cfg, k: int) -> dict:
    """A16: fold k -> Test = G_k, Validation = G_(k-1 mod 5), Train = remaining groups (ascending)."""
    G = [[int(u) for u in g] for g in cfg["pathfold"]["groups"]]
    n = len(G)
    if not 0 <= k < n:
        raise SystemExit(f"pathfold: fold {k} out of range 0..{n - 1}")
    prim = cfg["traces"]["path_pool"]
    if sorted(u for g in G for u in g) != sorted(int(u) for v in prim.values() for u in v):
        raise SystemExit("pathfold: groups must partition exactly the paths of traces.path_pool")
    v = (k - 1) % n
    pools = {"Train": sorted(u for i, g in enumerate(G) if i not in (k, v) for u in g), "Validation": G[v], "Test": G[k]}
    if set(pools) != set(prim):
        raise SystemExit(f"pathfold: split names {sorted(pools)} != path_pool keys {sorted(prim)}")
    return pools


def pathfold_roots(cfg, k: int, HV: dict) -> dict:
    """A16: rotated traces are written under pathfold.root/fold{k}; the frozen layout is never written."""
    if HV["name"] != "dataset" or HV["processed"] != ROOT / cfg["paths"]["processed"]:
        raise SystemExit("--pathfold requires --hetnet-variant dataset and no --trace-root")
    r = ROOT / cfg["pathfold"]["root"] / f"fold{k}"
    return dict(HV, name=f"dataset (A16 path-fold {k})", processed=r / "processed", heldout=r / "heldout",
                report=r / "traces_report.md", manifest=r / "manifest")


def run_traces(cfg, scenarios, a, log):
    V, P, T = cfg["veremi"], cfg["paths"], cfg["traces"]
    H, dt, vmax = T["handover"], cfg["hetnet"]["dt_s"], float(T["max_speed_mps"])
    HV = hetnet_variant_roots(cfg, getattr(a, "hetnet_variant", "dataset"), getattr(a, "trace_root", None))
    PF = getattr(a, "pathfold", None)
    if PF is not None:
        HV = pathfold_roots(cfg, PF, HV)
    paths = load_hetnet_paths(cfg, HV["long"])
    log.info("traces | HetNet path lengths: " + ", ".join(f"{u}:{p['L']:.0f}m" for u, p in sorted(paths.items())))
    out_root = HV["manifest"]
    out_root.mkdir(parents=True, exist_ok=True)
    R = [f"# Traces-stage report (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
         "Mobility: speed-driven spatial resampling of HetNet paths (reflecting traversal), driven by BASELINE speeds",
         "so all variants share identical handover timelines. Transfer faults are NOT baked in (applied in model.py).",
         "Label-dependent statistics: development splits only. Held-out split: label-free counts only.",
         f"HetNet variant: {HV['name']} ({HV['long'].name})"] + ([f"Path-fold rotation (A16): fold {PF}, pools {pathfold_pools(cfg, PF)}"] if PF is not None else []) + [""]
    for sc in scenarios:
        src = lambda sp: ROOT / (P["heldout"] if sp in V["heldout_splits"] else P["interim"]) / "veremi" / sc
        ids = {sp: set(pd.read_parquet(src(sp) / f"baseline_{sp}.parquet", columns=["sender_id"]).sender_id)
               for sp in V["splits"]}
        for sp in V["splits"]:
            held = sp in V["heldout_splits"]
            out = (HV["heldout"] if held else HV["processed"]) / sc / sp
            expected = [out / "handovers.parquet", out / "mapping.parquet"] + \
                       [out / f"{k}_{v}.parquet" for k in ("sessions", "evidence") for v in V["variants"]]
            if all(p.exists() for p in expected) and not a.force:
                log.info(f"traces | skip existing {out} (use --force)")
                R += [f"## {sc} / {sp}: skipped (exists)", ""]
                continue
            out.mkdir(parents=True, exist_ok=True)

            # ---- sessions + mobility + handovers (variant-independent)
            base = pd.read_parquet(src(sp) / f"baseline_{sp}.parquet", columns=["sender_id", "message_id", "send_ns", "s_spd"])
            u = base.drop_duplicates(["sender_id", "message_id"]).sort_values(["sender_id", "send_ns"])
            span = u.groupby("sender_id").send_ns.agg(first_ns="min", last_ns="max")
            span["lifetime_s"] = (span.last_ns - span.first_ns) / NS_PER_S
            short = span.lifetime_s < T["min_lifetime_s"]
            overlap = pd.Series(False, index=span.index)
            for other in (T.get("drop_overlap") or {}).get(sp, []):
                overlap |= span.index.isin(ids[other])
            n_all, n_short, n_ovl = len(span), int(short.sum()), int((overlap & ~short).sum())
            span = span[~short & ~overlap]
            n_clip = int((u[u.sender_id.isin(span.index)].s_spd > vmax).sum())
            pool = (pathfold_pools(cfg, PF) if PF is not None else T["path_pool"])[sp]
            map_rows, ho_rows = [], []
            for sid, g in u[u.sender_id.isin(span.index)].groupby("sender_id"):
                f_ns, l_ns = int(span.at[sid, "first_ns"]), int(span.at[sid, "last_ns"])
                life = float(span.at[sid, "lifetime_s"])
                rng = np.random.default_rng(stable_seed(cfg["seed"], sc, sp, sid))
                ue = int(rng.choice(pool))
                off = float(rng.uniform(0.0, paths[ue]["L"]))
                t_msg = (g.send_ns.to_numpy(np.int64) - f_ns) / NS_PER_S
                grid, rsrp, dist, nref = session_mobility(t_msg, g.s_spd.to_numpy(float), life, paths[ue], off, dt, vmax)
                hos = session_handovers(rsrp, grid, f_ns, l_ns, dt, H)
                for r in hos:
                    r["sender_id"] = sid
                ho_rows += hos
                map_rows.append(dict(sender_id=sid, path_ue=ue, offset_m=off, dist_m=dist, n_reflect=nref,
                                     init_edge=int(np.argmax(rsrp[0])), n_handovers=len(hos),
                                     first_ns=f_ns, last_ns=l_ns, lifetime_s=life))
            mp = pd.DataFrame(map_rows)
            ho = pd.DataFrame(ho_rows, columns=HO_COLS)
            mp.to_parquet(out / "mapping.parquet", index=False)
            ho.to_parquet(out / "handovers.parquet", index=False)
            lookup = {r["sender_id"]: (np.array([], np.int64), [r["init_edge"]]) for r in map_rows}
            for sid, g in ho.sort_values(["sender_id", "ho_idx"]).groupby("sender_id"):
                lookup[sid] = (g.t_exec_ns.to_numpy(np.int64), [lookup[sid][1][0]] + g.dst.astype(int).tolist())

            R += [f"## {sc} / {sp}{'  (HELD-OUT: label-free counts only)' if held else ''}",
                  f"- sessions: {n_all} baseline senders -> kept {len(mp)} "
                  f"(lifetime < {T['min_lifetime_s']} s: {n_short}; cross-split overlap dropped: {n_ovl})",
                  f"- baseline messages with speed > {vmax:g} m/s (clipped for mobility): {n_clip}",
                  f"- mean session speed (m/s) {_q(mp.dist_m / mp.lifetime_s)}",
                  f"- sessions that reflected at a path end: {float((mp.n_reflect > 0).mean()):.3f}",
                  f"- handovers: total {len(ho)}; per session {_q(mp.n_handovers)}; "
                  f"sessions with 0 HO: {float((mp.n_handovers == 0).mean()):.3f}",
                  f"- T_HO (s) {_q(ho.t_ho_s)}",
                  f"- dwell after HO (s) {_q(ho.dwell_s)}",
                  f"- ping-pong fraction: {(float(ho.pingpong.mean()) if len(ho) else float('nan')):.3f}",
                  f"- top HO pairs (src->dst): {ho.groupby(['src', 'dst']).size().nlargest(8).to_dict() if len(ho) else {}}"]

            # ---- per variant: evidence + labels
            delay_ns = int(round(T["report_delay_s"] * NS_PER_S))
            feat_rows = {}
            for var in V["variants"]:
                df = pd.read_parquet(src(sp) / f"{var}_{sp}.parquet")
                df = df[df.sender_id.isin(mp.sender_id)]
                ev = evidence_features(df)
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
                ev = ev[EV_COLS]
                ev.to_parquet(out / f"evidence_{var}.parquet", index=False)

                sess["n_evidence"] = ev.groupby("sender_id").size().reindex(sess.index).fillna(0).astype(int)
                sess = sess.reset_index()
                sess.insert(0, "session_uid", f"{sc}|{var}|{sp}|" + sess.sender_id)
                sess.insert(1, "scenario", sc)
                sess.insert(2, "variant", var)
                sess.insert(3, "split", sp)
                sess.to_parquet(out / f"sessions_{var}.parquet", index=False)

                if held:
                    R.append(f"- {var}: evidence rows {len(ev):,}")
                else:
                    if len(ho):
                        m = ho[["sender_id", "ho_idx", "src", "t_exec_ns"]].merge(
                            ev[["sender_id", "edge", "report_ns"]], left_on=["sender_id", "src"],
                            right_on=["sender_id", "edge"], how="left")
                        cnt = (m.report_ns < m.t_exec_ns).groupby([m.sender_id, m.ho_idx]).sum()
                        cov = f"{float((cnt > 0).mean()):.3f}; src items per HO {_q(cnt)}"
                    else:
                        cov = "n/a"
                    R.append(f"- {var}: evidence rows {len(ev):,}; compromised sessions "
                             f"{int(sess.compromised.sum())}/{len(sess)}; HOs with >=1 item at src: {cov}")
                    if sp == T["label_stats_split"] and var != "baseline":
                        for lab, d in ev.groupby("gt_msg_attack"):
                            feat_rows[f"{var} | attack={lab}"] = {
                                "n": len(d), "is_first": d.is_first.mean(), "dup": d.dup.mean(),
                                "msgid_back": d.msgid_back.mean(), "send_back": d.send_back.mean(),
                                "spd_resid_med": d.spd_resid.median(), "spd_resid_p90": d.spd_resid.quantile(0.9),
                                "dist_sr_med": d.dist_sr_m.median(), "dist_sr_p90": d.dist_sr_m.quantile(0.9),
                                "rate_1s_mean": d.rate_1s.mean(), "dt_send_med": d.dt_send_s.median()}
                log.info(f"traces | {sc} | {sp:10s} | {var:15s} | {len(sess):>5} sessions | {len(ev):>9,} evidence")
            if feat_rows:
                R += ["", f"### Feature separability by message label ({sp} only; sanity check, not a result)", "```",
                      pd.DataFrame(feat_rows).T.to_string(float_format=FMT), "```"]
            R.append("")
    report = HV["report"]
    report.write_text("\n".join(R))
    write_manifest(out_root, cfg, "preprocess.py", {"stage": "traces", "scenarios": scenarios, "force": a.force,
                                                    "hetnet_variant": HV["name"], **({"pathfold": PF} if PF is not None else {})})
    log.info(f"Done. Report: {report}")


# ============================================================================ summary
def write_summary(cfg, vprof, gtov, schema, hfile, hue, hev) -> Path:
    L = [f"# Data profile summary (auto-generated {datetime.now():%Y-%m-%d %H:%M})", ""]
    if vprof is not None:
        cols = ["variant", "split", "n_receiver_files", "n_msgs", "n_senders", "duration_s", "t_rcv_min_s",
                "lat_median_ms", "n_dup_receiver_messageID", "frac_has_attacker_field", "frac_attack_msgs",
                "n_attacker_senders", "frac_senders_label_constant", "lat_median_ms_attack",
                "lat_median_ms_benign", "aliases_per_sender_mean", "sender_spd_median"]
        cols = [c for c in cols if c in vprof.columns]
        for sc, d in vprof.groupby("scenario"):
            L += [f"## VeReMi: {sc}", "```", d[cols].to_string(index=False, float_format=FMT), "```", ""]
        L += ["## VeReMi ground-truth overlap (key = sender_id + messageID)", "```",
              gtov.drop(columns=["gt_file"]).to_string(index=False, float_format=FMT), "```", ""]
        L += ["## VeReMi raw JSON types (first message per variant)", "```json",
              json.dumps(schema, indent=1), "```", ""]
    if hfile is not None:
        nm = Counter(c for s in hfile.nonmonotonic for c in str(s).split(",") if c)
        dtv = sorted({x for s in hfile.dt_values for x in str(s).split(",") if x})
        L += ["## HetNet: file level", "```",
              f"files={len(hfile)}  rows={sorted(hfile.rows.unique().tolist())}  NaN total={int(hfile.n_nan.sum())}",
              f"time range=[{hfile.t_min.min()}, {hfile.t_max.max()}]  dt values={dtv}",
              f"files with non-monotonic counters={int((~hfile.all_counters_monotonic).sum())}  by counter={dict(nm)}",
              f"RSRP dBm: min={hfile.rsrp_min.min():.1f}  median-of-medians={hfile.rsrp_median.median():.1f}  "
              f"max={hfile.rsrp_max.max():.1f}  non-finite={int(hfile.n_rsrp_nonfinite.sum())}",
              f"LatencySum median (s)={hfile.latsum_median.median():.4g}  "
              f"LatencyLast median (s)={hfile.latlast_median.median():.4g}",
              f"final loss fraction: median={hfile.final_loss_frac.median():.4g}  max={hfile.final_loss_frac.max():.4g}",
              "```", ""]
        L += ["## HetNet: per UE (profiling handover heuristic, NOT frozen)", "```",
              hue.to_string(index=False, float_format=FMT), "```", ""]
        if len(hev):
            pw = hev.prep_window_s.dropna()
            L += [f"Total A3 handovers: {len(hev)}; ping-pong: {int(hev.pingpong.sum())}; "
                  f"prep window (s): p10={pw.quantile(0.1):.3g}  median={pw.median():.3g}  p90={pw.quantile(0.9):.3g}",
                  f"Handover pairs (src->dst) top 10: {hev.groupby(['src','dst']).size().nlargest(10).to_dict()}", ""]
    if vprof is not None and hue is not None:
        L += ["## Mobility comparison",
              f"VeReMi sender speed median (m/s) by scenario: "
              f"{vprof.groupby('scenario').sender_spd_median.median().round(2).to_dict()}",
              f"HetNet UE speed mean (m/s): {hue.spd_mean.mean():.2f}", ""]
    path = ROOT / cfg["paths"]["docs"] / "profile_summary.md"
    path.write_text("\n".join(L))
    return path


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["profile", "clean", "traces", "hetnet"])
    ap.add_argument("--scenarios", nargs="+", default=None, help="VeReMi scenarios (default: dev_scenario)")
    ap.add_argument("--skip-veremi", action="store_true")
    ap.add_argument("--skip-hetnet", action="store_true")
    ap.add_argument("--force", action="store_true", help="clean: overwrite existing parquet outputs")
    ap.add_argument("--hetnet-variant", choices=HETNET_VARIANTS, default="dataset",
                    help="A15: HetNet source for --stage hetnet/traces (dataset = frozen)")
    ap.add_argument("--trace-root", default=None, help="smoke tests only: write traces under this root")
    ap.add_argument("--pathfold", type=int, choices=range(5), default=None,
                    help="A16 path-fold rotation: fold k, written under pathfold.root/fold k")
    a = ap.parse_args()

    cfg = load_config()
    log = get_logger(f"preprocess_{a.stage}")
    set_seed(cfg["seed"])


    scenarios = a.scenarios or [cfg["veremi"]["dev_scenario"]]
    if scenarios == ["all"]:
        scenarios = list(cfg["veremi"]["scenarios"])
    unknown = set(scenarios) - set(cfg["veremi"]["scenarios"])
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {sorted(unknown)}")
    if a.stage == "hetnet":
        print("\n".join(clean_hetnet(cfg, log, variant=a.hetnet_variant)))
        return
    if a.stage == "clean":
        run_clean(cfg, scenarios, a, log)
        return
    if a.stage == "traces":
        run_traces(cfg, scenarios, a, log)
        return
    out = ROOT / cfg["paths"]["results"] / "profile"
    out.mkdir(parents=True, exist_ok=True)

    vprof = gtov = schema = hfile = hue = hev = None
    if not a.skip_veremi:
        vprof, gtov, schema = profile_veremi(cfg, scenarios, out, log)
    if not a.skip_hetnet:
        hfile, hue, hev = profile_hetnet(cfg, out, log)
    summary = write_summary(cfg, vprof, gtov, schema, hfile, hue, hev)
    write_manifest(out, cfg, "preprocess.py", {"stage": "profile", "scenarios": scenarios,
                                               "skip_veremi": a.skip_veremi, "skip_hetnet": a.skip_hetnet})
    log.info(f"Done. Summary: {summary.relative_to(ROOT)} | tables + manifest: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
