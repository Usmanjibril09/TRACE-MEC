#!/usr/bin/env python
"""TRACE-MEC train.py. Stage 'lstm' = baseline #10 (per-sender LSTM item detector); run with .venv_torch.
Train labels only. Train scores are out-of-fold (3-fold GroupKFold by sender, as the item model);
Validation scored by the final model; Test only via --score-split Test (called by evaluate.py --final)."""
import argparse, copy, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, yaml, torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["baseline", "timeDelayAttack", "dataReplay", "dosAttack"]
KEYS = ["receiver_id", "sender_id", "message_id", "rcv_ns"]
NUM = ["s_x", "s_y", "s_spd", "s_acl", "s_hed", "r_x", "r_y"]
GRID = [(10, 32), (10, 64), (20, 32), (20, 64)]          # (history K, hidden H), tuned on Train
TUNE_ROWS, FIT_ROWS, TUNE_EP, FIT_EP, BATCH = 300_000, 600_000, 3, 6, 2048
torch.set_num_threads(os.cpu_count() or 1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def first(*ps):
    for p in ps:
        if p.exists():
            return p
    raise FileNotFoundError(str(ps[0]))


def seq_table(sc, split, var):
    r = pd.read_parquet(first(ROOT / "data/interim/veremi" / sc / f"{var}_{split}.parquet",
                              ROOT / "data/heldout/veremi" / sc / f"{var}_{split}.parquet"),
                        columns=KEYS + ["send_ns"] + NUM)
    for c in NUM:
        r[c] = pd.to_numeric(r[c], errors="coerce").astype("float64")
    r = r.sort_values(["receiver_id", "sender_id", "rcv_ns"], kind="stable").drop_duplicates(KEYS).reset_index(drop=True)
    g = r.groupby(["receiver_id", "sender_id"], sort=False)
    f = lambda c: r[c].to_numpy(dtype="float64")
    dt = g["send_ns"].diff().to_numpy(dtype="float64") / 1e9
    d = np.hypot(g["s_x"].diff().to_numpy(dtype="float64"), g["s_y"].diff().to_numpy(dtype="float64"))
    v = np.abs(f("s_spd"))
    with np.errstate(invalid="ignore"):
        perr = d - v * np.where(dt > 0, dt, np.nan)
    X = np.column_stack([np.clip(dt, -5, 5), np.clip(d, 0, 1000), v, f("s_acl"),
                         np.clip(g["s_spd"].diff().to_numpy(dtype="float64"), -50, 50),
                         g["s_hed"].diff().to_numpy(dtype="float64"), np.clip(perr, -1000, 1000),
                         np.hypot(f("s_x") - f("r_x"), f("s_y") - f("r_y")),
                         np.clip((f("rcv_ns") - f("send_ns")) / 1e9, -5, 5)])
    return r[KEYS], X, g.cumcount().to_numpy()


def build(sc, split, labels):
    Xs, Ps, rows, off = [], [], [], 0
    for v in VARIANTS:
        k, X, pos = seq_table(sc, split, v)
        cols = (KEYS + ["gt_msg_attack"]) if labels else KEYS
        e = pd.read_parquet(first(ROOT / "data/processed" / sc / split / f"evidence_{v}.parquet",
                                  ROOT / "data/heldout" / sc / split / f"evidence_{v}.parquet"), columns=cols)
        m = e.merge(k.assign(_i=np.arange(len(k))), on=KEYS, how="left")
        miss = int(m["_i"].isna().sum())
        log(f"{sc}/{split}/{v}: {len(e):,} evidence items, {len(k):,} raw messages, unmatched {miss}")
        if miss > 0.01 * len(e):
            raise RuntimeError("more than 1% of evidence items not found in raw messages")
        m["_i"] = np.where(m["_i"].notna(), m["_i"].fillna(0) + off, -1).astype(np.int64)
        Xs.append(X); Ps.append(pos); rows.append(m.assign(variant=v)); off += len(X)
    X = np.vstack(Xs + [np.full((1, Xs[0].shape[1]), np.nan)])      # last row = empty-history sentinel
    pos = np.concatenate(Ps + [np.zeros(1, np.int64)])
    R = pd.concat(rows, ignore_index=True)
    R.loc[R["_i"] < 0, "_i"] = len(X) - 1
    return X, pos, R


def norm(X, mu, sd):
    return np.clip(np.nan_to_num((X - mu) / sd, nan=0.0), -10, 10).astype(np.float32)


def windows(Z, pos, idx, K):
    off = np.arange(K - 1, -1, -1)
    valid = pos[idx][:, None] >= off[None, :]
    W = Z[np.where(valid, idx[:, None] - off[None, :], idx[:, None])] * valid[..., None]
    return np.concatenate([W, valid[..., None].astype(np.float32)], axis=-1)


class Net(nn.Module):
    def __init__(self, F, H):
        super().__init__()
        self.rnn, self.out = nn.LSTM(F, H, batch_first=True), nn.Linear(H, 1)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.out(y[:, -1]).squeeze(-1)


@torch.no_grad()
def predict(net, Z, pos, idx, K):
    net.eval()
    out = np.empty(len(idx), np.float32)
    for b in range(0, len(idx), 8192):
        out[b:b + 8192] = net(torch.from_numpy(windows(Z, pos, idx[b:b + 8192], K))).numpy()
    return out


def fit(Z, pos, idx, y, grp, K, H, seed, epochs, max_rows):
    rs = np.random.default_rng(seed); torch.manual_seed(seed)
    if len(idx) > max_rows:
        s = rs.choice(len(idx), max_rows, replace=False); idx, y, grp = idx[s], y[s], grp[s]
    ug = np.unique(grp)
    va = np.isin(grp, rs.choice(ug, max(1, len(ug) // 10), replace=False))
    tr, vi = np.flatnonzero(~va), np.flatnonzero(va)
    net, lf = Net(Z.shape[1] + 1, H), nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = rs.permutation(tr)
        for b in range(0, len(perm), BATCH):
            j = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lf(net(torch.from_numpy(windows(Z, pos, idx[j], K))), torch.from_numpy(y[j].astype(np.float32)))
            loss.backward(); opt.step()
        vl = float(lf(torch.from_numpy(predict(net, Z, pos, idx[vi], K)), torch.from_numpy(y[vi].astype(np.float32))))
        log(f"  K={K} H={H} epoch {ep + 1}: inner-val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, state = vl, copy.deepcopy(net.state_dict())
        else:
            break
    net.load_state_dict(state)
    pm = float(np.clip(y[tr].mean(), 1e-6, 1 - 1e-6))
    return net, best, float(np.log(pm / (1 - pm)))


def write(out, R, llr, split):
    for v in VARIANTS:
        mk = (R["variant"] == v).to_numpy()
        R.loc[mk, KEYS].assign(llr_lstm=llr[mk]).drop_duplicates(KEYS).to_parquet(out / f"{v}_{split}.parquet", index=False)


def aucs(R, llr):
    res = {}
    for v in VARIANTS:
        mk = (R["variant"] == v).to_numpy(); yv = R.loc[mk, "gt_msg_attack"].to_numpy()
        if 0 < yv.sum() < len(yv):
            res[v] = round(float(roc_auc_score(yv, llr[mk])), 3)
    return res


def score(sc, split, out, P, labels):
    net = Net(len(P["mu"]) + 1, P["H"]); net.load_state_dict(P["state"])
    X, pos, R = build(sc, split, labels)
    llr = np.clip(predict(net, norm(X, P["mu"], P["sd"]), pos, R["_i"].to_numpy(), P["K"]) - P["base"], -P["clip"], P["clip"])
    write(out, R, llr, split)
    log(f"{sc}/{split}: scored {len(R):,} items" + (f"; item AUC {aucs(R, llr)}" if labels else " (label-free)"))


def stage_lstm(sc, seed):
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    clip = float(cfg["model"]["llr_clip"])
    out = ROOT / cfg["paths"]["results"] / "lstm" / sc
    out.mkdir(parents=True, exist_ok=True)
    X, pos, R = build(sc, "Train", True)
    with np.errstate(all="ignore"):
        mu = np.nan_to_num(np.nanmean(X[:-1], 0)); sd = np.nan_to_num(np.nanstd(X[:-1], 0), nan=1.0) + 1e-6
    Z = norm(X, mu, sd); del X
    idx, y = R["_i"].to_numpy(), R["gt_msg_attack"].to_numpy().astype(int)
    grp = R["sender_id"].astype(str).to_numpy()
    tune = {(K, H): fit(Z, pos, idx, y, grp, K, H, seed, TUNE_EP, TUNE_ROWS)[1] for K, H in GRID}
    K, H = min(tune, key=tune.get)
    log(f"grid inner-val loss (Train): { {f'K{k}_H{h}': round(v, 4) for (k, h), v in tune.items()} } -> K={K} H={H}")
    oof = np.zeros(len(idx))
    for f, (a, b) in enumerate(GroupKFold(n_splits=3).split(np.zeros((len(idx), 1)), groups=grp)):
        net, _, base = fit(Z, pos, idx[a], y[a], grp[a], K, H, seed + f, FIT_EP, FIT_ROWS)
        oof[b] = np.clip(predict(net, Z, pos, idx[b], K) - base, -clip, clip)
    write(out, R, oof, "Train")
    log(f"{sc}/Train: out-of-fold item AUC {aucs(R, oof)}")
    net, _, base = fit(Z, pos, idx, y, grp, K, H, seed, FIT_EP, FIT_ROWS)
    P = {"state": net.state_dict(), "K": K, "H": H, "mu": mu, "sd": sd, "base": base, "clip": clip}
    torch.save(P, out / "model.pt")
    score(sc, "Validation", out, P, True)
    (out / "meta.json").write_text(json.dumps({"scenario": sc, "seed": seed, "K": K, "H": H,
        "grid": {f"K{k}_H{h}": v for (k, h), v in tune.items()}, "torch": torch.__version__,
        "time": time.strftime("%Y-%m-%d %H:%M"), "runtime_s": round(time.time() - t0)}, indent=2))
    log(f"Done ({time.time() - t0:.0f} s): {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["lstm"])
    ap.add_argument("--scenario", default="InTAS_highway_2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-split", default=None, help="score a split with the saved model (Test: evaluate.py --final only)")
    a = ap.parse_args()
    if a.score_split:
        cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
        out = ROOT / cfg["paths"]["results"] / "lstm" / a.scenario
        score(a.scenario, a.score_split, out, torch.load(out / "model.pt", weights_only=False), a.score_split != "Test")
    else:
        stage_lstm(a.scenario, a.seed)


if __name__ == "__main__":
    main()
