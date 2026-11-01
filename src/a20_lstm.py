"""a20_lstm.py - LSTM-Det Test scores for the A20 position attacks with the frozen model.pt (runs in .venv_torch).
Uses train.score() unchanged; VARIANTS is narrowed to one attack so no existing score file can be rewritten."""
import pathlib, sys
import numpy as np, pandas as pd, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train as T

sc, var = sys.argv[1], sys.argv[2]
lst = ROOT / "results" / "lstm" / sc
P = torch.load(lst / "model.pt", map_location="cpu")
if var == "baseline":
    tmp = pathlib.Path(sys.argv[3]); tmp.mkdir(parents=True, exist_ok=True)
    T.VARIANTS = ["baseline"]
    T.score(sc, "Test", tmp, P, False)
    a, b = pd.read_parquet(tmp / "baseline_Test.parquet"), pd.read_parquet(lst / "baseline_Test.parquet")
    m = a.merge(b, on=T.KEYS, how="outer", suffixes=("_new", "_old"), indicator=True)
    both = float((m["_merge"] == "both").mean())
    dmax = float(np.nanmax(np.abs(m["llr_lstm_new"] - m["llr_lstm_old"])))
    ok = len(a) == len(b) and both == 1.0 and dmax <= 1e-4
    print(f"LSTM_CHECK {sc}: rows new {len(a):,} stored {len(b):,} | keys matched {both:.4f} | max|diff| {dmax:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 3)
dest = lst / f"{var}_Test.parquet"
if dest.exists():
    print(f"STOP: {dest} already exists", flush=True); sys.exit(4)
T.VARIANTS = [var]
T.score(sc, "Test", lst, P, False)
print(f"LSTM_SCORED {sc}/{var}: {len(pd.read_parquet(dest)):,} items", flush=True)
