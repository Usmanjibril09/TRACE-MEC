#!/usr/bin/env python3
"""Portable primary evaluation entry point; experimental code is unchanged."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def verify():
    expected = json.loads((ROOT / "SOURCE_SHA256.json").read_text())
    bad = [p for p, h in expected.items() if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit("Release checksum mismatch: " + ", ".join(bad))
    print("Release source checksums verified", flush=True)
    return expected

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["verify", "dryrun", "final", "seeds"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4])
    ap.add_argument("--torch-python", type=Path, default=ROOT / ".venv_torch/bin/python")
    a = ap.parse_args()
    hashes = verify()
    if a.mode == "verify": return
    sys.path.insert(0, str(ROOT / "src"))
    import evaluate as E
    # Source integrity replaces historical private Git commit comparisons only.
    # Model, scoring, data split restrictions, run locks and attempt logs remain intact.
    E.frozen_check = lambda self_committed: hashes
    E.TORCH_PY = a.torch_python.resolve()
    old_git = E.git
    def release_git(*args):
        if args and args[0] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, "public-release-1.0.0\n", "")
        return old_git(*args)
    E.git = release_git
    cfg, how = E.load_cfg()
    if a.mode == "dryrun": E.dryrun(cfg, E.SCEN, a.seed, how)
    elif a.mode == "final": E.final(cfg, a.seed, False, False)
    else: E.seeds_run(cfg, a.seeds)

if __name__ == "__main__": main()
