"""TRACE-MEC shared utilities: config, paths, logging, seeds, run manifests, safe patching."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return cfg


def get_logger(name: str) -> logging.Logger:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(logs / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def save_json(obj, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def write_manifest(out_dir: Path, cfg: dict, script: str, extra: dict | None = None) -> Path:
    """Record exactly which code/config/environment produced the outputs in out_dir."""
    pkgs = {}
    for k in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "pyyaml", "pyarrow"):
        try:
            pkgs[k] = md.version(k)
        except md.PackageNotFoundError:
            pkgs[k] = None
    m = {"script": script, "argv": sys.argv,
         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "git": git_info(), "config_sha256": cfg.get("_config_sha256"), "seed": cfg.get("seed"),
         "python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}
    if extra:
        m.update(extra)
    return save_json(m, Path(out_dir) / "manifest.json")


def patch_file(path: Path | str, replacements: list[tuple[str, str]]) -> None:
    """Exact-string patching. Aborts without writing unless every 'old' occurs exactly once."""
    path = Path(path)
    original = path.read_text()
    src = original
    for old, new in replacements:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"PATCH ABORTED (nothing written): expected 1 match, found {n} in {path}:\n{old[:300]}")
        src = src.replace(old, new)
    bak = path.with_name(path.name + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    bak.write_text(original)
    path.write_text(src)
    print(f"Patched {path} ({len(replacements)} edit(s)); backup: {bak.name}")
