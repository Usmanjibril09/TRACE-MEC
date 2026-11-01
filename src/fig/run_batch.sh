#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG="$1"; shift
for f in "$@"; do
  python "src/fig/$f.py" 2>&1 | tee -a "$LOG"; rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "BATCH FAIL at $f (exit $rc)" | tee -a "$LOG"; exit "$rc"; fi
done
echo "BATCH COMPLETE: $*" | tee -a "$LOG"
