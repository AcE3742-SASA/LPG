#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# )); then echo 'Use: bash recover_720_v1/launch.sh (no arguments)' >&2; exit 2; fi
python3 -B "$here/prepare.py"
note="$(python3 -B - "$here" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from prepare import REVIEW_NOTE
print(REVIEW_NOTE)
PY
)"
out="$here/../abstract_sweep_725_v1/runs/abstract_200_720_recovery_20260921"
python3 -u -B "$here/progress.py" "$out" "$$" &
monitor=$!
finish() {
  code=$?
  trap - EXIT
  kill "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  python3 -B "$here/progress.py" "$out" || true
  exit "$code"
}
trap finish EXIT
bash "$here/../abstract_sweep_725_v1/launch.sh" full \
  --check "$here/../abstract_sweep_725_v1/runs/check_720_repeat_20260921_190318" \
  --output "$out" \
  --review-note "$note"
