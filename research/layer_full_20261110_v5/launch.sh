#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 2 || $# > 3 )); then
  echo 'Usage: bash launch.sh BATCH OUTPUT [--collect-sealed-test]' >&2
  exit 2
fi
phase="$1"
out="$2"
release=()
if [[ "$phase" == test_* ]]; then
  [[ "${3:-}" == '--collect-sealed-test' ]] || { echo 'FAIL: sealed test flag required' >&2; exit 2; }
  release=(--collect-sealed-test)
elif (( $# != 2 )); then
  echo 'FAIL: unexpected argument' >&2
  exit 2
fi
python3 - "$here" "$out" <<'PY'
from pathlib import Path
import os, sys
here, out = (Path(value).resolve() for value in sys.argv[1:])
if here != Path('/home/eu/gemm-research/layer_full_20261110_v5') or os.geteuid() == 0:
    raise SystemExit('Run as eu from /home/eu/gemm-research/layer_full_20261110_v5')
if not out.is_relative_to(here/'runs') or out.exists():
    raise SystemExit('Use a fresh output directory inside this release runs directory')
PY
python3 -B "$here/compare.py" verify --phase "$phase" "${release[@]}"
mkdir -p "$out/launcher"
cp -- "${BASH_SOURCE[0]}" "$out/launcher/launcher_source.sh"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$out/launcher/exit_code.txt"
  python3 -B "$here/compare.py" pack --output "$out" --phase "$phase" "${release[@]}" || exit 1
  echo "RUN_DIR=$out"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
echo "RUN_DIR=$out"
echo "BATCH=$phase"
sudo -v
sudo -n true
command=(bash "$here/run_research_checked.sh" python -u -B "$here/compare.py" run --output "$out" --phase "$phase" "${release[@]}")
printf '%q ' "${command[@]}" > "$out/launcher/command.txt"
set +e
GEMM_RESEARCH_ROOT="$out/launcher/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" \
  "${command[@]}" 2>&1 | tee "$out/launcher/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$out/launcher/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
