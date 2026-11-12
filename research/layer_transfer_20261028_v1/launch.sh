#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phase="${1:-core_dev}"
release=()
case "$phase" in
  core_dev|conv_confirm|conv_diagnostic)
    if (( $# > 1 )); then echo 'FAIL: unexpected argument' >&2; exit 2; fi ;;
  core_locked)
    if [[ "${2:-}" != '--acknowledge-locked-test-release' || $# != 2 ]]; then
      echo 'FAIL: locked test disabled; a separate explicit release action is required.' >&2
      exit 2
    fi
    release=(--acknowledge-locked-test-release) ;;
  *) echo 'Usage: bash launch.sh [core_dev|conv_confirm|conv_diagnostic]' >&2; exit 2 ;;
esac
python3 - "$here" <<'PY'
from pathlib import Path
import os, sys
if Path(sys.argv[1]).resolve() != Path('/home/eu/gemm-research/layer_transfer_20261028_v1') or os.geteuid() == 0:
    raise SystemExit('Run as eu from /home/eu/gemm-research/layer_transfer_20261028_v1')
PY
python3 -B "$here/compare.py" verify --phase "$phase" "${release[@]}"
mkdir -p "$here/runs"
out="$(mktemp -d "$here/runs/${phase}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
mkdir -p "$out/launcher"
cp -- "${BASH_SOURCE[0]}" "$out/launcher/launcher_source.sh"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$out/launcher/exit_code.txt"
  python3 -B "$here/compare.py" pack --output "$out" --phase "$phase" "${release[@]}" || exit 1
  echo "RUN_DIR=$out"
  echo 'Run ended. Failed or interrupted output is preserved; restart uses a fresh run.'
  exit "$code"
}
trap finish EXIT
echo "RUN_DIR=$out"
echo "PHASE=$phase"
echo 'Ordinary calls only. No Graph, compile, training, or profiling.'
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
