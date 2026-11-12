#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# )); then echo 'Usage: bash launch.sh (fresh paired pilot; no arguments)' >&2; exit 2; fi
python3 - "$here" <<'PY'
from pathlib import Path
import os,sys
if Path('/home/eu/gemm-research').resolve() not in Path(sys.argv[1]).resolve().parents or os.geteuid()==0:
    raise SystemExit('Run as eu from /home/eu/gemm-research/scripts/layer_compare_v1')
PY
mkdir -p "$here/runs"
out="$(mktemp -d "$here/runs/layer_compare_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
mkdir -p "$out/launcher"
cp -- "${BASH_SOURCE[0]}" "$out/launcher/launcher_source.sh"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$out/launcher/exit_code.txt"
  python3 -B "$here/compare.py" pack --output "$out" || exit 1
  echo "RUN_DIR=$out"
  echo 'Run ended. Returning to the current terminal.'
  exit "$code"
}
trap finish EXIT
python3 -B "$here/compare.py" verify 2>&1 | tee "$out/launcher/source_check.log"
echo "RUN_DIR=$out"
echo 'Scope: 160 development cases, FP16/FP32, standard nn vs explicit GEMM, ordinary calls.'
echo '48 representative measurements first; continue to 1920 only after HEALTH PASS.'
echo 'No Nsight capture, Graph, compile, or model training.'
sudo -v
sudo -n true
command=(bash "$here/run_research_checked.sh" python -u -B "$here/compare.py" run --output "$out")
printf '%q ' "${command[@]}" > "$out/launcher/command.txt"
set +e
GEMM_RESEARCH_ROOT="$out/launcher/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" \
  "${command[@]}" 2>&1 | tee "$out/launcher/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$out/launcher/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
