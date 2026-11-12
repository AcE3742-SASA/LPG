#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite="$here/../gpu_sweep_v6"
mkdir -p "$here/runs"
out="$(mktemp -d "$here/runs/clock_range_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "CLOCK_RANGE_DIR=$out"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$out/exit_code.txt"
  python3 -B "$here/lower.py" pack --output "$out" || exit 1
  exit "$code"
}
trap finish EXIT
python3 -B "$here/lower.py" verify 2>&1 | tee "$out/source_check.log"
echo "Checking sudo authorization; clocks only use sudo, benchmark runs in ai_env as your user."
sudo -v
sudo -n true
command=(bash "$suite/run_research_checked.sh" python -u -B "$here/lower.py" run --output "$out")
printf '%q ' "${command[@]}" > "$out/command.txt"
set +e
GEMM_RESEARCH_ROOT="$out/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" "${command[@]}" 2>&1 | tee "$out/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$out/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
