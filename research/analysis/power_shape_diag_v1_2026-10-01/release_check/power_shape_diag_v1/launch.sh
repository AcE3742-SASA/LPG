#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# )); then
  echo 'Usage: bash launch.sh (fresh diagnostic run; no arguments)' >&2
  exit 2
fi
python3 - "$here" <<'PY'
from pathlib import Path
import os
import sys
root = Path('/home/eu/gemm-research').resolve()
if root not in Path(sys.argv[1]).resolve().parents or os.geteuid() == 0:
    raise SystemExit('Run as the normal user from /home/eu/gemm-research/scripts/power_shape_diag_v1')
PY
mkdir -p "$here/runs"
out="$(mktemp -d "$here/runs/power_shape_diag_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
mkdir -p "$out/launcher"
cp -- "${BASH_SOURCE[0]}" "$out/launcher/launcher_source.sh"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$out/launcher/exit_code.txt"
  python3 -B "$here/diagnose.py" pack --output "$out" || exit 1
  echo "RUN_DIR=$out"
  echo 'Run ended. Returning to the current terminal.'
  exit "$code"
}
trap finish EXIT
python3 -B "$here/diagnose.py" verify 2>&1 | tee "$out/launcher/source_check.log"
echo "RUN_DIR=$out"
echo 'Scope: FP32 prealloc, K=72 and K=127; 720/6801 MHz, existing 70 W.'
echo '128 diagnostic trials, then 2 Nsight Systems and 2 Nsight Compute captures.'
echo 'Expected budget: about 15-30 minutes; cooling/profiling can take longer.'
if [[ -n "${TMUX:-}" ]]; then
  echo 'Running in the current tmux session. Detach: press Ctrl+B, release, then D.'
else
  echo 'Keep this terminal and SSH connection open until completion.'
fi
echo 'sudo is used only to set/reset clocks. Measurements use the existing ai_env as the normal user.'
sudo -v
sudo -n true
command=(bash "$here/run_research_checked.sh" python -u -B "$here/diagnose.py" run --output "$out")
printf '%q ' "${command[@]}" > "$out/launcher/command.txt"
set +e
GEMM_RESEARCH_ROOT="$out/launcher/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" \
  "${command[@]}" 2>&1 | tee "$out/launcher/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$out/launcher/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
