#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
size="${1:-full}"
if (( $# )); then shift; fi
case "$size" in
  check) out="$here/runs/check_expansion_720"; extra=() ;;
  full) out="$here/runs/expansion_10000"; extra=(--check "$here/admission/abstract_200_720_recovery_20260921_20260922_024735_e61123.zip"
    --review-note '2026-10-28 user requested 10000 additional unique GEMM shapes, 180000 measurements. New 9000 train and 1000 development-validation shapes only. Preserve all existing splits and locked tests. Reuse reviewed 720/6801 MHz, 70 W, timing, correctness, telemetry, Graph policy and cleanup. Retain historical cumulative-counter ambiguity and flag-and-continue policy.') ;;
  *) echo 'Use: launch.sh check|full [--output PATH] [--retry-failed]' >&2; exit 2 ;;
esac
while (( $# )); do
  case "$1" in
    --output) out="${2:?Missing output path}"; shift 2 ;;
    --retry-failed) extra+=("$1"); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
python3 -B "$here/run.py" verify --size "$size"
# Validate before mkdir, sudo, or reading an existing result directory.
python3 - "$here" "$out" <<'PY'
from pathlib import Path
import sys
here,out=map(lambda p:Path(p).resolve(),sys.argv[1:])
if not out.is_relative_to(here/'runs'):
    raise SystemExit('Output must remain inside this release runs directory')
PY
mkdir -p "$out/launcher"
out="$(cd -- "$out" && pwd)"
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo 'Another launcher owns this output.' >&2; exit 1; }
invocation="$(mktemp -d "$out/launcher/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "${BASH_SOURCE[0]}" "$invocation/launcher_source.sh"
monitor=""
finish() {
  code=$?
  trap - EXIT
  if [[ -n "$monitor" ]]; then
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    python3 -B "$here/progress.py" "$out" "$size" || true
  fi
  printf '%s\n' "$code" > "$invocation/exit_code.txt"
  python3 -B "$here/run.py" pack --output "$out" || exit 1
  echo "RUN_DIR=$out"
  echo 'Run ended. Returning to the current terminal.'
  exit "$code"
}
trap finish EXIT
echo "RUN_DIR=$out"
echo 'Expected full collection: 10000 new shapes, 180000 measurements.'
echo 'Use the current tmux session. Detach: Ctrl+B, release, then D.'
echo 'sudo is used only to set/reset clocks. Measurements use the existing ai_env.'
sudo -v
sudo -n true
command=(bash "$here/run_research_checked.sh" python -u -B "$here/run.py" run --size "$size" --output "$out")
if (( ${#extra[@]} )); then command+=("${extra[@]}"); fi
printf '%q ' "${command[@]}" > "$invocation/command.txt"
python3 -u -B "$here/progress.py" "$out" "$size" "$$" &
monitor=$!
set +e
GEMM_RESEARCH_ROOT="$invocation/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" \
  "${command[@]}" 2>&1 | tee "$invocation/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$invocation/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
