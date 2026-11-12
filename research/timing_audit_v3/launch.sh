#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite="${1:-$here/../gpu_sweep}"
suite="$(cd -- "$suite" && pwd)"
mkdir -p "$here/runs"
out="$(mktemp -d "$here/runs/timing_audit_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "AUDIT_DIR=$out"
set +e
python3 -B "$here/audit.py" verify --suite "$suite" 2>&1 | tee "$out/source_check.log"
source_codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${source_codes[0]}" > "$out/source_check.exit_code.txt"
if (( source_codes[0] != 0 || source_codes[1] != 0 )); then
  printf '%s\n' "${source_codes[0]}" > "$out/exit_code.txt"
  python3 -B "$here/audit.py" pack --output "$out"
  if (( source_codes[0] != 0 )); then exit "${source_codes[0]}"; fi
  exit "${source_codes[1]}"
fi
command=(bash "$suite/run_research_checked.sh" python -u -B "$here/audit.py" run --suite "$suite" --output "$out")
printf '%q ' "${command[@]}" > "$out/command.txt"
set +e
GEMM_RESEARCH_ROOT="$out/checked" "${command[@]}" 2>&1 | tee "$out/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[0]}" > "$out/exit_code.txt"
printf '%s\n' "${codes[1]}" > "$out/tee_exit_code.txt"
# Packaging starts only after all experiment/telemetry/output logs are closed.
python3 -B "$here/audit.py" pack --output "$out"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
