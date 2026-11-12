#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
caller_dir="$PWD"
mode="${1:-pilot}"
if [[ "$mode" != pilot && "$mode" != full ]]; then
  echo 'Usage: bash launch.sh pilot|full [output_directory] [extra arguments]' >&2
  exit 1
fi
if (( $# > 0 )); then shift; fi
out="${1:-$here/runs/$mode}"
if (( $# > 0 )); then shift; fi
mkdir -p "$out/launcher"
out="$(cd -- "$out" && pwd)"
# The original runner changes cwd to a fresh RUN_DIR. Resolve user paths first.
command=(bash "$here/run_research_checked.sh" python -u "$here/$mode.py" --output "$out")
while (( $# > 0 )); do
  case "$1" in
    --pilot)
      if (( $# < 2 )); then echo '--pilot needs a directory' >&2; exit 1; fi
      pilot_dir="$2"
      if [[ "$pilot_dir" != /* ]]; then pilot_dir="$caller_dir/$pilot_dir"; fi
      command+=(--pilot "$pilot_dir")
      shift 2
      ;;
    --pilot=*)
      pilot_dir="${1#--pilot=}"
      if [[ -z "$pilot_dir" ]]; then echo '--pilot needs a directory' >&2; exit 1; fi
      if [[ "$pilot_dir" != /* ]]; then pilot_dir="$caller_dir/$pilot_dir"; fi
      command+=(--pilot "$pilot_dir")
      shift
      ;;
    --review-note)
      if (( $# < 2 )); then echo '--review-note needs text' >&2; exit 1; fi
      command+=("$1" "$2")
      shift 2
      ;;
    *) command+=("$1"); shift ;;
  esac
done
log_id="$(date +%Y%m%d_%H%M%S)_$$"
printf '%q ' "${command[@]}" > "$out/launcher/$log_id.command.txt"
set +e
GEMM_RESEARCH_ROOT="$out/checked" "${command[@]}" 2>&1 | tee "$out/launcher/$log_id.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[0]}" > "$out/launcher/$log_id.exit_code.txt"
if (( codes[0] == 2 )); then
  echo 'Exit 2: inspect summary.json. A completed sweep with gate=REVIEW needs review; the original wrapper labels every nonzero exit FAIL.'
fi
if (( codes[0] == 0 && codes[1] != 0 )); then exit "${codes[1]}"; fi
exit "${codes[0]}"
