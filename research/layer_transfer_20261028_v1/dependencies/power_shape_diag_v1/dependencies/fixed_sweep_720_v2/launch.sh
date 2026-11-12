#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
size="${1:-full}"
if (( $# )); then shift; fi
case "$size" in
  check) out="$here/runs/check_720"; extra=() ;;
  full) out="$here/runs/full_720"; extra=(--check "$here/../abstract_sweep_725_v1/runs/abstract_200_720_recovery_20260921_20260922_024735_e61123.zip"
    --review-note '2026-09-22 reviewed: completed 720 MHz 200-shape collection, 3600 PASS, three rounds, timing and sampled clocks pass, resets accepted. User approved the full 113382-measurement sweep and flag-and-continue for clock/SW power-cap observations. Keep cumulative counter ambiguity unresolved and preserve all prior evidence.') ;;
  *) echo 'Use: launch.sh check|full [--output PATH] [--check PATH] [--review-note TEXT]' >&2; exit 2 ;;
esac
while (( $# )); do
  case "$1" in
    --output) out="${2:?Missing output path}"; shift 2 ;;
    --check|--review-note) extra+=("$1" "${2:?Missing argument}"); shift 2 ;;
    --retry-failed) extra+=("$1"); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
# Import a stopped v1 run once; never replace files used by an active worker.
previous="$here/../fixed_sweep_720_v1/runs/full_720"
if [[ "$size" == full && ! -e "$out/manifest.json" && -e "$previous/manifest.json" ]]; then
  python3 -B "$here/prepare.py" --source "$previous" --output "$out"
fi
mkdir -p "$out/launcher"
out="$(cd -- "$out" && pwd)"
# The launcher lock also covers closed-log packaging and blocks overlapping launches.
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
  echo '실행 종료. 현재 터미널로 돌아갑니다.'
  exit "$code"
}
trap finish EXIT
python3 -B "$here/run.py" verify --size "$size" 2>&1 | tee "$invocation/source_check.log"
echo "RUN_DIR=$out"
if [[ -n "${TMUX:-}" ]]; then
  echo '현재 tmux 세션에서 실행합니다. 분리: Control+B를 누르고 손을 뗀 뒤 D.'
else
  echo '현재 터미널에서 실행합니다. 완료될 때까지 터미널과 SSH 연결을 유지해 주세요.'
fi
echo 'sudo는 클럭 설정·해제에만 사용하고, 측정은 기존 ai_env 일반 사용자로 실행합니다.'
sudo -v
sudo -n true
command=(bash "$here/../gpu_sweep_v6/run_research_checked.sh" python -u -B "$here/run.py" run
         --size "$size" --output "$out")
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
