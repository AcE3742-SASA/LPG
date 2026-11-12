#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
size=full
out="$here/runs/abstract_200_750"
extra=(--check "$here/../clock_recheck_v1/runs/clock_recheck_20260921_143452_YZ9HN5" --review-note "$(cat "$here/review_note.md")")
while (( $# )); do
  case "$1" in
    --output) out="${2:?Missing output path}"; shift 2 ;;
    --check|--review-note) extra+=("$1" "${2:?Missing argument}"); shift 2 ;;
    --retry-failed) extra+=("$1"); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$out/launcher"
out="$(cd -- "$out" && pwd)"
# The launcher lock also covers closed-log packaging and blocks overlapping launches.
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo 'Another launcher owns this output.' >&2; exit 1; }
invocation="$(mktemp -d "$out/launcher/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "${BASH_SOURCE[0]}" "$invocation/launcher_source.sh"
finish() {
  code=$?
  trap - EXIT
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
set +e
GEMM_RESEARCH_ROOT="$invocation/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" \
  "${command[@]}" 2>&1 | tee "$invocation/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$invocation/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
