#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
size="${1:-check}"
if [[ "$size" != check && "$size" != full ]]; then
  echo 'Usage: bash launch.sh check|full [--output PATH] [--check PATH --review-note TEXT] [--retry-failed]' >&2
  exit 2
fi
if (( $# )); then shift; fi
out="$here/runs/${size}_780"
extra=()
while (( $# )); do
  case "$1" in
    --output) out="${2:?Missing output path}"; shift 2 ;;
    --check|--review-note) extra+=("$1" "${2:?Missing argument}"); shift 2 ;;
    --retry-failed) extra+=("$1"); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo 'tmux가 없어 실행 전 중단했습니다. 설치 여부를 확인해 주세요. GPU 설정은 바꾸지 않았습니다.' >&2
    exit 1
  fi
  printf -v wrapped '%q ' bash "$here/launch.sh" "$size" --output "$out" "${extra[@]}"
  exec tmux new-session -s "gemm780_${size}" "$wrapped"
fi
mkdir -p "$out/launcher"
out="$(cd -- "$out" && pwd)"
# The launcher lock also covers closed-log packaging and blocks overlapping launches.
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo 'Another launcher owns this output.' >&2; exit 1; }
invocation="$(mktemp -d "$out/launcher/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$invocation/exit_code.txt"
  python3 -B "$here/run.py" pack --output "$out" || exit 1
  echo "RUN_DIR=$out"
  echo '실행 종료. tmux 창을 닫으려면 Enter.'
  read -r _ || true
  exit "$code"
}
trap finish EXIT
python3 -B "$here/run.py" verify --size "$size" 2>&1 | tee "$invocation/source_check.log"
echo "RUN_DIR=$out"
echo '분리: Control+B를 누르고 손을 뗀 뒤 D. 다시 접속: tmux attach -t gemm780_'"$size"
echo 'sudo는 클럭 설정·해제에만 사용하고, 측정은 기존 ai_env 일반 사용자로 실행합니다.'
sudo -v
sudo -n true
command=(bash "$here/../gpu_sweep_v6/run_research_checked.sh" python -u -B "$here/run.py" run
         --size "$size" --output "$out" "${extra[@]}")
printf '%q ' "${command[@]}" > "$invocation/command.txt"
set +e
GEMM_RESEARCH_ROOT="$invocation/checked" GEMM_RESEARCH_SUMMARY="$out/summary.json" \
  "${command[@]}" 2>&1 | tee "$invocation/launcher.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[1]}" > "$invocation/tee_exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
