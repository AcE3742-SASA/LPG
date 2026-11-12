#!/usr/bin/env bash
set -Eeuo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite="$here/../gpu_sweep_v6"
mkdir -p "$here/runs"
out="$(mktemp -d "$here/runs/clock_recheck_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "${BASH_SOURCE[0]}" "$out/launcher_source.sh"
echo "CLOCK_RECHECK_DIR=$out"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$out/exit_code.txt"
  python3 -B "$here/lower.py" pack --output "$out" || exit 1
  echo '실행 종료. 현재 터미널로 돌아갑니다.'
  exit "$code"
}
trap finish EXIT
python3 -B "$here/lower.py" verify 2>&1 | tee "$out/source_check.log"
if [[ -n "${TMUX:-}" ]]; then
  echo '현재 tmux 세션에서 실행합니다. 새 세션이나 창을 만들지 않습니다.'
else
  echo '현재 터미널에서 실행합니다. 완료될 때까지 터미널과 SSH 연결을 유지해 주세요.'
fi
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
