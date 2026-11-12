#!/usr/bin/env bash
# Two-kernel correctness/clock check; not a compute-ceiling search.
set -euo pipefail
out=$(mktemp -d /home/eu/cutlass_780_smoke_v3.XXXXXX)
prof=/home/eu/cutlass/build/tools/profiler/cutlass_profiler
monitor_pid=
profiler_pid=
printf '결과 폴더: %s\n' "$out"

finish() {
  status=$?
  trap - EXIT
  if [ -n "$profiler_pid" ]; then
    kill "$profiler_pid" 2>/dev/null || true
    wait "$profiler_pid" 2>/dev/null || true
  fi
  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$status" > "$out/runner.exit_code"
  tar -czf "$out.tar.gz" -C "$out" .
  printf '보낼 파일: %s.tar.gz (종료코드 %s)\n' "$out" "$status"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cp -- "${BASH_SOURCE[0]}" "$out/runner.sh"
git -C /home/eu/cutlass rev-parse HEAD > "$out/commit.txt"
git -C /home/eu/cutlass status --porcelain > "$out/source_status.txt"
sha256sum "$prof" > "$out/binary.sha256"
uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
[[ "$uuid" =~ ^GPU-[[:xdigit:]-]+$ ]]
exec 9>"/tmp/gemm-sweep-$uuid.lock"
if ! flock -n 9; then
  printf '다른 GEMM 실험이 GPU 잠금을 사용 중입니다.\n'
  exit 2
fi
nvidia-smi -i 0 -q > "$out/device_before.txt"
nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader > "$out/processes_before.txt"
if [ -s "$out/processes_before.txt" ]; then
  printf '다른 GPU 계산 프로세스가 있어 점검을 시작하지 않았습니다.\n'
  cat "$out/processes_before.txt"
  exit 2
fi
sudo -v
sudo -n nvidia-smi -i 0 -lgc 780,780 > "$out/lock_sm.log" 2>&1
sudo -n nvidia-smi -i 0 -lmc 6801,6801 > "$out/lock_mem.log" 2>&1
fields=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu

run_one() {
  local tag=$1 kernel=$2 dtype=$3 op=$4 status=0
  nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem,power.limit --format=csv,noheader,nounits > "$out/$tag.before.csv"
  python3 - "$out/$tag.before.csv" <<'PY_CHECK'
import csv, sys
with open(sys.argv[1]) as f:
    rows = list(csv.reader(f))
# Current idle clocks do not establish clocks during the workload.
# Lock-command failures already stop the shell; check actual clocks under load below.
assert len(rows) == 1 and len(rows[0]) == 3 and float(rows[0][2]) == 70, rows
PY_CHECK
  nvidia-smi -i 0 --query-gpu="$fields" --format=csv --loop-ms=200 > "$out/$tag.clocks.csv" 2> "$out/$tag.monitor.err" &
  monitor_pid=$!
  local args=("$prof" --operation=gemm --mode=profile --devices=0
    --kernels="$kernel" --gemm_kind=universal --m=8192 --n=8192 --k=8192
    --A="$dtype:row" --B="$dtype:column" --C="$dtype:column" --D="$dtype:column"
    --op_class="$op" --accum=f32 --alpha=1 --beta=0 --split_k_slices=1 --batch_count=1
    --dist=uniform,min:-2,max:2 --seed=20260921 --workspace-count=1
    --warmup-iterations=10 --profiling-iterations=0 --profiling-duration=1000 --min-iterations=10
    --verification-enabled=true --verification-providers=device --epsilon=0 --nonzero-floor=0
    --output="$out/$tag")
  printf '%q ' "${args[@]}" > "$out/$tag.command.txt"
  printf '\n' >> "$out/$tag.command.txt"
  printf '%s\tstart\t%s\n' "$tag" "$(date --iso-8601=ns)" >> "$out/timeline.tsv"
  printf '%s: device 검증과 짧은 측정 시작\n' "$tag"
  "${args[@]}" > "$out/$tag.log" 2>&1 &
  profiler_pid=$!
  wait "$profiler_pid" || status=$?
  profiler_pid=
  printf '%s\n' "$status" > "$out/$tag.exit_code"
  printf '%s\tend\t%s\n' "$tag" "$(date --iso-8601=ns)" >> "$out/timeline.tsv"
  kill -0 "$monitor_pid"
  kill "$monitor_pid"
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=
  nvidia-smi -i 0 -q > "$out/$tag.device_after.txt"
  if [ "$status" -ne 0 ]; then
    printf '%s: 프로파일러 실패, 종료코드 %s. 원로그 보존.\n' "$tag" "$status"
    return "$status"
  fi
python3 - "$out" "$tag" "$kernel" "$dtype" "$op" <<'PY'
import csv, json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
results = {}
tag, kernel, dtype, op = sys.argv[2:]
with (p / (tag + '.gemm.csv')).open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 1, (tag, 'expected one result', len(rows))
r = rows[0]
assert r['Provider'].lower() == 'cutlass' and r['Operation'] == kernel
correct = r['Status'].lower() == 'success' and r['Disposition'].lower() == 'passed'
assert all(int(r[x]) == 8192 for x in ('m', 'n', 'k'))
assert r['op_class'] == op and r['accum'] == 'f32'
assert all(r[x].split(':')[0] == dtype for x in ('A', 'B', 'C', 'D'))
ms = float(r['Runtime'])
assert math.isfinite(ms) and ms > 0
results[tag] = {'runtime_ms': ms, 'tflops_2mnk': 2 * 8192**3 / ms / 1e9, 'row': r}
with (p / (tag + '.clocks.csv')).open() as f:
    clocks = list(csv.DictReader(f, skipinitialspace=True))
number = lambda row, key: float(row[key].split()[0])
active = [r for r in clocks if number(r, 'utilization.gpu [%]') > 0]
clock_ok = len(active) >= 2 and all((number(r, 'clocks.current.sm [MHz]'), number(r, 'clocks.current.memory [MHz]'),
    number(r, 'power.limit [W]')) == (780, 6801, 70) for r in active)
monitor_ok = not (p / (tag + '.monitor.err')).read_text().strip()
results.update(correctness_pass=correct, clocks_pass=clock_ok, monitor_pass=monitor_ok, verification_provider='device')
results['active_clock_samples'] = len(active)
(p / (tag + '.check.json')).write_text(json.dumps(results, indent=2) + '\n')
if not (correct and clock_ok and monitor_ok):
    print(f'{tag}: FAIL (correctness={correct}, clocks={clock_ok}, monitor={monitor_ok}); 원자료 보존')
    sys.exit(1)
print(f"{tag}: PASS, {ms:.2f} ms, {results[tag]['tflops_2mnk']:.2f} TFLOP/s, 부하 표본 {len(active)}개 780/6801 MHz·70 W")
PY
}

run_one fp32_simt cutlass_simt_sgemm_256x128_8x5_tn_align1 f32 simt
run_one fp16_tensor cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8 f16 tensorop
printf '두 경로 smoke PASS. 최종 천장 확정에는 커널·크기·반복 비교가 남아 있습니다.\n'
