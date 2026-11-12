#!/usr/bin/env bash
# Two-kernel correctness/clock check; not a compute-ceiling search.
set -euo pipefail
out=$(mktemp -d /home/eu/cutlass_780_smoke.XXXXXX)
prof=/home/eu/cutlass/build/tools/profiler/cutlass_profiler
monitor_pid=
printf '결과 폴더: %s\n' "$out"

finish() {
  status=$?
  trap - EXIT
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
nvidia-smi -i 0 -q > "$out/device_before.txt"
nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader > "$out/processes_before.txt"
if [ -s "$out/processes_before.txt" ]; then
  printf '다른 GPU 계산 프로세스가 있어 점검을 시작하지 않았습니다.\n'
  cat "$out/processes_before.txt"
  exit 2
fi
fields=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu
nvidia-smi -i 0 --query-gpu="$fields" --format=csv > "$out/before.csv"
nvidia-smi -i 0 --query-gpu="$fields" --format=csv --loop-ms=200 \
  > "$out/clocks.csv" 2> "$out/monitor.err" &
monitor_pid=$!

run_one() {
  local tag=$1 kernel=$2 dtype=$3 op=$4 status=0
  local args=("$prof" --operation=gemm --mode=profile --devices=0
    --kernels="$kernel" --gemm_kind=universal --m=8192 --n=8192 --k=8192
    --A="$dtype:row" --B="$dtype:column" --C="$dtype:column" --D="$dtype:column"
    --op_class="$op" --accum=f32 --alpha=1 --beta=0 --split_k_slices=1 --batch_count=1
    --dist=uniform,min:-2,max:2 --seed=20260921 --workspace-count=1
    --warmup-iterations=10 --profiling-iterations=0 --profiling-duration=1000 --min-iterations=10
    --verification-enabled=true --verification-providers=cublas --epsilon=0 --nonzero-floor=0
    --output="$out/$tag")
  printf '%q ' "${args[@]}" > "$out/$tag.command.txt"
  printf '\n' >> "$out/$tag.command.txt"
  printf '%s\tstart\t%s\n' "$tag" "$(date --iso-8601=ns)" >> "$out/timeline.tsv"
  "${args[@]}" > "$out/$tag.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$out/$tag.exit_code"
  printf '%s\tend\t%s\n' "$tag" "$(date --iso-8601=ns)" >> "$out/timeline.tsv"
  if [ "$status" -ne 0 ]; then cat "$out/$tag.log"; return "$status"; fi
  kill -0 "$monitor_pid"
  printf '%s 실행 종료\n' "$tag"
}

run_one fp32_simt cutlass_simt_sgemm_256x128_8x5_tn_align1 f32 simt
run_one fp16_tensor cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8 f16 tensorop
kill -0 "$monitor_pid"
kill "$monitor_pid"
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
nvidia-smi -i 0 -q > "$out/device_after.txt"

python3 - "$out" <<'PY'
import csv, json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
results = {}
for tag, dtype, op in [('fp32_simt', 'f32', 'simt'), ('fp16_tensor', 'f16', 'tensorop')]:
    with (p / (tag + '.gemm.csv')).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, (tag, 'expected one result', len(rows))
    r = rows[0]
    assert r['Provider'].lower() == 'cutlass'
    assert r['Status'].lower() == 'success' and r['Disposition'].lower() == 'passed', (tag, r)
    assert all(int(r[x]) == 8192 for x in ('m', 'n', 'k'))
    assert r['op_class'] == op and r['accum'] == 'f32'
    assert all(r[x].split(':')[0] == dtype for x in ('A', 'B', 'C', 'D'))
    ms = float(r['Runtime'])
    assert math.isfinite(ms) and ms > 0
    results[tag] = {'runtime_ms': ms, 'tflops_2mnk': 2 * 8192**3 / ms / 1e9, 'row': r}
    print(f"{tag}: correctness PASS, {ms:.2f} ms, {results[tag]['tflops_2mnk']:.2f} TFLOP/s")
with (p / 'clocks.csv').open() as f:
    clocks = list(csv.DictReader(f, skipinitialspace=True))
number = lambda row, key: float(row[key].split()[0])
active = [r for r in clocks if number(r, 'utilization.gpu [%]') > 0]
assert len(active) >= 2, 'Insufficient active clock samples'
assert all((number(r, 'clocks.current.sm [MHz]'), number(r, 'clocks.current.memory [MHz]'),
            number(r, 'power.limit [W]')) == (780, 6801, 70) for r in active), 'Clock/power target mismatch'
assert not (p / 'monitor.err').read_text().strip(), 'Telemetry error'
results['active_clock_samples'] = len(active)
(p / 'smoke_summary.json').write_text(json.dumps(results, indent=2) + '\n')
print(f'부하 중 클록 표본 {len(active)}개: 780/6801 MHz, 70 W 확인')
PY
