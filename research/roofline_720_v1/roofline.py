"""720 MHz empirical Roofline: existing CUTLASS + BabelStream, Python stdlib only.

Run: bash run.sh
Recompute saved results without GPU access: python3 -B roofline.py --analyze RESULT_DIR
"""
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics as st
import subprocess as sp
import sys
import tarfile
import tempfile
import traceback

ROOT = Path(__file__).resolve().parent
REMOTE = Path('/home/eu')
CUTLASS = REMOTE / 'cutlass/build/tools/profiler/cutlass_profiler'
BABEL = REMOTE / 'BabelStream/build-cuda-sm89/cuda-stream'
MATRIX_SIZES = [4096, 8192, 16384]
ARRAY_SIZES = [67108864, 134217728, 268435456]
WEIGHTS = {'Copy': 2, 'Mul': 2, 'Add': 3, 'Triad': 3, 'Dot': 2}
ROUNDS = 5
TARGET = [720, 6801, 70]
FIELDS = ['timestamp', 'utilization.gpu', 'clocks.sm', 'clocks.mem', 'power.draw',
          'power.limit', 'temperature.gpu', 'clocks_event_reasons.sw_power_cap',
          'clocks_event_reasons.sw_thermal_slowdown', 'clocks_event_reasons.hw_thermal_slowdown',
          'clocks_event_reasons_counters.sw_power_cap']
SAMPLE_MS = 50
PINS = {'cutlass': '147295a3d4b75f3aeff247c25b8927cea9a7006a',
        'BabelStream': '17ab377b0e919e14fd3df2b67268761fdac8abb3'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def plan(kernels):
    require(len(kernels) == 16 and len({k['kernel'] for k in kernels}) == 16, 'kernel inventory')
    jobs = []
    for round_no in range(ROUNDS + 1):
        gemms = [dict(kind='gemm', round=round_no, size=n, spec=k,
                      tag=f"r{round_no}_{k['tag']}_n{n}") for n in MATRIX_SIZES for k in kernels]
        streams = [dict(kind='bw', round=round_no, size=n, tag=f'r{round_no}_bw_n{n}') for n in ARRAY_SIZES]
        if round_no:
            random.Random(20260921 + round_no).shuffle(gemms)
            random.Random(20260921 + round_no).shuffle(streams)
        jobs.extend(streams + gemms if round_no % 2 == 0 else gemms + streams)
    return dict(version='roofline_720_v1', target=TARGET, rounds=ROUNDS,
                warmup_round_excluded=0, matrix_sizes=MATRIX_SIZES, array_sizes=ARRAY_SIZES,
                sample_ms=SAMPLE_MS, sensor_resolution_verified=False,
                variation_review_percent=5, clocks_left_locked_on_exit=True, jobs=jobs)


def command(job, out):
    n = job['size']
    if job['kind'] == 'bw':
        return [str(BABEL), '--device', '0', '--float', '--arraysize', str(n), '--numtimes', '100', '--order', 'Classic', '--gigabytes', '--csv']
    k = job['spec']
    return [str(CUTLASS), '--operation=gemm', '--mode=profile', '--devices=0',
            '--kernels=' + k['kernel'], '--gemm_kind=universal', f'--m={n}', f'--n={n}', f'--k={n}',
            *['--' + x + '=' + k[x] for x in ('A', 'B', 'C', 'D')],
            '--op_class=' + k['op'], '--accum=f32', '--alpha=1', '--beta=0',
            '--split_k_slices=1', '--batch_count=1', '--dist=uniform,min:-2,max:2',
            '--seed=20260921', '--workspace-count=1', '--warmup-iterations=10',
            '--profiling-iterations=0', '--profiling-duration=1000', '--min-iterations=10',
            '--verification-enabled=true', '--verification-providers=device', '--epsilon=0',
            '--nonzero-floor=0', '--output=' + str(out / job['tag'])]


def telemetry(path):
    rows = list(csv.reader(io.StringIO(path.read_text()), skipinitialspace=True))
    require(bool(rows), 'missing telemetry')
    result = []
    for row in rows:
        require(len(row) == len(FIELDS), f'malformed telemetry: {row}')
        timestamp = dt.datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f')
        numbers = list(map(float, row[1:7]))
        require(all(math.isfinite(v) for v in numbers), 'nonfinite telemetry')
        require(0 <= numbers[0] <= 100, 'GPU utilization outside 0..100')
        flags = [x.strip().lower() for x in row[7:10]]
        require(all(x in ('active', 'not active') for x in flags), f'unsupported reason flags: {flags}')
        counter = int(row[10])
        result.append(dict(timestamp=timestamp, util=numbers[0], sm=numbers[1], mem=numbers[2],
                           power=numbers[3], limit=numbers[4], temp=numbers[5],
                           reasons=[x == 'active' for x in flags], power_cap_counter_us=counter))
    return result


def check_run(out, job):
    tag = job['tag']
    require((out / (tag + '.exit_code')).read_text().strip() == '0', f'{tag}: program failed')
    require(not (out / (tag + '.monitor.err')).read_text().strip(), f'{tag}: telemetry error')
    log = (out / (tag + '.log')).read_text()
    n = job['size']
    values = []
    if job['kind'] == 'gemm':
        rows = list(csv.DictReader(io.StringIO((out / (tag + '.gemm.csv')).read_text())))
        require(len(rows) == 1, f'{tag}: expected exactly one CUTLASS row')
        r, k = rows[0], job['spec']
        require(r['Provider'] == 'CUTLASS' and r['Operation'] == k['kernel'], f'{tag}: wrong kernel')
        require(r['Status'] == 'success' and r['Disposition'] == 'passed' and 'reference_device: Passed' in log, f'{tag}: correctness not passed')
        require(all(int(r[x]) == n for x in ('m', 'n', 'k')), f'{tag}: wrong shape')
        require(all(r[x] == k[x] for x in ('A', 'B', 'C', 'D')), f'{tag}: wrong dtype/layout')
        require(r['op_class'] == k['op'] and r['accum'] == 'f32', f'{tag}: wrong arithmetic')
        require([r[x] for x in ('alpha', 'beta', 'batch_count', 'split_k_slices')] == ['1', '0', '1', '1'], f'{tag}: wrong GEMM parameters')
        ms = float(r['Runtime'])
        require(math.isfinite(ms) and ms > 0, f'{tag}: invalid runtime')
        require(int(r['Flops']) == 2*n**3 + 2*n**2, f'{tag}: unexpected FLOP count')
        require(math.isclose(float(r['GFLOPs']), int(r['Flops'])/ms/1e6, rel_tol=2e-5), f'{tag}: throughput mismatch')
        values = [dict(function=k['kernel'], path=k['op'], value=2*n**3/ms/1e9, unit='TFLOP/s', runtime_ms=ms)]
    else:
        require(not re.search(r'FAILED|CUDA Error|NVML Error|exception', log, re.I), f'{tag}: BabelStream error')
        require('Memory Allocation: DEFAULT' in log, f'{tag}: unexpected memory allocation')
        start = log.index('function,num_times,')
        rows = list(csv.DictReader(io.StringIO(log[start:])))
        require(len(rows) == 5 and {r['function'] for r in rows} == set(WEIGHTS), f'{tag}: incomplete BabelStream rows')
        for r in rows:
            require((int(r['n_elements']), int(r['sizeof']), int(r['num_times'])) == (n, 4, 100), f'{tag}: wrong bandwidth settings')
            lo, hi, avg, bw = [float(r[x]) for x in ('min_runtime', 'max_runtime', 'avg_runtime', 'max_GB_per_sec')]
            require(all(math.isfinite(x) and x > 0 for x in (lo, hi, avg, bw)) and lo <= avg <= hi, f'{tag}: invalid bandwidth runtime')
            require(math.isclose(bw, WEIGHTS[r['function']]*4*n/lo/1e9, rel_tol=2e-5), f'{tag}: bandwidth arithmetic mismatch')
            values.append(dict(function=r['function'], path='bw', value=bw, unit='GB/s',
                               avg_time_based_GB_s=WEIGHTS[r['function']]*4*n/avg/1e9))
    clocks = telemetry(out / (tag + '.clocks.csv'))
    active = [r for r in clocks if r['util'] > 0]
    require(len(active) >= 2, f'{tag}: insufficient loaded clock samples')
    require(all([r['sm'], r['mem'], r['limit']] == TARGET for r in active), f'{tag}: loaded clocks/power limit differ from 720/6801/70')
    gaps = [(b['timestamp'] - a['timestamp']).total_seconds() for a, b in zip(clocks, clocks[1:])]
    require(gaps and min(gaps) > 0 and max(gaps) <= 0.5, f'{tag}: telemetry gap above 0.5 s or invalid timestamps')
    reason_counts = [sum(r['reasons'][i] for r in active) for i in range(3)]
    counter_delta = clocks[-1]['power_cap_counter_us'] - clocks[0]['power_cap_counter_us']
    require(counter_delta >= 0, f'{tag}: counter reset')
    return dict(tag=tag, kind=job['kind'], round=job['round'], size=n, values=values,
                status='REVIEW' if any(reason_counts) else 'PASS', samples=len(clocks), active_samples=len(active),
                loaded_reason_samples=dict(zip(FIELDS[7:10], reason_counts)),
                max_gap_s=max(gaps), max_power_W=max(r['power'] for r in clocks),
                temperature_range_C=[min(r['temp'] for r in clocks), max(r['temp'] for r in clocks)],
                power_cap_counter_delta_us=counter_delta)


def analyze(out):
    protocol = json.loads((out / 'plan.json').read_text())
    require(protocol == plan(json.loads((out / 'kernels.json').read_text())), 'saved plan differs from the protocol')
    completed, missing, errors = [], [], []
    for job in protocol['jobs']:
        if not (out / (job['tag'] + '.exit_code')).exists():
            missing.append(job['tag'])
            continue
        try:
            completed.append(check_run(out, job))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(dict(tag=job['tag'], error=str(exc)))
    groups = {}
    for r in completed:
        if r['round'] == 0:
            continue
        for v in r['values']:
            key = (v['path'], v['function'], r['size'])
            groups.setdefault(key, []).append((r['round'], v['value'], r['status']))
    summaries = []
    for (path, function, size), samples in sorted(groups.items()):
        samples.sort()
        values = [v for _, v, _ in samples]
        median = st.median(values)
        q1, _, q3 = st.quantiles(values, n=4, method='inclusive') if len(values) >= 2 else (median, median, median)
        spread = 100 * (max(values) - min(values)) / median
        good = [r for r, _, _ in samples] == list(range(1, ROUNDS+1)) and all(s == 'PASS' for _, _, s in samples) and spread <= 5
        summaries.append(dict(path=path, function=function, size=size, repeats=len(values),
                              median=median, min=min(values), max=max(values), iqr=q3-q1,
                              relative_iqr_pct=100*(q3-q1)/median, range_pct=spread,
                              first_to_last_pct=100*(values[-1]/values[0]-1),
                              status='PASS' if good else 'REVIEW'))
    candidates = {p: max((r for r in summaries if r['path'] == p and r['status'] == 'PASS'),
                         key=lambda r: r['median'], default=None) for p in ('simt', 'tensorop', 'bw')}
    complete = not missing and not errors and len(completed) == len(protocol['jobs'])
    if not complete:
        candidates = {p: None for p in candidates}
    status = 'PASS' if complete and all(r['status'] == 'PASS' for r in summaries) and all(r['status'] == 'PASS' for r in completed) else 'REVIEW'
    result = dict(status=status, collection_complete=complete, expected_runs=len(protocol['jobs']),
                  valid_runs=len(completed), missing=missing, errors=errors, target=TARGET,
                  summaries=summaries, ceiling_candidates=candidates,
                  total_power_cap_counter_delta_us=sum(r['power_cap_counter_delta_us'] for r in completed),
                  scope='Within-session empirical ceilings for tested kernels/sizes. Round 0 excluded. No guarantee between telemetry samples. Counter deltas are diagnostics, not timed-kernel throttling fractions.')
    save(out / 'analysis.json', result)
    with (out / 'summary.csv').open('w', newline='') as f:
        if summaries:
            w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    lines = [f'720 MHz Roofline: {status}', f'Completed valid runs: {len(completed)}/{len(protocol["jobs"])}',
             'Round 0 excluded; medians of five separate process invocations within this session.',
             'Candidate ceilings (inspect raw results, size saturation and telemetry before adopting):']
    for path, r in candidates.items():
        unit = 'GB/s' if path == 'bw' else 'TFLOP/s'
        lines.append(f'{path}: {r["median"]:.2f} {unit} ({r["function"]}, size {r["size"]}, range {r["range_pct"]:.2f}%)' if r else f'{path}: no eligible candidate')
    lines.append('Power-cap counter increments are retained separately; they do not prove timed-kernel throttling durations.')
    (out / 'SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    return result


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except sp.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_job(out, job, env):
    tag = job['tag']
    args = command(job, out)
    save(out / (tag + '.command.json'), args)
    monitor = worker = None
    with (out / (tag + '.clocks.csv')).open('w') as clocks, (out / (tag + '.monitor.err')).open('w') as err, (out / (tag + '.log')).open('w') as log:
        try:
            monitor = sp.Popen(['nvidia-smi', '-i', '0', '--query-gpu='+','.join(FIELDS), '--format=csv,noheader,nounits', f'--loop-ms={SAMPLE_MS}'], stdout=clocks, stderr=err, start_new_session=True)
            start = dt.datetime.now().astimezone().isoformat()
            worker = sp.Popen(args, stdout=log, stderr=sp.STDOUT, env=env, start_new_session=True)
            code = worker.wait(timeout=900)
            (out / (tag + '.exit_code')).write_text(str(code)+'\n')
            save(out / (tag + '.timing.json'), dict(start=start, end=dt.datetime.now().astimezone().isoformat()))
            require(monitor.poll() is None, f'{tag}: monitor exited prematurely')
        finally:
            stop(worker)
            stop(monitor)
    checked = check_run(out, job)
    save(out / (tag + '.check.json'), checked)
    return checked


def run():
    out = Path(tempfile.mkdtemp(prefix='roofline_720_', dir=REMOTE))
    code = 1
    lock = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    print(f'결과 폴더: {out}', flush=True)
    def capture(name, args):
        r = sp.run(args, text=True, stdout=sp.PIPE, stderr=sp.PIPE, timeout=30)
        (out / (name+'.txt')).write_text(r.stdout)
        (out / (name+'.err')).write_text(r.stderr)
        require(r.returncode == 0, f'{name}: exit {r.returncode}; see log')
        return r.stdout.strip()
    try:
        for name in ('run.sh', 'roofline.py', 'kernels.json', 'README.md'):
            shutil.copyfile(ROOT / name, out / name)
        protocol = plan(json.loads((ROOT / 'kernels.json').read_text()))
        save(out / 'plan.json', protocol)
        uuid = capture('gpu_uuid', ['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'])
        require(re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid), 'invalid GPU UUID')
        lock = open('/tmp/gemm-sweep-'+uuid+'.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not capture('processes_before', ['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader']), 'another GPU compute process is active')
        capture('device_before', ['nvidia-smi', '-i', '0', '-q'])
        preflight = capture('preflight', ['nvidia-smi', '-i', '0', '--query-gpu=power.limit,memory.free', '--format=csv,noheader,nounits'])
        limit, free = map(float, preflight.split(','))
        require(limit == 70 and free >= 8192, 'need existing 70 W limit and >=8192 MiB free VRAM')
        for repo, pin in PINS.items():
            require(capture(repo+'_commit', ['git', '-C', str(REMOTE/repo), 'rev-parse', 'HEAD']) == pin, repo+' source version changed')
            capture(repo+'_source_status', ['git', '-C', str(REMOTE/repo), 'status', '--porcelain'])
            require(not capture(repo+'_tracked_source_status', ['git', '-C', str(REMOTE/repo), 'status', '--porcelain', '--untracked-files=no']), repo+' tracked source files are modified')
        save(out / 'binary_hashes.json', {str(p): sha(p) for p in (CUTLASS, BABEL)})
        require(sha(CUTLASS) == '1e0608eb3eda54d329fd2cc017a28057aca952ae5732f8242aa5073abd7e8a4d', 'CUTLASS binary changed')
        require(sha(BABEL) == 'c465f7855e87a0a3de062ecb18812dc3d9b97569b94f041200902c45fc7790c5', 'BabelStream binary changed')
        capture('telemetry_probe', ['nvidia-smi', '-i', '0', '--query-gpu='+','.join(FIELDS), '--format=csv,noheader,nounits'])
        telemetry(out / 'telemetry_probe.txt')
        sp.run(['sudo', '-v'], check=True)
        capture('lock_sm', ['sudo', '-n', 'nvidia-smi', '-i', '0', '-lgc', '720,720'])
        capture('lock_mem', ['sudo', '-n', 'nvidia-smi', '-i', '0', '-lmc', '6801,6801'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
        save(out / 'environment.json', {k:env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'CUDA_MODULE_LOADING', 'CUDA_CACHE_DISABLE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS') if k in env})
        for i, job in enumerate(protocol['jobs'], 1):
            progress = f'[{i}/{len(protocol["jobs"])}] {job["tag"]}'
            print(progress, flush=True)
            with (out / 'progress.log').open('a') as f:
                f.write(dt.datetime.now().astimezone().isoformat()+' '+progress+'\n')
            checked = run_job(out, job, env)
            if checked['status'] != 'PASS':
                print('  REVIEW: 순간 전력/열 제한 표본 기록. 원자료를 유지하며 계속 수집.', flush=True)
        result = analyze(out)
        print((out / 'SUMMARY.txt').read_text(), flush=True)
        code = 0 if result['status'] == 'PASS' else 2
    except (Exception, SystemExit) as exc:
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        (out / 'failure.txt').write_text(traceback.format_exc())
        print(f'중단: {exc}. 원자료를 보존합니다.', flush=True)
        if (out / 'plan.json').exists():
            try:
                analyze(out)
            except Exception:
                (out / 'analysis_failure.txt').write_text(traceback.format_exc())
    finally:
        # Keep our GPU lock until final snapshots and the archive have been closed.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        try:
            capture('device_after', ['nvidia-smi', '-i', '0', '-q'])
        except Exception:
            (out / 'final_snapshot_failure.txt').write_text(traceback.format_exc())
            code = code or 1
        (out / 'runner.exit_code').write_text(str(code)+'\n')
        save(out / 'manifest.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'})
        archive = Path(str(out)+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as t:
            t.add(out, arcname=out.name)
        if lock:
            lock.close()
        print(f'보낼 파일: {archive} (종료코드 {code})', flush=True)
    return code


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--analyze':
        result = analyze(Path(sys.argv[2]))
        print(result['status'])
        sys.exit(0 if result['status'] == 'PASS' else 2)
    require(len(sys.argv) == 1, 'usage: bash run.sh OR python3 roofline.py --analyze RESULT_DIR')
    sys.exit(run())
