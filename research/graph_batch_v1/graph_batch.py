#!/usr/bin/env python3
"""Unprofiled fixed-size CUDA Graph comparison; no automatic full-sweep admission."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

import audit_support as support

HERE = Path(__file__).resolve().parent
BATCHES = (16, 32, 64, 128)
# Each size occupies each position once; all 12 directed transitions occur once.
ORDERS = ((16, 32, 128, 64), (32, 64, 16, 128),
          (64, 128, 32, 16), (128, 16, 64, 32))
CASES = [dict(id='g00', shape=[64, 64, 64], dtype='fp16'),
         dict(id='g01', shape=[64, 64, 64], dtype='fp32'),
         dict(id='g02', shape=[128, 128, 128], dtype='fp16')]
PLATEAU_MARGIN = .01


def tasks():
    jobs = []
    for rd, order in enumerate(ORDERS):
        cases = CASES[rd % len(CASES):] + CASES[:rd % len(CASES)]
        for case in cases:
            for batch in order:
                jobs.append(dict(id=f"r{rd}_{case['id']}_b{batch}", case_id=case['id'],
                    kind='gemm', mode='graph', phase='graph_batch_v1', split='diagnostic_only',
                    shape=case['shape'], dtype=case['dtype'], round=rd, gemms_per_replay=batch))
    return jobs


def capture_fixed(torch, fn, batch):
    """The v5 capture sequence, with an exact GEMM count instead of adaptive sizing."""
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('Unsupported GEMMs per replay')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def analyze_rows(rows, measure, protocol):
    expected = {t['id']: t for t in tasks()}
    if len(rows) != len(expected) or {r['task']['id'] for r in rows} != set(expected):
        raise ValueError('Missing or duplicate measurements')
    grouped = {}
    addresses = {}
    for row in rows:
        task, result = row['task'], row['result']
        if task != expected[task['id']] or result['input_seed'] != measure.task_seed(task):
            raise ValueError('Task/seed differs from plan')
        if result['correctness']['passed'] is not True:
            raise ValueError('Correctness failure')
        timing, batch = result['timing'], task['gemms_per_replay']
        blocks = timing['blocks']
        if len(blocks) != protocol['blocks']:
            raise ValueError('Wrong block count')
        if not math.isfinite(timing['setup']['settle_event_ms']) or timing['setup']['settle_event_ms'] < protocol['settle_ms']:
            raise ValueError('Insufficient settling')
        for block in blocks:
            if (type(block['repeats']) is not int or not 1 <= block['repeats'] <= protocol['max_repeats']
                    or block['operations_per_call'] != batch):
                raise ValueError('Wrong GEMM count')
            elapsed, latency = block['elapsed_ms'], block['latency_ms']
            if (not all(math.isfinite(v) and v > 0 for v in (elapsed, latency))
                    or not math.isclose(latency, elapsed / (block['repeats'] * batch), rel_tol=1e-12)):
                raise ValueError('Invalid per-GEMM normalization')
        if result['measured_gemm_calls'] != sum(b['repeats'] * batch for b in blocks):
            raise ValueError('Wrong total GEMM count')
        address_key = (task['round'], task['case_id'])
        pointers = result['tensor_addresses']
        if len(pointers) != 3 or any(type(p) is not int or p <= 0 for p in pointers):
            raise ValueError('Invalid tensor addresses')
        if addresses.setdefault(address_key, pointers) != pointers:
            raise ValueError('Tensor addresses changed within a case/process')
        stats = measure.summarize(blocks, protocol)
        if any(b['elapsed_ms'] < .8 * protocol['block_ms'] for b in blocks):
            stats['flags'].append('short_block')
        if any(timing[k] != stats[k] for k in stats):
            raise ValueError('Stored summary differs from raw blocks')
        grouped.setdefault((task['case_id'], batch), {})[task['round']] = stats
    cases = []
    for case in CASES:
        sizes = []
        for batch in BATCHES:
            rounds = grouped[(case['id'], batch)]
            values = [rounds[rd]['median_ms'] for rd in range(len(ORDERS))]
            median = statistics.median(values)
            spread = (max(values) - min(values)) / median
            flags = [f"round_{rd}:{flag}" for rd in range(len(ORDERS)) for flag in rounds[rd]['flags']]
            if spread > protocol['between_round_limit']:
                flags.append('between_round_spread')
            sizes.append(dict(gemms_per_replay=batch, median_ms=median, round_medians_ms=values,
                between_round_spread=spread, max_relative_iqr=max(r['relative_iqr'] for r in rounds.values()),
                max_abs_drift=max(abs(r['first_last_drift']) for r in rounds.values()), flags=flags))
        pairs = []
        for lower, upper in itertools.combinations(sizes, 2):
            changes = [b / a - 1 for a, b in zip(lower['round_medians_ms'], upper['round_medians_ms'])]
            pairs.append(dict(lower=lower['gemms_per_replay'], upper=upper['gemms_per_replay'],
                              paired_relative_changes=changes, median_change=statistics.median(changes),
                              within_margin_all_rounds=all(abs(v) <= PLATEAU_MARGIN for v in changes)))
        stable = all(not size['flags'] for size in sizes)
        candidate = next((batch for batch in BATCHES[:-1] if stable and all(
            p['within_margin_all_rounds'] for p in pairs if p['lower'] >= batch)), None)
        cases.append(dict(case=case, sizes=sizes, pairs=pairs, stability='PASS' if stable else 'REVIEW',
                          candidate_gemms_per_replay=candidate,
                          conclusion=('REVIEW_STABILITY' if not stable else
                                      'OBSERVED_PLATEAU_CANDIDATE' if candidate else 'NO_OBSERVED_PLATEAU')))
    return dict(measurements=len(rows), blocks=sum(len(r['result']['timing']['blocks']) for r in rows),
                stability='PASS' if all(c['stability'] == 'PASS' for c in cases) else 'REVIEW',
                plateau_margin=PLATEAU_MARGIN, cases=cases,
                interpretation='Four process runs; observed equivalence within 1%, not a confidence bound, '
                               'proof of zero submission cost, pure-kernel time or full-sweep permission.')


def analyze(out, measure, plan, suite):
    rows = []
    for rd in range(len(ORDERS)):
        rows.extend(json.loads((out / f'round_{rd}' / 'results.json').read_text()))
    result = analyze_rows(rows, measure, plan.PROTOCOL)
    suite.atomic_json(out / 'analysis.json', result)
    lines = ['# Graph 묶음 수 비교', '',
             '시간은 각 프로세스의 7블록 중앙값을 구한 뒤 4개 프로세스의 중앙값으로 집계했다.', '',
             '| 조건 | 묶음 | GEMM당 시간 (μs) | 실행 간 범위/중앙값 | 안정성 |',
             '|---|---:|---:|---:|---|']
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'M', 'N', 'K', 'dtype', 'gemms_per_replay', 'median_us',
                         'between_round_spread', 'max_relative_iqr', 'max_abs_drift', 'flags'])
        for case in result['cases']:
            spec = case['case']
            label = '×'.join(map(str, spec['shape'])) + ' ' + spec['dtype']
            for size in case['sizes']:
                writer.writerow([spec['id'], *spec['shape'], spec['dtype'], size['gemms_per_replay'],
                    size['median_ms']*1000, size['between_round_spread'], size['max_relative_iqr'],
                    size['max_abs_drift'], ';'.join(size['flags'])])
                lines.append(f"| {label} | {size['gemms_per_replay']} | {size['median_ms']*1000:.3g} | "
                             f"{size['between_round_spread']:.2%} | {'REVIEW' if size['flags'] else 'PASS'} |")
    lines += ['', '## 같은 실행에서 묶음을 늘렸을 때의 변화', '',
              '| 조건 | 비교 | 중앙 변화 | 4회 변화 범위 | 매회 ±1% 이내 |', '|---|---|---:|---:|---|']
    for case in result['cases']:
        for pair in case['pairs']:
            changes = pair['paired_relative_changes']
            lines.append(f"| {case['case']['id']} | {pair['lower']}→{pair['upper']} | {pair['median_change']:+.2%} | "
                         f"{min(changes):+.2%} ~ {max(changes):+.2%} | {pair['within_margin_all_rounds']} |")
        lines += ['', f"{case['case']['id']}: {case['conclusion']}; "
                   f"후보 묶음 수 = {case['candidate_gemms_per_replay']}", '']
    lines += ['1%는 이번 진단에 미리 지정한 실용 비교 기준이다. 신뢰구간이나 순수 GPU 시간 보증이 아니다.',
              '128회는 더 큰 묶음과 비교하지 않았으므로 단독으로 충분하다고 판정하지 않는다.',
              '동일 입력·출력 버퍼를 반복 재사용하는 조건이며 cold-cache latency나 전체 형상으로 일반화하지 않는다.',
              '실행 성공·안정성·묶음 수 후보를 구분하며 full은 자동 실행하지 않는다.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return result


def worker(args):
    expected, plan, measure, suite = support.load_suite(args.suite)
    import torch
    out = args.output / f'round_{args.round}'
    out.mkdir()
    env = suite.environment(torch)
    suite.atomic_json(out / 'environment.json', env)
    if suite.signature(env) != expected['environment_signature']:
        raise RuntimeError('GPU/software/precision differs from v5; inspect environment.json')
    if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError('Less than 4 GiB free GPU memory')
    support.check_processes(suite, env['gpu_uuid'], out / 'processes_before.json')
    rows = []
    jobs = [t for t in tasks() if t['round'] == args.round]
    with torch.inference_mode():
        timer = measure.Timer(torch, plan.PROTOCOL)
        x = torch.randn((1024, 1024), device='cuda') * .01
        y = torch.empty_like(x)
        started = time.perf_counter()
        print(f"round {args.round}: preheating {plan.PROTOCOL['preheat_seconds']} s", flush=True)
        while time.perf_counter() - started < plan.PROTOCOL['preheat_seconds']:
            timer.block(lambda: torch.mm(x, x, out=y), 100)
        del x, y
        for _, group in itertools.groupby(jobs, key=lambda t: t['case_id']):
            group = list(group)
            a, b, output, fn = measure.make_gemm(torch, group[0])
            # Keep the same tensors/addresses across all four graph sizes in this case.
            for task in group:
                support.check_processes(suite, env['gpu_uuid'], out / f"processes_{task['id']}.json")
                batch = task['gemms_per_replay']
                print(f"[{len(rows)+1}/{len(jobs)}] {task['id']} {task['shape']} {task['dtype']}", flush=True)
                capture_start = time.perf_counter()
                graph = capture_fixed(torch, fn, batch)
                capture_wall_ms = (time.perf_counter() - capture_start) * 1000
                timing = timer.run(graph.replay, batch)
                check = measure.check_mm(torch, a, b, output[0], task['dtype'], plan.task_seed(task), plan.PROTOCOL)
                rows.append(dict(task=task, result=dict(timing=timing, correctness=check,
                    input_seed=plan.task_seed(task), capture_and_warmup_wall_ms=capture_wall_ms,
                    tensor_addresses=[t.data_ptr() for t in (a, b, output[0])],
                    measured_gemm_calls=sum(block['repeats'] * batch for block in timing['blocks']))))
                suite.atomic_json(out / 'results.json', rows)
                del graph
                if not check['passed']:
                    raise RuntimeError('Correctness failed: ' + task['id'])
                print(f"  {timing['median_ms']*1000:.3g} us/GEMM; flags={timing['flags']}", flush=True)
            del a, b, output, fn
    support.check_processes(suite, env['gpu_uuid'], out / 'processes_after.json')
    suite.atomic_json(out / 'gpu_after.json', suite.command_text(['nvidia-smi', '-q']))


def run(args):
    expected, plan, measure, suite = support.load_suite(args.suite)
    out = args.output
    if (out / 'manifest.json').exists():
        raise RuntimeError('Existing output; use a new directory')
    out.mkdir(parents=True, exist_ok=True)
    snapshot = suite.command_text(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
    lines = snapshot.get('stdout', '').strip().splitlines()
    gpu_uuid = expected['environment_signature']['gpu_uuid']
    if snapshot.get('exit_code') != 0 or lines != [gpu_uuid]:
        raise RuntimeError('Expected the single v5 GPU; cannot acquire shared GPU lock')
    source_hashes = {p.name: support.sha256_file(p) for p in HERE.iterdir()
                     if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md')}
    suite.atomic_json(out / 'manifest.json', dict(version=1, diagnostic_only=True, tasks=tasks(),
        baseline=expected, source_hashes=source_hashes, profiler_enabled=False,
        capture_policy='exact GEMMs per replay; replaces adaptive graph sizing only',
        plateau_margin=PLATEAU_MARGIN, process_rounds=len(ORDERS),
        selection_rule='All pairwise differences among candidate and larger sizes within 1% in every '
                       'process round, with all timing/round stability gates passing. No 128-only candidate.'))
    for src, dest, names in ((HERE, out / 'source', source_hashes),
                             (args.suite, out / 'baseline_source', expected['code_hashes'])):
        for name in names:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, target)
    status = dict(collection='RUNNING', stability='NOT_ASSESSED', full_authorized=False)
    try:
        with suite.lock(Path('/tmp') / f'gemm-sweep-{gpu_uuid}.lock'):
            support.check_processes(suite, gpu_uuid, out / 'processes_before.json')
            with (out / 'telemetry.csv').open('w') as log:
                telemetry = subprocess.Popen(['nvidia-smi', '-i', gpu_uuid,
                    '--query-gpu=timestamp,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,power.limit',
                    '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
                try:
                    for rd in range(len(ORDERS)):
                        if telemetry.poll() is not None:
                            raise RuntimeError('Telemetry stopped')
                        print(f"Starting round {rd+1}/{len(ORDERS)}; progress: {out / f'round_{rd}.log'}", flush=True)
                        support.run_logged([sys.executable, '-u', '-B', str(HERE / 'graph_batch.py'),
                            'worker', '--suite', str(args.suite), '--output', str(out), '--round', str(rd)],
                            out, f'round_{rd}', timeout=1200)
                    if telemetry.poll() is not None:
                        raise RuntimeError('Telemetry stopped')
                finally:
                    telemetry.terminate()
                    try:
                        telemetry.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        telemetry.kill()
                        telemetry.wait()
            result = analyze(out, measure, plan, suite)
            support.verify_source(args.suite)
            if any(support.sha256_file(HERE / name) != digest for name, digest in source_hashes.items()):
                raise RuntimeError('Diagnostic source changed during execution')
            status.update(collection='COMPLETE', stability=result['stability'],
                          candidates={c['case']['id']: c['candidate_gemms_per_replay'] for c in result['cases']})
            print('COLLECTION COMPLETE', flush=True)
            print('STABILITY: ' + result['stability'], flush=True)
            print('BATCH CANDIDATES: ' + json.dumps(status['candidates']), flush=True)
            print('Interpret REPORT.md before changing the measurement protocol.', flush=True)
    except BaseException as error:
        status.update(collection='FAILED_OR_INTERRUPTED', error=str(error))
        raise
    finally:
        suite.atomic_json(out / 'status.json', status)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'run', 'worker', 'analyze', 'pack'))
    parser.add_argument('--suite', type=Path, default=HERE.parent / 'gpu_sweep')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--round', type=int, choices=range(len(ORDERS)))
    args = parser.parse_args()
    args.suite = args.suite.resolve()
    if args.action != 'verify' and args.output is None:
        parser.error('--output is required')
    if args.action == 'worker' and args.round is None:
        parser.error('--round is required for worker')
    if args.output:
        args.output = args.output.resolve()
    if args.action == 'verify':
        support.verify_source(args.suite)
        print('V5 SOURCE CHECK: PASS')
    elif args.action == 'pack':
        support.package(args.output)
    elif args.action == 'worker':
        worker(args)
    elif args.action == 'analyze':
        _, plan, measure, suite = support.load_suite(args.suite)
        analyze(args.output, measure, plan, suite)
    else:
        run(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
