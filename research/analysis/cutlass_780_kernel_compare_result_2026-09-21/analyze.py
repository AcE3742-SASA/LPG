"""Audit and preserve the uploaded 16-kernel comparison. Run: python3 -B analyze.py ARCHIVE"""
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import shlex
import tarfile

ROOT = Path(__file__).resolve().parent
archive = Path(sys.argv[1])
blob = archive.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
assert digest == '8e11151ffcad3c41282cbf6a8bf039160c506acb005c34736aead1a93ef7ef3d'
raw = ROOT / 'raw'
raw.mkdir(exist_ok=True)
data = {}
with tarfile.open(archive) as bundle:
    for member in bundle:
        p = PurePosixPath(member.name)
        if member.isdir() and str(p) == '.':
            continue
        assert member.isfile() and len(p.parts) == 1 and not p.is_absolute() and '..' not in p.parts
        assert p.name not in data and member.size < 1_000_000
        data[p.name] = bundle.extractfile(member).read()
assert len(data) == 154
for name, contents in {archive.name: blob, **data}.items():
    target = raw / name
    if target.exists():
        assert target.read_bytes() == contents
    else:
        target.write_bytes(contents)
text = {name: contents.decode() for name, contents in data.items()}
assert text['runner.exit_code'].strip() == '0'
assert not text['source_status.txt'].strip() and not text['processes_before.txt'].strip()
assert hashlib.sha256(data['runner.sh']).hexdigest() == '841895e71a83f35764f61667a1c34caa7164a71624bd68603f592bbd0abb70c2'
assert text['commit.txt'].strip() == '147295a3d4b75f3aeff247c25b8927cea9a7006a'
assert text['binary.sha256'].split()[0] == '1e0608eb3eda54d329fd2cc017a28057aca952ae5732f8242aa5073abd7e8a4d'
windows = {}
for tag, event, value in csv.reader(io.StringIO(text['timeline.tsv']), delimiter='\t'):
    assert event not in windows.setdefault(tag, {})
    windows[tag][event] = dt.datetime.fromisoformat(value.replace(',', '.'))
expected = json.loads((ROOT / 'kernels.json').read_text())
assert len(expected) == 16 and set(windows) == {x['tag'] for x in expected}
assert set(n for n in data if n.endswith('.gemm.csv')) == {x['tag']+'.gemm.csv' for x in expected}
results = {}
for spec in expected:
    tag, dtype, op = spec['tag'], spec['dtype'], spec['op']
    assert set(windows[tag]) == {'start', 'end'}
    assert windows[tag]['end'] > windows[tag]['start']
    rows = list(csv.DictReader(io.StringIO(text[tag + '.gemm.csv'])))
    assert len(rows) == 1
    r = rows[0]
    assert r['Provider'] == 'CUTLASS' and r['Status'] == 'success' and r['Disposition'] == 'passed'
    assert all(int(r[k]) == 8192 for k in ('m', 'n', 'k'))
    assert r['op_class'] == op and r['accum'] == 'f32'
    assert r['Operation'] == spec['kernel']
    assert [r[k] for k in ('A', 'B', 'C', 'D')] == [spec[k] for k in ('A', 'B', 'C', 'D')]
    assert (r['alpha'], r['beta'], r['split_k_slices'], r['batch_count']) == ('1', '0', '1', '1')
    assert text[tag + '.exit_code'].strip() == '0' and not text[tag + '.monitor.err'].strip()
    assert 'reference_device: Passed' in text[tag + '.log']
    command = shlex.split(text[tag + '.command.txt'])
    assert command[0] == '/home/eu/cutlass/build/tools/profiler/cutlass_profiler'
    args = dict(a[2:].split('=', 1) for a in command[1:])
    required = {'operation':'gemm','mode':'profile','devices':'0','m':'8192','n':'8192','k':'8192','alpha':'1','beta':'0','batch_count':'1','split_k_slices':'1','dist':'uniform,min:-2,max:2','seed':'20260921','workspace-count':'1','warmup-iterations':'10','profiling-iterations':'0','profiling-duration':'1000','min-iterations':'10','verification-enabled':'true','verification-providers':'device','epsilon':'0','nonzero-floor':'0'}
    assert all(args[k] == v for k, v in required.items()), tag
    assert args['kernels'] == r['Operation']
    assert all(args[k] == r[k] for k in ('A','B','C','D','op_class','accum'))
    ms = float(r['Runtime'])
    assert math.isfinite(ms) and ms > 0
    assert int(r['Flops']) == 2*8192**3 + 2*8192**2
    assert math.isclose(float(r['GFLOPs']), int(r['Flops'])/ms/1e6, rel_tol=1e-5)
    tflops = 2*8192**3/ms/1e9
    check = json.loads(text[tag + '.check.json'])
    assert check[tag]['row'] == r and check[tag]['tflops_2mnk'] == tflops
    assert all(check[k] is True for k in ('correctness_pass', 'clocks_pass', 'monitor_pass'))
    clocks = list(csv.DictReader(io.StringIO(text[tag + '.clocks.csv']), skipinitialspace=True))
    number = lambda c, key: float(c[key].split()[0])
    active = [c for c in clocks if number(c, 'utilization.gpu [%]') > 0]
    assert len(active) == check['active_clock_samples'] and len(active) >= 2
    assert all(tuple(number(c, k) for k in ('clocks.current.sm [MHz]', 'clocks.current.memory [MHz]', 'power.limit [W]')) == (780, 6801, 70) for c in active)
    times = [dt.datetime.strptime(c['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=windows[tag]['start'].tzinfo) for c in clocks]
    gaps = [(b-a).total_seconds() for a,b in zip(times, times[1:])]
    assert min(gaps) > 0 and windows[tag]['start'] <= times[0] < times[-1] <= windows[tag]['end']
    results[tag] = dict(row=r, runtime_ms=ms, tflops_2mnk=tflops, samples=len(clocks),
        active_samples=len(active), fully_busy_samples=sum(number(c, 'utilization.gpu [%]') == 100 for c in clocks),
        max_power_W=max(number(c, 'power.draw [W]') for c in clocks),
        temperature_range_C=[min(number(c, 'temperature.gpu') for c in clocks), max(number(c, 'temperature.gpu') for c in clocks)],
        all_sm_mhz=sorted({number(c, 'clocks.current.sm [MHz]') for c in clocks}),
        all_memory_mhz=sorted({number(c, 'clocks.current.memory [MHz]') for c in clocks}),
        max_poll_gap_s=max(gaps), profiler_wall_s=(windows[tag]['end']-windows[tag]['start']).total_seconds())
counters = {}
for name in ['device_before.txt'] + [x['tag']+'.device_after.txt' for x in expected]:
    counters[name] = int(re.search(r'SW Power Capping\s*:\s*(\d+) us', text[name]).group(1))
values=list(counters.values())
assert all(b >= a for a, b in zip(values, values[1:]))
best={op:max((tag for tag, row in results.items() if row['row']['op_class']==op), key=lambda tag:results[tag]['tflops_2mnk']) for op in ('simt','tensorop')}
summary = dict(status='COMPARISON_PASS_AT_780', target_sm_mhz=780, target_memory_mhz=6801,
    target_power_limit_W=70, current_roofline_target_sm_mhz=750, valid_for_750=False,
    final_compute_ceiling=False, independent_runs_per_kernel=1,
    archive_sha256=digest, files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in data.items()},
    results=results, best_observed_tags=best, sw_power_capping_counter_us=counters,
    sw_power_capping_counter_delta_us=values[-1]-values[0],
    totals=dict(samples=sum(r['samples'] for r in results.values()),
      active_samples=sum(r['active_samples'] for r in results.values()),
      fully_busy_samples=sum(r['fully_busy_samples'] for r in results.values()),
      max_power_W=max(r['max_power_W'] for r in results.values()),
      temperature_range_C=[min(r['temperature_range_C'][0] for r in results.values()), max(r['temperature_range_C'][1] for r in results.values())],
      wall_s=(windows[expected[-1]['tag']]['end']-windows[expected[0]['tag']]['start']).total_seconds()))
(ROOT / 'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
with (ROOT/'summary.csv').open('w',newline='') as f:
    fields=['tag','kernel','runtime_ms','tflops_2mnk','active_samples','samples','max_power_W','max_poll_gap_s']
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    for tag,r in results.items():
        writer.writerow(dict(tag=tag,kernel=r['row']['Operation'],**{k:r[k] for k in fields[2:]}))
print('PASS: original hashes, 16 exact kernels/layouts, device correctness, CSV/check agreement, 2MNK arithmetic, and active clocks at 780. Not 750 data; final ceilings pending.')
print(json.dumps(dict(best=best,totals=summary['totals'],counter_delta_us=summary['sw_power_capping_counter_delta_us']), indent=2))
for tag,r in results.items():
    print(f"{tag}: {r['runtime_ms']:.2f} ms, {r['tflops_2mnk']:.2f} TFLOP/s, {r['active_samples']}/{r['samples']} active")
