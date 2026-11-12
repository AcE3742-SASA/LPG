"""Audit and preserve the uploaded v3 smoke. Run: python3 -B analyze.py ARCHIVE"""
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
archive = Path(sys.argv[1])
blob = archive.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
assert digest == 'fabf311ab178a9f3174335dcf4950f3f21857c8169602b8686b71b0ccff444fb'
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
assert len(data) == 28
for name, contents in {archive.name: blob, **data}.items():
    target = raw / name
    if target.exists():
        assert target.read_bytes() == contents
    else:
        target.write_bytes(contents)
text = {name: contents.decode() for name, contents in data.items()}
assert text['runner.exit_code'].strip() == '0'
assert not text['source_status.txt'].strip() and not text['processes_before.txt'].strip()
assert hashlib.sha256(data['runner.sh']).hexdigest() == '28e0ab3e200843063b54fb249c1af1b36ec9e5e1b9e88769bd52e20b559d2b35'
assert text['commit.txt'].strip() == '147295a3d4b75f3aeff247c25b8927cea9a7006a'
assert text['binary.sha256'].split()[0] == '1e0608eb3eda54d329fd2cc017a28057aca952ae5732f8242aa5073abd7e8a4d'
windows = {}
for tag, event, value in csv.reader(io.StringIO(text['timeline.tsv']), delimiter='\t'):
    assert event not in windows.setdefault(tag, {})
    windows[tag][event] = dt.datetime.fromisoformat(value.replace(',', '.'))
results = {}
for tag, dtype, op in [('fp32_simt', 'f32', 'simt'), ('fp16_tensor', 'f16', 'tensorop')]:
    rows = list(csv.DictReader(io.StringIO(text[tag + '.gemm.csv'])))
    assert len(rows) == 1
    r = rows[0]
    assert r['Provider'] == 'CUTLASS' and r['Status'] == 'success' and r['Disposition'] == 'passed'
    assert all(int(r[k]) == 8192 for k in ('m', 'n', 'k'))
    assert r['op_class'] == op and r['accum'] == 'f32'
    assert [r[k] for k in ('A', 'B', 'C', 'D')] == [dtype+':row', dtype+':column', dtype+':column', dtype+':column']
    assert (r['alpha'], r['beta'], r['split_k_slices'], r['batch_count']) == ('1', '0', '1', '1')
    assert text[tag + '.exit_code'].strip() == '0' and not text[tag + '.monitor.err'].strip()
    assert 'reference_device: Passed' in text[tag + '.log']
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
        max_poll_gap_s=max(gaps), profiler_wall_s=(windows[tag]['end']-windows[tag]['start']).total_seconds())
counters = {}
for name in ('device_before.txt', 'fp32_simt.device_after.txt', 'fp16_tensor.device_after.txt'):
    counters[name] = int(re.search(r'SW Power Capping\s*:\s*(\d+) us', text[name]).group(1))
summary = dict(status='SMOKE_PASS', final_compute_ceiling=False, independent_runs_per_mode=1,
    archive_sha256=digest, files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in data.items()},
    results=results, sw_power_capping_counter_us=counters,
    sw_power_capping_counter_delta_us=counters['fp16_tensor.device_after.txt']-counters['device_before.txt'])
(ROOT / 'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
print('PASS: original bytes, correctness provider, CSV/check agreement, 2MNK arithmetic, and all 107 active clock samples. Final ceilings pending.')
