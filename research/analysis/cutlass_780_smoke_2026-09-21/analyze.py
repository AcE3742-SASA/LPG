"""Recheck this failed smoke archive: python3 -B analyze.py (no GPU work)."""
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent
raw = ROOT / 'raw'
archive = raw / 'cutlass_780_smoke.o9gaoQ.tar.gz'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == '714346ac831b780629ce9c9fa8535d54a955ca1b413e558b4dee606585fd86d6'
files = {}
with tarfile.open(archive) as bundle:
    for member in bundle:
        path = PurePosixPath(member.name)
        if member.isdir() and str(path) == '.':
            continue
        assert member.isfile() and len(path.parts) == 1 and not path.is_absolute()
        assert path.name not in files
        contents = bundle.extractfile(member).read()
        assert (raw / path.name).read_bytes() == contents
        files[path.name] = {'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
assert len(files) == 20
assert (raw / 'runner.exit_code').read_text().strip() == '1'
assert not (raw / 'monitor.err').read_text().strip()
windows = {}
for tag, event, when in csv.reader(io.StringIO((raw / 'timeline.tsv').read_text()), delimiter='\t'):
    assert event not in windows.setdefault(tag, {})
    windows[tag][event] = dt.datetime.fromisoformat(when.replace(',', '.'))
clocks = list(csv.DictReader(io.StringIO((raw / 'clocks.csv').read_text()), skipinitialspace=True))
number = lambda row, key: float(row[key].split()[0])
active = [r for r in clocks if number(r, 'utilization.gpu [%]') > 0]
busy = [r for r in clocks if number(r, 'utilization.gpu [%]') == 100]
assert len(clocks) == 26 and len(active) == 22 and len(busy) == 15
target = lambda r: tuple(number(r, key) for key in ('clocks.current.sm [MHz]', 'clocks.current.memory [MHz]', 'power.limit [W]')) == (780, 6801, 70)
results = {}
for tag, dtype, op in [('fp32_simt', 'f32', 'simt'), ('fp16_tensor', 'f16', 'tensorop')]:
    rows = list(csv.DictReader(io.StringIO((raw / (tag + '.gemm.csv')).read_text())))
    assert len(rows) == 1
    r = rows[0]
    assert r['Provider'] == 'CUTLASS' and r['Status'] == 'success'
    assert r['Disposition'] == 'not_verified'
    assert all(int(r[k]) == 8192 for k in ('m', 'n', 'k'))
    assert all(r[k].split(':')[0] == dtype for k in ('A', 'B', 'C', 'D'))
    assert r['accum'] == 'f32' and r['op_class'] == op
    assert int(r['Flops']) == 2 * 8192**3 + 2 * 8192**2
    assert (raw / (tag + '.exit_code')).read_text().strip() == '0'
    assert 'cuBLAS: Not run' in (raw / (tag + '.log')).read_text()
    runtime = float(r['Runtime'])
    assert math.isfinite(runtime) and runtime > 0
    assert math.isclose(float(r['GFLOPs']), int(r['Flops']) / runtime / 1e6, rel_tol=1e-5)
    window = windows[tag]
    assert window['start'] < window['end']
    inside = [c for c in busy if window['start'] <= dt.datetime.strptime(c['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=window['start'].tzinfo) <= window['end']]
    assert inside
    results[tag] = {'row': r, 'tflops_2mnk': 2 * 8192**3 / runtime / 1e9,
                    'fully_busy_sm_MHz': sorted({number(c, 'clocks.current.sm [MHz]') for c in inside})}
assert all(not target(r) for r in active)
summary = {'admission': 'FAIL', 'reasons': ['both_not_verified', 'all_active_clock_samples_off_target'],
           'archive_sha256': digest, 'files': files, 'results': results,
           'clock_samples': len(clocks), 'active_samples': len(active), 'fully_busy_samples': len(busy),
           'active_clock_mismatches': sum(not target(r) for r in active),
           'active_sm_MHz': sorted({number(r, 'clocks.current.sm [MHz]') for r in active}),
           'active_memory_MHz': sorted({number(r, 'clocks.current.memory [MHz]') for r in active}),
           'power_limit_W': sorted({number(r, 'power.limit [W]') for r in clocks}),
           'max_power_W': max(number(r, 'power.draw [W]') for r in clocks),
           'max_temperature_C': max(number(r, 'temperature.gpu') for r in clocks)}
(ROOT / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Archive/integrity/arithmetic checks PASS; 780 MHz smoke admission FAIL (verification and clocks).')
