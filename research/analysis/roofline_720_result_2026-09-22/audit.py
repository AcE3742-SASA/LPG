"""Read-only audit of originals, followed by offline replay in a temporary directory."""
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics as st
import tempfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
PACKAGE = ROOT.parents[1] / 'roofline_720_v1'
ARCHIVE = RAW / 'roofline_720_eif_37o1.tar.gz'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ARCHIVE) == '52351b316dd022a027a5e7f8bf7c1bc2a86daae5d4536325d4711c5c0b262f31'
manifest = json.loads((RAW / 'manifest.json').read_text())
assert len(manifest) == 2470 and all(sha(RAW/n) == h for n,h in manifest.items())
for name in ('roofline.py', 'run.sh', 'kernels.json', 'README.md'):
    assert (RAW/name).read_bytes() == (PACKAGE/name).read_bytes(), name
spec = importlib.util.spec_from_file_location('trusted_roofline_720_v1', PACKAGE/'roofline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
protocol = json.loads((RAW/'plan.json').read_text())
assert protocol == m.plan(json.loads((PACKAGE/'kernels.json').read_text()))
assert (RAW/'runner.exit_code').read_text().strip() == '0'
assert not (RAW/'failure.txt').exists()
for repo,pin in m.PINS.items():
    assert (RAW/(repo+'_commit.txt')).read_text().strip() == pin
    assert not (RAW/(repo+'_tracked_source_status.txt')).read_text().strip()
recomputed = ROOT/'recomputed'; recomputed.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='roofline-replay-') as temporary:
    work = Path(temporary)
    for name in manifest:
        if name not in ('analysis.json','summary.csv','SUMMARY.txt'):
            (work/name).symlink_to((RAW/name).resolve())
    result = m.analyze(work)
    assert result['status'] == 'PASS' and result['collection_complete']
    for name in ('analysis.json','summary.csv','SUMMARY.txt'):
        assert (work/name).read_bytes() == (RAW/name).read_bytes(), name
        shutil.copyfile(work/name, recomputed/name)
assert len(result['summaries']) == 63
assert all(r['repeats'] == 5 for r in result['summaries'])

runs, observations, times = [], [], []
previous_end = None
for job in protocol['jobs']:
    tag = job['tag']
    args = json.loads((RAW/(tag+'.command.json')).read_text())
    assert args == m.command(job, Path('/home/eu/roofline_720_eif_37o1'))
    checked = m.check_run(RAW,job)
    assert checked == json.loads((RAW/(tag+'.check.json')).read_text()), tag
    timing = json.loads((RAW/(tag+'.timing.json')).read_text())
    start,end = [dt.datetime.fromisoformat(timing[k]) for k in ('start','end')]
    assert start.tzinfo is not None and end > start
    if previous_end is not None: assert previous_end <= start
    previous_end = end; times.extend((start,end))
    clocks = m.telemetry(RAW/(tag+'.clocks.csv'))
    for c in clocks:
        c['timestamp'] = c['timestamp'].replace(tzinfo=start.tzinfo)
    active = [c for c in clocks if c['util'] > 0]
    assert all([c['sm'],c['mem'],c['limit']] == [720,6801,70] for c in active)
    assert not any(any(c['reasons']) for c in active)
    runs.append(dict(tag=tag,round=job['round'],kind=job['kind'],size=job['size'],
                     start=start.isoformat(),end=end.isoformat(),wall_s=(end-start).total_seconds(),
                     active_samples=len(active),samples=len(clocks),
                     monitor_lead_s=(start-clocks[0]['timestamp']).total_seconds(),
                     monitor_tail_s=(clocks[-1]['timestamp']-end).total_seconds(),
                     max_gap_s=checked['max_gap_s'],counter_delta_us=checked['power_cap_counter_delta_us']))
    observations.extend(clocks)
    for v in checked['values']:
        if job['round']:
            assert v['value'] > 0

candidate_details = {}
for path,candidate in result['ceiling_candidates'].items():
    matching = []
    for job in protocol['jobs']:
        if job['round'] == 0 or job['size'] != candidate['size']: continue
        checked = json.loads((RAW/(job['tag']+'.check.json')).read_text())
        for v in checked['values']:
            if v['path']==path and v['function']==candidate['function']:
                matching.append(dict(round=job['round'],value=v['value'],tag=job['tag']))
    assert len(matching)==5
    med = st.median(v['value'] for v in matching)
    assert med == candidate['median']
    candidate_details[path] = dict(candidate=candidate,repeats=matching,
        same_kernel_by_size=[s for s in result['summaries'] if s['path']==path and s['function']==candidate['function']])

def counters(name):
    text = (RAW/name).read_text()
    block = text.split('Clocks Event Reasons Counters',1)[1].split('Sparse Operation Mode',1)[0]
    return {k:int(re.search(re.escape(k)+r'\s*:\s*(\d+) us',block).group(1))
            for k in ('SW Power Capping','SW Thermal Slowdown','HW Thermal Slowdown','HW Power Braking')}

before,after = counters('device_before.txt'),counters('device_after.txt')
active = [c for c in observations if c['util']>0]
report = dict(status='AUDIT_PASS',archive_sha256=sha(ARCHIVE),original_files=2471,
              manifest_files_verified=2470,source_files_match=4,replay_outputs_byte_identical=True,
              calls=len(runs),gemm_calls=sum(r['kind']=='gemm' for r in runs),bw_calls=sum(r['kind']=='bw' for r in runs),
              measured_calls=sum(r['round']>0 for r in runs),groups=len(result['summaries']),
              start=min(times).isoformat(),end=max(times).isoformat(),wall_seconds=(max(times)-min(times)).total_seconds(),
              telemetry=dict(samples=len(observations),active_samples=len(active),fully_busy_samples=sum(c['util']==100 for c in observations),
                  all_sm_mhz=sorted({c['sm'] for c in observations}),all_mem_mhz=sorted({c['mem'] for c in observations}),
                  all_power_limits_W=sorted({c['limit'] for c in observations}),loaded_throttle_samples=sum(any(c['reasons']) for c in active),
                  all_throttle_samples=sum(any(c['reasons']) for c in observations),
                  temperature_range_C=[min(c['temp'] for c in observations),max(c['temp'] for c in observations)],
                  max_power_W=max(c['power'] for c in observations),max_sample_gap_s=max(r['max_gap_s'] for r in runs)),
              counter_delta_us={k:after[k]-before[k] for k in before},
              counter_sum_within_calls_us=result['total_power_cap_counter_delta_us'],
              max_group_range_pct=max(s['range_pct'] for s in result['summaries']),
              candidates=candidate_details,runs=runs)
(ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
print(json.dumps({k:v for k,v in report.items() if k not in ('runs','candidates')},indent=2))
for path,data in candidate_details.items():
    print(path,[(r['size'],r['median'],r['range_pct']) for r in data['same_kernel_by_size']])
