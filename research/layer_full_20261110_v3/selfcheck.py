"""CPU checks for shape protection, real quality analysis, sealed access and recovery."""
import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import build_manifest as bm
import collect_all
import compare
import diagnostic
from layers import correctness, make_pair


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def static_checks():
    import ast
    old_layers = ast.parse((compare.HERE/'previous_layers.py').read_text())
    new_layers = ast.parse((compare.HERE/'layers.py').read_text())
    for name in ('seed_for', 'make_pair', 'indices', 'tensor_identity'):
        nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for tree in (old_layers, new_layers)]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    v2_layers = ast.parse((compare.HERE/'v2_layers.py').read_text())
    for name in ('seed_for', 'make_pair', 'indices', 'sampled_reference', 'tensor_identity'):
        nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for tree in (v2_layers, new_layers)]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    assert compare.read(compare.HERE/'protocol.json')['tolerances'] == compare.read(
        compare.HERE/'inherited_v2_protocol.json')['tolerances']
    old_protocol = compare.read(compare.HERE/'previous_protocol.json')
    new_protocol = compare.read(compare.HERE/'protocol.json')
    assert {k: v for k, v in new_protocol.items() if k not in ('version', 'correctness_reference')} == {
        k: v for k, v in old_protocol.items() if k != 'version'}
    groups, phases, protected, audit = bm.build()
    for name, value in [('groups.json', groups), ('cases_by_phase.json', phases),
                        ('protected_shapes.json', protected), ('selection_audit.json', audit),
                        ('training_subsets.json', bm.training_subsets(groups))]:
        assert json.loads(json.dumps(value)) == compare.read(compare.HERE/name), name
    reserved, owners, case_ids = set(map(tuple, protected)), {}, set()
    for family in ('core', 'conv'):
        selected = [g for g in groups if g['family'] == family]
        assert Counter(g['split'] for g in selected) == dict(bm.SPLITS)
        assert set(Counter(g['stratum'] for g in selected).values()) == {25}
    for group in groups:
        for case in bm.make_cases(group):
            assert case['id'] not in case_ids
            case_ids.add(case['id'])
            for shape in bm.components(case):
                assert 0 < min(shape) <= max(shape) <= 4096
                assert not bm.overlaps([shape], reserved)
                owners.setdefault(shape, set()).add(group['group_id'])
    assert len(case_ids) == 1200
    for shape, ids in owners.items():
        assert len(ids) == 1
        assert all(owners[n] == ids for n in bm.neighbours(shape) if n in owners)
    counts = Counter()
    for phase, cases in phases.items():
        config = dict(compare.read(compare.HERE/'protocol.json'), phase=phase)
        assert compare.expected_records(cases, config) == 1080
        assert Counter(c['layer'] for c in cases) == dict.fromkeys(('linear', 'ffn', 'conv1', 'conv3'), 10)
        counts.update((c['layer'], c['split']) for c in cases)
    assert all(counts[layer, split] == count for layer in ('linear', 'ffn', 'conv1', 'conv3')
               for split, count in bm.SPLITS)
    subsets = bm.training_subsets(groups)
    for family, seeds in subsets['families'].items():
        allowed = {g['group_id'] for g in groups if g['family'] == family and g['split'] == 'train'}
        for seed in seeds.values():
            previous = set()
            for k in bm.K_VALUES:
                ids = seed['by_k'][str(k)]
                assert len(ids) == len(set(ids)) == k
                assert previous <= set(ids) <= allowed
                previous = set(ids)
    provenance = compare.read(compare.HERE/'source_provenance.json')
    for name, digest in provenance['source_hashes'].items():
        if name.startswith('dependencies/'):
            assert compare.sha(compare.HERE/name) == digest
    assert compare.sha(compare.HERE/'previous_layers.py') == provenance['source_hashes']['layers.py']
    for file in list(compare.HERE.glob('*.py')) + list(compare.HERE.glob('*.sh')):
        assert file.read_text().isascii(), file
    for phase in phases:
        compare.PHASE, compare.LOCKED_RELEASE = phase, True
        compare.verify()
    compare.PHASE, compare.LOCKED_RELEASE = 'test_01', False
    try:
        compare.verify()
    except RuntimeError as error:
        assert 'Sealed test' in str(error)
    else:
        raise AssertionError('Sealed test unexpectedly accessible')
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        write(out/'manifest.json', {'phase': 'test_01'})
        (out/'records').mkdir()
        (out/'records/unread.json').write_bytes(b'INVALID: MUST NOT BE PARSED OR HASHED')
        for phase in ('train_01', 'test_01'):
            result = subprocess.run([sys.executable, '-B', str(compare.HERE/'compare.py'), 'pack',
                                     '--output', str(out), '--phase', phase], capture_output=True, text=True)
            assert result.returncode != 0 and 'Sealed test access' in result.stderr
        assert not (out/'.run.lock').exists() and not (out/'result_hashes.json').exists()
    compare.PHASE, compare.LOCKED_RELEASE = 'train_01', False
    return compare.verify()


def analyzer_checks(config, cases, telemetry):
    sensor = {'clocks.sm': 720, 'clocks.mem': 6801, 'clocks_event_reasons.sw_power_cap': False,
              'utc': '2026-11-04T00:00:00.500000+00:00', **dict.fromkeys(telemetry.COUNTERS, 0)}
    rows = [dict(sensor, utc=f'2026-11-04T00:00:0{i}.500000+00:00') for i in range(5)]
    def window(subset, start, end):
        actual = telemetry.window(subset, start, end)
        assert actual == telemetry.window(rows, start, end)
        return actual
    loaded = dict(suite=SimpleNamespace(atomic_json=write),
                  telemetry=SimpleNamespace(read=lambda *args: (rows, {}), window=window))
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        write(out/'worker/telemetry_metadata.json', {'timezone_offset_seconds': 0})
        write(out/'correct.json', {'passed': True})
        write(out/'status.json', {'collection': 'COMPLETE', 'restoration': 'COMMANDS_ACCEPTED'})
        write(out/'worker/complete.json', {'complete': True})
        blocks = [dict(host_envelope_start_utc='2026-11-04T00:00:01+00:00',
                       host_envelope_end_utc='2026-11-04T00:00:03+00:00') for _ in range(7)]
        for case in cases:
            for dtype in config['dtypes']:
                for rd in range(3):
                    for implementation in compare.implementations(case, config):
                        write(compare.record_path(out, case, dtype, rd, implementation),
                              dict(id=compare.key(case, dtype, rd, implementation), case=case, dtype=dtype,
                                   timing=dict(median_ms=1 if implementation == 'standard' else 2,
                                               flags=[], blocks=blocks),
                                   memory=dict(extra_peak_allocated_bytes=10), prepared_weight_bytes=20,
                                   correctness_before='correct.json', correctness_after='correct.json'))
        result = compare.analyze(out, config, cases, loaded)
        assert result['gate'] == 'PASS' and result['recorded'] == 1080
        assert result['comparisons'] == []
        assert 'standard_ms' not in (out/'summary.json').read_text()
        assert compare.analyze(out, config, cases, loaded, health_only=True)['gate'] == 'PASS'
        rows[2]['clocks.sm'] = 705
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'REVIEW'
        rows[2]['clocks.sm'] = 720
        write(out/'correct.json', {'passed': False})
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'FAIL'
        write(out/'correct.json', {'passed': True})
        write(out/'status.json', {'collection': 'COMPLETE', 'restoration': 'FAILED'})
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'FAIL'
        write(out/'status.json', {'collection': 'COMPLETE', 'restoration': 'COMMANDS_ACCEPTED'})
        ffn = next(c for c in cases if c['layer'] == 'ffn')
        path = compare.record_path(out, ffn, 'fp16', 2, 'gelu')
        row = compare.read(path)
        row['timing']['median_ms'] = 3
        write(path, row)
        result = compare.analyze(out, config, cases, loaded)
        assert result['gate'] == 'REVIEW' and result['component_round_review']
        row['timing']['median_ms'] = 0
        write(path, row)
        try:
            compare.analyze(out, config, cases, loaded)
        except RuntimeError as error:
            assert 'Invalid measurement duration' in str(error)
        else:
            raise AssertionError('Invalid timing accepted')
        path.unlink()
        assert compare.analyze(out, config, cases, loaded)['collection'] == 'INCOMPLETE'


def recovery_checks():
    sources = compare.read(compare.HERE/'release_hashes.json')
    calls = []
    fail_once = True
    def fake_execute(command):
        nonlocal fail_once
        phase, out = command[2], Path(command[3])
        calls.append((phase, out.name))
        assert ('--collect-sealed-test' in command) == phase.startswith('test_')
        out.mkdir()
        if phase == 'train_02' and fail_once:
            fail_once = False
            write(out/'failure.json', {'interrupted': True})
            return 130
        write(out/'manifest.json', dict(phase=phase, source_hashes=sources))
        gate = 'REVIEW' if phase == 'validation_01' else 'PASS'
        write(out/'summary.json', dict(collection='COMPLETE', gate=gate, missing=0,
              recorded=1080, expected=1080, counts={'PASS': 1080},
              restoration='COMMANDS_ACCEPTED', worker_complete=True))
        archive = out.with_name(out.name + '_test.zip')
        with zipfile.ZipFile(archive, 'x') as package:
            for name in ('manifest.json', 'summary.json'):
                package.write(out/name, out.name + '/' + name)
        write(out/'archive_receipt.json', dict(filename=archive.name, sha256=compare.sha(archive)))
        return 2 if gate == 'REVIEW' else 0
    with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
        campaign = Path(temporary)
        assert collect_all.collect(campaign, fake_execute) == 130
        assert collect_all.collect(campaign, fake_execute) == 2
        assert Counter(p for p, _ in calls)['train_01'] == 1
        assert Counter(p for p, _ in calls)['train_02'] == 2
        assert len(calls) == 31
        before = len(calls)
        assert collect_all.collect(campaign, fake_execute) == 2 and len(calls) == before
        assert (campaign/'train_02_attempt_001/failure.json').is_file()
        summary = compare.read(campaign/'collection_summary.json')
        assert summary['review_batches'] == ['validation_01'] and summary['measurements'] == 32400
        path = campaign/'train_01_attempt_001/summary.json'
        original = path.read_text()
        altered = json.loads(original)
        altered['extra'] = 'changed after archival'
        write(path, altered)
        try:
            collect_all.collect(campaign, fake_execute)
        except RuntimeError as error:
            assert 'metadata differs' in str(error)
        else:
            raise AssertionError('Changed completion metadata accepted')
        path.write_text(original)
        (campaign/'train_01_attempt_001_test.zip').write_bytes(b'corrupt')
        try:
            collect_all.collect(campaign, fake_execute)
        except RuntimeError as error:
            assert 'archive is missing or damaged' in str(error)
        else:
            raise AssertionError('Corrupt completed archive accepted')
        try:
            collect_all.resolve_attempt(campaign, '../outside', 'train_01')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Outside attempt accepted')


def packing_checks():
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)/'test_01_attempt_001'
        write(out/'manifest.json', {'phase': 'test_01', 'fixture': 'synthetic'})
        write(out/'summary.json', {'gate': 'PASS', 'fixture': 'synthetic'})
        result = subprocess.run([sys.executable, '-B', str(compare.HERE/'compare.py'), 'pack',
                                 '--output', str(out), '--phase', 'test_01', '--collect-sealed-test'],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        receipt = compare.read(out/'archive_receipt.json')
        archive = out.parent/receipt['filename']
        assert compare.sha(archive) == receipt['sha256']
        assert archive.with_suffix('.zip.sha256').read_text().split()[0] == receipt['sha256']
        with zipfile.ZipFile(archive) as package:
            assert package.testzip() is None
            hashes = json.loads(package.read(out.name + '/result_hashes.json'))
            assert set(hashes) == {'manifest.json', 'summary.json'}
            assert hashes['manifest.json'] == compare.sha(out/'manifest.json')


def signal_checks():
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        fixture = out/'child.py'
        fixture.write_text('''import signal, subprocess, sys, time
from pathlib import Path
root, role = Path(sys.argv[1]), sys.argv[2]
child = None
def stop(signum, frame):
    if child is not None:
        child.wait(timeout=5)
    (root / (role + '.cleaned')).write_text('cleanup completed')
    raise SystemExit(143)
signal.signal(signal.SIGTERM, stop)
if role == 'child':
    child = subprocess.Popen([sys.executable, __file__, str(root), 'grandchild'])
(root / (role + '.ready')).write_text('ready')
while True:
    time.sleep(.05)
''')
        code = ('import sys; sys.path.insert(0, ' + repr(str(compare.HERE)) + '); '
                'import collect_all; raise SystemExit(collect_all.run_batch(' +
                repr([sys.executable, str(fixture), str(out), 'child']) + '))')
        bridge = subprocess.Popen([sys.executable, '-B', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not (out/'grandchild.ready').exists():
                assert bridge.poll() is None and time.monotonic() < deadline
                time.sleep(.02)
            bridge.terminate()
            stdout, stderr = bridge.communicate(timeout=10)
            assert bridge.returncode == 143, (stdout, stderr)
            assert (out/'child.cleaned').is_file() and (out/'grandchild.cleaned').is_file()
        finally:
            if bridge.poll() is None:
                bridge.terminate()
                bridge.wait(timeout=10)


def numeric_checks():
    import torch
    torch.set_num_threads(1)
    tolerances = compare.read(compare.HERE/'protocol.json')['tolerances']
    checked = 0
    with torch.inference_mode():
        for aligned in (True, False):
            d, f = (16, 32) if aligned else (17, 33)
            cases = [dict(layer='linear', rows=7, input_width=d, output_width=f),
                     dict(layer='ffn', rows=7, input_width=d, hidden_width=f)]
            cases += [dict(layer=f'conv{k}', batch=2, height=5, width=7, cin=d, cout=f, kernel=k) for k in (1, 3)]
            for case in cases:
                for dtype in ('fp16', 'fp32'):
                    pair = make_pair(torch, case, dtype, device='cpu')
                    assert correctness(torch, case, pair, dtype, tolerances)['passed'], (case, dtype)
                    functions, identity = diagnostic.prepare(torch, case, pair)
                    if case['layer'] == 'ffn':
                        h = functions['gemm1']()
                        torch.testing.assert_close(functions['gelu'](), torch.nn.functional.gelu(h, approximate='none'), rtol=0, atol=0)
                        torch.testing.assert_close(functions['gemm2'](), pair['functions']['decomposed'](), rtol=0, atol=0)
                        assert identity['gemm2_input']['sha256']
                    elif case['layer'].startswith('conv'):
                        torch.testing.assert_close(functions['output_pack'](), pair['functions']['decomposed'](), rtol=0, atol=0)
                    else:
                        assert functions == {} and identity['gemm_record'] == 'decomposed'
                    assert set(functions) == set(compare.implementations(case, {})) - {'standard', 'decomposed'}
                    checked += 1
    print(f'NUMERIC PASS: {checked} CPU cases, torch={torch.__version__}; GPU timing requires server pilot')
    worker_checks(torch)


def worker_checks(torch):
    cases = [dict(id='numeric_linear', layer='linear', rows=7, input_width=17, output_width=33),
             dict(id='numeric_ffn', layer='ffn', rows=7, input_width=17, hidden_width=33)]
    cases += [dict(id=f'numeric_conv{k}', layer=f'conv{k}', batch=2, height=5, width=7,
                   cin=17, cout=33, kernel=k) for k in (1, 3)]
    config = dict(compare.read(compare.HERE/'protocol.json'), phase='train_01', preheat_seconds=0,
                  health_case_ids=[case['id'] for case in cases])
    class Timer:
        def __init__(self, torch_module, timing):
            pass
        def block(self, fn, repeats, operations=1):
            fn()
            return {}
        def run(self, fn):
            self.block(fn, 1)
            return dict(median_ms=1.0, flags=[], blocks=[])
    with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
        out = Path(temporary)
        for name in ('worker', 'inputs', 'correctness', 'records', 'restricted_explicit/records'):
            (out/name).mkdir(parents=True, exist_ok=True)
        write(out/'manifest.json', dict(source_hashes=compare.read(compare.HERE/'release_hashes.json'),
                                       environment_signature='numeric_fixture'))
        suite = SimpleNamespace(atomic_json=write, now=lambda: 'numeric_fixture',
                environment=lambda _: dict(environment={}, gpu_uuid='numeric_fixture'),
                signature=lambda _: 'numeric_fixture')
        audit = SimpleNamespace(observed_timer=lambda timer: timer, check_processes=lambda *args: None)
        tensor = torch.ones((2, 2))
        loaded = dict(suite=suite, audit=audit, telemetry=None,
                      measure=SimpleNamespace(Timer=Timer, make_gemm=lambda *args: (tensor, tensor, tensor, lambda: tensor @ tensor)))
        frozen = SimpleNamespace(Guard=lambda *args: SimpleNamespace(check=lambda: None))
        original_prepare = diagnostic.prepare
        def prepare(module, case, pair):
            for dtype in config['dtypes']:
                if pair['input'].dtype == {'fp16': torch.float16, 'fp32': torch.float32}[dtype]:
                    break
            # Both whole forwards must be saved before any diagnostic buffers are allocated.
            for rd in range(3):
                if not (out/'inputs'/f'{case["id"]}_{dtype}_r{rd}_components.json').exists():
                    assert all(compare.record_path(out, case, dtype, rd, name).is_file()
                               for name in ('standard', 'decomposed'))
                    break
            return original_prepare(module, case, pair)
        stack.enter_context(patch.object(compare, 'make_pair', lambda module, case, dtype: make_pair(module, case, dtype, device='cpu')))
        stack.enter_context(patch.object(compare, 'memory', lambda module, fn: dict(extra_peak_allocated_bytes=0)))
        stack.enter_context(patch.object(compare, 'analyze', lambda *args, **kwargs: dict(gate='PASS')))
        stack.enter_context(patch.object(diagnostic, 'prepare', prepare))
        stack.enter_context(patch.object(torch.cuda, 'synchronize', lambda: None))
        stack.enter_context(patch.object(torch.cuda, 'mem_get_info', lambda: (8 * 1024**3, 8 * 1024**3)))
        import ffn_probe
        probe = stack.enter_context(patch.object(ffn_probe, 'run'))
        with contextlib.redirect_stdout(io.StringIO()):
            compare.worker(out, config, cases, frozen, loaded)
        probe.assert_called_once()
        records = [compare.read(path) for path in compare.saved_paths(out)]
        assert len(records) == 108 and compare.read(out/'worker/complete.json')['complete']
        for row in records:
            component = row['implementation'] not in ('standard', 'decomposed')
            assert bool(row['component_identity']) == component
            if component:
                assert f'_r{row["round"]}_components.json' in row['component_identity']
                assert (out/row['component_identity']).is_file()
            assert (out/row['correctness_after']).is_file()
    print('WORKER PASS: 108 CPU forward/component calls with mocked CUDA timing and controls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numeric', action='store_true')
    args = parser.parse_args()
    config, cases = static_checks()
    _, loaded = compare.load_frozen()
    analyzer_checks(config, cases, loaded['telemetry'])
    recovery_checks()
    packing_checks()
    signal_checks()
    if args.numeric:
        numeric_checks()
        import test_ffn_reference
        test_ffn_reference.numeric_checks()
        import test_conv_reference
        print(json.dumps(test_conv_reference.numeric_checks(), sort_keys=True))
    print('SELF CHECK PASS: manifests, protection, 1080-record analyzer, sealed access, resume and archive integrity')
