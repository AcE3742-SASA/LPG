"""Check the single proven exclusion, accounting and unchanged worker failure behavior."""
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import compare
import diagnostic
import layers


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def must_fail(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Invalid exclusion or unexpected correctness failure accepted')


def fixture():
    cases = compare.read(compare.HERE/'cases_by_phase.json')['validation_01']
    config = dict(compare.read(compare.HERE/'protocol.json'), phase='validation_01',
                  health_case_ids=[next(c for c in cases if c['layer'] == layer)['id']
                                   for layer in ('linear', 'ffn', 'conv1', 'conv3')])
    return config, cases


def checks():
    config, cases = fixture()
    exclusions = compare.excluded_conditions(cases, config)
    assert exclusions == config['correctness_exclusions']
    assert compare.planned_records(cases, config) == 1080
    assert compare.expected_records(cases, config) == 1062
    planned = compare.read(compare.HERE/'cases_by_phase.json')
    assert sum(compare.expected_records(cc, dict(config, phase=phase))
               for phase, cc in planned.items()) == 32382
    for changed in ([], exclusions*2, [dict(exclusions[0], dtype='fp16')],
                    [dict(exclusions[0], case_id=cases[0]['id'])]):
        must_fail(lambda: compare.excluded_conditions(cases, dict(config, correctness_exclusions=changed)))
    must_fail(lambda: compare.excluded_conditions(cases, dict(config, health_case_ids=[exclusions[0]['case_id']])))
    with patch.object(compare, 'sha', return_value='0'*64):
        must_fail(lambda: compare.excluded_conditions(cases, config))
    original_read = compare.read
    proof_path = compare.HERE/exclusions[0]['evidence']
    proof = original_read(proof_path)
    for field in ('passed', 'input_identity_matched'):
        changed = copy.deepcopy(proof)
        changed[field] = not changed[field]
        with patch.object(compare, 'read', lambda p: changed if p == proof_path else original_read(p)):
            must_fail(lambda: compare.excluded_conditions(cases, config))
    changed = copy.deepcopy(proof)
    changed['full_pair_adjudication']['implementations']['standard']['passed'] = True
    with patch.object(compare, 'read', lambda p: changed if p == proof_path else original_read(p)):
        must_fail(lambda: compare.excluded_conditions(cases, config))

    loaded = dict(suite=SimpleNamespace(atomic_json=write), telemetry=SimpleNamespace())
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        write(out/'correct.json', {'passed': True})
        write(out/'status.json', dict(collection='COMPLETE', restoration='COMMANDS_ACCEPTED'))
        write(out/'worker/complete.json', {'complete': True})
        for case in cases:
            for dtype in config['dtypes']:
                if not compare.eligible_condition(case, dtype, config):
                    continue
                for rd in range(3):
                    for implementation in compare.implementations(case, config):
                        write(compare.record_path(out, case, dtype, rd, implementation),
                              dict(id=compare.key(case, dtype, rd, implementation), case=case, dtype=dtype,
                                   timing=dict(median_ms=1.0, flags=[], blocks=[]),
                                   memory=dict(extra_peak_allocated_bytes=0), prepared_weight_bytes=0,
                                   correctness_before='correct.json', correctness_after='correct.json'))
        result = compare.analyze(out, config, cases, loaded)
        assert result['gate'] == 'REVIEW' and result['collection'] == 'COMPLETE'
        assert (result['planned'], result['expected'], result['recorded'], result['excluded'], result['missing']) == (1080,1062,1062,18,0)
        assert result['excluded_conditions'] == exclusions and result['counts'] == {'PASS':1062}
        assert result['comparison_counts'] == {'PASS':79}
        health = compare.analyze(out, config, cases, loaded, health_only=True)
        assert health['gate'] == 'PASS' and health['recorded'] == 108 and health['excluded'] == 0
        known = next(c for c in cases if c['id'] == exclusions[0]['case_id'])
        assert len([p for p in compare.saved_paths(out) if known['id']+'_fp16_' in p.name]) == 18
        assert not any(known['id']+'_fp32_' in p.name for p in compare.saved_paths(out))
        extra = compare.record_path(out, known, 'fp32', 0, 'standard')
        write(extra, {'id':compare.key(known,'fp32',0,'standard')})
        must_fail(lambda: compare.analyze(out, config, cases, loaded))
        extra.unlink()
        write(out/'correct.json', {'passed':False})
        assert compare.analyze(out, config, cases, loaded)['gate'] == 'FAIL'
        write(out/'correct.json', {'passed':True})
        compare.saved_paths(out)[0].unlink()
        assert compare.analyze(out, config, cases, loaded)['collection'] == 'INCOMPLETE'
    return dict(expected=1062, planned=1080, excluded=18, eligible_comparisons=79,
                all_campaign_expected=32382, healthy_records=108, tampered_exclusions_rejected=True)


def numeric_checks():
    import torch
    torch.set_num_threads(1)
    config, cases = fixture()
    config['preheat_seconds'] = 0
    known = config['correctness_exclusions'][0]['case_id']
    calls = []
    def make_pair(module, case, dtype):
        assert (case['id'],dtype) != (known,'fp32')
        calls.append((case['id'],dtype))
        numeric = dict(case)
        if case['layer'].startswith('conv'):
            numeric.update(batch=2, cin=2, cout=3, height=5, width=4)
        else:
            numeric.update(rows=7, input_width=17)
            numeric['hidden_width' if case['layer']=='ffn' else 'output_width'] = 33
        pair = layers.make_pair(module, numeric, dtype, device='cpu')
        pair['numeric_case'] = numeric
        return pair
    def correctness(module, case, pair, dtype, tolerances):
        return layers.correctness(module, pair['numeric_case'], pair, dtype, tolerances)
    original_prepare = diagnostic.prepare
    class Timer:
        def __init__(self, *args):
            pass
        def run(self, fn):
            fn()
            return dict(median_ms=1.0, flags=[], blocks=[])
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        base = Path(directory)
        suite = SimpleNamespace(atomic_json=write, now=lambda:'fixture',
            environment=lambda _:dict(environment={},gpu_uuid='fixture'), signature=lambda _:'fixture')
        tensor = torch.ones((2,2))
        loaded = dict(suite=suite, telemetry=None,
            audit=SimpleNamespace(observed_timer=lambda timer:timer, check_processes=lambda *args:None),
            measure=SimpleNamespace(Timer=Timer, make_gemm=lambda *args:(tensor,tensor,tensor,lambda:tensor@tensor)))
        frozen = SimpleNamespace(Guard=lambda *args:SimpleNamespace(check=lambda:None))
        stack.enter_context(patch.object(compare,'make_pair',make_pair))
        stack.enter_context(patch.object(compare,'correctness',correctness))
        stack.enter_context(patch.object(compare,'memory',lambda *args:dict(extra_peak_allocated_bytes=0)))
        stack.enter_context(patch.object(compare,'analyze',lambda *args,**kwargs:dict(gate='PASS')))
        stack.enter_context(patch.object(diagnostic,'prepare',lambda module,case,pair:original_prepare(module,pair['numeric_case'],pair)))
        stack.enter_context(patch.object(torch.cuda,'synchronize',lambda:None))
        stack.enter_context(patch.object(torch.cuda,'mem_get_info',lambda:(8*1024**3,8*1024**3)))
        def output(name):
            out=base/name
            for child in ('worker','inputs','correctness','records','restricted_explicit/records'):
                (out/child).mkdir(parents=True,exist_ok=True)
            write(out/'manifest.json',dict(source_hashes=compare.read(compare.HERE/'release_hashes.json'),environment_signature='fixture'))
            return out
        out=output('complete')
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            compare.worker(out,config,cases,frozen,loaded)
        assert captured.getvalue().count('EXCLUDED:') == 3
        records=[compare.read(p) for p in compare.saved_paths(out)]
        assert len(records)==1062 and compare.read(out/'worker/complete.json')['complete']
        assert Counter(calls)[known,'fp16']==3 and (known,'fp32') not in calls
        assert sum(r['case']['id']==known and r['dtype']=='fp16' for r in records)==18
        out=output('unexpected_failure')
        with patch.object(compare,'correctness',return_value={'passed':False}), contextlib.redirect_stdout(io.StringIO()):
            must_fail(lambda:compare.worker(out,config,cases,frozen,loaded))
        assert not (out/'worker/complete.json').exists() and not compare.saved_paths(out)
    return dict(cpu_worker_records=1062, excluded_before_pair_creation=True, retained_fp16_records=18,
                unexpected_correctness_failure_stops=True)


if __name__=='__main__':
    print(json.dumps(checks(),sort_keys=True))
    print(json.dumps(numeric_checks(),sort_keys=True))
