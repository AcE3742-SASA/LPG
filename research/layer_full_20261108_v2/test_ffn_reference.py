"""Regression tests for FP16 storage-aware FFN correctness."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import compare
import layers
import previous_layers
import ffn_probe


def write(path, value):
    path.write_text(json.dumps(value))


def must_fail(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid probe accepted')


def numeric_checks():
    import torch
    torch.set_num_threads(1)
    tolerances = compare.read(compare.HERE/'protocol.json')['tolerances']
    failed = next(c for c in compare.read(compare.HERE/'cases_by_phase.json')['train_02']
                  if c['id'] == 'lf_76b19fecaab55272')
    case = dict(failed, id='rounding_regression_8')
    with torch.inference_mode():
        pair = layers.make_pair(torch, case, 'fp16', device='cpu')
        old = previous_layers.correctness(torch, case, pair, 'fp16', tolerances)
        new = layers.correctness(torch, case, pair, 'fp16', tolerances)
        assert not old['passed'] and new['passed']
        assert not all(r['mathematical_fp64_diagnostic']['passed'] for r in new['implementations'].values())
        functions = pair['functions']
        pair['functions'] = {k: (lambda fn=fn: fn()+1) for k, fn in functions.items()}
        broken = layers.correctness(torch, case, pair, 'fp16', tolerances)
        assert broken['full_pair_passed'] and not broken['passed']
        pair['functions'] = {k: (lambda fn=fn: fn()*float('nan')) for k, fn in functions.items()}
        assert not layers.correctness(torch, case, pair, 'fp16', tolerances)['passed']
        pair['functions'] = functions
        for dtype in ('fp16', 'fp32'):
            exact_pair = layers.make_pair(torch, failed, dtype, device='cpu')
            assert layers.correctness(torch, failed, exact_pair, dtype, tolerances)['passed']
        linear = dict(layer='linear', rows=7, input_width=17, output_width=33)
        for c, dt in [(linear, 'fp16'), (dict(layer='ffn', rows=7, input_width=17, hidden_width=33), 'fp32')]:
            p = layers.make_pair(torch, c, dt, device='cpu')
            assert layers.correctness(torch,c,p,dt,tolerances) == previous_layers.correctness(torch,c,p,dt,tolerances)
        # The CUDA input identity gate must run before the corrected reference probe.
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(layers, 'make_pair', return_value=pair):
                must_fail(lambda: ffn_probe.run(torch, tolerances, Path(directory), write))
            probe_pair = layers.make_pair(torch, failed, 'fp16', device='cpu')
            identity = dict(input=layers.tensor_identity(probe_pair['input']),
                            weights=[layers.tensor_identity(t) for t in probe_pair['weights']],
                            prepared=[layers.tensor_identity(t) for t in probe_pair['prepared']],
                            seed=probe_pair['seed'])
            original_read = compare.read
            def read(path):
                return identity if path.name == 'lf_76b19fecaab55272_fp16.json' else original_read(path)
            with patch.object(layers, 'make_pair', return_value=probe_pair), patch.object(compare, 'read', read):
                ffn_probe.run(torch, tolerances, Path(directory), write)
                report = Path(directory)/'ffn_reference_probe.json'
                assert original_read(report)['passed']
                probe_pair['functions'] = {k: (lambda fn=fn: fn()+1) for k, fn in probe_pair['functions'].items()}
                must_fail(lambda: ffn_probe.run(torch, tolerances, Path(directory), write))
                assert not original_read(report)['passed']
    print('ROUNDING PASS: legacy false failure reproduced; corrected pass; shared errors and NaNs rejected')
    return dict(torch=torch.__version__, cpu_regression_old_max_ratio=max(
                    r['max_tolerance_ratio'] for r in old['implementations'].values()),
                cpu_regression_new_max_ratio=max(r['max_tolerance_ratio'] for r in new['implementations'].values()))


if __name__ == '__main__':
    print(json.dumps(numeric_checks(), sort_keys=True))
