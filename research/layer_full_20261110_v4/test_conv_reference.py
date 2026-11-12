"""CPU regression checks for exact-coordinate Conv pair adjudication."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import compare
import conv_probe
import layers
import v2_layers


def write(path, value):
    path.write_text(json.dumps(value))


def must_fail(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid correctness check accepted')


def cli_checks(torch, tolerances):
    signature = {'gpu': 'test'}
    env = dict(gpu_uuid='test', environment={}, signature=signature)
    source_hashes = {'layers.py': 'test'}
    manifest = dict(source_hashes=source_hashes, environment_signature=signature)
    suite = SimpleNamespace(environment=lambda _: env, signature=lambda value: value['signature'],
                            atomic_json=write)
    loaded = dict(suite=suite, audit=SimpleNamespace(check_processes=lambda *args: None))
    def read(path):
        return manifest if path.name == 'manifest.json' else source_hashes
    def probe(torch_arg, tolerances_arg, out, save):
        assert torch_arg is torch and tolerances_arg == tolerances
        assert torch.is_inference_mode_enabled()
        save(out/'conv_reference_probe.json', dict(passed=True))
    original_phase = compare.PHASE
    try:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'worker').mkdir()
            report = out/'conv_reference_probe.json'
            with patch.object(compare, 'verify', return_value=(dict(tolerances=tolerances), [])), \
                 patch.object(compare, 'authorize_result_access'), \
                 patch.object(compare, 'load_frozen', return_value=(None, loaded)), \
                 patch.object(compare, 'read', read), \
                 patch.object(compare, 'precision', return_value={'conv': 'ieee'}), \
                 patch.object(conv_probe, 'run', probe):
                assert conv_probe.main(['--output', directory]) == 0
                saved = report.read_bytes()
                result = json.loads(saved)
                assert result['passed'] and result['layer_precision'] == {'conv': 'ieee'}
                assert result['environment_signature'] == signature
                must_fail(lambda: conv_probe.main(['--output', directory]))
                assert report.read_bytes() == saved
                report.unlink()
                manifest['source_hashes'] = {'layers.py': 'changed'}
                must_fail(lambda: conv_probe.main(['--output', directory]))
                assert not json.loads(report.read_text())['passed']
                report.unlink()
                manifest['source_hashes'] = source_hashes
                manifest['environment_signature'] = {'gpu': 'different'}
                must_fail(lambda: conv_probe.main(['--output', directory]))
                assert not json.loads(report.read_text())['passed']
                report.unlink()
                manifest['environment_signature'] = signature
                env['environment']['CUDA_LAUNCH_BLOCKING'] = '1'
                must_fail(lambda: conv_probe.main(['--output', directory]))
                assert not json.loads(report.read_text())['passed']
    finally:
        compare.PHASE = original_phase


def numeric_checks():
    import torch
    torch.set_num_threads(1)
    tolerances = compare.read(compare.HERE/'protocol.json')['tolerances']
    case = dict(layer='conv3', batch=2, cin=2, cout=3, height=19, width=17, kernel=3)
    atol = tolerances['fp32'][0]
    with torch.inference_mode():
        for kernel in (1, 3):
            c = dict(case, layer='conv'+str(kernel), kernel=kernel)
            pair = layers.make_pair(torch, c, 'fp32', device='cpu')
            reference = torch.nn.functional.conv2d(pair['input'].double(), pair['weights'][0].double(),
                                                  padding=kernel//2)
            coordinates = torch.ones_like(reference, dtype=torch.bool).nonzero()
            actual = layers.conv_reference_points(torch, c, pair, coordinates)
            torch.testing.assert_close(actual, reference.flatten(), rtol=1e-12, atol=1e-12)

        pair = layers.make_pair(torch, case, 'fp32', device='cpu')
        pair['input'].zero_()
        shape = (case['batch'], case['cout'], case['height'], case['width'])
        standard = torch.full(shape, .75*atol)
        decomposed = torch.full(shape, -.75*atol)
        pair['functions'] = dict(standard=lambda: standard, decomposed=lambda: decomposed)
        old = v2_layers.correctness(torch, case, pair, 'fp32', tolerances)
        resolved = layers.correctness(torch, case, pair, 'fp32', tolerances)
        assert not old['passed'] and not resolved['full_pair_passed'] and resolved['passed']
        adjudication = resolved['full_pair_adjudication']
        assert adjudication['checked_count'] == adjudication['mismatch_count'] == standard.numel()

        standard.fill_(1.1*atol)
        broken = layers.correctness(torch, case, pair, 'fp32', tolerances)
        assert not broken['passed'] and not broken['full_pair_adjudication']['passed']

        standard.zero_()
        decomposed.zero_()
        sampled_rows = set(layers.indices(case['batch']*case['height']*case['width'], pair['seed']))
        unsampled = [p for p in torch.ones(shape, dtype=torch.bool).nonzero().tolist()
                     if (p[0]*case['height']+p[2])*case['width']+p[3] not in sampled_rows][:300]
        for coordinate in unsampled:
            standard[tuple(coordinate)] = .75*atol
            decomposed[tuple(coordinate)] = -.75*atol
        standard[tuple(unsampled[-1])] = 1.1*atol
        hidden_error = layers.correctness(torch, case, pair, 'fp32', tolerances)
        assert all(r['passed'] for r in hidden_error['implementations'].values())
        assert not hidden_error['passed'] and not hidden_error['full_pair_adjudication']['passed']
        assert hidden_error['full_pair_adjudication']['checked_count'] == len(unsampled)
        assert hidden_error['full_pair_adjudication']['coordinates'][-1] == unsampled[-1]

        for nonfinite in (float('nan'), float('inf')):
            standard.fill_(nonfinite)
            rejected = layers.correctness(torch, case, pair, 'fp32', tolerances)
            assert not rejected['passed'] and 'full_pair_adjudication' not in rejected
        pair['functions']['standard'] = lambda: torch.zeros(shape)[:, :, :-1, :]
        must_fail(lambda: layers.correctness(torch, case, pair, 'fp32', tolerances))

        unchanged = 0
        for c in [dict(layer='linear', rows=7, input_width=17, output_width=33),
                  dict(layer='ffn', rows=7, input_width=17, hidden_width=33),
                  dict(case, height=5, width=4),
                  dict(case, layer='conv1', kernel=1, height=5, width=4)]:
            for dtype in ('fp16', 'fp32'):
                p = layers.make_pair(torch, c, dtype, device='cpu')
                expected = v2_layers.correctness(torch, c, p, dtype, tolerances)
                actual = layers.correctness(torch, c, p, dtype, tolerances)
                assert expected['passed'] and actual == expected
                unchanged += 1

        probe_case = dict(case, id='lf_ce6440e85a4add93', height=5, width=4)
        probe_pair = layers.make_pair(torch, probe_case, 'fp32', device='cpu')
        identity = dict(input=layers.tensor_identity(probe_pair['input']),
                        weights=[layers.tensor_identity(t) for t in probe_pair['weights']],
                        prepared=[layers.tensor_identity(t) for t in probe_pair['prepared']], seed=probe_pair['seed'])
        original_read = compare.read
        def read(path):
            if path.name == 'cases_by_phase.json':
                return {'validation_01': [probe_case]}
            if path.name == 'lf_ce6440e85a4add93_fp32.json':
                return identity
            return original_read(path)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            report = out/'conv_reference_probe.json'
            with patch.object(layers, 'make_pair', return_value=probe_pair):
                must_fail(lambda: conv_probe.run(torch, tolerances, out, write))
            assert not original_read(report)['passed']
            assert not original_read(report)['input_identity_matched']
            with patch.object(layers, 'make_pair', return_value=probe_pair), patch.object(compare, 'read', read):
                conv_probe.run(torch, tolerances, out, write)
                assert original_read(report)['passed'] and original_read(report)['input_identity_matched']
                probe_pair['functions'] = {k: (lambda fn=fn: fn()+1)
                                           for k, fn in probe_pair['functions'].items()}
                must_fail(lambda: conv_probe.run(torch, tolerances, out, write))
                assert not original_read(report)['passed']
                assert original_read(report)['input_identity_matched']
    cli_checks(torch, tolerances)
    print('CONV PASS: every mismatch checked; unchanged tolerances; real errors and invalid probes rejected')
    return dict(torch=torch.__version__, adjudicated_points=adjudication['checked_count'],
                hidden_error_points=len(unsampled), unchanged_cases=unchanged,
                gpu_regression='original CUDA failure archived; known FP32 condition excluded in v4')


if __name__ == '__main__':
    print(json.dumps(numeric_checks(), sort_keys=True))
