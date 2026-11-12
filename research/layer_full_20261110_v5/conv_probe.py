"""Check the failed CUDA Conv inputs before resuming validation collection."""
import argparse
from pathlib import Path
import sys
import traceback
import compare

HERE = Path(__file__).resolve().parent


def run(torch, tolerances, out, save):
    from layers import correctness, make_pair, tensor_identity
    case = next(c for c in compare.read(HERE/'cases_by_phase.json')['validation_01']
                if c['id'] == 'lf_ce6440e85a4add93')
    result = dict(passed=False, case=case, input_identity_matched=False)
    report = out/'conv_reference_probe.json'
    try:
        pair = make_pair(torch, case, 'fp32')
        identity = dict(input=tensor_identity(pair['input']),
                        weights=[tensor_identity(t) for t in pair['weights']],
                        prepared=[tensor_identity(t) for t in pair['prepared']], seed=pair['seed'])
        result['input_identity'] = identity
        expected = compare.read(HERE/'regression/lf_ce6440e85a4add93_fp32.json')
        result['input_identity_matched'] = identity == expected
        if not result['input_identity_matched']:
            raise RuntimeError('Conv regression probe input hashes differ from the failed run')
        result.update(correctness(torch, case, pair, 'fp32', tolerances))
        if not result['passed']:
            raise RuntimeError('Conv regression probe failed; inspect conv_reference_probe.json')
    except Exception as error:
        result['passed'] = False
        result['error'] = dict(type=type(error).__name__, message=str(error))
        save(report, result)
        raise
    save(report, result)
    print('CONV REFERENCE PROBE PASS: original inputs, unchanged FP64 tolerances', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    compare.PHASE = 'validation_01'
    config, _ = compare.verify()
    out = args.output.resolve()
    compare.authorize_result_access(out, 'validation_01', False)
    _, loaded = compare.load_frozen()
    suite = loaded['suite']
    report = out/'conv_reference_probe.json'
    if report.exists():
        raise RuntimeError('Preserve the existing Conv probe; use a fresh batch attempt')
    try:
        import torch
        manifest = compare.read(out/'manifest.json')
        if manifest['source_hashes'] != compare.read(HERE/'release_hashes.json'):
            raise RuntimeError('Conv probe source snapshot mismatch')
        env = suite.environment(torch)
        suite.atomic_json(out/'worker/conv_probe_environment.json', env)
        if suite.signature(env) != manifest['environment_signature']:
            raise RuntimeError('GPU/software/base precision differs from reviewed baseline')
        if env['environment'].get('CUDA_LAUNCH_BLOCKING') not in (None, '', '0'):
            raise RuntimeError('CUDA_LAUNCH_BLOCKING is enabled')
        settings = compare.precision(torch)
        loaded['audit'].check_processes(suite, env['gpu_uuid'], out/'worker/conv_probe_processes.json')
        def save(path, result):
            suite.atomic_json(path, dict(result, layer_precision=settings,
                                        environment_signature=suite.signature(env)))
        with torch.inference_mode():
            run(torch, config['tolerances'], out, save)
    except Exception as error:
        if not report.exists():
            suite.atomic_json(report, dict(passed=False, stage='setup',
                                          error=dict(type=type(error).__name__, message=str(error))))
        raise
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('FAIL: '+str(error), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
