"""Validate the formerly failing CUDA inputs before a fresh collection."""
from pathlib import Path
import compare

HERE = Path(__file__).resolve().parent


def run(torch, tolerances, out, save):
    """Recheck the exact failed CUDA inputs before warming or collecting train_01."""
    from layers import correctness, make_pair, tensor_identity
    cases = compare.read(HERE/'cases_by_phase.json')['train_02']
    case = next(c for c in cases if c['id'] == 'lf_76b19fecaab55272')
    pair = make_pair(torch, case, 'fp16')
    identity = dict(input=tensor_identity(pair['input']),
                    weights=[tensor_identity(t) for t in pair['weights']],
                    prepared=[tensor_identity(t) for t in pair['prepared']], seed=pair['seed'])
    expected = compare.read(HERE/'regression/lf_76b19fecaab55272_fp16.json')
    if identity != expected:
        raise RuntimeError('FFN regression probe input hashes differ from the failed run')
    result = correctness(torch, case, pair, 'fp16', tolerances)
    result['input_identity'] = identity
    result['case'] = case
    save(out/'ffn_reference_probe.json', result)
    if not result['passed']:
        raise RuntimeError('FFN regression probe failed; inspect ffn_reference_probe.json')
    print('FFN REFERENCE PROBE PASS: original inputs, corrected FFN reference, unchanged tolerances', flush=True)
