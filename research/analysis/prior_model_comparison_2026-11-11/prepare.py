"""Freeze comparison choices and load the existing common development cohort."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT/'analysis/layer_full_fit_2026-11-10_v5'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    if (HERE/'protocol.json').exists():
        raise RuntimeError('Comparison protocol already exists')
    prior_inputs = json.loads((ROOT/'analysis/cost_data_efficiency_2026-11-11/input_manifest.json').read_text())
    for name, expected in prior_inputs.items():
        assert sha(ROOT/name) == expected, name
    save(HERE/'original_inputs.json', prior_inputs)
    source_hashes = {str(p.relative_to(HERE)): sha(p) for p in (HERE/'sources').rglob('*') if p.is_file()}
    save(HERE/'source_hashes.json', source_hashes)
    sys.path.insert(0, str(REFERENCE))
    spec = importlib.util.spec_from_file_location('reference_layer_pipeline', REFERENCE/'pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    import data
    frozen = REFERENCE/'prepared_layer_full'
    p = pipeline.verify_protocol(frozen)
    dataset = data.load_campaign(ROOT/'analysis/layer_full_result_2026-11-10/raw',
                                 diagnostic=True, protocol_path=frozen/'frozen_protocol.json')
    rows, coverage = [], []
    for layer in pipeline.LAYERS:
        for dtype in ('fp16', 'fp32'):
            cases, labels, quality = pipeline.cohort(dataset, p, layer, dtype)
            coverage.append(quality)
            for target in pipeline.TARGETS:
                rows.extend(dict(case=c, dtype=dtype, target=target, actual_ms=labels[c['id'],dtype,target]) for c in cases)
    save(HERE/'development.json', rows)
    save(HERE/'quality_coverage.json', coverage)
    protocol = dict(date='2026-11-11', status='BASELINES_FIXED_BEFORE_FIT',
        reference_protocol_sha256=sha(frozen/'frozen_protocol.json'),
        development_sha256=sha(HERE/'development.json'),
        source_hashes_sha256=sha(HERE/'source_hashes.json'),
        seeds=p['seeds'], budgets=p['budgets'], targets=p['targets'], dtypes=['fp16','fp32'],
        common_layers=['linear','conv1','conv3'], neusight_layers=['linear','ffn'],
        subsets=p['subsets'],
        selection='Minimum mean validation MAPE, then mean p90, training labels, k. Any invalid prediction disqualifies. Retain k=200 alongside selected results.',
        protocol_scope='Supplementary benchmark after the original test results were disclosed. New baseline choices use train/CV/validation only. Freeze before scoring saved test labels; preserve original models and test artifacts.',
        shared_pool='Each dtype and execution target has k Linear, k Conv1 and k Conv3 training labels. GEMMify pools all three. NeuralPower pools Conv1/Conv3 and fits FC separately. Inner CV keeps group_id together.',
        gemmify=dict(features=['M','N','K'], main='DecisionTreeRegressor with literal Table I grid; 5-fold group CV minimizing MAPE',
            max_depth=[None,5,10,15,20], max_leaf_nodes=[2,3,4,5,6], max_features=[None,'sqrt','log2'],
            capacity_sensitivity='Same grid with max_leaf_nodes=None; labelled a capacity adaptation',
            implementation='Paper-equation reimplementation; author repository not located in the documented search'),
        neuralpower=dict(features='Exact public FC degree-2 plus product / CONV degree-3 plus eight special terms',
            target='raw milliseconds', regularization='100-lambda LassoCV; up to 10 grouped folds; StandardScaler fitted on outer training subset only',
            lambda_ratio='0.01 when p>n, otherwise 1e-4', max_iter=100000, tolerance=1e-5,
            main_floor='minimum observed training latency within operator family',
            artifact_floor_sensitivity_ms=0.105,
            adaptations=['MATLAB lasso port to scikit-learn','keep original shared PASS cohort instead of author-specific runtime/shape filters','refit per dtype and execution target']),
        paleo=dict(upstream='Original forward FLOP/memory estimator; use_only_gemm path, bias/activation disabled',
            fp32_compute_gflops=6140, fp16_compute_gflops=35020, bandwidth_bytes_per_second=262.3e9,
            clock_mhz=720, platform_percent_of_peak=1,
            adaptations=['reuse existing empirical P/B calibration','FP16 byte-width and compute-throughput extension','explicit decomposed target is a supplementary application']),
        neusight=dict(upstream='Unmodified public model forward code and pretrained FP32 weights',
            tile_lookup='Unscaled Euclidean nearest metadata row over author features; batch result checked against upstream MetaTable',
            clock_profile=dict(Num_Sm=48,SingleFLOPs=8847.36,Dev_Mem=20,Mem_Bw=272.04,L2Cache=40),
            empirical_profile=dict(Num_Sm=48,SingleFLOPs=6140,Dev_Mem=20,Mem_Bw=262.3,L2Cache=40),
            hardware_evidence=['analysis/cutlass_inventory_2026-09-21/raw/help.txt','analysis/power_shape_diag_v1_result_2026-10-02/ncu_metrics.json','clock_audit_v1/baseline_v6.json'],
            adaptations=['CPU inference under PyTorch 2.15','batch nearest-neighbor tile lookup without global cache writes','new GPU with fixed clock','bias-trained LINEAR predictor applied to bias-free layers','FFN sums LINEAR+GELU+LINEAR'],
            unsupported=['FP16 weights','native cuDNN convolutions']),
        cost='Record raw labels, unique groups, model fit/prediction seconds, source model cost category. Published source training cost stays external/unknown; never count it as free.',
        test_in_training=False, original_inputs_unchanged=len(prior_inputs))
    save(HERE/'protocol.json', protocol)
    print('PASS: frozen baseline protocol; development rows', len(rows), '; original inputs', len(prior_inputs), flush=True)


if __name__ == '__main__':
    main()
