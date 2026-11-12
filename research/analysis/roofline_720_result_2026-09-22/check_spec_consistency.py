"""Read-only checks of retained results against declared architecture assumptions."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sources = {
    'datasheet': 'https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-rtx-4000-sff-ada-datasheet-2616456-web-1.pdf',
    'professional_ada_architecture': 'https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf',
    'nvidia_smi': 'https://docs.nvidia.com/deploy/nvidia-smi/index.html',
}
spec = dict(cuda_cores=6144, tensor_cores=192, fp32_boost_TFLOPs=19.2,
            sparse_fp8_boost_TFLOPs=306.8, memory_bandwidth_GBs=280,
            memory_bus_bits=160, power_limit_W=70)
# Derived from the professional Ada whitepaper, not a direct 4000 SFF measurement.
fp16_dense_flops_per_tc_cycle = 256
bw_adjusted = spec['memory_bandwidth_GBs'] * 6801 / 7001
checks = {}
inputs = []
for mhz in (720, 750):
    folder = ROOT / f'analysis/roofline_{mhz}_result_2026-09-22'
    parameter_file = folder / 'roofline_parameters.json'
    summary_file = folder / 'recomputed/summary.csv'
    inputs.extend([parameter_file, summary_file])
    p = json.loads(parameter_file.read_text())
    groups = list(csv.DictReader(summary_file.open()))
    peaks = {'simt': spec['cuda_cores'] * 2 * mhz / 1e6,
             'tensorop': spec['tensor_cores'] * fp16_dense_flops_per_tc_cycle * mhz / 1e6}
    out = {}
    for mode, param in [('simt', 'fp32_TFLOP_s'), ('tensorop', 'fp16_tensor_fp32_accum_TFLOP_s')]:
        best = max((r for r in groups if r['path'] == mode), key=lambda r: float(r['median']))
        actual = []
        for path in (folder / 'raw').glob('r[1-5]_*.gemm.csv'):
            for r in csv.DictReader(path.open()):
                if (r['Operation'], r['m']) != (best['function'], best['size']):
                    continue
                assert r['Disposition'] == 'passed' and r['Status'] == 'success'
                assert r['accum'] == 'f32' and int(r['batch_count']) == 1
                assert int(r['split_k_slices']) == 1
                rate = 2 * int(r['m']) * int(r['n']) * int(r['k']) / (float(r['Runtime']) * 1e9)
                actual.append(rate)
                inputs.append(path)
        assert len(actual) == 5
        assert math.isclose(statistics.median(actual), p[param], rel_tol=1e-12)
        out[mode] = {'computed_peak_TFLOPs': peaks[mode], 'measured_TFLOPs': p[param],
                     'fraction_of_computed_peak': p[param] / peaks[mode],
                     'within_computed_peak': p[param] <= peaks[mode],
                     'raw_five_run_recalculation': 'PASS'}
    out['bandwidth'] = {'adjusted_reference_GBs': bw_adjusted, 'measured_GBs': p['bandwidth_GB_s'],
                        'fraction_of_adjusted_reference': p['bandwidth_GB_s'] / bw_adjusted}
    checks[str(mhz)] = out

round_file = ROOT / 'analysis/abstract_200_720_recovery_result_2026-09-22/round_aggregates.csv'
inputs.append(round_file)
rounds = list(csv.DictReader(round_file.open()))
pilot = {}
for dtype, mode in [('fp32', 'simt'), ('fp16', 'tensorop')]:
    rows = [r for r in rounds if r['dtype'] == dtype]
    assert len(rows) == 600
    best = max(rows, key=lambda r: float(r['tflops']))
    value = float(best['tflops'])
    theoretical = checks['720'][mode]['computed_peak_TFLOPs']
    empirical = checks['720'][mode]['measured_TFLOPs']
    pilot[dtype] = {'maximum_TFLOPs': value, 'shape': [int(best[x]) for x in ('M', 'N', 'K')],
                    'mode': best['mode'], 'fraction_of_computed_peak': value / theoretical,
                    'fraction_of_CUTLASS_reference': value / empirical,
                    'combinations_above_CUTLASS_reference': sum(float(r['tflops']) > empirical for r in rows),
                    'all_within_computed_peak': all(float(r['tflops']) <= theoretical for r in rows)}
assert checks['720']['simt']['within_computed_peak']
assert checks['720']['tensorop']['within_computed_peak']
assert not checks['750']['tensorop']['within_computed_peak']
assert pilot['fp32']['combinations_above_CUTLASS_reference'] == 15
assert all(p['all_within_computed_peak'] for p in pilot.values())
result = {'calculation_verification': 'PASS', 'assessment': '720_RANGE_CONSISTENT_CEILING_NOT_UNIVERSAL_750_TENSOR_REVIEW',
          'official_sources': sources, 'specification': spec,
          'assumptions': {'fp16_dense_flops_per_tensor_core_cycle': fp16_dense_flops_per_tc_cycle,
                          'scope': 'Professional Ada throughput relation; 4000 SFF per-mode rate inferred, not separately specified in its datasheet',
                          'memory_reference': '280 GB/s times observed memory-clock ratio 6801/7001',
                          'clock_caveat': 'Sampled clocks do not independently establish continuous effective execution frequency'},
          'roofline_runs': checks, 'gemm_200_shapes': pilot,
          'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
(HERE / 'spec_consistency.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: result[k] for k in ('calculation_verification', 'assessment', 'roofline_runs', 'gemm_200_shapes')}, ensure_ascii=False, indent=2))
