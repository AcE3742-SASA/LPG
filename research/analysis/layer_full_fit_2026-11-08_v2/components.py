"""Static component regressors and their summed explicit-layer prediction."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEARNER_PATH = ROOT / 'analysis/learner_comparison_2026-10-28/compare_learners.py'
spec = importlib.util.spec_from_file_location('full_layer_component_learners', LEARNER_PATH)
learners = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learners)

FEATURES = learners.old.FEATURES + [
    'log_batch', 'log_height', 'log_width', 'log_cin', 'log_cout',
    'log_kernel', 'log_input_elements', 'log_output_elements',
]
COMPONENTS = {
    'linear': ('gemm',),
    'ffn': ('gemm1', 'gelu', 'gemm2'),
    'conv1': ('input_pack', 'mm', 'output_pack'),
    'conv3': ('unfold', 'input_pack', 'mm', 'output_pack'),
}


def component_names(layer):
    if layer not in COMPONENTS:
        raise ValueError('Unsupported layer')
    return COMPONENTS[layer]


def _positive_int(case, key):
    value = case[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('Invalid static dimension: ' + key)
    return value


def _dimensions(case, component):
    layer = case['layer']
    if component not in component_names(layer):
        raise ValueError('Component does not belong to layer')
    if layer.startswith('conv'):
        batch, height, width, cin, cout, kernel = (
            _positive_int(case, key) for key in
            ('batch', 'height', 'width', 'cin', 'cout', 'kernel'))
        if kernel != int(layer[-1]):
            raise ValueError('Convolution kernel mismatch')
        m, n, k = batch * height * width, cout, cin * kernel ** 2
    else:
        m = _positive_int(case, 'rows')
        cin = _positive_int(case, 'input_width')
        cout = _positive_int(case, 'output_width' if layer == 'linear' else 'hidden_width')
        batch = height = width = kernel = 1
        n, k = (cin, cout) if component == 'gemm2' else (cout, cin)
    if component in {'gemm', 'gemm1', 'gemm2', 'mm'}:
        inputs, outputs = m * k + k * n, m * n
    elif component == 'gelu':
        inputs = outputs = m * cout
    elif component == 'unfold':
        inputs, outputs = m * cin, m * cin * kernel ** 2
    elif component == 'input_pack':
        inputs = outputs = m * cin * kernel ** 2
    else:
        inputs = outputs = m * cout
    return (m, n, k), (batch, height, width, cin, cout, kernel, inputs, outputs)


def component_features(cases, dtype, component):
    if dtype not in ('fp16', 'fp32'):
        raise ValueError('Unsupported dtype')
    if not cases:
        return np.empty((0, len(FEATURES)), dtype=float)
    shapes, counts = zip(*(_dimensions(case, component) for case in cases))
    gemm_features, _ = learners.old.design(shapes, dtype)
    return np.column_stack((gemm_features, np.log(np.asarray(counts, dtype=float))))


def fit_sum(cases, dtype, learner, labels, seed):
    """Fit each log-time using train cases and (case_id, dtype, impl) labels."""
    if not cases or any(case.get('split') != 'train' for case in cases):
        raise ValueError('Component fit requires nonempty train-only cases')
    layer = cases[0]['layer']
    if any(case['layer'] != layer for case in cases):
        raise ValueError('Component fit requires one layer')
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate training case')
    if len({case['group_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate training group')
    models, details = {}, []
    for component in component_names(layer):
        implementation = 'decomposed' if component == 'gemm' else component
        y = np.asarray([labels[(case['id'], dtype, implementation)] for case in cases], dtype=float)
        if not (np.isfinite(y) & (y > 0)).all():
            raise ValueError('Component labels must be finite positive milliseconds')
        model, diagnostic = learners.fit(
            learner, component_features(cases, dtype, component), np.log(y), seed)
        models[component] = model
        details.append(dict(component=component, implementation=implementation, **diagnostic))
    bundle = dict(version='component_sum_v1', layer=layer, dtype=dtype, learner=learner,
                  seed=seed, features=FEATURES, models=models,
                  train_case_ids=[case['id'] for case in cases],
                  train_group_ids=[case['group_id'] for case in cases])
    diagnostic = dict(fit_seconds=sum(row['fit_seconds'] for row in details),
                      convergence_warnings=sum(row['convergence_warnings'] for row in details),
                      component_fits=len(details), components=details)
    return bundle, diagnostic


def predict_sum(bundle, cases):
    """Return (sum_ms, component_ms, valid_mask), keeping failed rows visible."""
    if bundle.get('version') != 'component_sum_v1' or bundle.get('features') != FEATURES:
        raise ValueError('Component model feature contract mismatch')
    names = component_names(bundle['layer'])
    if set(bundle['models']) != set(names):
        raise ValueError('Component model set mismatch')
    if any(case['layer'] != bundle['layer'] for case in cases):
        raise ValueError('Prediction layer mismatch')
    if not cases:
        return np.empty(0), {name: np.empty(0) for name in names}, np.empty(0, dtype=bool)
    predictions = {}
    for name in names:
        response = np.asarray(bundle['models'][name].predict(
            component_features(cases, bundle['dtype'], name)), dtype=float)
        if response.shape != (len(cases),):
            raise ValueError('Component prediction shape mismatch')
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            predictions[name] = np.exp(response)
    matrix = np.column_stack([predictions[name] for name in names])
    with np.errstate(over='ignore', invalid='ignore'):
        total = matrix.sum(axis=1)
    valid = (np.isfinite(matrix) & (matrix > 0)).all(axis=1) & np.isfinite(total) & (total > 0)
    total[~valid] = np.nan
    return total, predictions, valid
