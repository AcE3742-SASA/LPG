"""Pinned prior-model adapters; original source files remain unchanged."""
import importlib
import json
from pathlib import Path
import sys
import time
import types
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent


def validate_training(cases, y):
    if len(cases) != len(y) or len(cases) < 2:
        raise ValueError('Training cases and labels must be aligned and nonempty')
    if any(c.get('split') != 'train' for c in cases) or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Training requires unique train-only cases')
    if not (np.isfinite(y) & (np.asarray(y) > 0)).all():
        raise ValueError('Training labels must be finite positive milliseconds')


def package(name, path):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def gemm_shape(case):
    if case['layer'].startswith('conv'):
        return (case['batch'] * case['height'] * case['width'], case['cout'],
                case['cin'] * case['kernel'] ** 2)
    return case['rows'], case['output_width'], case['input_width']


def neuralpower_features(cases):
    """Exact runtime monomials from the authors' FC/CONV MATLAB scripts."""
    conv = cases[0]['layer'].startswith('conv')
    if any(c['layer'].startswith('conv') != conv for c in cases):
        raise ValueError('NeuralPower requires one operator family')
    if conv:
        raw = []
        for c in cases:
            b, h, w, ci, co, k = (c[n] for n in ('batch', 'height', 'width', 'cin', 'cout', 'kernel'))
            raw.append([b, h, w, co, k, k, ci, co, k//2, k//2, 1, 1, b, h, w, ci])
        p = np.asarray(raw, dtype=float)
        simple = p[:, [0, 1, 2, 5, 6, 7, 13]]
        special = np.column_stack([
            p[:, 1]*p[:, 2]*p[:, 3]*p[:, 4]*p[:, 5]*p[:, 6],
            p[:, 12]*p[:, 1]*p[:, 2]*p[:, 3]*p[:, 4]*p[:, 5]*p[:, 6],
            p[:, 12]*p[:, 1]*p[:, 2]*p[:, 4]*p[:, 5]*p[:, 6],
            p[:, 0]*p[:, 1]*p[:, 2]*p[:, 3], p[:, 4]*p[:, 5]*p[:, 6]*p[:, 7],
            p[:, 12]*p[:, 13]*p[:, 14]*p[:, 15], p[:, 12]*p[:, 14]*p[:, 15],
            p[:, 12]*p[:, 13]*p[:, 15]])
        degree = 3
    else:
        simple = np.asarray([[c['rows'], c['output_width'], 1, 1, c['input_width']]
                             for c in cases], dtype=float)
        special = np.prod(simple, axis=1).reshape(-1, 1)
        degree = 2
    return np.column_stack((PolynomialFeatures(degree, include_bias=False).fit_transform(simple), special))


def fit_neuralpower(cases, y, seed):
    validate_training(cases, y)
    start = time.perf_counter()
    x = neuralpower_features(cases)
    scaler = StandardScaler().fit(x)
    groups = np.array([c['group_id'] for c in cases])
    folds = list(GroupKFold(min(10, len(set(groups))), shuffle=True, random_state=seed).split(x, y, groups))
    # MATLAB lasso's default lambda ratio changes when p exceeds n.
    model = LassoCV(alphas=100, eps=0.01 if x.shape[1] > len(x) else 0.0001,
                    cv=folds, max_iter=100000, tol=1e-5, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    bundle = dict(kind='neuralpower', model=model, scaler=scaler, floor=float(np.min(y)))
    detail = dict(fit_seconds=time.perf_counter()-start, features=x.shape[1], folds=len(folds),
                  alpha=float(model.alpha_), iterations=int(model.n_iter_),
                  selected_dual_gap=float(model.dual_gap_),
                  convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught),
                  nonzero_coefficients=int(np.count_nonzero(model.coef_)),
                  warning_examples=list(dict.fromkeys(str(w.message) for w in caught))[:3])
    return bundle, detail


def fit_gemmify(cases, y, seed, capacity=False):
    validate_training(cases, y)
    start = time.perf_counter()
    x = np.asarray([gemm_shape(c) for c in cases], dtype=float)
    groups = np.array([c['group_id'] for c in cases])
    cv = GroupKFold(min(5, len(set(groups))), shuffle=True, random_state=seed)
    grid = dict(max_depth=[None, 5, 10, 15, 20], max_features=[None, 'sqrt', 'log2'],
                max_leaf_nodes=[None] if capacity else [2, 3, 4, 5, 6])
    search = GridSearchCV(DecisionTreeRegressor(random_state=seed), grid, cv=cv,
                          scoring='neg_mean_absolute_percentage_error', n_jobs=1, error_score='raise')
    search.fit(x, y, groups=groups)
    return dict(kind='gemmify', model=search.best_estimator_), dict(
        fit_seconds=time.perf_counter()-start, best_parameters=search.best_params_,
        inner_cv_mape_pct=float(-100*search.best_score_), candidates=len(search.cv_results_['params']),
        folds=cv.n_splits, convergence_warnings=0, features=3)


def predict_fit(bundle, cases, artifact_floor=False):
    if bundle['kind'] == 'neuralpower':
        raw = bundle['model'].predict(bundle['scaler'].transform(neuralpower_features(cases)))
        return np.maximum(raw, 0.105 if artifact_floor else bundle['floor'])
    return bundle['model'].predict(np.asarray([gemm_shape(c) for c in cases], dtype=float))


def paleo(cases, dtype):
    source = HERE / 'sources/paleo/paleo'
    for name, path in [('paleo', source), ('paleo.layers', source/'layers'),
                       ('paleo.profilers', source/'profilers')]:
        package(name, path)
    flops = importlib.import_module('paleo.profilers.flops_profiler')
    core = importlib.import_module('paleo.layers.core')
    conv = importlib.import_module('paleo.layers.conv')
    base = importlib.import_module('paleo.profilers.base')
    device = importlib.import_module('paleo.device')
    options = base.ProfilerOptions()
    options.use_cudnn_heuristics = False
    hardware = device.Device('RTX4000SFF_720_empirical', clock=720,
                             peek_gflop=35020 if dtype == 'fp16' else 6140,
                             mem_bandwidth=262.3e9/2**30, is_gpu=True)
    estimator = flops.FlopsProfiler(options, hardware)
    previous_bytes = flops._BYTES_FLOAT
    flops._BYTES_FLOAT = 2 if dtype == 'fp16' else 4
    result = []
    try:
        for c in cases:
            if c['layer'].startswith('conv'):
                layer = conv.Conv2d(c['id'], [c['batch'], c['height'], c['width'], c['cin']],
                    [c['kernel'], c['kernel'], c['cin'], c['cout']], [1, 1, 1, 1], 'SAME')
            else:
                layer = core.InnerProduct(c['id'], [c['rows'], c['input_width']], c['output_width'])
            result.append(estimator.profile(layer).total_time)
    finally:
        flops._BYTES_FLOAT = previous_bytes
    return np.asarray(result)


class NeuSight:
    """Use pinned upstream forward code, weights, and nearest-neighbor tile lookup."""
    def __init__(self):
        sys.path.insert(0, str(HERE/'runtime'))
        import pandas as pd
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        source = HERE/'sources/neusight/neusight'
        for name, path in [('neusight', source), ('neusight.Model', source/'Model'),
                           ('neusight.Dataset', source/'Dataset')]:
            package(name, path)
        mm = importlib.import_module('neusight.Model.mlp_wave_mm')
        vec = importlib.import_module('neusight.Model.mlp_wave_vec')
        data = importlib.import_module('neusight.Dataset.dataset')
        root = HERE/'sources/neusight/scripts/asplos/data'
        device_files = sorted((root/'dataset/train/device_configs').glob('*.json'))
        devices = pd.DataFrame([json.loads(p.read_text()) for p in device_files])
        self.models, self.tables = {}, {}
        for kind, cls, table in [('LINEAR', mm.MLPWaveMM, 'linear.csv'),
                                  ('BMM', mm.MLPWaveMM, 'bmm.csv'),
                                  ('VEC', vec.MLPWaveVec, 'collect/gelu.csv')]:
            path = root/'predictor/MLP_WAVE'/kind
            config = json.loads((path/'config.json').read_text())
            model = cls(config, tag=None, device='cpu')
            model.load_state_dict(torch.load(path/'model.pth', map_location='cpu', weights_only=True)['model'])
            model.eval()
            # Reuse upstream parsing without its write to the user's global cache.
            parser = data.Dataset.__new__(data.Dataset)
            parser.device_df = devices
            frame = parser.prepare(pd.read_csv(root/'dataset/train'/table), None)
            self.models[kind], self.tables[kind] = model, frame

    def predict(self, shapes, kind='BMM', empirical=False):
        from scipy.spatial.distance import cdist
        model, frame = self.models[kind], self.tables[kind]
        hardware = dict(Num_Sm=48, SingleFLOPs=6140 if empirical else 8847.36,
                        Dev_Mem=20, Mem_Bw=262.3 if empirical else 272.04, L2Cache=40)
        values = []
        for shape in shapes:
            dims = dict(B=1, M=shape[0], N=shape[2], K=shape[1]) if kind != 'VEC' else dict(
                B=shape[0], H=shape[1], OpsPerO=1, MemPerO=8)
            values.append([{**hardware, **dims}[name] for name in model.features])
        x = np.asarray(values, dtype=np.float32)
        reference = frame[model.features].to_numpy(dtype=float)
        tile_data = frame[['Tile1', 'Tile2']].to_numpy(dtype=np.float32)
        output, matched = [], []
        for start in range(0, len(x), 64):
            chunk = x[start:start+64]
            indices = np.argpartition(cdist(chunk, reference), 1, axis=1)[:, 0]
            tiles = tile_data[indices]
            with self.torch.inference_mode():
                y = model(['gelu' if kind == 'VEC' else kind.lower()],
                          self.torch.from_numpy(chunk), tiles=self.torch.from_numpy(tiles))
            output.extend(y.reshape(-1).numpy()*1000)
            matched.extend(indices.tolist())
        return np.asarray(output), dict(metadata_rows=len(frame), matched_indices=matched)
