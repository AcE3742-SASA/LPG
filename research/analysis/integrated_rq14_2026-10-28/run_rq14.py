#!/usr/bin/env python3
"""Frozen, validation-only RQ1/RQ4 execution. No test-label reader exists."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'analysis/gemm_dataset_v1_2026-10-02/main_complete_unlocked_conditions.csv'
ROLES = ROOT / 'analysis/layer_spec_v1_2026-10-02/gemm_internal_roles.json'
SEEDS = [20261028, 20261029, 20261030]
SIZES = [100, 300, 1000, 2652]
BUDGETS = [5, 10, 30, 100, 300]
DTYPES = ['fp16', 'fp32']
MODES = ['allocate', 'prealloc', 'graph']
RF = dict(n_estimators=100, max_depth=8, min_samples_leaf=2, n_jobs=1)
FEATURES = ['log_M', 'log_N', 'log_K', 'log_F', 'log_Q', 'log_L_ms',
            'log_M_over_N', 'log_M_over_K', 'log_N_over_K',
            'M_align8', 'N_align8', 'K_align8', 'M_align16', 'N_align16', 'K_align16']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shape_id(shape):
    return 'x'.join(map(str, shape))


def ordered(shapes, salt):
    return sorted(shapes, key=lambda s: hashlib.sha256(f'{salt}:{shape_id(s)}'.encode()).hexdigest())


def metadata(data=DATA, roles=ROLES):
    """Read identifiers only; the timing field is never converted or returned."""
    role_rows = json.loads(Path(roles).read_text())
    role_map = {tuple(row['shape']): row['internal_role'] for row in role_rows}
    groups = {}
    with Path(data).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['phase'] != 'main' or row['split'] not in {'legacy_train', 'legacy_validation'}:
                continue
            shape = tuple(json.loads(row['shape']))
            if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
                raise ValueError('Invalid shape metadata')
            if int(row['accepted_rounds']) != 3:
                raise ValueError('Incomplete round metadata')
            pair = (row['dtype'], row['mode'])
            if pair not in {(d, m) for d in DTYPES for m in MODES}:
                raise ValueError('Unexpected dtype or mode')
            key = (row['split'], shape)
            bucket = groups.setdefault(key, set())
            if pair in bucket:
                raise ValueError('Duplicate condition')
            bucket.add(pair)
    complete = {key for key, pairs in groups.items() if len(pairs) == 6}
    a = sorted(s for split, s in complete if split == 'legacy_train' and role_map.get(s) == 'A_train_candidate')
    c = sorted(s for split, s in complete if split == 'legacy_train' and role_map.get(s) == 'C_calibration')
    v = sorted(s for split, s in complete if split == 'legacy_validation')
    if set(a) & set(c) or (set(a) | set(c)) & set(v):
        raise ValueError('Shape split overlap')
    return a, c, v


def load_labels(split, shapes, modes, access_log, data=DATA):
    """The sole label gateway rejects non-development split and non-allowlisted modes."""
    if split not in {'legacy_train', 'legacy_validation'} or not set(modes) <= set(MODES):
        raise ValueError('Label access denied')
    allowed = set(map(tuple, shapes))
    if not allowed:
        raise ValueError('Empty label allowlist')
    labels = {}
    with Path(data).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['phase'] != 'main' or row['split'] != split or row['mode'] not in modes:
                continue
            shape = tuple(json.loads(row['shape']))
            if shape not in allowed:
                continue
            key = (shape, row['dtype'], row['mode'])
            value = float(row['median_of_round_medians_ms'])
            if key in labels or not math.isfinite(value) or value <= 0 or int(row['accepted_rounds']) != 3:
                raise ValueError('Invalid or duplicate allowed label')
            labels[key] = value
    expected = len(allowed) * len(DTYPES) * len(modes)
    if len(labels) != expected:
        raise ValueError(f'Incomplete label allowlist: {len(labels)} != {expected}')
    access_log.append(dict(timestamp=stamp(), split=split, shapes=len(allowed), modes=list(modes),
                           conditions=len(labels), shape_ids_sha256=hashlib.sha256('\n'.join(sorted(map(shape_id, allowed))).encode()).hexdigest()))
    return labels


def design(shapes, dtype):
    dims = np.asarray(shapes, dtype=float)
    m, n, k = dims.T
    flops = 2 * m * n * k
    nbytes = (2 if dtype == 'fp16' else 4) * (m*k + k*n + m*n)
    peak = 35.02e12 if dtype == 'fp16' else 6.14e12
    lower = np.maximum(flops / peak, nbytes / 262.3e9) * 1000
    x = np.column_stack((np.log(dims), np.log(flops), np.log(nbytes), np.log(lower),
                         np.log(m/n), np.log(m/k), np.log(n/k), dims % 8 == 0, dims % 16 == 0))
    assert x.shape[1] == len(FEATURES) and np.isfinite(x).all()
    return x, lower


def target(labels, shapes, dtype, mode):
    return np.asarray([labels[(tuple(s), dtype, mode)] for s in shapes])


def fit_model(method, x, y, lower, seed, source=None):
    start = time.perf_counter()
    if method == 'constant_efficiency':
        model = float(np.mean(np.log(y/lower)))
    elif method == 'launch_scale':
        model = nnls(np.column_stack((np.ones(len(y)), lower)), y)[0]
    elif method == 'zero':
        model = None
    elif method == 'constant_ratio':
        model = float(np.mean(np.log(y/source)))
    else:
        family = 'ridge' if method.startswith('ridge_') else 'rf'
        model = make_pipeline(StandardScaler(), Ridge(alpha=1)) if family == 'ridge' else RandomForestRegressor(**RF, random_state=seed)
        response = np.log(y)
        if method.endswith('_normalized'):
            response = response - np.log(lower)
        if method == 'residual':
            response = response - np.log(source)
        features = np.column_stack((x, np.log(source))) if method in {'residual', 'prior_direct'} else x
        model.fit(features, response)
    return model, time.perf_counter() - start


def predict(method, model, x, lower, source=None):
    start = time.perf_counter()
    with np.errstate(over='ignore', invalid='ignore'):
        if method == 'constant_efficiency':
            prediction = lower * np.exp(model)
        elif method == 'launch_scale':
            prediction = np.column_stack((np.ones(len(lower)), lower)) @ model
        elif method == 'zero':
            prediction = source.copy()
        elif method == 'constant_ratio':
            prediction = source * np.exp(model)
        else:
            features = np.column_stack((x, np.log(source))) if method in {'residual', 'prior_direct'} else x
            prediction = np.exp(model.predict(features))
            if method.endswith('_normalized'):
                prediction *= lower
            if method == 'residual':
                prediction *= source
    return prediction, time.perf_counter() - start


def errors(y, prediction):
    valid = np.isfinite(prediction) & (prediction > 0)
    relative = np.full(len(y), np.inf)
    relative[valid] = np.abs(prediction[valid] - y[valid]) / y[valid]
    return valid, relative


def metrics(y, prediction):
    valid, relative = errors(y, prediction)
    good = relative[valid]
    failed = int((~valid).sum())
    return dict(planned=len(y), valid=int(valid.sum()), failures=failed, coverage=float(valid.mean()),
                mape_pct=float(np.mean(relative)*100) if not failed else math.inf,
                p90_pct=float(np.percentile(relative, 90)*100) if not failed else math.inf,
                relative_rmse=float(np.sqrt(np.mean(relative**2))) if not failed else math.inf,
                mae_us=float(np.mean(np.abs(prediction-y))*1000) if not failed else math.inf,
                finite_mape_pct=float(good.mean()*100) if len(good) else math.inf,
                meets_15_30=bool(not failed and relative.mean() <= .15 and np.percentile(relative,90) <= .30))


def subgroups(shapes):
    dims = np.asarray(shapes)
    flops = 2*np.prod(dims.astype(float), axis=1)
    return {'small_F_le_1e7':flops <= 1e7, 'medium_F_1e7_to_1e9':(flops > 1e7)&(flops <= 1e9),
            'large_F_gt_1e9':flops > 1e9, 'aspect_ge_8':dims.max(axis=1)/dims.min(axis=1) >= 8,
            'aligned8':(dims%8 == 0).all(axis=1), 'unaligned8':(dims%8 != 0).any(axis=1),
            'aligned16':(dims%16 == 0).all(axis=1)}


def bootstrap_delta(y, a, b, seed=20261028):
    va, ea = errors(y,a)
    vb, eb = errors(y,b)
    if not (va.all() and vb.all()):
        return dict(delta_mape_pp=math.inf, ci95_low_pp=math.inf, ci95_high_pp=math.inf)
    delta = (ea-eb)*100
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0,len(y),size=(1000,len(y)))].mean(axis=1)
    return dict(delta_mape_pp=float(delta.mean()),ci95_low_pp=float(np.quantile(means,.025)),ci95_high_pp=float(np.quantile(means,.975)))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    a,c,v = metadata()
    if [len(a),len(c),len(v)] != [2652,693,716]:
        raise ValueError('Source counts differ from the adopted plan')
    protocol = dict(version='rq14_v1', created_utc=stamp(), script_sha256=sha(__file__),
        sources={str(DATA.relative_to(ROOT)):sha(DATA),str(ROLES.relative_to(ROOT)):sha(ROLES)},
        seeds=SEEDS, rq1_sizes=SIZES, rq4_budgets=BUDGETS, features=FEATURES, rf=RF,
        roofline=dict(fp16_tflops=35.02,fp32_tflops=6.14,bandwidth_gbs=262.3),
        rq1_methods=['rf_direct','rf_normalized','ridge_direct','ridge_normalized','constant_efficiency','launch_scale'],
        rq4_methods=['zero','constant_ratio','residual','direct','prior_direct'],
        ridge=dict(alpha=1,standardize_with_train_only=True),
        source_selection='RF direct vs normalized at n=2652; per dtype lowest mean allocate-validation MAPE across three seeds; p90 then method name breaks ties; no target labels',
        target_selection='Per dtype/mode: smallest k whose mean across seed MAPE<=15 and mean p90<=30 with zero failures; lowest MAPE then method name; no passing candidate is reported as not attained',
        bootstrap='1000 paired shape resamples, seed 20261028; delta first minus second, negative favors first',
        label_policy='Only legacy_train A/C and legacy_validation. RQ4 target validation exactly 30 shapes. No locked reader or final-test command.',
        graph_scope='Existing adaptive q policy only; q and probe time excluded from features',
        costs='target condition-round labels=2 dtypes *3 rounds*(k+30) per mode/seed; report cross-seed union separately; historical GPU wall time unknown',
        a_shapes=a,c_shapes=c,validation_shapes=v,
        rq4_validation_shapes=ordered(v,'rq4-target-validation-v1')[:30],
        a_orders={str(seed):ordered(a,f'A:{seed}') for seed in SEEDS},
        c_orders={str(seed):ordered(c,f'C:{seed}') for seed in SEEDS},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,platform=platform.platform(),
                     cpu_count=os.cpu_count(),rf_n_jobs=1,thread_env={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},threadpools=threadpool_info()),
        test_status='LOCKED_NOT_OPENED')
    save_json(out/'frozen_protocol.json',protocol)
    ids=[]
    for seed in SEEDS:
        for n in SIZES:
            ids += [dict(pool='A',seed=seed,n=n,shape=shape_id(s)) for s in protocol['a_orders'][str(seed)][:n]]
        for n in BUDGETS:
            ids += [dict(pool='C',seed=seed,n=n,shape=shape_id(s)) for s in protocol['c_orders'][str(seed)][:n]]
    ids += [dict(pool='RQ4_target_validation',seed='',n=30,shape=shape_id(s)) for s in protocol['rq4_validation_shapes']]
    save_csv(out/'frozen_subset_ids.csv',ids)
    print('PASS: protocol and subset IDs frozen; no timing labels loaded',flush=True)


def run(out):
    p=json.loads((out/'frozen_protocol.json').read_text())
    if p['script_sha256'] != sha(__file__):
        raise ValueError('Frozen script hash mismatch')
    for path,digest in p['sources'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Frozen input hash mismatch')
    if (out/'execution_started.json').exists():
        raise ValueError('Refusing to overwrite an existing execution')
    save_json(out/'execution_started.json',dict(time=stamp(),protocol_sha256=sha(out/'frozen_protocol.json')))
    (out/'models').mkdir()
    access=[]; scores=[]; subgroup_scores=[]; fits=[]; paired=[]
    source_models={}; source_predictions={}; rq1_predictions={}
    prediction_handle=(out/'validation_predictions.csv').open('w',newline='')
    prediction_writer=csv.DictWriter(prediction_handle,fieldnames=['rq','dtype','mode','method','n','seed','split','shape','actual_ms','predicted_ms','valid'])
    prediction_writer.writeheader()
    def record(info,shapes,y,pred,fit_seconds,predict_seconds,prior_train_seconds=0,prior_validation_seconds=0):
        row={**info,**metrics(y,pred)}
        scores.append(row)
        fits.append({**info,'fit_seconds':fit_seconds,'correction_or_direct_predict_seconds':predict_seconds,
                     'source_prior_train_predict_seconds':prior_train_seconds,'source_prior_validation_predict_seconds':prior_validation_seconds,
                     'pipeline_validation_predict_seconds':predict_seconds+prior_validation_seconds,
                     'train_conditions':info['n'] if info['method']!='zero' else 0,'validation_conditions':len(shapes)})
        for name,mask in subgroups(shapes).items():
            if mask.any():
                subgroup_scores.append({**info,'subgroup':name,**metrics(y[mask],pred[mask])})
        for shape,actual,predicted,valid in zip(shapes,y,pred,errors(y,pred)[0]):
            prediction_writer.writerow({**info,'split':'legacy_validation','shape':shape_id(shape),'actual_ms':actual,'predicted_ms':predicted,'valid':bool(valid)})
        return row
    a=[tuple(s) for s in p['a_shapes']]; c=[tuple(s) for s in p['c_shapes']]; v=[tuple(s) for s in p['validation_shapes']]
    # Freeze has already happened; begin with source-only labels and selection.
    source_labels=load_labels('legacy_train',a,['allocate'],access)
    source_val=load_labels('legacy_validation',v,['allocate'],access)
    start=time.perf_counter()
    def rq1_mode(mode,training,validation):
        for dtype in DTYPES:
            xv,lv=design(v,dtype); yv=target(validation,v,dtype,mode)
            for seed in SEEDS:
                order=[tuple(s) for s in p['a_orders'][str(seed)]]
                for n in SIZES:
                    shapes=order[:n]; x,l=design(shapes,dtype); y=target(training,shapes,dtype,mode)
                    predictions={}
                    for method in p['rq1_methods']:
                        model,seconds=fit_model(method,x,y,l,seed)
                        pred,ps=predict(method,model,xv,lv)
                        info=dict(rq='RQ1',dtype=dtype,mode=mode,method=method,n=n,seed=seed)
                        record(info,v,yv,pred,seconds,ps)
                        predictions[method]=pred
                        if n == 2652 and mode == 'allocate' and method.startswith('rf_'):
                            source_models[(dtype,seed,method)]=model
                        if n == 2652:
                            rq1_predictions[(dtype,mode,seed,method)]=pred
                    for family in ['rf','ridge']:
                        paired.append(dict(rq='RQ1',dtype=dtype,mode=mode,n=n,seed=seed,first=f'{family}_normalized',second=f'{family}_direct',**bootstrap_delta(yv,predictions[f'{family}_normalized'],predictions[f'{family}_direct'])))
            print(f'PASS: RQ1 {dtype} {mode}',flush=True)
    rq1_mode('allocate',source_labels,source_val)
    selected_source={}
    source_selection=[]
    for dtype in DTYPES:
        candidates=[]
        for method in ['rf_direct','rf_normalized']:
            rows=[r for r in scores if r['rq']=='RQ1' and r['dtype']==dtype and r['mode']=='allocate' and r['n']==2652 and r['method']==method]
            candidates.append((float(np.mean([r['mape_pct'] for r in rows])),float(np.mean([r['p90_pct'] for r in rows])),method))
        mape,p90,method=min(candidates)
        selected_source[dtype]=method
        source_selection.append(dict(dtype=dtype,method=method,n=2652,mean_validation_mape_pct=mape,mean_validation_p90_pct=p90,selection_label_scope='allocate_only'))
        for seed in SEEDS:
            joblib.dump(source_models[(dtype,seed,method)],out/'models'/f'source_{dtype}_{seed}_{method}.joblib')
    save_json(out/'source_selection_before_target_labels.json',dict(time=stamp(),protocol_sha256=sha(out/'frozen_protocol.json'),selection=source_selection))
    print('PASS: source selection frozen using allocate validation only',flush=True)
    c_consumed=sorted({tuple(s) for seed in SEEDS for s in p['c_orders'][str(seed)][:max(BUDGETS)]})
    calibration=load_labels('legacy_train',c_consumed,['prealloc','graph'],access)
    v30=[tuple(s) for s in p['rq4_validation_shapes']]
    target_val=load_labels('legacy_validation',v30,['prealloc','graph'],access)
    for dtype in DTYPES:
        xv,lv=design(v30,dtype)
        for mode in ['prealloc','graph']:
            yv=target(target_val,v30,dtype,mode)
            for seed in SEEDS:
                source=source_models[(dtype,seed,selected_source[dtype])]
                pv,pv_seconds=predict(selected_source[dtype],source,xv,lv)
                order=[tuple(s) for s in p['c_orders'][str(seed)]]
                for n in BUDGETS:
                    shapes=order[:n]; x,l=design(shapes,dtype); y=target(calibration,shapes,dtype,mode)
                    prior,prior_seconds=predict(selected_source[dtype],source,x,l)
                    predictions={}
                    for method in p['rq4_methods']:
                        model,seconds=fit_model(method,x,y,l,seed,prior)
                        pred,ps=predict(method,model,xv,lv,pv)
                        info=dict(rq='RQ4',dtype=dtype,mode=mode,method=method,n=n,seed=seed)
                        uses_prior=method!='direct'
                        record(info,v30,yv,pred,seconds,ps,
                               prior_seconds if uses_prior and method!='zero' else 0,pv_seconds if uses_prior else 0)
                        predictions[method]=pred
                    for first,second in [('residual','direct'),('prior_direct','direct'),('constant_ratio','zero'),('residual','prior_direct')]:
                        paired.append(dict(rq='RQ4',dtype=dtype,mode=mode,n=n,seed=seed,first=first,second=second,**bootstrap_delta(yv,predictions[first],predictions[second])))
            print(f'PASS: RQ4 {dtype} allocate to {mode}',flush=True)
    # Full target validation is used only for RQ1 after all RQ4 algorithms are frozen.
    training=load_labels('legacy_train',a,['prealloc','graph'],access)
    validation=load_labels('legacy_validation',v,['prealloc','graph'],access)
    for mode in ['prealloc','graph']:
        rq1_mode(mode,training,validation)
    prediction_handle.close()
    save_csv(out/'metrics.csv',scores); save_csv(out/'subgroup_metrics.csv',subgroup_scores)
    save_csv(out/'fit_times.csv',fits); save_csv(out/'paired_bootstrap.csv',paired)
    save_json(out/'label_access_log.json',access)
    save_csv(out/'source_selection.csv',source_selection)
    groups={}
    for row in scores:
        key=(row['rq'],row['dtype'],row['mode'],row['method'],row['n'])
        groups.setdefault(key,[]).append(row)
    aggregates=[]
    for key,rows in groups.items():
        rq,dtype,mode,method,n=key
        aggregates.append(dict(rq=rq,dtype=dtype,mode=mode,method=method,n=n,seeds=len(rows),
            mean_mape_pct=float(np.mean([r['mape_pct'] for r in rows])),mean_p90_pct=float(np.mean([r['p90_pct'] for r in rows])),
            failures=sum(r['failures'] for r in rows),coverage=float(np.mean([r['coverage'] for r in rows]))))
    save_csv(out/'aggregate_metrics.csv',aggregates)
    selections=[]
    for rq in ['RQ1','RQ4']:
        for dtype in DTYPES:
            for mode in MODES if rq=='RQ1' else ['prealloc','graph']:
                candidates=[r for r in aggregates if r['rq']==rq and r['dtype']==dtype and r['mode']==mode]
                passing=[r for r in candidates if r['mean_mape_pct']<=15 and r['mean_p90_pct']<=30 and not r['failures']]
                chosen=min(passing,key=lambda r:(r['n'],r['mean_mape_pct'],r['method'])) if passing else min(candidates,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['method']))
                effective_n=0 if chosen['method']=='zero' else chosen['n']
                selections.append({**chosen,'goal_attained':bool(passing),'effective_target_calibration_shapes':effective_n if rq=='RQ4' else '',
                    'target_validation_shapes':30 if rq=='RQ4' else 716,
                    'target_labels_per_mode_all_dtypes_three_rounds':6*(effective_n+30) if rq=='RQ4' else ''})
    save_csv(out/'selected_configurations_validation_only.csv',selections)
    cost=[]
    for n in BUDGETS:
        union=set(tuple(s) for seed in SEEDS for s in p['c_orders'][str(seed)][:n])
        cost.append(dict(k=n,seeds=3,per_seed_target_shape_count=n+30,per_seed_labels_one_mode_two_dtypes_three_rounds=6*(n+30),
            union_calibration_shapes=len(union),union_target_validation_shapes=30,union_labels_one_mode_two_dtypes_three_rounds=6*(len(union)+30),
            union_labels_both_modes_two_dtypes_three_rounds=12*(len(union)+30),historical_gpu_wall_time='UNKNOWN'))
    save_csv(out/'target_label_costs.csv',cost)
    save_json(out/'execution_completed.json',dict(time=stamp(),wall_seconds=time.perf_counter()-start,configurations=len(scores),
        rf_fits=sum(r['method'].startswith('rf_') or r['method'] in {'residual','direct','prior_direct'} for r in fits),
        total_fit_seconds=sum(r['fit_seconds'] for r in fits),locked_test_status='NOT_OPENED',source_hashes_unchanged=all(sha(ROOT/path)==digest for path,digest in p['sources'].items())))
    print(f'PASS: {len(scores)} validation configurations completed; locked test NOT OPENED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=HERE/'runs/v1')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.output)
    else:
        run(args.output)


if __name__=='__main__':
    main()
