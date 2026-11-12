#!/usr/bin/env python3
"""Matched RQ1/RQ4 learner comparison using development labels only."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'analysis/integrated_rq14_2026-10-28'
sys.path.insert(0, str(BASE))
import run_rq14 as old

LEARNERS = ['rf', 'ridge', 'mlp32', 'mlp64']
TARGETS = ['direct', 'normalized']
MLP = dict(activation='relu', solver='adam', alpha=0.0001, batch_size=128,
           learning_rate_init=0.001, max_iter=3000, tol=1e-6,
           n_iter_no_change=100, early_stopping=False, shuffle=True)


def fit(learner, x, response, seed):
    if learner == 'rf':
        model = RandomForestRegressor(**old.RF, random_state=seed)
    elif learner == 'ridge':
        model = make_pipeline(StandardScaler(), Ridge(alpha=1))
    elif learner in {'mlp32', 'mlp64'}:
        width = int(learner[3:])
        params = dict(MLP, batch_size=min(MLP['batch_size'], len(response)))
        model = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(width, width), random_state=seed, **params)),
            transformer=StandardScaler())
    else:
        raise ValueError('Unapproved learner')
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, response)
    seconds = time.perf_counter() - start
    neural = model.regressor_.named_steps['mlpregressor'] if learner.startswith('mlp') else None
    diagnostic = dict(fit_seconds=seconds, epochs=neural.n_iter_ if neural else '',
        final_scaled_loss=float(neural.loss_) if neural else '',
        convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        warning_messages=' | '.join(str(w.message) for w in caught),
        parameter_count=sum(w.size + b.size for w, b in zip(neural.coefs_, neural.intercepts_)) if neural else '')
    return model, diagnostic


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    previous = BASE / 'runs/v2/frozen_protocol.json'
    p = json.loads(previous.read_text())
    a, c, v = old.metadata()
    if [len(a), len(c), len(v)] != [2652, 693, 716]:
        raise ValueError('Adopted source shape counts changed')
    protocol = dict(version='matched_learners_v1', created_utc=old.stamp(),
        sources={str(path.relative_to(ROOT)): old.sha(path) for path in
                 [Path(__file__), BASE/'run_rq14.py', old.DATA, old.ROLES, previous,
                  BASE/'requirements.txt']},
        features=old.FEATURES, learners=LEARNERS, rq1_targets=TARGETS,
        rf=old.RF, ridge=dict(alpha=1, x_scaler='training subset only'), mlp=MLP,
        mlp_widths=[32,64], mlp_scalers='X and log-response standardized on the training subset only',
        mlp_stopping='Training-loss patience; no outer-validation early stopping or hyperparameter search. All n labels used for fitting.',
        seeds=old.SEEDS, rq1_sizes=old.SIZES+[3345], rq4_budgets=old.BUDGETS,
        rq4_methods=['zero','constant_ratio','residual','direct','prior_direct'],
        a_shapes=p['a_shapes'], c_shapes=p['c_shapes'], validation_shapes=p['validation_shapes'],
        a_orders=p['a_orders'], c_orders=p['c_orders'], rq4_validation_shapes=p['rq4_validation_shapes'],
        full_train_order='Existing A order followed by C order; full endpoint for RQ1 only, never RQ4 source.',
        source_selection='Within each learner/dtype, direct vs normalized at A=2652, mean allocate-validation MAPE then p90 then target name over three seeds.',
        rq4_selection='Minimum effective calibration n meeting mean MAPE<=15 and p90<=30, zero failures; then MAPE, p90, method.',
        fairness='Identical shapes, seeds, features, labels and evaluation within each comparison. One frozen configuration per learner; two MLP capacities. Numerical preprocessing differs by learner and is recorded.',
        caveat='Development validation was already examined in earlier runs. No independent test claim or equal compute-budget claim.',
        locked_test='NOT_OPENED; only the original unlocked label gateway is used',
        runtime=dict(python=sys.version, numpy=np.__version__, sklearn=old.sklearn.__version__,
                     thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}))
    old.save_json(out/'frozen_protocol.json', protocol)
    (out/'source').mkdir()
    for path in [Path(__file__), HERE/'selfcheck.py']:
        shutil.copy2(path, out/'source'/path.name)
    print('PASS: four learner configurations frozen before new fits; locked test closed', flush=True)


def run(out):
    p = json.loads((out/'frozen_protocol.json').read_text())
    if any(old.sha(ROOT/path) != digest for path,digest in p['sources'].items()):
        raise ValueError('Frozen source or data changed')
    if (out/'execution_started.json').exists():
        raise ValueError('Refusing to overwrite existing execution')
    old.save_json(out/'execution_started.json', dict(time=old.stamp(), protocol_sha256=old.sha(out/'frozen_protocol.json')))
    (out/'models').mkdir()
    access, scores, diagnostics, subgroup, source_rows = [], [], [], [], []
    source_models, source_targets, rq1_cache = {}, {}, {}
    a = list(map(tuple,p['a_shapes'])); c = list(map(tuple,p['c_shapes']))
    v = list(map(tuple,p['validation_shapes'])); v30 = list(map(tuple,p['rq4_validation_shapes']))
    started = time.perf_counter()
    prediction_path = out/'validation_predictions.csv'
    handle = prediction_path.open('w', newline='')
    writer = csv.DictWriter(handle,fieldnames=['rq','learner','dtype','mode','method','n','seed','shape','actual_ms','predicted_ms'])
    writer.writeheader()

    def record(info, shapes, y, predicted, detail, train_y=None, train_pred=None):
        score = {**info, **old.metrics(y, predicted)}
        scores.append(score)
        row = {**info, **detail}
        if train_y is not None:
            row.update({f'train_{k}':value for k,value in old.metrics(train_y,train_pred).items()})
        diagnostics.append(row)
        for name, mask in old.subgroups(shapes).items():
            if mask.any():
                subgroup.append({**info,'subgroup':name,**old.metrics(y[mask],predicted[mask])})
        for shape, actual, pred in zip(shapes,y,predicted):
            writer.writerow({**info,'shape':old.shape_id(shape),'actual_ms':actual,'predicted_ms':pred})

    def checkpoint(message):
        handle.flush()
        old.save_csv(out/'metrics.csv',scores)
        # Diagnostics share a fixed union of keys even for zero/ratio baselines.
        keys = list(dict.fromkeys(k for row in diagnostics for k in row))
        old.save_csv(out/'fit_diagnostics.csv',[{k:row.get(k,'') for k in keys} for row in diagnostics])
        old.save_json(out/'progress.json',dict(message=message, configurations=len(scores), elapsed_seconds=time.perf_counter()-started))
        old.save_json(out/'label_access_log.json',access)
        print(f'PASS: {message}; configurations={len(scores)} elapsed_s={time.perf_counter()-started:.1f}',flush=True)

    def rq1(mode, training, validation, sizes):
        for dtype in old.DTYPES:
            xv, lv = old.design(v,dtype); yv = old.target(validation,v,dtype,mode)
            for seed in old.SEEDS:
                order = list(map(tuple,p['a_orders'][str(seed)]+p['c_orders'][str(seed)]))
                for n in sizes:
                    shapes = order[:n]; x, lower = old.design(shapes,dtype)
                    y = old.target(training,shapes,dtype,mode)
                    for learner in LEARNERS:
                        for target in TARGETS:
                            response = np.log(y) - (np.log(lower) if target=='normalized' else 0)
                            model, detail = fit(learner,x,response,seed)
                            before = time.perf_counter()
                            with np.errstate(over='ignore',invalid='ignore'):
                                predicted = np.exp(model.predict(xv))*(lv if target=='normalized' else 1)
                                train_pred = np.exp(model.predict(x))*(lower if target=='normalized' else 1)
                            detail['predict_seconds'] = time.perf_counter()-before
                            info = dict(rq='RQ1',learner=learner,dtype=dtype,mode=mode,method=target,n=n,seed=seed)
                            record(info,v,yv,predicted,detail,y,train_pred)
                            if n==2652 and mode=='allocate':
                                source_models[(learner,dtype,seed,target)] = model
                            if n in {2652,3345}:
                                envelope = dict(model=model, learner=learner, target=target, features=old.FEATURES,
                                    dtype=dtype,mode=mode,n=n,seed=seed,train_shapes=shapes,
                                    protocol_sha256=old.sha(out/'frozen_protocol.json'))
                                name=f'rq1_{learner}_{dtype}_{mode}_{target}_{n}_{seed}.joblib'
                                joblib.dump(envelope,out/'models'/name)
                                assert np.array_equal(model.predict(xv),joblib.load(out/'models'/name)['model'].predict(xv))
                            rq1_cache[(learner,dtype,mode,target,n,seed)] = predicted
                checkpoint(f'RQ1 {dtype} {mode} seed={seed} n={sizes}')

    allocate_train = old.load_labels('legacy_train',a,['allocate'],access)
    allocate_val = old.load_labels('legacy_validation',v,['allocate'],access)
    rq1('allocate',allocate_train,allocate_val,old.SIZES)
    for learner in LEARNERS:
        for dtype in old.DTYPES:
            candidates=[]
            for target in TARGETS:
                rows=[r for r in scores if r['rq']=='RQ1' and r['learner']==learner and r['dtype']==dtype and r['mode']=='allocate' and r['n']==2652 and r['method']==target]
                candidates.append((np.mean([r['mape_pct'] for r in rows]),np.mean([r['p90_pct'] for r in rows]),target))
            mape,p90,target=min(candidates)
            source_targets[(learner,dtype)]=target
            source_rows.append(dict(learner=learner,dtype=dtype,target=target,n=2652,mean_mape_pct=float(mape),mean_p90_pct=float(p90)))
    old.save_json(out/'source_selection_before_target_access.json',dict(time=old.stamp(),selection=source_rows))
    consumed=sorted({tuple(s) for seed in old.SEEDS for s in p['c_orders'][str(seed)][:max(old.BUDGETS)]})
    calibration=old.load_labels('legacy_train',consumed,['prealloc','graph'],access)
    target_val=old.load_labels('legacy_validation',v30,['prealloc','graph'],access)
    for dtype in old.DTYPES:
        xv,lv=old.design(v30,dtype)
        for mode in ['prealloc','graph']:
            yv=old.target(target_val,v30,dtype,mode)
            for seed in old.SEEDS:
                for learner in LEARNERS:
                    starget=source_targets[(learner,dtype)]
                    source=source_models[(learner,dtype,seed,starget)]
                    before=time.perf_counter()
                    pv=np.exp(source.predict(xv))*(lv if starget=='normalized' else 1)
                    pv_seconds=time.perf_counter()-before
                    for n in old.BUDGETS:
                        shapes=list(map(tuple,p['c_orders'][str(seed)][:n])); x,lower=old.design(shapes,dtype)
                        y=old.target(calibration,shapes,dtype,mode)
                        before=time.perf_counter()
                        prior=np.exp(source.predict(x))*(lower if starget=='normalized' else 1)
                        prior_seconds=time.perf_counter()-before
                        for method in p['rq4_methods']:
                            detail=dict(fit_seconds=0.,epochs='',final_scaled_loss='',convergence_warnings=0,warning_messages='',parameter_count='')
                            before=time.perf_counter()
                            if method=='zero':
                                pred=pv.copy(); train_pred=prior.copy()
                            elif method=='constant_ratio':
                                scale=float(np.mean(np.log(y/prior)))
                                detail['fit_seconds']=time.perf_counter()-before
                                pred=pv*np.exp(scale); train_pred=prior*np.exp(scale)
                            else:
                                xp=np.column_stack((x,np.log(prior))) if method!='direct' else x
                                xvp=np.column_stack((xv,np.log(pv))) if method!='direct' else xv
                                response=np.log(y)-(np.log(prior) if method=='residual' else 0)
                                model,detail=fit(learner,xp,response,seed)
                                before=time.perf_counter()
                                with np.errstate(over='ignore',invalid='ignore'):
                                    pred=np.exp(model.predict(xvp))*(pv if method=='residual' else 1)
                                    train_pred=np.exp(model.predict(xp))*(prior if method=='residual' else 1)
                            detail['predict_seconds']=time.perf_counter()-before
                            detail['source_validation_predict_seconds']=pv_seconds if method!='direct' else 0
                            detail['source_train_predict_seconds']=prior_seconds if method not in {'direct','zero'} else 0
                            info=dict(rq='RQ4',learner=learner,dtype=dtype,mode=mode,method=method,n=n,seed=seed)
                            record(info,v30,yv,pred,detail,y,train_pred)
                checkpoint(f'RQ4 {dtype} {mode} seed={seed}')
    for mode in ['prealloc','graph']:
        training=old.load_labels('legacy_train',a,[mode],access)
        validation=old.load_labels('legacy_validation',v,[mode],access)
        rq1(mode,training,validation,old.SIZES)
    all_train=old.load_labels('legacy_train',a+c,old.MODES,access)
    all_val=old.load_labels('legacy_validation',v,old.MODES,access)
    for mode in old.MODES:
        rq1(mode,all_train,all_val,[3345])
    handle.close()
    old.save_csv(out/'subgroup_metrics.csv',subgroup)
    grouped=defaultdict(list)
    for row in scores:
        grouped[tuple(row[k] for k in ['rq','learner','dtype','mode','method','n'])].append(row)
    aggregate=[]
    for key,rows in sorted(grouped.items()):
        assert len(rows)==3
        row=dict(zip(['rq','learner','dtype','mode','method','n'],key))
        for metric in ['mape_pct','p90_pct','mae_us']:
            values=[r[metric] for r in rows]
            row['mean_'+metric]=float(np.mean(values)); row['sd_'+metric]=float(np.std(values,ddof=1))
        row['failures']=sum(r['failures'] for r in rows)
        aggregate.append(row)
    old.save_csv(out/'aggregate_metrics.csv',aggregate)
    selections=[]
    for learner in LEARNERS:
        for dtype in old.DTYPES:
            for mode in ['prealloc','graph']:
                rows=[r for r in aggregate if r['rq']=='RQ4' and r['learner']==learner and r['dtype']==dtype and r['mode']==mode]
                passing=[r for r in rows if r['mean_mape_pct']<=15 and r['mean_p90_pct']<=30 and r['failures']==0]
                cost=lambda r:0 if r['method']=='zero' else r['n']
                chosen=min(passing,key=lambda r:(cost(r),r['mean_mape_pct'],r['mean_p90_pct'],r['method'])) if passing else min(rows,key=lambda r:r['mean_mape_pct'])
                selections.append({**chosen,'effective_calibration_shapes':cost(chosen),'validation_shapes':30,'goal_attained':bool(passing)})
    old.save_csv(out/'rq4_selections_validation_only.csv',selections)
    paired=[]
    for learner in LEARNERS:
        for dtype in old.DTYPES:
            for mode in old.MODES:
                yv=old.target(all_val,v,dtype,mode)
                for n in p['rq1_sizes']:
                    for seed in old.SEEDS:
                        norm=rq1_cache[(learner,dtype,mode,'normalized',n,seed)]
                        direct=rq1_cache[(learner,dtype,mode,'direct',n,seed)]
                        paired.append(dict(learner=learner,dtype=dtype,mode=mode,n=n,seed=seed,first='normalized',second='direct',**old.bootstrap_delta(yv,norm,direct)))
    old.save_csv(out/'paired_target_bootstrap.csv',paired)
    old.save_json(out/'execution_completed.json',dict(time=old.stamp(),configurations=len(scores),
        wall_seconds=time.perf_counter()-started,total_fit_seconds=sum(r['fit_seconds'] for r in diagnostics),
        convergence_warning_fits=sum(bool(r['convergence_warnings']) for r in diagnostics),
        invalid_predictions=sum(r['failures'] for r in scores),locked_test='NOT_OPENED',
        source_hashes_unchanged=all(old.sha(ROOT/path)==digest for path,digest in p['sources'].items())))
    hashes={str(path.relative_to(out)):old.sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    old.save_json(out/'artifact_hashes.json',hashes)
    print(f'PASS: {len(scores)} matched validation configurations complete; locked test NOT OPENED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=HERE/'runs/v1')
    args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args.output)
