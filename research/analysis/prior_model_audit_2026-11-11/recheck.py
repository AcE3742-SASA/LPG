"""Fold-local scaling audit and fixed-target-budget comparison. Never edit v1."""
import argparse
from collections import defaultdict
import importlib.util
import itertools
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'analysis/prior_model_comparison_2026-11-11'
sys.path.insert(0,str(OLD))
import adapters as a
import compare as c
import gemm_compare as g
from prepare import save,sha,REFERENCE


def fold_fit(x,y,groups,seed):
    start=time.perf_counter()
    scaler=StandardScaler().fit(x)
    z=scaler.transform(x)
    maximum=np.max(np.abs(z.T@(y-y.mean())))/len(y)
    eps=.01 if x.shape[1]>len(y) else .0001
    alphas=np.geomspace(max(maximum,np.finfo(float).eps),max(maximum,np.finfo(float).eps)*eps,100)
    cv=list(GroupKFold(min(10,len(set(groups))),shuffle=True,random_state=seed).split(x,y,groups))
    losses=[];fold_info=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        for tr,va in cv:
            assert not set(groups[tr])&set(groups[va])
            local=StandardScaler().fit(x[tr])
            grid,coef,gaps=lasso_path(local.transform(x[tr]),y[tr]-y[tr].mean(),
                                    alphas=alphas,max_iter=100000,tol=1e-5)
            assert np.array_equal(grid,alphas)
            pred=local.transform(x[va])@coef+y[tr].mean()
            losses.append(np.mean((pred-y[va,None])**2,axis=0))
            fold_info.append(dict(train=len(tr),validation=len(va),scaler_samples=int(local.n_samples_seen_),group_overlap=0))
        mean=np.mean(losses,axis=0)
        best=int(np.argmin(mean))
        model=Lasso(alpha=float(alphas[best]),max_iter=100000,tol=1e-5).fit(z,y)
    return dict(kind='neuralpower',model=model,scaler=scaler,floor=float(y.min())),dict(
        fit_seconds=time.perf_counter()-start,alpha=float(alphas[best]),alpha_index=best,alphas=alphas.tolist(),
        cv_mean_mse=mean.tolist(),folds=fold_info,convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught),
        selected_dual_gap=float(model.dual_gap_),iterations=int(model.n_iter_))


def fit(cases,y,seed):
    a.validate_training(cases,y)
    return fold_fit(a.neuralpower_features(cases),y,np.array([x['group_id'] for x in cases]),seed)


def self_check():
    rng=np.random.default_rng(67)
    x=rng.normal(size=(80,5));x[:8]*=20
    y=np.maximum(1,5+x@np.array([.2,.1,.03,0,0])+rng.normal(0,.1,80))
    groups=np.repeat(np.arange(40),2)
    model,detail=fold_fit(x,y,groups,20261104)
    cv=GroupKFold(10,shuffle=True,random_state=20261104)
    reference=GridSearchCV(make_pipeline(StandardScaler(),Lasso(max_iter=100000,tol=1e-5)),
        {'lasso__alpha':detail['alphas']},cv=cv,scoring='neg_mean_squared_error',n_jobs=1)
    reference.fit(x,y,groups=groups)
    np.testing.assert_allclose(detail['cv_mean_mse'],-reference.cv_results_['mean_test_score'],rtol=2e-3,atol=1e-6)
    assert detail['alpha']==reference.best_params_['lasso__alpha']
    np.testing.assert_allclose(model['model'].predict(model['scaler'].transform(x)),reference.predict(x),rtol=1e-8,atol=1e-9)
    save(HERE/'fold_scaling_self_check.json',dict(status='PASS',reference='GridSearchCV(StandardScaler + Lasso)',
        folds=10,group_pairs_kept_together=True,selected_alpha=detail['alpha'],max_cv_mse_difference=float(np.max(np.abs(np.array(detail['cv_mean_mse'])+reference.cv_results_['mean_test_score'])))))
    print('PASS: warm-start CV matches independent sklearn Pipeline search',flush=True)


def owner():
    spec=importlib.util.spec_from_file_location('audit_original_layer_pipeline',REFERENCE/'pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def train():
    out=HERE/'rerun';out.mkdir(exist_ok=False);(out/'models').mkdir()
    self_check()
    p=c.read(OLD/'protocol.json');dev=c.read(OLD/'development.json')
    scores,predictions,refs,details=[],[],[],[]
    start=time.perf_counter()
    for dtype,target,seed,k,family in itertools.product(p['dtypes'],p['targets'],p['seeds'],p['budgets'],['fc','conv']):
        training=c.chosen_training(dev,p,dtype,target,seed,k,family)
        model,detail=fit([r['case'] for r in training],np.array([r['actual_ms'] for r in training]),seed)
        ref=f'models/np_{target}_{dtype}_{family}_{seed}_{k}.joblib';joblib.dump(model,out/ref)
        info=dict(dtype=dtype,target=target,seed=seed,k=k,train_labels=len(training),model_ref=ref,method='neuralpower_fold_scaled')
        refs.append(dict(info,family=family,train_keys=[[r['case']['group_id'],r['case']['layer'],dtype,target] for r in training],sha256=sha(out/ref)))
        details.append(dict(info,**detail))
        valid=[r for r in dev if r['dtype']==dtype and r['target']==target and r['case']['split']=='validation' and r['case']['layer'] in {x['case']['layer'] for x in training}]
        for floor in [False,True]:
            meta=dict(info,method='neuralpower_original_floor' if floor else info['method'])
            c.add_prediction(scores,predictions,meta,valid,a.predict_fit(model,[r['case'] for r in valid],floor))
        if family=='conv':print(f'PASS: layer NP {dtype} {target} seed={seed} k={k}',flush=True)
    pipe=owner()
    candidates=[r for r in c.read(REFERENCE/'prepared_layer_full/aggregate_metrics.json') if r['method']=='direct' and r['k']==200 and r['layer'] in p['common_layers'] and not r['failures']]
    grouped=defaultdict(list)
    for r in candidates:grouped[r['target'],r['layer'],r['dtype']].append(r)
    choices=[min(rs,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['learner'])) for rs in grouped.values()]
    oldpred=c.read(REFERENCE/'prepared_layer_full/validation_predictions.json')
    oldidx={(r['target'],r['layer'],r['dtype'],r['learner'],r['seed'],r['group_id']):r['predicted_ms'] for r in oldpred if r['method']=='direct' and r['k']==200}
    for chosen,seed in itertools.product(choices,p['seeds']):
        dtype,target,layer,learner=(chosen[k] for k in ['dtype','target','layer','learner'])
        family='conv' if layer.startswith('conv') else 'core'
        index={r['case']['group_id']:r for r in dev if r['dtype']==dtype and r['target']==target and r['case']['layer']==layer and r['case']['split']=='train'}
        training=[index[i] for i in p['subsets']['families'][family][str(seed)]['by_k']['200']]
        cases=[r['case'] for r in training];y=np.array([r['actual_ms'] for r in training])
        model,detail=pipe.previous.train(learner,'direct',pipe.features(cases,dtype),y,np.ones(len(y)),seed)
        envelope=dict(model=model,method='direct',seed=seed,learner=learner)
        ref=f'models/direct_{target}_{dtype}_{layer}_{seed}.joblib';joblib.dump(envelope,out/ref)
        info=dict(target=target,dtype=dtype,seed=seed,k=200,train_labels=len(y),model_ref=ref,method='ours_direct_k200',learner=learner)
        refs.append(dict(info,family=layer,train_keys=[[r['case']['group_id'],layer,dtype,target] for r in training],sha256=sha(out/ref)))
        details.append(dict(info,**detail))
        valid=[r for r in dev if r['dtype']==dtype and r['target']==target and r['case']['layer']==layer and r['case']['split']=='validation']
        values=pipe.predict(envelope,[r['case'] for r in valid],dtype,None)
        expected=[oldidx[target,layer,dtype,learner,seed,r['case']['group_id']] for r in valid]
        np.testing.assert_allclose(values,expected,rtol=1e-10,atol=1e-10)
        c.add_prediction(scores,predictions,info,valid,values)
    agg=c.aggregate(scores);grouped=defaultdict(list)
    for r in agg:
        if not r['failures']:grouped[r['target'],r['layer'],r['dtype'],r['method']].append(r)
    selected=[min(rs,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['mean_train_labels'],r['k'])) for rs in grouped.values()]
    for name,rows in [('validation_metrics',scores),('validation_predictions',predictions),('validation_aggregate',agg),('selected',selected),('model_references',refs),('diagnostics',details)]:save(out/(name+'.json'),rows)
    print('PASS: layer choices frozen; direct k=200 refits reproduce saved validation predictions',flush=True)
    freeze(out,start)
    train_gemm()


def freeze(out,start):
    save(out/'frozen.json',dict(status='FROZEN',wall_seconds=time.perf_counter()-start,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        code={str(p):sha(p) for p in [HERE/'recheck.py',HERE/'protocol.json',OLD/'adapters.py',OLD/'compare.py']},
        original_test_previously_disclosed=True,new_test_scoring_started=False))


def train_gemm():
    out=HERE/'gemm';out.mkdir(exist_ok=False);(out/'models').mkdir()
    spec=importlib.util.spec_from_file_location('original_expansion_audit',g.EXPANSION/'compare_expansion.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    p=c.read(g.EXPANSION/'fit_v1/frozen_protocol.json')
    labels,validations,access=original.source_labels(p)
    val={**validations['legacy'],**validations['expansion']}
    shapes=list(map(tuple,p['legacy_validation_shapes']+p['new_validation_shapes']))
    scores,predictions,refs,details=[],[],[],[];start=time.perf_counter()
    for dtype,mode,seed in itertools.product(['fp16','fp32'],['allocate','prealloc','graph'],p['seeds']):
        train_shapes=list(map(tuple,p['old_train_orders'][str(seed)]+p['new_train_orders'][str(seed)]))
        model,detail=fit(g.cases(train_shapes,'train'),np.array([labels[s,dtype,mode] for s in train_shapes]),seed)
        ref=f'models/{dtype}_{mode}_{seed}.joblib';joblib.dump(model,out/ref)
        info=dict(dtype=dtype,mode=mode,seed=seed,train_labels=11498,method='neuralpower_fold_scaled',model_ref=ref)
        refs.append(dict(info,sha256=sha(out/ref)));details.append(dict(info,**detail))
        for floor in [False,True]:
            g.record(scores,predictions,dict(info,method='neuralpower_original_floor' if floor else info['method']),shapes,
                     [val[s,dtype,mode] for s in shapes],a.predict_fit(model,g.cases(shapes,'validation'),floor))
        print(f'PASS: GEMM fold-local scaling {dtype} {mode} seed={seed}',flush=True)
    for name,rows in [('validation_metrics',scores),('validation_predictions',predictions),('models',refs),('diagnostics',details)]:save(out/(name+'.json'),rows)
    freeze(out,start)


def verify(out):
    frozen=c.read(out/'frozen.json')
    for name,digest in frozen['files'].items():assert sha(out/name)==digest,name
    for name,digest in frozen['code'].items():assert sha(Path(name))==digest,name


def evaluate():
    out=HERE/'rerun';verify(out);dest=out/'evaluation';dest.mkdir(exist_ok=False)
    rows=c.test_rows();scores,predictions=[],[];pipe=owner()
    requests={tuple(r[k] for k in ['target','layer','dtype','method','k']) for r in c.read(out/'selected.json')}
    for ref in c.read(out/'model_references.json'):
        model=joblib.load(out/ref['model_ref'])
        for floor in ([False,True] if ref['method']=='neuralpower_fold_scaled' else [False]):
            method='neuralpower_original_floor' if floor else ref['method']
            subset=[r for r in rows if r['target']==ref['target'] and r['dtype']==ref['dtype'] and r['case']['layer'] in ['linear','conv1','conv3']
                and (r['case']['layer']==ref['family'] if method=='ours_direct_k200' else (ref['family']=='conv')==r['case']['layer'].startswith('conv'))
                and (ref['k']==200 or (r['target'],r['case']['layer'],r['dtype'],method,ref['k']) in requests)]
            if not subset:continue
            cases=[r['case'] for r in subset]
            values=pipe.predict(model,cases,ref['dtype'],None) if method=='ours_direct_k200' else a.predict_fit(model,cases,floor)
            info={k:ref[k] for k in ['target','dtype','seed','k','train_labels','model_ref']};info['method']=method
            c.add_prediction(scores,predictions,info,subset,values)
    agg=c.aggregate(scores)
    for r in agg:r['selected_on_validation']=tuple(r[k] for k in ['target','layer','dtype','method','k']) in requests
    for name,data in [('test_metrics',scores),('test_predictions',predictions),('test_aggregate',agg)]:save(dest/(name+'.json'),data);c.csv_save(dest/(name+'.csv'),data)
    out=HERE/'gemm';verify(out);dest=out/'evaluation';dest.mkdir(exist_ok=False)
    old=c.read(OLD/'gemm/v1/evaluation/test_predictions.json');labels={}
    for r in old:
        s=tuple(map(int,r['shape'].split('x')));key=(s,r['dtype'],r['mode'])
        if key in labels:assert labels[key]==r['actual_ms']
        labels[key]=r['actual_ms']
    shapes=sorted({k[0] for k in labels});scores,predictions=[],[]
    for ref in c.read(out/'models.json'):
        model=joblib.load(out/ref['model_ref'])
        for floor in [False,True]:
            info=dict(ref,method='neuralpower_original_floor' if floor else ref['method'])
            g.record(scores,predictions,info,shapes,[labels[s,ref['dtype'],ref['mode']] for s in shapes],a.predict_fit(model,g.cases(shapes,'test_locked'),floor))
    for name,data in [('test_metrics',scores),('test_predictions',predictions),('test_aggregate',g.aggregate(scores))]:save(dest/(name+'.json'),data);c.csv_save(dest/(name+'.csv'),data)
    save(HERE/'rerun_complete.json',dict(status='PASS',layer_freeze=sha(HERE/'rerun/frozen.json'),gemm_freeze=sha(HERE/'gemm/frozen.json'),test_reuse='POST_HOC_DESCRIPTIVE'))
    print('PASS: corrected CV and equal-pool direct comparisons scored',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','self_check']);args=p.parse_args()
    {'train':train,'evaluate':evaluate,'self_check':self_check}[args.stage]()
