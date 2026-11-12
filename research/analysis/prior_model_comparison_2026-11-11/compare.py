"""Train paper baselines on frozen development splits, then evaluate saved tests."""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np

import adapters as a
from prepare import HERE, ROOT, REFERENCE, save, sha


def read(path):
    return json.loads(Path(path).read_text())


def csv_save(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict,list)) else v for k,v in r.items()} for r in rows)


def score(actual, predicted):
    y, pred = np.asarray(actual), np.asarray(predicted)
    valid = np.isfinite(pred) & (pred > 0)
    if not valid.all():
        return dict(groups=len(y), failures=int((~valid).sum()), mape_pct=None, p90_pct=None, rmspe_pct=None)
    error = np.abs(pred/y-1)*100
    return dict(groups=len(y), failures=0, mape_pct=float(np.mean(error)),
                p90_pct=float(np.percentile(error,90)), rmspe_pct=float(np.sqrt(np.mean(error**2))))


def aggregate(rows):
    groups = defaultdict(list)
    fields = ['target','layer','dtype','method','k']
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    output = []
    for key, group in groups.items():
        item = dict(zip(fields,key), seeds=len(group), failures=sum(r['failures'] for r in group),
                    mean_train_labels=float(np.mean([r['train_labels'] for r in group])))
        for metric in ['mape_pct','p90_pct','rmspe_pct']:
            item['mean_'+metric] = None if item['failures'] else float(np.mean([r[metric] for r in group]))
        output.append(item)
    return output


def add_prediction(scores, predictions, info, rows, values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (len(rows),)
    for layer in sorted({r['case']['layer'] for r in rows}):
        idx = [i for i,r in enumerate(rows) if r['case']['layer']==layer]
        meta = dict(info, layer=layer)
        y = [rows[i]['actual_ms'] for i in idx]
        scores.append(dict(meta, **score(y,values[idx])))
        for i in idx:
            predictions.append(dict(meta, group_id=rows[i]['case']['group_id'], actual_ms=rows[i]['actual_ms'],
                predicted_ms=float(values[i]) if np.isfinite(values[i]) else None))


def chosen_training(dev, p, dtype, target, seed, k, family):
    allowed = {}
    for f in ['core','conv']:
        allowed[f] = set(p['subsets']['families'][f][str(seed)]['by_k'][str(k)])
    return [r for r in dev if r['dtype']==dtype and r['target']==target and r['case']['split']=='train'
            and r['case']['layer'] in p['common_layers']
            and (family=='all' or (family=='conv') == r['case']['layer'].startswith('conv'))
            and r['case']['group_id'] in allowed['conv' if r['case']['layer'].startswith('conv') else 'core']]


def static_predictions(rows, p, scores, predictions, diagnostics):
    for dtype,target in itertools.product(p['dtypes'],p['targets']):
        selected = [r for r in rows if r['dtype']==dtype and r['target']==target
                    and r['case']['layer'] in p['common_layers']]
        start=time.perf_counter()
        values=a.paleo([r['case'] for r in selected],dtype)
        meta=dict(dtype=dtype,target=target,method='paleo_empirical',seed=0,k=0,train_labels=0,model_ref='upstream/paleo')
        add_prediction(scores,predictions,meta,selected,values)
        diagnostics.append(dict(meta,prediction_seconds=time.perf_counter()-start,fit_seconds=0))
    neo=a.NeuSight()
    for target in p['targets']:
        for empirical in [False,True]:
            for layer in p['neusight_layers']:
                selected=[r for r in rows if r['dtype']=='fp32' and r['target']==target and r['case']['layer']==layer]
                cases=[r['case'] for r in selected]
                start=time.perf_counter()
                if layer=='linear':
                    values,details=neo.predict([a.gemm_shape(c) for c in cases],'LINEAR',empirical)
                else:
                    one,d1=neo.predict([(c['rows'],c['hidden_width'],c['input_width']) for c in cases],'LINEAR',empirical)
                    two,d2=neo.predict([(c['rows'],c['input_width'],c['hidden_width']) for c in cases],'LINEAR',empirical)
                    gelu,d3=neo.predict([(c['rows'],c['hidden_width']) for c in cases],'VEC',empirical)
                    values,details=one+two+gelu,dict(gemm1=d1,gemm2=d2,gelu=d3)
                meta=dict(dtype='fp32',target=target,method='neusight_'+('empirical' if empirical else 'clock'),
                          seed=0,k=0,train_labels=0,model_ref='upstream/neusight')
                add_prediction(scores,predictions,meta,selected,values)
                diagnostics.append(dict(meta,layer=layer,prediction_seconds=time.perf_counter()-start,
                    fit_seconds=0,published_source_training_cost='EXTERNAL_UNKNOWN',tile_matches=details))


def train(out):
    p=read(HERE/'protocol.json')
    assert sha(HERE/'development.json')==p['development_sha256']
    out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir()
    start=time.perf_counter()
    dev=read(HERE/'development.json')
    validation=[r for r in dev if r['case']['split']=='validation']
    scores,predictions,diagnostics,refs=[],[],[],[]
    for dtype,target,seed,k in itertools.product(p['dtypes'],p['targets'],p['seeds'],p['budgets']):
        for method,family in [('neuralpower','fc'),('neuralpower','conv'),('gemmify_dt','all'),('gemmify_capacity','all')]:
            training=chosen_training(dev,p,dtype,target,seed,k,family)
            cases=[r['case'] for r in training]
            y=np.array([r['actual_ms'] for r in training])
            if method=='neuralpower':
                model,detail=a.fit_neuralpower(cases,y,seed)
            else:
                model,detail=a.fit_gemmify(cases,y,seed,method=='gemmify_capacity')
            ref=f'models/{target}_{dtype}_{method}_{family}_{seed}_k{k}.joblib'
            joblib.dump(model,out/ref)
            meta=dict(dtype=dtype,target=target,method=method,seed=seed,k=k,train_labels=len(training),model_ref=ref)
            train_keys=[[c['group_id'],c['layer'],dtype,target] for c in cases]
            refs.append(dict(meta,family=family,train_keys=train_keys,sha256=sha(out/ref)))
            diagnostics.append(dict(meta,family=family,**detail))
            valid=[r for r in validation if r['dtype']==dtype and r['target']==target
                   and r['case']['layer'] in {c['layer'] for c in cases}]
            before=time.perf_counter()
            values=a.predict_fit(model,[r['case'] for r in valid])
            add_prediction(scores,predictions,meta,valid,values)
            diagnostics[-1]['prediction_seconds']=time.perf_counter()-before
            if method=='neuralpower':
                alt=dict(meta,method='neuralpower_original_floor')
                add_prediction(scores,predictions,alt,valid,a.predict_fit(model,[r['case'] for r in valid],True))
        save(out/'progress.json',dict(dtype=dtype,target=target,seed=seed,k=k,completed_fit_calls=len(refs)))
        print(f'PASS: fits {len(refs)} {dtype} {target} seed={seed} k={k}',flush=True)
    static_predictions(validation,p,scores,predictions,diagnostics)
    agg=aggregate(scores)
    choices=defaultdict(list)
    for r in agg:
        if not r['failures']:
            choices[r['target'],r['layer'],r['dtype'],r['method']].append(r)
    selected=[min(rs,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['mean_train_labels'],r['k'])) for rs in choices.values()]
    for name,rows in [('validation_metrics',scores),('validation_predictions',predictions),('validation_aggregate',agg),
                      ('selected',selected),('fit_diagnostics',diagnostics),('model_references',refs)]:
        save(out/(name+'.json'),rows)
        if name!='fit_diagnostics':csv_save(out/(name+'.csv'),rows)
    hashes={str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file()}
    code={name:sha(HERE/name) for name in ['adapters.py','compare.py','protocol.json','source_hashes.json','development.json']}
    save(out/'selection_frozen.json',dict(status='BASELINE_SELECTION_FROZEN',artifacts=hashes,code=code,
         wall_seconds=time.perf_counter()-start,fit_calls=len(refs),test_labels_read=False))
    print('PASS: development complete; baseline choices frozen before test scoring',flush=True)


def test_rows():
    p=read(REFERENCE/'prepared_layer_full/frozen_protocol.json')
    cases={(c['group_id'],c['layer']):c for cs in p['cases'].values() for c in cs if c['split']=='test_locked'}
    labels={}
    path=ROOT/'analysis/final_test_2026-11-11/layers/test_predictions.csv'
    with path.open() as stream:
        for r in csv.DictReader(stream):
            key=r['target'],r['layer'],r['dtype'],r['group_id']
            y=float(r['actual_ms'])
            if key in labels:assert labels[key]==y
            labels[key]=y
    return [dict(target=t,dtype=d,case=cases[g,l],actual_ms=y) for (t,l,d,g),y in sorted(labels.items())]


def evaluate(out):
    p=read(HERE/'protocol.json')
    frozen=read(out/'selection_frozen.json')
    for name,digest in frozen['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in frozen['code'].items():assert sha(HERE/name)==digest,name
    destination=out/'evaluation'
    destination.mkdir(exist_ok=False)
    selected=read(out/'selected.json')
    refs=read(out/'model_references.json')
    requests={tuple(r[k] for k in ['target','layer','dtype','method','k']) for r in selected}
    requests.update(tuple(r[k] for k in ['target','layer','dtype','method','k']) for r in read(out/'validation_aggregate.json') if r['k']==200)
    rows=test_rows()
    scores,predictions,diagnostics=[],[],[]
    for ref in refs:
        for method in [ref['method']]+(['neuralpower_original_floor'] if ref['method']=='neuralpower' else []):
            required=[r for r in rows if (r['target'],r['case']['layer'],r['dtype'],method,ref['k']) in requests
                      and r['target']==ref['target'] and r['dtype']==ref['dtype']
                      and (ref['family']=='all' or (ref['family']=='conv')==r['case']['layer'].startswith('conv'))
                      and r['case']['layer'] in p['common_layers']]
            if not required:continue
            model=joblib.load(out/ref['model_ref'])
            start=time.perf_counter()
            values=a.predict_fit(model,[r['case'] for r in required],method=='neuralpower_original_floor')
            info={k:ref[k] for k in ['target','dtype','seed','k','train_labels','model_ref']}
            info['method']=method
            add_prediction(scores,predictions,info,required,values)
            diagnostics.append(dict(info,prediction_seconds=time.perf_counter()-start))
    static_predictions(rows,p,scores,predictions,diagnostics)
    agg=aggregate(scores)
    selected_ids={tuple(r[k] for k in ['target','layer','dtype','method','k']) for r in selected}
    for r in agg:
        r['selected_on_validation']=tuple(r[k] for k in ['target','layer','dtype','method','k']) in selected_ids
    for name,items in [('test_metrics',scores),('test_predictions',predictions),('test_aggregate',agg)]:
        save(destination/(name+'.json'),items);csv_save(destination/(name+'.csv'),items)
    save(destination/'prediction_costs.json',diagnostics)
    save(destination/'evaluation_complete.json',dict(status='PASS',selection_sha256=sha(out/'selection_frozen.json'),
         test_rows=len(rows),model_selection_changed=False,original_test_sha256=sha(ROOT/'analysis/final_test_2026-11-11/layers/test_predictions.csv')))
    print('PASS: selected and k=200 baseline test predictions saved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','evaluate'])
    parser.add_argument('--run',default='v1')
    args=parser.parse_args()
    if not args.run.replace('_','').isalnum():raise ValueError('Invalid run name')
    output=HERE/'runs'/args.run
    (train if args.stage=='train' else evaluate)(output)
