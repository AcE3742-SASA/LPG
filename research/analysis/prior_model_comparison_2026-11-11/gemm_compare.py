"""Full-budget GEMM comparison using the existing 11,498/1,611/1,000 split."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import itertools
import time

import joblib
import numpy as np

import adapters as a
from prepare import HERE, ROOT, save, sha
from compare import read, csv_save, score

EXPANSION=ROOT/'analysis/gemm_expansion_result_2026-11-06'
OUT=HERE/'gemm/v1'


def cases(shapes, split):
    return [dict(id='x'.join(map(str,s)),group_id='x'.join(map(str,s)),layer='linear',
                 split=split,rows=s[0],output_width=s[1],input_width=s[2]) for s in shapes]


def record(metrics,predictions,info,shapes,actual,values):
    metrics.append(dict(info,**score(actual,values)))
    for s,y,v in zip(shapes,actual,values):
        predictions.append(dict(info,shape='x'.join(map(str,s)),actual_ms=float(y),
                                predicted_ms=float(v) if np.isfinite(v) else None))


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['dtype'],r['mode'],r['method']].append(r)
    result=[]
    for (dtype,mode,method),items in grouped.items():
        failed=sum(x['failures'] for x in items)
        row=dict(dtype=dtype,mode=mode,method=method,seeds=len(items),failures=failed,
                 train_labels=items[0]['train_labels'],groups=items[0]['groups'])
        for m in ['mape_pct','p90_pct','rmspe_pct']:
            row['mean_'+m]=None if failed else float(np.mean([x[m] for x in items]))
        result.append(row)
    return result


def static(shapes,labels,metrics,predictions,diagnostics):
    cs=cases(shapes,'validation')
    for dtype in ['fp16','fp32']:
        values=a.paleo(cs,dtype)
        for mode in ['allocate','prealloc','graph']:
            info=dict(dtype=dtype,mode=mode,method='paleo_empirical',seed=0,train_labels=0)
            record(metrics,predictions,info,shapes,[labels[s,dtype,mode] for s in shapes],values)
    neo=a.NeuSight()
    for kind,empirical in itertools.product(['BMM','LINEAR'],[False,True]):
        start=time.perf_counter()
        values,detail=neo.predict(shapes,kind,empirical)
        method='neusight_'+kind.lower()+'_'+('empirical' if empirical else 'clock')
        diagnostics.append(dict(method=method,prediction_seconds=time.perf_counter()-start,
                                published_source_training_cost='EXTERNAL_UNKNOWN',**detail))
        for mode in ['allocate','prealloc','graph']:
            info=dict(dtype='fp32',mode=mode,method=method,seed=0,train_labels=0)
            record(metrics,predictions,info,shapes,[labels[s,'fp32',mode] for s in shapes],values)


def train():
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'models').mkdir()
    spec=importlib.util.spec_from_file_location('original_expansion',EXPANSION/'compare_expansion.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    p=read(EXPANSION/'fit_v1/frozen_protocol.json')
    protocol=dict(status='GEMM_BASELINES_FIXED_BEFORE_FIT',source_protocol_sha256=sha(EXPANSION/'fit_v1/frozen_protocol.json'),
        training_shapes=11498,validation_shapes=1611,test_shapes=1000,seeds=p['seeds'],
        methods=['neuralpower_fc_port','neuralpower_original_floor','paleo_empirical',
                 'neusight_bmm_clock','neusight_bmm_empirical','neusight_linear_clock','neusight_linear_empirical'],
        neuralpower_scope='Public FC runtime feature map retrained for each unbatched GEMM dtype/mode; full training budget fixed in advance',
        selection='Training-only grouped CV selects lasso penalty; no choice based on test or validation scores',
        profiles=read(HERE/'protocol.json')['neusight'],
        adaptations='BMM B=1 and bias-trained LINEAR are separately labelled proxies for torch.mm; Graph and preallocation change runtime boundaries.',
        input_hashes={str(path.relative_to(ROOT)):sha(path) for path in [EXPANSION/'compare_expansion.py',
            EXPANSION/'fit_v1/frozen_protocol.json',EXPANSION/'fit_v1/adopted_conditions.csv']},
        original_test_previously_disclosed=True)
    save(OUT/'protocol.json',protocol)
    train_labels,validation_sets,access=original.source_labels(p)
    validation={**validation_sets['legacy'],**validation_sets['expansion']}
    shapes=list(map(tuple,p['legacy_validation_shapes']+p['new_validation_shapes']))
    assert len(shapes)==len(set(shapes))==1611
    metrics,predictions,diagnostics,models=[],[],[],[]
    start=time.perf_counter()
    for dtype,mode,seed in itertools.product(['fp16','fp32'],['allocate','prealloc','graph'],p['seeds']):
        training=list(map(tuple,p['old_train_orders'][str(seed)]+p['new_train_orders'][str(seed)]))
        assert len(training)==len(set(training))==11498 and not set(training)&set(shapes)
        y=np.array([train_labels[s,dtype,mode] for s in training])
        model,detail=a.fit_neuralpower(cases(training,'train'),y,seed)
        ref=f'models/{dtype}_{mode}_{seed}.joblib'
        joblib.dump(model,OUT/ref)
        models.append(dict(dtype=dtype,mode=mode,seed=seed,model_ref=ref,sha256=sha(OUT/ref)))
        diagnostics.append(dict(dtype=dtype,mode=mode,seed=seed,**detail))
        for floor in [False,True]:
            info=dict(dtype=dtype,mode=mode,seed=seed,method='neuralpower_original_floor' if floor else 'neuralpower_fc_port',train_labels=11498)
            record(metrics,predictions,info,shapes,[validation[s,dtype,mode] for s in shapes],
                   a.predict_fit(model,cases(shapes,'validation'),floor))
        print(f'PASS: GEMM fit {dtype} {mode} seed={seed}',flush=True)
    static(shapes,validation,metrics,predictions,diagnostics)
    for name,value in [('validation_metrics',metrics),('validation_predictions',predictions),('validation_aggregate',aggregate(metrics)),('models',models),('diagnostics',diagnostics)]:
        save(OUT/(name+'.json'),value)
        if name!='diagnostics':csv_save(OUT/(name+'.csv'),value)
    save(OUT/'label_access.json',access)
    hashes={str(path.relative_to(OUT)):sha(path) for path in OUT.rglob('*') if path.is_file()}
    save(OUT/'selection_frozen.json',dict(status='GEMM_BASELINES_FROZEN',artifacts=hashes,
        code={name:sha(HERE/name) for name in ['gemm_compare.py','adapters.py']},wall_seconds=time.perf_counter()-start,
        test_labels_read=False,fit_calls=len(models)))
    print('PASS: GEMM baselines frozen before test scoring',flush=True)


def evaluate():
    frozen=read(OUT/'selection_frozen.json')
    for name,digest in frozen['artifacts'].items():assert sha(OUT/name)==digest,name
    for name,digest in frozen['code'].items():assert sha(HERE/name)==digest,name
    destination=OUT/'evaluation';destination.mkdir(exist_ok=False)
    path=ROOT/'analysis/final_test_2026-11-11/gemm_expanded/predictions.csv'
    labels={}
    with path.open() as stream:
        for r in csv.DictReader(stream):
            key=tuple(map(int,r['shape'].split('x'))),r['dtype'],r['mode']
            value=float(r['actual_ms'])
            if key in labels:assert labels[key]==value
            labels[key]=value
    shapes=sorted({k[0] for k in labels})
    assert len(shapes)==1000 and len(labels)==6000
    metrics,predictions,diagnostics=[],[],[]
    for r in read(OUT/'models.json'):
        model=joblib.load(OUT/r['model_ref'])
        for floor in [False,True]:
            info=dict(dtype=r['dtype'],mode=r['mode'],seed=r['seed'],train_labels=11498,
                      method='neuralpower_original_floor' if floor else 'neuralpower_fc_port')
            record(metrics,predictions,info,shapes,[labels[s,r['dtype'],r['mode']] for s in shapes],
                   a.predict_fit(model,cases(shapes,'test_locked'),floor))
    static(shapes,labels,metrics,predictions,diagnostics)
    for name,value in [('test_metrics',metrics),('test_predictions',predictions),('test_aggregate',aggregate(metrics))]:
        save(destination/(name+'.json'),value);csv_save(destination/(name+'.csv'),value)
    save(destination/'diagnostics.json',diagnostics)
    save(destination/'evaluation_complete.json',dict(status='PASS',selection_sha256=sha(OUT/'selection_frozen.json'),
        test_rows=len(labels),original_test_sha256=sha(path),model_selection_changed=False))
    print('PASS: GEMM test comparison complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate'])
    args=parser.parse_args();(train if args.stage=='train' else evaluate)()
