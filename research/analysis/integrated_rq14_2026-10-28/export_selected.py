#!/usr/bin/env python3
"""Replay frozen selected fits on allowed training labels; no reselection or test reader."""
import csv
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

import run_rq14 as runner

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/v2'


def main(cost_aware=False):
    suffix='_cost_aware' if cost_aware else ''
    protocol_path=OUT/'frozen_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    if runner.sha(runner.__file__) != protocol['script_sha256']:
        raise ValueError('Frozen runner mismatch')
    for path,digest in protocol['sources'].items():
        if runner.sha(runner.ROOT/path) != digest:
            raise ValueError('Frozen input mismatch')
    selection='selected_configurations_cost_aware_validation_only.csv' if cost_aware else 'selected_configurations_validation_only.csv'
    with (OUT/selection).open(newline='') as handle:
        configs=list(csv.DictReader(handle))
    reference={}
    with (OUT/'validation_predictions.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            key=(row['rq'],row['dtype'],row['mode'],row['method'],int(row['n']),int(row['seed']))
            reference.setdefault(key,[]).append((tuple(map(int,row['shape'].split('x'))),float(row['predicted_ms'])))
    with (OUT/'source_selection.csv').open(newline='') as handle:
        sources={r['dtype']:r['method'] for r in csv.DictReader(handle)}
    destination=OUT/f'selected_models{suffix}'
    destination.mkdir(exist_ok=False)
    access=[];verification=[]
    total_start=time.perf_counter()
    for config in configs:
        rq,dtype,mode,method=config['rq'],config['dtype'],config['mode'],config['method']
        n=int(config['n'])
        for seed in protocol['seeds']:
            key=(rq,dtype,mode,method,n,seed)
            expected=reference[key]
            validation_shapes=[s for s,_ in expected]
            xv,lv=runner.design(validation_shapes,dtype)
            source_ref=None;source_prior=None;pv=None
            effective_n=0 if method=='zero' else n
            order_key='a_orders' if rq=='RQ1' else 'c_orders'
            training_shapes=[tuple(s) for s in protocol[order_key][str(seed)][:effective_n]]
            x,l=runner.design(training_shapes,dtype) if training_shapes else (None,None)
            if rq=='RQ4' and method!='direct':
                source_ref=f'models/source_{dtype}_{seed}_{sources[dtype]}.joblib'
                source=joblib.load(OUT/source_ref)
                pv,_=runner.predict(sources[dtype],source,xv,lv)
                if training_shapes:
                    source_prior,_=runner.predict(sources[dtype],source,x,l)
            if training_shapes:
                labels=runner.load_labels('legacy_train',training_shapes,[mode],access)
                y=runner.target(labels,training_shapes,dtype,mode)
                model,fit_seconds=runner.fit_model(method,x,y,l,seed,source_prior)
            else:
                model=None;fit_seconds=0
            pred,_=runner.predict(method,model,xv,lv,pv)
            expected_prediction=np.asarray([p for _,p in expected])
            if not np.array_equal(pred,expected_prediction):
                raise ValueError('Selected fit does not exactly reproduce saved validation predictions')
            envelope=dict(model=model,rq=rq,dtype=dtype,mode=mode,method=method,seed=seed,n=n,
                effective_train_shapes=effective_n,training_shapes=training_shapes,feature_order=protocol['features'],
                target_features_include_source_prediction=method in {'residual','prior_direct'},
                protocol_sha256=runner.sha(protocol_path),frozen_runner_sha256=protocol['script_sha256'],source_ref=source_ref,
                source_sha256=runner.sha(OUT/source_ref) if source_ref else None,validation_only=True,locked_test_opened=False)
            filename=f'{rq}_{dtype}_{mode}_{method}_{n}_{seed}.joblib'
            joblib.dump(envelope,destination/filename)
            restored=joblib.load(destination/filename)
            restored_pred,_=runner.predict(method,restored['model'],xv,lv,pv)
            if not np.array_equal(restored_pred,expected_prediction):
                raise ValueError('Serialized selected model changed predictions')
            verification.append(dict(rq=rq,dtype=dtype,mode=mode,method=method,n=n,seed=seed,effective_train_shapes=effective_n,
                validation_predictions=len(expected),exact_prediction_match=True,serialized_exact_match=True,export_refit_seconds=fit_seconds,
                model_path=str((destination/filename).relative_to(OUT)),model_sha256=runner.sha(destination/filename)))
    runner.save_csv(OUT/f'selected_model_verification{suffix}.csv',verification)
    runner.save_json(OUT/f'selected_model_export{suffix}_label_access.json',access)
    runner.save_json(OUT/f'selected_model_export{suffix}.json',dict(timestamp=runner.stamp(),models=len(verification),selection_file=selection,
        wall_seconds=time.perf_counter()-total_start,export_refit_seconds=sum(r['export_refit_seconds'] for r in verification),
        protocol_sha256=runner.sha(protocol_path),export_script_sha256=runner.sha(__file__),new_validation_labels_loaded=0,new_test_labels_loaded=0,
        locked_test_status='NOT_OPENED',all_predictions_exact_match=all(r['exact_prediction_match'] for r in verification)))
    print(f'PASS: {len(verification)} selected model envelopes saved; exact replay and serialization checks passed; test NOT OPENED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cost-aware',action='store_true')
    main(parser.parse_args().cost_aware)
