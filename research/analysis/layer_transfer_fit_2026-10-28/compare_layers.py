#!/usr/bin/env python3
"""Matched RQ3 learners; use only the frozen standard development label loader."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

import fit_layers as layer

COMPARISON = layer.ROOT / 'analysis/learner_comparison_2026-10-28'
spec = importlib.util.spec_from_file_location('matched_gemm', COMPARISON/'compare_learners.py')
learners = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learners)
gemm = layer.gemm


def train(learner, method, x, y, prior, seed):
    if method in {'zero', 'constant_ratio'}:
        model, seconds = gemm.fit_model(method, x, y, np.ones(len(y)), seed, prior)
        return model, dict(fit_seconds=seconds, epochs='', convergence_warnings=0,
                           final_scaled_loss='', warning_messages='', parameter_count='')
    response = np.log(y) - (np.log(prior) if method == 'residual' else 0)
    xx = np.column_stack((x, np.log(prior))) if method in {'residual', 'prior_direct'} else x
    return learners.fit(learner, xx, response, seed)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    original = layer.HERE/'frozen_fit_protocol.json'
    frozen = layer.read(original)
    source_run = COMPARISON/'runs/v1'
    source_selection = layer.read(source_run/'source_selection_before_target_access.json')
    checked_hashes = layer.read(source_run/'artifact_hashes_final.json')
    weights, mapping = {}, {}
    for selected in source_selection['selection']:
        learner, dtype, target = (selected[k] for k in ['learner', 'dtype', 'target'])
        mapping[f'{learner}/{dtype}'] = target
        for seed in layer.SEEDS:
            name = f'models/rq1_{learner}_{dtype}_allocate_{target}_2652_{seed}.joblib'
            path = source_run/name
            if gemm.sha(path) != checked_hashes[name]:
                raise ValueError('Trusted source model checksum mismatch')
            weights[str(path.relative_to(layer.ROOT))] = checked_hashes[name]
    paths = [Path(__file__), layer.HERE/'fit_layers.py', COMPARISON/'compare_learners.py',
             original, source_run/'source_selection_before_target_access.json',
             source_run/'frozen_protocol.json', source_run/'artifact_hashes_final.json']
    protocol = dict(version='matched_layer_learners_v1', created_utc=gemm.stamp(),
        frozen_before_first_layer_label_access=True,
        timing='Returned archive was integrity-checked; no layer timing labels have been parsed.',
        inputs={str(p.relative_to(layer.ROOT)):gemm.sha(p) for p in paths},
        original_protocol_sha256=gemm.sha(original), source_weights=weights, source_targets=mapping,
        learners=learners.LEARNERS, seeds=layer.SEEDS, budgets=layer.BUDGETS,
        methods=layer.METHODS, features=layer.FEATURES, rf=gemm.RF,
        ridge=dict(alpha=1), mlp=learners.MLP, widths=[32,64],
        data_contract='Original frozen 30 train / 10 validation groups; same nested k and 3 seeds; median of 3 standard rounds; no group exclusion.',
        source_contract='Family-matched A=2652 GEMM allocate models, frozen source selection; sum logical GEMMs; no measured component times; no source fine-tuning.',
        preprocessing='Reuse matched GEMM fit: train-only X scaling for Ridge; train-only X and log-response scaling for MLP; training-loss stopping, no outer-validation early stopping.',
        accuracy_selection='Mean validation MAPE over 3 seeds, then p90, effective_k, learner, method; non-finite predictions disqualify.',
        cost_selection='Retain <=15% MAPE and <=30% p90 minimum effective_k selection separately; zero uses k=0.',
        label_policy='Only fit_layers.load_standard may access labels. Never traverse restricted_explicit or locked result records.',
        disclosure='Extension frozen after GPU collection, before label access. Validation model selection is development evidence, not an independent test.',
        runtime=dict(python=sys.version,numpy=np.__version__,sklearn=gemm.sklearn.__version__))
    gemm.save_json(output/'frozen_protocol.json', protocol)
    print('PASS: matched RQ3 protocol frozen before first layer label access', flush=True)


def choose(candidates, accuracy):
    valid = [r for r in candidates if r['failures']==0 and np.isfinite(r['mean_mape_pct'])]
    if not valid:
        raise ValueError('No finite model candidate')
    passing = [r for r in valid if r['mean_mape_pct']<=15 and r['mean_p90_pct']<=30]
    metric_key = lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['effective_k'],r['learner'],r['method'],r['k'])
    selected = min(valid,key=metric_key) if accuracy or not passing else min(passing,key=lambda r:(r['effective_k'],)+metric_key(r))
    return {**selected,'selection_rule':'accuracy_first' if accuracy else 'historical_minimum_cost',
            'historical_goal_attained':selected in passing}


def run(directory, output):
    protocol_path=output/'frozen_protocol.json';p=layer.read(protocol_path)
    if (output/'execution_started.json').exists():
        raise ValueError('Refusing to overwrite an execution')
    for name,digest in {**p['inputs'],**p['source_weights']}.items():
        if gemm.sha(layer.ROOT/name)!=digest:
            raise ValueError('Frozen dependency changed: '+name)
    original=layer.read(layer.HERE/'frozen_fit_protocol.json')
    labels,input_hashes=layer.load_standard(directory,original)
    gemm.save_json(output/'execution_started.json',dict(time=gemm.stamp(),protocol_sha256=gemm.sha(protocol_path)))
    gemm.save_json(output/'standard_input_hashes.json',input_hashes)
    start=time.perf_counter();scores=[];predictions=[];diagnostics=[];models={};paired=[]
    groups=original['groups'];valid_ids=original['validation_ids']
    for learner in p['learners']:
        for kind in ['linear','ffn']:
            for dtype in ['fp16','fp32']:
                vg=[groups[i] for i in valid_ids];xv=layer.features(vg,kind,dtype)
                yv=np.asarray([labels[(i,kind,dtype)] for i in valid_ids])
                target=p['source_targets'][f'{learner}/{dtype}']
                for seed in layer.SEEDS:
                    ref=f'analysis/learner_comparison_2026-10-28/runs/v1/models/rq1_{learner}_{dtype}_allocate_{target}_2652_{seed}.joblib'
                    envelope=joblib.load(layer.ROOT/ref)
                    if (envelope['learner'],envelope['dtype'],envelope['target'],envelope['n'],envelope['seed'])!=(learner,dtype,target,2652,seed):
                        raise ValueError('Source envelope identity mismatch')
                    prior_start=time.perf_counter();pv=layer.prior(vg,kind,dtype,envelope['model'],learner+'_'+target)
                    source_validation_seconds=time.perf_counter()-prior_start
                    for k in layer.BUDGETS:
                        ids=original['subsets']['seeds'][str(seed)]['by_k'][str(k)]
                        gg=[groups[i] for i in ids];x=layer.features(gg,kind,dtype)
                        y=np.asarray([labels[(i,kind,dtype)] for i in ids])
                        prior_start=time.perf_counter();pa=layer.prior(gg,kind,dtype,envelope['model'],learner+'_'+target)
                        source_train_seconds=time.perf_counter()-prior_start
                        method_predictions={}
                        for method in layer.METHODS:
                            model,detail=train(learner,method,x,y,pa,seed)
                            pred,seconds=gemm.predict(method,model,xv,np.ones(len(yv)),pv)
                            train_pred,_=gemm.predict(method,model,x,np.ones(k),pa)
                            info=dict(learner=learner,layer=kind,dtype=dtype,method=method,k=k,seed=seed)
                            scores.append({**info,**gemm.metrics(yv,pred),'train_mape_pct':gemm.metrics(y,train_pred)['mape_pct']})
                            diagnostics.append({**info,**detail,'correction_predict_seconds':seconds,
                                'source_prior_train_seconds':source_train_seconds if method not in {'direct','zero'} else 0,
                                'source_prior_validation_seconds':source_validation_seconds if method!='direct' else 0})
                            models[(learner,kind,dtype,method,k,seed)]=(model,ref,target)
                            method_predictions[method]=pred
                            predictions.extend({**info,'group_id':i,'actual_ms':float(a),'predicted_ms':float(b)} for i,a,b in zip(valid_ids,yv,pred))
                        for a,b in [('residual','direct'),('prior_direct','direct'),('residual','prior_direct'),('constant_ratio','zero')]:
                            paired.append(dict(learner=learner,layer=kind,dtype=dtype,k=k,seed=seed,first=a,second=b,
                                               **gemm.bootstrap_delta(yv,method_predictions[a],method_predictions[b])))
                print(f'PASS: RQ3 {learner} {kind} {dtype}',flush=True)
    aggregate=[]
    for learner in p['learners']:
        for kind in ['linear','ffn']:
            for dtype in ['fp16','fp32']:
                for method in layer.METHODS:
                    for k in layer.BUDGETS:
                        rr=[r for r in scores if (r['learner'],r['layer'],r['dtype'],r['method'],r['k'])==(learner,kind,dtype,method,k)]
                        aggregate.append(dict(learner=learner,layer=kind,dtype=dtype,method=method,k=k,effective_k=0 if method=='zero' else k,
                            mean_mape_pct=float(np.mean([r['mape_pct'] for r in rr])),
                            mean_p90_pct=float(np.mean([r['p90_pct'] for r in rr])),
                            mean_train_mape_pct=float(np.mean([r['train_mape_pct'] for r in rr])),failures=sum(r['failures'] for r in rr)))
    selected=[]
    for kind in ['linear','ffn']:
        for dtype in ['fp16','fp32']:
            for family in p['learners']+['all']:
                cc=[r for r in aggregate if r['layer']==kind and r['dtype']==dtype and (family=='all' or r['learner']==family)]
                for accuracy in [True,False]:
                    selected.append({**choose(cc,accuracy),'selection_scope':family,'validation_groups':10})
    (output/'models').mkdir();saved=set();weight_rows=[]
    for chosen in selected:
        for seed in layer.SEEDS:
            identity=tuple(chosen[n] for n in ['learner','layer','dtype','method','k'])+(seed,)
            if identity in saved:
                continue
            saved.add(identity);model,ref,target=models[identity]
            filename='_'.join(map(str,identity))+'.joblib'
            joblib.dump(dict(model=model,**chosen,seed=seed,source_ref=ref,source_target=target,
                source_sha256=p['source_weights'][ref],features=layer.FEATURES,protocol_sha256=gemm.sha(protocol_path),
                training_group_ids=original['subsets']['seeds'][str(seed)]['by_k'][str(chosen['k'])] if chosen['effective_k'] else [],
                validation_only=True,locked_test_opened=False),output/'models'/filename)
            loaded=joblib.load(output/'models'/filename)
            vg=[groups[i] for i in valid_ids];xx=layer.features(vg,chosen['layer'],chosen['dtype'])
            source=joblib.load(layer.ROOT/ref)['model'];pv=layer.prior(vg,chosen['layer'],chosen['dtype'],source,chosen['learner']+'_'+target)
            pred,_=gemm.predict(chosen['method'],loaded['model'],xx,np.ones(10),pv)
            expected=[r['predicted_ms'] for r in predictions if tuple(r[n] for n in ['learner','layer','dtype','method','k','seed'])==identity]
            if not np.array_equal(pred,np.asarray(expected)):
                raise ValueError('Selected model reload prediction mismatch')
            weight_rows.append(dict(file=filename,sha256=gemm.sha(output/'models'/filename),predictions_exact=True))
    for name,rows in [('metrics.csv',scores),('aggregate_metrics.csv',aggregate),('validation_predictions.csv',predictions),
                      ('fit_diagnostics.csv',diagnostics),('paired_bootstrap.csv',paired),('selected_configurations.csv',selected),
                      ('model_weight_verification.csv',weight_rows)]:
        gemm.save_csv(output/name,rows)
    gemm.save_json(output/'selection_frozen.json',dict(timestamp=gemm.stamp(),status='RQ3_SELECTION_FROZEN_VALIDATION_ONLY',
        protocol_sha256=gemm.sha(protocol_path),selected_configurations_sha256=gemm.sha(output/'selected_configurations.csv'),
        rq2_explicit_diagnostics_may_be_separately_reviewed=True,locked_test_status='NOT_OPENED',explicit_values_read=0,
        configurations=len(scores),actual_regressor_fits=sum(r['method'] not in {'zero','constant_ratio'} for r in scores),
        convergence_warnings=sum(r['convergence_warnings'] for r in diagnostics),saved_weights=len(weight_rows),
        wall_seconds=time.perf_counter()-start,total_model_fit_seconds=sum(r['fit_seconds'] for r in diagnostics),
        input_hashes_unchanged=all(gemm.sha(Path(directory)/n)==h for n,h in input_hashes.items())))
    print('PASS: four-family RQ3 selection frozen; explicit and locked timing labels not accessed',flush=True)


def selfcheck():
    x=np.arange(18,dtype=float).reshape(6,3)/20;prior=np.ones(6)*.1;y=prior*3
    for family in learners.LEARNERS:
        for method in layer.METHODS:
            model,_=train(family,method,x,y,prior,layer.SEEDS[0])
            pred,_=gemm.predict(method,model,x,np.ones(6),prior)
            assert np.isfinite(pred).all() and (pred>0).all()
            if method in {'constant_ratio','residual'}:
                assert np.allclose(pred,y,rtol=.02)
    candidates=[dict(learner='rf',method='zero',k=5,effective_k=0,mean_mape_pct=12,mean_p90_pct=20,failures=0),
                dict(learner='ridge',method='direct',k=30,effective_k=30,mean_mape_pct=3,mean_p90_pct=8,failures=0)]
    assert choose(candidates,True)['method']=='direct' and choose(candidates,False)['method']=='zero'
    print('PASS: synthetic four-family log-response, prior and selection contracts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','selfcheck'])
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    elif args.action=='prepare':prepare(args.output)
    else:run(args.input.resolve(),args.output)
