#!/usr/bin/env python3
"""CPU-only RQ3 development learner. No locked or explicit result reader."""
import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RQ14=ROOT/'analysis/integrated_rq14_2026-10-28'
PACKAGE=ROOT/'layer_transfer_20261028_v1'
SOURCE=RQ14/'runs/v2'
spec=importlib.util.spec_from_file_location('frozen_rq14',RQ14/'run_rq14.py')
gemm=importlib.util.module_from_spec(spec);spec.loader.exec_module(gemm)
SEEDS=[20261028,20261029,20261030]
BUDGETS=[5,10,20,30]
METHODS=['zero','constant_ratio','direct','prior_direct','residual']
FEATURES=gemm.FEATURES+['log_total_gemm_F','log1p_gelu_elements']


def read(path):
    return json.loads(Path(path).read_text())


def source_mapping():
    with (SOURCE/'source_selection.csv').open(newline='') as handle:
        return {r['dtype']:r['method'] for r in csv.DictReader(handle)}


def prepare(path):
    if path.exists():
        raise ValueError('Refusing to overwrite frozen fit protocol')
    groups=read(PACKAGE/'groups.json')
    dev={g['group_id']:g for g in groups if g['split'] in {'train','validation'}}
    if sum(g['split']=='train' for g in dev.values())!=30 or len(dev)!=40:
        raise ValueError('Unexpected group split')
    subsets=read(PACKAGE/'rq3_training_subsets.json')
    if not subsets['frozen_before_gpu_collection']:
        raise ValueError('Training subsets not frozen')
    for seed in SEEDS:
        previous=set()
        for k in BUDGETS:
            ids=subsets['seeds'][str(seed)]['by_k'][str(k)]
            if len(ids)!=k or len(set(ids))!=k or not previous<=set(ids) or any(dev[i]['split']!='train' for i in ids):
                raise ValueError('Invalid nested training subset')
            previous=set(ids)
    mapping=source_mapping()
    sources={str((SOURCE/'models'/f'source_{d}_{s}_{mapping[d]}.joblib').relative_to(ROOT)):
             gemm.sha(SOURCE/'models'/f'source_{d}_{s}_{mapping[d]}.joblib') for d in ['fp16','fp32'] for s in SEEDS}
    paths=[RQ14/'run_rq14.py',RQ14/'requirements.txt',SOURCE/'frozen_protocol.json',SOURCE/'source_selection.csv']
    paths += [PACKAGE/name for name in ['groups.json','cases_by_phase.json','rq3_training_subsets.json','protocol.json','release_hashes.json','compare.py','selfcheck.py']]
    protocol=dict(version='layer_transfer_fit_v1',created_utc=gemm.stamp(),status='PREPARED_NOT_EXECUTED',
        script_sha256=gemm.sha(__file__),seeds=SEEDS,budgets=BUDGETS,methods=METHODS,
        method_names=dict(zero='S_A',constant_ratio='constant_ratio',direct='D0',prior_direct='D+A',residual='R'),
        features=FEATURES,rf=gemm.RF,source_methods=mapping,source_weights=sources,
        input_hashes={str(p.relative_to(ROOT)):gemm.sha(p) for p in paths},
        package_release_hashes=read(PACKAGE/'release_hashes.json'),groups=dev,
        cases=read(PACKAGE/'cases_by_phase.json')['core_dev'],subsets=subsets,
        validation_ids=sorted(i for i,g in dev.items() if g['split']=='validation'),
        label_policy='core_dev standard only; 40 groups x2layers x2dtypes x3rounds=480 records; reject other phases before record access; never traverse restricted_explicit',
        quality_policy='Require all 480 standard records, correctness PASS, no standard measurement flags, all3round positive finite, between-round range<=5%; otherwise stop without dropping groups',
        source_policy='Sum source GEMM predictions of logical components; no H/C or measured component timings',
        selection='Per layer/dtype: mean across3seeds MAPE<=15 and p90<=30, nofailures; minimum effective_k (zero=0), then MAPE then method; otherwise minimum MAPE with goal=False',
        cost='12*(effective_k+10) standard records across2layers2dtypes3rounds; source preparation shared; explicit collection remains in whole-study costs',
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,sklearn=gemm.sklearn.__version__,rf_n_jobs=1,
                     thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}),
        locked_test_status='NOT_OPENED',explicit_diagnostics_status='CLOSED_UNTIL_SELECTION_FROZEN')
    path.parent.mkdir(parents=True,exist_ok=True);gemm.save_json(path,protocol)
    print('PASS: RQ3 fit protocol frozen; no layer labels or GPU accessed')


def load_standard(directory,protocol):
    directory=Path(directory)
    # This gate must run before any listing, hashing or reading of measurement files.
    manifest=read(directory/'manifest.json')
    if manifest.get('phase')!='core_dev':
        raise ValueError('Only core_dev standard labels are allowed')
    if manifest.get('kind')!='layer_transfer_20261028_v1' or manifest.get('split')!='train_validation_only':
        raise ValueError('Unexpected measurement manifest')
    if manifest.get('source_hashes')!=protocol['package_release_hashes'] or manifest.get('cases')!=protocol['cases']:
        raise ValueError('Measurement manifest differs from frozen package')
    config=manifest.get('protocol',{})
    if config.get('rounds')!=3 or config.get('dtypes')!=['fp16','fp32'] or config.get('phase')!='core_dev':
        raise ValueError('Measurement protocol mismatch')
    if any(config.get(k)!=v for k,v in [('sm_mhz',720),('memory_mhz',6801),('power_limit_w',70)]):
        raise ValueError('Measurement hardware policy mismatch')
    if config.get('timing',{}).get('between_round_limit')!=.05:
        raise ValueError('Measurement quality policy mismatch')
    summary=read(directory/'summary.json');status=read(directory/'status.json');finished=read(directory/'worker/complete.json')
    if summary.get('scope')!='core_dev' or summary.get('gate') not in {'PASS','REVIEW'} or summary.get('collection')!='COMPLETE' or summary.get('comparisons'):
        raise ValueError('Public quality summary is incomplete or exposes explicit values')
    if status.get('collection')!='COMPLETE' or status.get('restoration')!='COMMANDS_ACCEPTED' or finished.get('complete') is not True:
        raise ValueError('Collection or restoration incomplete')
    cases={c['id']:c for c in protocol['cases']}
    expected={f'{case_id}_{dtype}_r{rd}_standard' for case_id in cases for dtype in ['fp16','fp32'] for rd in range(3)}
    hashes={name:gemm.sha(directory/name) for name in ['manifest.json','summary.json','status.json','worker/complete.json']}
    records={}
    if (directory/'records').is_symlink() or (directory/'correctness').is_symlink():
        raise ValueError('Symlink measurement directory denied')
    paths=sorted((directory/'records').glob('*.json'))
    if len(paths)!=480:
        raise ValueError('Expected exactly 480 standard records')
    for path in paths:
        if path.is_symlink() or path.stem not in expected:
            raise ValueError('Unexpected standard record path')
        row=read(path);rid=row.get('id')
        if rid!=path.stem or rid in records or row.get('implementation')!='standard':
            raise ValueError('Invalid standard record identity')
        case=row.get('case',{});case_id=case.get('id')
        if case_id not in cases or case!=cases[case_id] or case.get('split') not in {'train','validation'}:
            raise ValueError('Unexpected or locked case')
        dtype=row.get('dtype');rd=row.get('round')
        if rid!=f'{case_id}_{dtype}_r{rd}_standard' or dtype not in ['fp16','fp32'] or type(rd) is not int or rd not in range(3):
            raise ValueError('Invalid dtype or round')
        value=row.get('timing',{}).get('median_ms')
        flags=summary.get('measurement_flags',{})
        if rid not in flags or flags[rid] or row['timing'].get('flags') or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Standard measurement quality is not PASS')
        for field in ['correctness_before','correctness_after']:
            requested=directory/row[field]
            check=requested.resolve()
            if requested.is_symlink() or not check.is_relative_to((directory/'correctness').resolve()) or not read(check).get('passed'):
                raise ValueError('Correctness failed or unsafe path')
            hashes[str(check.relative_to(directory.resolve()))]=gemm.sha(check)
        hashes[str(path.relative_to(directory))]=gemm.sha(path)
        records[rid]=value
    if set(records)!=expected:
        raise ValueError('Standard records are incomplete')
    labels={}
    for case_id,case in cases.items():
        for dtype in ['fp16','fp32']:
            values=[records[f'{case_id}_{dtype}_r{r}_standard'] for r in range(3)]
            median=statistics.median(values)
            if (max(values)-min(values))/median>.05:
                raise ValueError('Standard between-round quality is REVIEW')
            key=(case['group_id'],case['layer'],dtype)
            if key in labels:
                raise ValueError('Duplicate shape-group label')
            labels[key]=median
    if len(labels)!=160:
        raise ValueError('Unexpected condition count')
    return labels,hashes


def features(groups,layer,dtype):
    first=[(g['M'],g['F'],g['D']) for g in groups]
    x,_=gemm.design(first,dtype)
    total=np.asarray([2*g['M']*g['D']*g['F']*(2 if layer=='ffn' else 1) for g in groups],dtype=float)
    gelu=np.asarray([g['M']*g['F'] if layer=='ffn' else 0 for g in groups],dtype=float)
    return np.column_stack((x,np.log(total),np.log1p(gelu)))


def prior(groups,layer,dtype,model,method):
    shapes=[(g['M'],g['F'],g['D']) for g in groups]
    x,l=gemm.design(shapes,dtype);prediction,_=gemm.predict(method,model,x,l)
    if layer=='ffn':
        second=[(g['M'],g['D'],g['F']) for g in groups]
        x,l=gemm.design(second,dtype);p2,_=gemm.predict(method,model,x,l);prediction+=p2
    if not (np.isfinite(prediction)&(prediction>0)).all():
        raise ValueError('Source prior prediction failed')
    return prediction


def run(input_directory,output,protocol_path):
    p=read(protocol_path)
    if p['script_sha256']!=gemm.sha(__file__):
        raise ValueError('Frozen learner hash mismatch')
    for path,digest in {**p['input_hashes'],**p['source_weights']}.items():
        if gemm.sha(ROOT/path)!=digest:
            raise ValueError('Frozen metadata or trusted source weight hash mismatch')
    if output.exists():
        raise ValueError('Refusing to overwrite a fit output')
    labels,hashes=load_standard(input_directory,p)
    output.mkdir(parents=True);(output/'models').mkdir()
    gemm.save_json(output/'returned_standard_input_hashes.json',hashes)
    groups=p['groups'];valid_ids=p['validation_ids'];models={};scores=[];predictions=[];times=[];paired=[]
    start=time.perf_counter()
    for layer in ['linear','ffn']:
        for dtype in ['fp16','fp32']:
            vg=[groups[i] for i in valid_ids];xv=features(vg,layer,dtype)
            yv=np.asarray([labels[(i,layer,dtype)] for i in valid_ids])
            for seed in SEEDS:
                ref=f'analysis/integrated_rq14_2026-10-28/runs/v2/models/source_{dtype}_{seed}_{p["source_methods"][dtype]}.joblib'
                source=joblib.load(ROOT/ref)
                prior_started=time.perf_counter()
                pv=prior(vg,layer,dtype,source,p['source_methods'][dtype])
                prior_validation_seconds=time.perf_counter()-prior_started
                for k in BUDGETS:
                    ids=p['subsets']['seeds'][str(seed)]['by_k'][str(k)];gg=[groups[i] for i in ids]
                    x=features(gg,layer,dtype);y=np.asarray([labels[(i,layer,dtype)] for i in ids])
                    prior_started=time.perf_counter();pa=prior(gg,layer,dtype,source,p['source_methods'][dtype])
                    prior_train_seconds=time.perf_counter()-prior_started
                    method_predictions={}
                    for method in METHODS:
                        model,seconds=gemm.fit_model(method,x,y,np.ones(k),seed,pa)
                        prediction,ps=gemm.predict(method,model,xv,np.ones(10),pv)
                        key=(layer,dtype,method,k,seed);models[key]=(model,ref)
                        info=dict(layer=layer,dtype=dtype,method=method,k=k,seed=seed)
                        scores.append({**info,**gemm.metrics(yv,prediction)})
                        uses_prior=method!='direct'
                        times.append({**info,'model_fit_seconds':seconds,'correction_predict_seconds':ps,
                            'source_prior_train_predict_seconds':prior_train_seconds if uses_prior and method!='zero' else 0,
                            'source_prior_validation_predict_seconds':prior_validation_seconds if uses_prior else 0,
                            'pipeline_validation_predict_seconds':ps+(prior_validation_seconds if uses_prior else 0)})
                        for i,actual,pred in zip(valid_ids,yv,prediction):
                            predictions.append({**info,'group_id':i,'split':'validation','actual_ms':actual,'predicted_ms':pred})
                        method_predictions[method]=prediction
                    for a,b in [('residual','direct'),('prior_direct','direct'),('residual','prior_direct'),('constant_ratio','zero')]:
                        paired.append(dict(layer=layer,dtype=dtype,k=k,seed=seed,first=a,second=b,**gemm.bootstrap_delta(yv,method_predictions[a],method_predictions[b])))
            print(f'PASS: RQ3 {layer} {dtype} development fits',flush=True)
    aggregate=[]
    for layer in ['linear','ffn']:
        for dtype in ['fp16','fp32']:
            for method in METHODS:
                for k in BUDGETS:
                    rr=[r for r in scores if r['layer']==layer and r['dtype']==dtype and r['method']==method and r['k']==k]
                    aggregate.append(dict(layer=layer,dtype=dtype,method=method,k=k,effective_k=0 if method=='zero' else k,
                        mean_mape_pct=float(np.mean([r['mape_pct'] for r in rr])),mean_p90_pct=float(np.mean([r['p90_pct'] for r in rr])),failures=sum(r['failures'] for r in rr)))
    selected=[]
    for layer in ['linear','ffn']:
        for dtype in ['fp16','fp32']:
            candidates=[r for r in aggregate if r['layer']==layer and r['dtype']==dtype]
            passing=[r for r in candidates if r['mean_mape_pct']<=15 and r['mean_p90_pct']<=30 and not r['failures']]
            chosen=min(passing,key=lambda r:(r['effective_k'],r['mean_mape_pct'],r['method'],r['k'])) if passing else min(candidates,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['method'],r['k']))
            selected.append({**chosen,'goal_attained':bool(passing),'validation_groups':10,'all_layers_dtypes_standard_round_labels':12*(chosen['effective_k']+10)})
            for seed in SEEDS:
                model,ref=models[(layer,dtype,chosen['method'],chosen['k'],seed)]
                joblib.dump(dict(model=model,**chosen,seed=seed,features=FEATURES,source_ref=ref,source_sha256=p['source_weights'][ref],
                    protocol_sha256=gemm.sha(protocol_path),training_group_ids=p['subsets']['seeds'][str(seed)]['by_k'][str(chosen['k'])] if chosen['effective_k'] else [],
                    validation_only=True,locked_test_opened=False),output/'models'/f'{layer}_{dtype}_{seed}.joblib')
    for name,data in [('metrics.csv',scores),('aggregate_metrics.csv',aggregate),('validation_predictions.csv',predictions),('fit_times.csv',times),('paired_bootstrap.csv',paired),('selected_configurations.csv',selected)]:
        gemm.save_csv(output/name,data)
    costs=[]
    for k in BUDGETS:
        union={i for seed in SEEDS for i in p['subsets']['seeds'][str(seed)]['by_k'][str(k)]}
        costs.append(dict(k=k,per_seed_train_groups=k,validation_groups=10,per_seed_standard_round_labels=12*(k+10),
                          three_seed_train_union_groups=len(union),union_standard_round_labels=12*(len(union)+10),historical_gemm_gpu_wall_time='UNKNOWN'))
    gemm.save_csv(output/'label_costs.csv',costs)
    gemm.save_json(output/'selection_frozen.json',dict(timestamp=gemm.stamp(),protocol_sha256=gemm.sha(protocol_path),
        selected_configurations_sha256=gemm.sha(output/'selected_configurations.csv'),status='RQ3_SELECTION_FROZEN_VALIDATION_ONLY',
        rq2_explicit_diagnostics_may_be_separately_reviewed=True,locked_test_status='NOT_OPENED',standard_record_count=480,
        configurations=len(scores),rf_fits=144,wall_seconds=time.perf_counter()-start,total_model_fit_seconds=sum(r['model_fit_seconds'] for r in times),
        explicit_values_read=0,input_hashes_unchanged=all(gemm.sha(Path(input_directory)/name)==digest for name,digest in hashes.items())))
    print('PASS: RQ3 selection frozen for future diagnostics; test and explicit records NOT OPENED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--protocol',type=Path,default=HERE/'frozen_fit_protocol.json')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.protocol)
    elif args.input is None or args.output is None:
        parser.error('run requires --input and --output')
    else:
        run(args.input.resolve(),args.output,args.protocol)


if __name__=='__main__':
    main()
