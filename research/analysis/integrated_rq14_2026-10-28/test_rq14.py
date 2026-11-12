"""Small synthetic checks for the label boundary and model contracts."""
import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from run_rq14 import (BUDGETS, FEATURES, RF, SEEDS, design, fit_model,
                       load_labels, metrics, ordered, predict, bootstrap_delta)


def test_label_boundary():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'conditions.csv'
        fields=['phase','split','shape','dtype','mode','accepted_rounds','median_of_round_medians_ms']
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for dtype in ['fp16','fp32']:
                writer.writerow(dict(phase='main',split='legacy_train',shape='[64,64,64]',dtype=dtype,mode='allocate',accepted_rounds=3,median_of_round_medians_ms=.01))
                # If disallowed labels are parsed, the load fails.
                writer.writerow(dict(phase='main',split='fresh_holdout',shape='[64,64,64]',dtype=dtype,mode='allocate',accepted_rounds=3,median_of_round_medians_ms='DO_NOT_PARSE'))
                writer.writerow(dict(phase='main',split='legacy_validation',shape='[128,128,128]',dtype=dtype,mode='graph',accepted_rounds=3,median_of_round_medians_ms='DO_NOT_PARSE'))
        access=[]
        labels=load_labels('legacy_train',[(64,64,64)],['allocate'],access,path)
        assert len(labels)==2 and len(access)==1
        try:
            load_labels('fresh_holdout',[(64,64,64)],['allocate'],access,path)
        except ValueError as error:
            assert str(error)=='Label access denied'
        else:
            raise AssertionError('Locked access was accepted')


def test_models_and_failures():
    shapes=[(64+16*i,128+8*i,256) for i in range(40)]
    x,lower=design(shapes,'fp32')
    assert x.shape==(40,len(FEATURES)) and (lower>0).all()
    y=4*lower
    for method in ['constant_efficiency','launch_scale','rf_normalized','ridge_normalized']:
        model,_=fit_model(method,x,y,lower,SEEDS[0])
        pred,_=predict(method,model,x,lower)
        assert np.allclose(y,pred,rtol=1e-7,atol=1e-10),method
    for method in ['zero','constant_ratio','residual','direct','prior_direct']:
        model,_=fit_model(method,x,y,lower,SEEDS[0],y/2)
        pred,_=predict(method,model,x,lower,y/2)
        assert np.isfinite(pred).all() and (pred>0).all()
        if method in ['constant_ratio','residual']:
            assert np.allclose(y,pred)
    result=metrics(np.ones(4),np.array([1,np.inf,0,-1]))
    assert result['failures']==3 and result['coverage']==.25 and math.isinf(result['mape_pct'])
    assert not result['meets_15_30']
    ci=bootstrap_delta(y,y,2*y)
    assert np.isclose(ci['delta_mape_pp'],-100) and np.isclose(ci['ci95_high_pp'],-100)
    a=ordered(shapes,'A:20261028')
    assert a==ordered(list(reversed(shapes)),'A:20261028')
    assert set(a[:5])<=set(a[:10])<=set(a[:30])
    assert RF['n_jobs']==1 and BUDGETS==[5,10,30,100,300]


if __name__=='__main__':
    test_label_boundary()
    test_models_and_failures()
    print('PASS: synthetic label-boundary, model-contract, nested-sampling and failure checks')

# Shape roles are group-wide, including dtype and execution mode.
def test_shape_grouping():
    import json
    from run_rq14 import metadata
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'conditions.csv';roles=Path(tmp)/'roles.json'
        roles.write_text(json.dumps([dict(shape=[64,64,64],internal_role='A_train_candidate')]))
        fields=['phase','split','shape','dtype','mode','accepted_rounds','median_of_round_medians_ms']
        rows=[]
        for split,shape in [('legacy_train',[64,64,64]),('legacy_validation',[128,128,128])]:
            for dtype in ['fp16','fp32']:
                for mode in ['allocate','prealloc','graph']:
                    rows.append(dict(phase='main',split=split,shape=json.dumps(shape),dtype=dtype,mode=mode,accepted_rounds=3,median_of_round_medians_ms='DO_NOT_PARSE'))
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        a,c,v=metadata(path,roles)
        assert a==[(64,64,64)] and not c and v==[(128,128,128)]
        for row in rows:
            if row['split']=='legacy_validation':
                row['shape']='[64,64,64]'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        try:
            metadata(path,roles)
        except ValueError as error:
            assert str(error)=='Shape split overlap'
        else:
            raise AssertionError('Cross-split shape accepted')


if __name__=='__main__':
    test_shape_grouping()
    print('PASS: same-shape dtype/mode grouping and overlap guard')
