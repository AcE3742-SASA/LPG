"""Synthetic standard-only loader and model-contract checks, no GPU or real labels."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fit_layers as fit


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))


def fixture(out,p):
    config=fit.read(fit.PACKAGE/'protocol.json');config['phase']='core_dev'
    write(out/'manifest.json',dict(phase='core_dev',kind='layer_transfer_20261028_v1',split='train_validation_only',source_hashes=p['package_release_hashes'],cases=p['cases'],protocol=config))
    write(out/'status.json',dict(collection='COMPLETE',restoration='COMMANDS_ACCEPTED'))
    write(out/'worker/complete.json',dict(complete=True))
    write(out/'correctness/pass.json',dict(passed=True))
    flags={}
    for case in p['cases']:
        for dtype in ['fp16','fp32']:
            for rd in range(3):
                rid=f'{case["id"]}_{dtype}_r{rd}_standard';flags[rid]=[]
                write(out/'records'/f'{rid}.json',dict(id=rid,case=case,dtype=dtype,round=rd,implementation='standard',
                    timing=dict(median_ms=1+.001*rd,flags=[]),correctness_before='correctness/pass.json',correctness_after='correctness/pass.json'))
    write(out/'summary.json',dict(scope='core_dev',gate='PASS',collection='COMPLETE',comparisons=[],measurement_flags=flags))
    (out/'restricted_explicit').mkdir();(out/'restricted_explicit/DO_NOT_READ').write_text('PRIVATE TIMING: INVALID JSON')


def main():
    p=fit.read(fit.HERE/'frozen_fit_protocol.json')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)
        write(out/'manifest.json',dict(phase='core_locked'))
        with patch.object(Path,'glob',side_effect=AssertionError('Locked records listed')):
            try:
                fit.load_standard(out,p)
            except ValueError as error:
                assert str(error)=='Only core_dev standard labels are allowed'
            else:
                raise AssertionError('Locked phase accepted')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp);fixture(out,p)
        original_read=fit.read
        def guarded_read(path):
            assert 'restricted_explicit' not in Path(path).parts
            return original_read(path)
        with patch.object(fit,'read',side_effect=guarded_read):
            labels,hashes=fit.load_standard(out,p)
        assert len(labels)==160 and len([name for name in hashes if name.startswith('records/')])==480
        sample=next((out/'records').glob('*.json'));row=fit.read(sample)
        row['round']=3;write(sample,row)
        try:
            fit.load_standard(out,p)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid round accepted')
        row['round']=int(row['id'].split('_r')[-1].split('_')[0]);row['timing']['flags']=['clock_deviation'];write(sample,row)
        try:
            fit.load_standard(out,p)
        except ValueError:
            pass
        else:
            raise AssertionError('Quality REVIEW accepted')
    groups=[dict(M=64+16*i,D=64,F=256) for i in range(20)]
    class ConstantSource:
        def predict(self,x):
            return np.full(len(x),np.log(.1))
    linear=fit.prior(groups,'linear','fp16',ConstantSource(),'rf_direct')
    ffn=fit.prior(groups,'ffn','fp16',ConstantSource(),'rf_direct')
    assert np.allclose(ffn,2*linear)
    x=fit.features(groups,'ffn','fp16');assert x.shape==(20,len(fit.FEATURES))
    y=3*ffn
    for method in fit.METHODS:
        model,_=fit.gemm.fit_model(method,x,y,np.ones(20),fit.SEEDS[0],ffn)
        prediction,_=fit.gemm.predict(method,model,x,np.ones(20),ffn)
        assert np.isfinite(prediction).all() and (prediction>0).all()
        if method in ['constant_ratio','residual']:
            assert np.allclose(prediction,y)
    print('PASS: synthetic 480-standard loader, locked-before-list denial, private-read guard, round/quality rejection and prior/model contracts')
    print('NO REAL LAYER FIT OR GPU RUN: locked test and explicit records remain closed')


if __name__=='__main__':
    main()
