"""Check source contracts and execute public NeuSight operator test data."""
from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'prior_model_comparison_2026-11-11'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(OLD/'runtime'))
import adapters as a
from compare import read,csv_save,score
from prepare import sha,save


def contracts():
    frozen=read(OLD/'artifact_manifest.json')['artifacts']
    for name,digest in frozen.items():assert sha(OLD/name)==digest,name
    source=(OLD/'sources/neuralpower/predict_runtime_power.py').read_text()
    begin=source.index('def predict_runtime_power(');end=source.find('\ndef ',begin+5)
    if end<0:end=source.index('\nif __name__',begin)
    scope={'np':np};exec(source[begin:end],scope)
    dev=read(OLD/'development.json')
    cases={r['case']['id']:r['case'] for r in dev if r['case']['layer'] in ['linear','conv1','conv3']}
    rng=np.random.default_rng(20261111);checks=0;max_relative=0
    for case in cases.values():
        is_conv=case['layer'].startswith('conv')
        if is_conv:
            b,h,w,ci,co,k=(case[n] for n in ['batch','height','width','cin','cout','kernel'])
            raw=[b,h,w,co,k,k,ci,co,k//2,k//2,1,1,b,h,w,ci];kind='conv';power_len=128
            expected=(b*h*w,co,ci*k*k)
        else:
            raw=[case['rows'],case['output_width'],1,1,case['input_width']];kind='fc';power_len=66
            expected=(case['rows'],case['output_width'],case['input_width'])
        assert a.gemm_shape(case)==expected
        x=a.neuralpower_features([case])[0]
        for _ in range(4):
            weights=rng.uniform(.1,1,len(x)+1)
            coeff={(kind,'runtime'):weights,(kind,'power'):np.zeros(power_len)}
            original,_=scope['predict_runtime_power'](kind,raw,coeff)
            port=float(np.r_[x,1]@weights)
            np.testing.assert_allclose(port,original,rtol=1e-12,atol=1e-10)
            max_relative=max(max_relative,abs(port-original)/original);checks+=1
    linear=[c for c in cases.values() if c['layer']=='linear']
    for dtype,width,power in [('fp32',4,6.14e12),('fp16',2,35.02e12)]:
        actual=a.paleo(linear,dtype)
        expected=[]
        for c in linear:
            m,n,k=a.gemm_shape(c)
            expected.append(1000*(2*m*n*k/power+width*(m*k+k*n+m*n)/262.3e9))
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    result=dict(status='PASS',v1_files_unchanged=len(frozen),neuralpower_cases=len(cases),
        original_prediction_comparisons=checks,max_relative_difference=max_relative,
        gemmify_dimension_mappings=len(cases),paleo_unit_precision_comparisons=len(linear)*2,
        interpretation='Numerical feature, mapping and unit checks; author training-data scores are a separate requirement')
    save(HERE/'source_contracts.json',result)
    print('PASS: source feature, mapping and unit checks',result,flush=True)


def neusight_public():
    import pandas as pd
    neo=a.NeuSight();torch=neo.torch
    Dataset=importlib.import_module('neusight.Dataset.dataset').Dataset
    Meta=importlib.import_module('neusight.Model.meta').MetaTable
    directory=HERE/'sources/neusight_test'
    parser=Dataset.__new__(Dataset)
    parser.device_df=pd.DataFrame([read(p) for p in sorted((directory/'device_configs').glob('*.json'))])
    metrics,predictions=[],[];reference_checks=0
    for kind,file,op in [('LINEAR','linear.csv','linear'),('BMM','bmm.csv','bmm'),('VEC','vec.csv','gelu')]:
        raw=pd.read_csv(directory/file);raw=raw[raw['OPName']==op].copy()
        raw['SourceRow']=raw.index
        frame=parser.prepare(raw,None)
        assert len(raw)==len(frame)>0
        assert np.allclose(frame['Latency'].to_numpy()*1000,raw['Latency'].to_numpy(),rtol=1e-12,atol=1e-12)
        model=neo.models[kind];train=neo.tables[kind]
        x=frame[model.features].to_numpy(dtype=np.float32)
        reference=train[model.features].to_numpy(dtype=float)
        tile_data=train[['Tile1','Tile2']].to_numpy(dtype=np.float32)
        values=[];matched=[]
        for start in range(0,len(x),64):
            batch=x[start:start+64]
            idx=np.argpartition(cdist(batch,reference),1,axis=1)[:,0]
            with torch.inference_mode():
                out=model([op],torch.from_numpy(batch),tiles=torch.from_numpy(tile_data[idx])).reshape(-1).numpy()*1000
            values.extend(map(float,out));matched.extend(map(int,idx))
        values=np.array(values)
        meta=Meta.__new__(Meta);meta.features=model.features;meta.df=train
        for device,rs in frame.groupby('Device'):
            indices=rs.index.to_numpy()
            for i in indices[:3]:
                point=torch.from_numpy(x[i:i+1])
                tile=meta.get_tile(point,None,[op]).reshape(1,2)
                with torch.inference_mode():expected=float(model([op],point,tiles=tile).item()*1000)
                np.testing.assert_allclose(values[i],expected,rtol=2e-5,atol=1e-8)
                reference_checks+=1
            trained=device in set(train['Device'])
            actual=rs['Latency'].to_numpy()*1000
            metrics.append(dict(operator=kind,device=device,device_seen_in_public_train=trained,
                                **score(actual,values[indices])))
            for i,y in zip(indices,actual):
                predictions.append(dict(operator=kind,device=device,source_row=int(frame.loc[i,'SourceRow']),
                    actual_ms=float(y),predicted_ms=float(values[i]) if np.isfinite(values[i]) else None,
                    metadata_match=matched[i],device_seen_in_public_train=trained))
        print('PASS: public operator data scored',kind,len(frame),'rows',flush=True)
    csv_save(HERE/'neusight_public_metrics.csv',metrics)
    csv_save(HERE/'neusight_public_predictions.csv',predictions)
    save(HERE/'neusight_public_audit.json',dict(status='EXECUTED',original_scalar_forward_checks=reference_checks,
        rows=len(predictions),metrics=metrics,
        scope='Public operator measurements with authors device profiles and pretrained weights; independent of RTX target labels. This does not reproduce whole-network paper metrics.'))
    print('PASS: public NeuSight source-domain evaluation saved',flush=True)


if __name__=='__main__':
    contracts();neusight_public()
