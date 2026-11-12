"""Verify saved comparison results and trusted model weights, without source labels."""
from collections import Counter
import csv
import json

import joblib
import numpy as np

import compare_learners as c

OUT=c.HERE/'runs/v1'


def key(row,method=None):
    return tuple(row[k] if k!='method' or method is None else method
                 for k in ['rq','dtype','mode','method','n','seed','shape'])


def main():
    complete=json.loads((OUT/'execution_completed.json').read_text())
    assert complete['configurations']==1920 and complete['source_hashes_unchanged']
    hashes=json.loads((OUT/'artifact_hashes.json').read_text())
    # The runner writes its final status line after indexing the redirected console.
    assert all(c.old.sha(OUT/path)==digest for path,digest in hashes.items() if path!='console.log')
    reference={}
    with (c.BASE/'runs/v2/validation_predictions.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if (row['rq']=='RQ1' and row['method'].startswith(('rf_','ridge_'))) or row['rq']=='RQ4':
                reference[key(row)]=(float(row['actual_ms']),float(row['predicted_ms']))
    matched=0; predictions={}; counts=Counter()
    with (OUT/'validation_predictions.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            counts[row['rq']]+=1
            if row['rq']=='RQ1' and row['learner'] in {'rf','ridge'} and int(row['n'])<=2652:
                method=row['learner']+'_'+row['method']
            elif row['rq']=='RQ4' and row['learner']=='rf':
                method=row['method']
            else:
                method=None
            if method is not None:
                prior=reference[key(row,method)]
                assert prior==(float(row['actual_ms']),float(row['predicted_ms'])),key(row,method)
                matched+=1
            if row['rq']=='RQ1' and row['n'] in {'2652','3345'}:
                k=tuple(row[f] for f in ['learner','dtype','mode','method','n','seed'])
                predictions.setdefault(k,[]).append(float(row['predicted_ms']))
    assert counts=={'RQ1':720*716,'RQ4':1200*30}
    assert matched==len(reference)==215208
    protocol=json.loads((OUT/'frozen_protocol.json').read_text())
    v=list(map(tuple,protocol['validation_shapes']))
    verified=[]
    for path in sorted((OUT/'models').glob('*.joblib')):
        assert c.old.sha(path)==hashes[str(path.relative_to(OUT))]
        envelope=joblib.load(path)
        k=tuple(str(envelope[f]) for f in ['learner','dtype','mode','target','n','seed'])
        x,lower=c.old.design(v,envelope['dtype'])
        pred=np.exp(envelope['model'].predict(x))*(lower if envelope['target']=='normalized' else 1)
        assert np.array_equal(pred,np.asarray(predictions[k])),path.name
        assert len(envelope['train_shapes'])==envelope['n']
        verified.append(dict(file=path.name,validation_predictions=len(pred),exact_match=True,sha256=c.old.sha(path)))
    assert len(verified)==288
    c.old.save_csv(OUT/'model_weight_verification.csv',verified)
    verification=dict(original_baseline_predictions_exact=matched,endpoint_models_exact=len(verified),
        prediction_counts=dict(counts),artifact_hashes_verified=len(hashes)-int('console.log' in hashes),
        console_note='Final status line is appended after the primary artifact index; numerical artifacts are verified against the original index. Final closed-log hash is recorded in artifact_hashes_final.json.',
        locked_test='NOT_OPENED')
    c.old.save_json(OUT/'verification.json',verification)
    print('PASS: 215208 previous RF/Ridge predictions reproduced; 288 saved models exactly verified')


if __name__=='__main__':
    main()
