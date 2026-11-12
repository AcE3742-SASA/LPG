#!/usr/bin/env python3
"""Generate 10000 new stratified shapes without opening any timing labels."""
from bisect import bisect_left, insort
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
SEED=20261028
STRATA=list(itertools.product(['small','medium','large'],['compact','elongated'],['aligned8','unaligned8']))


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class Neighbours:
    def __init__(self,shapes=()):
        self.axes=defaultdict(list)
        for shape in sorted(set(map(tuple,shapes))):
            self.add(shape)

    def add(self,shape):
        for axis in range(3):
            key=(axis,tuple(shape[i] for i in range(3) if i!=axis))
            insort(self.axes[key],shape[axis])

    def near(self,shape):
        for axis in range(3):
            values=self.axes.get((axis,tuple(shape[i] for i in range(3) if i!=axis)),())
            position=bisect_left(values,shape[axis]-8)
            if position<len(values) and values[position]<=shape[axis]+8:
                return True
        return False


def held_shapes():
    meta=json.loads((HERE/'metadata/gemm_shape_metadata.json').read_text())
    shapes={tuple(r['shape']) for r in meta['shape_splits']}
    shapes.update(map(tuple,meta['nonmain_shapes']))
    shapes.update(map(tuple,json.loads((HERE/'metadata/protected_shapes.json').read_text())))
    groups=json.loads((HERE/'metadata/groups.json').read_text())
    for g in groups:
        shapes.update([(g['M'],g['F'],g['D']),(g['M'],g['D'],g['F'])])
    return shapes


def stratum(shape):
    f=2*math.prod(shape)
    work='small' if f<=1e7 else 'medium' if f<=1e9 else 'large'
    aspect='elongated' if max(shape)/min(shape)>=8 else 'compact'
    alignment='aligned8' if all(v%8==0 for v in shape) else 'unaligned8'
    return (work,aspect,alignment)


def generate():
    rng=random.Random(SEED)
    protected=held_shapes()
    neighbours=Neighbours(protected)
    buckets={s:[] for s in STRATA}
    quotas={s:(500 if s[1]=='compact' else 250) if s[0]=='small'
            else 1000 if s[0]=='medium' else 1125 for s in STRATA}
    train_quotas={s:math.floor(.9*quotas[s]) for s in STRATA}
    remaining=9000-sum(train_quotas.values())
    for s in sorted(STRATA,key=lambda s:(-(.9*quotas[s]-train_quotas[s]),s))[:remaining]:
        train_quotas[s]+=1
    attempts=0
    anchors=[64,128,256,512,1024,2048,4096]
    while any(len(buckets[s])<quotas[s] for s in STRATA):
        attempts+=1
        if attempts>3000000:
            raise RuntimeError('Insufficient unused shapes for fixed strata: '+str({str(s):len(v) for s,v in buckets.items()}))
        shape=[]
        for axis in range(3):
            if rng.random()<.25:
                value=rng.choice(anchors)+rng.choice([-8,0,8])
            else:
                value=8*round(math.exp(rng.uniform(math.log(64),math.log(4096)))/8)
            shape.append(max(64,min(4096,value)))
        if rng.random()<.5:
            axis=rng.randrange(3)
            shape[axis]+=rng.choice([-1,1,3,7])
        shape=tuple(shape)
        if any(not 64<=v<=4096 for v in shape):
            continue
        bucket=stratum(shape)
        if len(buckets[bucket])>=quotas[bucket] or neighbours.near(shape):
            continue
        buckets[bucket].append(shape)
        neighbours.add(shape)
    rows=[]
    for bucket in STRATA:
        ordered=sorted(buckets[bucket],key=lambda s:digest(dict(seed=SEED,shape=s,role='split')))
        for i,shape in enumerate(ordered):
            rows.append(dict(shape=list(shape),split='expansion_train' if i<train_quotas[bucket] else 'expansion_validation',
                stratum='_'.join(bucket),shape_id=digest(dict(shape=shape))[:24]))
    rows.sort(key=lambda r:r['shape'])
    return rows,dict(seed=SEED,attempts=attempts,shape_count=len(rows),expected_measurements=18*len(rows),
        split_counts=dict(Counter(r['split'] for r in rows)),
        stratum_counts=dict(Counter(r['stratum'] for r in rows)),
        split_stratum_counts=dict(Counter(r['split']+':'+r['stratum'] for r in rows)),
        sampling_allocation='1500 small-F (1000 compact, 500 elongated), 4000 medium-F, 4500 large-F; half aligned8 in each work/aspect cell',
        dimension_min=64,dimension_max=4096,protected_shape_count=len(protected),
        protection='Exclude exact and one-axis distance <=8 from all existing GEMM/main/control/boundary, protected layer and new layer component shapes; also exclude such neighbours across every new shape.',
        timing_labels_used=False,existing_splits_changed=False,new_final_test_created=False)


def main():
    if (HERE/'shape_manifest.json').exists():
        raise RuntimeError('Frozen shape manifest exists; refusing overwrite')
    rows,audit=generate()
    assert len(rows)==10000 and len({tuple(r['shape']) for r in rows})==10000
    assert audit['split_counts']=={'expansion_train':9000,'expansion_validation':1000}
    save('shape_manifest.json',rows)
    save('selection_audit.json',audit)
    config=json.loads((HERE/'protocol.json').read_text())
    cases=[]
    for bucket in [('small','compact','unaligned8'),('medium','elongated','aligned8'),('large','compact','aligned8'),('large','elongated','unaligned8')]:
        row=next(r for r in rows if r['stratum']=='_'.join(bucket) and r['split']=='expansion_train')
        cases.extend(dict(shape=row['shape'],dtype=dtype) for dtype in ['fp16','fp32'])
    config['check_cases']=cases
    config['reference_780_pilot']['conditions']=[]
    save('protocol.json',config)
    print('PASS: 10000 new unique shapes frozen; 9000 train, 1000 validation, 180000 measurements')


if __name__=='__main__':
    main()
