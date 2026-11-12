"""Reproduce development cases from frozen shape-only exclusions; no latencies."""
from collections import Counter
import json
from pathlib import Path

from compare import components, digest, forbidden_overlap


def select(reserved, seed):
    cases, anchors, rejected = [], {}, Counter()
    def allowed(group):
        return not any(forbidden_overlap(shape,reserved)
                       for case in group for shape in components(case))
    def conv_group(anchor):
        b,side,ci,co = anchor
        delta = -1 if (b+side+ci+co)%3==0 else 1
        variants = [('anchor',side,side,ci,co),('cin_neighbour',side,side,ci+delta,co),
                    ('cout_neighbour',side,side,ci,co-delta),('spatial_rectangle',side-1,side+1,ci,co)]
        return [dict(layer=f'conv{k}',batch=b,height=h,width=w,cin=i,cout=o,kernel=k,
                     split='development',selection_family=f'conv_b{b}_s{side}_ci{ci}_co{co}',
                     selection_reason=reason)
                for k in (1,3) for reason,h,w,i,o in variants]
    pool=[]
    for b in (1,2,4):
        for side in (8,16,32,64):
            for ci,co in ((16,32),(32,64),(64,128),(128,256),(256,128)):
                anchor=(b,side,ci,co)
                if allowed(conv_group(anchor)): pool.append(anchor)
                else: rejected['conv'] += 1
    seen_grid, seen_channels, chosen = set(), set(), []
    while len(chosen)<16:
        best=max(pool,key=lambda v:(10*(v[:2] not in seen_grid)+3*(v[2] not in seen_channels),
                                    digest([seed,'conv',v])))
        pool.remove(best); chosen.append(best)
        seen_grid.add(best[:2]); seen_channels.add(best[2])
        cases.extend(conv_group(best))
    assert len(seen_grid)==12 and len(seen_channels)==5
    anchors['conv']=chosen
    for kind,widths in [('linear',[(64,64),(128,256),(256,512),(512,1024)]),
                        ('ffn',[(64,256),(128,512),(256,1024),(512,2048)])]:
        def group(anchor):
            m,d,n=anchor
            dims=[('anchor',m,d,n),('row_neighbour',m+1,d,n),
                  ('input_neighbour',m,d+1,n),('output_neighbour',m,d,n+1)]
            result=[]
            for reason,rows,inp,out in dims:
                case=dict(layer=kind,rows=rows,input_width=inp,split='development',
                          selection_family=f'{kind}_m{m}_d{d}_n{n}',selection_reason=reason)
                case['output_width' if kind=='linear' else 'hidden_width']=out
                result.append(case)
            return result
        pool=[]
        for m in (64,256,512,1024,2048,4096):
            for d,n in widths:
                anchor=(m,d,n)
                if allowed(group(anchor)): pool.append(anchor)
                else: rejected[kind] += 1
        seen_rows,seen_widths,chosen=set(),set(),[]
        while len(chosen)<4:
            best=max(pool,key=lambda v:(10*(v[1:] not in seen_widths)+(v[0] not in seen_rows),
                                       digest([seed,kind,v])))
            pool.remove(best); chosen.append(best)
            seen_rows.add(best[0]); seen_widths.add(best[1:])
            cases.extend(group(best))
        assert len(seen_widths)==4
        anchors[kind]=chosen
    for case in cases: case['id']='lc_'+digest(case)[:16]
    return cases,dict(algorithm='greedy coverage with seed-hash tie order; shape exclusions only',
                      anchors=anchors,rejected_families=dict(rejected))


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    reserved={tuple(v) for v in json.loads((here/'forbidden_shapes.json').read_text())}
    protocol=json.loads((here/'protocol.json').read_text())
    actual,_=select(reserved,protocol['seed'])
    assert actual==json.loads((here/'cases.json').read_text())
    print('SELECTION PASS: 160 reproducible development cases; no timing values used')
