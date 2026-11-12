#!/usr/bin/env python3
"""CPU-only manifest, protection and collector checks. No GPU collection."""
from collections import Counter,defaultdict
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import build_manifest as build
import run


def main():
    rows=json.loads((run.HERE/'shape_manifest.json').read_text())
    rebuilt,audit=build.generate()
    assert rows==rebuilt
    assert audit==json.loads((run.HERE/'selection_audit.json').read_text())
    protected=build.Neighbours(build.held_shapes())
    selected=build.Neighbours()
    for row in rows:
        shape=tuple(row['shape'])
        assert not protected.near(shape)
        assert not selected.near(shape)
        selected.add(shape)
    loaded=run.load()
    full=run.make_plan('full',loaded)
    tasks=full['tasks']
    assert len(tasks)==180000 and len({t['id'] for t in tasks})==180000
    assert Counter(t['split'] for t in tasks)=={'expansion_train':162000,'expansion_validation':18000}
    conditions=defaultdict(set)
    seeds=defaultdict(set)
    for task in tasks:
        conditions[tuple(task['shape'])].add((task['dtype'],task['mode'],task['round']))
        seeds[(tuple(task['shape']),task['dtype'],task['round'])].add(loaded['plan'].task_seed(task))
    assert len(conditions)==10000 and all(len(v)==18 for v in conditions.values())
    assert all(len(v)==1 for v in seeds.values())
    assert len(run.make_plan('check',loaded)['tasks'])==48
    assert all(t['split']=='development' for t in run.make_plan('check',loaded)['tasks'])
    bad=[dict(r) for r in rows]
    bad[0]=dict(bad[0],split='fresh_holdout')
    original_read=run.read
    with patch.object(run,'read',side_effect=lambda p:bad if p.name=='shape_manifest.json' else original_read(p)):
        try:
            run.make_plan('full',loaded)
        except ValueError as error:
            assert 'Invalid expansion' in str(error)
        else:
            raise AssertionError('Foreign split accepted')
    assert all(name in run.own_hashes() for name in ['metadata/groups.json','shape_manifest.json','run.py'])
    with tempfile.TemporaryDirectory() as tmp:
        result=subprocess.run([sys.executable,'-B',str(run.HERE/'run.py'),'pack','--output',tmp],capture_output=True,text=True)
        assert result.returncode!=0 and 'Output must remain inside' in result.stderr
    for name in ['launch.sh','run_research_checked.sh','progress.py','run.py','build_manifest.py','selfcheck.py']:
        (run.HERE/name).read_text().encode('ascii')
        if name.endswith('.py'):
            ast.parse((run.HERE/name).read_text())
    for name in ['launch.sh','run_research_checked.sh']:
        subprocess.run(['bash','-n',str(run.HERE/name)],check=True)
    print('PASS: manifest reproducibility; 180000 tasks; split, neighbour, seed, output, release and ASCII checks')


if __name__=='__main__':
    main()
