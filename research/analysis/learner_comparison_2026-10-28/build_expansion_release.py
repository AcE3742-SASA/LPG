"""Create a separate release by adapting the preserved 720 MHz collector."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'gemm_expand_20261028_v1'


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'run.py').exists():
        raise RuntimeError('Release collector already exists; preserve it')
    deps=DEST/'dependencies'
    shutil.copytree(ROOT/'layer_transfer_20261028_v1/dependencies',deps)
    old=ROOT/'fixed_sweep_720_v2'
    for name,wanted in json.loads((old/'dependency_hashes.json').read_text()).items():
        origin=ROOT/name
        if hashlib.sha256(origin.read_bytes()).hexdigest()!=wanted:
            raise RuntimeError('Preserved dependency changed: '+name)
        target=deps/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(origin,target)
    source=(old/'run.py').read_text()
    source=source.replace('HERE.parent', 'DEPENDENCIES')
    source=source.replace("MODES = ('allocate', 'prealloc', 'graph')", "DEPENDENCIES = HERE / 'dependencies'\nMODES = ('allocate', 'prealloc', 'graph')")
    start=source.index('def own_hashes():')
    end=source.index('\ndef load(',start)
    source=source[:start]+'''def own_hashes():
    files=[*read(HERE/'release_hashes.json'),'release_hashes.json']
    return {name:sha(HERE/name) for name in sorted(files)}

'''+source[end:]
    source=source.replace('def load(sm_mhz=None):\n', "def load(sm_mhz=None):\n    release=read(HERE/'release_hashes.json')\n    if any(sha(HERE/name)!=digest for name,digest in release.items()):\n        raise RuntimeError('Expansion release hash mismatch')\n")
    start=source.index('        original = old.make_plan(\'full\')')
    end=source.index('    tasks = []',start)
    source=source[:start]+'''        rows = read(HERE/'shape_manifest.json')
        if (len(rows)!=10000 or len({tuple(r['shape']) for r in rows})!=10000
                or Counter(r['split'] for r in rows)!={'expansion_train':9000,'expansion_validation':1000}
                or any(len(r['shape'])!=3 or any(type(v) is not int or not 64<=v<=4096 for v in r['shape']) for r in rows)):
            raise ValueError('Invalid expansion shape manifest')
        bases=[]
        for round_index in range(3):
            ordered=sorted(rows,key=lambda r:old.digest(dict(round=round_index,shape=r['shape'])))
            for row in ordered:
                for dtype in ('fp16','fp32'):
                    bases.append(dict(kind='gemm',phase='main',shape=row['shape'],dtype=dtype,
                                      round=round_index,split=row['split'],group=row['stratum']))
        splits=rows
'''+source[end:]
    source=source.replace("kind='fixed_sweep_720_v2'", "kind='gemm_expand_20261028_v1'")
    source=source.replace("        args.output=args.output.resolve()", "        args.output=args.output.resolve()\n        if not args.output.is_relative_to(HERE/'runs'):\n            raise ValueError('Output must remain inside this release runs directory')")
    source=source.replace("                if path.is_file() and path.name!='.run.lock':", "                if path.is_symlink():\n                    raise ValueError('Symlinks are not allowed in result packages')\n                if path.is_file() and path.name!='.run.lock':")
    (DEST/'run.py').write_text(source)
    for name in ['analyze_sweep.py','dependency_hashes.json']:
        shutil.copy2(old/name,DEST/name)
    shutil.copy2(ROOT/'layer_transfer_20261028_v1/run_research_checked.sh',DEST/'run_research_checked.sh')
    config=json.loads((old/'protocol.json').read_text())
    config['minimum_free_disk_gib']=50
    config['expansion_version']='gemm_expand_20261028_v1'
    config['expected_expansion_shapes']=10000
    config['expected_expansion_measurements']=180000
    (DEST/'protocol.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    archive=ROOT/'analysis/abstract_200_720_recovery_result_2026-09-22/abstract_200_720_recovery_20260921_20260922_024735_e61123.zip'
    (DEST/'admission').mkdir()
    shutil.copy2(archive,DEST/'admission'/archive.name)
    (DEST/'metadata').mkdir()
    for name in ['protected_shapes.json','metadata/gemm_shape_metadata.json','groups.json']:
        shutil.copy2(ROOT/'layer_transfer_20261028_v1'/name,DEST/'metadata'/Path(name).name)
    provenance={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [old/'run.py',old/'analyze_sweep.py',old/'protocol.json',archive,
                             ROOT/'layer_transfer_20261028_v1/run_research_checked.sh']}
    (DEST/'source_provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print('PASS: separate collector copied; original sources and results unchanged')


if __name__=='__main__':
    main()
