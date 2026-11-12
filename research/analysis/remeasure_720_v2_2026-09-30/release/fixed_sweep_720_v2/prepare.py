#!/usr/bin/env python3
"""Copy a stopped v1 collection into v2 without changing the original run."""
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import tempfile

import run
from analyze_sweep import analyze


def prepare(source, output, loaded):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate directories')
    output.parent.mkdir(parents=True, exist_ok=True)
    suite = loaded['suite']
    with contextlib.ExitStack() as locks:
        locks.enter_context(suite.lock(output.parent/'.prepare.lock'))
        if output.exists():
            manifest = run.manifest_for(output, 'full', loaded)
            if run.read(output/'upgrade/provenance.json')['source_directory'] != str(source):
                raise RuntimeError('Existing output came from a different source')
            _, done = run.choices_and_done(output, manifest, loaded)
            return dict(output=str(output), imported=False, recorded=len(done))
        if not (source/'manifest.json').is_file():
            raise ValueError('Missing v1 source manifest')
        # The launcher retains its lock through cleanup and ZIP creation.
        for name in ('.launcher.lock', '.run.lock'):
            try:
                locks.enter_context(suite.lock(source/name))
            except RuntimeError as error:
                raise RuntimeError('v1 is still running or packaging. Press Control+C once in its '
                                   'tmux, wait for RESULT_ZIP and the prompt, then retry.') from error
        old = run.read(source/'manifest.json')
        wanted = run.read(run.HERE.parent/'fixed_sweep_720_v1/protocol.json')
        source_hashes = {Path(name).name: digest for name, digest in
                         run.read(run.HERE/'dependency_hashes.json').items()
                         if name.startswith('fixed_sweep_720_v1/')}
        expected = dict(run.make_plan('full', {**loaded, 'config': wanted}),
                        kind='fixed_sweep_720_v1', source_hashes=source_hashes,
                        dependency_hashes=run.read(run.HERE.parent/'fixed_sweep_720_v1/dependency_hashes.json'),
                        environment_signature=loaded['baseline']['environment_signature'],
                        pinned_graph_choices=run.pinned_choices(loaded))
        if old != json.loads(json.dumps(expected)):
            raise ValueError('Source is not the frozen v1 full plan')
        for folder, hashes in [('source', old['source_hashes']), ('dependencies', old['dependency_hashes'])]:
            if any(run.sha(source/folder/name) != digest for name, digest in hashes.items()):
                raise ValueError('Saved v1 source/dependency mismatch')
        launches = [run.read(p) for p in (source/'launches').glob('*/status.json')]
        if not launches or any(s['restoration'] != 'COMMANDS_ACCEPTED' for s in launches):
            raise RuntimeError('v1 clock cleanup is not confirmed; inspect the old run first')
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Unexpected symlink in the source run')
        files = {str(p.relative_to(source)): run.sha(p) for p in sorted(source.rglob('*'))
                 if p.is_file() and p.name not in ('.launcher.lock', '.run.lock')}
        with tempfile.TemporaryDirectory(prefix='.prepare_720_', dir=output.parent) as temp:
            stage = Path(temp)/output.name
            shutil.copytree(source, stage, ignore=shutil.ignore_patterns('.launcher.lock', '.run.lock'))
            if any(run.sha(stage/name) != digest for name, digest in files.items()):
                raise RuntimeError('Source copy mismatch')
            provenance = stage/'upgrade'
            provenance.mkdir()
            (stage/'manifest.json').rename(provenance/'source_manifest.json')
            (stage/'source').rename(provenance/'source')
            # Build the new manifest before attaching the copied database.
            replacement = Path(temp)/'new_manifest'
            replacement.mkdir()
            manifest = run.manifest_for(replacement, 'full', loaded)
            shutil.copytree(replacement, stage, dirs_exist_ok=True)
            _, done = run.choices_and_done(stage, manifest, loaded)
            if 'FAIL' in done.values():
                raise RuntimeError('Saved computation FAIL requires review before upgrading')
            flagged = run.require_clean_resume(stage, loaded)
            summary = analyze(stage, loaded)
            if not summary['all_restorations_accepted']:
                raise RuntimeError('v1 reset command evidence is invalid')
            if any(run.sha(source/name) != digest for name, digest in files.items()):
                raise RuntimeError('Source changed during preparation')
            suite.atomic_json(provenance/'source_files_sha256.json', files)
            suite.atomic_json(provenance/'provenance.json', dict(source_directory=str(source),
                source_manifest_sha256=files['manifest.json'], imported_records=len(done),
                raw_attempts=summary['attempts'], policy=loaded['config']['collection_policy'],
                source_unchanged=True, timestamp=suite.now()))
            suite.atomic_json(stage/'progress.json', dict(state='READY_TO_RESUME',
                recorded=len(done), expected=len(manifest['tasks']), flagged=flagged, updated_utc=suite.now()))
            stage.rename(output)
        return dict(output=str(output), imported=True, recorded=len(done),
                    remaining=len(manifest['tasks'])-len(done), flagged=flagged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, run.load()), indent=2))
