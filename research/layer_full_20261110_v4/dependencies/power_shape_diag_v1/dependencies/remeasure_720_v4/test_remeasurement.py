#!/usr/bin/env python3
"""CPU replay of the actual import plus existing worker safety checks."""
import argparse
import ast
import contextlib
import copy
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import analyze_sweep
import prepare
import progress
import run


def rejected(call, fragment):
    try:
        call()
    except RuntimeError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError('Expected rejection: ' + fragment)


def main(archive, output):
    previous = run.HERE.parent / 'fixed_sweep_720_v2'
    frozen = run.HERE.parent / 'remeasure_720_v3'
    old_tree, new_tree = [{n.name: ast.dump(n) for n in ast.parse((folder / 'run.py').read_text()).body
                           if isinstance(n, ast.FunctionDef)} for folder in (frozen, run.HERE)]
    for name in old_tree.keys() - {'load', 'manifest_for', 'worker'}:
        assert old_tree[name] == new_tree[name], name
    assert (run.HERE / 'analyze_sweep.py').read_bytes() == (previous / 'analyze_sweep.py').read_bytes()
    old = run.read(previous / 'protocol.json')
    new = run.read(run.HERE / 'protocol.json')
    assert new['reviewed_graph_cases'][:len(old['reviewed_graph_cases'])] == old['reviewed_graph_cases']
    assert len(new['reviewed_graph_cases']) == len(old['reviewed_graph_cases']) + 14
    assert {k: v for k, v in new.items() if k != 'remeasurement_schedule'} == run.read(frozen / 'protocol.json')
    spec = importlib.util.spec_from_file_location('frozen_v2_tests', previous / 'test_sweep.py')
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    names = [
        'test_only_exact_720_supported',
        'test_only_clock_and_sw_power_cap_are_nonfatal',
        'test_worker_continues_after_clock_flag_and_resumes_but_stops_on_fatal',
        'test_controller_sets_720_and_restores_on_failure',
        'test_denominator_or_accuracy_corruption_fails',
        'test_worker_resume_collects_only_pending_tasks',
        'test_manifest_change_and_output_lock_rejected',
        'test_full_coverage_modes_rounds_and_no_new_auxiliary',
    ]
    class GroupChecks(tests.Checks):
        def test_worker_separates_groups_and_resumes(self):
            fixture_tasks = self.manifest['tasks'][:9]
            directory, fixtures = self.synthetic_segment(fixture_tasks)
            by_id = {r['task']['id']: r for r in fixtures}
            config = copy.deepcopy(self.loaded['config'])
            config['remeasurement_schedule']['isolated_fp32_shapes'][0] = fixture_tasks[0]['shape']
            loaded = {**self.loaded, 'config': config}
            plan = run.make_plan('check', loaded)
            plan['tasks'] = fixture_tasks
            target = Path(self.tmp.name) / 'separated'
            target.mkdir()
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (10*1024**3,20*1024**3)),
                inference_mode=contextlib.nullcontext, randn=lambda *a, **k: 1., empty_like=lambda _: 0.)
            @contextlib.contextmanager
            def monitor(segment, *args):
                shutil.copy2(directory/'telemetry_metadata.json', segment/'telemetry_metadata.json')
                text = (directory/'telemetry.csv').read_text()
                if run.read(segment/'collection_group.json')['group'] == 'isolated_fp32':
                    text = text.replace(',720,6801,', ',705,6801,', 1)
                (segment/'telemetry.csv').write_text(text)
                yield SimpleNamespace(poll=lambda: None)
            def measured(torch, task, protocol, graph_nodes=None):
                return {k: copy.deepcopy(by_id[task['id']][k]) for k in ('timing','correctness','input_seed')}
            ticks = iter(range(0,100000,1000))
            args = argparse.Namespace(output=target, limit=7, retry_failed=False)
            with patch.object(run, 'make_plan', return_value=plan), patch.dict(sys.modules, torch=fake_torch), \
                 patch.object(loaded['suite'], 'environment', return_value=self.manifest['environment_signature']), \
                 patch.object(loaded['audit'], 'check_processes'), \
                 patch.object(loaded['telemetry'], 'collect', side_effect=monitor), \
                 patch.object(loaded['measure'], 'measure', side_effect=measured), \
                 patch.object(run.time, 'monotonic', side_effect=lambda: next(ticks)):
                run.manifest_for(target, 'check', loaded)
                self.assertEqual(run.worker(args, loaded), 0)
                self.assertEqual(len(list(run.history(target))), 7)
                args.limit = None
                self.assertEqual(run.worker(args, loaded), 0)
            rows = [r for _, r in run.history(target)]
            self.assertEqual([r['task']['id'] for r in rows],
                             [t['id'] for t in fixture_tasks[3:] + fixture_tasks[:3]])
            statuses = []
            for path in (target/'sessions').glob('*/segment_*/quality.json'):
                quality = run.read(path)
                group = run.read(path.parent/'collection_group.json')['group']
                expected = 'PASS' if group == 'other' else 'REVIEW'
                self.assertEqual({q['status'] for q in quality['measurements'].values()}, {expected})
                statuses.extend(q['status'] for q in quality['measurements'].values())
            self.assertEqual(statuses.count('PASS'), 6)
            self.assertEqual(statuses.count('REVIEW'), 3)
    suite = unittest.TestSuite(tests.Checks(n) for n in names)
    suite.addTest(GroupChecks('test_worker_separates_groups_and_resumes'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    original_sha = run.sha(archive)
    assert prepare.prepare(archive, output) == prepare.REDO
    loaded = run.load()
    manifest = run.manifest_for(output, 'full', loaded)
    choices, done = run.choices_and_done(output, manifest, loaded)
    ids = set(run.read(output / 'recovery/remeasure_ids.json'))
    pending = [t for t in manifest['tasks'] if t['id'] not in done]
    assert {t['id'] for t in pending} == ids and len(done) == prepare.KEEP
    assert sum(t['split'] == 'fresh_holdout' for t in pending) == 257
    batches = list(run.collection_batches(pending, loaded['config']))
    assert [(g,len(ts)) for g,ts in batches] == [('other',120)]*14 + [('other',69),('isolated_fp32',51)]
    ordered = [t for _,ts in batches for t in ts]
    assert len(ordered) == len({t['id'] for t in ordered}) == 1800
    proposal = run.read(run.HERE.parent / 'analysis/remeasure_720_v3_result_2026-10-01/next_groups_PROPOSED.json')
    assert batches[-1][1] == proposal['isolation_group']
    assert ordered[:-51] == proposal['other_remaining']
    # Every possible completed prefix must resume in order without mixing groups.
    for count in range(len(ordered)+1):
        remaining_ids = {t['id'] for t in ordered[count:]}
        remaining = [t for t in pending if t['id'] in remaining_ids]
        resumed = list(run.collection_batches(remaining, loaded['config']))
        assert [t for _,ts in resumed for t in ts] == ordered[count:]
        assert all(1 <= len(ts) <= 120 for _,ts in resumed)
        assert not any(g == 'other' for g,_ in resumed[next((i for i,(g,_) in enumerate(resumed) if g == 'isolated_fp32'),len(resumed)):])

    assert len(choices) == 12144 and loaded['plan'].digest(choices) == prepare.GRAPH_DIGEST
    assert prepare.prepare(archive, output) == prepare.REDO
    assert '[Remeasure] 0/1,800' in progress.line(output)
    assert '111,582/113,382' in progress.line(output)
    for name in ('.launcher.lock', '.run.lock'):
        with loaded['suite'].lock(output / name):
            rejected(lambda: prepare.prepare(archive, output), 'Another')
    with tempfile.TemporaryDirectory() as temp:
        unrelated = Path(temp) / 'existing'
        unrelated.mkdir()
        (unrelated / 'keep.txt').write_text('keep')
        rejected(lambda: prepare.prepare(archive, unrelated), 'incomplete or unrelated')
        bad = Path(temp) / 'wrong.zip'
        bad.write_bytes(b'wrong')
        rejected(lambda: prepare.prepare(bad, Path(temp) / 'new-output'), 'different reviewed source ZIP')
        assert (unrelated / 'keep.txt').read_text() == 'keep'
    # The last source segment was removed. A new SQLite row can reuse its seq.
    receipt = run.read(output / 'recovery_provenance.json')
    assert receipt['imported_last_seq'] < receipt['source_last_seq']
    simulated = {'task': next(t for t in pending if t['mode'] == 'allocate'), 'status': 'PASS'}
    store = loaded['suite'].Store(output)
    try:
        store.append(simulated)
        seq = store.con.execute('SELECT MAX(seq) FROM attempts').fetchone()[0]
        assert receipt['imported_last_seq'] < seq <= receipt['source_last_seq']
        assert prepare.verify_prepared(output, loaded) == prepare.REDO - 1
    finally:
        with store.con:
            store.con.execute('DELETE FROM attempts WHERE task_id=?', (simulated['task']['id'],))
        store.close()
    assert prepare.verify_prepared(output, loaded) == prepare.REDO
    evidence = run.read(output / 'recovery/imported_evidence_sha256.json')
    target = output / next(iter(evidence))
    saved = target.read_bytes()
    try:
        target.write_bytes(saved + b'\n')
        rejected(lambda: prepare.verify_prepared(output, loaded), 'Imported evidence changed')
    finally:
        target.write_bytes(saved)
    print('CPU telemetry replay: all imported segments; no GPU execution.', flush=True)
    summary = analyze_sweep.analyze(output, loaded)
    assert summary['recorded'] == prepare.KEEP and summary['missing'] == prepare.REDO
    assert summary['counts'] == {'PASS': prepare.KEEP} and summary['gate'] == 'INCOMPLETE'
    assert not summary['clock_flags'] and summary['all_restorations_accepted']
    assert prepare.verify_prepared(output, loaded) == prepare.REDO
    assert run.sha(archive) == original_sha == prepare.ARCHIVE_SHA
    result = dict(status='PASS', gpu_execution=False, existing_worker_checks=names,
                  imported=prepare.KEEP, pending=prepare.REDO, holdout_pending=257,
                  original_graph_choices=len(choices), retained_extra_pinned_graph_choices=14, trailing_seq_resume_checked=True,
                  other_tasks=1749, isolated_fp32_tasks=51, collection_segments=16,
                  all_1801_resume_prefixes_checked=True, worker_group_isolation_checked=True,
                  exact_pending_ids=True, idempotent_preparation=True, source_unchanged=True,
                  active_locks_rejected=True, wrong_archive_and_unrelated_output_rejected=True,
                  changed_evidence_rejected=True, telemetry_replay_summary=summary)
    (output.parent / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'telemetry_replay_summary'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.archive.resolve(), args.output.resolve())
