#!/usr/bin/env python3
"""CPU replay of the actual import plus existing worker safety checks."""
import argparse
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
    assert (run.HERE / 'run.py').read_bytes() == (previous / 'run.py').read_bytes()
    assert (run.HERE / 'analyze_sweep.py').read_bytes() == (previous / 'analyze_sweep.py').read_bytes()
    old = run.read(previous / 'protocol.json')
    new = run.read(run.HERE / 'protocol.json')
    assert new['reviewed_graph_cases'][:len(old['reviewed_graph_cases'])] == old['reviewed_graph_cases']
    assert len(new['reviewed_graph_cases']) == len(old['reviewed_graph_cases']) + 14
    assert {k: v for k, v in old.items() if k != 'reviewed_graph_cases'} == {
        k: v for k, v in new.items() if k != 'reviewed_graph_cases'}
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
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests.Checks(n) for n in names))
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
