"""Audit returned collection metadata and development quality without opening test labels."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSFER = Path('/Users/studyhard/Desktop/Transfer/resume_full_20261108_105717_8tqhrkir')
ANALYSIS = ROOT / 'analysis/layer_full_fit_2026-11-10_v5'
sys.path.insert(0, str(ANALYSIS))
import pipeline
import data
import costs


def audit():
    raw = HERE / 'raw'
    raw.mkdir(exist_ok=True)
    preserved = {}
    for path in sorted(TRANSFER.iterdir()):
        if not path.is_file() or path.name == '.campaign.lock':
            continue
        assert not path.is_symlink()
        target = raw / path.name
        digest = data.sha(path)
        if not target.exists():
            shutil.copy2(path, target)
        assert data.sha(target) == digest, path.name
        preserved[path.name] = digest
    assert len(list(raw.glob('*.zip'))) == 33
    state = data.read(raw / 'campaign.json')
    summary = data.read(raw / 'collection_summary.json')
    assert state['completed_batches'] == summary['batches'] == 30
    prepared = ANALYSIS / 'prepared_layer_full'
    protocol = pipeline.verify_protocol(prepared)
    dataset = data.load_campaign(raw, split='dev', diagnostic=True,
                                 protocol_path=prepared / 'frozen_protocol.json')
    assert dataset['provenance']['test_opened'] is False
    assert len(dataset['provenance']['phases']) == 25
    batches = []
    for phase in state['batch_order']:
        attempt = state['attempts'][phase][-1]
        with data.Artifact(raw, attempt) as artifact:
            manifest = artifact.read('manifest.json')
            s = artifact.read('summary.json')
            h = artifact.read('health_check.json')
            status = artifact.read('status.json')
            restore = artifact.read('restore.json')
            assert artifact.read('worker/complete.json')['complete'] is True
            assert not artifact.exists('failure.json')
            assert manifest['sealed_test'] is phase.startswith('test_')
            if phase in ('test_04', 'test_05'):
                package = ROOT / 'layer_full_20261110_v5'
                cases = data.read(package / 'cases_by_phase.json')[phase]
                frozen = data.read(package / 'protocol.json')
                exclusions = data.correctness_exclusions(artifact, phase, cases, frozen, state['source_hashes'])
                omitted = {row['case_id'] for row in exclusions}
                config = dict(frozen, phase=phase, health_case_ids=[next(c['id'] for c in cases
                              if c['layer'] == layer and c['id'] not in omitted) for layer in pipeline.LAYERS])
                assert manifest['kind'] == frozen['version'] and manifest['phase'] == phase
                assert manifest['cases'] == cases and manifest['protocol'] == config
                assert manifest['source_hashes'] == state['source_hashes']
                assert manifest['correctness_exclusions'] == s['excluded_conditions'] == exclusions
                assert manifest['planned_measurements'] == s['planned'] == 1080
                assert manifest['split'] == 'test_locked'
            assert s['collection'] == 'COMPLETE' and s['missing'] == 0
            assert s['recorded'] == s['expected'] == manifest['expected_measurements']
            assert s['counts'] == {'PASS': s['recorded']}
            assert not s['comparisons'] and not any(s['measurement_flags'].values())
            assert h['gate'] == 'PASS' and h['recorded'] == h['expected'] == 108
            assert not h['comparisons'] and not any(h['measurement_flags'].values())
            assert status == {'collection': 'COMPLETE', 'restoration': 'COMMANDS_ACCEPTED'}
            assert restore['commands_accepted'] is True
            correctness_files = [n for n in artifact.hashes if n.startswith('correctness/')]
            for name in correctness_files:
                assert artifact.read(name)['passed'] is True, name
            selected = {k: s[k] for k in ('gate', 'collection', 'expected', 'recorded', 'missing',
                                          'counts', 'comparison_counts', 'restoration')}
            selected.update(planned=s.get('planned', s['expected']), excluded=s.get('excluded', 0))
            reviews = s.get('component_round_review', [])
            assert all(len(row) == 3 and all(isinstance(v, str) for v in row) for row in reviews)
            batches.append(dict(phase=phase, attempt=attempt, collection_version=manifest['kind'],
                summary=selected, correctness_documents_passed=len(correctness_files),
                health=dict(gate=h['gate'], recorded=h['recorded'],
                            case_ids=manifest['protocol']['health_case_ids']),
                component_round_review=reviews, sensor_info=s['sensor_info']))
    cohorts = []
    for layer in pipeline.LAYERS:
        for dtype in ('fp16', 'fp32'):
            retained, _, details = pipeline.cohort(dataset, protocol, layer, dtype)
            details['retained_by_split'] = dict(Counter(c['split'] for c in retained))
            cohorts.append(details)
    held = [{k: row[k] for k in ('case_id', 'group_id', 'layer', 'dtype', 'split',
                                 'implementation', 'status', 'reasons', 'round_times_ms')}
            for row in dataset['excluded']]
    collection_costs = costs.collection_costs(raw)
    assert not collection_costs['all_attempts']['missing_ledgers']
    assert not collection_costs['all_attempts']['unverified_ledgers']
    totals = dict(completed_batches=len(batches), attempts=sum(map(len, state['attempts'].values())),
        recorded=sum(b['summary']['recorded'] for b in batches),
        planned=sum(b['summary']['planned'] for b in batches),
        excluded=sum(b['summary']['excluded'] for b in batches),
        review_batches=[b['phase'] for b in batches if b['summary']['gate'] == 'REVIEW'],
        correctness_documents_passed=sum(b['correctness_documents_passed'] for b in batches),
        dev_pass_conditions=len(dataset['rows']), dev_held_conditions=len(held),
        split_records={split: sum(b['summary']['recorded'] for b in batches if b['phase'].startswith(split))
                       for split in ('train', 'validation', 'test')})
    assert totals['recorded'] == summary['measurements'] == 32364
    assert totals['excluded'] == summary['excluded_measurements'] == 36
    assert totals['planned'] == 32400
    assert totals['review_batches'] == summary['review_batches']
    report = dict(status='PASS', source=str(TRANSFER), preserved_files=preserved, totals=totals,
        collection_summary=summary, batches=batches, common_cohorts=cohorts, dev_exclusions=held,
        dev_pass_by_split=dict(Counter(r['split'] for r in dataset['rows'])),
        costs=collection_costs, frozen_protocol_sha256=data.sha(prepared / 'frozen_protocol.json'),
        test_timing_labels_opened=False, model_training_performed=False,
        script_sha256=data.sha(Path(__file__)),
        reused_code_sha256={str(p.relative_to(ROOT)): data.sha(p)
                           for p in (ANALYSIS / 'data.py', ANALYSIS / 'pipeline.py', ANALYSIS / 'costs.py')})
    (HERE / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(totals, indent=2))
    print('Collection and development audit: PASS')


if __name__ == '__main__':
    audit()
