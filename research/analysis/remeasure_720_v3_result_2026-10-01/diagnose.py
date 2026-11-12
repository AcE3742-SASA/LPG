"""Compare observed flags across attempts without exporting timing values."""
from collections import Counter
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    paths = [ROOT / 'analysis/full_720_result_2026-09-30/recomputed/full_720',
             ROOT / 'analysis/remeasure_720_v1_result_2026-09-30/recomputed/full_720_remeasure_20260930',
             BASE / 'recomputed/full_720_remeasure2_20260930']
    runs = []
    for out in paths:
        tasks = {t['id']: t for t in read(out / 'manifest.json')['tasks']}
        receipt = out / 'recovery_provenance.json'
        if receipt.exists():
            provenance = read(receipt)
            cutoff = provenance.get('imported_last_seq', provenance['source_last_seq'])
            with (out / 'raw_attempts.jsonl').open() as stream:
                sessions = {r['session'] for line in stream
                            if (r := json.loads(line))['seq'] > cutoff}
        else:
            sessions = {p.name for p in (out / 'sessions').iterdir() if p.is_dir()}
        direct, segment_only, power = {}, set(), set()
        for session in sorted(sessions):
            for path in (out / 'sessions' / session).glob('segment_*/quality.json'):
                for task_id, quality in read(path)['measurements'].items():
                    flags = [f for f in quality['flags'] if not f.startswith('segment:')]
                    if flags:
                        direct[task_id] = flags
                    elif quality['status'] == 'REVIEW':
                        segment_only.add(task_id)
                    if 'observed_active:clocks_event_reasons.sw_power_cap' in flags:
                        power.add(task_id)
        runs.append(dict(direct=direct, segment_only=segment_only, power=power))
    latest = runs[-1]
    assert len(latest['direct']) == 48 and len(latest['segment_only']) == 1752
    review = {t['task']['id'] for t in read(paths[-1] / 'remeasure_tasks.json')}
    assert set(latest['direct']).isdisjoint(latest['segment_only'])
    assert set(latest['direct']) | latest['segment_only'] == review
    keys = {(tuple(tasks[i]['shape']), tasks[i]['dtype']) for i in latest['direct']}
    risk_ids = {i for i in review if (tuple(tasks[i]['shape']), tasks[i]['dtype']) in keys}
    repeated = set.intersection(*(r['power'] for r in runs))
    shape_counts = []
    for shape, dtype in sorted(keys):
        ids = {i for i in tasks if tasks[i]['shape'] == list(shape) and tasks[i]['dtype'] == dtype}
        shape_counts.append(dict(shape=shape, dtype=dtype,
            directly_flagged=[len(ids & set(r['direct'])) for r in runs],
            power_flagged_all_three=len(ids & repeated),
            remaining_tasks=len(ids & review)))
    result = dict(direct_tasks=48, segment_only_tasks=1752,
        current_direct_flags=dict(Counter(f for flags in latest['direct'].values() for f in flags)),
        directly_observed_shapes=shape_counts, repeated_power_tasks_all_three=len(repeated),
        direct_counts_by_attempt=[len(r['direct']) for r in runs],
        power_counts_by_attempt=[len(r['power']) for r in runs],
        proposed_isolated_tasks=len(risk_ids), other_remaining_tasks=len(review-risk_ids),
        policy_changed=False, reclassified_as_pass=0, holdout_latencies_disclosed=False)
    (BASE / 'diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    (BASE / 'next_groups_PROPOSED.json').write_text(json.dumps(dict(
        status='PROPOSED_NOT_EXECUTED', selection_basis='Observed clock flags only; no timing selection',
        isolation_group=[tasks[i] for i in tasks if i in risk_ids],
        other_remaining=[tasks[i] for i in tasks if i in review-risk_ids]), indent=2) + '\n')
    assert len(risk_ids) + len(review-risk_ids) == 1800
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
