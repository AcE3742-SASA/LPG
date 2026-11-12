#!/usr/bin/env python3
"""Account for observed collection cost and distinct development labels."""
import argparse
import json
import math
from pathlib import Path
import re


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Cost must be a finite nonnegative number of seconds')
    return float(value)


def label_budget(rows, rounds=3):
    """Rows explicitly name labels consumed; repeated methods/learners share labels."""
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError('Rounds must be a positive integer')
    by_seed = {}
    for row in rows:
        seed = str(row['seed'])
        if row['layer'] not in ('linear', 'ffn', 'conv1', 'conv3') or row['dtype'] not in ('fp16', 'fp32'):
            raise ValueError('Unknown label layer or dtype')
        allowed = {
            'linear': {'standard', 'decomposed'},
            'ffn': {'standard', 'decomposed', 'gemm1', 'gelu', 'gemm2'},
            'conv1': {'standard', 'decomposed', 'input_pack', 'mm', 'output_pack'},
            'conv3': {'standard', 'decomposed', 'unfold', 'input_pack', 'mm', 'output_pack'},
        }
        implementation = row.get('implementation', 'standard')
        if implementation not in allowed[row['layer']]:
            raise ValueError('Unknown label implementation')
        train, validation = row['train_group_ids'], row['validation_group_ids']
        if not isinstance(train, list) or not isinstance(validation, list):
            raise ValueError('Label group IDs must be lists')
        if set(train) & set(validation):
            raise ValueError('Train and validation groups overlap')
        labels = by_seed.setdefault(seed, set())
        for split, groups in (('train', train), ('validation', validation)):
            for group in groups:
                if not isinstance(group, str) or not group:
                    raise ValueError('Invalid label group ID')
                labels.add((split, group, row['layer'], row['dtype'], implementation))
    union = set().union(*by_seed.values()) if by_seed else set()
    train_groups = {key[1] for key in union if key[0] == 'train'}
    validation_groups = {key[1] for key in union if key[0] == 'validation'}
    if train_groups & validation_groups:
        raise ValueError('Group appears in train and validation across label rows')
    def count(labels):
        return dict(aggregated_labels=len(labels), round_records=rounds * len(labels),
                    train_round_records=rounds * sum(key[0] == 'train' for key in labels),
                    validation_round_records=rounds * sum(key[0] == 'validation' for key in labels),
                    by_implementation={name: rounds * sum(key[4] == name for key in labels)
                                       for name in sorted({key[4] for key in labels})})
    return dict(scope='Retrospective distinct labels; not measured collection time saved',
                rounds_per_label=rounds, per_seed={seed: count(labels) for seed, labels in sorted(by_seed.items())},
                all_seed_union=count(union))


def collection_costs(campaign):
    """Read only public metadata, including failed attempts and sealed-test cost."""
    from data import Artifact, validate_inheritance, verify_inherited_archive, inherited_attempt
    campaign = Path(campaign).resolve()
    state = json.loads((campaign / 'campaign.json').read_text())
    inheritance = validate_inheritance(campaign, state)
    phases = state['batch_order']
    if len(phases) != len(set(phases)) or any(not re.fullmatch(r'(train|validation|test)_\d{2}', p) for p in phases):
        raise ValueError('Invalid campaign phase order')
    if set(state['attempts']) - set(phases):
        raise ValueError('Unknown phase in campaign attempts')
    rows = []
    for phase in phases:
        names = state['attempts'].get(phase, [])
        if len(names) != len(set(names)):
            raise ValueError('Duplicate campaign attempt')
        for name in names:
            if not re.fullmatch(re.escape(phase) + r'_attempt_\d{3}', name):
                raise ValueError('Invalid campaign attempt name')
            row = dict(phase=phase, split=phase.rsplit('_', 1)[0], attempt=name,
                       controlled_collection_wall_seconds=None, recorded=None, expected=None,
                       collection=None, gate=None, integrity='unavailable')
            try:
                artifact = Artifact(campaign, name, require_hashes=False)
            except FileNotFoundError:
                rows.append(row)
                continue
            with artifact:
                inherited = inherited_attempt(inheritance, phase, name)
                if inherited:
                    verify_inherited_archive(artifact, inherited)
                if artifact.exists('cost_ledger.json'):
                    ledger = artifact.read('cost_ledger.json')
                    sources = inherited['source_hashes'] if inherited else state['source_hashes']
                    if ledger['phase'] != phase or ledger['source_hashes'] != sources:
                        raise ValueError('Cost ledger source or phase mismatch')
                    if (type(ledger['recorded']) is not int or type(ledger['expected']) is not int
                            or not 0 <= ledger['recorded'] <= ledger['expected']):
                        raise ValueError('Invalid cost ledger measurement counts')
                    row.update(controlled_collection_wall_seconds=_seconds(ledger['collection_wall_seconds']),
                               recorded=ledger['recorded'], expected=ledger['expected'],
                               ledger_scope=ledger['scope'],
                               integrity='checked' if 'cost_ledger.json' in artifact.hashes else 'unverified')
                if artifact.exists('summary.json'):
                    summary = artifact.read('summary.json')
                    row.update(collection=summary.get('collection'), gate=summary.get('gate'))
            rows.append(row)
    def total(selected):
        known = [row for row in selected if row['controlled_collection_wall_seconds'] is not None]
        missing = [row['attempt'] for row in selected if row['controlled_collection_wall_seconds'] is None]
        unchecked = [row['attempt'] for row in known if row['integrity'] != 'checked']
        observed = sum(row['controlled_collection_wall_seconds'] for row in known)
        return dict(attempts=len(selected), observed_controlled_wall_seconds=observed,
                    verified_complete_attempt_sum_seconds=observed if not missing and not unchecked else None,
                    missing_ledgers=missing, unverified_ledgers=unchecked,
                    recorded_measurements=sum(row['recorded'] for row in known),
                    scope='All attempts, including incomplete attempts; shared setup already included')
    return dict(scope='Controlled collection, clock setup/reset and analysis; never raw timing labels',
                excluded_costs=['checked-wrapper setup', 'archive packaging', 'campaign idle gaps'],
                campaign_elapsed_wall_seconds=None,
                missing_phase_attempts=[p for p in phases if not state['attempts'].get(p)],
                by_split={split: total([r for r in rows if r['split'] == split])
                          for split in ('train', 'validation', 'test')},
                all_attempts=total(rows), attempt_details=rows)


def _evidenced_cost(value):
    if value is None:
        return None
    if not isinstance(value.get('evidence'), str) or not value['evidence'].strip():
        raise ValueError('A measured cost requires an evidence reference')
    if not isinstance(value.get('scope'), str) or not value['scope'].strip():
        raise ValueError('A measured cost requires a comparable resource/accounting scope')
    return dict(seconds=_seconds(value['seconds']), evidence=value['evidence'], scope=value['scope'])


def break_even(common_source, direct_per_task, reuse_per_task):
    """Conditional task-count arithmetic for three comparable, evidenced costs."""
    source, direct, reuse = map(_evidenced_cost, (common_source, direct_per_task, reuse_per_task))
    result = dict(status='UNKNOWN', common_source=source, direct_per_task=direct, reuse_per_task=reuse,
                  first_nonloss_task_count=None, first_saving_task_count=None,
                  assumption='Each future task has the same measured direct/reuse preparation cost and comparable accuracy')
    if any(value is None for value in (source, direct, reuse)):
        result['reason'] = 'Missing measured common source or comparable per-task preparation costs'
        return result
    if len({value['scope'] for value in (source, direct, reuse)}) != 1:
        raise ValueError('Cannot combine different resource/accounting scopes')
    saving = direct['seconds'] - reuse['seconds']
    if saving <= 0:
        result.update(status='NO_BREAK_EVEN', reason='Reuse preparation has no per-task saving')
        return result
    ratio = source['seconds'] / saving
    result.update(status='CONDITIONAL', first_nonloss_task_count=max(1, math.ceil(ratio)),
                  first_saving_task_count=max(1, math.floor(ratio) + 1), per_task_saving_seconds=saving)
    return result


def fit_costs(timing):
    if timing is None:
        return dict(wall_seconds=None, model_fit_seconds=None, prediction_seconds=None)
    if isinstance(timing, list):
        return dict(wall_seconds=None,
                    model_fit_seconds=sum(_seconds(row['fit_seconds']) for row in timing),
                    prediction_seconds=sum(_seconds(row['predict_seconds']) for row in timing)
                    if all('predict_seconds' in row for row in timing) else None,
                    scope='Sum of recorded CPU operations; shared component fits must appear once')
    return dict(wall_seconds=_seconds(timing['wall_seconds']) if 'wall_seconds' in timing else None,
                model_fit_seconds=_seconds(timing['total_model_fit_seconds']) if 'total_model_fit_seconds' in timing else None,
                prediction_seconds=_seconds(timing['prediction_seconds']) if 'prediction_seconds' in timing else None,
                scope='Recorded CPU calls; serialization and other overhead remain in wall time; separate from GPU collection wall')


def build_report(campaign, selections=(), fit_timing=None, source_cost=None, protocol_access=()):
    return dict(collection=collection_costs(campaign),
                selected_recipe_label_budget=label_budget(selections),
                all_protocol_access_label_budget=label_budget(protocol_access),
                fitting=fit_costs(fit_timing), common_source_cost=_evidenced_cost(source_cost),
                break_even=break_even(source_cost, None, None),
                total_preparation_saving_seconds=None,
                interpretation='Selected recipe budgets and full search access are separate; bundled collection cost remains observed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    for name in ('selections', 'protocol-access', 'fit-timing', 'source-cost'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def load(path, default=None):
        return json.loads(path.read_text()) if path else default
    report = build_report(args.campaign, load(args.selections, []), load(args.fit_timing),
                          load(args.source_cost), load(args.protocol_access, []))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('PASS: cost report saved')


if __name__ == '__main__':
    main()
