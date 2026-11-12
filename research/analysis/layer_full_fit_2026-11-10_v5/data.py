"""Read frozen layer campaigns without extracting archives or crossing split gates."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path, PurePosixPath
import statistics
import zipfile

PACKAGE = Path(__file__).resolve().parents[2] / 'layer_full_20261110_v5'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe artifact path')
    return path


class Artifact:
    """A single attempt, read from its directory or its checked returned ZIP."""
    def __init__(self, campaign, attempt_name, require_hashes=True):
        if Path(attempt_name).name != attempt_name or not attempt_name:
            raise ValueError('Invalid attempt name')
        self.campaign = Path(campaign).resolve()
        self.root = self.campaign / attempt_name
        self.archive = None
        self.sha256 = None
        self.hashes = {}
        if self.root.is_symlink():
            raise ValueError('Symlink attempt denied')
        if self.root.is_dir():
            self.source = str(self.root)
        else:
            archives = sorted(self.campaign.glob(attempt_name + '_*.zip'))
            if not archives:
                raise FileNotFoundError('Missing attempt: ' + attempt_name)
            if len(archives) != 1:
                raise ValueError('Expected one returned archive for ' + attempt_name)
            path = archives[0]
            sidecar = path.with_suffix('.zip.sha256')
            fields = sidecar.read_text().strip().split()
            if len(fields) != 2 or fields[1].lstrip('*') != path.name or sha(path) != fields[0]:
                raise ValueError('Archive checksum mismatch')
            self.sha256 = fields[0]
            self.source = str(path)
            self.archive = zipfile.ZipFile(path)
            names = self.archive.namelist()
            if len(set(names)) != len(names):
                self.close()
                raise ValueError('Duplicate ZIP member')
            for item in self.archive.infolist():
                member = safe_name(item.filename)
                if member.parts[0] != attempt_name or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    self.close()
                    raise ValueError('Unsafe ZIP member')
        if self.exists('result_hashes.json'):
            self.hashes = self.read('result_hashes.json')
            for name in self.hashes:
                safe_name(name)
        elif require_hashes:
            self.close()
            raise ValueError('Missing result hash manifest')

    def exists(self, name):
        safe_name(name)
        if self.archive:
            return str(PurePosixPath(self.root.name) / name) in self.archive.namelist()
        path = self.root / name
        if path.is_symlink() or self.root not in path.resolve().parents:
            raise ValueError('Symlink or escaped artifact denied')
        return path.is_file()

    def bytes(self, name):
        safe_name(name)
        if self.archive:
            blob = self.archive.read(str(PurePosixPath(self.root.name) / name))
        else:
            if not self.exists(name):
                raise ValueError('Missing artifact: ' + name)
            blob = (self.root / name).read_bytes()
        if self.hashes and name != 'result_hashes.json':
            if self.hashes.get(name) != hashlib.sha256(blob).hexdigest():
                raise ValueError('Result hash mismatch: ' + name)
        return blob

    def read(self, name):
        return json.loads(self.bytes(name))

    def close(self):
        if self.archive:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def implementations(case):
    result = ['standard', 'decomposed']
    if case['layer'] == 'ffn':
        result += ['gemm1', 'gelu', 'gemm2']
    elif case['layer'].startswith('conv'):
        result += (['unfold'] if case['kernel'] == 3 else []) + ['input_pack', 'mm', 'output_pack']
    return result


def key(case, dtype, rd, implementation):
    return f'{case["id"]}_{dtype}_r{rd}_{implementation}'


def validate_inheritance(campaign, state):
    """Allow only the v2 results and v3 failure pinned in this release."""
    sources = read(PACKAGE / 'release_hashes.json')
    if state.get('source_hashes') != sources:
        raise ValueError('Campaign release mismatch')
    original = Path(campaign) / 'inherited_campaign.json'
    previous_path = Path(campaign) / 'previous_campaign.json'
    if (not any(name in state for name in ('inherited_batches', 'inherited_campaign_sha256', 'previous_campaign_sha256'))
            and not original.exists() and not previous_path.exists()):
        return None
    names = ('inheritance.json', 'inherited_v2_protocol.json', 'inherited_v2_release_hashes.json',
             'inherited_v3_protocol.json', 'inherited_v3_release_hashes.json', 'inherited_v3_campaign.json')
    if any(sources.get(name) != sha(PACKAGE / name) for name in names):
        raise ValueError('Inherited metadata differs from frozen release')
    inheritance = read(PACKAGE / 'inheritance.json')
    old_sources = read(PACKAGE / 'inherited_v2_release_hashes.json')
    old_protocol = read(PACKAGE / 'inherited_v2_protocol.json')
    if (inheritance.get('source_kind') != 'layer_full_20261108_v2'
            or old_protocol.get('version') != inheritance['source_kind']
            or inheritance.get('source_hashes') != old_sources
            or original.is_symlink() or not original.is_file()
            or sha(original) != inheritance.get('campaign_sha256')
            or state.get('inherited_campaign_sha256') != inheritance['campaign_sha256']):
        raise ValueError('Inherited campaign or release mismatch')
    previous = read(original)
    batches = inheritance['batches']
    complete = {phase: entry for phase, entry in batches.items() if entry['complete']}
    if (previous.get('source_hashes') != old_sources
            or previous.get('batch_order') != state.get('batch_order')
            or previous.get('attempts') != {phase: [entry['attempt']] for phase, entry in batches.items()}
            or previous.get('completed_batches') != len(complete)
            or state.get('inherited_batches') != complete):
        raise ValueError('Inherited batch metadata mismatch')
    for phase, entry in batches.items():
        attempts = state.get('attempts', {}).get(phase, [])
        if (not attempts or attempts[0] != entry['attempt']
                or entry['complete'] and attempts != [entry['attempt']]):
            raise ValueError('Inherited attempt history changed: ' + phase)
    prior_sources = read(PACKAGE / 'inherited_v3_release_hashes.json')
    prior_protocol = read(PACKAGE / 'inherited_v3_protocol.json')
    prior_state = read(PACKAGE / 'inherited_v3_campaign.json')
    if (prior_protocol.get('version') != 'layer_full_20261110_v3'
            or previous_path.is_symlink() or not previous_path.is_file()
            or sha(previous_path) != sources['inherited_v3_campaign.json']
            or state.get('previous_campaign_sha256') != sources['inherited_v3_campaign.json']
            or prior_state.get('source_hashes') != prior_sources
            or prior_state.get('batch_order') != state.get('batch_order')
            or prior_state.get('inherited_batches') != complete
            or prior_state.get('inherited_campaign_sha256') != inheritance['campaign_sha256']):
        raise ValueError('Previous v3 campaign provenance mismatch')
    expected_attempts = {phase: [entry['attempt']] for phase, entry in batches.items()}
    for entry in inheritance['prior_failures']:
        if entry.get('source_kind') != prior_protocol['version'] or entry.get('source_hashes') != prior_sources:
            raise ValueError('Inherited failed attempt source mismatch')
        expected_attempts.setdefault(entry['phase'], []).append(entry['attempt'])
    if prior_state.get('attempts') != expected_attempts:
        raise ValueError('Previous v3 campaign attempt history mismatch')
    for phase, attempts in expected_attempts.items():
        if state.get('attempts', {}).get(phase, [])[:len(attempts)] != attempts:
            raise ValueError('Inherited attempt history changed: ' + phase)
    if read(PACKAGE / 'protocol.json')['version'] == 'layer_full_20261110_v5':
        names = ('inherited_v4_protocol.json', 'inherited_v4_release_hashes.json', 'inherited_v4_campaign.json')
        if any(sources.get(name) != sha(PACKAGE / name) for name in names):
            raise ValueError('Inherited v4 metadata differs from frozen release')
        v4_sources = read(PACKAGE / names[1])
        v4_state = read(PACKAGE / names[2])
        v4_path = Path(campaign) / 'v4_campaign.json'
        if (v4_path.is_symlink() or not v4_path.is_file()
                or sha(v4_path) != sources[names[2]]
                or state.get('inherited_v4_campaign_sha256') != sources[names[2]]
                or v4_state.get('source_hashes') != v4_sources
                or v4_state.get('batch_order') != state['batch_order']
                or read(PACKAGE / names[0]).get('version') != 'layer_full_20261110_v4'
                or v4_state.get('inherited_batches') != complete
                or v4_state.get('inherited_campaign_sha256') != inheritance['campaign_sha256']
                or v4_state.get('previous_campaign_sha256') != sources['inherited_v3_campaign.json']):
            raise ValueError('Inherited v4 campaign provenance mismatch')
        for phase, attempts in v4_state['attempts'].items():
            if state['attempts'].get(phase, [])[:len(attempts)] != attempts:
                raise ValueError('Inherited v4 attempt history changed: ' + phase)
        completed = state.get('inherited_v4_batches', {})
        failures = state.get('inherited_v4_failures', [])
        wanted = {phase: v4_state['attempts'][phase][-1]
                  for phase in v4_state['batch_order'][:v4_state['completed_batches']]
                  if phase not in complete}
        if set(completed) != set(wanted) or len(failures) != 1:
            raise ValueError('Inherited v4 batch inventory mismatch')
        if any(state['attempts'].get(phase) != v4_state['attempts'][phase] for phase in completed):
            raise ValueError('Inherited v4 completed attempt history changed')
        entries = list(completed.items()) + [('test_04', failures[0])]
        for phase, entry in entries:
            if (entry.get('phase') != phase or entry.get('source_kind') != 'layer_full_20261110_v4'
                    or entry.get('source_hashes') != v4_sources
                    or entry.get('complete') is not (phase in completed)
                    or entry.get('attempt') != v4_state['attempts'][phase][-1]
                    or entry.get('attempt') != wanted.get(phase, 'test_04_attempt_001')
                    or Path(entry.get('filename', '')).name != entry.get('filename')
                    or not entry.get('filename', '').startswith(entry['attempt'] + '_')
                    or not entry['filename'].endswith('.zip')
                    or len(entry.get('sha256', '')) != 64
                    or any(c not in '0123456789abcdef' for c in entry['sha256'])):
                raise ValueError('Inherited v4 archive metadata mismatch')
            if phase == 'test_04' and (entry.get('phase') != phase
                    or entry['filename'] != 'test_04_attempt_001_c1bc4101.zip'
                    or entry['sha256'] != '047d2e27fb52c38978cc8ff620907370ad2a81b1e90e97713eb5a2c78e1cfc7f'):
                raise ValueError('Inherited v4 failure differs from pinned evidence')
        inheritance = dict(inheritance, v4_batches=completed, v4_failures=failures)
    return inheritance


def inherited_attempt(inheritance, phase, attempt):
    if inheritance is None:
        return None
    entry = inheritance['batches'].get(phase)
    if entry and attempt == entry['attempt']:
        return dict(entry, source_kind=inheritance['source_kind'], source_hashes=inheritance['source_hashes'])
    for entry in inheritance['prior_failures'] + inheritance.get('v4_failures', []):
        if (entry['phase'], entry['attempt']) == (phase, attempt):
            return dict(entry, complete=False)
    entry = inheritance.get('v4_batches', {}).get(phase)
    if entry and entry['attempt'] == attempt:
        return entry
    return None


def verify_inherited_archive(artifact, entry):
    if (artifact.archive is None or Path(artifact.source).name != entry['filename']
            or artifact.sha256 != entry['sha256']):
        raise ValueError('Inherited archive differs from pinned ZIP')


def correctness_exclusions(artifact, phase, cases, protocol, sources):
    if protocol['version'] not in ('layer_full_20261110_v4', 'layer_full_20261110_v5'):
        return []
    exclusions = protocol['correctness_exclusions']
    if len(exclusions) > (2 if protocol['version'].endswith('_v5') else 1):
        raise ValueError('Unexpected correctness exclusion policy')
    if len({(item.get('phase'), item.get('case_id'), item.get('dtype')) for item in exclusions}) != len(exclusions):
        raise ValueError('Duplicate correctness exclusion')
    for item in exclusions:
        allowed = [('validation_01', 'lf_ce6440e85a4add93'), ('test_04', 'lf_dfce839bd3ca0ef5')]
        if ((item.get('phase'), item.get('case_id')) not in allowed[:2 if protocol['version'].endswith('_v5') else 1]
                or item.get('dtype') != 'fp32'
                or item.get('reason') != 'standard_output_exceeds_original_fp64_tolerance'):
            raise ValueError('Unexpected correctness exclusion identity or reason')
        evidence = item['evidence']
        safe_name(evidence)
        if sources.get(evidence) != item.get('evidence_sha256'):
            raise ValueError('Correctness exclusion evidence hash mismatch')
        blob = artifact.bytes('source/' + evidence)
        if hashlib.sha256(blob).hexdigest() != item['evidence_sha256']:
            raise ValueError('Correctness exclusion evidence hash mismatch')
        report = json.loads(blob)
        adjudication = report.get('full_pair_adjudication', {})
        standard = adjudication.get('implementations', {}).get('standard', {})
        probe = item['phase'] == 'validation_01'
        if (report.get('passed') is not False or report.get('full_pair_passed') is not False
                or probe and (report.get('input_identity_matched') is not True
                    or report.get('case', {}).get('id') != item['case_id']
                    or report.get('case', {}).get('layer') != 'conv3'
                    or report.get('input_identity', {}).get('input', {}).get('dtype') != 'torch.float32')
                or protocol['tolerances']['fp32'] != [2e-5, 2e-4]
                or [report.get('atol'), report.get('rtol')] != protocol['tolerances']['fp32']
                or [adjudication.get('atol'), adjudication.get('rtol')] != protocol['tolerances']['fp32']
                or adjudication.get('passed') is not False or standard.get('passed') is not False
                or not _positive(standard.get('max_tolerance_ratio')) or standard['max_tolerance_ratio'] <= 1):
            raise ValueError('Correctness exclusion lacks an original-criterion failure')
        reference, observed = adjudication.get('reference', []), standard.get('observed', [])
        if (not reference or len(reference) != (2 if probe else 186) or len(reference) != len(observed)
                or not (adjudication.get('checked_count') == adjudication.get('mismatch_count') == len(reference))):
            raise ValueError('Correctness exclusion evidence is incomplete')
        atol, rtol = protocol['tolerances']['fp32']
        ratios = [abs(actual - expected) / (atol + rtol * abs(expected)) for actual, expected in zip(observed, reference)]
        if (not all(math.isfinite(value) for value in ratios) or max(ratios) <= 1
                or not math.isclose(max(ratios), standard['max_tolerance_ratio'], rel_tol=1e-10)):
            raise ValueError('Correctness exclusion does not reproduce the original-criterion failure')
        decomposed = adjudication.get('implementations', {}).get('decomposed', {})
        values = decomposed.get('observed', [])
        if (decomposed.get('passed') is not True or len(values) != len(reference)
                or not all(math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected)
                           for actual, expected in zip(values, reference))
                or not all(report['implementations'][name].get('finite_all') is True for name in ('standard', 'decomposed'))):
            raise ValueError('Correctness exclusion lacks an original-criterion failure')
        if not probe:
            identity_name = 'regression/' + item['case_id'] + '_fp32.json'
            identity = artifact.bytes('source/' + identity_name)
            if hashlib.sha256(identity).hexdigest() != sources.get(identity_name):
                raise ValueError('Correctness exclusion input identity hash mismatch')
            if json.loads(identity).get('input', {}).get('dtype') != 'torch.float32':
                raise ValueError('Correctness exclusion input dtype mismatch')
        if item['phase'] == phase and (not any(c['id'] == item['case_id'] and c['layer'] == 'conv3' for c in cases)
                or probe and report['case'] not in cases):
            raise ValueError('Correctness exclusion case differs from manifest')
    return [item for item in exclusions if item['phase'] == phase]


def check_freeze(protocol_path, freeze_path):
    if freeze_path is None:
        raise ValueError('Sealed test requires a frozen selection')
    path = Path(freeze_path).resolve()
    freeze = read(path)
    if (freeze.get('status') != 'LAYER_FULL_SELECTION_FROZEN'
            or freeze.get('protocol_sha256') != sha(protocol_path)):
        raise ValueError('Selection or protocol freeze mismatch')
    hashes = freeze.get('artifact_hashes', {})
    if not hashes or 'selected_configurations.json' not in hashes or not any(n.startswith('models/') for n in hashes):
        raise ValueError('Frozen artifact list is incomplete')
    for name, wanted in hashes.items():
        safe_name(name)
        artifact = path.parent / name
        if artifact.is_symlink() or path.parent not in artifact.resolve().parents or sha(artifact) != wanted:
            raise ValueError('Frozen artifact mismatch: ' + name)
    return freeze


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def check_timing(timing, protocol):
    """Recompute the frozen CUDA Event statistics and ordinary-call denominator."""
    blocks, setup = timing['blocks'], timing['setup']
    if len(blocks) != protocol['blocks'] or setup['settle_event_ms'] < protocol['settle_ms']:
        raise ValueError('Missing timing blocks or stabilization')
    if not math.isclose(sum(b['elapsed_ms'] for b in setup['settle_blocks']), setup['settle_event_ms'], rel_tol=1e-10):
        raise ValueError('Stabilization sum mismatch')
    for block in setup['settle_blocks'] + setup['calibration'] + blocks:
        if (type(block['repeats']) is not int or block['repeats'] < 1 or block['operations_per_call'] != 1
                or not _positive(block['elapsed_ms']) or not _positive(block['latency_ms'])
                or not math.isclose(block['elapsed_ms'] / block['repeats'], block['latency_ms'], rel_tol=1e-10)
                or datetime.fromisoformat(block['host_envelope_end_utc']) <= datetime.fromisoformat(block['host_envelope_start_utc'])):
            raise ValueError('Invalid event timing denominator or envelope')
    values = [b['latency_ms'] for b in blocks]
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    center = statistics.median(values)
    actual = dict(median_ms=center, relative_iqr=(q3-q1)/center,
                  first_last_drift=(values[-1]-values[0])/values[0], min_ms=min(values), max_ms=max(values))
    if any(not math.isclose(timing[name], value, rel_tol=1e-10, abs_tol=1e-12) for name, value in actual.items()):
        raise ValueError('Stored timing statistics mismatch')
    flags = []
    if actual['relative_iqr'] > protocol['relative_iqr_limit']:
        flags.append('relative_iqr')
    if abs(actual['first_last_drift']) > protocol['drift_limit']:
        flags.append('first_last_drift')
    if any(b['elapsed_ms'] < .8*protocol['block_ms'] for b in blocks):
        flags.append('short_block')
    if timing['flags'] != flags:
        raise ValueError('Stored timing flags mismatch')


def _load_batch(artifact, phase, cases, protocol, sources, diagnostic):
    # Check the time-free manifest before reading any record or restricted value.
    manifest = artifact.read('manifest.json')
    split = 'test_locked' if phase.startswith('test_') else phase.rsplit('_', 1)[0]
    exclusions = correctness_exclusions(artifact, phase, cases, protocol, sources)
    omitted = {(item['case_id'], item['dtype']) for item in exclusions}
    health_omitted = {case_id for case_id, _ in omitted} if protocol['version'].endswith('_v5') else set()
    config = dict(protocol, phase=phase, health_case_ids=[next(c['id'] for c in cases
                  if c['layer'] == layer and c['id'] not in health_omitted)
                  for layer in ('linear', 'ffn', 'conv1', 'conv3')])
    for case_id, dtype in omitted:
        prefixes = (f'inputs/{case_id}_{dtype}', f'records/{case_id}_{dtype}_',
                    f'restricted_explicit/records/{case_id}_{dtype}_')
        if any(name.startswith(prefixes) for name in artifact.hashes):
            raise ValueError('Excluded correctness condition has measured records or inputs')
    planned_count = sum(len(implementations(c)) * 6 for c in cases)
    expected_count = sum(len(implementations(c)) * 3 for c in cases for dtype in protocol['dtypes']
                         if (c['id'], dtype) not in omitted)
    if (manifest.get('kind') != protocol['version'] or manifest.get('phase') != phase
            or manifest.get('cases') != cases or manifest.get('protocol') != config
            or manifest.get('source_hashes') != sources or manifest.get('split') != split
            or manifest.get('sealed_test') is not phase.startswith('test_')
            or manifest.get('expected_measurements') != expected_count):
        raise ValueError('Batch manifest differs from frozen collection: ' + phase)
    if protocol['version'] in ('layer_full_20261110_v4', 'layer_full_20261110_v5') and (
            manifest.get('planned_measurements') != planned_count or manifest.get('correctness_exclusions') != exclusions):
        raise ValueError('Manifest correctness exclusions differ from frozen policy')
    for name, wanted in sources.items():
        if hashlib.sha256(artifact.bytes('source/' + name)).hexdigest() != wanted:
            raise ValueError('Measurement source mismatch: ' + name)
    summary = artifact.read('summary.json')
    state = artifact.read('status.json')
    complete = artifact.read('worker/complete.json')
    if (summary.get('scope') != phase or summary.get('collection') != 'COMPLETE'
            or summary.get('gate') not in ('PASS', 'REVIEW') or summary.get('comparisons')
            or summary.get('missing') != 0 or summary.get('recorded') != expected_count
            or summary.get('expected') != expected_count or summary.get('counts', {}).get('FAIL', 0)
            or state.get('collection') != 'COMPLETE' or state.get('restoration') != 'COMMANDS_ACCEPTED'
            or complete.get('complete') is not True or artifact.exists('failure.json')):
        raise ValueError('Batch collection or quality is incomplete: ' + phase)
    if protocol['version'] in ('layer_full_20261110_v4', 'layer_full_20261110_v5') and (
            summary.get('planned') != planned_count or summary.get('excluded') != planned_count - expected_count
            or summary.get('excluded_conditions') != exclusions or exclusions and summary['gate'] != 'REVIEW'):
        raise ValueError('Summary correctness exclusions differ from frozen policy')
    wanted_standard = {'records/' + key(c, d, r, 'standard') + '.json'
                       for c in cases for d in protocol['dtypes'] if (c['id'], d) not in omitted for r in range(3)}
    if {n for n in artifact.hashes if n.startswith('records/')} != wanted_standard:
        raise ValueError('Unexpected or missing standard record inventory')
    if diagnostic:
        wanted_extra = {'restricted_explicit/records/' + key(c, d, r, impl) + '.json'
                        for c in cases for d in protocol['dtypes'] if (c['id'], d) not in omitted for r in range(3)
                        for impl in implementations(c)[1:]}
        if {n for n in artifact.hashes if n.startswith('restricted_explicit/records/')} != wanted_extra:
            raise ValueError('Unexpected or missing component record inventory')
    rows, excluded = [], []
    for case in cases:
        for dtype in protocol['dtypes']:
            if (case['id'], dtype) in omitted:
                for implementation in implementations(case) if diagnostic else ['standard']:
                    excluded.append(dict(case_id=case['id'], group_id=case['group_id'], layer=case['layer'], dtype=dtype,
                                         split=split, case=case, implementation=implementation, time_ms=None,
                                         round_times_ms=[], status='FAIL', reasons=['correctness_exclusion'], record_costs=[]))
                continue
            identity_path = f'inputs/{case["id"]}_{dtype}.json'
            identity = artifact.read(identity_path)
            if not all(name in identity for name in ('input', 'weights', 'prepared', 'seed')):
                raise ValueError('Incomplete input identity')
            for implementation in implementations(case) if diagnostic else ['standard']:
                values, reasons, costs = [], set(), []
                for rd in range(3):
                    rid = key(case, dtype, rd, implementation)
                    directory = 'records' if implementation == 'standard' else 'restricted_explicit/records'
                    row = artifact.read(directory + '/' + rid + '.json')
                    if (row.get('id') != rid or row.get('case') != case or row.get('dtype') != dtype
                            or type(row.get('round')) is not int or row['round'] != rd
                            or row.get('implementation') != implementation or row.get('input_identity') != identity_path):
                        raise ValueError('Invalid measurement identity: ' + rid)
                    timing = row.get('timing', {})
                    value = timing.get('median_ms')
                    if not _positive(value):
                        raise ValueError('Invalid measurement time: ' + rid)
                    check_timing(timing, protocol['timing'])
                    flags = summary.get('measurement_flags', {})
                    if rid not in flags:
                        raise ValueError('Missing measurement quality: ' + rid)
                    reasons.update(flags[rid])
                    reasons.update(timing.get('flags', []))
                    for edge in ('before', 'after'):
                        expected = f'correctness/{case["id"]}_{dtype}_r{rd}_{edge}.json'
                        if row.get('correctness_' + edge) != expected:
                            raise ValueError('Unexpected correctness path')
                        if artifact.read(expected).get('passed') is not True:
                            reasons.add('correctness_failed')
                    if implementation not in ('standard', 'decomposed'):
                        expected = f'inputs/{case["id"]}_{dtype}_r{rd}_components.json'
                        if row.get('component_identity') != expected or not artifact.read(expected):
                            raise ValueError('Missing fixed-intermediate identity')
                    values.append(value)
                    costs.append(dict(id=rid, round=rd,
                                      timing_event_ms=sum(b['elapsed_ms'] for b in
                                          timing['setup']['settle_blocks'] + timing['setup']['calibration'] + timing['blocks']),
                                      preparation=row.get('preparation', {})))
                center = statistics.median(values)
                if (max(values) - min(values)) / center > protocol['timing']['between_round_limit']:
                    reasons.add('between_round_range')
                item = dict(case_id=case['id'], group_id=case['group_id'], layer=case['layer'], dtype=dtype,
                            split=split, case=case, implementation=implementation, time_ms=center,
                            round_times_ms=values, status='FAIL' if 'correctness_failed' in reasons else 'REVIEW' if reasons else 'PASS',
                            reasons=sorted(reasons), record_costs=costs)
                (excluded if reasons else rows).append(item)
    ledger = artifact.read('cost_ledger.json')
    if (ledger.get('phase') != phase or ledger.get('source_hashes') != sources
            or ledger.get('recorded') != expected_count or ledger.get('expected') != expected_count
            or not _positive(ledger.get('collection_wall_seconds'))):
        raise ValueError('Invalid collection cost ledger')
    return rows, excluded, ledger


def load_campaign(path, split='dev', *, allow_test=False, diagnostic=False, protocol_path=None, freeze_path=None):
    """Return PASS rows and explicit exclusions; opening test requires frozen artifacts."""
    if split not in ('dev', 'train', 'validation', 'test'):
        raise ValueError('Unknown analysis split')
    if protocol_path is None:
        raise ValueError('A pre-label frozen protocol is required')
    frozen = read(protocol_path)
    sources = read(PACKAGE / 'release_hashes.json')
    if frozen.get('status') != 'LAYER_FULL_PROTOCOL_FROZEN' or frozen.get('package_release_hashes') != sources:
        raise ValueError('Analysis protocol is not frozen against this release')
    if split == 'test':
        if not allow_test:
            raise ValueError('Sealed test requires explicit authorization')
        check_freeze(protocol_path, freeze_path)
    protocol = read(PACKAGE / 'protocol.json')
    cases_by_phase = read(PACKAGE / 'cases_by_phase.json')
    for name, wanted in sources.items():
        safe_name(name)
        if sha(PACKAGE / name) != wanted:
            raise ValueError('Local collection release changed: ' + name)
    order = [p for prefix in ('train_', 'validation_', 'test_') for p in sorted(cases_by_phase) if p.startswith(prefix)]
    prefixes = ('train_', 'validation_') if split == 'dev' else (split + '_',)
    phases = [p for p in order if p.startswith(prefixes)]
    campaign = Path(path).resolve()
    state = read(campaign / 'campaign.json')
    if state.get('source_hashes') != sources or state.get('batch_order') != order:
        raise ValueError('Campaign release or batch order mismatch')
    inheritance = validate_inheritance(campaign, state)
    rows, excluded, batch_costs, inputs = [], [], [], {}
    for phase in phases:
        attempts = state.get('attempts', {}).get(phase, [])
        if not attempts:
            raise ValueError('Missing required batch: ' + phase)
        attempt = attempts[-1]
        if not attempt.startswith(phase + '_attempt_') or Path(attempt).name != attempt:
            raise ValueError('Unexpected attempt identity')
        with Artifact(campaign, attempt) as artifact:
            inherited = inherited_attempt(inheritance, phase, attempt)
            if inherited:
                verify_inherited_archive(artifact, inherited)
                if not inherited['complete']:
                    raise ValueError('Inherited failed attempt requires a completed current-release retry: ' + phase)
                version = 'v2' if inherited['source_kind'] == 'layer_full_20261108_v2' else 'v4'
                batch_protocol = read(PACKAGE / ('inherited_' + version + '_protocol.json'))
                batch_sources = inherited['source_hashes']
            else:
                batch_protocol, batch_sources = protocol, sources
            accepted, held, ledger = _load_batch(artifact, phase, cases_by_phase[phase], batch_protocol, batch_sources, diagnostic)
            rows.extend(accepted)
            excluded.extend(held)
            batch_costs.append(dict(phase=phase, attempt=attempt, ledger=ledger))
            inputs[phase] = dict(attempt=attempt, source=artifact.source, archive_sha256=artifact.sha256,
                                 result_hashes=artifact.hashes, collection_version=batch_protocol['version'],
                                 inherited=inherited is not None)
    return dict(rows=rows, excluded=excluded, batch_costs=batch_costs,
                provenance=dict(campaign=str(campaign), campaign_sha256=sha(campaign/'campaign.json'),
                                protocol_sha256=sha(protocol_path), phases=phases, inputs=inputs,
                                test_opened=split == 'test', component_labels_opened=diagnostic))
