"""Read frozen layer campaigns without extracting archives or crossing split gates."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path, PurePosixPath
import statistics
import zipfile

PACKAGE = Path(__file__).resolve().parents[2] / 'layer_full_20261104_v1'


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
    config = dict(protocol, phase=phase, health_case_ids=[next(c['id'] for c in cases if c['layer'] == layer)
                  for layer in ('linear', 'ffn', 'conv1', 'conv3')])
    split = 'test_locked' if phase.startswith('test_') else phase.rsplit('_', 1)[0]
    expected_count = sum(len(implementations(c)) * 6 for c in cases)
    if (manifest.get('kind') != 'layer_full_20261104_v1' or manifest.get('phase') != phase
            or manifest.get('cases') != cases or manifest.get('protocol') != config
            or manifest.get('source_hashes') != sources or manifest.get('split') != split
            or manifest.get('sealed_test') is not phase.startswith('test_')
            or manifest.get('expected_measurements') != expected_count):
        raise ValueError('Batch manifest differs from frozen collection: ' + phase)
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
    wanted_standard = {'records/' + key(c, d, r, 'standard') + '.json'
                       for c in cases for d in protocol['dtypes'] for r in range(3)}
    if {n for n in artifact.hashes if n.startswith('records/')} != wanted_standard:
        raise ValueError('Unexpected or missing standard record inventory')
    if diagnostic:
        wanted_extra = {'restricted_explicit/records/' + key(c, d, r, impl) + '.json'
                        for c in cases for d in protocol['dtypes'] for r in range(3)
                        for impl in implementations(c)[1:]}
        if {n for n in artifact.hashes if n.startswith('restricted_explicit/records/')} != wanted_extra:
            raise ValueError('Unexpected or missing component record inventory')
    rows, excluded = [], []
    for case in cases:
        for dtype in protocol['dtypes']:
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
    rows, excluded, batch_costs, inputs = [], [], [], {}
    for phase in phases:
        attempts = state.get('attempts', {}).get(phase, [])
        if not attempts:
            raise ValueError('Missing required batch: ' + phase)
        attempt = attempts[-1]
        if not attempt.startswith(phase + '_attempt_') or Path(attempt).name != attempt:
            raise ValueError('Unexpected attempt identity')
        with Artifact(campaign, attempt) as artifact:
            accepted, held, ledger = _load_batch(artifact, phase, cases_by_phase[phase], protocol, sources, diagnostic)
            rows.extend(accepted)
            excluded.extend(held)
            batch_costs.append(dict(phase=phase, attempt=attempt, ledger=ledger))
            inputs[phase] = dict(attempt=attempt, source=artifact.source, archive_sha256=artifact.sha256,
                                 result_hashes=artifact.hashes)
    return dict(rows=rows, excluded=excluded, batch_costs=batch_costs,
                provenance=dict(campaign=str(campaign), campaign_sha256=sha(campaign/'campaign.json'),
                                protocol_sha256=sha(protocol_path), phases=phases, inputs=inputs,
                                test_opened=split == 'test', component_labels_opened=diagnostic))
