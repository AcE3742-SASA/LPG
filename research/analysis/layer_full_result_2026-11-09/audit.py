"""Read-only audit of the 2026-11-09 returned development archives."""
import hashlib
import importlib.util
import json
import csv
import io
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
import zipfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TRANSFER = Path('/Users/studyhard/Desktop/Transfer')
PACKAGE = ROOT / 'layer_full_20261108_v2'
spec = importlib.util.spec_from_file_location('frozen_layer_data', ROOT / 'analysis/layer_full_fit_2026-11-08_v2/data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def audit():
    campaign_blob = (TRANSFER / 'campaign.json').read_bytes()
    campaign = json.loads(campaign_blob)
    phases = [f'train_{n:02d}' for n in range(1, 21)] + ['validation_01']
    hash_prefixes = '8a42e28b 4f3b1a27 5b03561b 7a3f1389 218627bf 20dc5e05 a9f0a475 6ffa6a17 e83f445d 20ecd6f6 29824045 e6d0b7ca a4cceb8e 1f49f2a5 2e5a9ed3 3261254d c6981733 c3320935 4ffe550a 6ad931f3 61499c0f'.split()
    raw = Path(__file__).with_name('raw')
    raw.mkdir(exist_ok=True)
    cases = json.loads((PACKAGE / 'cases_by_phase.json').read_text())
    sources = campaign['source_hashes']
    assert sources == json.loads((PACKAGE / 'release_hashes.json').read_text())
    assert all(data.sha(PACKAGE / name) == value for name, value in sources.items())
    report = {'campaign_path': str(TRANSFER / 'campaign.json'), 'campaign_sha256': digest(campaign_blob),
              'campaign_completed_batches': campaign['completed_batches'], 'source_count': len(sources),
              'source_hashes_match_local_release': True, 'batches': []}
    all_ids = set()
    for phase, hash_prefix in zip(phases, hash_prefixes):
        attempt = phase + '_attempt_001'
        assert campaign['attempts'][phase] == [attempt]
        path = TRANSFER / (attempt + '_' + hash_prefix + '.zip')
        sha256 = data.sha(path)
        assert path.stem.rsplit('_', 1)[1] == hash_prefix
        prefix = attempt + '/'
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert len(set(names)) == len(names)
            for member in archive.infolist():
                name = data.safe_name(member.filename)
                assert name.parts[0] == attempt
                assert (member.external_attr >> 16) & 0o170000 != 0o120000
            # Reading every member below checks its ZIP CRC and its SHA-256 together.
            hashes = json.loads(archive.read(prefix + 'result_hashes.json'))
            actual_names = {n[len(prefix):] for n in names if not n.endswith('/')}
            assert actual_names == set(hashes) | {'result_hashes.json'}
            blobs = {}
            for name, expected_hash in hashes.items():
                data.safe_name(name)
                blob = archive.read(prefix + name)
                assert digest(blob) == expected_hash, (phase, name)
                if name.endswith('.json') and not name.startswith('source/'):
                    blobs[name] = blob
            def read(name):
                return json.loads(blobs[name])
            manifest = read('manifest.json')
            assert manifest['kind'] == 'layer_full_20261108_v2'
            assert manifest['phase'] == phase and not manifest['sealed_test']
            assert manifest['cases'] == cases[phase]
            assert manifest['source_hashes'] == sources
            for name, expected_hash in sources.items():
                assert digest(archive.read(prefix + 'source/' + name)) == expected_hash
            expected = {data.key(case, dtype, rd, impl) for case in manifest['cases']
                        for dtype in ('fp16', 'fp32') for rd in range(3)
                        for impl in data.implementations(case)}
            records = [read(name) for name in blobs if name.startswith(('records/', 'restricted_explicit/records/'))]
            record_ids = [row['id'] for row in records]
            assert len(record_ids) == len(set(record_ids))
            assert set(record_ids) <= expected and not (all_ids & set(record_ids))
            all_ids.update(record_ids)
            correctness = {name: read(name) for name in blobs if name.startswith('correctness/')}
            for row in records:
                assert row['id'] == data.key(row['case'], row['dtype'], row['round'], row['implementation'])
                assert row['case'] in manifest['cases']
                data.check_timing(row['timing'], manifest['protocol']['timing'])
                for edge in ('before', 'after'):
                    assert correctness[row['correctness_' + edge]]['passed'] is True
            summary = read('summary.json')
            ledger = read('cost_ledger.json')
            assert summary['recorded'] == len(records) == ledger['recorded']
            assert summary['expected'] == len(expected) == manifest['expected_measurements'] == ledger['expected']
            assert summary['missing'] == len(expected - set(record_ids))
            assert sum(summary['counts'].values()) == len(records)
            if phase.startswith('train'):
                assert set(record_ids) == expected and summary['collection'] == 'COMPLETE'
                assert summary['gate'] in ('PASS', 'REVIEW') and summary['counts'] == {'PASS': 1080}
            item = {'phase': phase, 'archive': str(path), 'sha256': sha256, 'crc_passed': True,
                    'filename_suffix_policy': 'random uuid4 prefix; independent of SHA256', 'archive_member_count': len(names), 'result_hashes_verified': len(hashes),
                    'embedded_source_hashes_verified': len(sources), 'manifest_matches_local_cases': True,
                    'summary': {k: summary[k] for k in ('gate','collection','expected','recorded','missing','counts','comparison_counts','restoration','execution_state','worker_complete')},
                    'component_round_review': summary.get('component_round_review', []),
                    'record_inventory_valid': True, 'record_timing_recomputed': len(records),
                    'record_correctness_links_passed': True,
                    'saved_correctness_documents': len(correctness),
                    'failed_correctness_documents': [name for name, value in correctness.items() if value['passed'] is not True],
                    'collection_wall_seconds': ledger['collection_wall_seconds'],
                    'environment_signature': manifest['environment_signature'],
                    'health_check': read('health_check.json'),
                    'status': read('status.json')}
            if 'ffn_reference_probe.json' in blobs:
                probe = read('ffn_reference_probe.json')
                item['ffn_reference_probe'] = {'passed': probe['passed'], 'case': probe['case']['id'],
                    'max_tolerance_ratio': {k: v['max_tolerance_ratio'] for k,v in probe['implementations'].items()},
                    'mathematical_fp64_passed': {k: v['mathematical_fp64_diagnostic']['passed'] for k,v in probe['implementations'].items()}}
            telemetry_fields = read('worker/telemetry_metadata.json')['fields']
            telemetry = list(csv.DictReader(io.StringIO(archive.read(prefix + 'worker/telemetry.csv').decode()), fieldnames=telemetry_fields, skipinitialspace=True))
            item['telemetry_rows'] = len(telemetry)
            item['telemetry_fields'] = telemetry_fields
            item['telemetry_ranges'] = {}
            for field in telemetry[0]:
                if any(word in field for word in ('clock', 'temperature', 'throttle')):
                    values = [r[field] for r in telemetry]
                    try:
                        numeric = [float(v) for v in values]
                        item['telemetry_ranges'][field] = [min(numeric), max(numeric)]
                    except ValueError:
                        item['telemetry_ranges'][field] = sorted(set(values))[:30]
            if phase.startswith('train'):
                artifact = SimpleNamespace(read=read, bytes=lambda n: archive.read(prefix + n), exists=lambda n: n in blobs, hashes=hashes)
                rows, excluded, _ = data._load_batch(artifact, phase, manifest['cases'], json.loads((PACKAGE / 'protocol.json').read_text()), sources, True)
                item['existing_loader_pass_conditions'] = len(rows)
                item['existing_loader_excluded'] = [{k:r[k] for k in ('case_id','layer','dtype','implementation','reasons','round_times_ms')} for r in excluded]
            report['batches'].append(item)
            copied = raw / path.name
            if copied.exists():
                assert data.sha(copied) == sha256
            else:
                shutil.copy2(path, copied)
                assert data.sha(copied) == sha256
            copied.with_suffix('.zip.sha256').write_text(sha256 + '  ' + copied.name + '\n')
            print(phase, 'records', len(records), 'hashes', len(hashes), 'PASS', summary['counts'].get('PASS', 0), flush=True)
    report['totals'] = {'archives': len(report['batches']), 'completed_train_batches': campaign['completed_batches'],
                        'recorded': len(all_ids), 'complete_train_records': sum(b['summary']['recorded'] for b in report['batches'] if b['phase'].startswith('train')),
                        'result_hashes_verified': sum(b['result_hashes_verified'] for b in report['batches']),
                        'collection_wall_seconds': sum(b['collection_wall_seconds'] for b in report['batches']),
                        'quality_counts': dict(sum((Counter(b['summary']['counts']) for b in report['batches']), Counter()))}
    assert report['totals']['archives'] == 21 and report['totals']['recorded'] == 21872
    assert report['totals']['complete_train_records'] == 21600
    copied_campaign = raw / 'campaign.json'
    if copied_campaign.exists():
        assert copied_campaign.read_bytes() == campaign_blob
    else:
        copied_campaign.write_bytes(campaign_blob)
    output = Path(__file__).with_name('audit.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    audit()
