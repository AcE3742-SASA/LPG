"""Audit returned archive bytes and collection metadata without opening test labels."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'layer_full_20261110_v5'
INPUT = Path('/Users/studyhard/Desktop/Transfer/resume_full_20261108_105717_8tqhrkir')
sys.path.insert(0, str(PACKAGE))
import compare
import inherit_campaign


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def audit():
    state = compare.read(INPUT / 'campaign.json')
    release = compare.read(PACKAGE / 'release_hashes.json')
    assert state['source_hashes'] == release
    assert all(compare.sha(PACKAGE / name) == value for name, value in release.items())
    inherit_campaign.validate_state(INPUT, state)
    attempts = {attempt for values in state['attempts'].values() for attempt in values}
    assert len(attempts) == sum(map(len, state['attempts'].values())) == 33
    archives = sorted(INPUT.glob('*.zip'))
    assert len(archives) == len(attempts)
    cases = compare.read(PACKAGE / 'cases_by_phase.json')
    report = dict(input_path=str(INPUT), campaign_sha256=compare.sha(INPUT / 'campaign.json'),
                  campaign_completed_batches=state['completed_batches'],
                  inheritance_validation_passed=True, current_release_hashes_verified=len(release),
                  test_policy='Hash opaque bytes; inspect collection metadata and record filenames only.',
                  archives=[])
    seen = set()
    for path in archives:
        assert not path.is_symlink()
        sidecar = path.with_suffix('.zip.sha256').read_text().split()
        archive_sha = compare.sha(path)
        assert sidecar == [archive_sha, path.name]
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            assert len(names) == len(set(names))
            attempt = names[0].split('/')[0]
            assert attempt in attempts and attempt not in seen
            seen.add(attempt)
            prefix = attempt + '/'
            for info in z.infolist():
                p = PurePosixPath(info.filename)
                assert not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename
                assert p.parts[0] == attempt and (info.external_attr >> 16) & 0o170000 != 0o120000
            hashes = json.loads(z.read(prefix + 'result_hashes.json'))
            assert set(names) == {prefix + n for n in hashes} | {prefix + 'result_hashes.json'}
            # Reading all bytes verifies ZIP CRC as well as SHA-256 without decoding labels.
            for name, expected in hashes.items():
                assert digest(z.read(prefix + name)) == expected, (path.name, name)
            manifest = json.loads(z.read(prefix + 'manifest.json'))
            summary = json.loads(z.read(prefix + 'summary.json'))
            phase = manifest['phase']
            assert attempt in state['attempts'][phase]
            assert manifest['cases'] == cases[phase]
            source = ROOT / manifest['kind']
            sources = compare.read(source / 'release_hashes.json')
            assert manifest['source_hashes'] == sources
            assert json.loads(z.read(prefix + 'source/release_hashes.json')) == sources
            for name, expected in sources.items():
                assert hashes['source/' + name] == expected
                assert compare.sha(source / name) == expected
            records = {n for n in hashes if n.startswith(('records/', 'restricted_explicit/records/'))}
            exclusions = manifest.get('correctness_exclusions', [])
            protocol = manifest['protocol']
            expected_records = {('records/' if impl == 'standard' else 'restricted_explicit/records/')
                                + compare.key(case, dtype, rd, impl) + '.json'
                                for case in cases[phase] for dtype in protocol['dtypes']
                                if not any((e['case_id'], e['dtype']) == (case['id'], dtype) for e in exclusions)
                                for rd in range(protocol['rounds'])
                                for impl in compare.implementations(case, protocol)}
            assert records <= expected_records
            assert len(records) == summary['recorded']
            assert len(expected_records) == manifest['expected_measurements'] == summary['expected']
            assert len(expected_records - records) == summary['missing']
            if summary['collection'] == 'COMPLETE':
                assert records == expected_records
                assert summary['gate'] in ('PASS', 'REVIEW') and summary['worker_complete']
                assert summary['restoration'] == 'COMMANDS_ACCEPTED'
            expanded = INPUT / attempt
            directory_hashes = 0
            extra_files = []
            if expanded.exists():
                assert expanded.is_dir() and not expanded.is_symlink()
                for name, expected in hashes.items():
                    target = expanded / name
                    assert not target.is_symlink() and compare.sha(target) == expected
                    directory_hashes += 1
                assert (expanded / 'result_hashes.json').read_bytes() == z.read(prefix + 'result_hashes.json')
                receipt = compare.read(expanded / 'archive_receipt.json')
                assert receipt['filename'] == path.name and receipt['sha256'] == archive_sha
                local_files = {str(p.relative_to(expanded)) for p in expanded.rglob('*') if p.is_file()}
                extra_files = sorted(local_files - set(hashes) - {'result_hashes.json'})
                assert extra_files == ['.run.lock', 'archive_receipt.json']
                assert (expanded / '.run.lock').read_bytes() == b''
            report['archives'].append(dict(filename=path.name, sha256=archive_sha,
                phase=phase, attempt=attempt, source_kind=manifest['kind'], zip_crc_passed=True,
                result_hashes_verified=len(hashes), source_hashes_verified=len(sources),
                local_source_match=True, expanded_directory_hashes_verified=directory_hashes,
                expanded_directory_additional_files=extra_files,
                record_inventory_matches=True, sealed_test=manifest['sealed_test'],
                summary={k: summary[k] for k in ('collection','gate','recorded','expected','missing','counts',
                           'worker_complete','restoration')},
                excluded_measurements=1080-len(expected_records)))
    assert seen == attempts
    complete = [a for a in report['archives'] if a['summary']['collection'] == 'COMPLETE']
    assert len(complete) == state['completed_batches'] == 30
    assert sum(a['summary']['recorded'] for a in complete) == 32364
    assert sum(a['excluded_measurements'] for a in complete) == 36
    report['totals'] = dict(archives=len(archives), complete_archives=len(complete),
        incomplete_archives=len(archives)-len(complete),
        result_hashes_verified=sum(a['result_hashes_verified'] for a in report['archives']),
        source_hashes_verified=sum(a['source_hashes_verified'] for a in report['archives']),
        expanded_directory_hashes_verified=sum(a['expanded_directory_hashes_verified'] for a in report['archives']),
        complete_records=sum(a['summary']['recorded'] for a in complete),
        excluded_measurements=sum(a['excluded_measurements'] for a in complete))
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    audit()
