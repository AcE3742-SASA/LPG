"""Read-only nvidia-smi sampling; a polling interval is not sensor resolution."""
import contextlib
import csv
import datetime as dt
import math
import statistics
import subprocess
import time

COUNTERS = tuple('clocks_event_reasons_counters.' + s for s in
                 ('sw_power_cap', 'sw_thermal_slowdown', 'hw_thermal_slowdown'))
REASONS = tuple('clocks_event_reasons.' + s for s in
               ('sw_power_cap', 'sw_thermal_slowdown', 'hw_thermal_slowdown'))
FIELDS = ('timestamp', 'uuid', 'pstate', 'temperature.gpu', 'utilization.gpu',
          'clocks.sm', 'clocks.mem', 'power.draw', 'power.limit',
          'clocks_event_reasons.active', *REASONS, *COUNTERS)


def parse_row(values, offset_seconds):
    if len(values) != len(FIELDS):
        raise ValueError('Unexpected telemetry CSV row: ' + str(values))
    row = {k: v.strip() for k, v in zip(FIELDS, values)}
    local = dt.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
    row['utc'] = local.replace(tzinfo=dt.timezone(dt.timedelta(seconds=offset_seconds))).astimezone(dt.timezone.utc).isoformat()
    for key in FIELDS[3:]:
        value = row[key]
        if value in ('N/A', '[N/A]', 'Not Supported', '[Not Supported]'):
            row[key] = None
        elif key in REASONS:
            if value not in ('Active', 'Not Active'):
                raise ValueError('Invalid clock reason: ' + value)
            row[key] = value == 'Active'
        elif key == 'clocks_event_reasons.active' or key in COUNTERS:
            row[key] = int(value, 0) if value.lower().startswith('0x') else int(value)
            if row[key] < 0:
                raise ValueError('Negative clock counter')
        else:
            row[key] = float(value)
            if not math.isfinite(row[key]):
                raise ValueError('Non-finite telemetry value')
    for key in ('clocks.sm', 'clocks.mem', 'power.draw', 'power.limit'):
        if row[key] is None or row[key] <= 0:
            raise ValueError('Unavailable required telemetry field: ' + key)
    return row


def read(path, offset_seconds, interval_ms):
    with path.open() as f:
        rows = [parse_row(r, offset_seconds) for r in csv.reader(f) if r]
    if len(rows) < 2:
        raise ValueError('Too few telemetry samples')
    times = [dt.datetime.fromisoformat(r['utc']).timestamp() for r in rows]
    intervals = [b-a for a,b in zip(times,times[1:])]
    if any(x < 0 for x in intervals):
        raise ValueError('Telemetry timestamps are not monotonic')
    return rows, dict(samples=len(rows), requested_interval_ms=interval_ms,
                     observed_interval_seconds_median=statistics.median(intervals),
                     observed_interval_seconds_max=max(intervals),
                     unavailable_fields=[key for key in FIELDS if any(r[key] is None for r in rows)],
                     sensor_resolution_verified=False)


def window(rows, start, end):
    start, end = dt.datetime.fromisoformat(start), dt.datetime.fromisoformat(end)
    if end <= start:
        raise ValueError('Invalid telemetry window')
    inside, before, after = [], None, None
    for row in rows:
        at = dt.datetime.fromisoformat(row['utc'])
        if at < start:
            before = row
        elif at <= end:
            inside.append(row)
        elif after is None:
            after = row
    deltas = {}
    for key in COUNTERS:
        a, b = (inside[0][key], inside[-1][key]) if len(inside) >= 2 else (None, None)
        deltas[key] = b-a if a is not None and b is not None and b >= a else None
    span = ((dt.datetime.fromisoformat(inside[-1]['utc']) - dt.datetime.fromisoformat(inside[0]['utc'])).total_seconds()
            if len(inside) >= 2 else None)
    return dict(start_utc=start.isoformat(), end_utc=end.isoformat(), samples_inside=len(inside),
                inside=inside, before=before, after=after,
                sm_mhz_median=statistics.median(r['clocks.sm'] for r in inside) if inside else None,
                counter_interval_seconds=span, counter_deltas_us=deltas,
                counter_note='Deltas cover first-to-last INSIDE sample only, not the whole block.')


def command(gpu_uuid, interval_ms=None):
    args = ['nvidia-smi', '-i', gpu_uuid, '--query-gpu=' + ','.join(FIELDS),
            '--format=csv,noheader,nounits']
    if interval_ms is not None:
        args.append('--loop-ms=' + str(interval_ms))
    return args


@contextlib.contextmanager
def collect(out, gpu_uuid, interval_ms, suite):
    offset = int(dt.datetime.now().astimezone().utcoffset().total_seconds())
    probe = suite.command_text(command(gpu_uuid))
    suite.atomic_json(out/'telemetry_probe.json', probe)
    if probe.get('exit_code') != 0:
        raise RuntimeError('Clock telemetry query failed; inspect telemetry_probe.json. No fields were silently dropped.')
    parsed = list(csv.reader(probe['stdout'].splitlines()))
    if len(parsed) != 1 or parse_row(parsed[0], offset)['uuid'] != gpu_uuid:
        raise RuntimeError('Telemetry preflight did not return the target GPU')
    args = command(gpu_uuid, interval_ms)
    metadata = dict(fields=FIELDS, command=args, requested_interval_ms=interval_ms,
                    timezone_offset_seconds=offset, started_utc=suite.now(), sensor_resolution_verified=False)
    suite.atomic_json(out/'telemetry_metadata.json', metadata)
    with (out/'telemetry.csv').open('w') as log, (out/'telemetry.stderr.log').open('w') as err:
        process = subprocess.Popen(args, stdout=log, stderr=err)
        try:
            time.sleep(.2)
            if process.poll() is not None:
                raise RuntimeError('Telemetry process stopped during startup')
            yield process
            if process.poll() is not None:
                raise RuntimeError('Telemetry stopped before measurement finished')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            metadata.update(stopped_utc=suite.now(), process_exit_code=process.returncode)
            suite.atomic_json(out/'telemetry_metadata.json', metadata)
