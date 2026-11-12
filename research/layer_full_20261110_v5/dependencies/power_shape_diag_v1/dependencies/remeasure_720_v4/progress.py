#!/usr/bin/env python3
"""Display the existing worker's small progress file; never query the GPU or database."""
import json
import os
from pathlib import Path
import sys
import time


def line(out, size="full"):
    path = out / 'progress.json'
    if path.exists():
        value = json.loads(path.read_text())
        state = 'Last saved state: ' + value['state']
    else:
        value = dict(recorded=0, expected=113382 if size == 'full' else 48)
        state = 'Preparing / warmup (180s)'
    done, total = value['recorded'], value['expected']
    receipt = out / 'recovery_provenance.json'
    redo = ''
    if receipt.exists():
        recovery = json.loads(receipt.read_text())
        redo = (f"[Remeasure] {max(0, done - recovery['imported_records']):,}"
                f"/{recovery['redo_records']:,} | ")
    return (redo + f'[Total] {done:,}/{total:,} ({100 * done / total:.1f}%)'
            f" | Remaining {total - done:,} | Review {value.get('flagged', 0):,} (finalized segments) | {state}")


if __name__ == '__main__':
    out, size = Path(sys.argv[1]), sys.argv[2]
    if len(sys.argv) == 3:
        print(line(out, size), flush=True)
        raise SystemExit(0)
    parent = int(sys.argv[3])
    try:
        while os.getppid() == parent:
            try:
                print(line(out, size), flush=True)
            except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
                print(f'[Progress] Cannot read latest record: {error}', flush=True)
            time.sleep(20)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
