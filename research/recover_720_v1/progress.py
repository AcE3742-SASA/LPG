#!/usr/bin/env python3
"""Display the existing worker's small progress file; never query the GPU or database."""
import json
import os
from pathlib import Path
import sys
import time


def line(out):
    path = out / 'progress.json'
    if path.exists():
        value = json.loads(path.read_text())
        state = '최근 저장 상태: ' + value['state']
    else:
        value = json.loads((out / 'recovery/initial_summary.json').read_text())
        state = '시작 준비 / 예열(180초)'
    done, total = value['recorded'], value['expected']
    return (f'[진행] {done:,}/{total:,} ({100 * done / total:.1f}%)'
            f' | 남음 {total - done:,} | 추가 수집 {max(0, done - 1680):,}/1,920 | {state}')


if __name__ == '__main__':
    out = Path(sys.argv[1])
    if len(sys.argv) == 2:
        print(line(out), flush=True)
        raise SystemExit(0)
    parent = int(sys.argv[2])
    try:
        while os.getppid() == parent:
            try:
                print(line(out), flush=True)
            except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
                print(f'[진행 표시] 최근 기록을 읽지 못함: {error}', flush=True)
            time.sleep(20)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
