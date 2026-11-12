#!/usr/bin/env python3
"""Read progress only, without touching GPU or raw measurements."""
import json
import os
from pathlib import Path
import sys
import time


def line(out,size='full'):
    path=out/'progress.json'
    value=json.loads(path.read_text()) if path.exists() else dict(recorded=0,expected=180000 if size=='full' else 48,state='PREPARING_OR_PREHEATING')
    done,total=value['recorded'],value['expected']
    return f"Progress {done}/{total} ({100*done/total:.1f}%) remaining={total-done} flagged={value.get('flagged',0)} state={value['state']}"


if __name__=='__main__':
    out,size=Path(sys.argv[1]),sys.argv[2]
    if len(sys.argv)==3:
        print(line(out,size),flush=True)
    else:
        parent=int(sys.argv[3])
        try:
            while os.getppid()==parent:
                try:
                    print(line(out,size),flush=True)
                except (OSError,ValueError,KeyError,ZeroDivisionError) as error:
                    print(f'Progress temporarily unavailable: {error}',flush=True)
                time.sleep(20)
        except (BrokenPipeError,KeyboardInterrupt):
            pass
