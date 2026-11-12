#!/usr/bin/env python3
"""Correct zero-baseline accounting using existing validation aggregates only."""
import csv
from pathlib import Path

from run_rq14 import save_csv

OUT=Path(__file__).resolve().parent/'runs/v2'


def main():
    with (OUT/'selected_configurations_validation_only.csv').open(newline='') as handle:
        original=list(csv.DictReader(handle))
    with (OUT/'aggregate_metrics.csv').open(newline='') as handle:
        aggregates=list(csv.DictReader(handle))
    corrected=[]
    for old in original:
        if old['rq']=='RQ1':
            corrected.append(old)
            continue
        candidates=[r for r in aggregates if r['rq']=='RQ4' and r['dtype']==old['dtype'] and r['mode']==old['mode']]
        passing=[r for r in candidates if float(r['mean_mape_pct'])<=15 and float(r['mean_p90_pct'])<=30 and int(r['failures'])==0]
        chosen=min(passing,key=lambda r:(0 if r['method']=='zero' else int(r['n']),float(r['mean_mape_pct']),r['method'],int(r['n']))) if passing else min(candidates,key=lambda r:(float(r['mean_mape_pct']),float(r['mean_p90_pct']),r['method'],int(r['n'])))
        effective_k=0 if chosen['method']=='zero' else int(chosen['n'])
        row={**chosen,'goal_attained':str(bool(passing)),'effective_target_calibration_shapes':str(effective_k),'target_validation_shapes':'30','target_labels_per_mode_all_dtypes_three_rounds':str(6*(effective_k+30))}
        corrected.append(row)
    save_csv(OUT/'selected_configurations_cost_aware_validation_only.csv',corrected)
    changed=[(a,b) for a,b in zip(original,corrected) if a['rq']=='RQ4' and a['method']!=b['method']]
    assert len(changed)==2
    assert sum(r['method']=='zero' for r in corrected if r['rq']=='RQ4')==3
    (OUT/'selection_correction.md').write_text('''# RQ4 zero baseline 선택 비용 정정

원래 v2 runner는 무보정 source를 모든 k의 비교 기준선으로 반복 기록했다. k=5 행의 무보정이 실제로 소비하는 calibration 형상은 0개다. 기존 선택 코드는 곡선 행의 n으로 먼저 정렬해서, zero와 k=5 보정을 같은 비용으로 취급했다. 그 결과 FP32 prealloc/Graph에서 이미 목표를 만족하는 zero 대신 오차가 더 작은 constant_ratio k=5를 골랐다.

이번 파생 정정은 정해진 validation 목표와 저장된 aggregate 값만 사용한다. 선택 키를 (effective_k, MAPE, method, grid_n)으로 적용하며 zero의 effective_k는 0이다. 세 seed 평균 MAPE 15% 이하와 p90 30% 이하, 실패 0개라는 기준은 그대로다. 새 label, 새 target validation 형상이나 잠금 시험은 열지 않았다.

정정 후 FP16 prealloc, FP32 prealloc와 FP32 Graph는 calibration 0개, FP16 Graph는 constant_ratio 5개를 선택한다. FP32의 k=5 상수 보정은 오차를 낮추지만 이 목표에 도달하기 위한 필수 비용은 아니다. 최종 시험 확인은 남아 있다.

원래 runner, frozen protocol, selected_configurations_validation_only.csv와 selected_models/는 보존했다. 정정 선택은 selected_configurations_cost_aware_validation_only.csv, 그 가중치는 selected_models_cost_aware/에 별도로 둔다. 이것은 zero 기준선의 비용 대응 오류를 고친 것이며 새 알고리즘 탐색이 아니다.
''')
    print('PASS: cost-aware selection corrected using saved validation aggregates; no labels loaded')


if __name__=='__main__':
    main()
