#!/usr/bin/env python3
"""Summarize saved validation artifacts without opening any measurement source."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/v2'


def rows(name):
    with (OUT/name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def table(headers,data):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)


def fmt(value):
    return f'{float(value):.2f}'


def main():
    complete=json.loads((OUT/'execution_completed.json').read_text())
    exported=json.loads((OUT/'selected_model_export.json').read_text())
    corrected_export=json.loads((OUT/'selected_model_export_cost_aware.json').read_text())
    protocol=json.loads((OUT/'frozen_protocol.json').read_text())
    aggregates=rows('aggregate_metrics.csv');individual=rows('metrics.csv')
    selected=rows('selected_configurations_cost_aware_validation_only.csv');source=rows('source_selection.csv');cost=rows('target_label_costs.csv')
    colors={'rf_direct':'#4361ee','rf_normalized':'#d44842','zero':'#777777','constant_ratio':'#ca8a04','residual':'#d44842','direct':'#4361ee','prior_direct':'#15803d'}
    for metric,threshold in [('mape_pct',15),('p90_pct',30)]:
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
        for di,dtype in enumerate(['fp16','fp32']):
            for mi,mode in enumerate(['allocate','prealloc','graph']):
                ax=axes[di,mi]
                for method in ['rf_direct','rf_normalized']:
                    group=[r for r in aggregates if r['rq']=='RQ1' and r['dtype']==dtype and r['mode']==mode and r['method']==method]
                    group.sort(key=lambda r:int(r['n']))
                    x=[int(r['n']) for r in group];y=[float(r['mean_'+metric]) for r in group]
                    ax.plot(x,y,marker='o',label=method,color=colors[method])
                    minimum=[];maximum=[]
                    for n in x:
                        vals=[float(r[metric]) for r in individual if r['rq']=='RQ1' and r['dtype']==dtype and r['mode']==mode and r['method']==method and int(r['n'])==n]
                        minimum.append(min(vals));maximum.append(max(vals))
                    ax.fill_between(x,minimum,maximum,color=colors[method],alpha=.12)
                ax.axhline(threshold,color='black',linestyle=':',linewidth=1)
                ax.set_title(f'{dtype} / {mode}');ax.set_xscale('log');ax.grid(alpha=.2)
                ax.set_ylabel(metric.replace('_',' '));ax.set_xlabel('A training shapes')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('RQ1 validation only: mean over 3 seeds; shading = seed min/max')
        fig.tight_layout();fig.savefig(OUT/f'rq1_{metric}.png',dpi=180);fig.savefig(OUT/f'rq1_{metric}.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for di,dtype in enumerate(['fp16','fp32']):
        for mi,mode in enumerate(['prealloc','graph']):
            ax=axes[di,mi]
            for method in ['zero','constant_ratio','residual','direct','prior_direct']:
                group=[r for r in aggregates if r['rq']=='RQ4' and r['dtype']==dtype and r['mode']==mode and r['method']==method]
                group.sort(key=lambda r:int(r['n']))
                ax.plot([int(r['n']) for r in group],[float(r['mean_mape_pct']) for r in group],marker='o',label=method,color=colors[method])
            ax.axhline(15,color='black',linestyle=':',linewidth=1)
            ax.set_title(f'{dtype}: allocate -> {mode}');ax.set_xscale('log');ax.grid(alpha=.2)
            ax.set_xlabel('Target calibration k (+30 validation shapes)');ax.set_ylabel('MAPE (%)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('RQ4 validation only: fixed 30 target shapes; zero uses no calibration labels')
    fig.tight_layout();fig.savefig(OUT/'rq4_mape_pct.png',dpi=180);fig.savefig(OUT/'rq4_mape_pct.pdf');plt.close(fig)
    rq1full=[]
    for dtype in ['fp16','fp32']:
        for mode in ['allocate','prealloc','graph']:
            group={r['method']:r for r in aggregates if r['rq']=='RQ1' and r['dtype']==dtype and r['mode']==mode and int(r['n'])==2652}
            rq1full.append([dtype,mode,fmt(group['rf_direct']['mean_mape_pct']),fmt(group['rf_normalized']['mean_mape_pct']),fmt(group['rf_direct']['mean_p90_pct']),fmt(group['rf_normalized']['mean_p90_pct'])])
    chosen=[[r['rq'],r['dtype'],r['mode'],r['method'],r['n'],r['effective_target_calibration_shapes'],fmt(r['mean_mape_pct']),fmt(r['mean_p90_pct']),r['goal_attained']] for r in selected]
    source_table=[[r['dtype'],r['method'],fmt(r['mean_validation_mape_pct']),fmt(r['mean_validation_p90_pct'])] for r in source]
    scope=[r for r in individual if r['rq']=='RQ1'];rq4scope=[r for r in individual if r['rq']=='RQ4']
    failures=sum(int(r['failures']) for r in individual)
    pool_union=cost[-1]['union_calibration_shapes']
    text='''# RQ1과 RQ4 기존 자료 실행 결과

2026-10-28 실행계획에 따라 CPU 학습과 validation 평가를 수행했다. 이 결과는 모델과 보정 예산을 정하기 위한 개발 결과다. 기존 fresh holdout의 시간, 원시 DB와 LOCKED timing 파일은 열지 않았다. 최종 시험을 여는 명령도 구현하지 않았다.

## 실행 범위와 접근 순서

A 학습 풀은 두 dtype과 세 mode가 모두 완비된 2,652형상이고, C 보정 후보는 693형상이다. RQ1 validation은 716형상이다. 두 dtype과 세 mode의 반복은 같은 형상 그룹으로 묶었다. 세 회차의 중앙값을 독립 학습 표본으로 늘리지 않았다.

먼저 시간과 무관한 hash로 세 seed의 중첩 부분집합과 RQ4 target validation 30형상을 정했다. 입력 파일과 코드 SHA-256, 특징, 회귀기와 선택 규칙을 `runs/v2/frozen_protocol.json`에 저장했다. 이후 allocate의 학습과 validation만 열어 source 모델을 선택하고 저장했다. RQ4에는 target validation 30형상만 제공했다. RQ4 평가를 마친 뒤 RQ1의 prealloc/Graph 전체 validation을 평가했다.

`runs/v1`은 target label을 열기 전 만든 미실행 초안이다. C 보정 접근을 실제 소비 합집합으로 좁히고 source 추론 비용을 분리한 `runs/v2`가 실제 실행본이다. 실행 중 핵심 코드와 고정 설정을 바꾸지 않았다.

주 회귀기는 RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=2, n_jobs=1)다. 두 목표 모두 log M/N/K, log FLOPs, log 논리적 byte 수, log L, 세 종횡비, 각 축 8/16 정렬 여부를 받았다. 정규화 모델에 새로운 실측 입력을 추가하지 않았다. L은 고정 FP16 35.02 TFLOP/s, FP32 6.14 TFLOP/s와 262.3 GB/s로 계산했다. 엄밀한 물리적 시간 하한으로 해석하지 않는다.

## RQ1: 전체 A 자료를 사용한 비교

아래 수치는 세 seed의 지표를 산술평균한 값이다. MAPE와 p90은 각각 별도로 평균했다. 잠금 시험 결과가 아니다.

'''
    text+=table(['dtype','mode','직접 MAPE (%)','정규화 MAPE (%)','직접 p90 (%)','정규화 p90 (%)'],rq1full)
    text+='\n'+'''RF 직접 목표와 정규화 목표의 paired bootstrap 1,000회 결과는 `runs/v2/paired_bootstrap.csv`에 seed별로 남겼다. 음수 차이는 정규화 쪽의 오차가 작다는 뜻이다. Ridge(alpha=1), 상수 효율과 nonnegative launch+scale 기준선은 `aggregate_metrics.csv`에서 같은 자료량으로 비교할 수 있다. Ridge의 표준화 fit도 각 train 부분집합 안에서만 수행했다.

전체 A 자료에서는 FP16의 직접 목표가 더 작고 FP32는 정규화 목표가 조금 더 작았다. FP32의 100형상에서는 정규화 MAPE가 약 10.55~10.69%로 직접 목표의 약 12.99~13.17%보다 낮았다. 다만 두 목표 모두 100형상에서 제안한 MAPE/p90 기준을 통과했다. 현재 확인한 자료량 격자에서 정규화가 목표 도달에 필요한 최소 형상 수를 줄였다고 결론 내릴 수는 없다.

## RQ4: allocate source 선택

source는 각 dtype에서 A 2,652형상으로 학습한 두 RF 목표 중 allocate validation의 세 seed 평균 MAPE가 작은 쪽이다. target의 시간은 source 선택에 사용하지 않았다.

'''
    text+=table(['dtype','고정 source','allocate validation MAPE (%)','p90 (%)'],source_table)
    text+='\n'+'''각 source 모델의 가중치는 `runs/v2/models/`에 dtype과 seed별로 저장했다. 보정은 무보정 source, 상수 곱셈, log 잔차 RF, target 직접 RF와 source 예측값을 입력으로 받는 직접 RF를 비교했다. 직접+prior와 잔차 모델은 같은 입력을 받는다. RQ1의 전체 target 모델을 RQ4의 소량 적응 모델로 섞지 않았다.

## Validation에서 고른 구성

고정 선택 규칙은 세 seed 평균 MAPE≤15%, 평균 p90≤30%이고 모든 예측이 유효한 구성 중 최소 자료량이다. 같은 자료량에서는 MAPE가 작은 것을 골랐다. 기준에 도달하지 못한 층은 최소 MAPE 구성과 `False`를 기록했다. 이 선택은 최종 시험 통과를 뜻하지 않는다.

아래 표는 무보정의 실제 calibration 비용 0을 반영한 파생 정정본이다. 원래 runner는 zero를 각 k의 기준선으로 반복 기록한 뒤 grid n으로 먼저 정렬해 FP32 두 층에서 zero와 상수 보정 5형상을 같은 비용으로 취급했다. 원래 선택표와 가중치는 보존하고, 저장된 validation aggregate에 (effective_k, MAPE, method, grid_n) 순서를 적용해 정정했다. 새 label이나 시험 결과는 사용하지 않았다. 자세한 대응은 `runs/v2/selection_correction.md`에 남겼다.

'''
    text+=table(['RQ','dtype','mode','방법','곡선상 n/k','실제 target 보정 형상','MAPE (%)','p90 (%)','validation 목표 도달'],chosen)
    text+='\n'+f'''RQ1은 {len(scope)}개 구성(주 RF 144 fit, 보조 Ridge 144 fit, 상수/launch 기준선 144개)이고, RQ4는 {len(rq4scope)}개 구성(RF 180 fit, 상수 보정 60개, 무보정 60개)이다. 전체 {len(individual)}개 구성의 평가에서 유효하지 않은 예측은 {failures}개였다. 비유한 값이나 0 이하 예측은 삭제하거나 clipping하지 않았으며 실패가 있으면 목표 충족에서 탈락시켰다. 구간별 planned/valid/coverage와 오차는 `subgroup_metrics.csv`에 남겼다.

## 대상 자료 비용

RQ4의 validation 30형상도 비용에 포함했다. 한 mode에서 두 dtype과 세 회차를 모두 합친 레이블 수는 6(k+30)이다. 무보정은 k=0으로 180측정의 validation만 사용한다. 그림의 무보정 선은 k와 무관한 기준선이다.

'''
    text+=table(['각 seed의 k','각 seed의 전체 형상(k+30)','한 target 측정/seed','세 seed 보정 합집합 형상','두 target 전체 측정 합집합'],[[r['k'],r['per_seed_target_shape_count'],r['per_seed_labels_one_mode_two_dtypes_three_rounds'],r['union_calibration_shapes'],r['union_labels_both_modes_two_dtypes_three_rounds']] for r in cost])
    text+='\n'+f'''최대 예산을 세 seed에서 모두 비교하므로 실제 target calibration 접근 합집합은 {pool_union}형상이다. 693형상 전체의 target labels를 읽지 않았다. 30형상 validation을 더한 두 target의 실제 고유 측정 사용량은 {cost[-1]['union_labels_both_modes_two_dtypes_three_rounds']}개다. 여러 k가 중첩된다는 이유로 같은 레이블을 매번 새 측정으로 세지 않았다. 이 수는 기존 자료에서 소비한 레이블 수이며, 새 GPU 수집시간 절약의 실증값이 아니다.

CPU 모델 fit 시간 합계는 {complete['total_fit_seconds']:.2f}초이고 본 실행 wall time은 {complete['wall_seconds']:.2f}초다. 원래 선택 가중치 보존의 재현 fit은 별도로 {exported['export_refit_seconds']:.2f}초, export wall time은 {exported['wall_seconds']:.2f}초였다. 비용 정정 선택의 별도 export는 fit {corrected_export['export_refit_seconds']:.2f}초, wall {corrected_export['wall_seconds']:.2f}초였다. source prior 생성과 보정/직접 추론 시간을 `fit_times.csv`에 나누어 기록했다. pipeline_validation_predict_seconds는 source 추론과 보정 추론을 합친 시간이다. source 학습 비용은 RQ1 해당 fit 로그에 있고 target별로 중복 합산하지 않는다. 같은 source 예측을 여러 k/방법이 재사용하므로 로그의 개별 구성 준비 비용을 연구 전체 wall time으로 더해서는 안 된다. 옛 GPU 수집 wall time은 이번에 복원하지 않았고 UNKNOWN으로 남겼다.

## 해석 범위와 남은 작업

정규화의 이점은 같은 특징과 고정 학습기에서 확인한 목표 표현 방식의 효과다. 이것만으로 새로운 Roofline 학습 원리나 NeuSight 전체 재현을 주장하지 않는다. mode 적응은 같은 GPU의 기존 adaptive Graph 측정 정책 아래의 결과다. q, probe 시간과 kernel ID는 입력에 넣지 않았다.

기존 train/validation의 인접 형상 때문에 이 validation으로 넓은 형상 외삽을 입증할 수 없다. target validation은 30형상이어서 작은 구간의 오차와 최소 k 선택은 불확실할 수 있다. 저장된 bootstrap은 seed별 형상 단위의 조건부 구간이며, 형상 풀/세션 자체의 변화나 서로 다른 seed의 변동을 모두 포함하는 구간은 아니다. 전체 716형상을 이미 RQ1 개발 평가에 썼으므로 이를 새로운 독립 시험으로 재명명하지 않는다.

기존 boundary/controls 파일의 시간은 이번 실행에 추가로 열지 않았다. 실패 구간은 정적 FLOPs와 종횡비, 정렬 기준으로 validation 안에서 집계했다. RQ2/RQ3 레이어 측정, 최종 GEMM 시험과 최종 보고서용 확정 결론은 남아 있다.

## 재현 파일

`run_rq14.py`의 라벨 입력은 단 하나의 unlocked 조건 CSV로 고정했다. loader는 main의 legacy_train과 legacy_validation만 허용하고 호출마다 형상과 mode allowlist를 확인한다. `label_access_log.json`에는 사용 조건 수와 허용 형상 hash가 남아 있다. 입력 SHA 전후 일치 여부는 `execution_completed.json`에 기록했다. 코드, synthetic test와 패키지 버전은 `runs/v2/source/`에 보존했다.

핵심 표는 `metrics.csv`, seed 평균은 `aggregate_metrics.csv`, 개별 validation 예측은 `validation_predictions.csv`, 현재 채택 선택은 `selected_configurations_cost_aware_validation_only.csv`, 비용은 `target_label_costs.csv`다. 그림은 `rq1_mape_pct.png`, `rq1_p90_pct.png`, `rq4_mape_pct.png`와 같은 이름의 PDF다.

원래 선택 구성은 `selected_models/`, 비용 정정 선택은 `selected_models_cost_aware/`에 각각 seed별 30개 envelope로 보존했다. 가중치, 특징 순서, dtype, method, seed, n, train 형상, 고정 protocol hash와 source 참조를 함께 담았다. `export_selected.py`는 고정된 선택표와 허용된 train만 읽어 재학습했으며 새 validation/test 레이블은 열지 않았다. 두 묶음 모두 30개 전부 기존 validation 예측과 원소별 완전 일치했고 joblib 직렬화 후에도 일치했다. 정정본의 검산은 `selected_model_verification_cost_aware.csv`, 별도 소요시간은 `selected_model_export_cost_aware.json`에 남겼다. joblib은 이 실행에서 직접 만든 파일만 읽는다.
'''
    (HERE/'RQ1_RQ4_실행결과_2026-10-28.md').write_text(text)
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    manifest['summarize_rq14.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['export_selected.py']=hashlib.sha256((HERE/'export_selected.py').read_bytes()).hexdigest()
    manifest['correct_selection.py']=hashlib.sha256((HERE/'correct_selection.py').read_bytes()).hexdigest()
    manifest['RQ1_RQ4_실행결과_2026-10-28.md']=hashlib.sha256((HERE/'RQ1_RQ4_실행결과_2026-10-28.md').read_bytes()).hexdigest()
    (OUT/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('PASS: validation report, plots and artifact hashes saved; no measurement source opened')


if __name__=='__main__':
    main()
