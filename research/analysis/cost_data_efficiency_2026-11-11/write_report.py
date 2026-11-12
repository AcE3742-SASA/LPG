#!/usr/bin/env python3
"""Render the Korean report from the computed CSV/JSON results."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / 'tables' / (name + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def md(headers, values):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in values])


def number(value, digits=0):
    return f'{float(value):,.{digits}f}'


def main():
    s = json.loads((HERE / 'summary.json').read_text())
    equal = rows('layer_equivalent_accuracy')
    portfolios = rows('layer_portfolios')
    def portfolio(target, method, goal='direct_mape'):
        return next(r for r in portfolios if (r['target'], r['method_family'], r['goal']) == (target, method, goal))
    direct = portfolio('standard', 'direct')
    reuse = portfolio('standard', 'prior_with_direct_fallback')
    saving_pct = 100 * (1 - float(reuse['total_labels_union']) / float(direct['total_labels_union']))
    assert float(direct['total_labels_union']) == 1945 and float(reuse['total_labels_union']) == 1200
    compare_tables = {}
    for target in ('standard', 'decomposed'):
        values = []
        for layer in ('linear', 'ffn', 'conv1', 'conv3'):
            for dtype in ('fp16', 'fp32'):
                rr = [r for r in equal if (r['target'], r['layer'], r['dtype'], r['goal']) == (target, layer, dtype, 'direct_mape')]
                base = next(r for r in rr if r['method_family'] == 'direct')
                cells = [layer + ' ' + dtype.upper(), number(base['target_mape_pct'], 2)]
                for method in ('direct', 'prior_direct', 'residual', 'component_sum'):
                    r = next(r for r in rr if r['method_family'] == method)
                    cells.append(number(r['k']) + ' / ' + number(r['total_labels_union']) if r['status'] == 'ATTAINED' else '관측 격자에서 미달')
                values.append(cells)
        compare_tables[target] = md(['조건', '목표 MAPE %', 'direct k / L', 'prior 입력 k / L', '잔차 보정 k / L', '구성합 k / L'], values)
    chosen = []
    for r in equal:
        if (r['goal'], r['target'], r['method_family']) == ('direct_mape', 'standard', 'prior_with_direct_fallback'):
            chosen.append([r['layer'] + ' ' + r['dtype'].upper(), r['method'], r['learner'], number(r['k']),
                           number(r['mean_mape_pct'], 2), number(r['total_labels_union']), number(r['raw_records_union'])])
    chosen_table = md(['조건', '회고적 구성', '학습기', 'k', 'MAPE %', '집계 L', '원측정 3L'], chosen)
    threshold = []
    for target in ('standard', 'decomposed'):
        for goal in ('mape_10', 'mape_15', 'mape_20', 'legacy_15_30'):
            rr = [portfolio(target, m, goal) for m in ('direct', 'prior_direct', 'residual', 'component_sum', 'prior_with_direct_fallback')]
            threshold.append([target, goal] + [r['attained_conditions'] + '/8' +
                (' (' + number(r['total_labels_union']) + ')' if r['total_labels_union'] else '') for r in rr])
    threshold_table = md(['대상', '목표', 'direct', 'prior 입력', '잔차', '구성합', 'prior 활용 + direct'], threshold)
    layer_costs = md(['분할', '시도', '저장 원측정', '통제 수집 초', '그중 실패 초'],
        [[r['split'], r['attempts'], number(r['recorded_measurements']), number(r['observed_controlled_wall_seconds'], 3), number(r['failed_seconds'], 3)]
         for r in rows('layer_collection_splits')])
    sessions = rows('gemm_worker_sessions')
    session_table = md(['수집 이력', '상태', '생성 원측정', 'worker 초'],
        [[r['session'], r['state'], number(r['new_records']), number(r['elapsed_seconds'], 3)] for r in sessions])
    cpu_table = md(['CPU 범위', 'wall 초', 'fit 호출 합계 초'],
        [[r['stage'], number(r['wall_seconds'], 3) if r['wall_seconds'] else '별도 wall 없음', number(r['fit_seconds'], 3)]
         for r in rows('gemm_cpu_costs')] + [['layer 전체 탐색, 선택, 저장', number(s['layer_cpu']['wall_seconds'], 3), number(s['layer_cpu']['model_fit_seconds'], 3)]])
    be = [r for r in rows('break_even_conditional') if r['goal'] == 'direct_mape' and r['method'] == 'prior_with_direct_fallback']
    be_table = md(['대상', '공통 준비비 범위', '단위', '초기 비용', '묶음당 차이', '첫 절감 묶음 수'],
        [[r['target'], r['source_scenario'], r['accounting'], number(r['source_cost'], 3 if 'seconds' in r['accounting'] else 0),
          number(r['saving_per_task'], 3 if 'seconds' in r['accounting'] else 0), r['first_saving_task_count']] for r in be])
    source_curve = rows('gemm_learning_curves')
    gt = []
    for dtype in ('fp16', 'fp32'):
        for mode in ('allocate', 'prealloc', 'graph'):
            rr = [r for r in source_curve if r['cohort'] == 'pooled_1611' and r['dtype'] == dtype and r['mode'] == mode]
            a = next(r for r in rr if float(r['n']) == 3345)
            b = next(r for r in rr if float(r['n']) == 11498)
            gt.append([dtype.upper(), mode, number(a['mean_mape_pct'], 3), number(b['mean_mape_pct'], 3),
                       number(float(a['mean_mape_pct']) - float(b['mean_mape_pct']), 3)])
    gemm_table = md(['정밀도', '실행 방식', '3,345형상 MAPE %', '11,498형상 MAPE %', '감소 %p'], gt)
    text = f'''# 비용과 데이터 효율 분석 (2026-11-11)

Linear와 FFN의 네 정밀도 조건은 5개 학습 형상의 잔차 보정으로 기존 direct 최선 validation MAPE를 충족했다. 표준 레이어 8조건을 함께 다루면, prior를 활용하고 Conv3×3 FP16에는 direct를 유지하는 회고적 구성의 고유 레이블 예산은 1,945개에서 1,200개로 {saving_pct:.2f}% 줄었다. validation 399개와 세 seed의 학습 집합 합집합을 포함한 수치다. 세 회차 기준 원측정은 5,835건과 3,600건이다.

확보된 실제 레이어 통제 수집은 실패를 포함해 133,109.178초(36.975시간)다. GEMM은 초기 중단과 세 번의 재측정, 10,000형상 확대를 포함한 7개 worker 세션에서 1,175,272.749초(326.465시간)를 복원했다. 레이블 예산은 선택한 구성에 필요한 자료를 회고적으로 세는 값이고, 실제 장부는 이미 수행한 수집과 탐색의 비용이다.

## 비교 기준과 집계 단위

주 비교의 목표는 각 레이어, 정밀도, 실행 대상의 동결된 direct 최선 평균 validation MAPE다. 기존 k=5, 10, 20, 30, 50, 100, 200과 zero k=0 격자에서 목표를 만족하는 최소 실제 학습 레이블 수를 찾았다. 동률은 MAPE, p90, 학습기, 방법, k 순서로 정한다. 기존 정확도 우선 선택 규칙의 지표와 실패 배제 기준을 유지하고, 자료량을 앞에 둔 이번 규칙은 사후 효율 분석으로 [policy.json](policy.json)에 기록했다.

보조 목표는 MAPE 10%, 15%, 20%, 기존 RQ4의 MAPE 15%와 p90 30% 동시 충족, direct의 MAPE와 p90 동시 충족이다. 15/30은 [기존 비용 정정](../integrated_rq14_2026-10-28/runs/v2/selection_correction.md)의 기준을 레이어에도 적용한 민감도 분석이다. 모든 평가는 세 seed 평균을 사용한다. [전체 비교표](tables/layer_equivalent_accuracy.csv)에 seed별 MAPE 최솟값과 최댓값, 세 seed 모두 목표를 충족하는지도 남겼다. 세 seed의 범위는 통계적 신뢰구간과 구분한다.

형상은 레이어의 정적 차원 조합이다. 집계 레이블 한 개는 특정 형상, 정밀도, 구현의 독립된 세 회차 측정을 집계한 시간값이다. 직접 예측, prior 입력, 잔차 보정은 형상당 전체시간 레이블 1개를 사용한다. 구성합은 Linear 1개, FFN과 Conv1×1 각각 3개, Conv3×3 4개를 사용한다. 모든 방법은 평가 대상의 validation 전체시간 레이블도 사용한다. 같은 레이블을 여러 seed와 학습기가 공유하면 한 번만 센다.

레이어 조립 오차 모델은 `log(T_layer / S)`를 학습하는 통합 잔차 보정기다. 여기서 S는 동결한 GEMM 예측시간의 합이며 실행 예측 차이, 보조 연산과 연결 실행 차이를 함께 보정한다. prior 입력은 정적 특징에 log(S)를 더한다. 구성합은 구성 연산별 정적 특징 모델을 별도로 학습하며 공통 GEMM prior 준비비를 요구하지 않는다.

이번 산출물은 기존 개발 결과를 재집계한 분석이다. 2026-11-11의 [최종 test 평가](../final_test_2026-11-11/README.md), 가중치와 선택은 그대로 보존했다. 레이어 선택 SHA-256은 `beed9b4314c3f258f0fcd6f86d19b15d271fac2d1abe705875aa50a94e99cbd4`다. 이 분석의 작은 k 구성은 개발 집합에서 계산한 회고적 비교이며, 해당 구성에 대한 최종 test 성능은 현재 표의 근거 범위 밖이다.

## 같은 정확도에 필요한 레이어 자료량

표의 k는 계획 학습 형상 수, L은 세 seed 합집합의 학습 레이블과 validation 레이블을 더한 수다. 원측정 수는 3L이다. FFN FP32 k=200의 실제 학습 형상은 198개이며 k=100은 seed에 따라 99~100개다. Conv3 FP32 validation은 49개이고 나머지는 50개다. 기존 품질 제외와 공통 PASS 집합을 유지했다.

### 표준 forward

{compare_tables['standard']}

Linear와 FFN의 잔차 보정은 k=5에서 네 조건 모두 direct 최선의 평균 MAPE보다 낮았다. 이 네 조건은 각 seed에서도 목표를 충족했다. k=5의 세 seed 학습 합집합은 14형상이므로 validation 50개를 더한 L은 64개다. Linear의 direct L=224와 비교하면 71.4%, FFN의 L=250 또는 248과 비교하면 약 74% 작은 예산이다. 학습 형상 5개와 validation 50개를 함께 세면 학습량만의 감소율보다 절감 폭이 작아진다.

Conv1×1 FP16은 잔차 보정 k=100, L=223에서 목표를 충족했다. Conv1×1 FP32는 prior 입력 k=200이 필요했다. Conv3×3 FP16은 direct가 가장 낮은 MAPE를 보였고, FP32 잔차 보정도 k=200을 사용했다. 자료 효율의 이득은 Linear와 FFN에 크게 나타났다.

prior 계열과 direct 가운데 조건별 최소 예산을 택한 분석용 조합은 다음과 같다. Linear는 zero만으로 direct 목표를 충족해 레이어 학습 레이블이 0개이고 validation만 센다. zero의 학습기 열은 반복 격자의 식별자이며 실제 source는 FP16 MLP32와 FP32 RF다. 전체 정확도 우선 동결 모델과 이번 최소 예산 구성은 목적이 다른 두 결과로 보존한다.

{chosen_table}

합계는 학습 레이블 801개, validation 399개, 총 1,200개다. direct는 학습 1,546개와 validation 399개로 총 1,945개다. 한 seed에서 8조건의 실제 학습 형상 수를 더하면 direct 1,398개, 분석 조합 710개이며, 세 seed와 dtype 간 중복까지 제거한 레이어 형상 합집합은 각각 774개와 414개다. 이 형상 합계와 레이블 수는 분모가 다르다. direct의 p90까지 동시에 맞추면 분석 조합은 1,242레이블로 늘어 절감률은 36.14%다.

### 분해 forward

{compare_tables['decomposed']}

분해 Conv는 구성합의 정확도가 좋지만 형상당 레이블 수가 크다. Conv1 FP16은 direct k=200, L=250과 같은 목표를 구성합 k=100에서 달성했으나 L=569를 사용했다. Conv3 FP32는 direct k=200, L=249에 대해 구성합 k=100, L=741이다. 형상 수가 절반이어도 구성 레이블을 포함하면 예산이 커진다. 공통 GEMM prior를 사용하는 방법과 direct의 최소 예산 조합은 전체 8조건에서 1,887 → 1,112레이블(41.07% 감소), 원측정 5,661 → 3,336건이다.

표준 Conv와 명시적 분해 Conv의 실행 경로를 각각 평가했다. 기존 validation에서 분해/표준 시간 중앙값은 약 2.02~2.98배였다. 구성합의 표준 Conv 오차가 큰 이유를 해석할 때 이 경로 차이를 함께 고려해야 한다. 근거는 [기존 validation 보고서](../layer_full_development_2026-11-11/README.md)의 구현 비교다.

### 고정 정확도 목표의 민감도

각 셀은 목표 달성 조건 수/8이며, 8조건을 모두 달성한 경우 괄호에 총 L을 적었다. `legacy_15_30`은 MAPE≤15%, p90≤30%다.

{threshold_table}

표준 Conv3 FP32는 관측된 격자에서 MAPE 15% 목표에 도달하지 못해 15% 기준의 전체 8조건 비용은 계산할 수 없다. 20% 목표에서는 direct 983개, prior 활용과 direct를 합친 구성이 614개다. 이 경우 비교하는 오차 수준은 앞의 조건별 direct 최선 기준보다 완화된다.

![표준 레이어의 레이블 학습곡선](figures/layer_standard.png)

[분해 레이어 그림](figures/layer_decomposed.png), [표준 PDF](figures/layer_standard.pdf), [분해 PDF](figures/layer_decomposed.pdf). 각 점은 해당 방법과 k에서 MAPE가 가장 낮은 학습기다. 축은 로그 척도이며, x축은 세 seed 고유 레이블 합집합에 validation을 포함한다. 그림의 선은 관측점을 연결하며 자료량 사이의 미관측 성능은 추정하지 않는다.

## GEMM 자체의 자료 효율과 고정 prior

기존 validation 716형상에서 MAPE 15%, p90 30% 목표는 FP16 1,000학습 형상, FP32 100형상으로 세 실행 방식 모두 달성했다. 세 seed 합집합과 validation을 포함한 실행 방식별 L은 FP16 2,718개, FP32 1,006개다. FP32의 MAPE 10% 목표는 300학습 형상, L=1,525에서 달성했다. FP16은 관측 격자에서 10%에 도달하지 못했다. [GEMM 목표별 예산](tables/gemm_target_budgets.csv)

확대 비교의 validation은 기존 716개와 새 895개를 합친 1,611형상이다. 이 집합에는 n=3,345, 4,345, 6,345, 11,498 결과가 있다. 같은 1,611형상에서 비교한 양 끝점은 다음과 같다.

{gemm_table}

1,611형상 validation에서도 최소 관측 n=3,345가 이미 여섯 조건의 15/30 목표를 충족한다. FP32는 10%도 충족한다. 확대는 정확도 개선을 얻는 선택이며, 목표 15/30만을 위한 최소 관측 예산은 작은 쪽에 있다. 716형상과 1,611형상 곡선의 차이는 평가 분포 변화도 포함하므로 각 곡선 안에서만 비교한다.

레이어 분석의 공통 prior는 기존에 동결한 n=11,498의 FP16 MLP32와 FP32 RF allocate 모델이다. 작은 GEMM 모델을 레이어에 연결한 개발 결과가 없으므로 레이어 효율 계산에는 이 여섯 가중치를 고정했다. 작은 GEMM 모델의 자체 MAPE를 근거로 레이어 조립 오차까지 동일하다고 가정하면 공통 준비비를 과소평가할 수 있다.

![GEMM 자료량 곡선](figures/gemm_curves.png)

[GEMM 그림 PDF](figures/gemm_curves.pdf), [곡선 표](tables/gemm_learning_curves.csv). 고정 prior의 allocate 두 dtype 학습 레이블은 22,996개, validation은 3,222개다. 합계 26,218레이블, 세 회차 78,654원측정이 선택된 source 구성의 회고적 예산이다. 세 실행 방식의 전체 공통 개발 비교는 78,654레이블, 235,962원측정이다. 세 seed는 같은 전체 source 형상을 공유한다.

## 실제 수행한 수집과 탐색 비용

### 레이어 장부 검산

33개 반환 ZIP과 비용 장부를 기존 `costs.collection_costs()`로 검증했다. 검증된 모든 시도 합계는 133,109.17767499853초, 저장 원측정은 32,666건이다.

{layer_costs}

실패 세 시도의 합은 1,545.723초, 302원측정이다. validation의 두 실패 시도는 272건과 0건, test 실패는 30건을 남겼다. 성공한 30묶음의 저장량은 32,364건이다. train과 validation만의 실제 통제 수집은 110,925.093초(30.813시간)이며 실패 1,244.796초를 포함한다. 성공한 개발 묶음의 26,982원측정은 8,994집계 레이블에 대응하고, 품질 REVIEW 두 집계를 제외한 PASS는 8,992개다.

기존 정확도 우선 선택의 표준과 분해 16조건 전체는 5,994레이블, 17,982원측정의 회고적 예산을 갖는다. 그중 train은 5,196레이블, validation은 798개다. 이번 표준 8조건의 최소 예산 1,200개와는 대상 범위와 정확도 목적이 다르다. [선택 예산 원본](../layer_full_fit_2026-11-10_v5/prepared_layer_full/cost_report.json), [시도별 장부](tables/layer_collection_attempts.csv)

공통 PASS 집합을 확인하는 과정에서는 표준, 분해와 구성 연산 레이블을 함께 열람했다. 특히 FFN FP32 두 형상의 제외는 구성 GELU의 품질에서 결정됐다. 좁은 구성의 L 예산은 이미 검증된 형상 집합을 사용하는 조건부 값이다. 같은 품질 판정을 새 자료에서 다시 확인하려면 구성별 검증 비용을 추가해야 한다. 현재 실제 개발 접근 비용 8,994개에는 이 비용이 포함된다.

세 회차는 동결 수집 기준이다. 이번 계산의 반복 측정 수는 모두 `원측정=3×고유 집계 레이블`로 환산했다. 7개 시간 블록과 내부 CUDA 반복은 각 원측정의 타이밍 절차다. 1회 또는 2회 측정으로 같은 예측 정확도와 품질 선별을 유지할 수 있는 최소 반복 수는 현재 분석으로 식별되지 않는다.

### GEMM 공통 준비비 복원

기존 결합 원자료의 session.json 여섯 개와 확대 원자료의 session.json 하나를 고유 session ID로 합쳤다. 중단된 0건 세션과 779건 세션, 최종 데이터에서 교체된 v3 재측정도 비용에 포함했다.

{session_table}

합계는 {number(s['gemm_recorded_worker_seconds'], 3)}초와 {number(s['gemm_all_session_records'])}생성 원측정이다. 기존 720 MHz 계보의 비용은 475,170.742초, 확대는 700,102.007초다. 초기 전체 수집 113,382건에 재측정 8,760건과 확대 180,000건을 더한 값이 302,142건이다. 최신 결합 자료와 확대 자료에 남은 고유 원측정 293,382건과 실제 생성량을 구분한다.

SQLite를 `mode=ro&immutable=1`로 열고 task의 split, phase, dtype, mode, session, task_wall_seconds만 SQL로 집계했다. 확대 task 시간 합은 기존 검산의 695,035.421초와 일치했다. worker 총시간과의 차이 5,066.586초에는 예열, 모니터링, 기록과 구간 분석 등 공통 작업이 들어 있다. 기존 자료는 교체된 원측정 때문에 최신 DB의 task 합과 과거 전체 session 합의 차이를 공유 준비비만으로 해석할 수 없다.

현재 남아 있는 train과 validation의 allocate task 시간 합은 {number(s['gemm_allocate_train_validation_retained_task_seconds'], 3)}초다. 이 값은 공통 PASS 밖의 부분 정상 조건도 포함하고 worker의 공유 작업을 제외한다. 따라서 선택된 26,218레이블만 다시 준비하는 전체 wall time으로 사용하기에는 비용 경계가 좁다. [분할과 실행 방식별 원장 추적](tables/gemm_task_costs_by_split_mode.csv), [세션 출처](tables/gemm_worker_sessions.csv)

GEMM session 시간은 worker 내부 예열과 수집, 기록, 구간 분석을 포함한다. 레이어 장부는 통제 수집, 클럭 설정과 복원, 분석을 포함한다. 각 합계의 경계를 유지한다. GPU 사용 전 환경 wrapper, launcher 바깥 검사, 압축, 장기 유휴 간격과 더 이른 교정 실험은 이 합계 밖에 남아 있다.

### CPU 학습 비용

{cpu_table}

GEMM 기존 비교와 확대 비교의 기록된 CPU wall 합은 6,524.970초(108.75분), 레이어는 789.398초(13.16분)다. 선택된 여섯 GEMM prior의 fit 호출 84.670초는 확대 탐색 안의 부분합이다. CPU와 GPU 시간은 자원과 작업 경계가 달라 별도로 보고한다. 레이어 fit 715.358초에는 탐색과 선택 모델의 refit이 포함되고, 기록된 예측은 9.127초다. 나머지 64.913초는 저장과 기타 wall overhead다. [방법별 CPU 호출](tables/layer_cpu_by_method.csv), [선택 source fit 여섯 건](tables/gemm_selected_prior_fit_calls.csv)

## 공통 prior 재사용의 손익분기점

독립적인 새 자료 묶음 q개에서 direct 준비비를 D, prior를 쓰는 준비비를 R, 공통 GEMM 준비비를 G라 두면 `direct=qD`, `prior=G+qR`이다. `D>R`일 때 처음 비용이 작아지는 정수 q는 `floor(G/(D−R))+1`이다. 동률까지 허용하면 `ceil(G/(D−R))`이며 최소 1로 둔다. 이미 준비된 prior의 추가 source 비용은 G=0이다.

표준 또는 분해의 한 묶음은 동일 hardware와 precision 계약에서 네 레이어 × 두 dtype의 8조건을 다시 다루는 경우다. combined는 표준과 분해를 함께 다루는 16조건 묶음이며 source 준비비를 한 번만 센다. 각 새 묶음이 현재와 같은 정확도, 레이블 요구량과 validation 크기를 가지며 묶음 간 레이어 레이블을 공유하지 않는다는 가정이다. 현재 관측은 표준 한 묶음과 분해 한 묶음이므로, q번 재사용의 이득은 이 가정에 따른 외삽이다.

{be_table}

allocate 선택 구성의 source 예산 26,218레이블을 처음부터 부담하면 표준 36묶음, 분해 34묶음부터 레이블 합계가 작아진다. 기존 prior를 재사용하는 현재 상태에서는 첫 새 묶음부터 표준 745개, 분해 775개의 조건부 레이블 차이가 생긴다. 실제 탐색 범위인 여섯 source 조건을 모두 초기 비용으로 배분하면 각각 106묶음과 102묶음이다. 모든 과거 생성 원측정 302,142건을 세 회차 단위로 환산한 행은 재측정과 시험, 진단까지 부담한 역사적 비용 시나리오다.

표준과 분해를 함께 다루는 16조건의 예산은 direct 3,832개, prior 활용과 direct를 합친 구성이 2,312개다. 차이는 1,520레이블이고 allocate source 준비비를 한 번 부담하면 18묶음부터 총 레이블 수가 작아진다. 위 표의 standard, decomposed, combined는 서로 다른 활용 시나리오다.

개별 레이어와 dtype만 반복 활용하는 경우에는 source 한 dtype의 13,109레이블을 사용하는 [조건별 계산표](tables/break_even_single_condition.csv)가 있다. 이 source 예산을 서로 다른 레이어에 각각 다시 더하면 공유 준비비를 중복 계산하게 된다.

CPU 호출 범위에서 표준 direct의 선택 후보 fit 합은 9.952초, 분석 조합의 fit과 prior 예측은 2.362초다. 공통 여섯 fit의 84.670초를 나누면 12묶음이며 분해는 19묶음이다. 이 계산은 실제로 기록된 특정 호출을 합친 것으로, 전체 탐색, 저장, 일반 예측과 환경 준비 wall은 포함 범위를 따로 둔다. CPU 반복 벤치마크가 없어 작은 호출 시간 차이의 재현성도 미확인이다.

### GPU 시간 배분 시나리오

개발 레이어의 실제 통제 수집 110,925.093초를 성공한 개발 집계 8,994개로 나누면 집계 레이블당 12.333초다. 실패 시도와 공유 준비비도 이 평균에 배분된다. 이 단가가 좁은 구성에서도 선형적으로 유지된다고 가정하면 표준 direct는 23,988.137초, 분석 조합은 14,799.879초이고 묶음당 차이는 9,188.258초다.

GEMM 역사적 worker 시간 1,175,272.749초 전체를 초기 비용으로 넣는 시나리오에서는 표준 128묶음, 분해 123묶음이다. 레이어 한계 단가를 평균의 0.5배와 2배로 바꾸면 표준은 256~64묶음, 분해는 246~62묶음으로 변한다. [시간 배분 민감도 표](tables/gpu_time_allocation_sensitivity.csv)

이 시간 계산은 두 장부의 서로 다른 경계와 선형 배분 가정을 사용한다. 실제 실패 확률, 예열의 고정비, 구성별 수집시간, 공통 PASS 판정과 새로운 작업 분포에 따라 손익분기점이 달라진다. 같은 경계에서 좁은 direct와 prior 구성을 각각 수집한 wall time이 없어 실측 기반 시간 손익분기점은 UNKNOWN으로 유지한다. 이번 캠페인의 실제 시간 절감량도 이 장부만으로 계산할 수 없다.

## 남은 UNKNOWN과 근거의 범위

| 항목 | 확인한 근거와 현재 판정 | 계산에 미치는 영향 |
| --- | --- | --- |
| 공통 GEMM core 수집 | 기존 6세션과 확대 1세션에서 326.465시간 복원 | 초기 중단과 재측정, 세션 내부 공유 작업 포함 |
| GEMM 전체 연구 준비 wall | session 바깥 launcher, 환경 검사, 초기 P/B 교정과 clock 탐색, 압축, 분석의 같은 경계 장부가 없음 | 전체 연구 프로젝트 비용은 복원된 수집보다 큼 |
| 선택된 source만의 수집 wall | allocate task 합과 선택 레이블 수는 확보, 공유 작업의 선택 구성 귀속은 미기록 | 레이블 기준 손익분기점과 시간 배분 시나리오로 제시 |
| 좁은 레이어 구성의 실측 비용 | 기존 수집은 표준, 분해와 구성 연산을 묶음으로 실행 | 실제 선택 구성 수집 절감량과 시간 손익분기점 UNKNOWN |
| 품질 판정만의 추가 비용 | 공통 PASS 판정에 전체 구현 자료를 사용 | 회고적 최소 L은 이미 검증된 형상 집합 조건 |
| 반복 측정 최소 횟수 | 세 회차 기준과 원측정 수 검산 | 1~2회로 줄였을 때 정확도와 품질 유지 여부 UNKNOWN |
| CPU 전체 프로젝트 비용 | 기존 비교, 확대 비교, 레이어 fit의 완료 장부 확보 | 이전 RQ14 탐색과 외부 분석, 검산 작업까지의 총 CPU 비용은 별도 |
| 전력량과 금전 비용 | 통제 시간은 확보, 일관된 전체 에너지 적분과 요율 장부는 범위에 없음 | 시간과 레이블 단위로 보고 |

미확인 비용은 항목별 경계와 필요한 근거를 표에 남겼다. 고정 GEMM prior가 준비된 조건에서 Linear와 FFN의 레이어 레이블 요구량이 크게 줄었고, Conv에서는 구현과 정확도 목표에 따라 이득이 달라졌다. 새 source를 한 번 준비하는 비용은 크므로 재사용 효과를 주장할 때 초기 비용과 추가 작업의 비용을 함께 밝혀야 한다.

## 산출물과 재현

계산은 [analyze.py](analyze.py), 보고서 생성은 [write_report.py](write_report.py), 경계 검사는 [test_analysis.py](test_analysis.py)에 있다. [summary.json](summary.json)은 주요 수치, [input_manifest.json](input_manifest.json)은 입력 SHA-256, [verification.json](verification.json)과 [독립 검산](independent_verification.json)은 보존과 검산 결과다. 계산 중 두 번의 구성 수 검사 실패는 analysis.log와 analysis_002.log에 남겼다. 실제 protocol의 격자 길이를 참조하도록 고친 실행은 analysis_003.log이며, 두 실행 대상을 함께 계산한 최종 로그는 [analysis_004.log](analysis_004.log)다.

[verify.py](verify.py)는 레이블 ID를 직접 열거해 2,304개 구성, 최소 예산 질의 768개, 묶음 집계 144개와 손익분기점 94개를 독립 검산했다. 입력 700개의 SHA-256이 유지됐고, 새 경계 검사 5개와 기존 비용 검사 2개가 통과했다. 그림 세 종류의 PNG와 PDF 렌더를 확인해 축·범례·텍스트의 가독성과 잘림을 점검했다.

```bash
cd /Users/studyhard/Claude/자율연구
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/research_efficiency_mpl
export XDG_CACHE_HOME=/private/tmp/research_efficiency_cache
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B -m unittest discover \\
  -s analysis/cost_data_efficiency_2026-11-11 -p 'test_*.py'
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \\
  analysis/cost_data_efficiency_2026-11-11/analyze.py
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \\
  analysis/cost_data_efficiency_2026-11-11/write_report.py
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/cost_data_efficiency_2026-11-11/verify.py
```

재현 명령은 이 분석 폴더의 파생 표와 그림만 다시 쓴다. 원자료, 동결 실행본, 모델과 최종 평가 산출물은 입력 해시로 검증한다. 연구 범위는 GEMM과 레이어이며 전체 신경망 end-to-end 검증은 이번 분석에서 제외했다.
'''
    (HERE / 'README.md').write_text(text)
    print('PASS: Korean report rendered from computed tables')


if __name__ == '__main__':
    main()
