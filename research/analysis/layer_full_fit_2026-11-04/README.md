# 확대 수집 후 학습, 최종 평가와 비용 분석

2026-11-04 준비본. `layer_full_20261104_v1`의 네 레이어, 두 dtype과 표준·분해 경로를 분석한다. 구성 연산별 모델과 그 예측 합도 포함한다. 실행 환경은 기존 `analysis/integrated_rq14_2026-10-28/.venv`를 재사용한다.

## 비교 범위

| 항목 | 고정한 내용 |
|---|---|
| 레이어 | Linear, exact GELU FFN, Conv 1x1, Conv 3x3 |
| 대상 시간 | 표준 전체 forward, 분해 전체 forward를 각각 평가 |
| 학습기 | RF100/depth8, Ridge(alpha=1), MLP(32,32), MLP(64,64) |
| 자료량 | 기존 고정 목록의 5/10/20/30/50/100/200 train 그룹, 3 seed |
| 검증·시험 | 종류별 validation 50그룹으로 선택, 별도 test 50그룹으로 평가 |
| 비교 방법 | GEMM 무보정 합, 상수 비율, 레이어 직접 예측, prior 입력 직접 예측, log 잔차 보정, 구성 연산별 예측의 합 |
| 선택 기준 | 3 seed의 평균 validation MAPE, p90, 실제 학습 레이블 수 순 |
| 구성 연산 | Linear GEMM; FFN GEMM1/GELU/GEMM2; Conv1 input pack/mm/output pack; Conv3 unfold/input pack/mm/output pack |

모든 모델 입력은 형상과 dtype에서 계산하는 정적 특징이다. 구성 연산 실측은 해당 모델의 학습 정답으로 사용한다. 예측 합은 각 구성 모델의 출력만 더한다. 표준/분해 전체 시간과 구성 연산 합의 관계도 별도 표로 출력한다. 고정 중간 텐서에서 측정한 구성 시간과 연결 실행의 전체 시간은 캐시·할당 상태에 따라 달라질 수 있다.

기존 10월 28일의 계열별 A=2,652 allocate GEMM 모델을 prior로 고정했다. 새 레이어 seed 20261104/05/06을 source seed 20261028/29/30에 순서대로 대응한다. 추가 GEMM 10,000형상으로 갱신할 prior는 새 분석 버전에서 동결한다. 현재 비교의 기준과 가중치는 `frozen_protocol.json`에 남긴다.

## 입력과 품질

입력은 반환 campaign 폴더다. `campaign.json`과 각 묶음의 최신 완료 시도 폴더 또는 ZIP·SHA-256을 같은 campaign 아래에 둔다. 실패·중단 시도의 자료도 비용 계산을 위해 함께 보존한다. 개발 분석에는 train 20묶음과 validation 5묶음의 완료 자료가 필요하다. test는 나중에 추가할 수 있다.

로더는 원본 ZIP을 풀거나 바꾸지 않고 읽는다. source·결과 해시, 고정 manifest와 분할, 입력 식별, 정확성, 7블록 통계와 세 회차를 확인한다. 센서 품질은 원래 수집 분석기가 저장한 해시 검증된 판정을 사용한다. REVIEW는 사유와 함께 남기고, 동일 layer/dtype의 모든 방법은 표준·분해·구성 연산이 함께 PASS인 형상 집합에서 비교한다. 제외 형상을 다른 형상으로 채우지 않으며 계획 k와 실제 유지 수를 함께 기록한다.

## 실행 순서

아래 명령은 이 연구 폴더를 현재 경로로 두고 Mac의 기존 CPU 분석 환경에서 실행한다. `CAMPAIGN_PATH`는 실제 반환 campaign 폴더 경로로 바꾼다. 이미 생성한 `prepared_layer_full`에는 시간값을 읽기 전에 고정한 분석 protocol이 들어 있다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-04/pipeline.py train \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-04/prepared_layer_full
```

새 실행 폴더를 만들려면 먼저 `pipeline.py prepare --output NEW_RUN`을 실행한다. 기존 실행 폴더 덮어쓰기는 차단한다. 실행 중 오류가 나면 기존 폴더를 보존하고 원인을 확인한 뒤 새 실행을 만든다. 계산은 CPU에서 수행하며 측정용 GPU를 사용하지 않는다.

학습이 끝나면 `selected_configurations.json`, 선택한 모델, 재현 확인, 학습곡선용 지표와 `selection_frozen.json`이 생긴다. 전체 후보 모델은 메모리에 쌓지 않고 선택된 모델을 재학습해 저장한다. 저장 후 검증 예측의 원소별 일치를 확인한다. 재학습 비용도 CPU 비용에 포함한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-04/report.py \
  --run analysis/layer_full_fit_2026-11-04/prepared_layer_full \
  --output analysis/layer_full_fit_2026-11-04/reports/development
```

최종 모델과 평가 규칙이 확정된 뒤 시험 명령을 별도로 실행한다. 이 명령의 준비와 실제 시험 열람은 구분한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-04/pipeline.py evaluate \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-04/prepared_layer_full \
  --evaluation-output analysis/layer_full_fit_2026-11-04/results/final_layers \
  --allow-sealed-test
```

시험은 동결한 모델과 파일 해시를 먼저 확인하고, 접근 시작 기록을 남긴 뒤 읽는다. 선택된 전체 최적 모델과 방법별 최적 모델을 그대로 평가한다. 같은 형상으로 seed별 paired bootstrap 1,000회를 수행하고, 정렬·연산량·종횡비와 GEMM 축 64 미만 구간을 따로 출력한다. 반복 시험 실행은 차단한다. 실패 시 접근 기록과 오류를 보존하고 복구 방식을 검토한다.

최종 보고서는 위 `report.py` 명령에 `--evaluation analysis/layer_full_fit_2026-11-04/results/final_layers`를 추가하고 별도 출력 폴더를 지정한다.

## 기존 GEMM 최종 평가

`prepared_gemm_final`은 기존 개발 비교에서 선택한 RQ1/RQ4의 시험용 모델 묶음이다. RQ1 선택 모델과 반대 목표 표현 모델은 보존 가중치를 사용한다. RQ4는 기존 개발용 보정 자료로 선택 가중치를 재현하고, 저장된 validation 예측과 원소별 일치를 확인했다. 모든 가중치와 코드, 시험의 시간 없는 형상 목록을 고정한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-04/gemm_final.py evaluate \
  --prepared analysis/layer_full_fit_2026-11-04/prepared_gemm_final \
  --output analysis/layer_full_fit_2026-11-04/results/final_gemm \
  --allow-sealed-test
```

이는 기존 10월 28일 비교의 최종 평가 명령이다. 추가 GEMM 자료로 재학습할 모델은 시험 전에 개발 자료에서 선택하고 별도로 고정해야 한다. 기존 fresh holdout 1,000형상은 실제 시험 실행 때만 읽는다.

## 비용과 산출물

실제 수집 비용은 train·validation·test별로 모든 시도를 합한다. 실패한 시도도 포함하며, 원장이 빠진 시도는 비용 미상으로 남긴다. 원장에 기록된 controlled collection wall은 클럭 설정·복원과 분석을 포함한다. wrapper 준비, ZIP 포장과 campaign 대기 시간은 별도 미상 항목이다.

선택한 모델을 준비하는 데 필요한 학습·검증 레이블 수와 전체 비교가 열람한 레이블 수를 따로 출력한다. seed·학습기·중복 선택 사이에 공유한 레이블은 한 번만 센다. 구성 연산 합산 모델은 구성 연산별 학습 정답도 비용에 포함한다. 이 자료량 예산은 이미 수행한 실제 묶음 수집 비용과 함께 보고한다.

CPU 학습 wall, 개별 fit 합계와 계측한 예측 시간을 기록한다. 기존 GEMM source의 전체 수집 비용은 현재 미상으로 남긴다. `costs.break_even`은 동일 자원 범위의 근거 있는 source·직접 학습·재사용 준비 비용을 모두 제공했을 때 손익분기 작업 수를 계산한다. 같은 정확도에서 비용을 비교할 수 있는지도 함께 판단해야 한다.

주요 산출물은 다음과 같다.

- `metrics` / `aggregate_metrics`: 형상 수별·학습기별·방법별 MAPE, p90, 실패율과 coverage.
- `component_metrics` / `implementation_diagnostics`: 구성 연산 예측과 실측 구현 비율.
- `quality_coverage` / `input_audit`: 제외 형상과 원본 대응.
- `selected_configurations` / `models` / `selection_frozen`: 최종 선택과 재현 가중치.
- `cost_report` / `fit_timing` / `protocol_access_budget`: 관측 비용과 레이블 예산.
- `test_metrics` / `test_slices` / `paired_bootstrap`: 고정 모델의 독립 평가.
- 보고서 폴더의 CSV, PNG, PDF, Markdown: 결과 공유용 표와 그림.

## 준비본 검증

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B -m unittest discover \
  -s analysis/layer_full_fit_2026-11-04 -p 'test_*.py' -v
```

검사는 합성 자료로 수행한다. 실제 수집본 학습은 반환 후 시작하며, 실제 시험 평가 상태는 각 실행의 접근 기록과 완료 기록으로 확인한다. 기존 측정 원본과 배포 ZIP은 그대로 보존한다.
