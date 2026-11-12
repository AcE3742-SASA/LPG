# 선택한 확대 GEMM 모델을 공통으로 쓰는 레이어 분석

2026-11-08 사용자가 선택한 방식에 따라, GEMM 검증에서 가장 낮은 오차를 기록한 FP16 MLP32와 FP32 RF를 네 레이어 학습기에 공통으로 연결했다. 각 모델은 11,498형상으로 학습한 allocate 정규화 모델이다. 기존 11월 4일 분석본과 수집 소스는 보존하고, 이 폴더에서 새 protocol을 고정한다.

## 연결한 모델과 비교 근거

| dtype | 공통 GEMM 모델 | 학습 형상 | 기존 최선 allocate MAPE | 확대 최선 allocate MAPE |
|---|---|---:|---:|---:|
| FP16 | MLP32, log(T/L) | 11,498 | 11.719% | 10.045% |
| FP32 | RF, log(T/L) | 11,498 | 7.043% | 6.477% |

지표는 같은 1,611검증 형상에서 세 seed의 평균이다. 기존 3,345형상과 확대 11,498형상에서 각각 선택된 최선 모델을 비교했다. [GEMM 결과 보고서](../gemm_expansion_result_2026-11-06/추가GEMM_10000형상_분석결과_2026-11-06.md)와 [선택 가중치 목록](../gemm_expansion_result_2026-11-06/fit_v1/accuracy_first_model_refs.csv)이 근거다. 레이어 오차 개선량은 반환한 레이어 자료의 학습과 검증에서 확인한다.

- 레이어 후속 학습기는 RF, Ridge, MLP32, MLP64다. 같은 dtype·seed에서는 네 학습기 모두 같은 GEMM prior를 사용한다. protocol의 키에 있는 `learner`는 레이어 학습기이며, source 명세의 `learner`는 GEMM 모델이다.
- 레이어 seed 20261104/05/06은 GEMM seed 20261028/29/30에 순서대로 대응한다. 실제 source 가중치는 6개이고 24개 레이어 학습기·dtype·seed 조합에서 공유한다.
- Linear는 GEMM 하나, FFN은 GEMM 둘의 예측 합, Conv는 `(batch*height*width, cout, cin*kernel**2)`의 GEMM 예측을 prior로 사용한다. GELU, unfold와 pack은 구성 연산 모델 및 레이어 보정으로 다룬다.
- 표준/분해 forward 직접 예측, prior 입력 직접 예측, 잔차 보정, 상수 비율, 무보정 합과 구성 연산별 예측 합을 비교한다. 구성 연산 모델도 기존 네 학습기와 정적 특징을 사용한다.
- GEMM 학습·검증 13,109형상과 새 레이어 1,200사례의 구성 GEMM 1,500개 사이에서 동일 형상 및 한 축 ±8 이내 충돌이 0개다. 이 검사는 시간값 없이 manifest만 사용한다.
- Conv의 GEMM 축 64 미만 구간은 기존 분석의 별도 평가 구역으로 유지한다. 전체 레이어 결과에서 이 범위의 성능을 함께 보고한다.

## 보존과 검증

`pipeline.py`는 [11월 4일 실행기](../layer_full_fit_2026-11-04/pipeline.py)의 source 연결을 새 버전으로 갱신한 파일이다. 기존 `components.py`, `data.py`, `costs.py`, `report.py`와 학습기·특징 계산을 재사용한다. 이 코드들과 manifest, source 선택표·모델·GEMM protocol의 해시를 [새 동결 protocol](prepared_layer_full/frozen_protocol.json)에 기록한다.

source 선택표는 GEMM 결과의 최종 artifact 해시와 대조한다. 가중치는 읽기 전에 해시를 검사하고, 읽은 뒤 learner, dtype, allocate mode, target, seed, 11,498형상, 특징 순서와 source protocol 해시를 확인한다. 개발용 모델 상태와 시험 잠금 상태도 검사한다. 모델이나 코드가 바뀌면 새 실행 폴더에 다시 준비해야 한다.

이번 source에는 기존 legacy_train 전체가 포함된다. 레이어 자료는 이 source와 분리되어 있다. 기존 RQ4는 독립된 A/C 분할과 실행본을 계속 사용한다.

11월 8일 기록상 train_01의 품질·정확성 점검과 train_02 중단 진단이 진행됐다. 이번 공통 prior 선택은 11월 6일 GEMM 검증의 동결된 선택을 따른다. 이 연결 준비 단계에서 레이어 실측 학습과 실제 시험 평가는 후속 작업으로 남긴다.

## 실행

연구 폴더를 현재 경로로 두고 기존 CPU 분석 환경을 사용한다. 먼저 계산 스레드를 고정한다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

`prepared_layer_full`은 준비된 실행 폴더다. 개발 분석 입력은 train 20묶음과 validation 5묶음이 완료된 campaign이며, 실패·중단 시도도 비용 계산을 위해 함께 보존한다. `CAMPAIGN_PATH`를 반환 campaign 경로로 바꿔 실행한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-08/pipeline.py train \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-08/prepared_layer_full

analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-04/report.py \
  --run analysis/layer_full_fit_2026-11-08/prepared_layer_full \
  --output analysis/layer_full_fit_2026-11-08/reports/development
```

새 실행 폴더가 필요하면 `pipeline.py prepare --output NEW_RUN`으로 만든다. 기존 실행의 덮어쓰기는 차단한다. 최종 시험 열람을 승인받고 모델 선택을 동결한 뒤에는 다음 별도 명령을 사용한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-08/pipeline.py evaluate \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-08/prepared_layer_full \
  --evaluation-output analysis/layer_full_fit_2026-11-08/results/final_layers \
  --allow-sealed-test
```

품질·입력 검산, 공통 PASS 집합, 부분집합별 실제 유지 수, 최종 모델 저장·재현, 별도 시험 접근 기록과 비용 산출은 [기존 분석 규칙](../layer_full_fit_2026-11-04/README.md)을 따른다. 공유 source의 비용은 공통 준비비로 취급한다. 기존 GEMM 전체 수집비와 비교 가능한 비용 근거가 확보되면 총비용과 손익분기점을 계산한다.

## 검사

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B -m unittest discover \
  -s analysis/layer_full_fit_2026-11-08 -p 'test_*.py' -v
```

`test_sources.py`는 선택된 실제 GEMM 모델의 개발 예측 9,666개 재현, 공통 모델 연결, 형상 변환, metadata·해시 거부와 형상 분리를 검사한다. `test_pipeline.py`는 합성 자료로 네 레이어의 학습·저장·재현·시험 분리 흐름을 확인한다. 최종 검사 기록은 `validation.json`에 둔다.
