# 지정한 정확성 실패 조건을 제외하는 v4 레이어 분석

완료된 v2 학습 20묶음과 v4 validation/test를 연결한다. `validation_01`의 Conv3×3 형상 `lf_ce6440e85a4add93`에서 FP32만 제외하며, 같은 형상의 FP16은 유지한다. 제외 근거는 v3 정확성 probe의 원래 FP64 허용오차 초과 기록이다. 원래 정확성 기준과 측정 코드는 수집본의 동결 명세를 따른다.

제외 대상은 세 회차와 여섯 구현의 18측정이다. 첫 validation 묶음은 원계획 1,080개, 제외 18개, 수집 대상 1,062개로 기록한다. 분석에서는 시간값이 없는 FAIL 조건 여섯 개를 남긴다. 이 조건은 측정 레이블 수에 더하지 않는다. 지정된 제외를 반영한 Conv3×3 FP32 validation 계획은 50형상 중 49형상이며, 모든 학습기와 예측 방식이 같은 공통 PASS 집합을 사용한다.

v2 학습 결과는 `inheritance.json`에 고정된 ZIP의 파일명, SHA-256, 소스와 protocol로 검증한다. 실패한 v2 `validation_01_attempt_001`과 v3 `validation_01_attempt_002`도 각각 원본 ZIP과 당시 campaign 이력으로 확인해 수집 비용에 포함한다. 완료된 v4 재시도를 validation 학습 입력으로 사용한다.

11,498형상 FP16 MLP32와 FP32 RF의 공통 GEMM prior 여섯 가중치, 학습기 비교, 모델 선택과 시험 잠금 규칙은 [기존 공통 모델 분석](../layer_full_fit_2026-11-08/README.md)을 유지한다.

## 준비와 학습

Mac 연구 폴더에서 아래 명령으로 분석 protocol을 별도 동결한다. 상속 명세와 실패 근거, 공통 GEMM 가중치를 함께 고정한다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-10_v4/pipeline.py prepare \
  --output analysis/layer_full_fit_2026-11-10_v4/prepared_layer_full
```

validation 5묶음 수집을 완료한 뒤 `CAMPAIGN_PATH`를 v4 반환 캠페인 경로로 바꿔 실행한다. 반환 폴더에는 `campaign.json`, `inherited_campaign.json`, `previous_campaign.json`, 이어받은 22개 원본 ZIP과 SHA-256 파일, v4 결과가 함께 있어야 한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-10_v4/pipeline.py train \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-10_v4/prepared_layer_full
```

최종 시험은 모델 선택 동결과 사용자 승인 후 `pipeline.py evaluate`로 진행한다. train_03과 train_09의 FP32 GELU 회차 변동도 기존 품질 제외 규칙을 따른다.

## 검증

```bash
MPLCONFIGDIR=/private/tmp/layer-full-report-mpl \
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B -m unittest discover \
  -s analysis/layer_full_fit_2026-11-10_v4 -p 'test_*.py' -v
```

지정한 정확성 제외와 근거 해시, 선언되지 않은 누락의 거부, 공통 validation 집합, 세 버전의 입력 구분과 두 실패 시도 비용을 검사한다. 기존 입력 검산, 공통 GEMM 모델 재현, 합성 자료의 학습과 모델 선택, 시험 분리, 보고서 검사도 포함한다.
