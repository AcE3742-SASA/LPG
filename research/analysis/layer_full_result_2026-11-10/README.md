# 2026-11-10 레이어 전체 수집 결과 검증

전체 30묶음이 완료됐다. 원계획 32,400측정 중 기존 정확성 제외 정책에 따른 36건을 제외하고 32,364건을 수집했다. 기존 분석 실행본으로 학습과 validation 분석을 진행할 수 있다. 공통 비교에 쓰는 FFN FP32 학습 형상은 198개, Conv3 FP32 validation 형상은 49개다.

사용자가 지정한 이번 범위는 수집 결과 정리와 다음 단계 판단이다. 아래 검증은 원본 무결성, 완료 상태, 공개 품질 정보와 개발 자료에 적용했다. 최종 test의 측정 시간 레이블은 잠금 상태를 유지한다.

## 완료 수와 보존 이력

| 구분 | 완료 묶음 | 원계획 측정 | 정확성 제외 | 수집 측정 |
|---|---:|---:|---:|---:|
| train | 20 | 21,600 | 0 | 21,600 |
| validation | 5 | 5,400 | 18 | 5,382 |
| test | 5 | 5,400 | 18 | 5,382 |
| 합계 | 30 | 32,400 | 36 | 32,364 |

측정 한 건은 형상·dtype·구현·회차 조합 하나다. 세 회차를 합친 분석 조건과 구분한다. 완료 자료는 v2 train 20묶음, v4 validation 5묶음과 test 3묶음, v5 test 2묶음으로 구성된다.

입력은 `/Users/studyhard/Desktop/Transfer/resume_full_20261108_105717_8tqhrkir`다. [raw](raw)에 ZIP 33개, 각 SHA-256 파일, campaign 이력 및 최종 요약 JSON을 합한 71개 파일을 동일 해시로 보존했다. 마지막 두 압축 해제 폴더도 각 ZIP과 일치한다. 압축 해제 폴더에만 있는 파일은 빈 실행 잠금 파일과 ZIP 해시가 일치하는 archive receipt다.

실패한 validation 첫 시도 272건, validation 둘째 시도 0건, test_04 첫 시도 30건도 각각 원본 ZIP으로 남겼다. 실패 시도까지 포함한 저장 이력은 32,666건이며, 완료 자료 32,364건과 구분한다.

## 무결성과 품질

ZIP 33개의 외부 SHA-256, CRC, 내부 파일 목록을 확인했다. 결과 파일 해시 61,401개와 내장·로컬 소스 해시 항목 4,982개가 일치한다. 이전 campaign 세 단계의 고정 해시와 승계 목록도 기존 검증 함수로 대조했다. 최신 압축 해제 폴더 두 개의 결과 해시 4,024개가 ZIP과 일치한다.

완료 30묶음의 개별 측정 품질 집계는 모두 PASS다. 각 묶음의 health 108측정이 통과했고 저장된 정확성 검사 문서 14,388개가 `passed=true`다. worker 완료 표식과 복원 명령 수락을 확인했다. v5 test_04에서는 교체한 Conv3 health 형상 `lf_bb6a106c15c571e5`가 실제 GPU 검사를 통과했다.

전체 campaign의 `gate=REVIEW`는 다음 네 묶음의 기존 이력을 유지한다.

| 묶음 | REVIEW 사유 | 분석 처리 |
|---|---|---|
| train_03 | FP32 FFN GELU의 회차 간 범위/중앙값 21.1131%, 기준 5% | 해당 구성 연산 조건과 공통 비교 형상 1개 제외 |
| train_09 | FP32 FFN GELU의 회차 간 범위/중앙값 5.21434%, 기준 5% | 해당 구성 연산 조건과 공통 비교 형상 1개 제외 |
| validation_01 | Conv3 FP32 `lf_ce6440e85a4add93`의 기존 정확성 실패 | 6구현 × 3회 = 18측정 제외 |
| test_04 | Conv3 FP32 `lf_dfce839bd3ca0ef5`의 기존 정확성 실패 | 6구현 × 3회 = 18측정 제외 |

train과 validation의 26,982개 저장 측정을 기존 v5 loader로 읽어 시간 형식·블록 통계·세 회차 변동·정확성 연결·입력 식별자를 검산했다. 집계 PASS 조건은 train 7,198개와 validation 1,794개, 합계 8,992개다. 별도 보류 목록에는 GELU REVIEW 2개와 시간값이 없는 정확성 제외 FAIL 6개를 남겼다.

센서 품질은 저장된 판정과 무결성을 확인했다. 센서 원시 구간의 독립 재계산은 이번 감사 범위에서 생략했다. 로그의 `sensor_resolution_verified=false`는 남아 있으며, 요청한 50ms 폴링과 센서 하드웨어 자체의 갱신 주기는 구분한다. 클럭 복원은 명령 수락까지 확인한 상태다.

## 공통 비교에 남는 형상

표준·분해·필요한 구성 연산이 모두 PASS인 동일 형상 집합을 모든 비교 방법에 적용한다.

| 레이어 | dtype | train | validation |
|---|---|---:|---:|
| Linear | FP16 / FP32 각각 | 200 | 50 |
| FFN | FP16 | 200 | 50 |
| FFN | FP32 | 198 | 50 |
| Conv1×1 | FP16 / FP32 각각 | 200 | 50 |
| Conv3×3 | FP16 | 200 | 50 |
| Conv3×3 | FP32 | 200 | 49 |

test의 최종 공통 PASS 집합은 모델 선택 동결 후 시험 단계에서 확정한다. 수집 정책상 Conv3 FP32 test 계획은 49형상이다.

## 수집 비용

실패 세 시도를 포함한 33개 cost ledger의 검증된 합계는 **133,109.178초, 약 36시간 58분 29초**다. 실패 시도 비용은 1,545.723초다. 이 값은 통제 수집·클럭 설정/복원·수집 판정에 든 시간의 합계이며, wrapper 준비·압축·시도 사이 대기 시간은 별도로 구분한다.

## 다음 단계 판단

현재 제외 정책과 품질 기준을 유지하고 v5의 학습·validation 분석으로 진행한다. 동결 protocol의 입력 47개, 공통 GEMM 가중치 6개와 현재 실행 환경 검증이 통과했다. prepared 폴더에는 기존 `frozen_protocol.json`이 있으며 준비 명령을 다시 실행할 필요가 없다.

후속 실행은 아래 명령을 사용한다. 이번 감사에서는 학습을 실행하지 않았다.

```bash
cd /Users/studyhard/Claude/자율연구
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-10_v5/pipeline.py train \
  --campaign analysis/layer_full_result_2026-11-10/raw \
  --output analysis/layer_full_fit_2026-11-10_v5/prepared_layer_full
```

FP16 MLP32·FP32 RF 공통 GEMM prior를 유지하고 RF·Ridge·MLP32·MLP64, 기존 예측 방식과 자료량을 비교한다. validation으로 모델을 선택한 뒤 가중치와 선택 기준을 동결하고 최종 test 평가로 넘어간다. FFN FP32의 최대 학습량과 Conv3 FP32의 validation 수는 실제 유지 형상 수를 보고한다.

전체 재수집보다 이 개발 분석을 먼저 진행하는 것이 적절하다. GELU 두 조건의 보완 측정은 개발 결과를 확인한 뒤 필요성을 판단할 수 있다. 현재 정확성 제외 두 형상과 품질 제외 이력은 후속 결과에도 함께 표시한다.

## 검증 근거와 재현

[archive_audit.json](archive_audit.json)은 원본 ZIP·승계·압축 해제 폴더 검증, [audit.json](audit.json)은 보존 파일 해시·완료 상태·개발 자료 품질·공통 형상·비용을 담는다. 감사 스크립트는 기존 수집·분석 helper를 재사용하며 원본 수집본과 동결 분석 코드는 유지했다.

```bash
python3 -B analysis/layer_full_result_2026-11-10/archive_audit.py
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_result_2026-11-10/audit.py
```

두 스크립트의 assert 검사와 전체 실행이 통과했다. test는 파일 바이트 해시·공개 완료/품질 메타데이터·파일명 목록만 검사했으며 시간 레이블 열람과 예측 성능 평가는 후속 시험 단계로 남겼다.
