# RQ3 레이어 학습과 네 학습기 비교

2026-10-28 후속: 실제 `core_dev` 반환 자료의 검산과 학습을 완료했다. [결과 보고서](레이어_개발실험_결과_2026-10-28.md)에 네 학습기의 960구성, 원 RF 예측 2,400개 일치와 선택 모델 87개 독립 검산, RQ2 구현 차이와 실제 비용을 정리했다. `runs/dev_fit_v1`은 기존 수집 전 RF protocol의 결과이고, `runs/matched_v1`은 반환 후 실제 정답 접근 전에 동결한 RF·Ridge·MLP32/64 확장 비교다. 아래의 준비·미실행 설명은 최초 작성 시점의 이력이다. 잠금 시험은 계속 닫혀 있다.

추가한 `compare_layers.py`는 기존 label loader, 정적 특징과 네 학습기 fit을 재사용한다. `selfcheck`는 합성 자료만 검사하고, `prepare --output NEW_RUN`은 기존 source 가중치의 해시·선택과 학습 규칙을 저장한다. `run --input CORE_DEV --output NEW_RUN`은 존재하는 protocol의 의존성 해시를 대조한 뒤 표준 자료만 읽는다. 같은 output을 재실행해 덮어쓸 수 없다. `summarize_results.py`는 실제 완료한 matched_v1의 지표와 가중치를 독립 검산하고 보고서·그림·비용을 만든다. RQ2 원자료 접근에는 선택 동결 표시와 별도 전체 검산 결과가 필요하다.

2026-10-28 통합 실행계획의 새 Linear/FFN 학습기를 준비했다. 실제 레이어 자료 학습과 새 GPU 실험은 실행하지 않았다. 기존 RQ1/RQ4의 고정 특징, RandomForest 설정, 오차 계산과 source A 가중치를 재사용한다. 환경은 `analysis/integrated_rq14_2026-10-28/.venv`다.

`frozen_fit_protocol.json`은 새 레이어 시간을 보기 전에 생성했다. train 30그룹, validation 10그룹, 두 레이어와 두 dtype, 세 seed와 k=5/10/20/30의 ID는 수집 패키지의 `rq3_training_subsets.json`에서 가져온다. static metadata, 학습기 코드, 비교군, source 6개 가중치와 패키지 release hash를 고정했다. `precheck_protocol_draft.json`은 검토 중 만든 미실행 초안이며 실행에는 사용하지 않는다.

학습기는 반환된 `core_dev` 결과 폴더의 manifest를 먼저 검사한다. 다른 phase이면 record 목록, hash와 내용을 읽기 전에 거부한다. manifest의 case ID, 형상, split, dtype, round와 source hash를 고정 패키지와 대조한다. 표준 `records/`의 480측정만 읽으며 `restricted_explicit/`은 목록에도 넣지 않는다. standard 품질 플래그, correctness와 회차 간 5% 기준을 모두 통과하고 세 회차가 완비되어야 fit을 시작한다. 문제가 있으면 형상을 조용히 제외하지 않고 중단한다. 원래 자료는 보존된다.

RQ3의 prior S_A는 해당 seed의 allocate A가 예측한 논리적 GEMM 시간을 더한 값이다. Linear는 (M,F,D) 하나, FFN은 (M,F,D)와 (M,D,F) 둘이다. H/C 모델이나 실측 구성 연산 시간은 입력에 넣지 않는다. 입력은 첫 GEMM의 정적 특징과 전체 GEMM FLOPs, GELU 원소 수다. 레이어와 dtype별로 따로 학습한다.

같은 k그룹에서 S_A, 상수 곱셈, D0, D+A와 잔차 R을 비교한다. D+A와 R은 같은 정적 특징과 log S_A를 받는다. 주 RF는 100 trees, depth 8, leaf 2이고 CPU 한 thread를 사용한다. 새 선택은 validation의 세 seed 평균 MAPE≤15%와 p90≤30%, 실패 0개를 만족하는 최소 실제 k다. S_A는 실제 k=0이며 validation 10그룹만 비용에 포함한다. 이 규칙으로 무보정 기준선을 k=5와 같은 비용으로 취급하지 않는다.

한 레이어와 한 dtype의 표준 레이블 비용은 3(k+10)측정이다. 두 레이어와 두 dtype를 함께 수집한 비용은 12(k+10)측정이고, 세 seed의 실제 train 합집합도 별도 계산한다. source 공통 준비비와 새 explicit 짝 수집비는 전체 연구 비용에서 별도 귀속한다. 모델 fit, prior 생성, 보정 예측과 전체 pipeline validation 추론시간을 나누어 저장한다. 각 구성에서 공유한 prior 시간을 전체 연구 wall time으로 중복 합산하지 않는다.

## 준비와 검사

다음 명령의 `prepare`는 이미 실행했다. 고정 protocol이 존재하면 덮어쓰기를 거부한다. 재동결이 필요하면 이유를 남기고 `--protocol`에 새로운 이름을 지정한다.

```bash
cd /Users/studyhard/Claude/자율연구
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/layer_transfer_fit_2026-10-28/fit_layers.py prepare
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/layer_transfer_fit_2026-10-28/selfcheck.py
```

synthetic selfcheck는 표준 480측정 fixture의 160조건 집계, 잠금 phase의 record 목록 접근 전 거부, explicit 읽기 차단, 잘못된 회차와 REVIEW 거부, source 합산과 비교 모델 계약을 확인한다. 실제 GPU correctness, 센서와 측정시간은 검증하지 않았다.

## `core_dev` 반환 후 실행

반환 ZIP을 보존하고 별도 폴더에 푼 뒤 `manifest.json`이 들어 있는 실제 run 폴더를 `LAYER_DEV_RESULT`에 지정한다. 아래 명령은 CPU 로컬 학습이다. 결과 폴더는 새 이름이어야 한다. 현재는 실제 반환 자료가 없어 이 명령을 실행하지 않았다.

```bash
cd /Users/studyhard/Claude/자율연구
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LAYER_DEV_RESULT='/absolute/path/to/returned/core_dev_run'
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/layer_transfer_fit_2026-10-28/fit_layers.py run \
  --input "$LAYER_DEV_RESULT" \
  --output analysis/layer_transfer_fit_2026-10-28/runs/dev_fit_v1
```

성공하면 validation 표와 예측, bootstrap, 비용, source 참조와 선택 가중치 12개, 실제 입력 hash를 저장한다. `selection_frozen.json`이 `RQ3_SELECTION_FROZEN_VALIDATION_ONLY` 상태를 기록한 뒤 RQ2 explicit 진단을 별도 검토할 수 있다. 이 파일은 잠금 시험을 열거나 자동으로 explicit 값을 읽게 하지 않는다. CLI에는 test/final 실행이 없다. 저장 모델은 이 작업에서 만든 source 가중치만 hash를 확인해 읽으며 반환 자료 안의 joblib은 사용하지 않는다.
