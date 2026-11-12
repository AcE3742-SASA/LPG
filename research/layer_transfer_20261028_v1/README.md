# 레이어 재사용 추가 실험 실행본

2026-10-28 통합 개정계획의 RQ2/RQ3 수집 준비본이다. 기존 `layer_compare_v2`의 타이머, IEEE 정밀도, 정확성 검사, GPU 잠금, 클럭 설정·해제, 센서, 품질 판정과 환경 래퍼를 새 폴더에 복사했다. 원본 실행본과 측정 자료는 변경하지 않았다. CUDA 실행은 아직 하지 않았다.

## 목록과 접근 제한

| phase | 범위 | 측정 / 본 시간 블록 |
|---|---|---:|
| `core_dev` 기본값 | 30 train + 10 validation 그룹, Linear/FFN, 두 dtype와 두 구현, 3회차 | 960 / 6,720 |
| `core_locked` 잠금 | 20 test 그룹. 현재 실행 금지 | 480 / 3,360 |
| `conv_confirm` 선택 | 새로운 기저 10그룹의 Conv 1×1/3×3, 모두 확인용 | 240 / 1,680 |
| `conv_diagnostic` 선택 | 기존 개발 Conv 1×1 4형상과 3×3 4형상, 전체 경로와 구성 연산 | 264 / 1,848 |

필수 전체는 60그룹, 1,440측정이다. 현재 기본 실행에서는 잠금 test의 시간 수집도 하지 않는다. 각 core split은 정렬과 비정렬을 절반씩 포함한다. 모든 M,D,F는 4,096 이하이고 F/D는 약 2 또는 4다. 정적 FLOP 1억 이하를 small, 그 위를 large로 고정했다. train은 small 14/large 16, validation은 4/6, test는 10/10이다. 이 경계는 실측 시간으로 정하지 않았다.

시간 없는 기존 manifest의 형상, 비본측정 형상, 기존 개발 160형상, 기존 layer_spec validation/test의 구성 GEMM을 보호했다. 보호 집합은 2,991개의 M/N/K다. 정확 일치 또는 한 축 ±8의 충돌이 없고, 같은 기저 anchor의 변형을 다른 그룹에 중복 채택하지 않았다. 120개의 고유 core 구성 GEMM 가운데 A 학습 후보와 정확히 같은 것은 1개, 새 M/N/K는 119개다. 원래 보호 ID나 split은 해제·재배정하지 않았다. `groups.json`, `cases_by_phase.json`, `selection_audit.json`, `protected_shapes.json`에 목록과 근거가 있다. 생성기는 `build_manifest.py`이며 시간 CSV나 SQLite를 읽지 않는다.

RQ3의 학습 표본도 새 시간값을 보기 전에 `rq3_training_subsets.json`에 고정했다. seed 20261028/20261029/20261030마다 train 30그룹 안에서 k=5/10/20/30의 중첩 집합을 만든다. 정렬·비정렬과 정적 small·large bucket 안에서 seed hash로 순서를 정하고 bucket을 번갈아 선택한다. k=5는 각 분류가 3:2, k=10/20은 각각 반반이며 k=30은 전체 train 분포를 유지한다. 두 레이어, dtype, 구현과 비교 방법 모두 같은 그룹 ID를 쓴다. `groups.json`의 `nested_train_position`은 수집용 순서이며 학습 표본 순서로 사용하지 않는다. 수집기·평가 계약·표본 ID를 준비한 단계이고 RQ3 학습기는 새 train/validation 정답이 생긴 뒤 이 ID를 사용해 구현한다.

기본 실행의 표준 경로 정답은 `records/`, explicit 및 선택 구성 연산은 `restricted_explicit/records/`에 분리한다. 전체 비교의 시간·비율은 `restricted_explicit/comparison.json`과 CSV에만 쓴다. 공개 `summary.json`과 `health_check.json`은 시간값 없이 품질과 누락을 보고한다. RQ3의 모델·k 선택을 동결하기 전에는 `restricted_explicit/`을 진단하거나 학습에 사용하지 않는다. 폴더 분리는 실수 방지 장치이며 사용자 계정의 파일 접근권한 격리는 아니다.

## 측정 계약

기존 환경은 RTX 4000 SFF Ada, SM 720 MHz, memory 6801 MHz, 70 W, 기존 Conda `ai_env`와 Python 3.12다. 래퍼의 CUDA 13.4 확인은 `/usr/local/cuda-13.4`의 nvcc toolkit에 대한 검사다. 보존된 PyTorch CUDA runtime 13.2와 구분하며 PyTorch를 13.4 빌드로 교체하지 않는다. 실제 서버의 환경 서명과 GPU UUID가 보존 기준과 다르면 실행을 중단한다. 장비·기존 GPU 프로세스 확인은 실행 시 다시 수행한다. 설치와 CUDA/Conda 경로를 변경하지 않는다.

ordinary inference 호출, bias 없음, FP16과 IEEE FP32, exact GELU를 유지한다. 전역 예열 180초, 측정별 3,000 event-ms 안정화, 목표 100 ms의 7블록이다. core에서는 train의 대표 Linear·FFN 각 한 조건의 24측정을 먼저 수행하고 HEALTH PASS 뒤 나머지로 진행한다. 선택 Conv는 각 커널 한 조건부터 확인한다. 정확성, 센서 누락·클럭·전력, IQR·drift·회차 간 5% 기준과 기본 클럭 해제 검사를 유지한다. REVIEW에서 자동으로 허용오차를 넓히지 않는다.

Conv 구성 진단은 원래 입력과 실제 explicit 중간 tensor에서 별도 연산 함수를 준비한다. 1×1은 input pack/mm/output pack 3개, 3×3은 unfold/input pack/mm/output pack 4개다. 전체 두 경로 96측정과 구성 168측정을 합쳐 264측정이다. 고정 중간 tensor의 형상·stride·값 해시를 저장하고 연결 결과가 원래 explicit 경로와 정확히 일치하는지 실행 중 검사한다. 단독 호출 시간의 합을 연결 경로의 인과 기여율로 해석하지 않는다. 새 Conv 확인은 기저 10그룹이며 독립 20그룹으로 세지 않는다.

`cost_ledger.json`에는 통제 수집·클럭 설정/해제·집계의 실제 wall time과 시도한 예산을 저장한다. 환경 래퍼의 비용은 launcher 로그에서 별도로 계산한다. 각 record의 준비·사전 정확성 비용은 두 구현에 중복 기록되므로 case/dtype/round 단위로 한 번만 합산한다. 표준과 explicit은 함께 수집하므로 전체 연구 비용에 둘 다 포함한다.

## 연구 PC 실행 명령

배포 ZIP과 같은 이름의 `.zip.sha256`을 `/home/eu/gemm-research/`에 전송한다. 다음 명령의 ZIP 이름은 같이 제공한 실제 파일명으로 바꾼다. 새 폴더가 이미 있으면 멈추고 내용을 보존한다.

```bash
(
set -euo pipefail
cd /home/eu/gemm-research
sha256sum -c layer_transfer_20261028_v1_source.zip.sha256
test ! -e /home/eu/gemm-research/layer_transfer_20261028_v1
unzip -n layer_transfer_20261028_v1_source.zip
python3 -B /home/eu/gemm-research/layer_transfer_20261028_v1/selfcheck.py
bash /home/eu/gemm-research/layer_transfer_20261028_v1/launch.sh core_dev
)
```

ZIP 안의 최상위 폴더명은 `layer_transfer_20261028_v1`이다. 기존 터미널 또는 tmux에서 일반 사용자 eu로 실행한다. launcher가 기존 `ai_env`를 활성화하고 환경을 확인한다. 다른 측정·학습과 동시에 실행하지 않는다. 기본 명령은 `bash .../launch.sh`로 생략해도 `core_dev`다. 선택 실험은 필요성이 결정된 뒤 다음 명령을 사용한다.

```bash
bash /home/eu/gemm-research/layer_transfer_20261028_v1/launch.sh conv_confirm
bash /home/eu/gemm-research/layer_transfer_20261028_v1/launch.sh conv_diagnostic
```

잠금 test 명령은 최종 설정·모델·k와 코드 해시를 동결하고 사용자가 별도로 허용한 뒤에만 사용한다. 현재는 실행하지 않는다.

```bash
bash /home/eu/gemm-research/layer_transfer_20261028_v1/launch.sh core_locked --acknowledge-locked-test-release
```

명시 플래그가 없는 `core_locked`는 verify/run/worker/analyze/pack에서 거부한다. 기본 phase에서 다른 phase의 결과를 analyze 또는 pack하려 해도 manifest phase를 검사한 뒤 record 열람·해시·압축 전에 거부한다. 잠금 자료의 시간은 이 준비 작업에서 수집·조회하지 않았다.

## 보존과 중단 복구

매 실행은 이 폴더의 `runs/<phase>_<시각>_<고유값>/`에 새로 저장한다. source와 의존성 사본, manifest, 명령, 환경·센서·실패, correctness, tensor 해시와 결과 해시를 보존한다. 종료·중단 시 결과 ZIP과 SHA-256을 같은 runs 안에 만들고 `RESULT_ZIP=`을 출력한다. 배포 ZIP과 결과 ZIP을 `/home/eu` 바로 아래에 만들지 않는다.

실패 뒤에는 원인을 기록하고 같은 phase를 새 run에서 다시 실행한다. 기존 manifest가 있는 디렉터리에는 재개·덮어쓰기하지 않는다. 부분 run은 보존하고 새 run과 합쳐 3회차 완료로 간주하지 않는다. 정상 종료 뒤 재집계는 해당 phase를 명시한다.

```bash
python3 -B /home/eu/gemm-research/layer_transfer_20261028_v1/compare.py analyze --phase core_dev --output /home/eu/gemm-research/layer_transfer_20261028_v1/runs/RETURNED_RUN_NAME
```

## 로컬 확인과 남은 검증

`selfcheck.py`는 목록 재현, 30/10/20과 정렬·작업량 분포, 보호/인접 충돌, 잠금 거부, 예산, 실제 보존 telemetry window를 이용한 합성 960/264 record 집계와 실패 조건을 검사한다. 본측정이 없는 상태의 합성 검사다. `compare.py verify`는 실행본과 보존 의존성의 해시도 확인한다. 새 Python과 shell의 문법, 기본/선택 phase와 잠금 명령 차단은 로컬에서 확인했다.

CPU PyTorch 수치 검증과 새 CUDA 실행은 하지 않았다. 원래 레이어 경로·정확성 함수는 바이트 그대로 재사용하며 새 Conv 구성 진단에는 실행 중 연결 결과 assert를 둔다. 새 형상에서 correctness와 시간/센서 품질은 실제 예비 실행으로 확인해야 한다. 사용되지 않는 과거 launch/progress 파일은 원래 의존성 해시를 위해 보존되지만 실행하지 않는다. 실제 호출하는 새 launcher·래퍼·측정 진행 출력은 ASCII 영어다.
