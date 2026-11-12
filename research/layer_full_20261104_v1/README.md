# 레이어 전체 확대 수집 — 2026-11-04

Linear, exact GELU FFN, Conv 1×1, Conv 3×3을 각각 **새 형상 300개**로 측정한다. 종류마다 train 200 / validation 50 / sealed test 50이다. 전체 forward와 실제 중간 텐서를 사용하는 구성 연산을 FP16·IEEE FP32로 세 회차 수집한다.

## 수집량

| 레이어 | 표준·분해 전체 forward | 별도 구성 연산 | 합계 |
|---|---:|---:|---:|
| Linear | 3,600 | 0 | 3,600 |
| FFN | 3,600 | GEMM1 / GELU / GEMM2 5,400 | 9,000 |
| Conv 1×1 | 3,600 | input pack / mm / output pack 5,400 | 9,000 |
| Conv 3×3 | 3,600 | unfold / input pack / mm / output pack 7,200 | 10,800 |
| 합계 | **14,400** | **18,000** | **32,400** |

Linear의 분해 경로는 GEMM 한 번이므로 같은 기록을 구성 연산 정답으로 사용한다. 한 측정은 안정화와 반복 수 보정 뒤 일곱 시간 블록으로 구성된다. 본 시간 블록은 총 226,800개다. 같은 기저 형상을 공유하는 Linear·FFN과 Conv 1×1·3×3을 각각 한 그룹으로 묶으므로 독립 기저 그룹은 600개다.

새 수집은 이전 자료와 겹치지 않는 별도 코호트다. 기존 실행본·원자료·split은 그대로 보존한다. 이전 960측정의 재수집이나 미수집 984개의 단순 실행과 구분되는 확대본이다.

## 형상과 분할

`groups.json`, `cases_by_phase.json`, `selection_audit.json`에 실제 형상과 예산을 고정했다. 생성기는 시간 정답을 읽지 않는다. core 300그룹과 Conv 300그룹에서 정렬/비정렬 × 연산량 3구간 × 종횡비 2구간에 각각 25그룹을 배치한다. 연산량 경계는 1억·10억 FLOP, 종횡비 경계는 8이다. core는 첫 GEMM, Conv는 3×3의 구성 GEMM으로 구간을 정한다.

core의 M/D/F는 64~4096, F/D는 약 2 또는 4다. Conv는 batch 1/2/4/8/16, H/W 8~64의 후보, Cin/Cout 16 이상을 사용하며 환산된 M/N/K를 4096 이하로 제한한다. 비정렬은 기본 채널·폭의 ±1 변형이다. Conv 기저 300그룹 중 179개는 어느 구성 GEMM에 축 크기 64 미만을 포함하므로 기존 GEMM 학습 범위 밖의 적용 조건으로 구분한다. 사전에 정의한 형상 구간을 균등하게 조사한다.

기존 GEMM·대조·레이어 목록과 추가 GEMM 10,000형상까지 합친 16,607개 구성 형상을 보호했다. 동일 또는 한 축 ±8 이내 충돌을 제외하고, 새 그룹 사이에도 같은 검사를 적용한다. 같은 기저의 정렬/비정렬 변형도 중복 선택하지 않는다. 신규 구성 GEMM과 기존 GEMM 학습 자료의 직접 중복을 제외한 조건이다.

`training_subsets.json`에는 각 계열의 세 seed와 중첩 k=5/10/20/30/50/100/200 학습 그룹을 고정했다. 두 레이어·dtype·구현·반복은 모두 동일 split에 속한다.

## 측정 조건

기존 `layer_transfer_20261028_v1`의 `layers.py`와 측정·환경·센서·클럭 복원 의존성을 바이트 그대로 재사용한다. RTX 4000 SFF Ada, SM 720 MHz / memory 6801 MHz / 70 W, 기존 `ai_env`와 Python 3.12, CUDA_HOME `/usr/local/cuda-13.4`를 유지한다. 보존된 PyTorch runtime은 13.2이며 실제 GPU UUID·소프트웨어 서명은 기존 기준과 실행 중 대조한다.

ordinary inference forward, bias 없음, exact GELU, contiguous NCHW/2D, FP16·IEEE FP32 조건이다. 입력과 weight는 GPU에 상주하며 prepared weight 배치는 측정 전에 준비한다. Conv의 unfold와 input/output pack은 분해 전체 forward에 포함된다. GPU 시간은 기존 CUDA Event 반복 묶음 타이머로 측정한다. 각 묶음 전역 예열 180초, 측정마다 event 시간 기준 3초 안정화, 목표 100ms의 7블록, 3회차와 기존 IQR·drift·회차 변동 5% 기준을 사용한다.

표준·분해 전체 forward를 먼저 측정하고 구성 연산의 중간 버퍼를 만든다. FFN은 실제 첫 GEMM 출력과 GELU 출력, Conv는 실제 unfold/pack 중간 텐서를 고정해 단독 호출한다. 구성 연산의 연결 출력과 분해 전체 출력을 대조하고 tensor shape·stride·값 해시를 기록한다. 전체 경로의 CPU FP64 표본 참조, 전체 유한값 검사와 표준/분해 비교도 유지한다. 구성 연산 단독 시간은 각각의 고정 중간 상태에서 얻는 진단값이며, 그 합을 전체 경로의 인과 기여율로 해석할 때는 캐시·할당·연결 실행 차이를 고려해야 한다.

## 30개 묶음과 복구

한 묶음은 core 10그룹 + Conv 10그룹으로 네 종류 각 10형상, 총 1,080측정이다. train 20묶음 → validation 5묶음 → sealed test 5묶음 순서로 수집한다. 각 묶음은 대표 네 종류의 건강 검사부터 시작한다. 건강 검사 REVIEW/FAIL, 정확성 실패, 센서·클럭·전력·열 문제나 환경 불일치에서는 수집을 중단한다. 건강 검사 통과 후 완료된 묶음의 REVIEW는 그대로 표시하고 다음 묶음으로 진행한다. 허용오차를 자동으로 넓히지 않는다.

`collect_all.py`는 묶음별 새 시도 폴더를 만들고 ZIP·SHA-256을 보존한다. 재실행은 `--resume`으로 같은 campaign을 지정한다. source hash와 결과 ZIP 해시가 일치하고 COMPLETE로 끝난 묶음은 건너뛴다. 중단된 묶음은 새 시도 폴더에서 세 회차 전체를 다시 측정하고, 이전 부분 결과는 보존한다. 한 묶음 안에서 서로 다른 시도의 원측정을 합치지 않는다. 손상된 완료 ZIP이나 바뀐 소스는 재개를 차단한다.

묶음마다 5 GiB 이상의 여유 디스크를 요구한다. 전체 안정화와 본 시간 블록만 약 33.3 GPU 시간이며, 30회 예열·보정·정확성·메모리·포장 시간이 추가된다. 따라서 GPU 전용 시간을 최소 하루 반 이상 확보하고 실제 첫 묶음의 비용 기록으로 남은 시간을 산정한다. 다른 GPU 측정·학습과 동시에 실행하지 않는다.

## 시험 자료 보관

사용자의 전체 수집 요청에 따라 새 test 50그룹도 마지막에 수집한다. 기본 단일 묶음 명령은 train이며, test 묶음에는 `--collect-sealed-test`를 명시한다. 전체 실행기는 이 플래그를 자동으로 전달한다. 기존 다른 실험의 잠금 시험은 읽거나 수집하지 않는다.

test는 별도 `test_*` 시도 폴더·ZIP에 저장된다. 공개 `summary.json`, `health_check.json`, `collection_summary.json`은 품질·누락·개수만 보고하고 레이어 시간·비율을 내보내지 않는다. 고정 품질 검사는 원시 시간을 사용한다. 예측 모델 평가와 학습은 실행하지 않는다. 최종 모델·규칙을 확정할 때까지 test의 `records/`와 `restricted_explicit/` 시간값을 열람하지 않는다. 이 분리는 실수 방지용이며 OS 권한 격리는 적용하지 않는다.

## 연구 PC 실행

ZIP과 `.zip.sha256` 두 파일을 `/home/eu/gemm-research/`에 옮긴 뒤, 기존 tmux 세션에서 일반 사용자 eu로 실행한다. 최초 실행:

```bash
(
set -euo pipefail
cd /home/eu/gemm-research
sha256sum -c layer_full_20261104_v1_source.zip.sha256
test ! -e layer_full_20261104_v1
unzip -n layer_full_20261104_v1_source.zip
python3 -B layer_full_20261104_v1/selfcheck.py
python3 -B layer_full_20261104_v1/collect_all.py
)
```

출력된 `CAMPAIGN_DIR=` 경로를 보관한다. 터미널 연결을 유지하려면 tmux를 사용하고, 분리는 Ctrl+B를 누른 뒤 D다. 중단 후 재개는 출력된 실제 campaign 경로로 실행한다:

```bash
python3 -B /home/eu/gemm-research/layer_full_20261104_v1/collect_all.py --resume /home/eu/gemm-research/layer_full_20261104_v1/runs/full_ACTUAL_RUN_NAME
```

모든 스크립트 출력은 ASCII 영어다. 종료 시 클럭 복원과 결과 포장을 수행한다. campaign 폴더의 `campaign.json`은 시도 이력, `collection_summary.json`은 전체 수집 완료 여부다. 반환 자료는 각 묶음의 최신 완료 시도 ZIP·SHA-256과 이 두 JSON이며, 중단·실패 시도도 보존한다. 기존 패키지를 덮어쓰거나 CUDA/Conda를 재설치하지 않는다.

## 로컬 검증

`python3 -B selfcheck.py`는 목록 재현·분할·보호 형상·예산, 원 의존성 해시, 1,080개 합성 기록의 실제 품질 분석, 시험 접근 차단, 중단/재개·REVIEW 유지·손상 ZIP 및 변경된 완료 메타데이터 거부를 검사한다. 결과 ZIP 포장과 종료 신호 전달·자식 프로세스 정리도 검사한다. `python selfcheck.py --numeric`은 설치된 PyTorch의 CPU에서 두 dtype·정렬/비정렬·네 종류의 16개 수치 사례와, CUDA 타이머·장비 제어를 대역으로 바꾼 실제 worker의 108개 기록을 검사한다. 검증 기록은 `validation.json`에 저장한다. 실제 CUDA correctness·품질과 서버 환경은 각 묶음의 시작 검사로 확인한다.
