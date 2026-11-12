2026-11-09 레이어 확대 수집 중단 분석

`full_20261108_105717_8tqhrkir`는 train 20묶음을 완료한 뒤 validation 첫 묶음의 Conv3×3 FP32 정확성 검사에서 중단됐다. 사용자는 PC가 켜진 상태에서 프로그램만 멈췄다고 확인했다. 완료된 train 결과를 보존하고, 실패 형상의 수치 차이를 진단한 뒤 validation부터 이어가는 방향이 적절하다.

| 구역 | 완료 묶음 | 저장 원측정 | 판정 |
|---|---:|---:|---|
| train_01~20 | 20/20 | 21,600/21,600 | 개별 측정 전부 PASS. train_03·09의 GELU 회차 간 변동은 REVIEW |
| validation_01 | 0/1 | 272/1,080 | 저장된 272개 PASS, 정확성 검사에서 중단, 808개 미수집 |
| validation_02~05 | 0/4 | 0/4,320 | campaign에 시도 기록 없음 |
| test_01~05 | 0/5 | 0/5,400 | campaign에 시도 기록 없음 |

첨부 campaign 기준 완료는 30묶음 중 20개이며 저장된 측정은 21,872개다. 마지막 터미널 진행 표시 `Saved=269` 이후의 최종 저장 수는 ZIP의 `summary.json`과 원자료에서 확인한 272개다. Controlled collection wall 합계는 89,039.936초, 약 24시간 44분이다. train에 87,797.620초, 실패한 validation 시도에 1,242.317초가 기록됐다.

실패는 [worker.log](evidence/validation_01_attempt_001/worker/worker.log)의 `Correctness failed`에서 시작해 worker exit 1, 상위 wrapper의 `Worker exited 1`, campaign의 STOP으로 전파됐다. `RESTORE: COMMANDS ACCEPTED`는 종료 처리 중 GPU 그래픽·메모리 클럭 초기화 명령 두 개가 exit 0으로 끝났다는 뜻이다.

실패한 형상은 `lf_ce6440e85a4add93`, Conv3×3, FP32, 첫 회차의 측정 전 검사다. batch=8, Cin=303, Cout=463, H=20, W=12이며 출력은 888,960개다. 분해 경로는 unfold 후 GEMM(M=1,920, N=463, K=2,727)을 수행한다.

| 정확성 검사 | 결과 | 최대 절대 오차 | 최대 허용오차 비율 |
|---|---|---:|---:|
| 표준 Conv2d 대 CPU FP64, 256표본 | PASS | 1.536333676e-5 | 0.5179601 |
| unfold+GEMM 대 CPU FP64, 256표본 | PASS | 5.766813282e-7 | 0.0189074 |
| 표준 대 분해, 전체 출력 | FAIL | 4.094839096e-5 | 로그에 미기록 |

두 구현의 전체 유한성·출력 형상·연속 배치 검사는 통과했다. 전체 쌍 비교는 각 원소에 `abs(standard-decomposed) <= 2e-5 + 2e-4*abs(standard)`를 적용한다. 최대 절대 차이가 난 좌표와 실제 허용오차를 넘은 좌표는 다를 수 있다. [정확성 원본 JSON](evidence/validation_01_attempt_001/correctness/lf_ce6440e85a4add93_fp32_r0_before.json)은 전체 실패 좌표·개수·각 출력값을 저장하지 않아, 해당 위치에서 어느 경로가 FP64 기준을 얼마나 벗어났는지는 추가 재현이 필요하다.

현재 근거로 확정할 수 있는 수치 원인은 두 구현 사이의 차이가 전체 비교 기준을 넘었다는 것이다. cuDNN Conv와 GEMM의 알고리즘·누적 순서에 따른 반올림 차이가 후보에 해당한다. PyTorch도 수학적으로 같은 계산에서 연산 순서에 따라 부동소수점 결과가 달라질 수 있다고 설명한다. 이 일반적 설명만으로 이번 실패를 정상 오차로 판정할 수는 없으므로 실패 좌표의 FP64 검산이 필요하다. [PyTorch 수치 정확성 문서](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

기록된 환경은 RTX 4000 SFF Ada, PyTorch 2.14.0+cu132, cuDNN 92400이다. Matmul·Conv FP32는 모두 IEEE 설정이며, cuDNN은 benchmark=true, deterministic=false다. 재실행 시 autotuning이 다른 알고리즘을 고를 수 있어 단순 재개만으로 해결됐는지 판단하기 어렵다. [PyTorch cuDNN 재현성 설명](https://docs.pytorch.org/docs/2.14/notes/randomness.html#cuda-convolution-benchmarking)

train_01에 포함된 기존 FP16 FFN 회귀 probe는 원래 CUDA 입력 해시 대조와 수정 정확성 검사를 통과했다. 이번 실패 위치는 Conv3 FP32의 전체 쌍 비교다.

train의 REVIEW 두 건은 별도로 보존한다. 기준은 구성 연산의 세 회차 중앙값에 대한 `(최대-최소)/중앙값 > 5%`다.

| 묶음 | FFN 형상 ID | 구성 연산 | 회차별 중앙값(ms) | 회차 상대 범위 |
|---|---|---|---|---:|
| train_03 | lf_1df050dd847f7ac8 | FP32 GELU | 0.176639996 / 0.145691021 / 0.146586824 | 21.1131% |
| train_09 | lf_40b5206c5b6879dd | FP32 GELU | 0.007646034 / 0.007266775 / 0.007273369 | 5.21434% |

개별 원측정의 PASS와 회차를 합친 REVIEW는 서로 다른 검사다. 기존 분석 loader를 train 20묶음에 실행해 집계 레이블 7,198개 PASS와 GELU 두 개 제외를 확인했다. 학습 pipeline은 표준·분해·구성 연산이 공통 PASS인 형상을 비교하므로 FFN FP32 공통 학습 집합은 200개 중 198개가 된다. 원측정 여섯 개와 REVIEW 이력을 유지하고 재측정 대상으로 남긴다.

재현 진단에서는 같은 CUDA seed `1079309840`으로 입력을 만들고 저장된 SHA-256을 먼저 대조한다. input은 `4e2f93af17822df3b0c184889a37431e66ad673aeba72a1114fded85bc7461c4`, weight는 `452625a31d2ac7b9d36196cfb280e76c72f4cc94e04864315917a0bb731c1e82`다. 전체 쌍 비교의 실패 좌표·개수·최대 허용오차 비율과 해당 좌표의 두 출력값을 저장한 뒤 CPU FP64 직접 내적으로 검산한다. 이 결과로 검사 기준과 계산 경로 중 수정 대상을 결정한다.

현재 `collect_all.py --resume`는 완료된 train 20묶음을 해시·완료 메타데이터로 확인한 뒤 건너뛰고 `validation_01_attempt_002`에서 그 묶음의 1,080측정을 새로 시작한다. 기존 실패 시도의 272개는 보존된다. 동일 소스·seed로 재개하면 같은 검사에서 다시 멈출 수 있다. 소스를 수정한 실행본은 campaign의 release hash 대조에 걸리므로 별도 버전과 명시적인 결과 승계가 필요하다. 측정 보존본을 직접 덮어쓰는 방식은 피한다.

첨부된 정확한 ZIP 21개와 campaign을 `raw/`에 복사하고 SHA-256을 대조했다. ZIP CRC, 내부 결과 해시 40,788개, ZIP마다 release source 145개, campaign·로컬 v2·내장 source의 대응, 형상·원측정 ID와 누락·중복, 저장된 21,872개 timing 통계와 전후 정확성 링크를 검산했다. 모든 보존·검산 조건이 통과했다. 파일명 끝의 8자리는 생성 시 붙인 UUID 일부이며 SHA-256 접두사와 별개다.

21묶음의 환경 signature는 같았다. 저장된 센서 1,696,262표본에서 SM=720 MHz·메모리=6801 MHz였고 전체 온도는 31~66°C, 실패한 validation 시도는 46~60°C였다. 열 제한 상태·누적 카운터는 0으로 기록됐다. 전력 제한은 표본 상태가 Not Active여도 누적 카운터 증가가 있어 전체 구간의 비활성을 단정할 수 없다.

원본 ZIP·campaign과 검산 결과의 위치는 [audit.json](audit.json)에 기록한다. 검산은 `python3 -B analysis/layer_full_result_2026-11-09/audit.py`로 재실행할 수 있다. 실패 증거를 별도로 추출한 파일은 [해시 목록](evidence/extracted_sha256.json)과 대조할 수 있다. 기존 연구 상태는 [AcEVault 실행 기록](/Users/studyhard/Claude/AcEVault/wiki/research-2학기-y축예측-계획.md)을 참조했다.

이번 분석 범위는 반환 자료와 코드 검산이다. GPU 재현·수집 재개·모델 학습·최종 시험 평가는 후속 작업이며, 실패 좌표 검산 전 허용오차를 완화하면 실제 수치 오류까지 통과시킬 위험이 있다.
