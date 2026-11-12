# FFN 정확성 검사 수정과 전체 재수집

2026-11-08 개정본. 사용자 변경 요청에 따라 `train_01`부터 30묶음 전체를 새로 측정한다. 기존 v1 실행본·결과·실패 기록은 그대로 보관한다.

## 오류와 수정

기존 FFN 기준값은 입력·가중치를 FP64로 변환한 뒤 `GELU(XW1)W2` 전체를 FP64로 계산했다. 실제 FP16 FFN은 첫 GEMM과 GELU의 출력을 각각 FP16 텐서로 저장한다. 따라서 중간 반올림이 둘째 GEMM에 전파되며, 특히 0에 가까운 출력에서 기존 단일 연산 수준의 허용오차를 넘을 수 있다. [PyTorch 2.14 수치 정확성 설명](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)은 저정밀 계산과 계산 순서에 따른 차이를 설명한다.

새 기준값은 CPU FP64 연산을 사용하되 GEMM1과 GELU 뒤에 FP16 저장 반올림을 반영한다. FP16 `atol=0.001`, `rtol=0.005`, 전체 출력의 두 경로 비교, 유한성·형상·배치 검사와 실패 시 중단은 유지한다. 기존 수학적 FP64 오차도 `mathematical_fp64_diagnostic`에 함께 저장한다. Linear·Conv·FP32 기준값, 실제 timed forward·구성 연산, 타이머, 데이터 분할, 입력 seed, 하드웨어 조건은 기존과 같다.

반환된 실패 사례는 `lf_76b19fecaab55272`, rows=143/input_width=1631/hidden_width=3263, FP16이다. 표준·분해 경로에서 각 256개 표본 중 (88,1492) 한 점이 기준을 넘었다. 허용오차 대비 최대 비율은 1.04005/1.10435였다. CPU에서는 같은 형상과 별도 고정 seed의 회귀 사례에서 기존 실패(1.05697), 수정 통과(0.07403)를 재현했다. CPU와 CUDA의 난수열은 다르므로 원래 CUDA 입력에 대한 확인은 아래 자동 probe가 수행한다.

`train_01` worker는 예열 전에 원래 실패한 입력·가중치를 다시 만들고 반환 자료의 SHA-256과 대조한다. 수정한 검사에 통과하면 `FFN REFERENCE PROBE PASS`를 출력하고 수집을 진행한다. 보고서는 `ffn_reference_probe.json`에 저장된다. 이 검사가 실패하면 해당 시도 결과와 ZIP을 보존하고 중단한다.

## 연구 PC 실행

이 폴더 전체를 `/home/eu/gemm-research/layer_full_20261108_v2`에 둔다. 일반 사용자 `eu`의 tmux에서 실행한다.

```bash
(
set -euo pipefail
cd /home/eu/gemm-research/layer_full_20261108_v2
python3 -B selfcheck.py
python3 -B collect_all.py
)
```

`collect_all.py`는 v2의 `runs/full_<timestamp>_<id>`에 새 캠페인을 만든다. 첫 묶음 수집에 앞서 원래 실패한 FFN 입력의 해시와 수정된 정확성 검사를 확인한다. `FFN REFERENCE PROBE PASS` 뒤에는 원래 예열·health 검사·본 측정이 진행된다. 위 명령을 다시 실행하면 새 캠페인이 생기므로, 중간에 중단된 v2 실행을 이어갈 때는 출력된 `RESUME:` 명령을 사용한다.

```bash
python3 -B /home/eu/gemm-research/layer_full_20261108_v2/collect_all.py --resume /home/eu/gemm-research/layer_full_20261108_v2/runs/full_ACTUAL_RUN_NAME
```

전체 계획은 train 20 → validation 5 → test 5, 30묶음·32,400측정이다. 네 종류 각 300형상(200/50/50), 두 dtype, 세 회차를 유지한다. 표준/분해 전체 forward 14,400개와 구성 연산 18,000개다. 새 test는 별도 수집·품질 검사하며 모델 평가와 시간값 열람은 별도로 통제한다. Ubuntu 터미널 출력은 ASCII 영어다. 첫 실행 probe가 실패하면 그 시도의 `ffn_reference_probe.json`과 결과 ZIP을 확인한다.

## 검증과 후속 분석

`selfcheck.py`는 형상 보호·분할·품질 분석·시험 접근·중단 재개·클럭 복원 경로 및 timed 함수/허용오차의 보존을 검사한다. `--numeric`은 PyTorch CPU의 16개 수치 사례, 실제 worker의 CUDA 제어 대역 108기록, 반올림 실패 재현과 오염 출력·NaN 거부, probe의 입력 불일치·실패 중단을 추가 검사한다. `validation.json`에 로컬 증거와 CUDA 확인 상태를 기록한다.

후속 분석은 `analysis/layer_full_fit_2026-11-08_v2`를 사용하며 v2 결과만 읽는다. 이전 분석 준비본과 동결된 protocol은 그대로 보존한다. 새 분석 protocol은 `prepare`로 별도 생성한다. 새 GPU 결과의 학습과 최종 시험은 이 수정 작업에 포함되지 않는다.
