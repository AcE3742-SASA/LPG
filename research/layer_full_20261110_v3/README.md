# 기존 train 20묶음을 유지하는 validation 재개본

`layer_full_20261110_v3`는 반환 자료로 검산한 v2 캠페인 `full_20261108_105717_8tqhrkir`를 이어받는다. 원래 train 20묶음·21,600측정과 실패한 validation 첫 시도의 ZIP을 새 캠페인에 복사하고, 각 ZIP의 고정 SHA-256과 원래 campaign을 검증한다. 원래 폴더·측정 소스·ZIP·REVIEW 판정은 보존된다.

Conv3 FP32에서 두 구현 사이의 전체 비교가 실패하면, 실패한 모든 좌표를 실제 입력·가중치의 CPU FP64 직접 내적으로 검산한다. 표준 Conv2d와 unfold+GEMM 각각이 기존 `atol=2e-5`, `rtol=2e-4`를 만족할 때만 통과한다. 기존 표본·유한성·형상 검사를 유지하며 `full_pair_passed=false`와 새 `full_pair_adjudication`을 함께 저장한다. 입력 seed, timed forward·구성 연산, cuDNN 설정, 형상·분할·타이머·클럭·품질 기준은 v2와 같다.

## 연구 PC 실행

새 폴더 전체를 `/home/eu/gemm-research/layer_full_20261110_v3`에 둔다. 기존 v2 캠페인 폴더에는 `campaign.json`과 반환 ZIP 21개가 그대로 있어야 한다. 일반 사용자 `eu`의 기존 tmux에서 실행한다.

```bash
(
set -euo pipefail
cd /home/eu/gemm-research/layer_full_20261110_v3
python3 -B selfcheck.py
python3 -B collect_all.py --import-campaign /home/eu/gemm-research/layer_full_20261108_v2/runs/full_20261108_105717_8tqhrkir
)
```

새 캠페인은 `/home/eu/gemm-research/layer_full_20261110_v3/runs/resume_full_20261108_105717_8tqhrkir`다. 같은 import 명령을 다시 실행해도 같은 새 캠페인을 이어간다. train 20묶음은 `SKIP INHERITED`로 표시하고 `validation_01_attempt_002`부터 1,080측정을 새로 진행한다. 이후 validation_02~05, test_01~05를 순서대로 수집한다. test는 별도 수집·품질 검사만 하며 최종 모델 평가는 기존 잠금 절차를 따른다.

validation_01 시작 전 별도 프로세스가 원래 실패한 Conv FP32 입력·가중치의 SHA-256과 정확성을 확인한다. 별도 프로세스를 사용해 사전 검사의 cuDNN autotuning cache가 본 worker에 남지 않게 했다. `CONV REFERENCE PROBE PASS`가 뜨면 본 worker가 기존 예열·health 검사·수집을 시작한다. 이 사전 검사 비용은 해당 시도의 수집 wall에 포함된다.

```bash
python3 -B /home/eu/gemm-research/layer_full_20261110_v3/collect_all.py --resume /home/eu/gemm-research/layer_full_20261110_v3/runs/resume_full_20261108_105717_8tqhrkir
```

다시 중단되면 현재 시도의 ZIP과 `conv_reference_probe.json`, `worker/conv_probe.log`, `worker/worker.log`를 보존한다. FP64 직접 검산까지 기존 기준을 넘는 경우에는 실패 좌표·두 출력·FP64 정답·허용오차 비율을 저장하고 중단한다. 실제 원래 CUDA 입력의 통과 여부는 연구 PC의 자동 사전 검사에서 확인한다.

## 결과 보존과 후속 분석

기존 train_03·09의 FP32 FFN GELU 회차 변동 REVIEW를 유지한다. 두 조건은 기존 공통 PASS 정책에 따라 분석에서 제외한다. train의 재수집과 원래 실패한 validation 시도의 부분 측정 병합은 수행하지 않으며, 실패 시도의 비용도 이력에 포함한다.

후속 분석은 `analysis/layer_full_fit_2026-11-10_v3`를 사용한다. 고정된 v2 train과 v3 validation/test를 각 release의 소스·protocol·원본 ZIP 해시로 구분해 검증하고, 기존 FP16 MLP32·FP32 RF GEMM prior와 모델 선택 규칙을 유지한다. 캠페인 폴더 전체에서 ZIP·SHA sidecar·`campaign.json`·`inherited_campaign.json`을 함께 반환한다. 개발 구역 완료 후 새 분석 README의 명령을 사용한다.

[검증 기록](validation.json), [고정된 승계 대상](inheritance.json), [이전 반환 분석](../analysis/layer_full_result_2026-11-09/README.md)에 근거를 남겼다. 일반적인 부동소수점 연산 순서 차이는 [PyTorch 문서](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)를 참고했다. 실제 정확성 판정은 기존 FP64 오차 기준을 사용한다.
