# 완료 28묶음 보존 · test_04 정확성 제외 · v5 재개

v4 `test_04_attempt_001`은 health 첫 회차에서 Conv3 FP32 `lf_dfce839bd3ca0ef5`의 사전 정확성 검사에 실패했다. 30건이 저장됐고 복원 명령은 수락됐다. 전체 출력의 불일치 186좌표를 원래 FP64 정답으로 검산한 결과 표준 Conv2d는 183좌표에서 실패했고 unfold+GEMM은 186좌표 모두 통과했다. 표준 출력의 최대 허용오차 비율은 2.0881282000588635다.

사용자가 선택한 기존 정책을 적용해 이 FP32 조건만 제외한다. `atol=2e-5`, `rtol=2e-4`, timed 연산, 정밀도, 클럭, 품질 기준은 유지한다. 이전 validation_01의 제외 조건도 유지한다. 제외 근거와 입력 식별자를 해시로 고정했으며 다른 정확성 실패는 로그를 저장하고 중단한다.

## 연구 PC 실행

이 폴더 전체를 `/home/eu/gemm-research/layer_full_20261110_v5`에 둔다. 기존 v4 캠페인의 폴더와 ZIP, SHA sidecar, 세 campaign JSON을 유지한다. 일반 사용자 `eu`의 tmux에서 실행한다.

```bash
(
set -euo pipefail
cd /home/eu/gemm-research/layer_full_20261110_v5
python3 -B selfcheck.py
python3 -B collect_all.py --import-campaign /home/eu/gemm-research/layer_full_20261110_v4/runs/resume_full_20261108_105717_8tqhrkir
)
```

v5 importer는 사용자가 보낸 v4 campaign 상태를 고정 해시로 확인한다. train 20묶음, validation 5묶음, test_01~03의 완료 28묶음과 실패 시도 3개를 새 캠페인으로 복사한다. ZIP은 총 31개다. 원래 v4 폴더와 소스는 보존하며 부분 실패 측정을 새 결과에 합치지 않는다.

기존 22개 원본과 새 test_04 실패 ZIP은 전달받은 고정 해시로 검증한다. 새로 완료된 v4 ZIP 8개는 연구 PC에서 SHA sidecar, 결과 파일 전체 해시, 내장 소스, 원래 protocol, 완료 판정, 정확한 측정 파일 목록을 검증한 뒤 새 campaign에 해시를 고정한다. 검증할 때 sealed test의 시간값은 해석하지 않는다. 어떤 파일이 빠지거나 검사가 실패하면 수집 시작 전에 중단한다.

train 20묶음은 `SKIP INHERITED`, 새 완료 8묶음은 `SKIP INHERITED V4`로 표시한다. 첫 새 실행은 **test_04_attempt_002**, 목표는 **1,062측정 + 18제외**다. 이어서 test_05의 1,080측정을 수집한다. 전체 계획은 **32,364측정 + 제외 36건 = 원계획 32,400건**이다. 기존 REVIEW 판정도 보존한다.

## health 검사와 품질 판정

이번 실패 형상은 원래 Conv3 health 대상이었다. v5는 제외 대상이 없는 첫 Conv3 형상 `lf_bb6a106c15c571e5`를 health에 배치한다. 네 레이어·두 dtype·세 회차의 기존 health 108측정을 그대로 수행한다. 기존 실패 형상의 FP16 18건은 pilot에서 수집한다. 대체 health가 정확성·센서·시간 안정성 검사를 통과해야 나머지 수집을 시작한다.

test_04는 대상 1,062건이 모두 저장되면 `collection=COMPLETE`, 제외 이력을 반영해 `gate=REVIEW`가 된다. summary에는 원계획·실제 목표·제외 건수와 두 실패 조건 중 해당 phase의 근거가 구분된다. 제외한 FP32의 구현 6개는 분석에서 시간값 없는 FAIL로 남는다. Conv3 FP32의 validation과 test 계획은 각각 49/50형상이며 실제 평가 시 모든 방법은 동일한 공통 정상 집합을 사용한다.

## 중단 후 재개와 반환

```bash
python3 -B /home/eu/gemm-research/layer_full_20261110_v5/collect_all.py --resume /home/eu/gemm-research/layer_full_20261110_v5/runs/resume_full_20261108_105717_8tqhrkir
```

중단된 시도를 보존하고 다음 번호로 재시도한다. 같은 import 명령도 기존 v5 캠페인으로 이어진다. 반환할 때는 v5 캠페인의 모든 ZIP과 SHA sidecar, `campaign.json`, `inherited_campaign.json`, `previous_campaign.json`, `v4_campaign.json`을 함께 보낸다.

후속 분석은 `analysis/layer_full_fit_2026-11-10_v5`를 사용한다. v2 train, v3 실패 비용, v4 완료·실패 자료, v5 새 test 자료를 각각 원래 소스와 protocol로 검증한다. 모델 후보·GEMM prior·선택 규칙과 최종 시험 승인 절차는 유지한다. 이번 수집 수정은 test 예측값이나 시간 오차 평가를 수행하지 않는다.

[검증 기록](validation.json), [실제 실패 분석](../analysis/layer_full_test04_result_2026-11-10/README.md)에 근거가 있다. v5 대체 health의 실제 GPU 통과와 나머지 수집은 연구 PC 실행 단계다.
