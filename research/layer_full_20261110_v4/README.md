# train 20묶음 보존 · 알려진 정확성 실패 제외 · validation 재개

`layer_full_20261110_v4`는 v2 train 20묶음·21,600측정과 validation 실패 시도 001·002의 원본 ZIP 22개를 고정 SHA-256으로 검증해 새 캠페인에 복사한다. 기존 폴더와 소스는 보존한다.

v3 사전 검사에서 `lf_ce6440e85a4add93`의 표준 Conv3 FP32 출력 두 좌표가 CPU FP64 정답 대비 기존 허용오차를 초과했다. 최대 절대오차는 `2.53605546e-5`, 허용오차 대비 비율은 `1.24537199`였다. unfold+GEMM은 같은 좌표에서 통과했다. 사용자가 선택한 정책에 따라 기존 `atol=2e-5`, `rtol=2e-4`를 유지하고 이 형상의 FP32 조건만 측정·분석에서 제외한다. 제외 근거 JSON과 해시를 protocol에 고정했다.

## 연구 PC 실행

새 폴더 전체를 `/home/eu/gemm-research/layer_full_20261110_v4`에 둔다. 기존 v3 캠페인에는 `campaign.json`, `inherited_campaign.json`, 원본 ZIP 22개가 있어야 한다. 일반 사용자 `eu`의 기존 tmux에서 실행한다.

```bash
(
set -euo pipefail
cd /home/eu/gemm-research/layer_full_20261110_v4
python3 -B selfcheck.py
python3 -B collect_all.py --import-campaign /home/eu/gemm-research/layer_full_20261110_v3/runs/resume_full_20261108_105717_8tqhrkir
)
```

train_01~20은 `SKIP INHERITED`로 넘어가고 `validation_01_attempt_003`부터 시작한다. 이번 validation_01은 1,062건을 수집하고 18건(구현 6개 × 회차 3개)을 명시적으로 제외한다. 해당 형상의 FP16은 계속 수집한다. 이후 validation_02~05, test_01~05는 기존 순서와 측정 예산을 따른다. 전체 완료 목표는 32,382건과 제외 18건이다.

새 캠페인은 `/home/eu/gemm-research/layer_full_20261110_v4/runs/resume_full_20261108_105717_8tqhrkir`다. 같은 import 명령을 다시 실행하면 이 캠페인을 이어간다. 직접 재개할 때는 다음 명령을 쓴다.

```bash
python3 -B /home/eu/gemm-research/layer_full_20261110_v4/collect_all.py --resume /home/eu/gemm-research/layer_full_20261110_v4/runs/resume_full_20261108_105717_8tqhrkir
```

실패가 입증된 조건은 pair 생성 전에 건너뛴다. v3의 중단 원인이었던 별도 사전 probe 호출도 수집 경로에서 제거했다. 본 worker의 예열·health 검사·유한성·정확성·온도·클럭 검사는 유지한다. 새 정확성 실패나 health 실패가 발생하면 해당 시도와 로그를 보존하고 중단한다.

## 판정과 분석

validation_01은 대상 1,062건이 모두 저장되면 `collection=COMPLETE`, 제외 이력을 반영해 `gate=REVIEW`가 된다. summary에 `planned=1080`, `expected=1062`, `excluded=18`, 제외 조건과 증거를 기록한다. 제외한 조건은 분석에서 `FAIL`과 빈 시간값으로 남기며 실제 label이나 측정 비용을 만들지 않는다. Conv3 FP32 validation의 공통 비교 대상은 49/50이다.

기존 train_03·09의 FP32 FFN GELU 회차 변동 REVIEW도 유지한다. 원래 train 집계 label 7,200개 중 기존 정책으로 채택되는 7,198개와 제외 2개가 그대로 유지된다. 실패 시도 001·002의 부분 결과는 새 validation label에 합치지 않고 비용 이력에 보존한다.

후속 분석은 `analysis/layer_full_fit_2026-11-10_v4`를 사용한다. v2 train, v3 실패 비용, v4 새 측정을 각 소스·protocol·ZIP 해시로 구분한다. 모델 후보·GEMM prior·선택 규칙·sealed test 잠금은 유지한다. 캠페인 반환 시 모든 ZIP·SHA sidecar와 `campaign.json`, `inherited_campaign.json`, `previous_campaign.json`을 함께 보낸다.

[검증 기록](validation.json), [고정된 승계 대상](inheritance.json), [실패 분석](../analysis/layer_full_probe_result_2026-11-10/README.md)에 근거를 남겼다. v4 실제 GPU 수집은 연구 PC에서 실행해 확인한다.
