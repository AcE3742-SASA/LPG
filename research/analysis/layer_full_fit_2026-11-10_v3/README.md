# v2 학습 결과를 이어받는 v3 레이어 분석

`layer_full_20261110_v3` 수집본의 분석 실행본이다. 완료된 v2 학습 20묶음을 그대로 사용하고, v3에서 재수집한 validation과 이후 test를 연결한다. 11,498형상 FP16 MLP32와 FP32 RF의 공통 GEMM prior, 학습기 비교, 모델 선택과 시험 잠금 규칙은 [기존 공통 모델 분석](../layer_full_fit_2026-11-08/README.md)을 유지한다.

v2 입력은 수집본의 `inheritance.json`에 고정된 20개 ZIP만 허용한다. 원래 campaign 해시, 소스와 protocol, ZIP 파일명과 SHA-256과 시도 이력을 대조하고, 해당 묶음을 다른 ZIP이나 압축 해제 폴더로 대체하면 거부한다. v3 validation과 test에는 v3 manifest와 소스 검사를 적용한다. 실패한 v2 `validation_01_attempt_001`의 비용도 원래 ZIP에서 읽어 합산하며, 학습에는 완료된 v3 재시도를 사용한다.

## 준비와 학습

Mac 연구 폴더에서 아래 명령으로 분석 protocol을 별도 동결한다. 입력 상속 명세와 기존 GEMM 가중치도 함께 고정한다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-10_v3/pipeline.py prepare \
  --output analysis/layer_full_fit_2026-11-10_v3/prepared_layer_full
```

validation 5묶음 수집을 완료한 뒤 `CAMPAIGN_PATH`를 v3 반환 캠페인 경로로 바꿔 실행한다. 반환 폴더에는 `campaign.json`, `inherited_campaign.json`, 이어받은 21개 원본 ZIP과 SHA-256 파일과 v3 결과가 함께 있어야 한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-10_v3/pipeline.py train \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-10_v3/prepared_layer_full
```

최종 시험은 모델 선택 동결과 사용자 승인 후 `pipeline.py evaluate`로 진행한다. train_03과 train_09의 FP32 GELU 회차 변동은 기존 품질 제외 규칙을 따른다.

## 검증

```bash
MPLCONFIGDIR=/private/tmp/layer-full-report-mpl \
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B -m unittest discover \
  -s analysis/layer_full_fit_2026-11-10_v3 -p 'test_*.py' -v
```

혼합 버전 입력의 성공 및 거부 사례, 실패 시도 비용, 기존 입력 검산, 공통 GEMM 모델 고정, 합성 자료 학습, 선택, 시험 분리와 보고서를 검사한다. 실제 학습은 개발 구역 수집 완료 후 실행한다.
