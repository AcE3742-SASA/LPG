# 수정 수집본 v2의 후속 분석

`layer_full_20261108_v2`로 처음부터 수집한 결과를 읽는 실행본이다. 2026-11-08 사용자가 선택한 공통 GEMM prior인 11,498형상 FP16 MLP32·FP32 RF를 유지한다. 최신 [공통 모델 분석 설명](../layer_full_fit_2026-11-08/README.md)의 학습·선택·시험 분리·비용 규칙을 따른다. 기존 두 분석본과 동결 폴더는 보존한다.

수집 버전과 source 해시를 v2에 맞춘 `data.py`를 사용하며 v1 측정을 혼합하지 않는다. `pipeline.py`는 공통 모델 분석 실행기의 선택 규칙을 그대로 사용한다. 구성 모델·비용·보고서 코드는 이전 검증본과 같다.

## 준비 및 학습

Mac 연구 폴더에서 기존 CPU 환경을 사용한다. 새 protocol 준비는 시간 레이블을 열기 전에 수행한다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-08_v2/pipeline.py prepare \
  --output analysis/layer_full_fit_2026-11-08_v2/prepared_layer_full
```

개발 구역 train 20묶음·validation 5묶음의 수집 완료 후 `CAMPAIGN_PATH`를 반환 경로로 바꿔 실행한다.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B \
  analysis/layer_full_fit_2026-11-08_v2/pipeline.py train \
  --campaign CAMPAIGN_PATH \
  --output analysis/layer_full_fit_2026-11-08_v2/prepared_layer_full
```

최종 시험은 모델 선택 동결과 사용자 승인 후 `pipeline.py evaluate`의 별도 명령으로 진행한다. 이번 수정 작업은 새 실측의 학습·시험을 실행하지 않는다.

## 검증

```bash
MPLCONFIGDIR=/private/tmp/layer-full-report-mpl \
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B -m unittest discover \
  -s analysis/layer_full_fit_2026-11-08_v2 -p 'test_*.py' -v
```

검증에는 source 가중치·선택의 고정, 실제 GEMM 개발 예측 재현, v2 입력 검산, 합성 자료의 학습·선택·시험 분리·보고서를 포함한다. 레이어 실측 학습은 전체 개발 구역 수집 이후 단계다.
