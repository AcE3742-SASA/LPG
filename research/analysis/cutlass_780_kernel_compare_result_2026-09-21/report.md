# CUTLASS 16개 커널 비교 결과 — 780 MHz

검토 대상은 `cutlass_780_kernel_compare.xk43QJ.tar.gz`다. 설정 로그·실행 스크립트·부하 표본 모두 780 MHz 실행임을 확인했다. 현재 연구 목표인 750 MHz 결과로 분류하지 않는다.

## 판정

- 16개 커널 모두 `Status=success`, `Disposition=passed`, `reference_device: Passed`이며 프로파일러·전체 실행 종료코드는 0이다.
- 예상한 FP32 SIMT 8개·FP16 입출력/FP32 누산 Tensor Core 8개와 A/B/C/D 배치가 정확히 일치한다. 각 8192³, alpha=1, beta=0, batch=1, split-K=1이다.
- 예열 10회, profiling-duration=1000 ms·최소 10회, 동일 입력 분포·seed 및 device 참조 검증 명령을 확인했다. 커널당 프로파일러 호출 1회이며 독립 반복성은 아직 평가하지 않았다.
- 전체 669개 클록 표본 중 GPU 사용률>0인 605개 모두 780/6801 MHz·70 W다. 사용률 100% 표본은 581개이며 최대 표본 간격은 0.207초다. 표본 사이의 순간 변동까지 배제하는 검사는 아니다.
- 관측 온도 40–51°C, 최대 전력 47.9 W, 첫 프로파일러 시작부터 마지막 종료까지 136.8초다.
- SW Power Capping 누적 카운터는 약 76.67초 증가했다. 낮은 전력 표본만으로 전력 제한 부재를 주장하지 않으며, 이 증가분을 timed GEMM의 제한시간·제한비율로 환산하지 않는다. 이전부터 남아 있던 누적 카운터 해석은 미해결이다.

## 커널별 관측값

처리율은 `2MNK / Runtime` 기준이다. CUTLASS 보고 GFLOPs의 epilogue 포함 연산량과 구분해 재계산했다. 아래 값은 이번 단일 비교의 관측값이며 최종 천장이 아니다.

| 경로 | 커널 구성 | A/B 배치 | 시간 (ms) | TFLOP/s |
|---|---|---|---:|---:|
| FP32 SIMT | 128x128_8x2 | nn | 160.07 | 6.87 |
| FP32 SIMT | 128x128_8x2 | nt | 159.58 | 6.89 |
| FP32 SIMT | 128x128_8x2 | tn | 164.28 | 6.69 |
| FP32 SIMT | 128x128_8x2 | tt | 158.38 | 6.94 |
| FP32 SIMT | 256x128_8x5 | nn | 160.53 | 6.85 |
| FP32 SIMT | 256x128_8x5 | nt | 170.18 | 6.46 |
| FP32 SIMT | 256x128_8x5 | tn | 171.07 | 6.43 |
| FP32 SIMT | 256x128_8x5 | tt | 160.11 | 6.87 |
| FP16 Tensor | s1688gemm_f16_256x128_32x2 | nn | 38.61 | 28.48 |
| FP16 Tensor | s1688gemm_f16_256x128_32x2 | nt | 37.60 | 29.24 |
| FP16 Tensor | s1688gemm_f16_256x128_32x2 | tn | 38.01 | 28.93 |
| FP16 Tensor | s1688gemm_f16_256x128_32x2 | tt | 37.74 | 29.14 |
| FP16 Tensor | s16816gemm_f16_256x128_32x3 | nn | 27.98 | 39.30 |
| FP16 Tensor | s16816gemm_f16_256x128_32x3 | nt | 27.94 | 39.35 |
| FP16 Tensor | s16816gemm_f16_256x128_32x3 | tn | 27.99 | 39.28 |
| FP16 Tensor | s16816gemm_f16_256x128_32x3 | tt | 28.06 | 39.19 |

`n`은 column, `t`는 row 저장 배치이며 C/D는 column이다.

이번 비교의 최고 관측값은 FP32 `cutlass_simt_sgemm_128x128_8x2_tt_align1`의 6.94 TFLOP/s와 FP16 `cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_align8`의 39.35 TFLOP/s다. 작은 커널 간 차이를 안정적인 우위로 확정하지 않는다.

## 현재 목표와 다음 단계

사용자가 선택한 목표는 750/6801 MHz·70 W다. 이미 준비한 `../cutlass_750_kernel_compare_2026-09-21/cutlass_750_kernel_compare.sh`로 같은 16개를 측정한 뒤 750 MHz에서 후보를 판단한다. 기존 GPU 측정이 끝나고 정리된 뒤 사용자가 실행한다. 새 스크립트 생성·원격 실행·설정 변경은 이번 검토에서 하지 않았다.

750 MHz의 BabelStream 대역폭 측정과 후보별 행렬 크기·독립 반복 확인이 남아 있다. 기존 780 MHz의 261.2 GB/s 및 이번 연산값을 750 MHz 결과에 섞거나 클록 비례 환산하지 않는다.

## 원본과 재현

- 원본 154파일과 압축을 `raw/`에 바이트 그대로 보존했고 파일별 SHA-256은 `analysis.json`에 기록했다.
- 압축 SHA-256: `8e11151ffcad3c41282cbf6a8bf039160c506acb005c34736aead1a93ef7ef3d`.
- runner SHA-256: `841895e71a83f35764f61667a1c34caa7164a71624bd68603f592bbd0abb70c2`. 배포한 780 MHz 비교본과 동일하다.
- CUTLASS 커밋 `147295a3d4b75f3aeff247c25b8927cea9a7006a`, 소스 변경 목록 없음. 바이너리 SHA-256 `1e0608eb3eda54d329fd2cc017a28057aca952ae5732f8242aa5073abd7e8a4d`로 이전 실행과 같다.
- 재현: `python3 -B analyze.py raw/cutlass_780_kernel_compare.xk43QJ.tar.gz`. 압축 해시, 파일·커널 수, 검증 상태·배치·부하 클록, CSV/JSON 일치, 처리율을 다시 확인한다. 상세 원값은 `analysis.json`, 비교표는 `summary.csv`다.
