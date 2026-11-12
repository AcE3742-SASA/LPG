# 실제 측정 블록 진단

수집 구조 검증: PASS_WITH_WARNINGS. 측정 품질: 검토 필요.

각 조건은 일반 실행, Nsight 실행, 일반 재실행으로 비교했다. 시간은 블록 평균의 중앙값이다.

| 조건 | 일반 전 (μs) | Nsight (μs) | 일반 후 (μs) | Nsight/일반 | 커널 사이 간격 비중 |
|---|---:|---:|---:|---:|---:|
| 64x64x64 fp16 prealloc | 3.97 | 6.68 | 4.08 | 1.66 | 70.2% |
| 64x64x64 fp32 prealloc | 6.32 | 8.66 | 6.32 | 1.37 | 36.6% |
| 128x128x128 fp16 prealloc | 3.96 | 6.65 | 4.09 | 1.65 | 68% |
| 3216x520x776 fp16 prealloc | 80.2 | 81.3 | 80.9 | 1.01 | 1.21% |
| 4096x4096x4096 fp16 prealloc | 3.27e+03 | 3.27e+03 | 3.26e+03 | 1 | 0.0311% |
| 4096x4096x4096 fp32 prealloc | 1.28e+04 | 1.27e+04 | 1.29e+04 | 0.986 | 0.0217% |
| 64x64x64 fp16 graph | 1.73 | 1.88 | 1.73 | 1.08 | 3.59% |
| 64x64x64 fp32 graph | 5.05 | 5.32 | 5.05 | 1.05 | 2.24% |

## 측정 중 GPU 작업 사이 간격

모든 내부 간격을 다음 GPU 작업에 연결된 CUDA API 시점으로 분할했다. memcpy·memset과 모든 연산 커널은 GPU 작업으로 세었다.

| 조건 | API 진입 전 (μs/GEMM) | API 실행 중 | API 반환 후 |
|---|---:|---:|---:|
| c00 64x64x64 fp16 prealloc | 1.31 | 1.87 | 1.52 |
| c01 64x64x64 fp32 prealloc | 0.0484 | 1.17 | 1.97 |
| c02 128x128x128 fp16 prealloc | 1.17 | 1.83 | 1.52 |
| c03 3216x520x776 fp16 prealloc | 0 | 0 | 0.985 |
| c04 4096x4096x4096 fp16 prealloc | 0 | 0 | 1.02 |
| c05 4096x4096x4096 fp32 prealloc | 0 | 0.274 | 2.05 |
| c06 64x64x64 fp16 graph | 0 | 0 | 0.0674 |
| c07 64x64x64 fp32 graph | 0 | 0 | 0.119 |

각 열은 블록별 값을 먼저 구한 뒤 중앙값을 낸 것이다. 열 중앙값의 합은 전체 간격 중앙값과 다를 수 있다.
API 진입 전은 다음 제출 함수에 아직 도달하지 않은 시간이다. Python·프레임워크·OS·계측 비용을 구분한 값은 아니다. API 반환 후에는 Graph 내부 순서, 의존성, driver/device scheduling 및 미관측 간섭 등이 남는다.
이 표는 프로파일링된 실행의 진단이며 일반 실행의 대기 시간으로 그대로 대입하거나 보정하지 않는다.

## 시작 경계와 첫 블록


## Nsight 자체 경고

- capture.1.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.2.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.3.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.4.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.5.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.6.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.7.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.
- capture.8.sqlite: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.

간격 비중은 첫 커널 시작~마지막 커널 끝에서 기록된 커널 구간을 제외한 비중이다. 전체 GPU 유휴율 또는 원래 본측정의 오차율이 아니다.

CPU NVTX 범위에는 event 초기화와 마지막 동기화도 포함된다. event 시간에서 CUPTI 커널 시간을 뺀 차이를 전부 CPU 제출 비용으로 해석하지 않는다.

조건별 변동 표식과 블록별 API·커널 이름, 큰 간격의 다음 launch 시점은 analysis.json에 있다. 자동 판정만으로 full을 시작하지 않는다.
