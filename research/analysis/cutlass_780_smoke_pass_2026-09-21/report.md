# 780 MHz CUTLASS 실행 점검 통과

2026-09-21 15:31 KST 실행의 두 커널 모두 `Status=success`, `Disposition=passed`, `reference_device: Passed`, 프로파일러 및 runner 종료코드 0이다. 이번 검증은 CUTLASS의 GPU 참조 구현과의 비교이며 cuBLAS 검증 성공을 뜻하지 않는다. 두 모드 모두 8192³에서 한 번의 프로파일러 호출로 측정한 값이고 최종 연산 천장은 아직 미확정이다.

| 경로 | 실행시간 | 2MNK 기준 처리율 | 부하 표본 |
|---|---:|---:|---:|
| FP32 SIMT, FP32 입출력·누산 | 170.89 ms | 6.43 TFLOP/s | 61 |
| FP16 Tensor Core, FP16 입출력·FP32 누산 | 28.00 ms | 39.27 TFLOP/s | 46 |

SM 780 MHz·메모리 6801 MHz·70 W가 사용률>0 표본 107개 모두에서 확인됐다. 전체 표본은 115개, 사용률 100%는 103개, 최대 표본 간격은 0.206초다. 전체 관측 온도는 47–52°C, 최대 관측 전력은 47.4 W다. 표본 사이의 연속 클록 유지나 장시간 열평형을 입증하는 것은 아니다. 시작 전 `210/405 MHz`였던 첫 실행도 실제 부하에서는 목표로 올라갔으므로 무부하 사전 검사를 고친 v3의 진행 방식이 이번 실행에서 작동했다.

정확한 커널은 `cutlass_simt_sgemm_256x128_8x5_tn_align1`과 `cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8`이다. 두 경로의 alpha=1, beta=0, split-K=1, batch=1, A row/B column/C·D column, seed 20260921, uniform [-2,2] 기본 scale, epsilon=0, workspace=1, 예열 10회, profiling-duration 1000 ms·min-iterations 10을 유지했다. profiler 내부 반복의 개별 원시 시간은 CSV에 없으므로 이번 두 행으로 실행 간 분산·IQR을 구하지 않는다.

원자료 28파일·배포 runner 해시·기존 바이너리와 소스 커밋을 확인했고, `.check.json`의 원행과 CSV가 정확히 일치한다. `2MNK/Runtime`으로 처리율을 독립 재계산했다. CUTLASS GFLOPs에는 2MN epilogue가 추가되어 이 기준보다 약 0.012% 높다. CSV의 GB/s는 논리적 GEMM byte 수에 기초한 GiB/s이며 DRAM 대역폭 천장이 아니다. 원본과 정밀 수치·구성 파일 해시는 `raw/`와 `analysis.json`에 보존했다.

`SW Power Capping` 누적 카운터는 전체 구간에서 약 5.98초 증가했다. 관측 클록 유지와 별도로 남겨야 하는 진단 정보이며, 이번의 낮은 샘플 전력만으로 전력 제한이 전혀 없었다고 결론내리지 않는다. 누적값 증가를 곧바로 두 측정 커널의 실제 제한 시간으로 배분하지 않는다. 기존 연구에서 보류 중인 이 카운터의 해석을 이번 smoke PASS가 해소한 것은 아니다.

다음 단계는 설치된 동일 정밀도 커널 8개씩 총 16개를 8192³에서 비교하는 것이다. 이후 유망 커널의 행렬 크기와 독립 반복을 비교해야 경험적 연산 천장 후보를 확정할 수 있다. 이 과정은 기존 10,000조건 연구 본수집과 구분한다. [BabelStream의 읽기 중심 대역폭 후보](../babelstream_780_bw_2026-09-21/report.md) 261.2 GB/s는 유지한다.

압축 원본 SHA-256: `fabf311ab178a9f3174335dcf4950f3f21857c8169602b8686b71b0ccff444fb`. 재현: `python3 -B analyze.py raw/cutlass_780_smoke_v3.Vq21ey.tar.gz`.
