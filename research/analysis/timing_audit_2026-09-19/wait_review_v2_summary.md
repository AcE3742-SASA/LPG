# 기존 v2 trace의 측정 중 대기 분해

새 GPU 실행 결과가 아니다. 기존 56개 측정 범위를 재분석했고 원본 157개 파일 hash는 모두 동일하다. 기존 커널 수·시간·간격 통계도 유지됐다.

| 조건 | API 진입 전 (μs/GEMM) | API 실행 중 | API 반환 후 |
|---|---:|---:|---:|
| 64x64x64 fp16 prealloc | 1.31 | 1.87 | 1.52 |
| 64x64x64 fp32 prealloc | 0.0484 | 1.17 | 1.97 |
| 128x128x128 fp16 prealloc | 1.17 | 1.83 | 1.52 |
| 3216x520x776 fp16 prealloc | 0 | 0 | 0.985 |
| 4096x4096x4096 fp16 prealloc | 0 | 0 | 1.02 |
| 4096x4096x4096 fp32 prealloc | 0 | 0.274 | 2.05 |
| 64x64x64 fp16 graph | 0 | 0 | 0.0674 |
| 64x64x64 fp32 graph | 0 | 0 | 0.119 |

각 값은 블록별 GEMM당 시간의 중앙값이다. 서로 다른 열의 중앙값 합이 전체 간격 중앙값과 같을 필요는 없다.

64³·128³ FP16 prealloc은 다음 GPU 작업에 대응하는 CUDA API에 진입하기 전 공백이 관측됐다. 이는 관측된 제출 지연이며 Python 자체 비용과 OS·framework·profiler 비용을 분리한 결과는 아니다. CUDA Graph에서는 GPU 작업이 이미 하나의 launch에 묶여 제출되므로 내부 노드 사이 간격을 다음 Python 제출 대기로 세지 않는다.

memset·memcpy·모든 커널은 GPU 작업으로 세어 간격에서 제외했다. API 반환 후의 간격은 하드웨어 원인 하나로 확정하지 않는다. 다른 프로세스의 작업을 모두 기록한 것은 아니므로 전체 GPU idle time이라고 부르지 않는다.

이 결과에는 v2에서 확인된 NSYS 시간 증가, profiler 시작 경계 문제 및 수집 경고가 그대로 적용된다. 프로파일링 없는 일반 실행에 이 수치를 그대로 대입하거나 차감하지 않는다. 수정 후 대조는 timing_audit_v4로 수행하며 아직 GPU 실행하지 않았다.

상세 결과: wait_review_v2/report.md 및 analysis.json. 분석 코드: timing_audit_v4/analyze.py. 검증 기록: v4_validation.json.
