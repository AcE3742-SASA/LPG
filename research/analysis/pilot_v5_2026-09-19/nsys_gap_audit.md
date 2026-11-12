# 2026-09-19 v5 측정 구간 재검토

현재 판단: 대규모 본측정에 앞서 실제 측정 루프의 타임라인과 측정 대상의 정의를 확인해야 한다. 앞선 본측정 수집 권고는 실행 성공·수치 검사·반복 변동에 근거했으며, 커널 사이 간격과 측정 대상의 적합성을 충분히 검증하지 못했다. 기존 수집 자료가 모두 무효라는 뜻은 아니다.

`Timer.block`은 시작 CUDA Event, Python 반복 호출, 종료 CUDA Event 사이를 잰다. GPU가 CPU의 다음 제출을 기다리는 구간도 두 event 사이에 있으면 포함된다. IQR이 작더라도 반복적으로 포함되는 간격은 제거되지 않는다. CPU 호출 시간은 GPU 실행과 겹칠 수 있으므로 전체 CPU 시간을 단순히 더하거나 빼면 안 된다.

현재 `profiling.worker`는 별도 프로세스에서 준비 뒤 함수를 20번 호출한다. 실제 7개 event 측정 블록을 추적하지 않는다. NSYS PASS는 실제 커널 기록과 성공적인 보고서 생성을 검증하며, 본측정 내부에 불필요한 작업이나 대기가 없다는 판정은 아니다.

## 저장된 trace의 간격

세 trace는 batch 64, 폭 1024의 FP16 행렬곱 4층 연결이며 Graph는 캡처된 반복을 포함한다. 전체 GPU의 유휴율이나 v5 본측정의 오차율이 아니다. 각 trace에서 첫 커널 시작부터 마지막 커널 끝까지를 분모로 하고, 기록된 커널 실행 구간의 합집합을 제외한 비중을 계산했다. 시작 전·종료 후 event 경계는 포함하지 않는다.

| 모드 | 커널 수 | 첫 시작~마지막 끝 | 커널 실행 구간 | 사이 간격 | 간격 비중 |
|---|---:|---:|---:|---:|---:|
| prealloc | 80 | 587 μs | 524 μs | 63.6 μs | 10.8% |
| sync_each | 80 | 1.22e+03 μs | 531 μs | 692 μs | 56.6% |
| graph | 4,800 | 3.09e+04 μs | 3.06e+04 μs | 214 μs | 0.694% |

수치는 프로파일러를 켠 별도 실행의 관측이다. 간격의 원인을 이 집계만으로 CPU 제출·GPU 스케줄링·프로파일러 영향 중 하나로 확정하지 않는다. `sync_each`는 의도적으로 층 사이를 동기화한 대조다. Graph는 실행 방식과 반복 수가 다르므로 위 비중 차이를 단일 원인의 효과로 해석하지 않는다. 자세한 값·원본 경로·SHA-256은 [nsys_gap_audit.json](nsys_gap_audit.json)에 보존했다. SQLite는 immutable/read-only로 읽었으며 읽기 전후 hash가 일치했다.

## 필요한 확인

1. 작은 GEMM, 큰 GEMM, REVIEW 형상을 대상으로 원래 준비·반복·event 순서를 유지하고 각 실제 측정 블록 바깥에 NVTX 표식을 붙여 Nsight Systems로 짧게 수집한다. 각 호출 안에는 추가 표식을 넣지 않아 관측 영향을 줄인다.
2. 블록별 event 시간과 커널 실행 구간, 커널 사이 간격, CPU launch·메모리 복사·할당·동기화·동시 실행의 위치를 대조한다. NVTX는 CPU 범위이므로 GPU event 경계와 동일하다고 가정하지 않는다.
3. 같은 조건의 프로파일링 없는 실행을 대조해 관측 영향을 확인한다. 프로파일링된 시간을 본측정 정답으로 교체하지 않는다.
4. 원인 후보가 커널 내부라면 Nsight Compute로 DRAM·캐시·SM/Tensor Core 이용률을 확인한다. Nsight Compute의 replay·직렬화·캐시 제어가 실행을 바꾸므로 NCU 실행 중 event 시간을 일반 latency로 쓰지 않는다.
5. 커널 실행 성능과 Python/PyTorch 제출 간격을 포함한 실행 성능 중 예측할 대상을 명시한다. CUDA Graph도 별도 실행 조건이므로 자동으로 순수 커널 시간과 같다고 취급하지 않는다.

새 GPU 진단은 아직 실행하지 않았다. full_v5의 실행 여부도 미확인이다. 기존 v5 원자료·소스·배포 ZIP·5% 기준은 그대로 보존한다.

근거: [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#range-and-precision), [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/), v5 저장 소스 `measure.py`·`profiling.py`, 세 NSYS SQLite.
