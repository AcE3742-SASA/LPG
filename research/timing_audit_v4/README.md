# timing_audit_v4 — 측정 중 대기 진단

사용자가 지적한 작은 GEMM의 제출 대기를 우선 진단한다. v3가 확인하려던 큰 GEMM의 시작 경계도 같은 실행에 포함한다. **v3를 아직 실행하지 않았다면 v4만 실행한다.** 기존 v1/v2/v3/v5와 원본 자료는 보존한다.

## 실행

ZIP을 풀어 생긴 `timing_audit_v4` 폴더를 GPU 컴퓨터의 `/home/eu/gemm-research/scripts/`에 놓는다. 그 옆에는 기존 `gpu_sweep` v5가 있어야 한다.

```bash
bash /home/eu/gemm-research/scripts/timing_audit_v4/launch.sh
```

정상 시작에는 `AUDIT_DIR=...`, `V5 SOURCE CHECK: PASS`, `ENV CHECK: PASS`, `Starting plain_before`가 나온다. 상세 진행은 표시된 `plain_before.log`, `nsys.log`, `plain_after.log`에 남는다. 각 실행은 180초 예열로 시작하므로 처음 3분은 조건 완료 출력이 없을 수 있다.

일반 전 → Nsight → 일반 후, 세 실행의 예열만 총 9분이다. 이번에는 NSYS 준비 구간도 저장하므로 파일 크기와 export 시간이 늘 수 있다. **계획용 예상 12~20분**이며 v4 GPU 실측 시간은 아니다. 터미널을 유지하고 다른 GPU 계산을 동시에 실행하지 않는다.

완료 후 `COLLECTION COMPLETE`, `RESULT_ZIP=...`가 나온다. 성공·실패 모두 종료 코드·로그·원자료를 ZIP으로 남기므로 표시된 결과 ZIP을 전달한다. 프로세스는 기존 v5와 같은 GPU lock을 사용한다. 환경 재설치나 clock·power 변경은 없다.

## 조건과 수집 순서

| ID | M,N,K | 정밀도 | 방식 |
|---|---|---|---|
| c00 | 64,64,64 | FP16 | prealloc |
| c01 | 64,64,64 | FP32 | prealloc |
| c02 | 128,128,128 | FP16 | prealloc |
| c03 | 3216,520,776 | FP16 | prealloc |
| c04 | 4096,4096,4096 | FP16 | prealloc |
| c05 | 4096,4096,4096 | FP32 | prealloc |
| c06 | 64,64,64 | FP16 | Graph |
| c07 | 64,64,64 | FP32 | Graph |

v2의 8개 task와 순서·입력 seed 조건을 그대로 보존한다. 8조건 × 3회 = 24개 조건 실행, NSYS 측정 범위는 56블록이다. 각 조건에서 **Nsight 시작 → 기존 3초 안정화·반복 보정 → 실제 7블록 측정** 순서로 진행한다. 준비 범위 8개도 NVTX로 구분해 저장하지만 측정 통계에는 넣지 않는다.

v5 `Timer.block/run/prepare`와 source hash 13개, 예열·안정화·100 ms 목표 블록·적응 반복 수·정확성 검사·5% 변동 기준은 유지한다. profiler가 안정화와 보정 중에도 켜져 있으므로 각 pass의 반복 수는 다를 수 있다. 첫 블록은 삭제하지 않는다.

## 무엇을 진단하는가

각 측정 범위의 CUDA API와 GPU 작업을 프로세스·correlation ID로 연결한다. 단일 stream에서 관측된 커널·memcpy·memset의 합집합을 GPU 작업 구간으로 삼고, 작업 사이 모든 내부 간격을 다음 GPU 작업에 대응하는 API 시점으로 분해한다.

1. **API 진입 전:** 앞선 GPU 작업이 끝난 뒤에도 CPU가 다음 launch 또는 메모리 API에 아직 들어가지 않은 부분이다.
2. **API 실행 중:** GPU 작업 사이 간격과 다음 API 실행 시간이 겹치는 부분이다.
3. **API 반환 후:** 다음 API가 반환한 뒤 다음 GPU 작업이 시작하기까지 남은 부분이다.

세 부분의 합이 해당 간격과 정확히 일치하는지 검증한다. 앞선 GPU 작업이 끝났을 때 다음 API가 미진입·실행 중·반환 완료 중 어느 상태였는지도 센다. `report.md`에는 GEMM당 시간의 블록 중앙값을, `analysis.json`에는 모든 간격의 합계·개수·큰 간격의 예시를 보존한다. 처음 GPU 작업 이전과 마지막 작업 이후의 event 경계 구간은 이 내부 간격 집계에 포함하지 않는다.

64³ FP32의 주 커널과 reduction 커널은 모두 GPU 작업으로 센다. 4096³ FP32의 2 KiB memset도 GPU 작업이다. Graph의 여러 노드가 하나의 launch에 대응하는 경우 각 노드를 같은 launch에 연결하며, 이미 제출된 Graph 내부 간격을 Python의 다음 제출 대기로 분류하지 않는다. 여러 stream이나 해당 프로세스의 미연결 커널이 관측되면 원인 분류 불가로 표시한다.

이 분류는 **시간 관계**를 나타낸다. API 진입 전 시간에는 Python·프레임워크·OS 스케줄링·프로파일러 비용이 섞일 수 있다. API 반환 후 시간도 driver/device scheduling·의존성·Graph 내부 순서·다른 context 등의 원인을 자동으로 구분하지 못한다. 전체 GPU의 유휴 시간이나 일반 실행의 정확한 대기 시간이라고 부르지 않는다. 더 깊은 원인 분리는 관측된 구간에 맞춰 판단한다.

## 일반 실행·Graph 대조의 역할

프로파일링 없는 일반 전후 시간과 변동을 기준으로 NSYS 관측 영향을 확인한다. 같은 작은 형상의 prealloc·Graph 대조로 제출 방식에 대한 민감성을 확인한다. Graph는 캐시·실행 방식도 달라질 수 있으므로 두 방식의 시간 차이 전체를 순수 Python 비용으로 단정하지 않는다. NSYS에서 관측한 간격을 일반 시간에서 일괄 차감하지 않는다.

다음 항목을 함께 검토한다.

- 작은 prealloc의 간격 중 다음 API 진입 전 부분이 얼마나 되는가.
- Graph에서는 다음 API가 이미 반환한 상태의 내부 간격으로 어떻게 바뀌는가.
- profiler를 켰을 때 일반 실행과 시간이 얼마나 달라지는가.
- profiler 시작 후 안정화했을 때 큰 GEMM의 첫 블록 변동이 줄어드는가.
- 모든 기대 블록·커널과 준비 범위가 기록됐고 수집 경고가 남는가.

## 출력과 판정

`report.md`·`conditions.csv`·`analysis.json`에 일반/NSYS 시간, 간격 분해, 첫 블록·안정화 경계, Nsight 자체 경고를 기록한다. 원본은 각 pass의 `results.json`, `traces/*.nsys-rep`·`*.sqlite`, telemetry·환경·프로세스 목록·종료 코드다. SQLite는 immutable/read-only로 읽고 hash를 확인한다. 준비 구간은 원본 trace에 보존하되 Python 분석에서는 측정 범위에 겹치는 행만 읽어 메모리 사용을 줄인다.

`COLLECTION INTEGRITY: PASS`는 구조 검증이며, Nsight 자체 경고가 있거나 진단 테이블이 없으면 `PASS_WITH_WARNINGS`다. `MEASUREMENT QUALITY: REVIEW REQUIRED`는 결과 해석이 남았다는 뜻이며 실행 실패가 아니다. 관측된 경로에서 기대한 커널 수가 바뀌거나 간격 분류가 불가능하면 표식을 남긴다. 수치 검사 실패·측정 블록 누락·source 불일치·도구 오류는 실패로 기록한다.

이 결과만으로 자동으로 full을 시작하지 않는다. 최종 예측 대상이 일반 실행의 stream 경과 시간인지, GPU 작업 중심 시간인지도 대기 진단 결과와 함께 명시해야 한다. NCU는 이후 필요할 경우 커널 내부 병목을 확인하며 일반 latency 측정값을 대체하지 않는다.

## 로컬 검사

```bash
python3 -B -m unittest discover -s timing_audit_v4 -p 'test_*.py' -v
```

GPU 재실행은 아직 수행하지 않았다. 로컬 검사와 기존 v2 trace 재분석은 v4가 GPU에서 정상 수집됐다는 증거가 아니다.

출처: [NVIDIA의 CUDA API·GPU 실행·대기 구간 설명](https://developer.nvidia.com/blog/?p=20916), [Nsight Systems capture API](https://docs.nvidia.com/nsight-systems/UserGuide/), 사용자 제공 v2 진단 ZIP. 세 구간 분할은 이 도구의 timestamp를 이용한 본 진단의 분석 방법이다.
