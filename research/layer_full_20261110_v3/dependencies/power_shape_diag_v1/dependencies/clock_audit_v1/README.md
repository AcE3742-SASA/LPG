# 두 조건의 클럭·Nsight 진단

v6 파일럿에서 round 변동이 남은 두 조건만 재확인한다. 기존 v6 코드·입력·정확성·5% 기준을 유지하고, GPU 클럭과 전력 제한 설정은 바꾸지 않는다. 결과는 별도 진단 자료이며 학습용 y값에 합치지 않는다.

## GPU 컴퓨터에서 실행

ZIP을 풀어 `clock_audit_v1` 폴더를 `/home/eu/gemm-research/scripts/`에 놓는다. 옆에 이전에 실행한 `gpu_sweep_v6` 폴더가 있어야 한다. 기존 `ai_env`를 그대로 사용한다.

```bash
bash /home/eu/gemm-research/scripts/clock_audit_v1/launch.sh
```

`AUDIT_DIR=...`, `V6 SOURCE CHECK: PASS`, 환경 검사, `Starting sparse_before (1/5)`가 나오면 정상 시작이다. 각 실행은 180초 예열로 시작한다. 자세한 진행은 표시된 pass별 로그에 `[1/8]`부터 `[8/8]`까지 남고, 메인 터미널에는 pass 완료가 표시된다.

총 20~25분을 예상한다. 예열만 15분이고 실제 소요시간은 아직 GPU에서 확인하지 않았다. 다른 GPU 실험과 동시에 실행하지 않는다. 끝나면 성공·실패 모두 `RESULT_ZIP=...`로 결과 ZIP을 표시한다. 그 ZIP을 가져와 검토한다.

## 순서와 비교 대상

| 단계 | 실행 | nvidia-smi 조회 간격 |
|---|---|---:|
| sparse_before | 프로파일러 없는 실행 | 1초 |
| dense_before | 프로파일러 없는 실행 | 0.1초 |
| nsys | Nsight Systems | 0.1초 |
| dense_after | 프로파일러 없는 실행 | 0.1초 |
| sparse_after | 프로파일러 없는 실행 | 1초 |

각 단계는 새 프로세스로 실행하며 180초씩 예열한다. 1초 대 0.1초 비교는 조회 빈도 증가의 영향, 0.1초 대 Nsight 비교는 프로파일러 관측 영향을 살펴보기 위한 것이다. 전후 실행을 모두 보존한다. 서로 다른 프로세스·시간대의 차이도 포함하므로 비율 전체를 순수 계측 비용으로 해석하지 않는다.

| 대상 `(M,N,K)` | 정밀도 | 방식 | Graph 재생당 GEMM |
|---|---|---|---:|
| (1888,832,632) | FP16 | prealloc·Graph | 29회 |
| (392,2776,80) | FP32 | prealloc·Graph | 56회 |

각 단계에서 네 가지 조건을 두 round씩 측정한다. 첫 round는 첫 형상의 prealloc→Graph, 다음 형상의 prealloc→Graph이고, 다음 round는 형상과 방식 순서를 모두 뒤집는다. 8작업×5단계=40작업·280블록이다. 두 입력 seed와 Graph 묶음은 v6 파일럿에서 확인한 값이다.

각 측정의 안정화 3초, 목표 100 ms, 7블록, 적응 반복 수, CPU FP64 표본 대조는 기존 v6 함수를 사용한다. Graph는 최초 probe로 다시 묶음 수를 고르지 않고 29/56회를 사용한다. 첫 블록을 버리지 않는다.

짧은 진단이므로 원래 192작업의 앞선 부하 순서를 모두 재현하지 않는다. 변동이 재현되지 않더라도 원래 REVIEW를 지우거나 안정성 문제가 해소됐다고 자동 판정하지 않는다.

## 클럭 기록과 구간 대응

nvidia-smi에서 timestamp, GPU UUID, pstate, 온도·사용률, SM·메모리 클럭, 전력·전력 제한값, 현재 클럭 제한 사유, 전력·SW/HW thermal 카운터를 조회한다. 권한이나 지원 문제로 필요한 query가 실패하면 원문 오류를 저장하고 중단한다. 선택적인 사유·카운터가 N/A라면 그 값을 누락으로 보존하고 검토 표식을 남긴다.

0.1초는 요청한 조회 간격이다. 실제 표본 간격을 계산해 저장하고, 센서 자체의 갱신 주기가 0.1초라고 가정하지 않는다. 원본 CLI 출력과 사전 query 결과·명령을 보존한다.

기존 Timer.block 호출 직전·직후의 UTC 시각을 이벤트 구간 밖에서 기록한다. 안정화·보정·실제 측정 블록의 host 범위에 들어온 표본만 집계하며 직전·직후 표본을 별도 저장한다. 이 host 범위에는 CUDA Event 초기화와 동기화도 포함돼 GPU Event의 정확한 경계와 같지 않다. 블록 안 표본이 없으면 비워 두며 보간하지 않는다.

카운터 차이는 첫 내부 표본과 마지막 내부 표본 사이에서만 계산한다. 표본이 하나뿐이거나 카운터가 누락·감소하면 값은 미확정으로 남긴다. 이를 전체 블록의 throttling 시간으로 확대하지 않는다. 더 높은 빈도의 조회 자체는 위의 일반 실행 대조로 검토한다.

## Nsight 기록

검증된 v4의 수집·분석 함수를 재사용한다. 순서는 Nsight 시작 → 기존 안정화·보정 → 7블록이다. 준비 범위 8개와 측정 범위 56개를 구분하며 Graph는 node 단위로 추적한다. trace는 준비 구간도 보존하고, Python의 GPU 작업 분석은 실제 측정 범위에 적용한다.

CUDA API와 커널·memcpy·memset을 프로세스와 correlation ID로 연결한다. 내부 간격은 다음 API 진입 전·실행 중·반환 후로 나누고 커널 이름·GEMM당 개수·GPU 작업 시간도 저장한다. 새로운 두 형상의 커널 수를 한 개로 가정하지 않는다. 관측된 경로 변화와 불완전한 수집 경고를 남기며, 개수 일치만으로 전체 GPU의 간섭 부재나 추적 완전성을 주장하지 않는다.

## 판정과 파일

`COLLECTION COMPLETE`는 40작업과 trace 수집·분석 완료를 뜻한다. `MEASUREMENT QUALITY: REVIEW`는 블록 변동, round·세션 차이, 관측 영향, 샘플 누락, Nsight 경고 등 검토할 항목이 남았다는 뜻이다. exit 2는 이번 summary가 완성됐을 때만 기존 v6 래퍼가 REVIEW로 표시한다. 사전 검사·수치 오류·수집 실패는 FAIL로 끝나며 가능한 원자료와 로그를 압축한다. 자동 full 실행은 없다.

| 파일 | 내용 |
|---|---|
| `report.md`, `conditions.csv`, `summary.json` | 세 기록 방식 비교와 검토 표식 |
| `analysis.json` | Nsight 경고·커널 경로·작업 시간·제출 간격·조회 간격 통계 |
| `clock_windows.json` | 안정화·보정·측정 블록의 내부·직전·직후 표본과 제한된 카운터 차이 |
| 각 pass의 `results.json` | 두 round의 모든 정확성·타이밍 원자료 |
| 각 pass의 `telemetry.csv`, `telemetry_metadata.json`, `telemetry_probe.json` | 조회 원본, 명령, 시간대·간격·누락 정보 |
| `traces/` | 8개 `.nsys-rep`와 내보낸 SQLite |
| `source/`, `baseline_source/`, `manifest.json` | 진단·v6 소스 사본과 hash·사전 조건 |
| 환경·프로세스·종료 코드·로그 | 실패와 경고를 포함한 실행 기록 |

결과는 `clock_audit_v1/runs/clock_audit_날짜_식별자/`에 매번 새로 저장한다. 중단된 결과를 덮어쓰거나 이어서 수정하지 않는다. 원인을 확인한 뒤 필요하면 같은 명령으로 새 진단을 시작한다. GPU lock은 기존 스윕과 공유한다.

## 검증 상태

새 로컬 검사 10개, 기존 v4 진단 검사 22개와 v6 검사 33개를 통과했다. 측정 함수 호출·고정 묶음·타이머 복원·실패 시 monitor 정리, CSV 파싱·시각/카운터 구분, 40작업·56 trace 합성 분석과 누락·경고 처리를 포함한다. 실제 GPU의 새 query 지원, 0.1초 조회 영향, 신규 두 조건의 NSYS 경로와 안정성은 GPU 실행에서 확인해야 한다. 기존 v6와 v4는 수정하지 않았다.

근거: [NVIDIA nvidia-smi와 클럭 카운터](https://docs.nvidia.com/deploy/nvidia-smi/#clock-event-reasons-counters), [Nsight Systems capture-range 안내](https://docs.nvidia.com/nsight-systems/UserGuide/), 사용자 v6 파일럿 검산(`analysis/pilot_v6_2026-09-20/`).
