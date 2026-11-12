# timing_audit_v3 — 프로파일러 시작 경계 재검증

v2에서 큰 GEMM의 첫 블록이 빠르고, 안정화 뒤 Nsight를 켜는 경계에 약 0.19초의 공백이 관측됐다. v3는 **Nsight 시작 → 기존 3초 안정화·반복 보정 → 실제 7블록 측정**으로 순서만 바로잡는 별도 진단이다. GPU에서 수정 효과를 아직 검증하지 않았다.

## 실행

이 ZIP을 풀어 생긴 `timing_audit_v3` 폴더를 GPU 컴퓨터의 `/home/eu/gemm-research/scripts/`에 놓는다. 옆의 `gpu_sweep`는 기존 v5 그대로 사용한다.

```bash
bash /home/eu/gemm-research/scripts/timing_audit_v3/launch.sh
```

정상 시작: `AUDIT_DIR=...`, `V5 SOURCE CHECK: PASS`, `ENV CHECK: PASS`, `Starting plain_before`가 나온다. 각 실행의 상세 진행은 표시된 `plain_before.log`, `nsys.log`, `plain_after.log`에 남는다. 예열 중에는 한동안 조건 완료 출력이 없을 수 있다.

일반 전 → Nsight → 일반 후의 각 실행에 기존 예열 180초를 유지한다. 예열만 총 9분이므로 전체 **약 10~15분**을 계획용으로 예상한다. v3 GPU 실측 시간은 아니다. 터미널을 유지하고 다른 GPU 계산을 동시에 실행하지 않는다.

정상 종료: `COLLECTION COMPLETE` 후 `RESULT_ZIP=...`가 나온다. 성공·실패 모두 실제 종료 코드와 로그를 담은 ZIP을 생성한다. 그 결과 ZIP을 전달하면 된다. 환경 재설치, 서비스 종료, clock·power 변경이나 full 자동 실행은 없다. 중단 후 재실행도 새로운 출력 폴더를 사용한다.

## 비교 조건

| ID | M,N,K | 정밀도 | 방식 |
|---|---|---|---|
| c03 | 3216,520,776 | FP16 | prealloc |
| c04 | 4096,4096,4096 | FP16 | prealloc |
| c05 | 4096,4096,4096 | FP32 | prealloc |

v2의 task ID·형상·정밀도·입력 seed 조건을 그대로 보존한다. 세 조건 × 세 실행 = 9개 조건 실행이며, NSYS에서는 준비 범위 3개와 실제 측정 범위 21개를 추적한다. 조건 수는 반복 블록 수와 다르다. 작은 GEMM의 관측 영향은 v2에서 확인했으므로 이번에는 반복하지 않는다.

## 바뀐 내용

- Nsight 시작 비용은 안정화 **앞**으로 옮겼다. `profiler_start_wall_ms`에는 시작 API 호출의 CPU 경과 시간을 따로 저장한다.
- 원래 `Timer.prepare`의 probe·3초 안정화·적응 반복 보정을 profiler가 켜진 상태에서 수행한다. `prepare/c03` 등의 NVTX 범위에 보존하며 실제 측정 통계에는 넣지 않는다.
- 실제 `Timer.block/run`은 재사용한다. 블록 안의 GEMM 호출마다 계측을 추가하지 않는다. 첫 블록 포함 7블록을 모두 보존하며 5% 기준을 바꾸지 않는다.
- 분석기는 준비 범위가 먼저 끝났는지, 같은 CPU thread인지, 최소 안정화 시간과 GPU 작업이 기록됐는지 확인하고 준비 종료→첫 측정 NVTX 간격을 출력한다.
- `DIAGNOSTIC_EVENT`의 원문을 `analysis.json`에 보존한다. 경고가 있으면 `PASS_WITH_WARNINGS`로 구분하고 보고서에 경고를 표시한다. 진단 테이블이 없을 때도 경고로 표시한다. 경고가 없다는 사실만으로 전체 GPU 추적을 보증하지 않는다.
- 메모리 작업은 첫 커널 시작보다 앞선 memset까지 포함하도록 전체 NVTX 범위의 CUDA API correlation으로 별도 집계한다. 원래 커널 구간 내 집계도 남긴다.
- 준비·측정 중 오류가 나면 capture를 닫고 실패를 전파한다. 기존 source hash 검사, GPU lock, 원본 프로세스 기록 및 정확한 GNOME portal 경로 예외를 유지한다.

조건 목록이 짧아져 v2와 선행 부하 순서는 다르다. v3 안의 일반 전·NSYS·일반 후 비교를 우선하며, v2와의 차이 전체를 시작 순서 변경의 순수 인과 효과로 해석하지 않는다. 반복 수는 각 실행에서 원래 방식으로 보정하므로 서로 같지 않을 수 있다.

## 결과 판정

`COLLECTION INTEGRITY: PASS` 또는 `PASS_WITH_WARNINGS`는 구조적 검증 결과다. `MEASUREMENT QUALITY: REVIEW REQUIRED`는 사람이 결과를 해석해야 한다는 표시이며 실행 실패가 아니다. 원본 타이밍 IQR·첫/끝 변동, 첫 블록 시간, 전후 일반 실행 차이와 Nsight 경고를 함께 검토한다. 기준을 넘거나 경고가 남으면 해결된 것으로 처리하지 않는다.

성능값은 프로파일링 없는 실행에서 얻고, Nsight는 시간 구간의 구성과 계측 영향을 진단하는 데 사용한다. Nsight 시간을 일괄 빼서 보정하거나 첫 블록을 삭제하지 않는다. 이 진단은 작은 GEMM의 관측 영향을 제거하는 수정이나 최종 y값 정의의 확정이 아니다. 후속 full 진행은 별도로 판단한다.

주요 출력은 `report.md`, `conditions.csv`, `analysis.json`, 세 실행의 `results.json`, `traces/*.nsys-rep`·`*.sqlite`, telemetry·환경·프로세스·종료 코드이다. raw와 기존 v1/v2/v5는 보존한다.

## 로컬 검사

```bash
python3 -B -m unittest discover -s timing_audit_v3 -p 'test_*.py' -v
```

CPU 검사는 capture 시작 순서, 준비와 측정의 구분, 실패 정리, 경고 보존, 첫 커널 이전 memset 집계, 누락 거부 및 ZIP의 종료 코드 보존을 확인한다. GPU 실행 검증은 결과를 받은 뒤 수행한다.

출처: [Nsight Systems capture API와 반복 수집](https://docs.nvidia.com/nsight-systems/UserGuide/), 이전 진단 원자료 `timing_audit_20260919_175307_zz6mlt.zip`. profiler를 시작한 뒤 안정화하는 순서는 이번 관측 문제에 대한 실험 설계 변경이다.
