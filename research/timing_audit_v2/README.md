# v5 실제 측정 블록 진단

GPU 컴퓨터의 기존 `gpu_sweep` v5를 읽어 실행하는 별도 진단이다. v5 코드·manifest·pilot 원자료는 수정하지 않는다. 새 측정값은 모두 진단용이며 학습 데이터에 자동 합치지 않는다.

## v2 수정

사용자가 `xdg-desktop-portal-gnome` 때문에 실행이 막힌다고 보고했다. v5의 기존 검사에는 GNOME portal 경로가 없으므로 정상 데스크톱 서비스도 경쟁 작업으로 차단할 수 있었다. 이 진단의 공통 프로세스 검사에서 `/usr/libexec/xdg-desktop-portal-gnome`만 추가로 허용한다. 이 경로는 [Ubuntu 26.04 패키지 파일 목록](https://packages.ubuntu.com/resolute/amd64/xdg-desktop-portal-gnome/filelist)과 대조했다. 사용자 오류 전문에서도 같은 경로(PID 6470, 9 MiB)가 확인됐다. 실패는 부모 실행의 측정 전 검사에서 발생했다. 다른 경로나 원인이면 계속 차단한다.

원본 GPU 프로세스 목록은 그대로 저장하고 `*_classification.json`에 허용한 portal 행과 차단한 행을 따로 남긴다. portal의 GPU 영향이 없다는 뜻은 아니며 Nsight와 전후 일반 실행을 함께 검토한다. 부모 실행과 모든 worker의 검사에 동일하게 적용한다. 다른 Python/CUDA 작업·동명 다른 경로·접미사가 붙은 실행 파일·조회 실패·손상된 CSV는 계속 차단한다. v5의 소스와 hash·타이머·프로토콜은 바꾸지 않는다. 서비스 종료나 재설치를 요구하지 않는다. 기존 v1 결과와 ZIP은 보존하며 v2도 매번 새 출력 폴더를 쓴다.

## 실행

압축을 풀어 생긴 `timing_audit_v2` 폴더를 GPU 컴퓨터의 `/home/eu/gemm-research/scripts/`에 놓는다. 그 옆에 기존 `/home/eu/gemm-research/scripts/gpu_sweep/` v5가 있어야 한다. full_v5는 아직 실행하지 않은 상태에서 이 진단을 먼저 한다.

아래 명령 하나를 실행한다.

```bash
bash /home/eu/gemm-research/scripts/timing_audit_v2/launch.sh
```

기존 위치가 다를 때만 launch.sh의 첫 인자로 v5 폴더의 절대경로를 지정한다. 기존 `run_research_checked.sh`가 `ai_env`와 CUDA·도구 환경을 확인한다. 재설치하거나 GPU 클럭·전력 설정을 바꾸지 않는다.

정상 시작 시 `AUDIT_DIR=...`, `V5 SOURCE CHECK: PASS`, `ENV CHECK: PASS`, `Starting plain_before`가 나온다. 각 실행의 상세 진행은 출력된 `plain_before.log`, `nsys.log`, `plain_after.log`에 기록된다. 한 실행마다 초기 예열 3분이 있다.

세 번의 예열만 9분이고 측정·프로파일 보고서 변환이 추가된다. 계획용 예상은 약 12~20분이며 GPU에서 확인한 소요시간은 아니다. 터미널을 유지하고 다른 GPU 계산을 동시에 실행하지 않는다. 기존 v5와 같은 GPU lock 및 경쟁 프로세스 검사를 재사용한다. GNOME 등의 기존 허용 데스크톱 프로세스는 로그에 남으며 영향이 없다고 가정하지 않는다.

결과는 매번 새 `timing_audit_v2/runs/timing_audit_날짜_시각_고유값/`에 저장된다. 완료 후 같은 위치에 결과 ZIP과 SHA-256 파일을 자동으로 만들고 `RESULT_ZIP=...`를 출력한다. 성공·실패 모두 로그와 실제 종료 코드를 보존한다. 중단 시 이미 저장된 조건은 남고 다음 실행은 새 폴더를 사용한다. 파일럿이나 full을 자동으로 시작하지 않는다.

## 조건과 비교

8조건을 동일한 순서로 일반 실행 → Nsight 실행 → 일반 재실행한다. 한 조건당 7개 측정 블록으로, Nsight에 대응할 블록은 56개다.

| M,N,K | 정밀도 | 실행 방식 | 확인할 항목 |
|---|---|---|---|
| 64,64,64 | FP16·FP32 | prealloc | 짧은 GEMM의 제출 간격 |
| 128,128,128 | FP16 | prealloc | 작은 형상의 기본 실행 |
| 3216,520,776 | FP16 | prealloc | 주 측정 REVIEW 형상 |
| 4096,4096,4096 | FP16·FP32 | prealloc | 큰 GEMM의 비교 기준 |
| 64,64,64 | FP16·FP32 | graph | 같은 작은 형상의 Graph 대조 |

각 실행은 기존 180초 예열을 수행한다. 측정은 기존 `measure.measure`와 `Timer.run`, `Timer.block`을 재사용하며, 입력 seed·출력 재사용·조건별 3초 안정화·100 ms 목표·7블록·반복 수 보정·5% 변동 기준·측정 후 정확성 검사를 유지한다. Nsight 실행만 `Timer`를 상속해 prepare 직후 수집을 시작하고 실제 block 바깥에 NVTX 표식을 붙인다. GEMM마다 Python 콜백이나 NVTX 호출을 추가하지 않는다. 마지막 블록이 끝나면 수집을 멈춘다. 조건별 trace 8개를 만들며 워밍업·정확성 검사는 수집 구간에서 제외한다.

반복 수와 Graph node 수는 v5 방식으로 각 실행에서 결정하고 원자료에 남긴다. Nsight 때문에 선택된 반복 수가 달라질 수 있다. 조건은 같지만 호출 수까지 고정한 실험은 아니며, Graph node 수가 달라지면 비교표에 표식을 남긴다. 전후 일반 실행의 차이도 함께 확인해야 프로파일러 영향과 세션 변동을 구분할 수 있다.

이번 1차 진단은 주 측정 루프를 확인한다. pressure 대조의 개별 event 구간과 연결 조립 실험은 포함하지 않는다. 커널 내부의 변화가 남으면 해당 조건의 NCU 지표를 다음 단계에서 확인한다. NCU를 켠 상태의 CUDA Event 시간을 일반 latency로 사용하지 않는다.

## 결과 읽기

- `report.md`, `conditions.csv`: 일반 전·Nsight·일반 후 시간과 커널 사이 간격 비중.
- `analysis.json`: 블록별 CUDA API·커널 이름·커널 수·메모리 복사 기록·가장 큰 간격과 다음 CPU launch의 시점·변동 표식.
- `traces/*.nsys-rep`, `*.sqlite`: Nsight UI에서 직접 확인할 원본 trace와 분석용 export.
- 실행별 `results.json`: 안정화·보정·7블록·event/host 시간·정확성 원자료.
- `telemetry.csv`, 환경·프로세스 기록, 명령·출력·종료 코드: 재현과 간섭 검토 근거.

`COLLECTION INTEGRITY: PASS`는 24개 조건 실행과 56개 trace 블록의 구조·정규화·커널 대응을 확인했다는 뜻이다. `MEASUREMENT QUALITY: REVIEW REQUIRED`는 타임라인의 해석이 남았다는 뜻이며 실행 실패가 아니다. 수치 검사 실패, 블록·커널 누락, source 불일치, 도구 오류는 종료 코드 1로 실패 기록을 남긴다. CPU 검사는 실제 GPU 동작을 보증하지 않는다.

NVTX는 CPU 범위다. 그 안에는 event 초기화와 마지막 동기화도 있으므로 CUDA Event의 GPU 시작·끝과 같다고 보지 않는다. 커널 사이 간격은 첫 커널 시작~마지막 커널 끝에서 커널 구간의 합집합을 뺀 값이다. 전체 GPU의 유휴율이나 본측정 오차율이 아니며, 다른 프로세스의 GPU 작업이 모두 기록되는 것도 아니다. Event 시간에서 CUPTI 커널 시간을 뺀 차이를 전부 launch 비용으로 해석하거나 본측정에 일괄 보정하지 않는다.

프로파일링 자체가 실행에 영향을 줄 수 있다. 원본 v5 소스와 GPU 환경을 SHA-256 및 저장된 환경 signature로 대조하고, trace SQLite는 immutable/read-only로 열어 읽기 전후 hash가 같은지 확인한다. 새 진단도 GPU에서 실행한 결과를 검토하기 전까지 full 진행 근거가 되지 않는다.

## 로컬 검증

```bash
python3 -B -m unittest discover -s timing_audit_v2 -p 'test_*.py' -v
```

출처: [Nsight Systems 수집 구간과 repeat 옵션](https://docs.nvidia.com/nsight-systems/UserGuide/), [Nsight SQLite의 NVTX와 CUDA correlation](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html), [Nsight Compute 시간 측정의 영향](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#range-and-precision).
