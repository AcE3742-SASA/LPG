# 두 FP32 형상의 전력 제한 원인 분리 v1

대상은 `(4096,4096,72)`와 `(4096,4096,127)`이다. 같은 회복 조건에서 K=72의 제한이 다시 나타나는지, 반복 실행에 따라 달라지는지, 실제 선택된 커널과 연산·메모리 활동에 차이가 있는지 확인한다. 기존 전수 자료를 교체하는 재측정이 아니라 별도 진단이다.

## 실행

ZIP의 `power_shape_diag_v1/` 폴더를 연구 PC의 `/home/eu/gemm-research/scripts/` 아래에 둔다. 기존 tmux에서 다음 명령을 실행한다.

```bash
bash /home/eu/gemm-research/scripts/power_shape_diag_v1/launch.sh
```

기존 `ai_env`, CUDA, Nsight 설치를 사용한다. 검산된 기존 코드 110파일은 이 실행본의 `dependencies/`에 원래 바이트와 상대 경로로 포함했다. 이전 실행본을 덮어쓰거나 재설치할 필요가 없다. 전체 설치물이 아니라 연구 코드의 보존 사본이다.

매번 새로운 `runs/power_shape_diag_<시각>_<고유값>/`을 만들며 기존 결과를 재개하거나 덮어쓰지 않는다. 출력·로그·결과 ZIP은 모두 해당 연구 폴더 안에 남는다. 끝에 출력되는 `RESULT_ZIP=`의 ZIP을 전달하면 된다. 보통 15~30분의 실행 예산을 잡되, 이는 실측 소요 시간이 아니다. 회복 대기와 프로파일러 재실행 횟수에 따라 더 걸릴 수 있다.

## 비교 조건

| 항목 | 고정하거나 기록하는 조건 |
|---|---|
| GPU·소프트웨어 | 기존 검토 기준의 GPU UUID, 드라이버, PyTorch/CUDA, 정밀도·환경 서명 대조 |
| 클럭·전력 | SM 720 MHz, 메모리 6801 MHz, 현재 70 W 확인. 전력 한도 변경 없음 |
| 연산 | FP32 IEEE, `prealloc`, 기존 형상별 입력 seed와 정확성 기준 |
| 입력·할당 | 한 묶음의 16회 동안 같은 A·B·C 재사용. 할당과 첫 정확성 검사는 회복 대기 전에 수행 |
| 라운드 | 4회. A=K72, B=K127일 때 AB / BA / BA / AB |
| 회복 | 각 형상 묶음 시작 전 55°C 이하의 센서 표본이 10초 이어질 때 시작. 최대 300초 뒤 미달이면 중단 |
| 시작 온도 대조 | 같은 라운드의 두 시작 온도 차이가 2°C를 넘으면 `START_TEMPERATURE_UNMATCHED` 표시 |
| 실행 횟수 | 각 묶음 16회. 첫 1회는 회복 후 첫 실행, 이후 15회는 이어서 실행. 8묶음 × 16 = 128회 |
| 시간 측정 | 매회 기존 3000 event-ms 안정화·보정 후 약 100 ms 블록 7개. 전역 180초 예열은 회복 조건을 위해 생략 |
| 센서 | 50 ms 요청 간격. 클럭·전력·온도·제한 플래그·카운터와 지원되는 추가 센서 |

회복 후 첫 실행에도 기존 3초 안정화가 들어간다. 따라서 ‘차가운 상태의 즉시 지연시간’을 뜻하지 않는다. 최초 probe, 안정화, 보정, 본 측정 블록의 호스트 구간을 모두 남겨 제한이 어느 단계에서 시작했는지 대조한다. 55°C는 과거 문제 시작 표본인 65°C보다 낮게 둔 이번 진단의 회복 기준이며, 하드웨어의 열 제한 온도가 아니다.

각 측정 후 정확성 검사·다른 GPU 프로세스 검사·저장을 수행한다. 이 짧은 간격이 있는 반복 부하이며, 완전히 끊기지 않는 GPU 부하는 아니다. `gap_from_previous_trial_ms`와 센서 원자료로 실제 간격을 확인한다. 시작 온도가 기준 이내라도 팬·전압·메모리 상태까지 같다고 가정하지 않는다.

## 커널·활동 진단

일반 128회 수집 뒤 각 형상에 Nsight Systems 1회와 Nsight Compute 1회, 총 4개 프로파일을 별도로 수집한다. 각각 새 worker에서 같은 입력을 만들고 회복·안정화를 수행한다.

- Systems는 `cudaProfilerApi` 구간 안의 GEMM 20호출을 추적하고 커널 이름·호출 수·GPU 시간 요약을 저장한다. CPU sampling은 끈다.
- Compute는 같은 구간 안의 GEMM 1호출을 대상으로 `SpeedOfLight`, `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `Occupancy`, `LaunchStats`를 수집한다. 단일 GEMM이 여러 커널을 쓰면 모두 대상이다.
- Compute는 `--clock-control none --cache-control none`으로 실행한다. 계수기를 수집하면서 여러 번 replay할 수 있어, 프로파일링 시간·전력은 일반 측정의 값과 합치지 않는다.
- `.ncu-rep`와 `.ncu-repz`를 모두 인식하며, 보고서 파일뿐 아니라 실제 커널 지표가 있어야 완료로 인정한다. 설치 누락·권한 오류는 로그를 남기고 중단한다.
- `power.draw.average`, `power.draw.instant`, `fan.speed`, `voltage.graphics`, `utilization.memory`를 먼저 개별 조회한다. 지원되지 않는 값은 `sensor_capabilities.json`에 명시하며, 없는 값을 추정하지 않는다.

커널 이름·타일·연산/메모리 지표 차이는 설명 후보를 좁히는 근거다. 커널을 강제로 바꾸는 실험은 아니므로 특정 커널이 전력 차이의 유일한 원인이라고 확정하지 않는다. 전압을 읽지 못하면 누설 전력·전압 기여도도 미확정으로 남긴다.

## 결과 해석과 중단

`summary.json`의 `collection=COMPLETE`는 128회와 4개 프로파일 수집 완료를 뜻한다. 이 진단은 해석이 필요하므로 정상 완료도 `gate=REVIEW`, 종료 코드 2이다. 기존 전수 측정의 품질 판정이나 최종 54건 검토 목록을 바꾸지 않는다. 단순히 REVIEW가 다시 나왔다는 이유로 실험이 실패한 것은 아니다.

K72가 맞춘 시작 온도에서 반복적으로 먼저 제한되고 K127은 그렇지 않다면 형상에 따른 실행 경로·활동 차이를 지지한다. 두 형상 모두 처음에는 안정적이다가 뒤쪽에서 제한되면 지속 부하와 온도·제어 이력의 기여를 지지한다. 라운드 순서나 시작 온도에 따라 달라지면 그 영향을 먼저 분리해야 한다. 이런 양상은 함께 나타날 수도 있다.

SW power-cap·720 MHz 미만 표본은 진단 대상이므로 계속 기록한다. 정확성/CUDA 오류, GPU 환경 불일치, 70 W 설정 변경, 열 제한·예상하지 않은 제한 비트, 85°C 이상, 센서 중단·역순 시각·5초 초과 공백은 중단한다. 메모리 클럭 불일치는 별도 표시하며 조건을 맞춘 비교로 인정하지 않는다. 센서 표본의 활성 비율을 실제 제한 시간 비율로 바꾸지 않는다. 50 ms는 조회 요청 간격이지 전력 센서의 갱신 주기가 아니다.

실행을 중단하거나 일부 클럭 설정만 성공해도 worker 종료 후 그래픽·메모리 클럭 해제를 모두 시도한다. 해제는 기존 코드와 같이 기본 자동 클럭으로 돌아가는 명령이며, 이전 수동 고정값을 재설정하는 것은 아니다. `RESTORE: COMMANDS ACCEPTED`는 명령 수락만 뜻한다. 종료 로그와 가능한 부분 결과도 ZIP으로 묶는다.

## 보존 파일과 로컬 검증

반환 ZIP에는 manifest·프로토콜·입력 seed·소스/의존 해시와 사본, 단계별 환경·명령·로그, 50 ms 센서 CSV, 회복 시작 조건, trial별 시간 블록·정확성, Nsight 원본, 요약과 클럭 해제 증거를 담는다. `all_event_blocks.json`은 최초 probe도 포함한다. 프로파일러 실행 자료는 `nsys_k*`, `ncu_k*`에 분리된다.

```bash
python3 -B power_shape_diag_v1/test_diagnostic.py
python3 -B power_shape_diag_v1/diagnose.py verify
```

위 검사는 CPU에서 조건·seed·실제 worker 제어 흐름, 원래 Timer의 3초 안정화와 7블록, 프로파일러 구간, 회복 조건·중간 온도 상승·시간 초과, 센서 오류, 부분 설정 실패·중단 시 두 클럭 해제, 128회 반환 자료 분석, 시작 온도 불일치와 프로파일 누락, ASCII 출력을 검증한다. CUDA 실행과 실제 커널/센서 지원 여부는 연구 PC에서 확인해야 한다.

## 근거

[NVIDIA SMI의 전력·클럭 정의](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), [프로파일링의 clock/cache/replay 제어](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)를 기준으로 옵션과 해석 범위를 정했다. 이번에 비교할 기존 자료의 ZIP·manifest SHA-256은 `protocol.json`에 기록했다.
