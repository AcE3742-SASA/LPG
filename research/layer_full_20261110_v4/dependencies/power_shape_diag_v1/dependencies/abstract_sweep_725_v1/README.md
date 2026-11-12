# 725 MHz 요청 / 720 MHz 대체 허용 — GEMM 점검과 200형상 수집

사용자가 코어 725 MHz를 선택했고, 정확히 지원되지 않으면 지원 여부를 확인한 뒤 720 MHz로 진행하도록 허용했다. 메모리 6801 MHz와 기존 전력 한도 70 W는 유지한다. 실제 GPU의 지원 목록은 Ubuntu 실행 시 조회한다. Mac에서 해당 GPU의 725/720 MHz 지원 여부나 측정 안정성을 확인한 것은 아니다.

## 먼저 실행할 명령

ZIP 안의 `abstract_sweep_725_v1` 폴더를 Ubuntu `/home/eu/gemm-research/scripts/`에 넣는다. 기존 `abstract_sweep_750_v1` 등 의존 폴더와 `ai_env`는 유지한다. 기존 결과 폴더를 덮어쓰지 않는다.

```bash
bash /home/eu/gemm-research/scripts/abstract_sweep_725_v1/launch.sh check
```

현재 tmux 세션과 창에서 그대로 실행한다. 새 tmux를 만들거나 종료 후 Enter 입력을 기다리지 않는다. `check`를 생략해도 점검만 실행한다. 예상 약 10~15분이며 실제 소요시간은 형상과 시스템 상태에 따라 달라진다.

지원 graphics clock 목록에 725가 있으면 725, 없고 720이 있으면 720을 선택한다. 둘 다 없거나 조회 결과를 읽을 수 없으면 클럭 설정 전에 중단한다. 명령에 입력한 725를 실제 725 동작으로 간주하지 않는다. NVIDIA의 `-lgc`는 가까운 지원 주파수를 설정할 수 있기 때문이다. [공식 nvidia-smi 설명](https://docs.nvidia.com/deploy/nvidia-smi/)

실행 중 다음과 같이 선택값을 출력한다.

```text
Requested: 725 MHz; selected exact supported target: 720 MHz
```

이는 지원 목록에서 720을 선택했다는 예시다. 부하 중 실제 클럭은 이후 센서 기록으로 별도 검증한다. 요청값·허용 대체값·선택값·지원 목록 조회 원문을 `clock_selection_*.json`에 보존하고, `manifest.json`의 `selected_clock_mhz`와 `config.sm_mhz`에 실제 선택값을 고정한다. 재개와 본측정에서는 기존 결과 또는 검토한 점검의 클럭을 유지하며 임의로 바꾸지 않는다.

## 점검 구성

아래 4형상 × FP32/FP16 × allocate/prealloc/Graph × 두 회차 = **48측정·336블록·16묶음**이다. 두 회차는 별도 worker에서 실행하며 각 worker는 기존대로 180초 예열한다.

1. `(2329,3998,66)` — 이번 750 MHz에서 문제가 난 형상.
2. `(3128,680,488)` — 앞선 780 MHz에서 문제가 난 형상.
3. `(4096,4096,4096)` — 큰 행렬 대조.
4. `(64,64,64)` — 작은 행렬 대조.

첫 회차는 새 문제 형상을 먼저 확인하고, 두 번째는 순서를 뒤집는다. 기존 Timer·CUDA Event·측정별 3,000 event ms 안정화·약 100 ms 목표 7블록·정확성 검사·5% 기준을 유지한다. Graph 묶음은 기존 50조건에 새 문제 형상의 기록값(FP32 8개, FP16 14개)을 추가해 52조건을 고정했다. 이 값의 재사용은 725/720 MHz 품질 통과를 뜻하지 않는다.

종료하면 `runs/check_low_clock_날짜_식별자.zip`과 SHA-256 파일이 생기고 `RESULT_ZIP=...`를 표시한다. **그 ZIP을 반환해 점검 결과를 검토한 뒤 본측정으로 진행한다. 자동으로 본측정을 시작하지 않는다.** 누적 전력 카운터 해석 미해결 때문에 정상 완료에도 전체 `REVIEW`·exit 2가 나올 수 있으므로 완료 건수, 개별 측정 판정, 관측 클럭, 회차 간 변동과 해제 근거를 함께 확인한다.

## 본측정 구성과 진입 조건

본측정은 기존과 같은 **200형상 × 두 정밀도 × 세 방식 × 세 회차 = 3,600측정·25,200블록·1,200묶음**이다. `selection.json`, 선정 함수, task ID·입력 seed·순서·split은 이전 750/780 MHz 본측정과 동일하다. 본측정 예상 예산은 약 4~6시간이다.

검토를 마친 뒤 쓰는 명령 형식은 다음과 같다. `--review-note`에는 실제 검토 내용을 입력한다.

```bash
bash /home/eu/gemm-research/scripts/abstract_sweep_725_v1/launch.sh full \
  --review-note '점검 결과를 검토한 내용'
```

기본 점검 폴더는 `runs/check_low_clock`, 본측정 폴더는 `runs/abstract_200_low_clock`이다. 별도 경로는 `--check PATH`와 `--output PATH`로 지정할 수 있다. full은 점검 manifest와 소스, 모든 측정·센서·정확성·두 worker 분리·해제 근거를 다시 검산한다. 분석 산출물은 임시 복사본에서 재생성한다. 미완료, 시간·클럭·정확성 문제, 클럭 불일치 또는 검토 메모 누락이면 본측정에 진입하지 않는다. 과거 750 MHz 점검이나 실패 결과로 새 클럭 본측정의 진입을 대신할 수 없다.

## 중단·재개와 보존

실행 터미널에서 Control+C를 한 번 누르면 현재 측정 저장과 worker 종료를 기다린 뒤 코어·메모리 고정을 해제하고 가능한 자료를 ZIP으로 만든다. 쉘 프롬프트와 `RESULT_ZIP`가 돌아올 때까지 기다린다. tmux 분리는 Control+B를 누르고 손을 뗀 뒤 D이며, 이는 측정 중단이 아니다.

정상 저장된 부분 실행은 같은 명령으로 재개한다. 저장된 센서 품질을 다시 확인해 클럭·센서·데이터 문제가 있으면 `resume refused`로 차단한다. 기존 raw FAIL은 검토 후 명시적 `--retry-failed`가 필요하다. 750/780 MHz 결과를 새 실행에 이어 붙이거나 REVIEW를 삭제하지 않는다.

정확성·실행 FAIL은 저장 후 중단하고, 시간 변동만의 REVIEW는 기록을 남기며 수집한다. 기존대로 최대 120작업마다 센서를 검산해 클럭 이탈·순간 전력 제한·센서 문제가 있으면 다음 구간 전에 멈춘다. 전력 한도를 바꾸지 않으며, 다른 GPU 작업을 강제로 종료하지 않는다. 코어·메모리 해제와 결과 ZIP 생성 경로는 기존 코드를 재사용한다.

## 로컬 확인

```bash
python3 -B abstract_sweep_725_v1/run.py verify --size check
python3 -B abstract_sweep_725_v1/run.py verify --size full
python3 -B abstract_sweep_725_v1/test_low_clock.py
```

로컬 검사 통과는 새 GPU 실측 성공을 의미하지 않는다. 작은 점검의 통과 역시 200형상 전체 안정성을 보장하지 않으며, 본측정에서도 같은 품질 감시를 유지한다. 원래 전체 10,000조건 재측정 계획과 과거 원자료는 별도로 보존한다.
