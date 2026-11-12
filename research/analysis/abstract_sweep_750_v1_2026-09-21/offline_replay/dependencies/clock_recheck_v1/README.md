# 780 MHz 클럭 문제 재검증

기존 `clock_range_v1`의 클럭제어, worker와 분석기를 재사용해 780→750→780 MHz를 비교한다. 메모리는 6801 MHz, 기존 전력 한도는 70 W로 유지한다. 새 코어 운용값을 자동 선택하거나 200형상 수집을 재개하지 않는다.

## 실행

`clock_recheck_2026-09-21_v1.zip`의 `clock_recheck_v1` 폴더를 Ubuntu `/home/eu/gemm-research/scripts/` 아래에 둔다. 기존 의존 폴더와 `ai_env`는 그대로 필요하며 재설치하지 않는다.

```bash
bash /home/eu/gemm-research/scripts/clock_recheck_v1/launch.sh
```

현재 tmux 세션과 창을 그대로 사용한다. tmux 밖이면 현재 터미널에서 실행하고 SSH 연결을 유지한다. 종료 후 Enter 입력 없이 프롬프트로 돌아온다. 기존 240건 결과를 읽거나 이어쓰지 않으며 실행마다 새 결과 폴더를 만든다.

시작 출력은 `DEPENDENCY CHECK: PASS; 3 stages / 108 measurements / 756 blocks / 36 mode triplets`다. 이어 각 단계의 로그 경로와 180초 예열 안내가 나온다. 단계별 로그에 `[1/36]`부터 진행 기록이 남는다. 계획 예산은 준비와 검산을 포함 약 20~25분이다. 기존 작업당 약 3.9초와 단계별 180초 예열로 계산하면 약 16분이며 실제 시간은 달라질 수 있다.

## 측정 구성

| 순서 | 코어 | 목적 |
|---|---:|---|
| s0_780_before | 780 MHz | 문제 재현 |
| s1_750 | 750 MHz | 더 낮은 후보의 유지 여부 |
| s2_780_after | 780 MHz | 처음과 같은 조건으로 돌아와 앞뒤 차이 확인 |

각 단계는 다음 세 형상 각각 FP32와 FP16, allocate/prealloc/Graph, 두 라운드로 36건이다.

- `(3128,680,488)`: 200형상 부분수집에서 FP32의 세 방식 모두 765 MHz와 순간 전력 제한이 관측된 형상.
- `(4096,4096,4096)`: 큰 행렬 대조.
- `(64,64,64)`: 작은 행렬 대조.

문제 형상의 입력 seed와 Graph 묶음은 반환 ZIP `abstract_200_780_20260921_083037_8b68d7.zip`의 실제 저장값을 사용한다. FP32는 seed 1456857178, Graph당 GEMM 5개이고 FP16은 seed 2682971897, Graph당 GEMM 20개다. 원본 ZIP SHA-256은 `f722419bea5b83b54894a4d0db8efdd9c680d63f6e2f1e25c14a7110c759ac20`이다. 대조 형상은 기존 고정 seed와 Graph 묶음을 유지한다.

첫 라운드는 문제 형상부터 시작하고 두 번째는 형상과 방식 순서를 뒤집는다. v6 Timer, 단계별 새 프로세스와 180초 예열, 각 측정의 3,000 event ms 안정화, 약 100 ms 목표 7블록, 정확성 검사와 5% 변동 기준을 유지한다. 전체 출력 비교 36건, 표본 비교 72건이 계획된다.

## 진단의 종료 규칙

이 실행은 클럭 후보를 비교하는 진단이다. 목표 클럭 이탈, 순간 전력 제한과 시간 변동은 REVIEW로 남기고 다음 후보까지 수집한다. 최종 단계별 `timing_and_sampled_clocks_pass`에서 실제 유지 여부를 판단한다. 이전 200형상 본수집의 자동 중단 규칙이나 기준을 변경한 것은 아니다.

계산 정확성 실패, 설정 거부, 다른 GPU 작업 감지, 센서 프로세스 종료와 worker 오류는 후속 부하를 중단한다. GPU 잠금과 sudo 권한 갱신을 유지한 상태에서 worker 종료 뒤 코어와 메모리 고정을 각각 해제한다. 초기 검사에서 실패해 설정을 건드리지 않았으면 해제 미시도로 기록한다.

`RESTORE: COMMANDS ACCEPTED`, `CLOCK RECHECK: PASS/REVIEW/FAIL`, `RESULT_ZIP=...`를 확인한다. exit 2는 검토 표식, exit 1은 실행/검증/해제 실패이며 처리 가능한 중단은 130이다. 실패나 중단도 가능한 로그와 완료된 기록을 ZIP에 남긴다. 새 결과는 `clock_recheck_v1/runs/clock_recheck_날짜_식별자.zip`으로 만들어지며 그 ZIP을 반환한다. 소스와 로그를 ZIP 안에 보존하고 SHA-256 파일도 생성한다.

## 결과 해석

`summary.json`, `report.md`, `measurements.csv`, `review_queue.csv`, `paired_metrics.csv`, `repeat_780.csv`와 단계별 원자료를 함께 확인한다. 누적 SW Power Cap 카운터 해석은 여전히 미해결이며 순간 활성/클럭 판정과 분리해 남긴다. 표본 개수나 누적 카운터를 전력 제한 시간 비율로 환산하지 않는다.

이 진단은 이전 163번째 작업에 도달하기까지의 전체 부하 이력을 재현하지 않는다. 문제가 재현되지 않아도 앞선 클럭 이탈을 없던 일로 처리하지 않는다. 세 형상의 통과는 200형상 전체의 통과가 아니며, `validated_for_full=false`, `full_sweep=HOLD`, `selected_clock_mhz=null`을 유지한다. 실제 GPU 결과를 검토한 뒤 후속 수집 조건과 재측정을 결정한다.

## 로컬 검증

```bash
python3 -B clock_recheck_v1/lower.py verify
python3 -B clock_recheck_v1/test_lower.py
```

합성 데이터와 가짜 GPU 명령으로 입력/분모/누락/클럭 판정, 오류와 중단 후 해제, 현재 tmux 재사용, ZIP 포장을 검사한다. Mac 검사 통과는 새 GPU 실험 완료를 뜻하지 않는다.
