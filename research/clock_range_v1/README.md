# 780–1200 MHz 코어 클럭 비교

현재 780 MHz 파일럿이 끝난 뒤 대표 10형상/dtype 조합으로 여덟 코어 후보를 비교한다. 메모리 6801 MHz와 기존 전력 한도 70 W를 유지한다. 마지막에 780 MHz를 다시 측정해 앞뒤 시간 변화를 확인한다. 사용자는 다른 그래픽 작업을 현재 780 파일럿 시작 전에 모두 껐다고 밝혔다. 이는 사용자 확인 사항이며, 이번 실행이 그래픽 작업 유무의 인과 효과를 분리하는 실험은 아니다.

## 실행

현재 실행의 클럭이나 소스를 바꾸지 않는다. 현재 파일럿이 끝나고 `RESTORE: COMMANDS ACCEPTED`와 `RESULT_ZIP=...`가 나온 뒤 이 비교를 실행한다.

`clock_range_2026-09-21_v1.zip`을 Ubuntu `/home/eu/gemm-research/scripts/`에 풀어 `clock_range_v1/` 폴더를 둔다. 새 폴더의 7파일만 배포한다. 기존 `gpu_sweep_v6`, `clock_audit_v1`, `clock_hold_v1`, `clock_steps_v1`, `fixed_pilot_v1`은 그대로 필요하다. 기존 42파일의 의존 hash가 다르면 설정 전에 중단한다. 기존 설치를 다시 하지 않는다.

```bash
bash /home/eu/gemm-research/scripts/clock_range_v1/launch.sh
```

위 명령은 일반 사용자 eu로 실행한다. 기존 checked runner가 `ai_env`를 활성화한다. 클럭 설정·해제에만 sudo를 사용하며 처음 암호를 요청할 수 있다. 실행 중 그래픽 작업을 다시 시작하거나 다른 GPU 실험을 겹쳐 실행하지 않는다. 현재 파일럿과 공유하는 GPU 잠금이 겹친 실행을 거부한다.

시작 시 `DEPENDENCY CHECK: PASS; 9 stages / 540 measurements / 3780 blocks / 180 mode triplets`를 확인한다. 각 단계 로그에는 `[현재/60]` 진행 상황이 남는다. 예상 60~75분은 이전 네 단계 실측 27분 50초를 기준으로 잡은 계획값이다. 이번 아홉 단계의 실제 시간은 아직 확인하지 않았으며 각 worker 제한은 기존 20분이다.

## 순서와 범위

| 단계 | 코어 요청 MHz | 역할 |
|---|---:|---|
| s0_780_before | 780 | 시작 기준 |
| s1_810 | 810 | 낮은 후보 |
| s2_840 | 840 | 낮은 후보 |
| s3_870 | 870 | 900 아래 후보 |
| s4_900 | 900 | 과거 일부 FP32에서 미유지된 후보 재확인 |
| s5_960 | 960 | 900 위 후보 |
| s6_1050 | 1050 | 900 위 후보 |
| s7_1200 | 1200 | 900 위 후보 |
| s8_780_after | 780 | 마지막 기준 재측정 |

각 단계는 기존 `clock_lower_v1`의 10형상/dtype × allocate/prealloc/Graph × 두 round, 60측정이다. 전체 540측정, 3780블록, 180개 세 방식 묶음이다. 전체 출력 정확성 검사 108측정, 256원소 표본 검사 432측정으로 구성한다.

기존 900 MHz에서 REVIEW였던 FP32 네 조건 `(1888,832,632)`, `(3216,520,776)`, `(1016,2048,504)`, `(1999,646,425)`, 기존 대조 `(1888,832,632)` FP16과 `(392,2776,80)` FP32, 4096³과 64³ 각각 FP16/FP32를 사용한다. 전체 24형상 파일럿이나 새로운 holdout 조건은 열지 않는다.

입력 seed, 저장된 Graph 묶음, v6 Timer와 정확성 함수, 단계별 새 프로세스와 180초 예열, 조건별 3000 event ms 안정화, 100 ms 목표 7블록, IQR·drift·round 차이 5%, 50 ms telemetry를 유지한다. 방식 시작 위치를 형상별 순환하고 두 번째 round에서 형상·방식 순서를 뒤집는다. allocate도 기존 캐싱 할당자를 쓰며 방식 차이를 순수 할당 비용이나 CPU 대기로 해석하지 않는다.

## 결과 판정

종료 시 `RESTORE: COMMANDS ACCEPTED`, `CLOCK RANGE: PASS/REVIEW/FAIL`, `RESULT_ZIP=...`를 확인한다. 초기 검사나 worker 실패는 traceback과 `EXPERIMENT: FAIL`로 끝날 수 있으며 가능한 로그를 ZIP에 보존한다. 결과는 `/home/eu/gemm-research/scripts/clock_range_v1/runs/clock_range_<실행 시각>_<고유값>.zip`과 `.zip.sha256`에 남는다. exit 0은 자동 검토 표식 없음, 2는 REVIEW, 1은 실행·검증·해제 실패이며 처리 가능한 중단은 130이다.

목표 클럭과 다른 표본, 순간 전력·열 제한, 내부 표본 누락, 시간 변동을 REVIEW로 보존한다. 높은 클럭의 REVIEW를 숨기거나 낮은 클럭의 성능으로 바꿔 적지 않는다. 수집 자체가 성공하면 다음 후보를 계속 측정한다. 설정 거부·worker 오류·정확성 실패는 후속 부하를 중단한다. 이때 완료된 단계 원자료와 실패 요약을 남기며 전체 비교가 완료됐다고 표시하지 않는다.

요청 클럭은 유지가 검증된 값이 아니다. 설정 성공과 실제 부하 중 유지를 나눠 확인한다. NVIDIA도 전력 제한이 요청 클럭보다 실제 클럭을 낮출 수 있다고 설명한다. [NVIDIA nvidia-smi 문서](https://docs.nvidia.com/deploy/nvidia-smi/).

누적 카운터의 과거 불일치는 미해결로 남기고 시간/관측 클럭 판정과 따로 기록한다. 순간 비활성 중 증가, 표본 간격보다 큰 누적 증가량을 제한 시간 비율로 환산하지 않는다. 약 100 ms 묶음 평균은 단일 GEMM 지연 스파이크의 부재를 증명하지 않는다. 그래픽 작업 종료만으로 이전 900 MHz 미유지 원인이 해결됐다고 판단하지 않는다.

`report.md`와 `summary.json`은 단계별 판정, `measurements.csv`와 `review_queue.csv`는 측정별 검토, `paired_metrics.csv`는 세 방식의 시간을 담는다. `repeat_780.csv`는 시작/마지막 780의 두 round 중앙값을 비교한다. 차이 5% 초과는 REVIEW다. 원자료는 단계별 디렉터리, 전체 분석 근거는 `analysis.json`과 `clock_windows.json`에 있다.

공유 GPU 잠금 안에서 worker 종료 후 두 클럭 고정 해제를 각각 시도한다. `RESTORE: FAILED`이면 `restore.json`과 출력된 수동 해제 명령을 확인한다. 기본 자동 클럭으로 해제하며 이전 수동 고정을 재적용하지 않는다. 사전 진입 실패는 기존 설정을 건드리지 않는다. 전원 차단·SIGKILL은 종료 처리가 불가능할 수 있다.

780 MHz는 비교 기준이다. 이번 비교의 `selected_clock_mhz`는 새 후보를 아직 선택하지 않았다는 뜻으로 null이며 `validated_for_full=false`, `full_sweep=HOLD`다. 결과 검토 뒤 유지되는 후보의 전체 192측정 파일럿을 판단한다. 이번 ZIP은 새 Nsight 캡처나 본수집을 실행하지 않는다. 시간 변화에는 단계 순서·프로세스·시간대 효과가 남으며 두 round는 독립 세션이 아니다.

로컬 검사는 `python3 -B test_lower.py`로 실행한다. 합성 데이터와 가짜 GPU 명령을 사용하므로 실제 GPU 검증을 대신하지 않는다.
