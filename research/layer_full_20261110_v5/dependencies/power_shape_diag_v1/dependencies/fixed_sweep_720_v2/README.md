# 720 MHz 전수 수집: 클럭 이탈 표시 후 계속

사용자 요청으로 클럭 이탈 시 중단하던 v1을 변경한 별도 실행본이다. 720/6801 MHz와 기존 전력 한도 70 W, 113,382작업, 입력, 순서, 시간 측정과 정확성 기준은 그대로다. 기존 `fixed_sweep_720_v1`과 그 결과는 변경하지 않는다.

## 실행 중인 v1에서 전환

1. v1이 실행 중인 tmux에서 Control+C를 한 번 누른다. 현재 측정 저장, 클럭 해제와 ZIP 생성이 끝나 `RESULT_ZIP=`과 셸 프롬프트가 나올 때까지 기다린다. 강제 종료하지 않는다.
2. ZIP의 `fixed_sweep_720_v2` 폴더를 Ubuntu `/home/eu/gemm-research/scripts/`에 넣는다. 기존 의존 폴더, v1과 완료한 200형상 결과 ZIP은 유지한다.
3. 같은 tmux에서 아래 명령을 실행한다.

```bash
bash /home/eu/gemm-research/scripts/fixed_sweep_720_v2/launch.sh
```

기존 기본 위치인 `../fixed_sweep_720_v1/runs/full_720/`가 있으면 정상 중단과 소스, 데이터, 해제 증거를 확인한 뒤 새 `runs/full_720/`로 복사해 이어받는다. 원본 폴더는 그대로 보존한다. 실행 중이거나 ZIP 생성 중인 v1은 잠금 검사에서 거부된다. 저장된 task ID와 Graph 묶음 선택을 유지하고 이미 기록된 task는 다시 측정하지 않는다. v2를 다시 실행하면 그 v2 결과에서 이어간다. v1 결과가 없으면 처음부터 수집한다.

기존 v1을 다른 출력 경로로 실행했다면, 먼저 `python3 -B fixed_sweep_720_v2/prepare.py --source /home/eu/gemm-research/기존결과폴더 --output /home/eu/gemm-research/scripts/fixed_sweep_720_v2/runs/full_720`으로 준비한 뒤 위 명령을 사용한다.

## 표시하고 계속하는 항목

- 코어가 720 MHz 또는 메모리가 6801 MHz에서 벗어난 관측값, 순간 SW power-cap 표식.
- 해당 120측정 구간 전체를 REVIEW로 표시한다. 정확한 클럭 기준을 느슨하게 하거나 이탈값을 정상으로 바꾸지 않는다. 블록별 관측값과 전체 센서 기록도 남긴다.
- 기존 시간 IQR, 앞뒤 drift, 짧은 블록 REVIEW는 계속 저장한다. 회차 간 변동은 종료 분석에서 표시한다.
- 이상 구간이 반복돼도 클럭/SW power-cap만의 문제라면 다음 구간을 계속 수집한다. 즉시 재시도하지 않으므로 특정 조건에서 재시도 루프에 빠지지 않는다.

정확성 실패, CUDA 실행 오류, 잘못된 타이밍 데이터, 센서 중단/누락/큰 간격, 열 제한, 70 W 한도 변경, GPU나 환경 불일치, 다른 GPU 작업 발견은 계속 중단 사유다. 정상 종료와 중단 모두 기존 클럭 해제 경로를 유지한다.

## 결과와 진행률

- 20초마다 완료/전체/남음과 검토 대상 수를 표시한다. 검토 대상 수는 끝난 구간에서 확정한 값이며, 진행 중 구간과 종료 후 회차 간 검사는 나중에 반영된다.
- 결과와 로그, 종료 ZIP은 `/home/eu/gemm-research/scripts/fixed_sweep_720_v2/runs/` 내부에 저장한다. 새 tmux는 만들지 않는다.
- `metrics.csv`와 `fresh_holdout_LOCKED.csv`는 모든 기록과 최종 PASS/REVIEW/FAIL을 포함한다. REVIEW 구간도 남아 있으므로 이 파일 전체를 그대로 학습에 넣지 않는다.
- `review_queue.csv`와 `review_fresh_holdout_LOCKED.csv`에 재검토 대상이 모인다. `remeasure_tasks.json`에는 시간값 없이 task와 이유를 모으므로 holdout 성능을 열지 않고도 재측정 대상을 확인할 수 있다.
- 학습 자료는 위 검토 목록에 있는 task를 제외해 구성한다. 누락 없이 수집을 끝낸 상태와 모든 측정이 품질 기준을 통과한 상태를 구분한다.
- 원시 DB 행의 PASS는 당시 계산/시간 검사 결과다. 최종 클럭 검토는 `quality.json`, CSV와 summary에 반영한다. 원시 행을 덮어쓰지 않는다.
- 누적 카운터 해석 미해결은 그대로이며 전체 REVIEW/종료 코드 2가 남을 수 있다. 이번 변경은 720 MHz 안정성의 새 증명이 아니다.

전체 구성은 주 측정 108,036, 대조 3,942, 경계 1,350, 연산 기준 54의 총 113,382측정과 793,674블록이다. 옛 10,000조건은 allocate/prealloc/Graph와 세 회차로 재측정한다. fresh holdout 1,000형상과 split을 유지한다. 약 5~6일은 예비 수집에서 외삽한 계획값이다.

로컬 CPU 검사는 `python3 -B fixed_sweep_720_v2/test_sweep.py`, 전체 목록 검사는 `python3 -B fixed_sweep_720_v2/run.py verify --size full`이다. 실제 완료한 200형상 ZIP을 다른 곳에 둔 경우 시험용 경로는 `GEMM_TEST_PILOT_ZIP`로 지정한다. GPU 측정 실행은 사용자가 위 명령을 입력한 뒤 시작된다.
