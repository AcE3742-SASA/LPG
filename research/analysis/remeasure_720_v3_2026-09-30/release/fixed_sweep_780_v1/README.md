# 780 MHz 세 방식 GEMM 수집

코어 780 MHz·메모리 6801 MHz·기존 전력 한도 70 W에서 일반 `torch.mm(A,B)`(allocate), `torch.mm(A,B,out=C)`(prealloc), CUDA Graph를 각각 수집한다. 전력 한도는 변경하지 않는다. 기존 v6의 Timer·입력 seed·정확성 검사·180초 예열·3000 event ms 안정화·100 ms 목표 7블록·5% 기준을 그대로 가져온다. 기존 파일은 수정하지 않는다.

## 지금 할 일: 약 10–15분 점검

ZIP의 `fixed_sweep_780_v1` 폴더를 Ubuntu의 `/home/eu/gemm-research/scripts/` 아래에 놓는다. 기존 gpu_sweep_v6, clock_audit_v1, clock_hold_v1, clock_steps_v1, fixed_pilot_780_v1 폴더가 같은 scripts 안에 있어야 한다. 실행 전 42개 의존 파일의 hash를 확인한다. 패키지·환경을 다시 설치하지 않는다.

```bash
bash /home/eu/gemm-research/scripts/fixed_sweep_780_v1/launch.sh check
```

tmux가 있으면 `gemm780_check` 세션을 열고 그 안에서 sudo 비밀번호를 물어본다. tmux가 없으면 GPU 설정 전에 멈춘다. 원격 환경의 tmux 설치 여부는 아직 확인하지 않았다. `Control+B`를 누르고 손을 뗀 뒤 `D`를 누르면 실행을 유지한 채 분리된다. Command+D가 아니다. 이후 SSH를 종료할 수 있다.

다시 볼 때는 `tmux attach -t gemm780_check`로 연결한다. 완료 뒤 결과 ZIP 경로가 보이고 Enter를 누르면 tmux 창이 닫힌다. tmux도 머신 재부팅이나 세션 강제 종료를 견디는 서비스는 아니다.

대표 10형상/dtype × 세 방식 × 두 round = **60측정·420블록·20묶음**이다. 첫 worker가 30개를 저장하고 종료한 뒤 두 번째 worker가 SQLite에서 이어서 나머지 30개를 측정한다. 각 worker가 180초 예열하므로 예상 10–15분이다. 서로 다른 프로세스지만 같은 머신의 인접 실행이며 독립된 날짜의 재현성 검증은 아니다. 소요시간은 이전 측정에 기초한 계획값이다.

첫 30개와 다음 30개 사이에 저장 결과·Graph 선택을 이어 쓰는지, 새 실행기의 세 방식 결과가 완전한지 확인한다. 두 round의 차이와 지난 780 파일럿 대비 차이는 각각 summary.json 및 reference_780_comparison.csv에 남긴다. 과거 대비 변화율은 기술적 비교이며 자동 보정에 사용하지 않는다.

점검 결과는 `fixed_sweep_780_v1/runs/check_780/`에 저장되고, 닫힌 로그를 포함한 `check_780_날짜_식별자.zip`이 runs 안에 생긴다. **그 ZIP을 검토한 뒤 본수집을 시작한다. 점검에서 본수집으로 자동 전환하지 않는다.**

## 확인할 상태

- `summary.json`: 누락·FAIL·세 방식 묶음·IQR/drift/round 차이·시간/관측 클럭 판정.
- `progress.json`: 현재 작업, 기록 수, worker 상태. worker FINISHED만으로 컨트롤러의 클럭 해제까지 끝났다는 뜻은 아니다.
- `launches/*/status.json`, `restore.json`: 컨트롤러별 완료 상태와 두 해제 명령의 실제 결과.
- `launcher/*/exit_code.txt`, `checked/runs/*/exit_code.txt`: 실행 래퍼 종료 코드.
- `reference_780_comparison.csv`: 점검의 공통 20조건/방식과 이전 780 파일럿의 시간 비교.

누적 SW Power Cap 카운터 해석 문제를 `UNRESOLVED`로 유지하므로, 모두 수집되고 시간·관측 클럭이 통과해도 전체는 **REVIEW, exit 2**다. `timing_and_sampled_clocks_pass=true`, 누락/FAIL 0, 20묶음 완성, 두 worker, 두 클럭 해제 성공을 함께 확인해야 한다. REVIEW 코드만 보고 성공/실패를 판단하지 않는다.

순간 전력 제한·클럭 불일치·센서 누락이 생기면 현재 로그 구간을 검토한 뒤 후속 수집을 멈춘다. 정확성/실행 오류는 해당 측정을 저장한 즉시 멈춘다. 시간 변동은 기존 v6처럼 REVIEW로 남기고 수집을 계속하며, check에 남은 시간 변동 문제는 full 진입을 막는다. 누적 카운터의 기존 모호함만으로는 후속 측정을 멈추지 않으며, 원자료와 이상 블록을 남긴다.

## 본수집 범위

기존 v6 full의 GEMM 주 측정·대조·경계·연산 기준을 유지하면서 각 조건에 세 방식을 채운다. phase와 boundary group은 보존한다. 같은 shape가 주 측정과 대조 또는 여러 경계 family에 등장하는 것은 원래 계획의 별도 맥락이며, 작업 ID는 서로 다르다. 새로운 고유 형상 수로 중복 계산하지 않는다.

| 구분 | 측정 수 |
|---|---:|
| 주 측정: 6,002형상 × 2 dtype × 3방식 × 3 round | 108,036 |
| 대조 GEMM: 219형상 × 2 dtype × 3방식 × 3 round | 3,942 |
| 경계 GEMM: 기존 75 family 항목 × 2 dtype × 3방식 × 3 round | 1,350 |
| 연산 기준 GEMM: 3형상 × 2 dtype × 3방식 × 3 round | 54 |
| 합계 | **113,382** |

총 793,674개 측정 블록, 37,794개 세 방식 묶음이다. **옛 5,000형상 × FP16/FP32의 10,000조건은 주 측정에 세 방식·세 round의 90,000측정으로 포함**된다. 이를 113,382개에 다시 더하지 않는다. 기존 1,000개 fresh holdout 형상과 shape split을 유지한다. 연결 실험·프로파일링·pressure 개입·복사 대역폭은 이 GEMM 실행본의 작업 목록에 들어 있지 않으며 기존 v6 코드는 그대로 보존돼 있다.

세 방식 순서는 round마다 allocate→prealloc→Graph, prealloc→Graph→allocate, Graph→allocate→prealloc으로 바뀐다. 입력값은 동일 seed를 쓰며, 검증한 48조건의 Graph 묶음은 그대로 고정한다. 나머지는 기존 초기 probe로 선택하고 저장된 성공/검토 기록에서 재사용한다. 완료되지 않은 최초 Graph 시도에 저장된 선택이 없다면 다음 시도에서 다시 선택될 수 있다.

안정화+7블록의 목표 시간만 계산하면 약 **116.5시간(4.86일)**이다. 지난 파일럿의 예열 제외 속도를 적용한 참고치는 약 **124시간(5.2일)**이며, 큰/새 형상·정확성 검사·로그·재시작에 따라 더 걸릴 수 있다. 실제 점검 결과를 받은 뒤 예상 시간을 갱신한다. 출력 디스크의 여유 공간은 시작/재시작 전에 20 GiB 이상이어야 한다. 여러 결과 ZIP을 보관할 공간은 별도로 필요하다.

## 본수집 명령: 점검 ZIP 검토 후 사용

```bash
bash /home/eu/gemm-research/scripts/fixed_sweep_780_v1/launch.sh full \
  --check /home/eu/gemm-research/scripts/fixed_sweep_780_v1/runs/check_780 \
  --review-note '실제로 검토한 점검 결과와 카운터·실행 간 차이의 판단을 여기에 기록'
```

이 문구는 형식 예시다. 점검을 검토하기 전에 그대로 실행하지 않는다. full은 같은 코드/프로토콜/환경의 완전한 check, 두 worker의 기록, 통과한 시간·관측 클럭, 두 해제 명령과 검토 메모를 확인한다. 점검의 미해결 카운터와 검토 메모도 보존한다. full 결과는 `runs/full_780/`, tmux 세션은 `gemm780_full`이다.

## 중단과 이어하기

Ctrl+C나 처리 가능한 SIGTERM/SIGHUP을 받으면 worker에 종료를 요청하고 현재 측정을 저장할 시간을 준다. 필요하면 이 실행이 만든 프로세스 그룹만 종료한다. worker 종료 확인 뒤 공유 GPU 잠금과 sudo 갱신을 유지한 채 코어·메모리 고정을 해제한다. GPU가 응답하지 않거나 SIGKILL/전원 차단을 받으면 종료 후 처리까지 보장할 수 없으므로 상태 파일을 확인한다.

같은 명령과 같은 output 경로로 재실행하면 저장된 측정은 건너뛴다. 이전 시도와 실패 기록은 지우지 않는다. 코드·프로토콜·입력·환경이 바뀌면 이어 쓰기를 거부한다. 저장된 FAIL은 원인 검토 후에만 같은 명령에 `--retry-failed`를 붙여 다시 시도한다. 클럭·센서 문제로 멈춘 경우도 결과를 먼저 검토한다. 재시작은 예열을 다시 수행하며, 새 세션은 기존 데이터와 구분해 기록한다.

점검을 처음부터 별도로 반복하려면 `--output /home/eu/gemm-research/scripts/fixed_sweep_780_v1/runs/check_780_repeat`처럼 새 경로를 쓴다. 기존 폴더를 삭제하거나 덮어쓰지 않는다.

진행 상황만 읽을 때:

```bash
python3 -B /home/eu/gemm-research/scripts/fixed_sweep_780_v1/run.py status \
  --output /home/eu/gemm-research/scripts/fixed_sweep_780_v1/runs/check_780
```

## 데이터와 해석

SQLite에 측정별 commit을 하고, 재시도도 새 행으로 남긴다. raw_attempts.jsonl은 모든 시도를 순서대로 반출한 파일이다. metrics.csv·blocks.csv·triplets.csv에는 방식별 시간과 상태를 따로 저장한다. fresh holdout 시간은 별도 LOCKED CSV와 원자료에 남긴다. LOCKED는 접근 권한이 아니라 모델 선택 전 값을 보지 않는 연구 절차다.

50 ms telemetry는 120측정마다 파일을 나눠 저장한다. 분석은 파일 하나씩 읽고 이진 탐색으로 블록 시간 경계를 찾는다. 파일 분할 사이에는 예열이나 측정 기준을 바꾸지 않는다. 센서 조회 간격은 연속 관측이나 유효 커널 주파수를 보장하지 않는다. 약 100 ms 묶음의 평균이므로 개별 호출 스파이크 분포도 아니다. Graph 차이를 순수 CPU 대기 시간으로, allocate 차이를 순수 할당 비용으로 해석하지 않는다.

로컬 점검:

```bash
python3 -B -m unittest discover -s fixed_sweep_780_v1 -p 'test_sweep.py' -v
python3 -B fixed_sweep_780_v1/run.py plan --size full
```

Mac 로컬 검사는 GPU 측정 성공의 증거가 아니다. 실제 새 실행기는 위 check 결과로 확인한다.
