# 780 MHz 고정 클럭 파일럿

기존 v6 파일럿의 192측정을 코어 780 MHz·메모리 6801 MHz·조회 간격 50 ms에서 다시 수행한다. 전력 한도는 기존 70 W인지 검사하며 변경하지 않는다. 사용자가 코어 780 MHz를 선택했다. 이 파일럿에서 더 넓은 조건의 유지를 확인하며 full은 결과 검토 후 결정한다.

## 설치와 실행

`fixed_pilot_780_2026-09-20_v1.zip`을 GPU 컴퓨터의 `/home/eu/gemm-research/scripts/`에 풀면 `fixed_pilot_780_v1/` 폴더가 생긴다. ZIP에는 새 폴더의 7파일만 들어 있다. 앞서 사용한 `gpu_sweep_v6`, `clock_audit_v1`, `clock_hold_v1`, `clock_steps_v1`이 같은 scripts 폴더에 있어야 한다. 이 네 폴더의 35파일 해시를 먼저 검사하며 기존 설치나 패키지를 수정하지 않는다. 의존 파일이 없거나 해시가 다르면 GPU 설정 전에 중단하므로 `source_check.log`를 확인한다.

```bash
bash /home/eu/gemm-research/scripts/fixed_pilot_780_v1/launch.sh
```

시작 시 sudo 암호를 요청할 수 있다. 클럭 설정·해제만 sudo를 사용한다. 기존 checked runner가 `ai_env`를 활성화하므로 위 명령 전체에 sudo를 붙이지 않는다. GPU에서 다른 실험과 동시에 실행하지 않는다.

처음에 `DEPENDENCY CHECK: PASS; 192 measurements / 1344 blocks / 96 pairs`가 나오고, 이후 `FIXED PILOT: 180 s preheat + 192 measurements`와 `worker.log` 경로가 표시된다. worker.log에는 `[현재/192]` 진행 상황이 저장된다. 예상 15~25분이며 이전 파일럿의 측정별 소요 시간 합계 약 751초와 예열 180초를 기준으로 잡았다. 780 MHz의 192측정 전체 실행 시간은 아직 검증하지 않은 계획용 추정이다. worker 제한 시간은 40분이다.

끝나면 `RESTORE: COMMANDS ACCEPTED`, `FIXED PILOT: PASS/REVIEW`, `RESULT_ZIP=...`를 확인한다. 결과는 `/home/eu/gemm-research/scripts/fixed_pilot_780_v1/runs/fixed_pilot_780_<실행 시각>_<고유값>.zip`과 같은 위치의 `.zip.sha256`에 남는다. 생성된 ZIP을 전달하면 된다. 초기 검사나 수집이 실패해도 가능한 로그를 ZIP으로 묶는다. exit 2는 수집을 마치고 검토할 사항이 남았다는 뜻이다. `RESTORE: FAILED`는 해제 실패이므로 출력된 수동 해제 명령과 restore.json을 확인한다.

## 같은 조건으로 유지한 항목

24형상 × FP16/FP32 × prealloc/Graph × 두 round = 192측정, 1344블록, 96쌍이다. 기존 `make_plan('pilot')`의 작업 ID와 순서를 그대로 사용한다. 입력 seed, 180초 예열, 3초 event 안정화, 100 ms 목표 블록, 블록 7개와 IQR·drift·round 차이의 5% 기준을 유지한다.

48개 형상·정밀도 조합의 Graph 묶음 수는 실제 v6 파일럿에서 저장된 값으로 고정한다. 기존 결과 ZIP과 raw_attempts.jsonl의 해시는 protocol.json에 남겼다. 첫 probe로 묶음 수를 새로 선택하는 대신 이전에 선택된 값을 재사용한다. 타이머·입력 생성·GEMM·정확성 함수는 원본 v6을 그대로 호출하고, 기존 clock_audit worker의 작업 목록·Graph 묶음·조회 간격만 프로세스 안에서 임시 연결한다.

작은 조건의 전체 출력 오차 검사 16개와 나머지 조건의 256원소 표본 검사 176개를 그대로 구분한다. 새로운 Nsight 측정, 자동 코어 부하 대조군, 모델 학습은 포함하지 않는다. 같은 프로세스의 두 round이며 독립 세션 검증은 아니다.

## 클럭과 결과 판정

대상 GPU UUID·다른 계산 프로세스·기존 70 W를 검사하고 공유 GPU 잠금을 획득한 상태에서 780/6801 MHz를 설정한다. 기존 허용된 데스크톱 프로세스는 기록에 남긴다. 기존 sudo 승인은 실행 중 45초마다 비대화식으로 갱신한다. 정상 완료·오류·처리 가능한 중단에서 worker가 종료된 뒤 같은 GPU 잠금 안에서 `-rgc`와 `-rmc`를 각각 시도한다. 두 명령은 기본 설정으로 되돌리며 실행 전 수동 잠금을 복원하는 것은 아니다. 진입 검사 실패 시 기존 클럭 설정은 건드리지 않는다. 전원 차단·SIGKILL에서는 해제 코드를 실행할 수 없다.

각 블록의 호스트 경계 내부 표본으로 코어·메모리·전력 한도와 제한 사유를 확인한다. 내부 표본이 없으면 REVIEW로 남기고 이웃 표본으로 채우지 않는다. 예열 등을 포함한 worker 전체 표본의 클럭 불일치·제한 활성도 별도로 기록한다. 50 ms 요청은 센서 갱신 주기나 연속 관측의 보증이 아니다.

순간 제한 상태가 비활성인데 누적 카운터가 증가하는 문제는 별도 REVIEW로 보존한다. `timing_and_sampled_clocks_pass`는 시간과 관측 클럭 기준만 구분해 보여 주는 항목이다. 이것이 true여도 전체 `gate`는 REVIEW일 수 있다. 카운터를 짧은 GEMM 제한 비율로 환산하지 않으며 기존 이상을 자동으로 면제하지 않는다. PASS여도 full은 HOLD다.

## 결과 파일

- `pilot/results.json`: 완료된 측정마다 누적 저장한 원래 타이머·정확성 결과.
- `pilot/telemetry.csv`, `telemetry_metadata.json`, `gpu_after.json`: 원시 상태와 카운터.
- `analysis.json`, `clock_windows.json`: 시간·round·클럭·카운터 검토와 표본 연결.
- `measurements.csv`, `paired_metrics.csv`, `review_queue.csv`: 192측정·96쌍·검토 대상 표. round 문제는 별도 열에 남는다.
- `manifest.json`, `source/`, `dependencies/`: 조건과 실행 소스 사본·해시.
- `restore.json`, `gpu_after_restore.json`, `status.json`, `summary.json`: 해제 명령 수락과 이후 관측값, 수집 및 최종 상태.
- `worker.log`, `checked/`, `launcher.log`: 환경·진행·오류 기록. 종료한 로그까지 포함해 ZIP을 만든다.

중간에 실패한 실행은 원본 그대로 보존하고 새 실행 폴더를 만든다. 같은 결과 디렉터리에 재개하거나 기존 v6 full 승인 자료로 넘기는 경로는 제공하지 않는다.

로컬 검사는 `python3 -B test_fixed.py`로 실행한다. 합성 데이터·가짜 GPU 명령을 사용하므로 실제 GPU 성능이나 클럭 유지를 검증하지는 않는다.

780 MHz 선택은 사용자 결정이며, 세 후보 중 통계적으로 가장 우수한 값으로 판정했다는 뜻은 아니다. 앞선 10형상/dtype 진단의 관측 유지·작은 블록 변동을 바탕으로 더 넓은 48형상/dtype 조합을 확인한다. summary의 selected_clock_mhz는 780이고 validated_for_full은 false다. 누적 카운터 해석 문제와 약 100 ms 평균이 단발 GEMM 스파이크를 가릴 수 있는 한계는 그대로 남는다.
