# 900/6801 MHz 부하 확인

사용자가 GPU의 코어 900 MHz·메모리 6801 MHz 설정 성공 출력을 확인했다. 이 패키지는 두 설정이 실제 GEMM 부하에서도 관측되는지와 시간 안정성을 확인한다. 기존 `clock_audit_v1`의 worker·타이머·수치 검사·telemetry를 그대로 재사용한다.

`clock_hold_v1` 폴더를 `/home/eu/gemm-research/scripts/`에 둔다. 같은 위치에 기존 `clock_audit_v1`과 `gpu_sweep_v6`가 있어야 한다. 다음 명령 하나를 실행한다.

```bash
bash /home/eu/gemm-research/scripts/clock_hold_v1/launch.sh
```

약 4~5분 예상이다. 시작할 때 sudo 인증을 요청할 수 있다. 실험 Python은 일반 사용자로 실행하고 GPU 클럭 설정·복원에만 sudo를 쓴다. 진단 9소스와 v6 14소스 hash·환경을 검사하고 기존 공유 GPU 잠금을 획득한다. 다른 GPU 계산이 확인되면 시작하지 않는다. 기존 환경·benchmark 파일을 수정하거나 재설치하지 않는다.

실행은 다음 순서다.

1. 현재 GPU 상태와 실행 소스를 보존한다.
2. 대상 UUID에 코어 900 MHz·메모리 6801 MHz 설정을 다시 요청하고 응답을 저장한다. 전력 한도 변경 명령은 없다.
3. 기존 180초 예열 뒤 (1888,832,632) FP16과 (392,2776,80) FP32의 prealloc·Graph를 각각 두 round씩 측정한다. 두 번째 round는 형상·방식 순서를 뒤집는다. Graph 묶음 29/56회·입력 seed·3초 안정화·100 ms 목표·7블록을 유지한다. 총 8측정·56블록이다. Nsight는 켜지 않는다.
4. 기록된 host 범위 내부 표본에서 코어 900 MHz·메모리 6801 MHz·전력 한도 70 W를 확인한다. 표본이 없는 블록·설정과 다른 표본·관측된 전력/thermal 제한 신호는 REVIEW로 남긴다. IQR·첫/끝 변화·round 범위/중앙값의 기존 5% 기준을 유지한다.
5. 드라이버가 설정 단계를 시작한 뒤에는 정상 종료·측정 오류·처리 가능한 중단 모두에서 코어 `-rgc`와 메모리 `-rmc` 복원 명령을 각각 시도한다. 두 명령이 끝날 때까지 공유 GPU 잠금을 유지한다. 복원 명령 수락 결과와 이후 GPU 조회를 보존한다.
6. 로그가 닫힌 뒤 새 결과 ZIP과 SHA-256을 만들고 `RESULT_ZIP=` 경로를 출력한다.

`HOLD CHECK: PASS`는 이 단일 프로세스에서 수집한 표본·정확성·시간 기준을 통과했다는 뜻이다. 명령 수락만으로 연속 시간의 실제 고정을 증명하지 않으며, 100 ms 조회도 센서 갱신 주기를 뜻하지 않는다. 시간 밖 직전·직후 표본은 내부 표본에 섞지 않고, 카운터를 짧은 구간의 제한 시간 비율로 바꾸지 않는다. 정확성 범위는 기존 v6 검사를 그대로 따른다.

`REVIEW`(exit 2)는 수집을 완료했으나 클럭·표본·시간 품질 표식이 남았다는 뜻이다. `FAIL`(exit 1)은 명령 실패·수집/검산 실패·복원 실패다. 원시 결과와 REVIEW는 모두 보존한다. 성공해도 full을 시작하지 않는다. 새 자동 코어 대조군이 없으므로 원인 확정·본측정 클럭 최종 결정은 별도다. 후속 대조 실험은 양쪽 메모리 조건을 같게 해야 한다.

복원은 기본 자동 클럭으로 돌아가는 명령이며 원래 두 수동 잠금을 유지하는 동작이 아니다. sudo 권한 만료나 하드웨어 오류로 복원 명령이 거절되면 `RESTORE: FAILED`와 수동 복원 명령을 출력하고 성공으로 판정하지 않는다. 사전 소스·환경·GPU 잠금·다른 작업 검사에서 막히면 이 실행은 설정에 손대지 않으므로 앞서 사용자가 걸어 둔 수동 잠금이 남을 수 있다. 터미널 강제 종료·SIGKILL·전원 차단에서도 자동 복원을 보장하지 않는다. 이때는 GPU 실험이 끝난 상태에서 다음 두 명령을 차례로 실행한다.

```bash
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
```

주요 결과는 `dense_before/results.json`·`telemetry.csv`, `analysis.json`, `clock_windows.json`, `measurement_summary.json`, 최종 `summary.json`·`status.json`·`restore.json`·`gpu_after_restore.json`이다. `dense_before`는 재사용 worker의 디렉터리 이름이며 이번 실행에서는 고정 클럭 측정만 담는다.

로컬에서는 새 검사 7개와 재사용 검사 10개를 실행한다. 8측정·56블록의 합성 검산, 클럭/메모리 불일치·표본 누락·round 변화·수치 실패 판정, 공유 잠금 안에서 정상/오류/중단 뒤 두 복원 시도, 첫 복원 실패 후 두 번째 시도, 중단한 launcher가 정리 후 ZIP을 만드는 경로를 검사한다. Mac에서 실제 GPU 설정·부하 유지·복원을 실행한 것은 아니다.

설정과 복원 명령: [NVIDIA nvidia-smi 문서](https://docs.nvidia.com/deploy/nvidia-smi/).
