# 720 MHz 전수 결과의 REVIEW 5,160건 재측정

2026-09-30 사용자가 승인한 재측정 실행본이다. 원본 `full_720_20260927_192931_e069ad.zip`에서 PASS 108,222건을 별도 결과 폴더로 가져오고, REVIEW 5,160건만 기존 측정기로 다시 수집한다. 원본 폴더와 ZIP은 바꾸지 않는다.

## 실행

이 폴더를 Ubuntu `/home/eu/gemm-research/scripts/remeasure_720_v1/`에 둔다. 기존 `fixed_sweep_720_v2`, 그 의존 폴더, 200형상 완료 ZIP은 그대로 유지한다. 완료한 전수 ZIP은 다음 위치를 기본으로 사용한다.

```text
/home/eu/gemm-research/scripts/fixed_sweep_720_v2/runs/full_720_20260927_192931_e069ad.zip
```

기존 SSH의 tmux 안에서 실행한다.

```bash
bash /home/eu/gemm-research/scripts/remeasure_720_v1/launch.sh
```

먼저 CPU에서 원본 ZIP의 SHA-256, 원시 기록, Graph 설정과 복사한 근거 파일을 확인한다. 정상 기록의 원래 순서를 유지하며 재측정 목록만 비워 둔 새 DB를 만든다. 준비 완료 표시는 `REMEASUREMENT READY: imported=108222, pending=5160`이다. 이후 sudo 인증, 기존 실행 환경 확인, 정상 자료의 센서 재검산과 180초 예열이 이어진다. 측정이 시작되기 전 CPU 검산에 시간이 걸릴 수 있다.

진행 표시는 `재측정 0/5,160 | 전체 108,222/113,382`에서 시작한다. 이전 측정 속도를 적용하면 GPU 수집은 약 5.5~6시간이고, 준비와 최종 재검산·압축 시간이 추가된다. 파일 전송을 끝낸 뒤 실행한다.

원본 ZIP이 다른 위치에 있으면 `launch.sh full --archive /절대경로/원본.zip`으로 지정할 수 있다. 정상적인 일시중단 후에는 같은 명령을 다시 실행하면 된다. 진행 중인 실행을 겹쳐 시작하면 잠금 검사에서 거부한다.

## 보존하는 조건

`run.py`와 `analyze_sweep.py`는 `fixed_sweep_720_v2`와 바이트 단위로 같다. task ID, 입력 seed, 형상 split, 순서, 정확성 검사, 7블록 측정과 세 회차, 720/6801 MHz 및 70 W 조건을 유지한다. PASS 자료만 가져오면 이전 Graph 설정이 사라지는 14개 조건은 원본에서 확인한 묶음 수를 추가로 고정했다. 전체 12,144개 Graph 선택의 digest가 원본과 일치해야 실행을 준비할 수 있다. Graph 크기를 새 측정값으로 다시 선택하지 않는다.

클럭 이탈과 순간 SW power-cap은 해당 구간을 REVIEW로 남기고 계속한다. 정확성 실패, CUDA 오류, 센서 누락·중단, 열 제한, 전력 한도 변경, 환경 불일치와 다른 GPU 작업은 기존처럼 중단 사유다. 같은 실행에서 REVIEW를 무한 재시도하지 않는다.

기존의 모든 시도와 제외 근거는 새 결과의 `recovery/source_run.zip`에 보존한다. 가져온 정상 기록은 원래 seq와 내용을 유지하고, 원래 session 및 launch의 상태를 성공으로 고쳐 쓰지 않는다. 가져온 원자료와 새 시도는 서로 다른 세션이다. 종료 분석은 이 둘을 함께 읽어 세 회차 변동을 다시 검사한다.

fresh holdout 798건도 원래 split 그대로 재측정한다. 해당 시간값은 LOCKED 파일에 보존하며 학습·모델 선택용으로 공개하지 않는다. 누적 카운터 해석은 계속 미해결이므로 재측정이 끝나도 전체 REVIEW와 종료 코드 2가 남을 수 있다. 완료 수, 남은 재측정 목록, 센서 품질과 클럭 해제 기록으로 결과를 판단한다.

## 결과

결과, 로그와 종료 ZIP은 모두 다음 폴더 안에 생긴다.

```text
/home/eu/gemm-research/scripts/remeasure_720_v1/runs/
```

마지막 `RESULT_ZIP=` 경로의 ZIP을 전달하면 된다. 수집 100% 이후에도 검산과 클럭 해제, 압축이 끝날 때까지 기다린다.

로컬 CPU 검증은 `python3 -B remeasure_720_v1/test_remeasurement.py --archive 원본ZIP --output 검증용새폴더`로 수행한다. 이 검증은 GPU를 실행하지 않는다.
