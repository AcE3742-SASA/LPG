# 720 MHz 남은 REVIEW 1,800건의 2차 재측정

2026-09-30 사용자의 재측정 요청을 이어받는 실행본이다. 검산한 1차 재측정 ZIP에서 PASS 111,582건을 보존하고 남은 REVIEW 1,800건(일반 1,543건, 잠금 시험 257건)만 다시 수집한다. 1차 재측정에서는 5,160건 중 3,360건이 PASS로 바뀌었다. 남은 1,800건이 재측정해도 통과할 수 없다고 판정한 것은 아니다.

## 실행

이 폴더를 Ubuntu `/home/eu/gemm-research/scripts/remeasure_720_v3/`에 업로드한 뒤 기존 tmux에서 실행한다.

```bash
bash /home/eu/gemm-research/scripts/remeasure_720_v3/launch.sh
```

기본 원본 ZIP은 `/home/eu/gemm-research/scripts/remeasure_720_v1/runs/full_720_remeasure_20260930_20260930_201957_8a23fb.zip`이다. 다른 위치라면 `launch.sh full --archive /absolute/path/source.zip`으로 지정한다. 기존 의존 폴더와 검산된 200형상 ZIP은 유지한다. 출력은 v3 폴더의 `runs/full_720_remeasure2_20260930/`에 저장되며 이전 측정 폴더를 덮어쓰지 않는다.

CPU 준비 후 `REMEASUREMENT READY: imported=111582, pending=1800`이 표시된다. sudo 인증, 기존 ai_env 환경 확인, 가져온 정상 자료의 센서 재검산과 180초 예열을 거쳐 측정한다. 진행률은 `[Remeasure] 0/1,800 | [Total] 111,582/113,382`에서 시작한다. 직전 속도를 적용한 GPU 수집 예상은 약 2시간이며 준비, 재검산과 압축 시간이 추가된다. 모든 터미널 안내는 ASCII 영어다.

## 보존 조건과 재개

측정기 `run.py`, 분석기 `analyze_sweep.py`, 프로토콜과 의존 해시는 v2와 같다. 720/6801 MHz, 70 W, task ID, seed, split, Graph 선택 12,144개와 정확성 및 시간 안정성 기준을 유지한다. 순간 전력 제한이나 클럭 이탈은 해당 구간을 REVIEW로 표시하고 계속 수집하며 정확성, CUDA, 센서, 열 제한, 전력 한도 및 환경 오류는 중단한다. 처음 전달한 ZIP은 새 결과의 `recovery/source_run.zip`에, 그 이전 원본과 시도 이력도 해당 ZIP 내부에 보존된다.

가져온 기록의 seq와 내용을 보존한다. 이번 원본의 마지막 구간이 제외되므로 SQLite가 새 기록에 그 구간의 번호를 재사용할 수 있다. 재개 검사는 실제 가져온 마지막 seq를 경계로 삼고 원본의 마지막 seq도 별도로 기록한다. 측정 로직에는 변경이 없다.

정상 중단 후 같은 명령을 실행하면 저장되지 않은 작업부터 이어 간다. 실행이 끝나도 REVIEW가 남을 수 있으며 최종 요약과 목록을 다시 검토한다. 같은 완료 폴더에서 단순 재실행하면 이미 저장된 REVIEW를 반복 측정하지 않는다.

로컬 검증은 `test_remeasurement.py`에서 실제 ZIP 가져오기, 정확한 남은 목록, 원본과 정상 근거 및 Graph 보존, 재개·잠금·오류 거부와 센서 재계산을 확인한다. 새 GPU 측정 결과는 아직 없다.
