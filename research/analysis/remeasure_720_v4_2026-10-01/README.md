# 720 MHz 분리 재측정 v4 준비

2026-10-01 사용자가 새 실행본 제작을 승인했다. v3 결과에서 전력 제한이 직접 관측된 6개 FP32 형상의 남은 51개 task를 다른 1,749개 task와 별도 수집 구간에 배치한다. 수집 대상 합계는 기존 REVIEW 1,800건 그대로이며 정상 111,582건은 보존한다.

## 실행본과 변경 범위

실행본은 [remeasure_720_v4](../../remeasure_720_v4/README.md), 배포는 [v4 ZIP](../../remeasure_720_review_2026-10-01_v4_isolated_ascii.zip)이다. 입력 ZIP SHA-256은 `e8d2423393a69903eb21552b177689acef38eff02f846662fe7c906fae39ff05`, 배포 ZIP SHA-256은 `d3da18b4644d79fb21c3218600726b6edc18d312ddda4b0464029f96ca239162`다.

일반 그룹부터 120건 × 14구간과 69건 구간을 측정하고, 문제 형상 51건을 마지막 구간에 측정한다. 총 16구간이다. 각 그룹 안에서는 기존 manifest 순서를 유지하며 재개 시에도 그룹을 섞지 않는다. 전체 task 목록과 ID, 입력 seed·split·Graph 선택은 바꾸지 않았다. 순서·구간 경계 변경을 `protocol.json`의 `remeasurement_schedule`과 복구 기록의 `collection_schedule.json`에 명시하고 해시를 검증한다. 실제 구간마다 그룹 및 예정 task ID도 저장한다.

720/6801 MHz·70 W, 예열·안정화·시간 측정·정확성·치명적 오류 중단 조건을 유지한다. 측정 의존성 100파일과 분석기는 기존본과 같다. 한 구간의 클럭/전력 제한 때문에 그 구간 전체가 REVIEW가 되는 정책도 유지하므로 분리가 PASS를 보장하지 않는다. 이번 변경은 기존 자료의 재분류가 아닌 새 수집 순서와 구간의 변경이다.

원본 ZIP은 새 결과의 `recovery/source_run.zip`에 보존되며 이전 시도와 원본도 이 ZIP 안에 이어서 보존된다. 정상 자료의 seq·내용·근거 파일을 유지한다. 새 기록과 가져온 기록의 경계는 실제 마지막 imported seq로 구분한다.

## 실행 경로

ZIP을 기존 연구 PC의 `/home/eu/gemm-research/scripts/`에 푼 뒤 기존 tmux 터미널에서 실행한다.

```bash
bash /home/eu/gemm-research/scripts/remeasure_720_v4/launch.sh
```

기본 입력은 `/home/eu/gemm-research/scripts/remeasure_720_v3/runs/full_720_remeasure2_20260930_20260930_232816_56d6fb.zip`이다. 파일을 옮겼다면 `--archive /actual/path/to/result.zip`을 붙인다. 원본 누락이나 해시 불일치는 GPU 설정 전에 거부한다.

기본 출력은 `/home/eu/gemm-research/scripts/remeasure_720_v4/runs/full_720_remeasure3_20261001/`이다. 로그·압축 결과도 연구 폴더 내부에 저장한다. 마지막 `RESULT_ZIP=`에 새 ZIP 경로를 표시한다. 이전 세션의 실제 속도에 따른 측정 예상은 약 2시간이며 준비·검산·압축은 별도다. 이번 작업에서는 GPU를 실행하지 않았다.

## 검증 근거

실제 자료 검산과 배포 검산을 모두 통과했다. [validation.json](validation.json), [delivery_validation.json](delivery_validation.json), [test.log](test.log)에 근거를 보존했다. 검증은 CPU 작업이며 새 GPU 측정은 하지 않았다.

기존 안전 검사 8개와 분리·부분 재개 worker 검사 1개가 통과했다. 문제 그룹에 모의 클럭 이탈을 넣어도 다른 그룹은 PASS를 유지했다. 실제 v3 원본에서 가져온 111,582건, Graph 선택 12,144개, task ID와 정확한 남은 1,800건을 대조했다. 재개 경계 1,801곳에서 순서와 그룹 분리가 유지됐다. 반복 준비·잠금·잘못된 원본/출력·변조 근거 거부와 SQLite seq 재사용 경계를 검증했다.

정상 931구간의 센서 로그를 기존 분석기로 재계산한 결과는 111,582 PASS, 미수집 1,800건, 회차 간 추가 플래그 0건이다. 준비 단계의 gate는 아직 1,800건을 수집하지 않아 INCOMPLETE다. 원본 ZIP과 정상 근거는 그대로다.

배포 10파일의 ZIP CRC·추출 바이트·실제 검증에 사용한 source snapshot 일치, 기존 의존 100파일, ASCII 콘솔 출력·셸 문법, 준비 실패 시 GPU 단계 차단, REVIEW 종료 코드 2와 결과 압축 및 기존 tmux 사용을 확인했다.
