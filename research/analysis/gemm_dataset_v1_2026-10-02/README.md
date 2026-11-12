# 최종 GEMM 데이터셋 v1

2026-10-02 로컬 정리본. 자료 버전은 `gemm_720_adopted_v1_2026-10-02`다. 원자료 113,382측정에서 최종 검토 54건을 보류하고 113,328건의 채택 가능 측정을 식별했다. 품질 기준이나 기존 split을 바꾸지 않았다. 모델 학습은 아직 수행하지 않았다.

## 자료와 집계 단위

원본은 [v4 반환 ZIP](../../full_720_remeasure3_20261001_20261001_163605_88321a.zip)이며 SHA-256은 `319c8480f15fd5f310bd0c211b326104ed341276b76e214bec688b673ecdc5c4`다. 내장 소스, 의존성, 복구 이력과 센서를 보존한 최신 결합본을 사용한다. [반환 검산](../remeasure_720_v4_result_2026-10-01/README.md)과 원본 대응을 유지한다.

측정은 task ID 한 개의 한 회차다. 조건 행은 `(phase, split, group, M,N,K, dtype, mode)`가 같은 세 회차를 합친다. 형상 수는 M/N/K만의 고유 개수다. 7블록이나 내부 반복 수를 독립 형상 수로 세지 않는다. 집합 간 형상 수를 더하면 중복이 포함될 수 있다. 특히 controls는 main과 같은 형상을 다시 측정했고 boundary는 중앙 형상이 여러 group에 나타난다.

개별 품질은 PASS 113,331 / REVIEW 51 / FAIL 0이다. 회차 간 기준 때문에 PASS 3건도 보류하므로 채택 가능 수는 113,328이다. 보류를 제외한 main은 108,000측정이지만, 불완전 회차에 남은 정상 18측정은 진단용으로 보존한다. 세 회차가 모두 정상인 main 조건 행은 35,994개다. 이 수에는 검증, development, 탐색 이력이 있는 시험 및 잠금 시험이 포함된다.

## split과 phase별 수량

| phase | 기존 split | 전체 측정 | 채택 가능 측정 | 보류 | 고유 M/N/K | 세 회차 정상 조건 행 | 완전 FP16/FP32 형상 |
|---|---|---:|---:|---:|---:|---:|---:|
| boundary | boundary_diagnostic | 1,350 | 1,350 | 0 | 73 | 450 | 73/73 |
| compute_reference | calibration | 54 | 54 | 0 | 3 | 18 | 3/3 |
| controls | development | 3,942 | 3,924 | 18 | 219 | 1,308 | 219/217 |
| main | development | 3,942 | 3,930 | 12 | 219 | 1,308 | 219/217 |
| main | fresh_holdout | 18,000 | 18,000 | 0 | 1,000 | 6,000 | 1,000/1,000 |
| main | legacy_test_explored | 12,924 | 12,924 | 0 | 718 | 4,308 | 718/718 |
| main | legacy_train | 60,264 | 60,243 | 21 | 3,348 | 20,079 | 3,348/3,345 |
| main | legacy_validation | 12,906 | 12,903 | 3 | 717 | 4,299 | 717/716 |

legacy_train main의 60,243측정 중 불완전 조건의 정상 6측정을 제외하면 세 회차 정상 조건은 20,079행이다. 아직 C 구역을 떼기 전 학습 후보 풀이다. legacy_validation의 정상 조건은 4,299행이다. development는 설계 및 예비 검토 구역이며 A 학습에 자동 편입하지 않는다. `legacy_test_explored`는 과거 탐색 이력이 있어 최종 미열람 시험으로 사용하지 않는다. `fresh_holdout`은 1,000형상의 18,000측정, 6,000조건을 잠금 시험으로 유지한다.

후속 정적 점검에서 기존 main split 사이에 한 축만 1~8 다른 인접 형상 113쌍을 확인했다. train/validation 26쌍, train/fresh_holdout 1쌍 등이 있다. [인접 대응표](../layer_spec_v1_2026-10-02/original_split_neighbours.csv)에 시간값 없이 남기고 기존 split은 유지한다. 정확히 같은 형상의 미관측과 인접 형상까지 멀리 분리된 일반화를 구분한다. 새 레이어 그룹과 제안한 A/C 내부 역할에는 같은 인접 교차가 없다.

controls, boundary, compute_reference는 대조 및 기준 자료다. main 학습 행과 합치거나 동일 task의 복제 표본처럼 증식시키지 않는다. 보존된 기준 천장은 FP32 6.14 TFLOP/s, FP16 35.02 TFLOP/s, B 262.3 GB/s다. 새 학습에 넣을 P/B 값은 별도 버전으로 고정하며 compute_reference의 우연히 빠른 시간으로 사후 교체하지 않는다.

## 54건의 보류 기준과 적용 범위

보류 집합은 기존 분석기의 `remeasure_tasks.json` 그대로다. 전력 제한 및 SM 이탈을 직접 관측한 51건과 `(4096,2049,64)` FP32의 회차 간 상대 범위 초과로 추가된 정상 3건을 포함한다. 보류 여부는 task와 회차, phase별로 적용한다. REVIEW를 PASS로 바꾸거나 원본에서 삭제하지 않는다.

| M,N,K | main FP32 보류/전체 | 정상 FP32 회차 | controls FP32 보류/전체 |
|---|---:|---|---:|
| 2056,4095,72 | 3/9 | 1,2 | 해당 없음 |
| 2704,3720,72 | 3/9 | 1,2 | 해당 없음 |
| 4096,2049,64 | 9/9 | 없음 | 해당 없음 |
| 4096,4088,72 | 9/9 | 없음 | 해당 없음 |
| 4096,4095,72 | 9/9 | 없음 | 9/9 |
| 4096,4096,72 | 3/9 | 1,2 | 9/9 |

표의 정상 회차는 세 실행 방식에 공통이다. 모든 여섯 형상의 main FP16 9건씩은 채택 가능하며 FP32의 문제를 이유로 제외하지 않는다. controls가 있는 두 형상의 FP16도 정상이다. FP32의 정상 부분 회차 18건을 이용해 두 회차 평균으로 학습 행을 만들거나 한 회차를 복제하지 않는다. FP32 여섯 형상의 18개 main 조건은 완전 회차 모델 표에서 빠진다. 다른 형상이나 모든 K=72, 큰 M/N 전체까지 배제하는 규칙을 만들지 않았다.

이 제외는 현재 720/6801 MHz, 70 W, 예열 후 반복 실행에서 확보한 정상 품질 자료의 적용 범위를 정한다. 문제 형상의 지연시간 예측 정확도를 검증했다는 의미가 없다. 레이어 구성에 해당 FP32 형상이 생기면 scope 경고를 남기고 새 구성 연산과 레이어의 품질을 각각 확인한다. 널리 재현되거나 모델 오차 및 핵심 결론에 영향을 주면 원인 진단을 다시 검토한다. [10월 2일 두 형상 진단](../power_shape_diag_v1_result_2026-10-02/README.md)은 별도 자료이며 54건의 품질 재분류에 쓰지 않았다.

## 원본 대응과 재사용

`measurement_index.csv`의 113,382행은 모든 원래 task ID, split, phase, 회차와 seed를 보존한다. `seq`, 원래 session 및 segment, 품질 파일 경로, 최신 v4 ZIP SHA와 최초 해당 세션을 담은 반환 ZIP SHA가 들어 있다. `results.sqlite3:attempts:<seq>`를 이용해 v4 원자료의 기록으로 돌아갈 수 있다.

| 최종 기록의 최초 수집본 | 최종 보존 기록 수 |
|---|---:|
| full_720 두 세션 | 108,222 |
| v1 재측정 세션 | 3,360 |
| v4 재측정 세션 | 1,800 |
| 합계 | 113,382 |

v3의 1,800건은 보존 이력으로 남지만 최신 결합 자료에서는 v4로 대체됐다. 새로운 정상 기록을 선택할 때 과거 회차의 가장 빠른 값을 고르지 않았다. 현행 v4가 가져온 정상 자료와 새 재측정 자료의 대응을 사용한다.

재사용 키에는 M/N/K 외에 dtype, API, A/B/C shape와 stride, 실제 전치, precision 설정, mode, 할당과 입력 정책, Graph 반복 묶음 및 환경이 포함된다. 기존 GEMM은 contiguous A[M,K], B[K,N]와 `torch.mm` 경로다. Graph 선택값은 식별용 메타데이터이며 무실측 예측에 자동으로 추가하지 않는다. 새 레이어의 실제 중간 입력과 기존 무작위 GEMM 입력이 같다고 가정하지 않는다. 기존 GEMM 예측기 A의 재사용과 단독 실측 S_G 값의 재사용을 따로 판정한다.

## 파생 파일

| 파일 | 내용과 용도 |
|---|---|
| [measurement_index.csv](measurement_index.csv) | 시간값 없는 전체 대응표 |
| [quarantine_54.csv](quarantine_54.csv) | 시간값 없는 보류 54건, 직접 품질 및 회차 플래그 |
| [accepted_unlocked_measurements.csv](accepted_unlocked_measurements.csv) | 잠금 시험을 제외한 채택 가능 측정과 원래 median_ms |
| [accepted_locked_index.csv](accepted_locked_index.csv) | 잠금 시험 식별자만 보존. 시간값 없음 |
| [main_complete_unlocked_conditions.csv](main_complete_unlocked_conditions.csv) | 정상 세 회차 중앙값들의 중앙값. 29,994행, 역할별 필터 필요 |
| [main_complete_locked_index.csv](main_complete_locked_index.csv) | 잠금 시험 6,000조건 식별자만 보존 |
| [incomplete_conditions.csv](incomplete_conditions.csv) | 불완전 조건과 남은 정상 회차 수. 시간값 없음 |
| [split_phase_counts.csv](split_phase_counts.csv), [shape_scope.csv](shape_scope.csv) | 집합별 수와 여섯 형상의 dtype 및 phase별 범위 |
| [summary.json](summary.json), [derived_hashes.json](derived_hashes.json) | 자료 버전, 입력 및 파생 파일 해시, 보존 검사 결과 |
| [build_dataset.py](build_dataset.py), [verification.log](verification.log) | stdlib 생성기와 실행 검사 |

잠금 시험의 시간 및 블록 파일은 열지 않았다. SQLite에서도 task/session/segment/seed/Graph 식별 정보만 SQL로 추출했다. 잠금 시험의 시간은 원본 ZIP과 기존 LOCKED 파일에 그대로 남아 있다. 재계산으로 시험 시간이 공개되거나 새로운 학습 표에 들어가지 않는다.

## 로컬 검증과 다음 연결

전체 ID 누락 및 중복 0, 품질 113,331/51, 보류 54, 완전 main 35,994조건, 잠금 6,000조건과 원본 파일의 전후 해시 일치를 검사했다. 생성 전 manifest와 unlocked metrics 등의 해시도 기존 반환 ZIP 검산 기록과 대조했다. 보존 DB는 `mode=ro&immutable=1`로 읽어 sidecar를 변경하지 않았다.

```bash
python3 -B analysis/gemm_dataset_v1_2026-10-02/build_dataset.py --self-check
python3 -B analysis/gemm_dataset_v1_2026-10-02/build_dataset.py
```

정상 출력은 `PASS: derived dataset, source preservation and locked-test boundary`다. 두 명령은 로컬 파생 자료 작업이며 GPU를 호출하지 않는다. C 구역의 내부 역할은 [레이어 명세 후보](../layer_spec_v1_2026-10-02/README.md)에서 별도 목록으로 준비했다. 원래 split을 덮어쓰지 않고 A 학습 전에 내부 역할을 고정한다. 실제 레이어 수집과 모델 학습은 다음 단계다.
