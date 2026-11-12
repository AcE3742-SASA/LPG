# 추가 GEMM 10,000형상 반환 자료

2026-11-06 사용자가 전달한 `expansion_10000_20261105_181927_9a759e.zip`의 보존·검산과 네 학습기 비교 폴더다. 원본 SHA-256은 `ec461b0f1d1eebc23ff34e362757385b84f09b79ecea1411674c6a21f85559d0`이다. 사용자 Transfer 원본과 해시가 같은 사본을 이 폴더에 보존했다.

검산과 학습을 완료했다. [결과 보고서](추가GEMM_10000형상_분석결과_2026-11-06.md)에 576구성·432 fit, 103,104개 기존 예측·144개 새 가중치의 재현, 네 학습곡선과 조건별 선택을 정리했다. 현재 선택은 FP16 MLP32, FP32 RF이며 여섯 조건 모두 11,498형상 전체 보강 모델이다. 수렴 경고와 무효 예측은 0개다.

## 검산

`verify_expansion.py extract`는 정답을 파싱하기 전에 고정 패키지의 task·split·source와 대조하고 안전한 경로 및 CRC를 검사해 별도 `raw/`에 푼다. 새 task는 expansion_train/expansion_validation만 허용한다. 배포 source 223개와 의존성 91개가 일치한다. 원래 source와 데이터 및 보류 이력은 그대로 둔다.

`recompute`는 원래 로컬 분석기를 `recomputed/`에서 실행한다. 원시 SQLite와 source는 읽기용 사본으로 연결하고 다시 쓰는 quality.json·CSV·JSONL은 독립된 경로에 만든다. `verify`는 180,000개 고유 task·저장된 정확성·SQLite 무결성·원본 해시를 확인하고 1,500개 구간 품질과 전체 exports의 재현을 검사한다. 파이썬 내부 정수형 회차 key는 저장된 JSON의 문자열 key로 비교한다. 이 형식 비교를 수정한 과정은 처음 recompute.log에 남아 있으며, 저장 집계 전체와 원본의 일치 및 verify 통과를 확인했다.

최종 기록은 `verification.json`이다. 측정 누락과 task 중복은 0개, 개별 품질은 PASS 174,000/REVIEW 6,000이고 회차 간 보류를 포함한 총 보류는 6,041개다. 정상 사용 가능 원측정은 173,959개다. 실험의 REVIEW 판정을 유지한다. 원본 archive와 해제 자료를 덮어쓰는 명령은 없다.

## 학습 비교

`prelabel_comparison_protocol.json`은 새 시간값 접근 전에 채택 규칙과 비교 격자·모델 설정·선택 방법을 고정한 기록이다. `compare_expansion.py selfcheck`는 합성 자료로 공통 형상 집계·보류 제외·중복 거부·잠금 구역의 정답 접근 전 거부를 검사한다.

`prepare`는 검산 완료를 요구한다. 세 회차와 두 dtype·세 mode가 모두 정상인 공통 형상만 같은 비교에 사용한다. 새 train 8,153형상과 새 validation 895형상을 채택하고, 기존 train 3,345형상·validation 716형상을 유지한다. 부분적으로 정상인 자료의 상태는 `fit_v1/shape_eligibility.csv`에 남긴다. 네 학습기가 같은 정답을 공유한다.

`run`은 기존 3,345형상 endpoint 모델 144개를 재사용하고 새 형상 1,000/3,000/8,153개를 더해 432회 fit한다. 네 학습기 RF·Ridge·MLP32/64, 두 목표 log(T)/log(T/L), 두 dtype·세 mode·세 seed를 동일한 설정으로 비교한다. 총 576구성을 기존·새 검증 구역에서 평가한다. `fit_v1/frozen_protocol.json`에 ID·순서·코드·입력·기존 가중치 해시가 있다. 원래 output이 있으면 실행을 덮어쓰지 않는다.

`summarize_comparison.py`는 완료 결과의 원 예측·지표·가중치를 독립 검산하고, 구역별/합친 검증 표·자료량 곡선·그룹 단위 paired bootstrap과 결과 보고서를 만든다. 주 추천은 11,498형상의 전체 보강 endpoint에서 합친 1,611검증 그룹의 MAPE를 최소화하는 구성이다. RQ4는 source A와 보정 C를 분리한 기존 protocol을 따른다. 최종 시험은 잠금 상태로 유지한다.

실행 환경은 기존 `analysis/integrated_rq14_2026-10-28/.venv`이며 CPU thread 설정도 이전 비교와 같다. `prepare_fit.log`, `fit_console.log`, `fit_v1/progress.json`에 실제 진행을 저장한다.
