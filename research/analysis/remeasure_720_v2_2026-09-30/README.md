# Ubuntu 텍스트 모드용 영어 출력 개정

2026-09-30 사용자 요청을 반영해 `remeasure_720_v2`와 `remeasure_720_review_2026-09-30_v2_ascii.zip`을 준비했다.

- 터미널 진행률·안내·오류를 영어 ASCII로 바꿨다. 기존 공통 환경 점검 래퍼의 번역본을 패키지에 포함하고 `LC_ALL=C`를 적용했다.
- `run.py`, `analyze_sweep.py`, `prepare.py`, `protocol.json`, `dependency_hashes.json`은 v1과 바이트 단위로 같다. 측정·분석·준비 로직 및 품질 기준은 유지한다. 한국어 설명 문서와 저장 보고서는 그대로 둔다.
- v1 전체 CPU 재생 검증은 `../remeasure_720_v1_2026-09-30/validation.json`에 있다. 이 개정은 메시지 변경이므로 원자료 전체 재생 검증을 반복하지 않았다.
- `test_delivery.py`는 CPU에서 ASCII 진행률, 셸 구문, ZIP CRC와 추출본 동일성, 의존 100파일 해시, 전체 113,382 task 목록을 검사한다. 실제 GPU 대신 명령 대역으로 준비 선행·실패 시 중단·기존 tmux 사용·REVIEW 종료 코드와 최종 압축 동작을 검사한다. 결과는 `delivery_validation.json`이다.
- v1 ZIP·소스·검증 자료 및 기존 의존 파일은 보존했다. 새 실행 기본 출력은 새 v2 폴더의 `runs/full_720_remeasure_20260930`이다. v1 실행 중 소스를 덮어쓰거나 동일 출력 폴더로 버전을 바꾸지 않는다.
- 새 GPU 측정은 실행하지 않았다.
