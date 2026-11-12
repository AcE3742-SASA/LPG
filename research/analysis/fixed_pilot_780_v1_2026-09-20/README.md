# 780 MHz 파일럿 실행본 검증 — 2026-09-20

기존 `fixed_pilot_780_v1` 초안을 `fixed_pilot_v1` 원본과 대조하고 실행 ZIP으로 완성했다. 사용자는 코어 780 MHz, 메모리 6801 MHz, 기존 전력 한도 70 W와 Mac ZIP → Ubuntu 실행 방식을 확정했다. 실제 GPU에서는 아직 실행하지 않았다.

## 재사용과 변경

기존 `fixed_pilot_v1`의 24형상 × FP16/FP32 × prealloc/Graph × 두 round, 192측정과 1344블록, 96쌍을 유지한다. 프로토콜의 측정 관련 변경은 코어 900→780 MHz뿐이다. 사용자 선택과 full 미검증을 표시하는 메타데이터를 추가했다. 기존 Timer, 작업 순서, 입력 seed, 저장된 Graph 묶음, 정확성 범위, 180초 예열, 3000 event ms 안정화, 100 ms 목표, 7블록, 5% 기준, 50 ms 조회는 같다.

실행기는 프로토콜의 코어와 메모리 값을 설정 명령에 넘긴다. 대상 UUID, 공유 GPU 잠금, 다른 계산 작업, 기존 70 W 검사와 ai_env 일반 사용자 실행, sudo 승인 갱신, worker 종료 후 두 클럭 해제, 닫힌 로그 ZIP 보존을 재사용한다. 새 Nsight 실행 경로와 full 진입 경로는 추가하지 않았다.

`preserved_hashes.json`의 기존 49파일은 변경하지 않았다. 실행에 필요한 기존 35파일은 배포본의 `dependency_hashes.json`과 일치한다. ZIP에는 새 폴더의 7파일만 있으며 기존 환경을 재설치하지 않는다.

## 검증 결과

배포 ZIP을 임시 폴더에 풀어 CRC, 7파일 원본 바이트, Python AST와 Bash 문법, 의존 해시를 확인했다. 해당 복사본에서 45개의 서로 다른 합성 검사를 통과했다. 최종 실행 집계 55회에는 `test_hold.py`가 함께 발견한 audit 검사 10회의 중복이 포함된다.

최초 병렬 검사에서 기존 steps의 SIGINT 처리 테스트가 자식 종료를 기다리는 5초 제한을 한 번 초과했다. 실패 로그와 `parallel_test_attempt.json`을 보존했고, 코드나 제한 시간을 바꾸지 않고 해당 suite를 단독 재실행했을 때 10개가 모두 통과했다. 타임아웃의 원인을 확정하지 않는다. 새 780 실행본의 중단 및 ZIP 보존 검사는 처음과 배포본 검사 모두 통과했다.

클럭 설정값 780/6801, 전력 변경 명령 부재, 설정/worker/분석/해제 실패와 중단 처리, 누락·중복·소스 변조·정확성·분모·Graph 묶음·round 변동 판정을 확인했다. 시간/관측 클럭 PASS와 누적 카운터 REVIEW를 구분하며 `full_sweep=HOLD`, `validated_for_full=false`를 유지한다.

검사별 로그, 해시와 실행 환경은 `validation.json`에 있다. 합성 검사는 실제 GPU의 클럭 유지나 성능을 입증하지 않는다. Ubuntu 결과 ZIP을 받은 뒤 누락, FAIL/REVIEW, IQR, drift, round 변동과 관측 클럭을 검토한다.

## 전달 파일

- `fixed_pilot_780_2026-09-20_v1.zip`
- SHA-256: `cb9cfc2cd1742d40c3340fbb44925ca8c6161da25d94ff667c9258e5c699db35`
- 같은 이름의 `.zip.sha256` 파일을 함께 제공한다.
- 예상 15~25분은 계획값이며 worker 제한은 40분이다.
