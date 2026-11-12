# v6 배포 검증

일반 prealloc·Graph를 같은 조건에서 별도로 수집하는 gpu_sweep_v6를 준비했다. GPU 파일럿과 full은 아직 실행하지 않았다.

- 로컬 unittest 33개, Python AST·Bash 구문, 실제 192작업 계획의 합성 결과·96쌍·full 진입 검사 통과.
- v5 원본 13개 소스 hash, 타이머·정확성·입력 생성·정밀도 설정과 profiler 원본 유지 확인.
- pilot 192작업, full 78,952작업. 전체 계획은 pilot_plan.json·full_plan.json에 저장했다.
- 배포 ZIP 16파일을 로컬 파일과 byte 단위로 대조했고 CRC 검사를 통과했다.
- 배포본: `gpu_sweep_2026-09-19_v6.zip`
- SHA-256: `210b9eb91820b0c16f12d118b7a27fcd3202de39837c37b1a3446f62a52916b0`

`validation.json`은 로컬 검증, `unittest.log`는 실제 테스트 출력, `package_verification.json`은 배포 파일 hash를 담는다. `verify_release.py`의 합성 데이터는 임시 디렉터리에서 삭제되며 GPU 관측값으로 쓰지 않는다.

GPU 실행 명령과 시간 해석·남은 검증은 [README](../../gpu_sweep_v6/README.md), [검증 기록](../../gpu_sweep_v6/VALIDATION.md)에 있다.
