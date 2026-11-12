# 두 조건의 클럭 진단 v1 준비

사용자의 “오케이” 승인에 따라 별도 clock_audit_v1를 준비했다. v6 benchmark와 원자료는 유지한다.

40작업·280블록, Nsight 8개 준비 범위·56개 측정 범위를 수집한다. 1초→0.1초→Nsight+0.1초→0.1초→1초의 다섯 프로세스 비교이며 180초 예열·3초 안정화·7블록·100 ms 목표·5% 기준을 유지한다. Graph는 29/56회, 입력 seed는 v6 파일럿과 같다.

로컬 검사 10+22+33개, AST·Bash, 원본 v6 14개·v5 13개 hash, 재사용 11함수의 AST 동일성을 확인했다. 배포 9파일의 ZIP CRC·byte 대조를 통과했다.

배포본: `clock_audit_2026-09-20_v1.zip`

SHA-256: `72c4f47ee579e90cae200f3214d712ea8e09422635aba40b2ef5de273ed721ae`

실제 GPU query 지원·0.1초 관측 영향·Nsight 경로·안정성은 아직 확인하지 않았다. GPU 실행 및 full은 미완료다. 상세 검증은 validation.json·test_*.log·package_verification.json, 실행 안내는 [README](../../clock_audit_v1/README.md)에 있다.
