# CUTLASS 검증 재점검 v2

제공된 `cutlass_build_check.yx7T9I.txt`에서 `CUTLASS_ENABLE_CUBLAS:BOOL=ON`, CUDA/CXX의 `CUTLASS_ENABLE_CUBLAS=1`, `libcublas.so.13`과 `libcublasLt.so.13`의 실제 연결 경로를 확인했다. 따라서 빌드 비활성화나 두 라이브러리의 연결 누락을 뒷받침하는 증거는 없다. 런타임 검증이 건너뛰어진 정확한 원인은 여전히 미확정이다. 현재 자료만으로 재설치·재빌드를 진행하지 않는다. 원본과 점검 결과는 `../cutlass_780_smoke_2026-09-21/raw/cutlass_build_check.yx7T9I.txt`, `../cutlass_780_smoke_2026-09-21/build_check.json`에 보존했다.

기존 v1을 재사용한 `cutlass_780_smoke_v2.sh`는 검증 제공자를 `device`로 바꾼다. [설치본 체크아웃과 동일 커밋의 공식 소스](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/profiler/src/gemm_operation_profiler.cu#L1400-L1553)에서 GPU 기준 구현의 GEMM 결과를 별도 계산한 뒤 CUTLASS 출력과 비교하는 경로를 확인했다. 같은 CUTLASS 프로젝트가 제공하는 참조 구현이며 cuBLAS 검증 PASS나 독립 라이브러리 간 교차 검증으로 표기하지 않는다. 실제 설치 바이너리에서 이 경로가 수행되는지는 새 GPU 결과로 확인해야 한다.

v1과 같은 두 커널, 8192³, 자료형·누산·입력 분포·seed, epsilon=0, 예열 10회 및 1000 ms 이상/최소 10회 조건을 유지한다. 기존 GEMM 실험과 동일한 GPU UUID 기반 `/tmp/gemm-sweep-<GPU UUID>.lock`을 사용하고 다른 계산 프로세스를 확인한다. 그 뒤 sudo 인증 및 SM 780·메모리 6801 MHz 설정을 다시 수행한다. 각 커널 시작 전에 780/6801 MHz·70 W를 확인하며 전력 한도를 새로 변경하지 않는다. 각 실행별 200 ms 모니터와 정확성·클록·모니터 상태를 검사하고 실패 시 다음 커널을 실행하지 않는다. 원자료와 실패 종료코드는 새 결과 폴더 및 tar.gz로 보존한다. 종료 시 클록 고정을 해제하지 않으므로 이후 작업은 해당 상태를 확인해야 한다.

기존 테스트를 재사용해 정상, not_verified, 부하 클록 불일치, 프로파일러 실패, 다른 계산 프로세스, 공유 잠금 사용 중, 실행 전 클록 불일치, 고정 명령 실패, 모니터 오류의 9개 대역 실행을 확인했다. 검사 실패 시 후속 커널 중단, 원로그·압축 생성, 모니터 종료를 확인했고 Bash 문법 검사도 통과했다. 테스트는 native GPU 도구의 대역 실행이며 CUDA 정확성·실제 클록 유지·온도 안정성·최종 천장 검증은 아니다. 재현 명령은 `python3 -B test_smoke.py`다.

Ubuntu의 `/home/eu/cutlass_780_smoke_v2.sh`로 파일을 옮긴 뒤 기존 터미널/tmux에서 `bash /home/eu/cutlass_780_smoke_v2.sh`를 실행한다. 비밀번호 요청은 클록 설정용 sudo 인증이다. 두 경로 PASS·종료코드 0이면 짧은 실행 점검 통과이며, 최종 연산 천장에는 커널·형상·반복 비교가 더 필요하다. 성공/실패와 관계없이 마지막에 표시되는 tar.gz를 전달한다. device 참조 계산 때문에 v1보다 오래 걸릴 수 있고 정량 소요시간은 아직 측정하지 않았다.

스크립트 SHA-256: `0e96a77e8e295a672dafa5467789eb34e9205f28b8962755885e01cf8d99dbd6`.
