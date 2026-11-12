# CUTLASS 연산 천장 측정 준비 확인

2026-09-21, 입력 원본: `cutlass_inventory.c50Nc9.tar.gz`.

- 목록의 GEMM 연산은 239개이며 239개 모두 `Status=success`, `Disposition=not_run`이다. 목록 생성 성공을 나타내며 커널 실행·정확성 검증·성능 측정 성공은 아니다.
- FP32 입력·출력·누산의 SIMT 커널 8개와 FP16 입력·출력, FP32 누산의 Tensor Core 커널 8개를 확인했다. FP16 입력·FP32 누산·FP32 출력 커널도 별도로 8개 있다. 혼합 입력형, BF16, FP8, TF32 및 정수 커널은 이번 후보에서 제외했다.
- 재빌드 없이 첫 실행을 점검할 수 있다. 대표 커널 선정은 동작 확인 목적이며 가장 빠른 커널을 선정한 결과가 아니다.

| 점검 모드 | 대표 커널 | A / B / C·D 배치 |
|---|---|---|
| FP32 SIMT | `cutlass_simt_sgemm_256x128_8x5_tn_align1` | row / column / column |
| FP16 Tensor Core, FP32 누산 | `cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8` | row / column / column |

두 커널 모두 M=N=K=8192, alpha=1, beta=0, batch=1, split-K slices=1로 점검한다. 입력은 seed 20260921과 CUTLASS uniform 분포(-2부터 2, 기본 scale)로 고정한다. workspace 1개, 예열 10회, profiling-duration 1000 ms, min-iterations 10을 명시한다. CUDA Graph 옵션은 설치본 도움말에서 확인되지 않아 지정하지 않았다.

cuBLAS 검증을 켜고 epsilon=0, nonzero-floor=0을 사용한다. 프로세스 종료코드뿐 아니라 CSV의 `Disposition=passed`, `Status=success`, 문제 크기·자료형·누산·연산 모드를 확인한다. TFLOP/s는 Runtime(ms)와 `2MNK`로 다시 계산하며, CSV의 Flops가 epilogue 연산을 포함할 가능성을 구분한다. 이 입력 분포에서의 도구 내장 검증이며 일반적인 모든 실수 입력의 정확성 평가를 대신하지 않는다.

`cutlass_780_smoke.sh`는 원시 로그, CSV, 실행 명령, 시각, 200 ms 주기 클록 기록, 실행 전후 GPU 상태, Git 커밋·소스 상태·실행파일 해시를 새 폴더에 보존한다. 다른 계산 프로세스가 있으면 시작하지 않고, 오류가 나도 자료를 압축해 남긴다. 새로운 tmux 세션 생성이나 클록·전력 설정 변경은 하지 않는다. 부하 표본의 780/6801 MHz·70 W 일치를 확인하며 커널별 정확한 시간 범위와 표본 사이 변동은 반환 자료에서 별도로 검토한다.

`--version` 출력에 `@CUTLASS_VERSION@`, `@CUTLASS_REVISION@`가 남아 있어 정확한 버전·리비전을 이 문자열에서 확정할 수 없다. 빌드 시각은 2026-09-16 20:20:55로 표시돼 있다. 다음 실행에 실제 Git 커밋과 바이너리 해시를 기록한다. 도움말 장치 정보의 MHz 값도 부하 중 클록 기록을 대신하지 않는다.

원본 4파일과 압축 파일은 `raw/`, 분류한 목록은 `all_operations.csv`, `fp32_simt.csv`, `fp16_tensor.csv`, 파일 해시와 선택 결과는 `inventory.json`에 있다. 원본 압축 SHA-256: `aece49011c9587b24a15da42eb4e84da7d87b9b274b197972868c2053943ebb9`.

로컬에서 대역 명령만 사용하는 `python3 -B test_smoke.py`로 정상 경로, 미검증 결과 거부, 클록 불일치, 프로파일러 실패, 다른 계산 프로세스 존재 시 시작 중단, 모든 경로의 압축·모니터 종료를 확인했다. 실제 GPU 실행은 아직 하지 않았다. 두 커널을 한 번씩 실행하는 이 단계는 최종 연산 천장 측정이 아니다. 이후 커널·행렬 크기 탐색과 반복 측정이 남아 있다.

[공식 CUTLASS Profiler 문서](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md)
