# CUTLASS 780 MHz smoke 검토 — 재측정 필요

2026-09-21 사용자가 제공한 `cutlass_780_smoke.o9gaoQ.tar.gz` 전체를 확인했다. 두 커널은 실행됐지만 정확성 검증이 수행되지 않았고 목표 SM 클록도 유지되지 않았다. 이번 처리율은 780 MHz 경험적 연산 천장에 사용할 수 없다. FP32 SIMT와 FP16 입력·FP32 누산 Tensor Core 천장은 미확정이다.

| 확인 항목 | FP32 SIMT | FP16 Tensor Core |
|---|---|---|
| Status / 프로파일러 종료코드 | success / 0 | success / 0 |
| Disposition / 검증 provider | not_verified / cuBLAS Not run | not_verified / cuBLAS Not run |
| 실행시간 | 96.13 ms | 16.59 ms |
| 2MNK 처리율, 목표 클록 결과로 사용 불가 | 11.44 TFLOP/s | 66.27 TFLOP/s |
| 각 실행 구간의 GPU 사용률 100% 표본 SM 클록 | 1455–1470 MHz | 1380 MHz |

형상은 각각 8192³, alpha=1, beta=0, batch=1, split-K=1이다. FP32는 SIMT·FP32 입출력, FP16은 TensorOp·FP16 입출력이며 둘 다 FP32 누산이다. 커널 이름과 CSV 원행, 반올림 전 재계산값은 `analysis.json`에 보존했다. CUTLASS FLOPs에는 2MN epilogue 연산이 포함돼 연구 기준 2MNK와 소폭 다르다. CSV의 GB/s는 GEMM 논리적 데이터량/실행시간이며 DRAM 천장으로 쓰지 않는다.

전체 모니터 표본 26개 중 사용률>0은 22개, 100%는 15개다. 부하 표본 22개 모두 SM 780 MHz와 불일치한다. 전환 구간을 포함한 SM 값은 1380·1410·1455·1470·1530·2295 MHz이고, 이 22개 표본의 메모리는 모두 6801 MHz다. 전력 한도는 전체 표본에서 70 W, 최대 관측 전력은 69.5 W, 최대 온도는 50°C다. 사용률과 클록의 샘플링 간격 및 보고 시점 때문에 각 표본을 특정 내부 커널의 순간 상태로 단정하지 않는다. 그 한계를 고려해도 780 MHz 실행으로 인정할 근거는 없다.

프로파일러 시작 전에는 SM 210/메모리 405 MHz, 종료 후에는 SM 2085/메모리 7001 MHz였다. 과거 고정 명령 성공과 BabelStream 실행 중 유지가 이번 CUTLASS 실행까지 이어졌다고 가정할 수 없다. 고정이 언제, 왜 유효하지 않게 됐는지는 이 자료로 알 수 없으며 SSH 탓으로 판단하지 않는다. `SW Power Capping` 누적값도 실행 전후 약 4.06초 증가했지만 이 차이를 개별 커널의 정확한 제한 시간으로 해석하지 않는다.

기존 스크립트는 실행 후 `not_verified`를 감지해 종료코드 1로 중단했고 원자료를 압축했다. 그 assertion이 먼저 실패해 뒤의 클록 검사가 실행되지 않았지만 독립 검토에서 클록 문제도 확인했다. 기존 클록 고정을 가정해 실행 전 재설정/확인을 하지 않은 점은 안내 스크립트의 보완 사항이다. 업로드된 runner는 배포한 v1과 바이트가 같다. 원본 v1과 실패 기록은 수정하지 않는다.

## 검증 경로 진단과 다음 단계

설치본이 보고한 소스 커밋은 `147295a3d4b75f3aeff247c25b8927cea9a7006a`, 소스 작업 상태는 clean이다. 바이너리 SHA-256은 `1e0608eb3eda54d329fd2cc017a28057aca952ae5732f8242aa5073abd7e8a4d`다. 커밋은 체크아웃 식별값이며 바이너리가 반드시 해당 소스로 만들어졌다는 단독 증명은 아니다.

[동일 커밋의 공식 GEMM profiler 소스](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/profiler/src/gemm_operation_profiler.cu#L1190-L1219)에서는 `CUTLASS_ENABLE_CUBLAS`가 검증 코드를 제어한다. 같은 파일 1345–1351행에는 cuBLASLt 알고리즘을 찾지 못하거나 dispatcher 상태가 정상이 아니면 비교 없이 돌아오는 경로도 있다. 따라서 이번 `Not run`만으로 cuBLAS 미설치나 빌드 누락을 확정하지 않는다. [공식 profiler 문서](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#gemm)도 cuBLAS 참조 검증이 빌드 설정에 의존함을 설명한다. 확인한 고정 소스는 `gemm_operation_profiler_147295.cu`로 보존했다.

다음 사용자 단계는 기존 CMakeCache·실제 컴파일 정의·cuBLAS 링크의 읽기 전용 확인이다. 결과에 따라 cuBLAS 경로를 바로잡거나 CUTLASS의 device reference로 검증할 수 있는지 판단한다. 다음 측정 스크립트에는 부하 직전 780/6801 MHz 설정 성공과 70 W 확인, 실행별 정확성 및 부하 클록 판정을 넣고 실패하면 후속 측정을 중단한다. 이 검토에서 GPU 실행·설정 변경·재빌드는 하지 않았다.

앞서 다른 아카이브로 검증한 [BabelStream 보고서](../babelstream_780_bw_2026-09-21/report.md)의 읽기 중심 261.2 GB/s 후보는 유지한다. 이번 CUTLASS 값을 클록 비율로 환산해 빈 연산 천장을 채우지 않는다.

## 재현 및 보존

`raw/`에 업로드 압축 원본과 20개 구성 파일을 그대로 보존했다. 압축 SHA-256은 `714346ac831b780629ce9c9fa8535d54a955ca1b413e558b4dee606585fd86d6`이다. `python3 -B analyze.py`로 원본/추출본 일치, 상태·수치·클록 실패 근거를 다시 확인하고 전체 정밀값과 구성 파일 해시를 `analysis.json`에 생성한다. 이 분석 검사의 PASS는 측정의 PASS가 아니다.
