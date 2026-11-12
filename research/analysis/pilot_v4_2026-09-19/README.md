# 2026-09-19 v4 GPU 파일럿 검토

판정: 풀 스윕을 보류한다. 253작업의 기록은 모두 있지만 마지막 Graph 프로파일링이 FAIL이며, 측정 변동도 남아 있다. v5에서 추적 설정과 조건별 안정화 시간을 바꾸고 새 파일럿을 확인한다.

## 원자료 검증

사용자가 제공한 `pilot_v4-20260919T043621Z-1-001.zip`의 SHA-256은 `5e55d40a509e9cee264657dee754a97cff7b6047c195be8c0238bdd7a14ab0c0`이다. 출처는 [source_manifest.json](source_manifest.json), 추출한 84개 파일의 hash는 [raw_sha256.json](raw_sha256.json)에 저장했다. 원본 ZIP과 `raw/pilot_v4/`는 변경하지 않았다. 첨부된 소스는 실행하지 않고 배포본과 대조했다.

SQLite integrity, 빈 WAL, DB와 JSONL 253행, manifest의 작업 ID와 순서, 실행 소스 13개와 v4 배포본 hash, summary digest가 일치했다. 일반 측정 246개의 7블록 통계, 변동 판정, GEMM 처리량, copy payload 대역폭과 연결 잔차를 재계산했다. 저장된 정확성 검사는 246개 모두 통과했다. 큰 GEMM은 표본 검사, 연결은 층별 검사이며 전체 연결의 FP64 오차 상계는 아니다. 로컬에서 CUDA 계산을 다시 한 것은 아니다.

| 항목 | 이전 파일럿 | v4 |
|---|---:|---:|
| 기록 / 예정 | 247 / 253 | 253 / 253 |
| 미실행 | 6 | 0 |
| PASS / REVIEW / FAIL | 243 / 3 / 1 | 244 / 8 / 1 |
| 일반 측정 정확성 검사 | 246 / 246 | 246 / 246 |
| 상대 IQR 중앙값 | 0.208% | 0.157% |
| 상대 IQR 95백분위 | 2.12% | 2.67% |
| round 간 상대 범위 5% 초과 | 15묶음 | 9묶음 |
| 최대 round 상대 범위 | 27.8% | 26.0% |
| 수집 세션 시간 | 10분 45초 | 14분 10초 |

이전 수치는 [기존 분석](../pilot_2026-09-19/README.md)의 검증 결과를 참조했다. v4는 예열 설정이 달라 별도 실행으로 비교하며 원자료를 합치지 않는다. 253은 작업 기록 수다. 마지막 FAIL을 포함하므로 전체 성공을 뜻하지 않는다. 래퍼와 수집 세션 종료 코드는 모두 3이다.

## 마지막 FAIL의 원인

NCU 4작업은 모두 종료 코드 0이고 `.ncu-repz`와 유한한 실제 커널 metric을 포함한다. v4의 NCU 수정은 이번 GPU 실행에서 확인됐다.

NSYS의 일반 제출과 층 사이 동기화는 각 80개 커널을 기록했다. 마지막 CUDA Graph 작업은 종료 코드 0과 비어 있지 않은 `.nsys-rep`를 만들었지만 `cuda_gpu_kern_sum`이 `does not contain CUDA kernel data`로 SKIPPED됐다. 해당 SQLite에는 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 21행이 있으며 이 중 양의 시각에서 실행된 20행과 준비 단계의 음의 시각 1행이 있다. `CUPTI_ACTIVITY_KIND_KERNEL` 테이블은 없다.

이는 그래프 실행이 기록됐지만 요구한 개별 커널 자료를 수집하지 못한 상태다. NSYS 기본 graph 추적은 전체 그래프를 기록하고 개별 노드를 생략한다. [NVIDIA 공식 문서](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-graph-trace)에 따라 v5는 `--cuda-graph-trace=node`를 명시한다. 검사 기준을 graph 기록으로 낮춰 기존 FAIL을 PASS로 바꾸지 않는다. 실제 node 자료 수집은 GPU 재실행으로 확인한다. 재설치나 관리자 권한 변경이 필요한 오류로 보지 않는다.

NSYS 프로파일링은 batch 64, 폭 1024의 GEMM 4층이며 아래 일반 연결 실험의 2층과 다르다. 프로파일링 시간은 일반 event 타이밍에 합치지 않는다.

## 측정 변동

REVIEW 8개는 주 측정 5개와 pressure 대조 3개다. round 간 5% 초과는 주 측정 7묶음과 pressure 대조 2묶음이다. 서로 겹칠 수 있으므로 두 개수를 독립 조건 수처럼 더하지 않는다. 전체는 [review_conditions.csv](review_conditions.csv)와 [between_round.csv](between_round.csv)에 있다.

| 주 측정 REVIEW 조건 | round | 상대 IQR | 첫·마지막 차이 |
|---|---:|---:|---:|
| (2968, 976, 160), FP16 | 0 | 3.18% | +6.72% |
| (392, 2776, 80), FP32 | 0 | 4.61% | −9.72% |
| (768, 1723, 331), FP32 | 1 | 20.3% | −22.5% |
| (549, 3639, 73), FP32 | 1 | 6.22% | −6.54% |
| (2968, 976, 160), FP16 | 1 | 1.29% | +17.4% |

(768, 1723, 331) FP32의 round 중앙값은 163→126 μs이며 상대 범위는 26.0%다. round 1의 블록은 약 161, 164, 140, 126, 125, 125, 125 μs로 변했다. 안정화 마지막 블록과 보정 블록 뒤에도 변화가 이어졌다. 블록 시간이 80 ms 아래로 줄어 `short_block`도 붙었다. 이전 파일럿의 79.7 ms 경계 사례와 달리, 이번에는 측정 중 연산 시간이 크게 줄어든 사례다. 앞 블록을 버려 변동을 숨기지 않는다.

이전의 (3216, 520, 776) FP32는 381→491 μs였지만 v4에서는 479→501 μs, round 상대 범위 4.40%였다. 타이밍 구간 내 표본의 온도는 두 round 모두 71°C, 클럭은 930/900 MHz다. 이 조건은 개선됐지만 다른 조건의 변동이 남아 전체 안정화 성공으로 해석하지 않는다.

예열 종료 무렵 72°C가 관측됐으며 전체 세션은 33~74°C다. `nvidia-smi -q`의 실행 전후 SW Power Capping 누적 시간은 312초 증가했고 SW/HW Thermal Slowdown 누적 시간은 증가하지 않았다. 이 전후 구간에는 예열과 프로파일링도 포함된다. 특정 GEMM 변동의 원인이나 열 스로틀링을 확정하는 근거는 아니다. 1초 telemetry는 짧은 측정 구간을 놓치며, 최악 조건의 round 1 타이밍 구간 내부에는 표본이 없었다. [telemetry_alignment.csv](telemetry_alignment.csv)는 구간 내부 표본만 연결한다. v4 pressure 블록은 시작 시각이 없어 이 표에서 제외했다.

## 보존할 예비 관측

FP16 (1024,64,264)→(1024,65,264)의 경계 차이는 두 round 모두 4.98→9.93 μs로 반복됐다. 작은 batch 64, 폭 256→512→128 연결 실험 12작업도 정확성과 변동 검사를 통과했다. FP16 연속 제출의 조립 잔차는 3.72%, 3.59%, 층 사이 동기화는 두 round 모두 41.5%였다. Graph는 −1.66%, −1.78%였다. 각 값은 블록별 `(연결 시간−개별 시간 합)/연결 시간`의 중앙값이다. [원래 정밀도의 결과표](chain_results.csv)를 보존했다. 작은 연결의 관측이며 예측기 정확도나 전체 신경망 일반화 결과는 아니다.

## v5와 다음 실행

v5는 NSYS node 추적을 명시하고 커널 CSV의 실제 행을 검사한다. 이름, 양의 Instances, 유한하고 양의 Total Time이 필요하며 통계 명령의 종료 코드도 별도 저장한다. 헤더만 있는 CSV나 Graph SKIPPED는 여전히 실패다. 현재 실제 NSYS CSV 2건은 새 검사에 통과하고 Graph CSV는 거부된다. 로컬 회귀 검사 23개와 Python/Bash 구문 검사를 통과했다.

조건별 안정화는 관측 event 시간 750 ms에서 3,000 ms로 늘렸다. 3초는 v4의 전환 중 변동을 점검하기 위한 다음 시험 설정이며, GPU에서 효과가 확인된 값은 아니다. 초기 예열 180초, 7블록, 목표 100 ms, 반복 수 보정 90~110%, 정확성 허용오차와 변동 5% 기준은 유지한다. setup에 저장한 조건별 안정화·보정 블록과 측정 블록을 보존하며 좋은 블록을 선택하거나 통과할 때까지 자동 반복하지 않는다. pressure 블록 시작 시각도 저장한다.

코드와 프로토콜이 바뀌었으므로 기존 폴더에서 `--retry-failed`로 이어갈 수 없다. 기존 `runs/`를 보존하고 `gpu_sweep_2026-09-19_v5.zip`의 코드를 GPU 폴더에 반영한 뒤 다음 명령 하나를 실행한다.

```bash
bash /home/eu/gemm-research/scripts/gpu_sweep/launch.sh pilot /home/eu/gemm-research/scripts/gpu_sweep/runs/pilot_v5
```

환경 검사와 180초 예열 뒤 `[1/253]`부터 시작한다. v4 시간과 조건별 안정화 증가분을 단순 합산하면 약 33분이며, 실제로는 더 걸릴 수 있다. 완료 후 누락 0·FAIL 0, REVIEW와 round 간 변동을 검토한다. 종료 코드 0은 전체 PASS, 2는 summary가 REVIEW인지 확인할 대상, 3은 실패 또는 미완료다. 풀 스윕은 이 검토 뒤 결정한다.

재분석 명령: `python3 -B analysis/pilot_v4_2026-09-19/analyze.py`. 검증 수치는 [verification.json](verification.json), 원자료 hash는 [raw_sha256.json](raw_sha256.json), 전체 블록은 [blocks.csv](blocks.csv)에 있다. v5 GPU 결과는 아직 없다.
