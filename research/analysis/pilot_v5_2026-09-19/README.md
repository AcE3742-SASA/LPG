# 2026-09-19 v5 GPU 파일럿 검토

## 2026-09-19 측정 범위 검토에 따른 판단 정정

현재 권고는 실제 측정 루프의 Nsight Systems 확인 후 본측정 여부를 결정하는 것이다. 아래의 앞선 수집 권고와 명령은 당시 판단 이력이며 현재의 실행 권고로 사용하지 않는다. 프로파일러 PASS는 본측정 구간의 순수성을 보장하지 않았고, 별도 일반 제출 trace에서 커널 사이 간격 10.8%를 확인했다. 이 비중은 본측정의 오차율이 아니다. 상세 근거와 미검증 범위는 [측정 구간 재검토](nsys_gap_audit.md)에 정리했다. 기존 원자료와 판정 이력은 보존한다.

판정: v5의 본측정 수집을 진행할 수 있다. 실행 실패는 없고 모든 작업이 기록됐다. 남은 변동 표식은 유지하며, 본측정 수집 허용과 측정값 전체의 품질 확정을 구분한다.

## 터미널의 FAIL 표시

`summary.json`은 `gate=REVIEW`, PASS 247·REVIEW 6·FAIL 0·누락 0이다. 세션 상태는 `FINISHED`, 수집 시간은 32분 38초다. 수집기와 두 실행 래퍼의 실제 종료 코드는 2다. 기존 `run_research_checked.sh`는 0이 아닌 종료 코드를 모두 `EXPERIMENT: FAIL`로 표시하므로 검토가 필요한 정상 수집도 FAIL로 보인다. 이번 출력은 수치 검사나 프로파일러의 실행 실패를 뜻하지 않는다.

수집 코드와 배포 v5를 그대로 유지한다. 원본 summary를 PASS로 바꾸거나 종료 코드를 0으로 덮어쓰지 않는다.

## 원자료 확인

사용자 ZIP은 `pilot_v5-20260919T075844Z-1-001.zip`이며 SHA-256은 `6a3ff38d023c76238429d218d63c78ccc497b307de52e443588848ed54c590f5`이다. 출처는 [source_manifest.json](source_manifest.json)에 있다. 추출 파일 87개와 ZIP의 hash, SQLite integrity와 빈 WAL, DB/JSONL 253행, 작업 ID·순서·결과 digest, manifest의 실행 소스 13개와 v5 배포본을 대조했다. 현재 로컬 소스도 GPU 실행 소스와 일치한다.

일반 측정 246개의 저장된 정확성 검사, 7블록 통계·변동 표식·GEMM 처리량·copy payload 대역폭·연결 잔차가 일치했다. 큰 GEMM은 표본 검사, 연결은 층별 검사다. 저장된 검사를 검증한 것이며 로컬 CUDA 재실행이나 전체 연결의 FP64 오차 보증은 아니다. 원자료와 기존 시도는 변경하지 않았다.

| 항목 | v4 | v5 |
|---|---:|---:|
| 기록 / 예정 | 253 / 253 | 253 / 253 |
| FAIL | 1 | 0 |
| REVIEW 작업 | 8 | 6 |
| 주 측정 REVIEW / 전체 | 5 / 96 | 1 / 96 |
| round 간 5% 초과 | 9묶음 | 3묶음 |
| 최대 round 상대 범위 | 26.0% | 8.90% |
| 상대 IQR 중앙값 | 0.157% | 0.0928% |
| 상대 IQR 95백분위 | 2.67% | 1.48% |

프로토콜이 다른 v4와 v5는 개선 양상을 비교할 뿐 하나의 데이터셋으로 합치지 않는다. 3초 안정화 설정에서 변동이 줄었으나, 별도 세션 비교이므로 이 변화 전부를 안정화 시간만의 인과 효과로 단정하지 않는다.

## 프로파일러 검증

NCU 4건은 모두 종료 코드 0과 유한한 실제 커널 metric을 남겼다. NSYS 3건도 profile과 stats 종료 코드가 모두 0이다. 실제 CSV와 SQLite에서 일반 제출·층 사이 동기화 각각 80개, Graph에서 4,800개 커널을 확인했다. 세 NSYS 명령 모두 `--cuda-graph-trace=node`를 포함한다. v4의 Graph 내부 커널 미수집 문제는 v5 GPU 실행에서 해소됐다. Graph의 커널 수에는 캡처된 반복이 포함되며 별도 학습 표본 수가 아니다.

## 남은 REVIEW와 사용 판단

| 작업 | round | 판정 원인 |
|---|---:|---|
| 주 측정 (3216,520,776) FP16 prealloc | 1 | 첫·마지막 블록 −10.3% |
| pressure (512,512,512) FP16 | 0 | 첫·마지막 블록 +5.90% |
| pressure (64,64,64) FP16 | 0 | 첫·마지막 블록 +8.07% |
| pressure (64,64,64) FP16 | 1 | 상대 IQR 5.28% |
| pressure (64,64,64) FP32 | 1 | 첫·마지막 블록 +5.61% |
| pressure (128,128,128) FP16 | 1 | 첫·마지막 블록 −8.22% |

round 간 표식 3묶음은 주 측정 (3216,520,776) FP16의 81.1→74.2 μs(8.90%), pressure (256,256,256) FP16의 6.30%, pressure (64,64,64) FP16의 8.45%다. 작업별 REVIEW와 round 표식은 겹칠 수 있으므로 독립 조건 수처럼 더하지 않는다. [전체 round 비교표](between_round.csv)와 [REVIEW 원자료](review_conditions.csv)를 보존했다.

주 측정의 남은 한 작업은 7블록이 약 82.2, 75.9, 73.9, 74.2, 74.2, 73.8, 73.8 μs였다. 첫 블록을 삭제하지 않으며 확정된 안정 측정값으로 취급하지 않는다. round 0의 타이밍 구간 내부에는 telemetry 표본이 없고 round 1에는 71°C·1185 MHz 표본이 1개 있다. 이 자료만으로 변동의 원인을 확정할 수 없다. 해당 형상은 본측정에서 주 측정 3 round와 prealloc/allocate/Graph/pressure 대조 3 round에 모두 포함돼 재확인이 가능하다.

pressure는 GEMM 앞에 256 MiB 쓰기 연산을 삽입하는 별도 대조다. 짧은 GEMM만 event로 분리해 측정하며, 이 변동을 주 prealloc 측정 전체의 오류로 일반화하지 않는다. pressure 값도 삭제하거나 일반 측정과 섞지 않고 원래 표식을 유지한다. 상세 원인은 미확정이다.

이전 최악 조건 (768,1723,331) FP32는 v5에서 두 round 모두 PASS이며 166→164 μs였다. 대역폭 2작업, 연산 기준 8작업, 작은 연결 12작업도 저장된 정확성과 변동 판정을 통과했다. 작은 파일럿만으로 모든 본측정 조건의 안정성을 보장하지 않는다.

## 본측정 수집 결정과 실행

파일럿의 누락·수치 검사·프로파일러 실행 문제는 해소됐다. 주 측정 96작업 중 95개는 작업 내 기준을 통과했고, 남은 한 형상과 pressure 대조의 변동 범위·원자료를 확인했다. 코드나 5% 기준을 다시 바꾸지 않고 3 round의 본측정을 수집한다. 이는 잔여 변동을 통과로 인정한다는 뜻이 아니다. 본측정에서도 REVIEW와 round 표식을 그대로 남겨 수집 완료 후 재확인하고, 미해결 조건을 안정한 학습 정답으로 확정하지 않는다. holdout 성능은 모델 선택 전에 보지 않는다.

검토 메모와 로컬 호환성 검사 결과는 [full_collection_review.json](full_collection_review.json)에 저장했다. 이 검사는 원자료를 복사한 임시 폴더에서 현재 v5 코드와 저장된 GPU 환경을 대조했다. 실제 GPU 환경은 full 실행 시 다시 검사한다. 원래 원자료의 hash는 검사 뒤에도 같았다.

현재 본측정 계획은 42,490작업이다. 파일럿의 종류·실행 방식별 실제 소요시간 중앙값을 적용하고 연결 실험을 층수와 블록 수에 따라 환산하면 약 70시간, 대략 3일이다. 큰 형상, 프로파일러, 중단·재시작에 따라 더 걸릴 수 있다. 실측 전체 소요시간이 아닌 계획용 추정이며 [산출 내역](runtime_estimate.json)을 보존했다.

GPU 컴퓨터에서 아래 명령 하나를 실행한다. v5 코드를 그대로 사용하며 결과는 새 `runs/full_v5`에 저장한다. `nohup`으로 터미널을 닫아도 실행이 이어지게 하고 바깥 로그는 append한다. 컴퓨터가 절전되거나 꺼지면 측정은 진행되지 않는다.

```bash
nohup bash /home/eu/gemm-research/scripts/gpu_sweep/launch.sh full /home/eu/gemm-research/scripts/gpu_sweep/runs/full_v5 --pilot /home/eu/gemm-research/scripts/gpu_sweep/runs/pilot_v5 --review-note 'v5 검토: 253작업 완료, FAIL 0, 정확성 246건과 프로파일러 7건 통과. 주 측정 FP16 (3216,520,776)과 pressure 변동 표식 유지, 3 round 수집 후 품질 재검토.' >> /home/eu/gemm-research/scripts/gpu_sweep/full_v5_launcher.log 2>&1 &
```

정상 시작 시 터미널에 작업번호와 PID가 표시된다. 위 로그에 환경 검사, `Preheating GPU for 180 seconds`, `[1/42490]` 순서가 나타나야 한다. 본측정 실행은 아직 사용자에게서 확인받지 않았다. 코드나 manifest hash를 수정하지 않는다.

재분석은 `python3 -B analysis/pilot_v5_2026-09-19/analyze.py`로 수행한다. [verification.json](verification.json), [raw_sha256.json](raw_sha256.json), [blocks.csv](blocks.csv), [telemetry_alignment.csv](telemetry_alignment.csv)에 상세 검증과 자료를 보존했다. pressure telemetry 구간에는 pressure 연산 시간도 포함되며 GEMM-only event 구간과 구분한다.
