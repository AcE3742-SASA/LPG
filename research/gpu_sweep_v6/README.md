# GEMM 일반 prealloc·CUDA Graph 측정 v6

같은 형상·정밀도·입력값에서 일반 prealloc과 CUDA Graph를 각각 측정한다. 두 시간은 별도 지표로 저장하며 평균하거나 하나의 학습 정답으로 합치지 않는다. 기존 `gpu_sweep/` v5와 결과는 보존하고, 이 패키지는 `gpu_sweep_v6/`에서 실행한다.

## 지금 실행할 명령

ZIP을 풀어 `gpu_sweep_v6` 폴더 전체를 GPU 컴퓨터의 `/home/eu/gemm-research/scripts/` 아래로 옮긴 뒤 다음 명령 하나를 실행한다.

```bash
bash /home/eu/gemm-research/scripts/gpu_sweep_v6/launch.sh pilot
```

기존 `/home/eu/miniconda3`의 `ai_env`를 사용한다. 설치·재설치나 GPU 전원·클럭 변경은 하지 않는다. 다른 GPU 실험과 동시에 실행하지 않는다.

환경 검사를 통과하면 `Preheating GPU for 180 seconds`가 표시된다. 약 3분 예열 뒤 `[1/192]`부터 `[192/192]`까지 측정한다. 예상 시간은 약 16~20분이며 v6 GPU 실측 시간은 아직 확인하지 않았다. 작업별 소요시간과 진행 상황은 콘솔에 나온다.

결과는 `/home/eu/gemm-research/scripts/gpu_sweep_v6/runs/pilot_v6/`에 저장된다. 종료 시 로그까지 자동으로 압축하고 `RESULT_ZIP=...zip`을 출력한다. 그 ZIP을 가져와 검토한다. 파일럿에서 전체 측정으로 자동 진행하지 않는다.

## 파일럿에서 확인하는 것

파일럿은 24형상 × FP32/FP16 × prealloc/Graph × 2 round로, 192작업·96쌍·1,344개 측정 블록이다. 같은 형상과 dtype의 두 방식을 연달아 측정하고, 다음 round에서는 방식·dtype 순서를 뒤집고 형상 순서를 다시 섞는다. 두 round는 같은 프로세스 안의 반복이다.

초기 예열 180초, 조건별 관측 안정화 3초, 블록 목표 100 ms, 7블록, IQR·drift·round 차이 5% 기준은 v5와 같다. 큰 변동을 없애기 위해 블록을 버리거나 통과할 때까지 재측정하지 않는다.

이번 파일럿은 바뀐 두 방식의 GEMM 수집 경로를 확인한다. 대역폭·pressure·연결 실험·Nsight 수집은 포함하지 않는다. 해당 보조 실험은 기존 코드를 유지해 full에 남겼으며, v6 파일럿 성공이 그 경로를 새로 GPU 검증했다는 뜻은 아니다.

| 항목 | 확인 내용 |
|---|---|
| 정확성 | 64×64 출력은 전체 원소, 큰 출력은 결정적으로 선택한 최대 16×16개 원소를 CPU FP64 결과와 비교한다. GPU dtype으로 반올림된 입력을 기준으로 하며 전체 출력의 유한성도 검사한다. 큰 출력 전체의 정확성 검사는 아니다. |
| 블록 변동 | 상대 IQR 또는 첫·끝 블록 차이가 5%를 넘으면 REVIEW다. 목표보다 짧은 블록도 검토 대상으로 남긴다. |
| round 변동 | 같은 방식·형상·dtype의 round 중앙값 범위/중앙값이 5%를 넘으면 REVIEW다. |
| 짝 비교 | 같은 입력 seed인지 검사하고 `prealloc_ms`, `graph_ms`, `graph_over_prealloc`을 별도 열에 저장한다. 한쪽이 없거나 FAIL이면 미완성 쌍으로 남긴다. |
| Graph 묶음 | GEMM/재생 횟수가 같은 형상·dtype의 round·재시작·pilot→full 사이에 유지되는지 확인한다. |
| GPU 상태 | 1초 간격 telemetry와 시작·100작업마다의 프로세스 목록을 보존한다. |

FP32는 IEEE 경로를 사용하고 FP16의 지원되는 reduced-precision reduction/accumulation 옵션을 끈다. 오차 기준은 FP32 `atol=2e-5, rtol=2e-4`, FP16 `atol=1e-3, rtol=5e-3`이다. 허용오차나 프로토콜을 바꾸면 새 파일럿부터 실행한다.

## 무엇을 재는가

두 방식 모두 입력과 출력 메모리를 측정 전에 확보한다. 같은 seed로 같은 입력값을 생성하지만 각 방식은 별도로 메모리를 할당하므로 두 측정의 버퍼 주소가 같다는 뜻은 아니다.

- `prealloc_ms`: CUDA Event 사이에 일반 `torch.mm(..., out=...)`를 반복한 stream 경과 시간을 GEMM 횟수로 나눈 값이다. GPU가 CPU의 다음 제출을 기다리는 구간도 들어갈 수 있다.
- `graph_ms`: 같은 GEMM을 여러 번 담은 CUDA Graph를 반복 재생한 stream 경과 시간을 실제 GEMM 횟수로 나눈 값이다. Graph 내부 작업 간격과 재생 사이 간격을 포함한다.

Graph는 CPU 제출 영향을 줄이지만 순수 커널 시간과 같지 않다. 두 방식의 차이 전체를 정확한 CPU 비용이나 대기 시간으로 해석하지 않는다. 두 값 모두 입력 준비부터 결과 반환까지의 전체 응용 프로그램 시간도 아니다. 블록의 host wall 시간은 별도로 기록한다.

처음 수집한 5,000형상 × 2정밀도 데이터는 출력 할당을 포함한 일반 호출 경로였다. 그 결과와 이번 prealloc·Graph 결과는 실행 방식이 다르므로 하나의 y값으로 섞지 않는다. 연결 모델의 시간 합도 같은 실행 방식끼리 검증한다.

## Graph 묶음 수

별도 묶음 비교에서 검증한 64³ FP16, 64³ FP32, 128³ FP16은 재생당 GEMM 64회로 고정한다. 네 독립 프로세스의 64→128회 비교에서 가장 큰 감소는 0.345%였고, 사전 지정 1% 비교 기준과 5% 안정성 기준을 통과했다. 이 근거는 세 조건에 한정된다.

그 밖의 형상은 처음 세 번의 일반 호출 probe로 `ceil(2 ms / probe의 GEMM당 시간)`을 계산하고 1~64회로 제한한다. 이 값은 목표 길이를 위한 초기 선택이며 실제 Graph 한 번이 2 ms라는 보증은 아니다. 첫 선택을 원자료에 저장하고 같은 형상·dtype의 다른 round, 대조 실험, 재시작에서도 재사용한다. full의 공통 형상에는 검토한 v6 pilot의 선택을 이어 쓴다. 저장된 선택이 서로 다르면 중단한다.

`graph_nodes`는 이 코드에서 캡처한 GEMM 호출 수다. GEMM 한 번이 여러 CUDA 커널을 실행할 수 있으므로 실제 커널 수와 구분한다. 이 규칙은 개별 GEMM 측정에 적용하며, 연결 실험의 Graph는 기존의 별도 적응 묶음 방식을 유지한다. 모든 형상에서 묶음 수 수렴을 확인한 것은 아니다.

캡처 전 별도 stream에서 예열하고, Graph가 참조하는 입력·출력 tensor를 재생 동안 유지한다. API 동작은 [PyTorch CUDA Graph 문서](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)를 따른다.

## 결과 판정과 재시작

| 종료 상태 | 의미 |
|---|---|
| `EXPERIMENT: PASS`, exit 0 | 모든 작업을 수집했고 현재 판정 기준을 통과했다. |
| `EXPERIMENT: REVIEW`, exit 2 | 누락·FAIL 없이 수집했으나 블록 또는 round 변동 검토가 필요하다. |
| `EXPERIMENT: FAIL`, exit 1 또는 3 등 | 환경·실행·정확성 오류 또는 중단으로 미완료 상태다. 로그를 확인한다. |

잘못된 인자도 exit 2를 만들 수 있다. 따라서 래퍼는 이번 실행에서 새로 생성한 `summary.json`이 `gate=REVIEW`, 누락 0, FAIL 0일 때만 REVIEW라고 표시한다. 이전 실행의 summary나 종료 코드만으로 판정하지 않는다. 정상 종료뿐 아니라 실패 시에도 가능한 로그를 자동 압축한다.

Ctrl+C를 한 번 누르면 현재 작업을 저장한 뒤 중단한다. 같은 명령으로 다시 실행하면 SQLite에 저장된 작업을 건너뛴다. 코드·프로토콜·GPU·드라이버·정밀도·PyTorch 환경이 달라지면 이어 쓰기를 거부한다. 저장된 FAIL은 원인을 확인한 뒤에만 `--retry-failed`로 재측정하며 과거 시도는 남긴다.

데스크톱 프로그램은 기존 세 경로와 `/usr/libexec/xdg-desktop-portal-gnome`의 정확한 경로만 기록 후 허용한다. 다른 Python/CUDA 작업, 비슷한 이름의 다른 경로, 조회 실패는 계속 차단한다. 허용한 데스크톱 프로그램의 GPU 영향이 없음을 보장하지 않는다.

## 파일럿 검토 뒤의 전체 측정

v6 파일럿 결과의 누락·FAIL·IQR·drift·round 차이와 telemetry를 먼저 검토한다. full은 해당 v6 파일럿의 전체 작업 목록, 프로토콜, 코드 hash, GPU 환경과 검토 메모를 확인한다. 과거 v5 파일럿으로 v6 full을 시작할 수 없다.

full은 78,952작업이다. 주요 구성은 다음과 같다.

| 구분 | 작업 수 | 구성 |
|---|---:|---|
| 주 측정 | 72,024 | 6,002형상 × 2정밀도 × 2방식 × 3 round |
| 대조 측정 | 5,256 | 대표·기존 검토 형상의 prealloc/allocate/Graph/pressure |
| 경계 측정 | 900 | M/N/K 경계 family에서 prealloc·Graph 각각 수집 |
| 연산 기준 | 18 | 큰 정방형 GEMM의 prealloc 경로 |
| 대역폭 기준 | 9 | device-to-device copy |
| 연결 실험 | 702 | 연속 제출·층별 동기화·Graph, 단독 합과 연결 시간 |
| 프로파일링 | 43 | 타이밍 종료 뒤 별도 NCU·NSYS 수집 |

주 측정과 경계 측정의 모든 조건을 두 방식으로 수집한다. 보조 실험 전체가 두 방식의 짝으로 구성된 것은 아니다. prealloc·Graph 쌍은 대조 실험까지 포함해 37,776개다. 프로파일러 측정값은 일반 실행의 타이밍에 합치지 않는다.

전체 실행에는 여러 날이 필요하다. 주 측정만 조건별 안정화 3초와 목표 측정 0.7초를 곱하면 약 74시간이며, 실제로는 보정·정확성 검사·대조·연결·프로파일링 시간이 더 든다. 이는 계획 계산이고 실측 예상은 v6 파일럿 뒤 갱신한다.

검토를 마친 뒤 아래 형태로 별도 실행한다. 검토 메모는 실제 판단으로 작성한다.

```bash
bash /home/eu/gemm-research/scripts/gpu_sweep_v6/launch.sh full /home/eu/gemm-research/scripts/gpu_sweep_v6/runs/full_v6 --pilot /home/eu/gemm-research/scripts/gpu_sweep_v6/runs/pilot_v6 --review-note '실제 파일럿 검토 내용'
```

## 저장 파일과 분석 규칙

| 파일 | 내용 |
|---|---|
| `results.sqlite3`, `raw_attempts.jsonl` | 성공·REVIEW·실패를 포함한 모든 시도. SQLite가 재시작의 기준이다. |
| `metrics.csv`, `blocks.csv` | 조건별 통계와 블록 원자료. 블록을 독립 학습 표본으로 세지 않는다. |
| `paired_metrics.csv` | 같은 조건의 두 시간, 비율, Graph 묶음 수, 개별 상태·round 검토 표식. |
| `summary.json`, `review_queue.csv`, `report.html` | 완료 상태, 실패·변동 목록, 파일럿 블록 그래프. |
| `manifest.json`, `shape_splits.csv` | 실행 전 확정한 작업·분할·프로토콜·코드 hash. |
| `fresh_holdout_LOCKED.csv`, `paired_fresh_holdout_LOCKED.csv` | 새 확인용 형상의 개별·짝 결과. |
| `sessions/`, `checked/runs/`, `launcher/` | 환경·telemetry·프로세스·실제 종료 코드·콘솔 로그. |
| `source/`, `inputs/provenance.json` | 실행 코드 사본과 기존 입력의 원본 hash. |
| `profiles/` | full의 별도 프로파일러 보고서·명령·로그. |

기존 5,000형상에 파일럿 추가 형상 2개와 새 확인용 1,000형상을 더한다. 같은 형상의 dtype·방식·round는 같은 split에 둔다. 기존 데이터는 이미 탐색했으므로 `legacy_test_explored`를 미관측 시험 데이터라고 쓰지 않는다.

새 holdout 1,000형상은 모델과 hyperparameter를 확정하기 전 성능값을 열어보지 않는다. 전체 품질 검사는 수행하지만 HTML과 일반 짝 CSV에서는 holdout 시간을 제외한다. raw 파일에도 값은 들어 있으며, LOCKED는 접근 권한 잠금이 아닌 연구 절차상의 구분이다.

원자료는 반올림하지 않는다. 표·본문을 작성할 때 유효숫자를 맞춘다. 측정 안정성 통과와 예측 성능 검증은 별개이며 이 패키지는 학습 결과를 만들어 넣지 않는다.

## 로컬 검증

```bash
python3 -B -m unittest discover -s . -p 'test_*.py'
python3 -B suite.py plan pilot
python3 -B suite.py plan full
```

로컬 검사 33개, Python 구문·Bash 구문·패키지 파일 대조를 검증한다. 자세한 범위는 `VALIDATION.md`에 있다. v6의 실제 CUDA 실행과 안정성은 새 GPU 파일럿에서 확인한다.
