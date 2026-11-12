# GEMM 파일럿과 풀 스윕

먼저 파일럿만 실행한다. 끝나면 결과를 가져와 측정 양상을 확인하고, 문제가 있으면 코드를 고쳐 새 파일럿을 실행한다. 풀 스윕은 별도 명령이며 자동으로 이어지지 않는다. 기존 설치를 바꾸거나 패키지를 설치하지 않는다.

## GPU 컴퓨터에서 첫 실행

압축파일을 풀어 `gpu_sweep` 폴더 전체를 GPU 컴퓨터로 옮긴다. 그 폴더에서 터미널을 열고 다음 명령 하나를 실행한다. 기존 `/home/eu/miniconda3`의 `ai_env`를 사용하며 일반 사용자로 실행한다. v2부터 `launch.sh`가 동봉한 원본 `run_research_checked.sh`를 먼저 실행한다.

```bash
bash launch.sh pilot runs/pilot_v5
```

환경 검사 뒤 `Preheating GPU for 180 seconds`가 표시되고 약 3분 동안 예열한다. 이후 `[1/...] bandwidth ...`와 각 측정의 `PASS` 또는 `REVIEW`가 표시된다. 결과는 이 폴더의 `runs/pilot_v5/`에 저장된다. 측정 조건은 24형상 × FP32/FP16 × 2회 순서 교대이며, 여기에 대조 측정·기준값 측정·작은 연결 실험·프로파일링을 더한다. 블록 반복은 별도 학습 샘플로 세지 않는다.

그 전에 기존 래퍼가 `ai_env`, Python 3.12, CUDA_HOME, GPU·compute capability, PyTorch 등 7개 패키지 import, `nvcc` 13.4, `ncu`·`nsys`·`nvidia-smi` 실행을 검사한다. 환경이 맞지 않으면 측정을 시작하지 않는다. 새 `RUN_DIR`는 `runs/pilot_v5/checked/runs/날짜_식별자/`이며 기존 `environment.txt`, `output.log`, `exit_code.txt`가 그대로 생긴다. 환경 검사에서 먼저 실패하면 원본 래퍼의 `exit_code.txt`는 없을 수 있지만 바깥쪽 `launcher/*.exit_code.txt`에 종료 코드가 남는다. 전체 패키지 검사는 버전 고정 검사가 아니라 import 성공 여부와 버전 기록이다.

v4의 실제 파일럿은 253번째 작업까지 14분 10초가 걸렸으며 마지막 Graph 프로파일링이 FAIL이었다. v5는 조건별 안정화를 750 ms에서 3,000 ms로 늘려 약 33분 이상을 예상한다. 연결 실험의 각 층과 연결 전체도 블록마다 안정화하므로 시간이 늘어난다. v5 전체 소요시간은 GPU에서 아직 확인하지 않았다. 출력 중 `REVIEW`가 나와도 남은 조건은 측정한다. 수치 오류나 실행 오류는 저장한 뒤 멈춘다.

완료 후 `runs/pilot_v5/report.html`을 브라우저로 연다. `PASS`라도 그래프와 telemetry를 확인한다. 이 단계에서 풀 스윕을 시작하지 말고 아래 명령으로 묶은 파일을 가져오면 된다.

```bash
python suite.py pack --output runs/pilot_v5
```

`runs/pilot_v5_날짜_시간.zip`과 `.sha256`이 생긴다. Python은 `conda activate ai_env` 환경을 사용한다. GPU 없이도 보고서·상태·압축 명령은 Python 표준 라이브러리만으로 실행할 수 있다.

## 파일럿 결과에서 볼 것

| 항목 | 판정과 확인 내용 |
|---|---|
| 수치 정확성 | 64×64 출력은 전체, 큰 출력은 결정적으로 선택한 최대 16×16개 원소를 CPU FP64 결과와 비교. 입력은 실제 GPU dtype으로 반올림된 값이다. 전체 출력의 유한성도 별도 확인한다. 큰 출력 전체의 정확성을 증명하는 검사는 아니다. |
| 블록 안정성 | 7블록 상대 IQR >5%, 첫·마지막 블록 차이 >5%, 목표보다 짧은 블록은 REVIEW. raw 블록을 삭제하거나 좋은 값만 남기지 않는다. |
| round 재현성 | 같은 조건의 round 중앙값 범위/중앙값 >5%면 REVIEW. 두 round는 같은 프로세스 안의 반복이며 독립 재부팅·세션 실험이 아니다. |
| 순서 효과 | 형상 순서를 다시 섞고 FP32/FP16 및 대조 조건 순서를 뒤집는다. 기존 순서 고정의 영향을 점검한다. |
| 짧은 GEMM | `prealloc`, `allocate`, `graph`의 차이와 event 시간·host wall 시간을 확인한다. 차이 전체를 순수 launch 비용이라고 부르지 않는다. |
| 캐시 가설 | `pressure`는 매 측정 전에 256 MiB 버퍼에 쓰기 연산을 한다. 압박 연산은 GEMM event 구간 밖에 있다. 완전히 차가운 캐시를 보장하지 않는다. |
| GPU 상태 | `sessions/*/telemetry.csv`에 1초 간격 온도, 클럭, 전력, 사용률을 기록한다. 전원 설정이나 다른 프로세스를 변경하지 않는다. |
| 프로파일러 | Nsight Compute의 `all`/`none` cache control과 Nsight Systems의 연결 실행 trace를 분리 저장한다. 프로파일링 시간은 일반 타이밍 데이터에 합치지 않는다. |

FP32는 IEEE 경로, FP16은 지원되는 reduced-precision reduction/accumulation 옵션을 끈다. TF32와 FP32를 섞지 않는다. GPU 이름·compute capability·실행 환경을 검사한다. 기본 오차 기준은 FP32 `atol=2e-5, rtol=2e-4`, FP16 `atol=1e-3, rtol=5e-3`이며 실패 원소의 값과 기준값을 보존한다. 허용오차를 바꾸면 새 파일럿부터 실행한다.

종료 코드는 0=전체 PASS, 2=전체 수집 완료·변동 검토 필요, 3=미완료/실험 실패, 1=사전 검사나 실행 오류다. 성공 출력만 보고 데이터 품질이 확보됐다고 판단하지 않는다.

원본 래퍼는 0이 아닌 종료 코드를 모두 `EXPERIMENT: FAIL`로 표시한다. 2로 끝났다면 `summary.json`의 `gate=REVIEW`와 누락·실패 수를 확인한다. 잘못된 명령 인자도 2를 반환할 수 있으므로 종료 코드만으로 REVIEW라고 판단하지 않는다. `python pilot.py` 직접 실행은 셸 래퍼 검사를 거치지 않으므로 현장 실행에는 `bash launch.sh ...`를 사용한다.

## v5: Graph 커널 추적과 조건별 안정화 재검증

v4 원자료는 253/253작업, PASS 244·REVIEW 8·FAIL 1이었다. 일반 측정 246개는 저장된 정확성 검사를 통과했다. NCU 4건과 일반/층별 동기화 NSYS 2건은 커널 데이터를 포함했다. 마지막 Graph NSYS도 종료 코드 0으로 그래프 실행 20회를 기록했지만, 개별 커널 자료가 없어 `cuda_gpu_kern_sum`이 SKIPPED됐다. 기존 FAIL은 보존한다.

NSYS 명령에 `--cuda-graph-trace=node`를 명시했다. 커널 CSV는 이름, 양의 Instances, 유한하고 양의 Total Time을 확인한다. 통계 명령의 실제 종료 코드를 `stats_exit_code.txt`에 따로 저장한다. 그래프 전체 추적과 노드 추적은 NVIDIA의 [CUDA Graph Trace 안내](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-graph-trace)를 따른다. 프로파일링 시간은 일반 측정값에 합치지 않는다.

초기 예열 180초 뒤에도 REVIEW 8개와 round 간 5% 초과 9묶음(최대 26.0%)이 남았다. 조건별 안정화를 관측 event 시간 750→3,000 ms로 늘려 효과를 시험한다. 3초는 다음 파일럿의 시험 설정이며 안정화가 검증된 값은 아니다. 7블록, 100 ms 목표, 90~110% 반복 수 보정과 5% 판정 기준, 수치 허용오차는 유지한다. 통과할 때까지 재측정하거나 앞 블록을 삭제하지 않는다. pressure 블록에는 telemetry 대조용 시작 시각을 추가했다.

프로토콜은 3이며 코드가 바뀌었다. `runs/pilot_v4`를 보존하고 새 결과 폴더를 쓴다. 기존 ai_env 설치는 그대로 사용한다.

```bash
bash /home/eu/gemm-research/scripts/gpu_sweep/launch.sh pilot /home/eu/gemm-research/scripts/gpu_sweep/runs/pilot_v5
```

파일럿의 누락·실패와 변동을 검토하기 전에는 풀 스윕을 시작하지 않는다. v5 GPU 검증은 미완료다.

## v4: 프로파일러 완료 판정 수정과 예열 보강 (이력)

09-19 업로드된 파일럿에서 Nsight Compute는 종료 코드 0, 실제 커널 CSV 및 `capture.ncu-repz`를 만들었다. 기존 완료 검사가 `.ncu-rep`와 `Metric Name` 열만 기대해 FAIL로 잘못 판정했다. v4는 두 보고서 확장자와 raw CSV의 metric별 열 형식을 지원한다. 헤더만 있거나 유한한 metric 값이 없으면 통과시키지 않는다. 실제 실패 시 콘솔에도 오류와 프로파일러 로그 위치를 표시한다.

일반 측정 246작업의 저장된 정확성 검사는 모두 통과했지만, round 간 변동 15조건과 온도 상승·클럭 하락이 관측됐다. 시작 예열을 10초에서 180초로 늘리고, 블록 반복 수 보정의 허용 구간을 목표 100 ms의 90~110%로 좁혀 80 ms 검토 기준과 여유를 뒀다. 개별 조건 settle 750 ms와 7블록, 정확성 허용오차, 변동 5% 기준은 유지한다. 늘린 예열의 효과는 새 파일럿으로 확인해야 한다.

코드·프로토콜이 바뀌었으므로 기존 결과 폴더를 이어 쓰지 않는다. 기존 `runs/`를 보존하고 v4 코드로 새 파일럿을 실행한다. 기존 manifest나 저장된 FAIL을 수동 수정하지 않는다.

```bash
bash launch.sh pilot runs/pilot_v4
```

새 파일럿의 누락·실패가 없고 변동 검토가 끝나기 전에는 풀 스윕으로 넘어가지 않는다. 기존 파일럿의 분석상 프로파일러 성공을 전체 파일럿 완료로 간주하지 않는다.

## v3: 데스크톱 프로세스 검사 수정 이력

v2의 첫 GPU 실행은 환경 검사를 통과한 뒤 원격 데스크톱·파일 관리자·터미널을 경쟁 작업으로 분류해 측정 전에 중단됐다. v3는 `/usr/libexec/gnome-remote-desktop-daemon`, `/usr/bin/nautilus`, `/usr/bin/ptyxis` 세 경로만 기록 후 허용한다. 다른 Python/CUDA 작업이나 알려지지 않은 실행 경로는 계속 차단한다. 시작 시점과 100작업마다 같은 판정을 적용한다. 메모리 점유량으로 작업 종류를 구분하지 않는다.

허용한 프로세스도 원본 `gpu_processes*.json`과 콘솔 로그에 남는다. 원격 화면 전송이나 창 조작의 측정 영향이 없다고 가정하지 않으며 telemetry와 파일럿 변동을 함께 확인한다. 코드가 바뀌었으므로 기존 `runs/pilot`을 재사용하지 않고 새 `runs/pilot_v3`에서 시작한다. 기존 로그는 그대로 둔다.

```bash
bash launch.sh pilot runs/pilot_v3
```

## 중단·재시작과 진행 확인

Ctrl+C를 한 번 누르면 진행 중인 조건을 저장한 뒤 멈춘다. 프로파일링 중에는 종료 처리에 시간이 더 걸릴 수 있다. 전원이 꺼져도 SQLite에 commit된 조건은 남는다. 같은 명령·같은 폴더로 다시 실행하면 저장된 조건을 건너뛴다. 기존 실패는 원인을 확인한 뒤 `--retry-failed`로 다시 측정하며, 실패 이력은 그대로 남는다.

```bash
bash launch.sh pilot runs/pilot --retry-failed
```

코드·프로토콜·GPU·드라이버·정밀도·PyTorch 환경이 바뀌면 같은 결과 폴더에 이어 쓰지 못한다. 코드를 수정했다면 `runs/pilot_v2`처럼 새 이름을 사용한다. 다른 세션의 데이터를 삭제하거나 덮어쓰지 않는다.

실행 중 다른 터미널에서 아래 명령으로 진행을 확인할 수 있다. 측정에 신호를 보내거나 데이터를 수정하지 않는다.

```bash
python suite.py status --output runs/pilot
```

## 파일럿 검토 후 풀 스윕

파일럿의 수치 정확성·누락·프로파일링 문제가 해소돼야 한다. `REVIEW`가 남으면 원인과 사용 판단을 먼저 기록한다. 코드나 안정화 설정을 수정해야 하면 파일럿을 다시 실행한다. 다음 명령의 검토 메모는 사람이 실제로 확인한 내용으로 바꾼다.

```bash
bash launch.sh full runs/full --pilot runs/pilot --review-note '파일럿 그래프와 검토 목록을 확인한 내용'
```

터미널을 닫고 계속 돌려야 하면 같은 명령 앞에 `nohup`을 붙이고 뒤에 `> full_launcher.log 2>&1 &`를 붙인다. 컴퓨터가 절전되거나 전원이 꺼지면 측정은 진행되지 않는다. OS 설정은 이 코드가 바꾸지 않는다.

풀 스윕은 현재 FP32/FP16 GEMM 예측·경계 분석·조립 잔차 연구의 후속 분석에 쓸 원자료를 수집한다.

| 데이터 | 구성 |
|---|---|
| 주 데이터 | 기존 5,000형상, 파일럿 전용 추가 형상, 새 확인용 1,000형상. FP32/FP16, 3 round, preallocated `torch.mm`. |
| 변동·경로 대조 | 기존 재확인 206형상과 대표 형상. prealloc/allocate/Graph/cache pressure, 두 dtype, 3 round. 기존 raw CSV는 변경하지 않는다. |
| 경계 대조 | M/N/K 각 축의 128/256/512/1024/2048 경계에서 -8/-1/0/+1/+8. 같은 family를 이어 측정하고 다음 round에서 순서를 뒤집는다. |
| 연산 기준값 | 2048³/4096³/8192³의 동일 PyTorch 실행 경로를 3회 측정. 관측 최고값은 backend 기준값 후보이며 하드웨어 절대 천장이 아니다. |
| 대역폭 기준값 | 256/512/1024 MiB GPU 버퍼의 device-to-device copy, 읽기+쓰기 payload 기준 GB/s, 3 round. 실제 DRAM 트래픽 카운터와 구분한다. |
| 조립 실험 | batch 32/64/128/512, width 256/1024/2048, 깊이 2/4/8의 36개 연결과 혼합 폭 3개 연결. 각 FP32/FP16, 연속 제출/층 사이 동기화/Graph, 3 round. |
| 조립 원자료 | 블록마다 개별 GEMM과 전체 연결을 순서 교대 측정. `T_iso_sum`, `T_chain`, 부호 있는 `E_composition`을 저장한다. 예측 합은 모델 학습 뒤 추가한다. |
| 프로파일링 | 대표 경계·짧은/큰 연산·기존 불안정 조건의 ncu 보고서, 연속/동기화/Graph 연결의 nsys 보고서. 타이밍 측정 종료 뒤 별도 프로세스에서 수집한다. |

INT8/INT4, CNN/Transformer 전체 모델, fusion은 이번 데이터의 적용 범위에 포함하지 않는다. 이들은 실제 kernel 경로와 정확성 검증을 새로 정의해야 한다. 세 round만으로 다른 날짜의 독립 재현성을 주장하지 않는다. 독립 세션이 필요하면 같은 확정 코드로 다른 출력 폴더에서 반복하고 `session`을 보존해 분석한다.

## 데이터 사용 규칙과 파일

`manifest.json`에 실행 전에 모든 조건·순서·정밀도·분할·코드 hash를 고정한다. 같은 형상의 dtype과 반복은 같은 split에 둔다. 파일럿 및 재확인용 형상은 development로 예약하고, 남은 기존 형상을 약 70/15/15로 나눈다. 기존 데이터는 이미 탐색했으므로 `legacy_test_explored`도 완전히 보지 않은 시험이라고 쓰면 안 된다. 경계 family는 진단용이다.

새 1,000형상은 `fresh_holdout_LOCKED.csv`로 따로 내보낸다. 모델과 hyperparameter를 확정하기 전에는 이 파일의 성능값을 보지 않는다. 품질 점검은 전체에 적용하지만 HTML은 새 holdout의 성능 그래프를 표시하지 않는다. 이것은 접근 권한으로 강제한 잠금이 아니라 연구 절차상의 구분이다.

| 파일 | 내용 |
|---|---|
| `results.sqlite3`, `raw_attempts.jsonl` | 성공·변동·실패를 포함한 모든 시도. SQLite가 재시작의 정본이다. |
| `metrics.csv`, `blocks.csv` | 조건별 통계 및 블록별 원자료. 블록은 독립 학습 표본이 아니다. |
| `summary.json`, `review_queue.csv`, `report.html` | 누락, 실패, IQR·drift·round 차이와 양상. |
| `shape_splits.csv`, `manifest.json` | 형상 분할과 사전 확정 조건. |
| `sessions/`, `launcher/` | 환경, telemetry, 실제 종료 코드, 콘솔 로그. shell history와 별개로 보존한다. |
| `checked/runs/` | 원본 `run_research_checked.sh`가 만든 실행별 RUN_DIR와 환경·출력·종료 코드. |
| `profiles/` | 프로파일러 명령·원본 보고서·CSV·실패 로그. |
| `source/`, `inputs/provenance.json` | 실행 코드 사본 및 기존 데이터의 원본 hash. |

모든 raw 값은 반올림하지 않는다. 논문 표·본문을 만들 때 유효숫자를 맞춘다. IQR, drift, round 차이를 통과한 것과 예측 성능이 좋다는 것은 별개다. 이 패키지는 데이터를 수집하며 학습 결과나 논문 결론을 만들어 넣지 않는다.

## GPU 없이 하는 검증

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python suite.py plan pilot
python suite.py plan full
```

조건 생성, 분할 중복, SQLite 재시작·실패 이력, 판정, 결과 내보내기, 측정 준비 로직을 로컬에서 검증한다. 실제 CUDA event·Graph·Nsight 동작 확인은 GPU 파일럿에 남아 있다.

측정 API와 해석 기준: [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html), [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/).
