# 표준 PyTorch 레이어와 GEMM 분해 경로 비교 v1

2026-10-06 사용자 승인과 형상 확대 요청을 반영한 실행본이다. 아직 연구 PC의 CUDA 측정 결과는 없다. 두 경로 모두 PyTorch를 사용하며 자체 CUDA 커널을 작성하지 않는다. 형상별 실제 값은 [cases.json](cases.json), 고정 조건은 [protocol.json](protocol.json)에 있다.

## 범위와 실행 순서

| 레이어 | 형상 수 | 표준 경로 | 분해 경로 |
|---|---:|---|---|
| 1×1 Conv | 64 | `nn.Conv2d`, bias 없음 | NCHW 입력 pack → `torch.mm` → NCHW 출력 pack |
| 3×3 Conv | 64 | `nn.Conv2d`, padding 1, bias 없음 | `F.unfold` → patch pack → `torch.mm` → NCHW 출력 pack |
| Linear | 16 | `nn.Linear`, bias 없음 | 준비된 weight로 `torch.mm` |
| GELU FFN | 16 | `nn.Linear → nn.GELU → nn.Linear` | `torch.mm → F.gelu → torch.mm` |

160형상 × 2 dtype × 2경로 × 3회차 = **1,920측정**, 본 시간 블록은 13,440개다. 먼저 네 종류에서 각 한 형상씩 FP16/FP32로 48측정을 수행한다. 정확성, 5% 시간 변동 기준, 시간 블록의 센서 표본과 클럭·전력 제한 검사를 모두 통과하면 나머지 1,872측정으로 자동 진행한다. 대표 조건의 REVIEW/FAIL에서는 전체 수집을 시작하지 않고 결과를 보존한다. 이 48측정도 1,920에 포함된다.

Conv는 batch 1/2/4, 공간 8/16/32/64, 입력 채널 기준 16/32/64/128/256을 포함한 16개 기준 묶음으로 구성했다. 각 묶음에서 기준, 입력 채널 ±1, 출력 채널 ±1, 공간 `(side−1, side+1)`을 비교한다. Linear/FFN은 네 기준 묶음에서 행 수·입력 폭·출력 또는 hidden 폭을 하나씩 바꾼다. 이 목록이 실제 네트워크 분포를 대표한다고 검증된 것은 아니다. 일부 구성 GEMM은 기존 자료의 64~4096 범위 밖이므로 나중에 기존 예측기로 평가할 때 외삽 조건을 별도로 표시해야 한다.

시간값 없는 `measurement_index.csv`의 전체 식별자를 사용해 기존 GEMM validation 717형상·탐색 시험 718형상·fresh holdout 1,000형상과 레이어 명세 v1 validation 49·test_locked 50 구성 GEMM 형상을 제외했다. 합집합은 2,534형상이다. 정확히 같은 M/N/K와 한 축만 ±8 차이 나는 후보도 제외했다. [selection_audit.json](selection_audit.json)에 메타데이터 해시와 선택 근거를 남겼고 [선택 코드](select_cases.py)로 목록을 재현할 수 있다. 전부 개발용 비교 자료이며 새로운 최종 시험 자료가 아니다. 기존 split과 자료를 교체하거나 모델 학습에 넣지 않는다.

## 측정 계약

- GPU에 상주하는 동일 입력과 원래 weight 값을 두 경로가 공유한다. seed와 실제 tensor 해시를 세 회차 대조한다. Conv 입출력은 contiguous NCHW, Linear/FFN은 2D다. 출력 배치가 다르면 조용히 변환하지 않고 중단한다.
- FP32는 matmul와 cuDNN Conv 모두 IEEE로 설정한다. FP16 reduced-precision reduction/accumulation 설정은 기존 검증 코드를 따른다. inference mode, bias 없음, exact GELU이며 autocast·Graph·compile·backward를 쓰지 않는다.
- 표준 Conv는 `cudnn.benchmark=True`, `deterministic=False`다. 알고리즘 선택과 첫 호출은 준비 단계에 들어가며 설정과 cuDNN 버전을 저장한다. 이 기준을 모든 최적화 방식의 최고 성능이라고 부르지 않는다. channels_last 비교는 이번 실행에 포함하지 않았다.
- 원래 weight와 분해용 `[K,N]` weight 준비는 시간 정답 밖에 둔다. Conv의 unfold와 실제 배치 복사는 매 순전파 안에 포함한다. 준비·정확성 검사·타이머 안정화 비용도 별도 저장한다.
- 기존 CUDA Event 타이머로 전역 180초 예열, 측정별 누적 3000 event-ms 안정화와 100ms 목표의 7블록을 사용한다. 경로 순서를 번갈아 배치하고 회차별 조건 순서를 바꾼다. CPU dispatch 때문에 Event 사이 GPU 유휴 시간이 포함될 수 있으며 순수 커널 시간이라고 해석하지 않는다.
- 기존 720/6801MHz와 70W를 유지한다. 전력 한도는 확인만 한다. 센서는 50ms 간격을 요청하며 관측 간격과 원자료를 보존한다. IQR·first/last drift·회차 간 상대 범위의 5% 기준, 블록 센서 결측, 클럭 이탈과 SW power-cap을 REVIEW로 남긴다. 정확성 오류, CUDA/환경 오류, 센서 중단·5초 초과 결측과 열 제한은 중단한다.
- 기존 GPU 잠금·sudo credential 갱신·프로세스 검사·클럭 해제 코드를 보존해 재사용한다. 종료 후 기본 클럭으로 해제하는 명령의 수락을 기록한다. 이전의 임의 수동 lock 상태를 복구한다는 뜻은 아니다.

## 정확성 및 메모리

실제 dtype으로 생성한 입력·weight를 CPU FP64로 옮겨 고정된 출력 위치의 수학적 참조를 계산한다. Conv는 padding과 batch를 포함한 원래 patch를 직접 선택하고, FFN은 두 GEMM과 exact GELU 전체를 계산한다. 두 경로 모두 참조와 대조하고 출력 전체의 finite 및 경로 간 일치도 확인한다. 검사 위치에 처음·마지막·중앙 위치를 포함한다.

FP32 `atol=2e-5, rtol=2e-4`, FP16 `atol=1e-3, rtol=5e-3`을 예비 기준으로 고정했다. 기존 GEMM 기준에서 출발한 보수적 경계이며 새 레이어의 CUDA 검증 완료를 뜻하지 않는다. 실패하면 근거와 원자료를 남기고 중단하며 허용오차를 자동으로 높이지 않는다. 큰 출력의 FP64 참조는 위치별 표본 검사이고 전체 출력의 FP64 검증이 아니다.

메모리는 시간 측정 뒤 예열된 순전파 한 번에서 `max_memory_allocated − 준비 상태의 memory_allocated`로 기록한다. 두 weight 배치가 함께 준비된 상태에서의 추가 transient allocated 메모리이며 전체 애플리케이션 메모리나 실제 DRAM traffic이 아니다. 분해용 weight의 영구 추가 바이트와 allocator reserved peak는 별도로 남긴다. correctness와 메모리 검사는 본 시간 블록 밖에 있다.

## 연구 PC 실행

배포 ZIP을 `/home/eu/gemm-research/scripts/`에 풀어 `layer_compare_v1/` 폴더를 만든다. 기존 설치와 실행본을 덮어쓰거나 재설치하지 않는다. 기존 tmux에서 정상 사용자 `eu`로 실행한다.

```bash
bash /home/eu/gemm-research/scripts/layer_compare_v1/launch.sh
```

기존 `ai_env`, CUDA와 센서를 사용한다. 터미널 출력은 영어 ASCII다. 실행본 내부 `runs/layer_compare_<시각>_<고유값>/`에 매번 새 결과를 만들고 원본 소스·의존 파일·manifest·센서·실패를 보존한다. 종료 시 정상·REVIEW·실패 상태를 모두 압축하고 `RESULT_ZIP=`을 출력한다. 해당 ZIP을 반환하면 비교 결과를 분석할 수 있다.

본측정의 명목 안정화와 본 블록만 약 118.4분이고 전역 예열 3분과 초기 보정·정확성·저장·메모리 검사가 추가된다. 실행 예산은 약 2~3시간으로 잡되 실제 wall time은 실행 후 확인한다. 실행본의 worker timeout은 6시간이다. 다른 GPU 측정·학습과 동시에 실행하지 않는다.

## 결과와 로컬 검증

`comparison.csv`는 형상·dtype별 두 경로의 중앙값, 시간 비율, 절대 차이와 추가 최대 메모리를 제공한다. 회차별 비율·변동과 품질 판정은 `summary.json`, 원래 블록은 `records/`, 입력 해시는 `inputs/`, 정확성은 `correctness/`에 있다. REVIEW/FAIL 비율을 검증된 속도 차이로 사용하지 않는다. 전체 완료와 품질 통과를 별도로 판단한다.

이 값은 **구현 경로의 시간 차이**다. 기존 식의 `Delta_assembly`, 예측 오차나 보정 모델의 유효성으로 해석하지 않는다. 차이가 큰 대표 조건의 구성 연산 측정은 반환 결과를 검토한 다음 단계다. 이번에는 Nsight를 실행하지 않는다.

```bash
python3 -B compare.py verify
python3 -B test_compare.py
python3 -B select_cases.py
python3 -B test_compare.py --cpu
```

앞의 세 검사는 PyTorch 없이 정적 범위·해시·품질 집계와 목록 재현을 검산한다. `--cpu`는 PyTorch가 설치된 환경에서 작은 비정렬·직사각형·batch 및 padding 조건과 실제 대표 형상의 두 경로, 독립된 전체 FP64 참조와 주입한 weight 오류를 검사한다. 로컬 CPU 검사는 macOS 임시 환경의 PyTorch 2.14.1을 사용하며 연구 PC의 PyTorch 2.14.0+cu132, CUDA 커널·시간·메모리는 확인하지 않는다. 검증 결과는 [validation.json](validation.json)에 있다.

API 근거: [Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html), [Unfold](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Unfold.html), [CUDA precision](https://docs.pytorch.org/docs/2.14/notes/cuda.html), [NVIDIA convolution algorithms](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html).
