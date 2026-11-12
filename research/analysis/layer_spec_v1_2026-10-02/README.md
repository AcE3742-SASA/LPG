# 네 레이어 실험 명세 v1 검토본

2026-10-02. 실제 제출본의 네 레이어와 예측식을 유지하고, 구현 경로와 측정 경계를 아래처럼 구체화했다. [50그룹 목록](groups.json), [구성 연산 목록](operations.json), [기존 GEMM 대응표](existing_gemm_overlap.csv)와 [A/C 내부 역할](gemm_internal_roles.json)을 로컬에서 생성하고 검산했다. GPU 구현, 수집 및 정확성 검증은 아직 수행하지 않았다.

이 문서는 구현할 계약과 검토 가능한 후보 목록이다. 사용자에게 Graph 정책을 먼저 물었고, 답이 오기 전 검토본에는 기존 정책을 이어 쓰는 유료 probe를 기본안으로 적었다. 고정 묶음을 선택하면 Graph 정책과 목록 버전을 바꾼다. 새 GPU 실행이나 환경 변경의 승인은 별도로 받는다. 수치 허용오차와 실제 capture 메모리는 예비 검증 뒤 고정한다.

## 연구 범위와 기준

기준은 [실제 제출본](../../../AcEVault/sources/휴먼테크_제출본_2026-09-22.md)의 §2.2~2.4와 [9월 22일 수행안](../../연구계획_일정_개정_2026-09-22.md)이다. [9월 30일 컨텍스트](../../실험명세_확정용_컨텍스트_2026-09-30.md)의 오래된 품질 수치는 [최종 GEMM 자료](../gemm_dataset_v1_2026-10-02/README.md)로 갱신한다. 이전 SwiGLU나 2,760건 계획은 사용하지 않는다.

질문은 GEMM 예측 및 오차 모델 재사용과 레이어별 조립 보정이 직접 레이어 학습보다 적은 추가 비용으로 같은 정확도에 도달하는가이다. 현재 정확도 및 비용 절감 결과는 없다. 대상은 Linear/FC, 1×1 Conv2D, GELU FFN, n×n Conv2D(explicit GEMM)의 순전파다. v1의 n은 3으로 정한다. bias, dropout, residual addition, backward, cuDNN 합성곱 시간, compile/fusion은 제외한다. `conv2d`는 정확성 참조에만 사용한다.

FP16/FP32, 목표 SM 720 MHz, 메모리 6801 MHz, 기존 전력 한도 70 W를 유지한다. 저장된 환경은 PyTorch 2.14.0+cu132, runtime 13.2, driver 615.71.09이며 실행 전 동일성을 확인한다. 기존 `configure`, `Timer`, `capture`, `select_graph_nodes`를 새 실행본에서 해시로 고정해 재사용한다. 기존 측정에 쓰인 파일은 덮어쓰지 않는다.

## 입력과 실행 순서

공통 그룹의 B=batch, H=W=side, M=BHW, D=FFN 입력 폭, F=hidden 폭, Ci=Cin, Co=Cout으로 둔다. 전체 그룹은 `groups.json`에 실제 값으로 들어 있다. 선형 입력은 `[B,H*W,D]` contiguous, Conv 입력은 `[B,Ci,H,W]` NCHW contiguous다. weight를 원래 수학적 방향에서 GEMM의 `[K,N]` contiguous로 준비하는 변환은 측정 전에 한 번만 한다. 이 준비 비용은 별도 비용 장부에 남긴다.

### Linear/FC

수식은 `Y=XW`다. X를 `[M,D]` view로 보고, 사전 준비한 W[D,Co]와 `torch.mm(X2,W)`를 한 번 호출한다. `(M,N,K)=(M,Co,D)`다. Y[M,Co]를 `[B,H*W,Co]`로 view한다. stride는 X2 `(D,1)`, W `(Co,1)`, Y2 `(Co,1)`이다. reshape가 view라는 조건을 검사하고 불필요한 contiguous 복사를 추가하지 않는다.

### 1×1 Conv2D

stride=1, padding=0, dilation=1, groups=1이다. 공간 위치마다 같은 W를 곱한다. X NCHW의 stride는 `(Ci*H*W,H*W,W,1)`이다. 다음 순서 전체를 한 레이어로 측정한다.

```python
A = X.permute(0, 2, 3, 1).contiguous().view(B*H*W, Ci)
Y2 = torch.mm(A, W)                 # W[Ci,Co], (M,N,K)=(BHW,Co,Ci)
Y = Y2.view(B, H, Wd, Co).permute(0, 3, 1, 2).contiguous()
```

위의 `Wd`는 이미지 폭이며 weight 변수 W와 구분한다. 실제 코드는 이미지 폭을 `width`, weight를 `weight`로 쓴다. A와 GEMM 출력은 각각 `(Ci,1)`, `(Co,1)` stride다. 최종 출력은 contiguous NCHW, stride `(Co*H*Wd,H*Wd,Wd,1)`이다. 입력 pack과 출력 pack의 실제 복사는 T_layer에 들어가며 H의 단독 측정 두 개에도 대응한다. im2col을 추가하지 않는다. batch별 GEMM 반복이나 bmm 대신 전체 BHW를 한 GEMM의 M으로 합친다.

### GELU FFN

순서는 `Z=XW1`, `U=GELU(Z)`, `Y=UW2`다. W1[D,F], W2[F,D]를 모두 contiguous로 준비한다. GEMM은 `(M,F,D)`와 `(M,D,F)` 두 번이다. 중간 및 출력은 Z[M,F], U[M,F], Y[M,D]다. `torch.nn.functional.gelu(Z, approximate='none')`를 사용하고 in-place 연산, tanh 근사 및 fusion을 추가하지 않는다. H의 대상은 GELU 한 번이다. GELU의 정의와 근사 인자는 [PyTorch 2.14 공식 문서](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.gelu.html)에서 확인했다.

### 3×3 Conv2D(explicit GEMM)

kernel=(3,3), stride=1, padding=1, dilation=1, groups=1로 고정한다. 출력 H_o=H, W_o=W, L=H_o*W_o, K=9Ci다. 원래 weight[Co,Ci,3,3]를 `[Co,K]`로 펼치고 transpose 후 contiguous로 만든 Wmat[K,Co]를 준비한다.

```python
U = F.unfold(X, kernel_size=(3, 3), dilation=1, padding=1, stride=1)
A = U.transpose(1, 2).contiguous().view(B*L, K)
Y2 = torch.mm(A, Wmat)              # (M,N,K)=(B*L,Co,9Ci)
Y = Y2.view(B, Ho, Wo, Co).permute(0, 3, 1, 2).contiguous()
```

U[B,K,L]의 반환 shape/stride를 기록하고, A `(K,1)`, Wmat `(Co,1)`, Y2 `(Co,1)`, 최종 NCHW 출력 stride를 검사한다. U를 수정하지 않는다. H의 대상은 unfold, U→A의 배치 복사, 출력 NCHW 복사의 세 단독 호출이다. unfold가 4차원 입력을 받는 것은 [공식 API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.unfold.html)로 확인했으며 실제 GPU 반환 배치는 예비 검증에서 대조한다. cuDNN의 implicit GEMM, FFT, Winograd로 이 결과를 일반화하지 않는다.

## 측정 경계와 단독 연산

예측 대상은 GPU에 상주하는 입력과 weight를 예열한 뒤 고정된 실행 방식으로 반복한 레이어 순전파 1회당 CUDA Event 시간이다. 첫 호출, host 전송, 입력 생성, weight 준비, capture, 정확성 검사, 저장 및 센서 준비는 시간 정답 밖에 둔다. GPU Event 사이에 발생한 dispatch 간격은 실제 측정값에 포함될 수 있다. CPU wall time은 별도로 저장한다.

일반 호출은 매번 GEMM 결과를 반환하는 `torch.mm(a,b)`를 사용한다. 위 순서의 GELU, unfold, contiguous copy도 매 호출에 수행한다. Graph에서는 GEMM 중간 및 출력 버퍼를 사전 할당하고 `torch.mm(a,b,out=buffer)`로 호출한다. `torch.mm`의 두 행렬 shape와 out 인자는 [공식 API](https://docs.pytorch.org/docs/2.14/generated/torch.mm.html)에 따른다. GELU, unfold와 배치 복사는 같은 API를 capture 안에서 수행한다. 이 API들의 capture 중 할당은 Graph 메모리 풀에 속하며 ordinary의 host allocator 경로와 같다거나 순수 할당 차이라고 해석하지 않는다.

| 레이어 | 단독 GEMM 수 | H 단독 연산 | 전체 시간에 포함되는 실제 복사 |
|---|---:|---|---|
| Linear | 1 | 없음 | 없음 |
| 1×1 Conv | 1 | 입력 pack, 출력 pack | 두 pack |
| GELU FFN | 2 | GELU | 없음 |
| explicit 3×3 Conv | 1 | unfold, 펼침 pack, 출력 pack | 펼침 pack와 출력 pack |

view와 permute의 메타데이터 변경은 전체 호출 경로에 있지만 별도 GPU 보조 연산으로 중복 계상하지 않는다. `T_aux`는 위 실제 보조 함수의 단독 시간 합이다. `Delta_assembly`에서 T_aux를 이미 뺀 뒤 동일 복사를 다시 가산하지 않는다.

S_G의 단독 GEMM은 해당 레이어의 실제 중간 입력과 weight를 측정 전에 만들어 보존한 뒤 실행한다. H도 실제 대응 중간 입력을 보존하고 정확히 위 함수만 반복한다. 순전파 전체와 단독 연산은 각각 안정화 및 측정을 거친다. 데이터, 커널의 호출 경로, Graph 묶음과 캐시 재사용이 완전히 같다는 가정은 두지 않는다. 차이는 조립 정답에 남을 수 있다.

단독 함수별 source tensor의 hash, seed 및 operand 역할을 저장한다. 같은 M/N/K나 논리적 aux 키만으로 실측값을 재사용하지 않는다. 예컨대 FFN의 두 번째 GEMM 입력은 GELU 결과이므로 기존 무작위 A 행렬과 동일하지 않다. 키와 실제 operand 정책까지 일치해야 S_G 또는 T_aux의 실측 중복을 제거한다. 이 엄격한 판정과 A 예측기의 재사용은 별개다.

## 시간 프로토콜과 Graph 기본안

전체 180초 예열, 조건별 CUDA Event 누적 3,000 ms 이상 안정화, 목표 100 ms의 7블록, 세 회차를 유지한다. 블록 대표값은 Event 시간/내부 반복 수이며 조건 대표값은 블록별 시간의 중앙값이다. 모델용 조건 행은 세 회차 대표값의 중앙값으로 만든다. 회차는 한 장비의 반복 pass이며 독립 세션 세 개라고 부르지 않는다.

Graph 기본안은 최초 조건의 3회 uncaptured Graph-body probe에서 `q=max(1,min(64,ceil(2 ms/probe_ms)))`를 선택하고 다음 회차에서 재사용한다. probe에서도 Graph용 GEMM prealloc 함수를 사용한다. q_layer는 한 replay에 넣은 레이어 순전파 횟수다. 단독 GEMM과 H는 자기 조건의 probe로 q_component를 정한다. 서로 다른 q를 강제로 동일하게 취급하지 않는다. legacy의 검증된 64회 고정 사례는 legacy GEMM에만 적용하고 새 레이어로 자동 이식하지 않는다.

```text
T_layer,graph = Event_elapsed_ms / (replays * q_layer)
T_component,graph = Event_elapsed_ms / (replays * q_component)
```

FFN의 GEMM 두 개 때문에 레이어 Graph 분모를 `2*q_layer`로 바꾸지 않는다. Graph capture 자체, sizing probe 및 예열은 정답 밖에 두되 실제 횟수, wall time과 준비 비용을 장부에 넣는다. 반복 q는 CUDA 내부 노드 수와 다르다. 비용을 포함한 조건부 예측으로 해석하며 q를 얻기 위한 실측을 무실측 예측이라고 부르지 않는다. v1의 예측 입력에는 probe 시간이나 q를 넣지 않고 해당 정책 자체를 고정한다. q를 입력에 추가하는 비교는 별도 유료 probe 비교로 정의해야 한다.

capture 동안 CPU와 GPU 동기화, 동적 shape, 데이터 의존 Python 분기를 넣지 않는다. replay의 입력 주소와 tensor shape를 고정하고 capture stream의 예열 및 stream 대기를 기존 capture 함수처럼 처리한다. [PyTorch CUDA Graph 제약](https://docs.pytorch.org/docs/2.14/notes/cuda.html#constraints)을 따른다. 실제 GPU에서 API별 capture 가능 여부, 마지막 순전파 출력, 메모리 최대값과 풀 수명을 검사해야 한다. OOM을 이유로 q를 조용히 낮추거나 실패 조건을 지우지 않는다. 변경하면 정책, manifest와 비용을 새 버전으로 남긴다.

## 50그룹과 시험 잠금

`build_plan.py`는 시간값을 사용하지 않고 B∈{1,2,4}, H=W∈{8,12,16,24,32}, D∈{64,72,128,136,256,264,512,520,1024}, F/D∈{2,4}, Ci∈{64,72,128,136,256}, Co∈{64,128,256,512,1024}에서 후보를 만든다. M/F/K/Co가 4096을 넘는 후보를 제외한다. seed 20261002로 후보 순서를 고정한다. 정렬된 처음 50개를 그대로 고른 것이 아니라 구성 GEMM 및 H 키가 이미 선택한 다른 그룹과 겹치면 건너뛴다. 보존된 검증, 탐색 시험, fresh holdout 및 별도 대조 형상과의 충돌도 건너뛴다.

선택한 50그룹의 ID hash로 30 train / 10 validation / 10 test_locked를 배정했다. 한 그룹의 네 레이어, 두 dtype, ordinary/Graph와 모든 회차는 같은 split에 둔다. 5/10/20/30그룹은 `nested_train_position`의 앞부분을 사용한다. 실제 모든 매개변수와 분할은 `groups.json`에 있다. 시간값을 본 뒤 바꾸지 않는다.

로컬 검사에서 그룹 간 구성 GEMM 중복 0, H 논리 키 중복 0이다. GEMM의 한 축만 1~8 달라지는 인접 형상도 서로 다른 그룹에 들어가지 않도록 후보 단계에서 제외하고 검산했다. 이는 기존 boundary 진단의 한 축 변형 범위를 따라 정한 정적 규칙이다. 모든 dtype/mode를 함께 묶어 검사했으므로 이들을 바꾼 같은 연산도 다른 split에 들어가지 않는다. 그룹 내부의 공유는 유지한다. 추가 인접 변형은 원래 group에 붙이고 다른 그룹과 다시 검사한다. 기존 목록 밖의 변형은 새 manifest로 남긴다.

250개의 구성 GEMM 호출 위치 중 기존 legacy_train 형상과 일치하는 위치는 train 2, validation 1, test 1이다. 나머지 246위치는 기존 main M/N/K와 일치하지 않는다. 고유 구성 형상은 242개다. 이 일치는 시간 재사용 확인이 아니다. 시험에서는 기존 구성 GEMM을 본 새 레이어 조합과 구성 형상까지 새로운 경우를 따로 표시한다. validation의 기존 학습 풀 형상 중복 1위치도 별도 표시한다.

잠금 시험은 그룹과 실행 목록의 정적 정보만 설계에 사용한다. 시험 T_layer/S_G/T_aux, probe 시간, kernel 이름 및 성능 순위는 모델 선택 전에 공개하거나 특징에 넣지 않는다. 최종 입력 생성 경로에는 실측 S_G/T_aux가 들어가지 않는다. 시험 측정은 저장과 품질 확인 뒤 잠그고, 설정 및 평가 코드 동결 후에만 시간을 평가한다.

## A, C, H와 g_l의 연결

```text
T_roof = max(2MNK/P, s(MK+KN+MN)/B)
A: T_pred,GEMM = T_roof / eta_pred
S_P = sum(A(component))
S_G = sum(measured_standalone_component_GEMM)
T_aux = sum(measured_standalone_aux)
Delta_execution = S_P - S_G
Delta_assembly = T_layer - (S_G + T_aux)
T_hat_layer = S_P - sum(C(component)) + sum(H(aux)) + g_l(layer)
```

C는 개별 GEMM의 부호 있는 `A−실측`을 학습하는 공통 모델로 시작하고 구성별 예측을 합한다. A를 먼저 고정한다. g_l은 레이어별로 부호 있는 조립 정답을 학습한다. H는 위 실제 aux 종류별 시간을 예측한다. 두 Δ 정답에는 같은 S_G가 들어 있으므로 통계적 독립을 전제하지 않는다. 검증 단계에서 signed error, 공분산과 최종 오차의 상쇄 및 증폭을 확인한다.

A 학습 전에 기존 legacy_train 3,348형상을 2,655 A 후보와 693 C 보정 형상으로 내부 분리한 목록을 준비했다. 시간값 없는 hash 순서로 약 20%인 670개를 먼저 예약하되 레이어 train 구성과 일치하는 기존 2개 형상은 C에 먼저 넣는다. 이후 기존 train 안에서 한 축만 1~8 다른 인접 형상 연결을 모두 C로 묶어 23개가 추가됐다. A/C 사이의 해당 인접 쌍은 0이다. 기존 split은 여전히 legacy_train이며 validation/test를 옮기지 않는다. 정상 세 회차 기준 A 후보는 15,921조건, C 후보는 4,158조건이다. 원래 보류 54건과 불완전 조건을 다시 넣지 않는다. 이 내부 역할은 A 학습 시작 전에 버전으로 고정할 구체안이다.

기존 main split 자체에는 같은 인접 정의의 교차 쌍 113개가 있었다. train/validation 26쌍, train/fresh_holdout 1쌍 등이 포함되며 [원래 split 인접 대응표](original_split_neighbours.csv)에 시간값 없이 보존했다. 원래 시험 분할을 바꾸지 않는다. 기존 평가의 '미관측 형상'은 정확히 같은 M/N/K가 없다는 뜻으로 구분하고, 모든 시험이 멀리 떨어진 형상이라고 해석하지 않는다. 새 레이어 그룹과 A/C 내부 분리에서는 이 인접 교차를 차단했다. 최종 시험의 인접/비인접별 사후 집계는 미리 정의할 수 있지만 시험 시간으로 목록을 재설계하지 않는다.

C를 추가 학습할 때 레이어 train의 실제 단독 GEMM을 사용하면 그 비용을 기록한다. validation/test의 구성 시간은 A/C 재학습에 쓰지 않는다. H와 g_l도 train 그룹만 학습하고 검증 연산 키의 중복을 검사한다. g_l 5그룹 곡선에서 C/H가 이미 대상 30그룹을 사용했다면 대상 총자료가 5그룹이라고 쓰지 않는다. 재사용 A/C/H 버전을 고정한 적응 곡선과 이들의 대상 자료량도 줄이는 곡선을 구분한다.

정적 입력은 레이어 구조, M/N/K와 비율 및 정렬 잔여, dtype/mode, shape/stride/layout, 연산 종류, 논리적 중간 텐서 크기다. 사전 예측 A/H의 값은 허용한다. 실제 클럭 하락, 시험 kernel, 시간이나 실제 DRAM traffic은 입력에서 제외한다. Roofline 효율은 양수로 예측하되 경험적 기준 천장의 한계 때문에 1 이하로 강제하지 않는다. C/g_l 정답은 음수도 유지한다.

## 비교와 비용 계약

제출본 비교군인 무보정 S_P, S_P+H, 종류별 직접 시간 예측, 다른 종류에서 학습 후 대상에 적응하는 공통 보정기를 유지한다. 같은 정적 입력과 사전 A/H 값을 받는 직접 예측, C만 추가, g_l만 추가와 전체 결합을 비교한다. 작은 상수/선형 보정과 통합 잔차 `T_layer−(S_P+H)`도 비교 후보에 넣는다. 통합 잔차는 제출본 주 방법을 대체하지 않는다. 동일 split, 정보 접근과 탐색 예산을 맞춘다.

| 자료 | 중복 제거 전 구체안 |
|---|---:|
| 레이어 전체, 50×4×2×2×3 | 2,400측정 |
| 구성 GEMM, 50×(1+1+2+1)×2×2×3 | 3,000측정 |
| H, 50×(0+2+1+3)×2×2×3 | 3,600측정 |
| 합계 | 9,000측정 |

현재 정적 키만 합치면 구성 GEMM의 최소 후보는 2,904회, H는 3,000회다. 합계 8,304회는 operand hash와 Graph 조건까지 일치해 모두 공유할 수 있을 때의 하한이다. 확인된 시간 재사용은 아직 0건이며 9,000회를 보수적 작업량으로 둔다. 실제 중복 제거와 기존 시간 재사용은 구현이 저장한 키 및 입력을 대조한 뒤 정한다. 9,000회에는 sizing/timer probe, 안정화, correctness, 실패 재측정, profiling이 빠져 있으므로 실제 GPU 호출 수와 총시간은 더 크다. 과거 6,600회를 확정 수량으로 쓰지 않는다.

장부에는 operation ID, source hash, 최초/재사용, 생성/측정/안정화/probe/capture/정확성/검증 wall time, 각 모델의 optimizer step과 실제 학습시간, 탐색 실패 및 profiling을 남긴다. 처음부터 드는 총비용과 A/C/H가 이미 있는 상황의 추가 비용을 함께 보고한다. 같은 목표 오차에 필요한 비용을 비교하고 미도달은 미도달로 남긴다.

A 단계에서도 Roofline, 상수 효율, A만 사용, 상수/선형 실행 보정, A+C와 같은 예산의 직접 GEMM 재학습을 비교해 C가 필요한지 확인한다. 평가량은 mean absolute relative error, `sqrt(mean(((pred−truth)/truth)^2))`, p90 absolute relative error와 μs absolute error다. 레이어/dtype/mode별로 나누고 그룹 단위로 불확실성을 표시한다. 비유한 값과 0 이하 시간 예측은 실패로 집계하며 몰래 제거하거나 clipping해 개선하지 않는다. 모델 크기 소수 후보, seed 반복과 목표 오차는 예비 학습 뒤 검증 자료로 정하고 시험을 열기 전에 고정한다.

## 품질, 저장과 다음 단계

기존 상대 IQR, first/last drift, 회차 간 상대 범위의 각 5% 기준과 50 ms 센서 요청 간격을 유지한다. 최대 120측정의 구간별로 저장하고 환경/telemetry hash를 보존한다. 순간 SW power-cap 또는 SM/memory 이탈은 해당 구간 REVIEW로 유지한다. correctness 실패, CUDA 오류, 비유한 값, 센서 결측/큰 간격, 열 제한, 전력 한도 변경 및 환경 불일치는 중단한다. 누적 전력 제한 카운터의 증가만을 제한 시간 비율로 환산하지 않는다.

full output finite 검사와 CPU FP64 참조를 사용한다. 작은 형상은 전체 출력, 큰 형상은 사전에 고정한 점을 검사한다. GEMM 단독 검사, GELU 계산 참조, unfold patch 인덱스, 전체 FFN 및 Conv 결과와 ordinary/Graph 동일 출력 검사를 모두 준비한다. 기존 GEMM tolerance는 연결된 레이어의 허용오차로 자동 이식하지 않는다. 레이어별 atol/rtol과 sampling plan은 train 예비 형상에서 오차 분포를 대조하고 본측정 전에 고정한다. 이 값의 GPU 확인은 아직 남아 있다.

task ID에는 schema version, group, layer/component 역할, dtype/mode, round 및 실행 계약 hash를 넣는다. 재개 시 완료 ID와 환경/source hash를 대조하며 실패나 REVIEW를 덮어쓰지 않는다. 품질 보류 그룹은 전체 조건을 본측정 학습 표에 넣지 않는다. GPU에서 다른 측정이나 학습을 동시에 실행하지 않는다. 원격 script/output/log/ZIP은 모두 `/home/eu/gemm-research/` 내부에 둔다. 안내, 오류와 helper 출력은 ASCII 영어로 작성한다.

구현할 순서는 위 계약으로 새 레이어 함수를 만들고 CPU 참조와 정적 호출 기록을 대조한 뒤, 사용자 승인 범위의 train 대표 조건에서 correctness/capture/메모리를 검증하는 것이다. 본측정 목록과 실제 재사용량은 그 결과로 동결한다. Graph 선택 외에 초기 사용자의 추가 선택을 요구하지 않고 이 검토본을 준비했으며, 예비 결과가 필요한 항목은 아래에 남긴다.

| 상태 | 항목 |
|---|---|
| 제출본 범위 유지 | 네 종류, dtype, 제외 연산, Δ 부호와 최종 예측식 |
| 로컬 검토본 준비 | 실제 호출 순서, 복사/할당/시간 경계, 50그룹 분할, C 선예약, 대응표, 비용 집계 |
| 사용자 응답 대기, 검토본 기본안 있음 | Graph 유료 probe 또는 고정 묶음 |
| GPU 예비 검증 필요 | 실제 stride, correctness 허용오차, API capture, q별 메모리 및 source tensor 재사용 |
| 시험 전에 고정 | 모델/seed 탐색 예산, 목표 오차, 최종 비용/평가 코드 |

`python3 -B analysis/layer_spec_v1_2026-10-02/build_plan.py`는 stdlib로 목록과 충돌 검사를 다시 생성한다. 정상 출력은 `PASS: 50 groups, 30/10/10 split, zero cross-group operation overlap, C-before-A reservation`이다. [summary.json](summary.json)과 [derived_hashes.json](derived_hashes.json)에 수량, 중복 결과 및 버전을 남겼다. 이 명령에는 GPU, SSH나 Nsight 호출이 없다.
