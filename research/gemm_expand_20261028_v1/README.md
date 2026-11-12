# GEMM 추가 10,000형상 수집

사용자가 2026-10-28에 선택한 추가 10,000형상, 180,000측정의 별도 실행본이다. 기존 GEMM 자료와 split, 품질 보류, 모델과 잠금 시험은 보존한다. 현재 패키지는 코드와 목록이며 실제 GPU 수집 결과가 아니다.

새 형상은 기존 범위인 M/N/K=64~4096 안에서 계산량 3구간, 종횡비 2구간과 8정렬 여부 2구간의 12개 층을 채운다. 작은 계산량 1,500개, 중간 4,000개, 큰 계산량 4,500개이며 각 계산량·종횡비 구간 안에서 정렬과 비정렬을 반씩 배정한다. 작은 계산량은 compact 1,000개, elongated 500개다. 모든 층에 같은 수를 강제한 최초 생성은 중복·인접 제외 뒤 충분한 후보를 확보하지 못해 저장하지 않았고, 작은 계산량의 한정된 영역을 과도하게 반복하지 않도록 이 배분을 사용했다. 완전 정렬값과 타일 크기 주변의 비정렬값을 포함하며 형상 선택에 시간 정답을 사용하지 않는다. 9,000형상을 학습, 1,000형상을 새 개발 검증으로 고정한다. 이 검증 구역을 최종 미열람 시험으로 재명명하지 않는다. 기존 main/control/boundary와 보호된 레이어 구성 형상에 대한 정확 중복 및 한 축 ±8 인접 형상을 제외한다. 새 형상 사이에서도 같은 인접 관계를 제외한다. 여기서 인접 검사는 다른 두 축이 정확히 같은 경우를 뜻하며 모든 의미의 가까운 형상까지 차단하는 거리 규칙은 아니다.

각 형상에 FP16/FP32, allocate/prealloc/Graph, 세 회차를 적용하므로 10,000×2×3×3=180,000측정이다. 모든 품질 조건을 통과한다면 새 학습 조건은 54,000개, 기존 완비 학습 3,345형상에 합친 최대 학습 형상은 12,345개가 된다. 회차를 별도 학습 표본으로 세지 않는다. 실제 채택 개수는 반환 후 품질 검사로 결정한다.

보존된 `fixed_sweep_720_v2`에서 목록과 의존성 위치를 바꾼 새 `run.py`를 사용한다. CUDA Event 타이머, 실제 mm/Graph 구현, correctness와 telemetry 검사는 기존 의존 파일을 그대로 사용한다. `analyze_sweep.py`와 checked 환경 래퍼도 보존본을 재사용한다. 720/6801 MHz, 70 W, 세 회차, 50 ms 센서, 120측정 구간, 기존 adaptive Graph 묶음 정책과 클럭 해제 경로를 유지한다. historical cumulative-counter 문제와 clock/SW power-cap REVIEW 후 계속 수집하는 정책도 유지한다. 정확성 실패, 센서 중단, 열 제한, 환경 불일치 등 fatal 문제에서는 중단한다. REVIEW를 자동으로 정상 학습 자료에 넣지 않는다.

원래 완료된 720 MHz 200형상 admission ZIP은 바이트 그대로 포함한다. 실행 시 그 ZIP의 기존 manifest/summary와 해시를 확인하며 원시 시간 DB를 읽는 용도로 사용하지 않는다. 새 full manifest에는 expansion_train과 expansion_validation만 들어 있다. 기존 holdout을 수집하거나 평가하는 명령은 없다.

Ubuntu에서는 소스 ZIP과 체크섬을 `/home/eu/gemm-research/`에 복사한 뒤 현재 tmux에서 실행한다. 기존 폴더를 덮어쓰지 않는다.

```bash
cd /home/eu/gemm-research
(
  set -euo pipefail
  sha256sum -c gemm_expand_20261028_v1_source.zip.sha256
  test ! -e gemm_expand_20261028_v1
  unzip -n gemm_expand_20261028_v1_source.zip
  python3 -B gemm_expand_20261028_v1/selfcheck.py
  bash gemm_expand_20261028_v1/launch.sh full
)
```

이미 압축을 풀어 selfcheck까지 확인했다면 본수집 명령은 다음 한 줄이다.

```bash
bash /home/eu/gemm-research/gemm_expand_20261028_v1/launch.sh full
```

결과, launcher/worker 로그와 종료 ZIP은 실행본 내부 `runs/`에 저장된다. 결과의 source snapshot에는 실행 코드, 형상 목록과 metadata, admission ZIP 및 의존성을 함께 보존한다. 같은 명령으로 중단한 새 수집을 이어갈 수 있으며 기존 full 결과를 가져와 덮어쓰지 않는다. Ctrl+C를 한 번 누르면 현재 작업 저장, 클럭 해제와 ZIP 생성을 기다린다. 기본으로 50 GiB 이상의 여유 디스크를 요구한다. 환경은 기존 ai_env/Python 3.12, CUDA toolkit 13.4와 보존된 실제 GPU/PyTorch signature를 확인하며 설치 위치나 패키지를 바꾸지 않는다. 터미널 출력은 영어 ASCII다.

선택적으로 새 학습 형상 네 개를 사용하는 48측정 환경 확인을 별도 `check` 출력에서 실행할 수 있다. 이 선택 수집은 본수집 180,000건과 별도이고 자동으로 full로 전환하지 않는다.

```bash
bash /home/eu/gemm-research/gemm_expand_20261028_v1/launch.sh check
```

10,000형상 보강의 실제 GPU 소요시간은 아직 측정하지 않았다. 과거 113,382측정에 대한 5~6일 계획값을 단순 비례하면 약 8~10일이지만 실측 ETA가 아니다. 시작 후 worker task wall time과 저장 진행률로 갱신한다. 품질 검산 후에는 고정된 네 학습기 비교 설정으로 기존 학습 자료와 합친 학습곡선을 다시 계산하고, 기존 validation과 새 validation의 오차를 각각 보고한다. 추가 자료로 오차가 줄어드는지는 그 비교로 확인한다.
