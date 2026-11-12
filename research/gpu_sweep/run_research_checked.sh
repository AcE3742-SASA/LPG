#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# == 0 )); then
  echo "사용법: $0 <실험 명령> [인자 ...]" >&2
  exit 2
fi

research_root="${GEMM_RESEARCH_ROOT:-$HOME/gemm-research}"
runs_dir="$research_root/runs"
conda_setup="$HOME/miniconda3/etc/profile.d/conda.sh"

if [[ ! -r "$conda_setup" ]]; then
  echo "FAIL: Conda 초기화 파일을 찾을 수 없습니다: $conda_setup" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$conda_setup"
conda activate ai_env

export CUDA_HOME=/usr/local/cuda-13.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$runs_dir"
run_dir="$(mktemp -d "$runs_dir/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
environment_log="$run_dir/environment.txt"

printf -v experiment_command '%q ' "$@"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "working_directory=$(pwd)"
  echo "execution_directory=$run_dir"
  echo "command=$experiment_command"
  echo "runner_sha256=$(sha256sum "$0" | cut -d ' ' -f 1)"
  echo "python=$(command -v python)"
  echo "nvcc=$(command -v nvcc)"
  echo "CUDA_HOME=$CUDA_HOME"
  echo

  python - <<'PY'
import importlib.metadata as metadata
import os
import sys

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import torch
import torchvision

packages = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
)

problems = []
if os.environ.get("CONDA_DEFAULT_ENV") != "ai_env":
    problems.append("Conda 환경이 ai_env가 아닙니다.")
if sys.version_info[:2] != (3, 12):
    problems.append(f"Python 3.12가 아닙니다: {sys.version.split()[0]}")
if os.environ.get("CUDA_HOME") != "/usr/local/cuda-13.4":
    problems.append(f"CUDA_HOME이 예상과 다릅니다: {os.environ.get('CUDA_HOME')}")
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available()가 False입니다.")

print(f"python_version={sys.version.split()[0]}")
for package in packages:
    print(f"{package}={metadata.version(package)}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(f"gpu={gpu_name}")
    print(f"compute_capability={capability[0]}.{capability[1]}")
    if "RTX 4000 SFF Ada" not in gpu_name:
        problems.append(f"예상 GPU가 아닙니다: {gpu_name}")
    if capability != (8, 9):
        problems.append(f"예상 compute capability 8.9가 아닙니다: {capability}")

if problems:
    print("ENV CHECK: FAIL")
    for problem in problems:
        print(f"- {problem}")
    raise SystemExit(1)

print("ENV CHECK: PASS")
PY

  echo
  nvcc --version
  ncu --version
  nsys --version
  nvidia-smi
} 2>&1 | tee "$environment_log"

if ! grep -Fq "release 13.4" "$environment_log"; then
  echo "FAIL: nvcc 13.4를 확인하지 못했습니다." | tee -a "$environment_log" >&2
  exit 1
fi

echo "RUN_DIR=$run_dir"
set +e
(
  cd "$run_dir"
  "$@"
) 2>&1 | tee "$run_dir/output.log"
experiment_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$experiment_status" > "$run_dir/exit_code.txt"

if (( experiment_status == 0 )); then
  echo "EXPERIMENT: PASS"
else
  echo "EXPERIMENT: FAIL (exit $experiment_status)" >&2
fi

exit "$experiment_status"
