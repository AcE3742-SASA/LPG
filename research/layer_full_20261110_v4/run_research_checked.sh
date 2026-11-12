#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

if (( $# == 0 )); then
  echo "Usage: $0 <experiment command> [arguments ...]" >&2
  exit 2
fi

wrapper_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
research_root="${GEMM_RESEARCH_ROOT:-$wrapper_here}"
runs_dir="$research_root/runs"
conda_setup="$HOME/miniconda3/etc/profile.d/conda.sh"

if [[ ! -r "$conda_setup" ]]; then
  echo "FAIL: Conda setup file not found: $conda_setup" >&2
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
    problems.append("Conda environment is not ai_env.")
if sys.version_info[:2] != (3, 12):
    problems.append(f"Expected Python 3.12, got: {sys.version.split()[0]}")
if os.environ.get("CUDA_HOME") != "/usr/local/cuda-13.4":
    problems.append(f"Unexpected CUDA_HOME: {os.environ.get('CUDA_HOME')}")
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False.")

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
        problems.append(f"Unexpected GPU: {gpu_name}")
    if capability != (8, 9):
        problems.append(f"Expected compute capability 8.9, got: {capability}")

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
  echo "FAIL: Could not verify nvcc 13.4." | tee -a "$environment_log" >&2
  exit 1
fi

echo "RUN_DIR=$run_dir"
experiment_started_ns=$(python -c 'import time; print(time.time_ns())')
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
elif (( experiment_status == 2 )) && python - "${GEMM_RESEARCH_SUMMARY:-}" "$experiment_started_ns" <<'PY'
import json
import os
import sys
try:
    with open(sys.argv[1]) as f:
        result = json.load(f)
    valid = (os.stat(sys.argv[1]).st_mtime_ns >= int(sys.argv[2]) and result['gate'] == 'REVIEW'
             and result['counts'].get('FAIL', 0) == 0
             and result['collection'] in ('COMPLETE', 'PRECHECK_STOPPED'))
except (OSError, ValueError, KeyError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
  echo "EXPERIMENT: REVIEW (inspect summary.json; the startup check may have stopped collection; exit 2)"
else
  echo "EXPERIMENT: FAIL (exit $experiment_status)" >&2
fi

exit "$experiment_status"
