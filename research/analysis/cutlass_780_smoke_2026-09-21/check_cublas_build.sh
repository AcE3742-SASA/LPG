#!/usr/bin/env bash
# Read-only build diagnosis; no benchmark or settings changes.
set -u
check=$(mktemp /home/eu/cutlass_build_check.XXXXXX.txt) || exit 1
{
  date --iso-8601=seconds
  grep -Ei 'CUTLASS_ENABLE_CUBLAS|CUBLAS.*(FILEPATH|BOOL)' /home/eu/cutlass/build/CMakeCache.txt
  grep -E 'CUTLASS_ENABLE_CUBLAS' /home/eu/cutlass/build/tools/profiler/CMakeFiles/cutlass_profiler.dir/flags.make
  ldd /home/eu/cutlass/build/tools/profiler/cutlass_profiler | grep -Ei 'cublas|not found'
} > "$check" 2>&1
cat "$check"
printf '\n저장: %s\n' "$check"
