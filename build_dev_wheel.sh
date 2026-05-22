#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [ -z "${HOME:-}" ] || [ ! -w "${HOME}" ]; then
    export HOME="${REPO_ROOT}/.build-home"
fi

export CONAN_HOME="${CONAN_HOME:-${HOME}/.conan2}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"
export PYTHONNOUSERSITE=1
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
export PATH="${REPO_ROOT}/build_tools:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/cpp/build/_deps/cutlass-src/python:${REPO_ROOT}/cpp/build/_deps/flashmla-src/csrc/cutlass/python:${REPO_ROOT}/cpp/build/_deps/deepgemm-src/third-party/cutlass/python:${PYTHONPATH:-}"
mkdir -p "${CONAN_HOME}" "${PIP_CACHE_DIR}"

# Build TensorRT-LLM for H100, B200, and B300 only.
python3 ./scripts/build_wheel.py --cuda_architectures "90-real;100-real;120-real" --no-venv --clean
