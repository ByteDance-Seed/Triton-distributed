#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_units() {
  python3 python/little_kernel/tests/integration/test_sm90_gemm.py
  python3 python/little_kernel/tests/unit/test_codegen_comprehensive.py
  python3 python/little_kernel/tests/unit/test_cuda_asm_intrinsics.py
  python3 python/little_kernel/tests/unit/test_stmt.py
  python3 python/little_kernel/tests/unit/test_struct_stub.py
  python3 python/little_kernel/tests/unit/test_type_promotion.py 
  python3 python/little_kernel/tests/unit/test_type_system.py
}

run_units