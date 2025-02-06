#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PARAMS_FILE=${SCRIPT_DIR}/aot_kernels.txt
mapfile -t AOT_KERNELS < "$PARAMS_FILE"

echo $SCRIPT_DIR
echo $PARAMS_FILE
AOT_KERNELS=()
while IFS= read -r line; do
    resolved_line="${line//\$\{SCRIPT_DIR\}/$SCRIPT_DIR}"
    echo $line
    echo $resolved_line
    AOT_KERNELS+=("$resolved_line")
done < $PARAMS_FILE

echo $AOT_KERNELS

rm -rf ${SCRIPT_DIR}/../third_party/distributed/csrc/triton_aot_generated
python3 ${SCRIPT_DIR}/../third_party/distributed/distributed/tools/compile_aot.py --workspace ${SCRIPT_DIR}/../third_party/distributed/csrc/triton_aot_generated --kernels ${AOT_KERNELS[@]}  --library triton_distributed_kernel
