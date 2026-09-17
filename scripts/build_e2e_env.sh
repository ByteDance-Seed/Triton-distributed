################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

#!/bin/bash

# --- NVIDIA CUDA ---
if command -v nvcc &> /dev/null; then
    echo "NVIDIA CUDA compiler (nvcc) found. Proceeding with CUDA-specific installations."
    cuda_version_full=""
    parsed_cuda_slug=""

    cuda_version_output=$(nvcc --version)
    if [[ $cuda_version_output =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
        cuda_major="${BASH_REMATCH[1]}"
        cuda_minor="${BASH_REMATCH[2]}"
        cuda_version_full="${cuda_major}.${cuda_minor}"
        parsed_cuda_slug="cu${cuda_major}${cuda_minor}"
        echo "Detected CUDA version: $cuda_version_full (Using slug: $parsed_cuda_slug)"
    else
        echo "Error: Could not parse CUDA version from 'nvcc --version' output."
        echo "Output was: $cuda_version_output"
        exit 1
    fi

    pytorch_version_full=""
    parsed_pytorch_slug=""

    PY_EXECUTABLE=""
    if command -v python3 &> /dev/null; then
        PY_EXECUTABLE="python3"
    elif command -v python &> /dev/null; then
        PY_EXECUTABLE="python"
    else
        echo "Error: Neither 'python3' nor 'python' command found. Cannot detect PyTorch version."
        exit 1
    fi

    pytorch_version_output=$($PY_EXECUTABLE -c "import sys; sys.path.append('.'); import torch; print(torch.__version__)" 2>/dev/null)
    if [[ -n "$pytorch_version_output" ]]; then
        pytorch_cleaned_version=$(echo "$pytorch_version_output" | sed -E 's/\+.*//' | sed -E 's/\.dev.*//')
        if [[ $pytorch_cleaned_version =~ ^([0-9]+)\.([0-9]+) ]]; then
            torch_major="${BASH_REMATCH[1]}"
            torch_minor="${BASH_REMATCH[2]}"
            parsed_pytorch_slug="torch${torch_major}.${torch_minor}"
            echo "Detected PyTorch version: $pytorch_version_output (Using slug: $parsed_pytorch_slug)"
        else
            echo "Error: Could not parse major.minor PyTorch version from '$pytorch_version_output'."
            exit 1
        fi
    else
        echo "Error: Could not import PyTorch or get its version."
        echo "Please ensure PyTorch is installed in the current Python environment ($PY_EXECUTABLE)."
        exit 1
    fi

    # --- flashinfer and flash-attn ---
    if [[ -z "$parsed_cuda_slug" || -z "$parsed_pytorch_slug" ]]; then
        echo "Error: CUDA slug or PyTorch slug could not be determined. Exiting."
        exit 1
    fi

    echo "Installing CUDA-specific libraries..."

    # Check for local flashinfer whl files first
    local_flashinfer_whl=$(find . -name "*flashinfer*.whl" | head -1)
    if [[ -n "$local_flashinfer_whl" ]]; then
        echo "Found local flashinfer whl: $local_flashinfer_whl"
        echo "Installing local flashinfer whl..."
        pip install "$local_flashinfer_whl"
    else
        echo "No local flashinfer whl found, installing from remote..."
        flashinfer_whl_url="https://flashinfer.ai/whl/${parsed_cuda_slug}/${parsed_pytorch_slug}/"
        pip install flashinfer-python -i "$flashinfer_whl_url"
    fi
    python3 -c "import flash_attn" >/dev/null 2>&1 || pip3 install flash-attn --no-build-isolation --no-deps
    echo "Finished installing CUDA-specific libraries."

# --- AMD ROCm ---
elif command -v hipcc &> /dev/null; then
    echo "AMD ROCm compiler (hipcc) found. Proceeding with ROCm-specific installations."
    echo "Note: flashinfer does not currently support ROCm and will be skipped."
    python3 -c "import flash_attn" >/dev/null 2>&1 || pip3 install flash-attn --no-build-isolation --no-deps
    echo "Finished installing ROCm-specific libraries."
else
    echo "NVIDIA CUDA compiler (nvcc) and AMD ROCm compiler (hipcc) not found."
    echo "Assuming a non-GPU environment. Skipping flashinfer and flash-attn installation."
fi

# --- Install common packages ---
echo "Installing common packages: transformers and numpy..."
pip install transformers==4.51.3 numpy==1.26.4 termcolor
# deepspeed is not imported by triton_dist itself, but transformers==4.51.3
# auto-probes and imports it while loading models (is_deepspeed_available()).
# deepspeed 0.19.3 eagerly imports transformers.models.opt at package init,
# which re-enters a still-initializing transformers.modeling_utils and dies with
# a circular import ("cannot import name 'PreTrainedModel'"). Pin to the last
# release that imports cleanly with transformers 4.51.3 (0.19.2 verified good;
# 0.19.3 is the first bad one) instead of always taking latest via --upgrade.
pip install deepspeed==0.19.2

# --- Define Hugging Face models to download ---
MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-32B"
  "Qwen/Qwen3-30B-A3B"
)

download_model=false
skip_model_download=false

for arg in "$@"; do
    case $arg in
        --download_model)
            download_model=true
            shift
            ;;
        --skip_model_download)
            skip_model_download=true
            shift
            ;;
        *)
            ;;
    esac
done

if [ "$download_model" = true ] && [ "$skip_model_download" = false ]; then
    # GitHub AMD e2e only needs the 0.6B checkpoint. The default list used
    # to include 8B/32B/MoE and retried forever, so one timed-out 32B fetch
    # could hold the job until the workflow cap.
    if [ -n "${E2E_MODELS:-}" ]; then
        read -r -a MODELS <<< "${E2E_MODELS}"
    fi
    max_attempts="${E2E_DOWNLOAD_ATTEMPTS:-3}"
    for MODEL_NAME in "${MODELS[@]}"; do
    attempt=1
    while [ "${attempt}" -le "${max_attempts}" ]; do
        echo "Attempting to download model: $MODEL_NAME (timeout: 120s, attempt ${attempt}/${max_attempts})..."
        timeout 120s huggingface-cli download "$MODEL_NAME"

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
        echo "Model '$MODEL_NAME' downloaded successfully! 🎉"
        break
        elif [ $EXIT_CODE -eq 124 ]; then
        echo "Download timed out for '$MODEL_NAME'."
        else
        echo "Download failed for '$MODEL_NAME' with exit code $EXIT_CODE."
        fi
        if [ "${attempt}" -eq "${max_attempts}" ]; then
            echo "error: giving up on '$MODEL_NAME' after ${max_attempts} attempts" >&2
            exit 1
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    done

    echo "All specified models have been downloaded."
elif [ "$skip_model_download" = true ]; then
    echo "Skipping model download (using pre-downloaded model weights)"
    echo "Ensure HF_HOME or HUGGINGFACE_HUB_CACHE environment variable points to the correct model cache directory"
else
    echo "No Model download required. If you want to download model, add '--download_model' option."
fi

pip install accelerate

# In the out-of-tree plugin model the patched, plugin-hosting Triton (built by
# scripts/build_triton.sh and pip-installed as the `triton` dist that
# libtriton_dist.so binds to) is a HARD dependency of triton_dist -- not a
# vendored tree on PYTHONPATH. Never `pip uninstall triton` here: that removes the
# very Triton the plugin loads into, so `import triton` -> `import triton_dist` ->
# _plugin.find_plugin() all break and plugin discovery returns empty (this is what
# `pip uninstall triton -y`, inherited from the old monolithic layout, used to do
# and it silently broke the megakernel/e2e jobs). If an e2e dependency installed
# above pulled a stock PyPI `triton` wheel that shadowed the patched build,
# re-assert the patched Triton so it keeps hosting the plugin.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if python3 -c "from triton._C.libtriton import passes; assert hasattr(passes, 'plugin')" 2>/dev/null; then
    echo "Patched plugin-hosting Triton is intact; leaving it in place."
else
    echo "Patched Triton missing or shadowed by a stock wheel; reinstalling the plugin-hosting Triton..."
    bash "${SCRIPT_DIR}/build_triton.sh"
fi

# --- Final check ---
if [[ $? -eq 0 ]]; then
    echo "E2E environment installation successful."
else
    echo "Error: E2E environment installation failed."
    exit 1
fi
