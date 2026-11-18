#!/usr/bin/env bash
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# Dataflow verification entry point: CPU contracts and SM100 numerical gates.
#
# This script is intentionally a thin wrapper over the documented, typed
# runner. It selects CPU/dataflow coverage (CPU unit/role checks + SM100
# numerical/generated-kernel gates) but no longer treats missing SM100
# hardware as success: required-but-absent hardware fails admission with
# exit code 3 (UNAVAILABLE); a required gate that ran but verified nothing,
# and real failures, exit 1. Usage errors exit 2.
#
#   ./tools/nv/sm100/verify_dataflow.sh                         # cpu + sm100 suites
#   ./tools/nv/sm100/verify_dataflow.sh --references none       # PyTorch-only bench checks
#   ./tools/nv/sm100/verify_dataflow.sh --allow-missing         # developer mode (never a pass)
#
# Every Python/test invocation stays inside locked uv; there is no external
# virtualenv activation and no direct python/pytest invocation.
set -euo pipefail
cd "$(dirname "$0")/../../.."

exec uv run --locked python tools/nv/common/verify_legacy.py --suites cpu,sm100 "$@"
