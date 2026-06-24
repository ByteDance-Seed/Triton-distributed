#!/bin/bash
set -x

TRITON_URL="https://github.com/MetaX-MACA/mcTriton.git"
TRITON_SUBMODULE="3rdparty/triton"

if [[ $# -eq 1 ]]; then
  TRITON_BRANCH="3.6"
  TRITON_COMMIT="$1"
else
  TRITON_BRANCH="${1:-3.6}"
  TRITON_COMMIT="${2:-}"
fi

if [[ -z "${TRITON_COMMIT}" ]]; then
  echo "Usage: $0 [branch] <commit>"
  echo "Example: $0 3.6 70eb31eb1899ec64f1a301b3b065e7cf2c9f5802"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"

cd "${PROJECT_ROOT}"

git config -f .gitmodules "submodule.${TRITON_SUBMODULE}.url" "${TRITON_URL}"
git config -f .gitmodules "submodule.${TRITON_SUBMODULE}.branch" "${TRITON_BRANCH}"
git submodule sync -- "${TRITON_SUBMODULE}"

git submodule deinit -f -- "${TRITON_SUBMODULE}" || true
rm -rf "${TRITON_SUBMODULE}"

git submodule update --init --remote -- "${TRITON_SUBMODULE}"
git -C "${TRITON_SUBMODULE}" checkout --detach "${TRITON_COMMIT}"
