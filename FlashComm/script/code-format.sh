#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to the project root (one level up from script/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_only=0
fail_on_diff=0

while [ "$#" -gt 0 ]; do
  case $1 in
  -h | --help)
    echo "Usage: $0 [--fail-on-diff] [--show-only]"
    exit 0
    ;;
  --fail-on-diff)
    fail_on_diff=1
    shift
    ;;
  --show-only)
    show_only=1
    shift
    ;;
  *)
    echo "error: unrecognized option $1"
    exit 1
    ;;
  esac
done

# Collect C++/CUDA files (exclude build/)
files_to_check_cpp=$(find "$PROJECT_ROOT" \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.cuh" -o -name "*.cc" -o -name "*.hpp" -o -name "*.c" \) \
  ! -path "*/build/*" 2>/dev/null)

# Collect Python files (exclude build/, dist/, __pycache__/, egg-info/)
files_to_check_py=$(find "$PROJECT_ROOT" -name "*.py" \
  ! -path "*/build/*" ! -path "*/dist/*" ! -path "*/__pycache__/*" ! -path "*.egg-info/*" 2>/dev/null)

if [ -z "$files_to_check_cpp" ] && [ -z "$files_to_check_py" ]; then
  echo "No files to format."
  exit 0
fi

# Check tools
if ! clang-format --version >/dev/null 2>&1; then
  echo "error: clang-format not found, can be installed by: pip3 install clang-format"
  exit 1
fi

if ! command -v yapf &>/dev/null; then
  echo "error: yapf not found, can be installed by: pip3 install yapf"
  exit 1
fi

if ! command -v ruff &>/dev/null; then
  echo "error: ruff not found, can be installed by: pip install ruff"
  exit 1
fi

if [ "$show_only" -eq 1 ]; then
  has_diff=0

  # C++/CUDA files
  for f in $files_to_check_cpp; do
    [ -L "$f" ] && continue
    result=$(diff "$f" <(clang-format -style=file --assume-filename="$f" "$f"))
    if [ "$result" != "" ]; then
      echo "clang-format ===== $f ====="
      echo -e "$result"
      has_diff=1
    fi
  done

  # Python files
  for f in $files_to_check_py; do
    [ -L "$f" ] && continue
    result=$(diff "$f" <(yapf "$f"))
    if [ "$result" != "" ]; then
      echo "yapf ===== $f ====="
      echo -e "$result"
      has_diff=1
    fi
    ruff_result=$(ruff check --config "$PROJECT_ROOT/pyproject.toml" --diff "$f" 2>/dev/null)
    if [ "$ruff_result" != "" ]; then
      echo "ruff ===== $f ====="
      echo -e "$ruff_result"
      has_diff=1
    fi
  done
else
  # Format C++/CUDA files
  if [ -n "$files_to_check_cpp" ]; then
    echo "Formatting C++/CUDA files by clang-format..."
    echo "$files_to_check_cpp" | xargs clang-format -i -style=file
  fi

  # Format Python files by yapf
  if [ -n "$files_to_check_py" ]; then
    echo "Formatting Python files by yapf..."
    for f in $files_to_check_py; do
      yapf -i "$f"
    done
  fi

  # Lint+fix Python files by ruff
  if [ -n "$files_to_check_py" ]; then
    echo "Formatting Python files by ruff..."
    echo "$files_to_check_py" | xargs ruff check --config "$PROJECT_ROOT/pyproject.toml" --fix
  fi

  # Ensure final newline for Python files
  echo "Ensuring final newline for Python files..."
  for f in $files_to_check_py; do
    if [ -s "$f" ] && [ -n "$(tail -c1 "$f")" ]; then
      echo >> "$f"
    fi
  done
fi

if [ "$fail_on_diff" -eq 1 ] && [ "$has_diff" -eq 1 ]; then
  echo "code format check failed, please run: bash script/code-format.sh"
  exit 1
fi

echo "Code formatting complete."
