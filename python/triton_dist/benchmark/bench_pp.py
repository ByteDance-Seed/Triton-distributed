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

import subprocess
import re
import csv


def run_test(max_tokens, hidden_size, num_sms):
    """Runs the test command and returns the output."""
    command = [
        'bash', 'scripts/launch.sh', 'python/triton_dist/test/nvidia/test_pp.py', '--max_tokens',
        str(max_tokens), '--hidden_size',
        str(hidden_size), '--num_sms',
        str(num_sms)
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with parameters: {max_tokens}, {hidden_size}, {num_sms}):")
        print(f"Error output: {e.stderr}")
        return None


def parse_output(output):
    """Parses the output content to extract the required timing information."""
    put_get_pattern = r"Triton put/get: (\d+\.\d+) ms"
    triton_torch_pattern = r"Triton: (\d+\.\d+) ms, Torch: (\d+\.\d+) ms"

    triton_put_get_time = None
    triton_time = None
    torch_time = None

    put_get_match = re.search(put_get_pattern, output)
    if put_get_match:
        triton_put_get_time = float(put_get_match.group(1))

    triton_torch_match = re.search(triton_torch_pattern, output)
    if triton_torch_match:
        triton_time = float(triton_torch_match.group(1))
        torch_time = float(triton_torch_match.group(2))

    return triton_put_get_time, triton_time, torch_time


def main():
    max_tokens_values = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    hidden_size_values = [6144]
    num_sms_values = [1, 2, 4, 6, 8, 10, 12, 14, 16]

    results = []
    results.append(
        ["max_tokens", "hidden_size", "num_sms", "triton_put_get_time(ms)", "triton_time(ms)", "torch_time(ms)"])

    # Iterate over all parameter combinations
    for max_tokens in max_tokens_values:
        for hidden_size in hidden_size_values:
            for num_sms in num_sms_values:
                print(f"Testing with: max_tokens={max_tokens}, hidden_size={hidden_size}, num_sms={num_sms}")

                output = run_test(max_tokens, hidden_size, num_sms)
                if not output:
                    continue

                put_get_time, triton_time, torch_time = parse_output(output)

                print(f"Parsed results: Triton put/get={put_get_time}ms, Triton={triton_time}ms, Torch={torch_time}ms")

                results.append([max_tokens, hidden_size, num_sms, put_get_time, triton_time, torch_time])

    with open('triton_benchmark_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print("Tests complete. Results saved to triton_benchmark_results.csv")


if __name__ == "__main__":
    main()
