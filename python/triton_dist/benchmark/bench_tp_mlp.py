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
import os
import subprocess
import csv
import argparse
import math
import time


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run MLP performance tests with modes.')
    parser.add_argument('--resume', action='store_true', help='Resume from last run')
    parser.add_argument('--output', default='mlp_results_with_modes.csv', help='Output CSV file name')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for failed commands')
    return parser.parse_args()


def load_completed_tasks(filename):
    """Load completed tasks from CSV file"""
    completed = set()
    if not os.path.exists(filename):
        return completed

    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['model'], row['M'], row['mode'])
                completed.add(key)
    except Exception as e:
        print(f"Warning: Could not load completed tasks from {filename}: {str(e)}")

    return completed


def initialize_csv(filename):
    """Initialize CSV file with header if it doesn't exist"""
    if not os.path.exists(filename):
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'model', 'M', 'mode', 'torch_mean', 'torch_std', 'dist_triton_mean', 'dist_triton_std',
                    'speedup_mean', 'timestamp'
                ])
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Error initializing CSV file: {str(e)}")
            raise


def append_result_immediately(filename, results):
    """Write single result immediately to CSV with forced flushing"""
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                results['model'], results['M'], results['mode'], results['torch_mean'], results['torch_std'],
                results['dist_triton_mean'], results['dist_triton_std'], results['speedup_mean'],
                time.strftime('%Y-%m-%d %H:%M:%S')
            ])
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Critical error writing to results file: {str(e)}")
        raise


def parse_output(output, model, M_value, mode):
    """Parse command output to extract performance data"""
    torch_times = []
    dist_triton_times = []
    speedups = []

    if not output:
        return None

    lines = output.split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == 'torch':
            try:
                time = float(parts[-1])
                torch_times.append(time)
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 5 and parts[0] == 'dist-triton':
            try:
                time = float(parts[-2])
                speedup = float(parts[-1].replace('x', ''))
                dist_triton_times.append(time)
                speedups.append(speedup)
            except (ValueError, IndexError):
                continue

    if torch_times and dist_triton_times and speedups:
        torch_mean = sum(torch_times) / len(torch_times)
        torch_std = math.sqrt(sum((t - torch_mean)**2 for t in torch_times) / len(torch_times))

        triton_mean = sum(dist_triton_times) / len(dist_triton_times)
        triton_std = math.sqrt(sum((t - triton_mean)**2 for t in dist_triton_times) / len(dist_triton_times))

        speedup_mean = sum(speedups) / len(speedups)

        return {
            'model': model, 'M': M_value, 'mode': mode, 'torch_mean': round(torch_mean, 6), 'torch_std':
            round(torch_std, 6), 'dist_triton_mean': round(triton_mean, 6), 'dist_triton_std': round(triton_std, 6),
            'speedup_mean': round(speedup_mean, 6)
        }
    return None


def run_command(model, M_value, mode, retries=3):
    """Execute command with retries, show full output on failure"""
    env = os.environ.copy()
    env['NVSHMEM_DISABLE_CUDA_VMM'] = '1' if mode == 'ag_rs' else '0'
    env['NVSHMEM_SYMMETRIC_SIZE'] = '10g'

    cmd = [
        'bash', 'scripts/launch.sh', './python/triton_dist/test/nvidia/test_tp_mlp.py', '--M',
        str(M_value), '--mode', mode
    ]

    print("\n=== Executing Command ===")
    print(f"Command: {' '.join(cmd)}")
    print(
        f"Env: NVSHMEM_DISABLE_CUDA_VMM={env['NVSHMEM_DISABLE_CUDA_VMM']}, NVSHMEM_SYMMETRIC_SIZE={env['NVSHMEM_SYMMETRIC_SIZE']}"
    )
    print("========================")

    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
                                    timeout=3600)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"\n=== Command Failed (Attempt {attempt + 1}/{retries}) ===")
            print(f"Exit code: {e.returncode}")
            print("\n--- STDOUT ---")
            print(e.stdout)
            print("\n--- STDERR ---")
            print(e.stderr)
            print("--------------")
        except subprocess.TimeoutExpired:
            print(f"\n=== Command Timed Out (Attempt {attempt + 1}/{retries}) ===")
            print("Command exceeded 1-hour timeout")
        except Exception as e:
            print(f"\n=== Unexpected Error (Attempt {attempt + 1}/{retries}) ===")
            print(str(e))

        if attempt < retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)

    print(f"\n=== All {retries} Attempts Failed ===")
    return None


def main():
    args = parse_arguments()

    try:
        from tqdm import tqdm
    except ImportError:
        print("Error: tqdm library not found. Install with 'pip install tqdm'")
        return

    initialize_csv(args.output)
    completed_tasks = load_completed_tasks(args.output)

    models = ["Qwen/Qwen3-32B"]
    modes = ["ag_rs", "gemm_ar"]
    M_values = [512 * (2**i) for i in range(13)]

    print(f"M values to test: {M_values}")
    print(f"Modes to test: {modes}")

    total_tasks = len(models) * len(M_values) * len(modes)

    completed_count = sum(1 for task in completed_tasks
                          if task[0] in models and task[1] in map(str, M_values) and task[2] in modes)

    print(f"Found {completed_count} completed tasks out of {total_tasks}")

    pbar = tqdm(total=total_tasks, initial=completed_count, unit='task')

    try:
        print("\n=== Starting MLP tests ===")
        for model in models:
            for mode in modes:
                for M in M_values:
                    task_key = (model, str(M), mode)
                    if args.resume and task_key in completed_tasks:
                        tqdm.write(f"Skipping: model={model}, M={M}, mode={mode}")
                        continue

                    tqdm.write(f"\nProcessing: model={model}, M={M}, mode={mode}")
                    output = run_command(model, M, mode, args.retries)

                    if output:
                        results = parse_output(output, model, M, mode)
                        if results:
                            append_result_immediately(args.output, results)
                            tqdm.write(f"Successfully saved results for model={model}, M={M}, mode={mode}")
                            completed_count += 1
                        else:
                            tqdm.write("Failed to parse results from output")
                    else:
                        tqdm.write("No valid output received from command")

                    pbar.update(1)

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Partial results saved.")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
    finally:
        pbar.close()
        print(f"\nProcess finished. Results in {args.output}")
        print(f"Completed {completed_count}/{total_tasks} tasks")


if __name__ == "__main__":
    main()
