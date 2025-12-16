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

import torch
import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prof_dir", type=str, default="prof")
    parser.add_argument("--M-range", "--M_range", type=str, default="128-1024-128")
    parser.add_argument("--N-range", "--N_range", type=str)
    parser.add_argument("--K-range", "--K_range", type=str)
    parser.add_argument("-M", "--M", type=int)
    parser.add_argument("-N", "--N", type=int)
    parser.add_argument("-K", "--K", type=int)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--trans_b", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


class IntFilter:
    """
    A class to filter integers based on a defined rule.
    The rule can be an integer, a [min, max] list for a range, or None for a wildcard.
    """

    def __init__(self, rule):
        """
        Initializes the filter with a specific rule.

        Args:
            rule: Can be None (matches all), an int (exact match),
                  or a list of two ints [min, max] (range match).

        Raises:
            TypeError: If the rule is not an int, list, or None.
            ValueError: If the list rule is not properly formatted.
        """
        if rule is None or isinstance(rule, int):
            self.rule = rule
        elif isinstance(rule, list):
            if len(rule) != 2 or not all(isinstance(i, int) for i in rule):
                raise ValueError("List rule must be a [min, max] pair of two integers.")
            if rule[0] > rule[1]:
                raise ValueError("In a range [min, max], min cannot be greater than max.")
            self.rule = rule
        else:
            raise TypeError("Rule must be an int, a list of two ints, or None.")

    def match(self, val: int) -> bool:
        """
        Checks if a given integer value matches the filter's rule.

        Args:
            val: The integer value to check.

        Returns:
            True if the value matches, False otherwise.
        """
        # Rule is None: This is a wildcard and matches any integer.
        if self.rule is None:
            return True

        # Rule is an int: Check for an exact match.
        if isinstance(self.rule, int):
            return val == self.rule

        # Rule is a list: Check if the value is within the range [min, max].
        if isinstance(self.rule, list):
            min_val, max_val = self.rule
            return min_val <= val <= max_val

        return False  # Should not be reached due to __init__ validation

    def __repr__(self):
        """Provides a clear string representation of the filter object."""
        return f"IntFilter(rule={self.rule})"

    def is_int(self):
        return isinstance(self.rule, int)


def parse_range(range_str):
    start, end, step = range_str.split("-")
    start, end, step = int(start), int(end), int(step)
    assert start < end
    assert step > 0
    return start, end, step


def parse_int_range_args(value, value_range):
    if value:
        return IntFilter(value)
    v_start, v_end, _ = parse_range(value_range)
    return IntFilter([v_start, v_end])


def _match_col(filename):
    import re
    pattern = r"N_\d+_K_\d+"
    # Search for the pattern in the string
    match = re.search(pattern, filename)

    if match:
        result = match.group()
        return (result)  # Output: N_4096_K_7168
    else:
        print("Pattern not found")
        return filename


TORCH_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
}
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _parse_col(col_name):
    # '(256, 8192, 7392, torch.float16)'
    col_name = col_name.replace("(", "").replace(")", "")
    M, N, K, dtype, trans_b = col_name.split(", ")
    return int(M), int(N), int(K), TORCH_DTYPE_MAP[dtype], bool(trans_b)


def find_best_topk(df, topk=1, policy="greedy"):
    df = df.loc[df.min(axis=1) <= 1.15]
    rows = df.index.tolist()
    rows_select = []
    logging.info(f"Find best topk {topk}")
    for _ in range(topk):
        best = [df.loc[rows_select + [row]].to_numpy().min(axis=0).mean() for row in rows]
        x = rows[np.array(best).argmin()]
        rows.remove(x)
        rows_select.append(x)

    x = df.loc[rows_select].min(axis=0)
    return rows_select, (x.mean(), x.quantile(0.9), x.quantile(0.99), x.quantile(0.999), x.max())


def find_best_topk_fast(df: pd.DataFrame, topk: int = 1, threshold: float | None = 1.15,  # row filter on row-min
                        objective: str = "mean",  # "mean" or "minimax" over column-wise minima
                        allow_na: bool = False,  # if False, drop rows with NaNs
                        random_tiebreak: bool = False,  # tie-breaking strategy
                        ):
    """
    Greedy: pick `topk` rows to minimize a score of the column-wise minima over chosen rows.
    Score options:
      - "mean":   minimize mean(col_mins)
      - "minimax": minimize max(col_mins)
    Returns (rows_selected, (mean, p90, p99, p999, max)) computed on the final col_mins.
    """
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise ValueError("df must be a non-empty DataFrame")

    work = df.copy()

    if not allow_na:
        work = work.dropna(axis=0, how="any")
    if work.empty:
        raise ValueError("No rows remain after dropping NaNs")

    if threshold is not None:
        mask = work.min(axis=1) <= threshold
        work = work.loc[mask]
        if work.empty:
            raise ValueError("No rows pass the threshold filter")

    # NumPy arrays for speed
    rows = work.index.to_list()
    X = work.to_numpy(dtype=float)  # shape: (n_rows, n_cols)
    n, c = X.shape

    if topk > n:
        logging.warning("topk > available rows; clipping")
        topk = n

    # current column-wise minima; start with +inf so first pick becomes that row
    cur_min = np.full(c, np.inf, dtype=float)
    chosen_idx = []  # indices in X
    remaining = np.arange(n)

    def score(col_mins: np.ndarray) -> float:
        if objective == "mean":
            return float(np.mean(col_mins))
        elif objective == "minimax":
            return float(np.max(col_mins))
        else:
            raise ValueError(f"unknown objective: {objective}")

    for _ in range(topk):
        # Broadcasted trial minima for ALL remaining candidates at once
        trial_mins = np.minimum(cur_min, X[remaining])  # (m, c)

        if objective == "mean":
            scores = trial_mins.mean(axis=1)
        else:  # minimax
            scores = trial_mins.max(axis=1)

        best_pos = np.argmin(scores)
        # Optional stable tie-break
        if random_tiebreak:
            ties = np.where(scores == scores[best_pos])[0]
            best_pos = np.random.choice(ties)

        pick = remaining[best_pos]
        chosen_idx.append(pick)
        cur_min = np.minimum(cur_min, X[pick])
        # remove picked from remaining
        remaining = np.delete(remaining, best_pos)

    # Final metrics
    col_mins = pd.Series(cur_min, index=work.columns)
    # Use numpy quantiles to avoid dtype surprises
    p90 = float(np.quantile(cur_min, 0.90))
    p99 = float(np.quantile(cur_min, 0.99))
    p999 = float(np.quantile(cur_min, 0.999))
    out = (float(col_mins.mean()), p90, p99, p999, float(col_mins.max()))

    # Map back to original df indices
    selected_rows = [rows[i] for i in chosen_idx]
    return selected_rows, out


if __name__ == "__main__":
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    M_range = parse_int_range_args(args.M, args.M_range)

    autotune_rel_files = glob.glob(f"{args.prof_dir}/**/autotune_*_rel.csv",
                                   recursive=True) + glob.glob(f"{args.prof_dir}/autotune_*_rel.csv")
    logging.info(f"{len(autotune_rel_files)} autotune files found")

    df_full = []
    df_mean_full = []
    configs_full = set()
    topks = {topk: [] for topk in range(1, 6)}
    for idx, autotune_rel_file in enumerate(autotune_rel_files):
        df = pd.read_csv(autotune_rel_file, index_col=0)
        col_name = _match_col(Path(autotune_rel_file).stem)

        mean_df = df[["mean"]].rename(columns={"mean": col_name})
        df_mean_full.append(mean_df)
        del df["mean"]
        df_full.append(df)

        df = df.rename(columns=_parse_col)
        col_filter = lambda M, N, K, dtype_, trans_b_: M_range.match(M) and dtype == dtype_ and trans_b_ == args.trans_b
        columns = [col for col in df.columns if col_filter(*col)]
        df = df[columns]

        configs = set(df.idxmin().tolist())
        configs_full.update(configs)
        n = len(configs)
        logging.info(f"{col_name} {n} configs found")

        for topk in range(1, min(n, 6)):
            configs, stats = find_best_topk_fast(df, topk=topk)
            mean_value, p90_value, p99_value, p999_value, max_value = stats
            print(
                f"{col_name} topk {topk} mean {mean_value:0.3f} p90 {p90_value:0.3f} p99 {p99_value:0.3f} p999 {p999_value:0.3f} max {max_value:0.3f}"
            )
            topks[topk].append(sorted(configs))

    print(f"total {len(configs_full)} configs need to cover all best configs for all cases")

    # Loop through each DataFrame in df_mean_full and check for duplicates
    for i, df in enumerate(df_mean_full):
        if df.index.duplicated().any():
            print(f"DataFrame at index {i} has duplicate indexes: from {autotune_rel_files[i]}")
            print(df.index[df.index.duplicated()])  # Show duplicate values

    df = pd.concat(df_mean_full, axis=1, ignore_index=False, join="outer")
    df.to_csv(f"{args.prof_dir}/autotune_rel.csv")

    df_full = pd.concat(df_full, axis=1, ignore_index=False, join="outer")
    print("try using topk")
    for topk in range(1, 11):
        configs, stats = find_best_topk_fast(df_full, topk=topk)
        mean_value, p90_value, p99_value, p999_value, max_value = stats
        print(
            f"topk {topk} mean {mean_value:0.3f} p90 {p90_value:0.3f} p99 {p99_value:0.3f} p999 {p999_value:0.3f} max {max_value:0.3f}"
        )
    for c in configs:
        print(c)
