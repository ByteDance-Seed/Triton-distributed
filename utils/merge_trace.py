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
import glob
import gzip
import json
import os
import pathlib
import sys
from typing import Optional
import concurrent.futures
import chardet
import re

merge_pool = concurrent.futures.ProcessPoolExecutor(max_workers=10)
unzip_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
# install:
#    tkinterdnd2-universal : refer to https://github.com/pmgagne/tkinterdnd2/issues/7


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result["encoding"]


def clean_json_file(input_file, output_file):
    try:
        # 读取文件内容
        with open(input_file, "rb") as f:
            content = f.read()

        # 清理
        content = content.decode("utf-8", "ignore")
        pattern = re.compile(r'[^a-zA-Z0-9()\[\]{}_.,;:\-!?\'" ]')
        cleaned_content = pattern.sub("", content)

        # 解析JSON内容
        data = json.loads(cleaned_content)

        # 将清理后的内容写入新文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        print(f"Cleaned JSON file saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


def load_json_with_auto_encoding(file_path):
    print(file_path)
    clean_json_file(file_path, file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def unzip(infile, outfile):
    with gzip.open(infile, "rb") as f:
        with open(outfile, "wb") as fout:
            fout.write(f.read())


def unzip_if_not_exist(infile: str, outfile: Optional[str] = None):
    if outfile is None:
        assert infile.endswith(".gz") or infile.endswith(".tgz")
        outfile = pathlib.Path(infile).with_suffix("")
        print(f"outfile is not set, using input without .gz suffix: {outfile}")
    if pathlib.Path(outfile).is_file:
        print(f"{outfile} already exist from pathlib, skip")
        if os.path.exists(outfile):
            print(f"{outfile} already exist, skip")
            return
    unzip(infile, outfile)


def unzip_jsons(indir):
    print(f"unzip all gz files from: {indir}")
    to_merge_files = glob.glob(f"{indir}/*json.gz") + glob.glob(f"{indir}/*json.tgz")
    futures = []
    for infile in to_merge_files:
        futures.append(unzip_pool.submit(unzip_if_not_exist, infile))
        # unzip_if_not_exist(infile)
    for future in futures:
        future.result()


def merge_json(indir, output_json):
    print(f"merge json from {indir} to {output_json}")
    unzip_jsons(indir)
    events = []
    to_merge_files = glob.glob(f"{indir}/*json")

    for tl_file in to_merge_files:
        if tl_file.endswith("merged.json"):
            continue
        full_tl_json = load_json_with_auto_encoding(tl_file)
        # with open(tl_file, "r") as f:
        #     full_tl_json = json.load(f)

        rank = full_tl_json["distributedInfo"]["rank"]
        world_size = full_tl_json["distributedInfo"]["world_size"]
        for e in full_tl_json["traceEvents"]:
            e["pid"] = f"{e['pid']}_{rank}"
            if isinstance(e["tid"], int):
                e["tid"] = e["tid"] * world_size + rank
            if e["name"] == "thread_name":
                e["args"]["name"] = f'{e["args"]["name"]}_{rank}'
            if e["name"] == "thread_sort_index":
                e["args"]["sort_index"] = e["args"]["sort_index"] * world_size + rank
        events.extend(full_tl_json["traceEvents"])

    with open(output_json, "w") as f:
        full_tl_json["traceEvents"] = events
        json.dump(events, f)


def merge_json_callback(e):
    callback_data = e.data
    print(f"callback_data: {type(callback_data)} `{callback_data}`")

    jobs = []

    for indir in callback_data.split():
        indir = pathlib.Path(indir)
        if not indir.is_dir():
            print(f"{indir} is not a valid directory")
            return
        merged_jsons = list(indir.glob("*merged.json"))
        if merged_jsons:
            print(f"{merged_jsons} already exists")
            return
        output_json = indir / "merged.json"
        jobs.append(merge_pool.submit(merge_json, indir, output_json))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=None)
    args = parser.parse_args()

    if args.indir is not None:
        merge_json(args.indir, pathlib.Path(args.indir) / "merged.json")
        sys.exit(0)

    # import tkinter as tk

    # from tkinterdnd2 import DND_FILES, TkinterDnD

    # root = TkinterDnD.Tk()  # notice - use this instead of tk.Tk()
    # lb = tk.Listbox(root)
    # lb.insert(1, "drag directories including *.json/*.json.gz here")

    # # register the listbox as a drop target
    # lb.drop_target_register(DND_FILES)
    # lb.dnd_bind("<<Drop>>", merge_json_callback)

    # lb.pack()
    # root.mainloop()
