# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/safe_rlhf")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "PKU-Alignment/PKU-SafeRLHF-prompt"

    dataset = datasets.load_dataset(data_source)["train"]
    if args.max_examples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_examples))

    dataset_dict = dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = dataset_dict["train"], dataset_dict["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            prompt_source = example.pop("prompt_source")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "safety",
                "reward_model": {"style": "model", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "prompt_source": prompt_source
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.max_examples:
        train_dataset.to_parquet(os.path.join(local_dir, f"train_{args.max_examples}.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, f"test_{args.max_examples}.parquet"))
    else:
        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
