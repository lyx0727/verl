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
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/safe_rlhf")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--wo_safety", action="store_true")
    parser.add_argument("--wo_test_jailbreak", action="store_true")
    parser.add_argument("--wo_test_xstest", action="store_true")
    parser.add_argument("--w_jailbreak", action="store_true")
    parser.add_argument("--w_math_jailbreak", action="store_true")

    args = parser.parse_args()

    safe_data_source = "PKU-Alignment/PKU-SafeRLHF-prompt"

    math_data_source = "openai/gsm8k"
    code_data_source = "ganler/code-r1-12k"

    columes_to_keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    safe_dataset = datasets.load_dataset(safe_data_source)["train"]
    math_dataset = datasets.load_dataset(math_data_source, "main")["train"]
    code_dataset = datasets.load_dataset(code_data_source)["train"]

    def safe_dataset_mapper(item, idx):
        prompt = item['prompt']
        return {
            "data_source": "PKU-Alignment/PKU-SafeRLHF-prompt",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "safety",
            "reward_model": {"style": "model", "ground_truth": ""},
            "extra_info": {
                "split": "train",
                "index": idx,
                "query": prompt,
            }
        }

    def math_dataset_mapper(item, idx):
        question_raw = item['question']
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
        question = question_raw + " " + instruction_following
        answer_raw = item['answer']
        solution = extract_solution(answer_raw)
        return {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "train",
                "index": idx,
                "query": item['question'],
            }
        }
    
    def code_dataset_mapper(item, idx):
        return {
            k: item[k]  for k in columes_to_keep if k != 'extra_info'
        } | {
            'extra_info': {
                'split': 'train',
                'index': idx,
                'query': item['prompt'][-1]['content'],
            }
        }

    safe_dataset = safe_dataset.map(safe_dataset_mapper, with_indices=True).select_columns(columes_to_keep)
    math_dataset = math_dataset.map(math_dataset_mapper, with_indices=True).select_columns(columes_to_keep)
    code_dataset = code_dataset.map(code_dataset_mapper, with_indices=True).select_columns(columes_to_keep)

    if args.max_examples:
        safe_dataset = safe_dataset.shuffle(seed=args.seed).select(range(args.max_examples))
        # 安全数学代码取 2:1:1 暂时
        math_dataset = math_dataset.shuffle(seed=args.seed).select(range(args.max_examples // 2))
        code_dataset = code_dataset.shuffle(seed=args.seed).select(range(args.max_examples // 2))
    else:
        print("Not supported for now")
        exit(1)

    print("safe", safe_dataset)
    print("math", math_dataset)
    print("code", code_dataset)

    if args.w_jailbreak or args.w_math_jailbreak:
        import pandas as pd
        jailbreak_templates = []
        for dataset in ['SGB_art', 'SGB_autodan']:
            data = pd.read_json(f'data/safety/{dataset}.json')
            templates = data['jailbreak_prompt'].unique().tolist()
            jailbreak_templates += [
                {
                    'template': template,
                    'source': f'{dataset}_{index}',
                }
                for index, template in enumerate(templates)
            ]
        print("use jailbreak_templates", len(jailbreak_templates))

        def jailbreak(dataset, jailbreak_templates):
            jailbroken_dataset = []
            for example in dataset:
                prompt = example.pop('prompt')
                data_source = example.pop('data_source')
                for template in jailbreak_templates:
                    jailbroken_dataset.append(example | {
                        'prompt': [{ 'role': 'user', 'content': template['template'].format(query=prompt[0]['content']) }],
                        'data_source': data_source + '_jailbroken_' + template['source']
                    })
            jailbroken_dataset = datasets.Dataset.from_list(jailbroken_dataset)
            return jailbroken_dataset

        if args.w_jailbreak:
            jailbroken_safe_dataset = jailbreak(safe_dataset, jailbreak_templates)
            # TODO: 2:1 混合
            jailbroken_safe_dataset = jailbroken_safe_dataset.shuffle(seed=args.seed).select(range(len(safe_dataset) // 2))
            safe_dataset = datasets.concatenate_datasets([safe_dataset, jailbroken_safe_dataset])

        if args.w_math_jailbreak:
            jailbroken_math_dataset = jailbreak(math_dataset, jailbreak_templates)
            # TODO: 2:1 混合
            jailbroken_math_dataset = jailbroken_math_dataset.shuffle(seed=args.seed).select(range(len(math_dataset)))
            math_dataset = datasets.concatenate_datasets([math_dataset, jailbroken_math_dataset])

    dataset = datasets.concatenate_datasets([safe_dataset, math_dataset, code_dataset])
    dataset = dataset.shuffle(seed=args.seed)

    # 把安全去掉
    if args.wo_safety:
        dataset = dataset.filter(lambda x: x['ability'] != 'safety')

    dataset_dict = dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = dataset_dict["train"], dataset_dict["test"]

    # TEST
    # todo: 在 test_dataset 加入越狱攻击的数据
    if not args.wo_test_jailbreak:
        SAFE_TEST_DATASETS = [
            # 'SGB_art',
            # 'SGB_autodan',
            'SGB_pair',
            'WildJailbreak',
            # 'ALERT',
            # 'SaladBench',
        ]
        def get_jailbreak_query(item):
            if item.get('jailbreak_prompt'):
                if item.get('reference_responses'):
                    return item['jailbreak_prompt'].format(query=item['query'], reference_responses=item['reference_responses'][0])
                else:
                    return item['jailbreak_prompt'].format(query=item['query'])
            return item['query']
        for dataset in SAFE_TEST_DATASETS:
            with open(f'data/safety/{dataset}.json') as f:
                safe_test_dataset = json.load(f)
            safe_test_dataset = [
                {
                    "data_source": dataset.upper() if dataset == 'SGB_pair' else dataset,
                    "prompt": [
                        {
                            "role": "user",
                            "content": get_jailbreak_query(item),
                        }
                    ],
                    "ability": "safety",
                    "reward_model": {"style": "model", "ground_truth": ""},
                    "extra_info": {
                        "split": "test",
                        "index": idx,
                        "query": item['query'],
                    }
                }
                for idx, item in enumerate(safe_test_dataset)
            ]
            test_dataset = datasets.concatenate_datasets([
                test_dataset, 
                datasets.Dataset.from_list(safe_test_dataset).select_columns(columes_to_keep).shuffle(seed=args.seed).select(range(len(safe_test_dataset) // 10))
            ])

    if not args.wo_test_xstest:
        # 过度拒绝测试
        with open(f'data/safety/XSTest.json') as f:
            xstest_dataset = json.load(f)
        xstest_dataset = [
            {
                "data_source": 'XSTest',
                "prompt": [
                    {
                        "role": "user",
                        "content": item['query'],
                    }
                ],
                "ability": "overrefusal",
                "reward_model": {"style": "model", "ground_truth": ""},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "query": item['query'],
                }
            }
            for idx, item in enumerate(xstest_dataset)
        ]
        test_dataset = datasets.concatenate_datasets([
            test_dataset, 
            datasets.Dataset.from_list(xstest_dataset).select_columns(columes_to_keep).shuffle(seed=args.seed).select(range(len(xstest_dataset) // 10))
        ])

    print("test_dataset", len(test_dataset))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.wo_safety:
        suffix = '_wo_safety'
    else:
        suffix = ''
        if args.w_jailbreak:
            suffix += '_w_jailbreak'
        if args.w_math_jailbreak:
            suffix += '_w_math_jailbreak'

    if args.wo_test_jailbreak:
        suffix += '_wo_test_jailbreak'
    if args.wo_test_xstest:
        suffix += '_wo_test_xstest'

    if args.max_examples:
        train_dataset.to_parquet(os.path.join(local_dir, f"train_{args.max_examples}{suffix}.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, f"test_{args.max_examples}{suffix}.parquet"))
    else:
        train_dataset.to_parquet(os.path.join(local_dir, f"train{suffix}.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, f"test{suffix}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
