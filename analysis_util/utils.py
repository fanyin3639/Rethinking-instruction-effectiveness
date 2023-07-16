import os
import re
import sys
import string
import json
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric, Dataset

data_path = sys.argv[1]
split_path = sys.argv[2]

def extract_definitions(tasks_path, lines):
    files = [os.path.join(tasks_path, f) for f in os.listdir(tasks_path) if os.path.isfile(os.path.join(tasks_path, f))]
    def_dict = {}
    demos = []
    for file in files:
        if file.endswith('.json'):
            if not file[len(tasks_path) + 1:-5] in lines:
                continue
            with open(file) as f:
                data = json.load(f)
            pos_examples = []
            for idx, pos_example in enumerate(data["Positive Examples"][:2]):
                pos_example_str = f" Positive Example {idx + 1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_examples.append(pos_example_str)
            # definition = data['Definition'][0] + '\n' + ''.join(pos_examples)
            def_dict[data['Definition'][0]] = file[len(tasks_path) + 1:-5]
    return def_dict


def extract_definition_for_split():
    full_def_dict = {}
    for split in ['train', 'dev', 'test']:
        with open(os.path.join(split_path, f'{split}_tasks.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        split_dict = extract_definitions(data_path, lines)
        full_def_dict.update(split_dict)
    return full_def_dict

def predict_run(trainer, in_dataset):
    predict_results = trainer.predict(
        in_dataset, metric_key_prefix="predict"
    )
    metrics = predict_results.metrics
    overall_rougeL = -metrics['predict_loss']
    per_data_rougeL = np.array(metrics['predict_rougeL_per_data'])
    return overall_rougeL, per_data_rougeL

def flatten(phrases):
    result = []
    for i in phrases:
        if not isinstance(i, list):
            result.append(i)
        else:
            result.extend(flatten(i))

    return result

def traverse(trainer, predict_dataset, metric, tree, ori_instruct, instruct_cache={}):
    max_metric = 0.0
    track_perf = {}
    for i, subtree in enumerate(tree):
        if isinstance(subtree, list):
            remove_str = ' '.join(flatten(subtree)).strip()
        else:
            if subtree in string.punctuation:
                continue
            remove_str = subtree
        ori_instruct_removed = ori_instruct.replace(remove_str, '').strip()
        ori_instruct_removed = re.sub(' +', ' ', ori_instruct_removed)
        if ori_instruct_removed in instruct_cache:
            overall_rougeL = instruct_cache[ori_instruct_removed]
        else:
            for idx, data in enumerate(predict_dataset):
                predict_dataset[idx]['Definition'][0] = ori_instruct_removed
            df = pd.DataFrame(predict_dataset)
            dataset = Dataset.from_pandas(df)
            overall_rougeL, per_data_rougeL = predict_run(trainer, dataset)
            instruct_cache[ori_instruct_removed] = overall_rougeL
        if overall_rougeL - metric >= max_metric:
            max_metric = overall_rougeL - metric
        track_perf[i] = (remove_str, overall_rougeL - metric)
    track_perf = {k: v for k, v in sorted(track_perf.items(), key=lambda item: item[1][1], reverse=True)}
    ablate_list = []
    for subtree_idx, v in track_perf.items():
        text, perf_change = v[0], v[1]
        if perf_change < 0:
            break
        pruned_tree = [subt for i, subt in enumerate(tree) if i not in [subtree_idx] + ablate_list]
        remove_str = ' '.join(flatten(pruned_tree)).strip()
        for idx, data in enumerate(predict_dataset):
            predict_dataset[idx]['Definition'][0] = remove_str
        df = pd.DataFrame(predict_dataset)
        dataset = Dataset.from_pandas(df)
        overall_rougeL, per_data_rougeL = predict_run(trainer, dataset)
        if overall_rougeL - metric < max_metric:
            break
        ablate_list.append(subtree_idx)


    pruned_tree = []
    for i, subt in enumerate(tree):
        if i not in ablate_list:
            if not isinstance(subt, list):
                pruned_tree.append(subt)
            else:
                pruned_tree.extend(subt)

    if not any(isinstance(item, list) for item in tree):
        return pruned_tree
    else:
        compressed_instruct = ' '.join(flatten(pruned_tree))
        return traverse(trainer, predict_dataset, max_metric + metric, pruned_tree, compressed_instruct, instruct_cache=instruct_cache)


if __name__ == "__main__":
    full_def_dict = extract_definition_for_split()


