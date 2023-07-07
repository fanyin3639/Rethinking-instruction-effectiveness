import os
import sys
import string
import json

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


if __name__ == "__main__":
    full_def_dict = extract_definition_for_split()


