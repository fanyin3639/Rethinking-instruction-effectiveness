import os
import sys
import json
import string
import random
import nltk
import itertools
import argparse
import numpy as np


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import supar
from supar import Parser
print(supar.NAME)
const_parser = Parser.load('crf-con-roberta-en')

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count == total_count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree:
            continue
        if tree.label() == '_':
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree):
            leaves.append(detokenize(tree.leaves()))
        else:
            leaves.append(collect_leaves(tree))
    return leaves


def get_phrases(instruction):
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = const_parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    # phrases = flatten(phrases)
    return phrases



def main(args):
    with open(os.path.join(args. output_path, f'{args.task_name}_parsed.json'), 'w') as fout:
        file = os.path.join(args.tasks_path, f'{args.task_name}.json')
        with open(file) as f:
            data = json.load(f)
        definition = data['Definition'][0]
        phrases = get_phrases(definition)
        json.dump({f'{args.task_name}': phrases}, fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks_path",
        type=str,
        default='/local2/fanyin/Rethinking-instruction-effectiveness/data/tasks',
        help="The directory where you store the task data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='/local2/fanyin/Rethinking-instruction-effectiveness/analysis_util/parsed_instructions',
        help="The directory where you want to store the parsed sentence"
    )
    parser.add_argument(
        "--task_name",
        default='task1670_md_gender_bias_text_modification',
        type=str,
        help="The task name whose instruction is to be parsed"
    )
    args = parser.parse_args()

    main(args)
