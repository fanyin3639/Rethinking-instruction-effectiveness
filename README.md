# Rethinking-instruction-effectiveness
The codebase for our ACL2023 paper: [Did You Read the Instructions? Rethinking the Effectiveness of Task Definitions in Instruction Learning](https://arxiv.org/abs/2306.01150).

## Pre-requisite 
Make sure you have CUDA 11 installed as the deepspeed version used in the codebase only supports CUDA 11. The code is tested with Python 3.8.
When you have the above environment ready,

<code>pip install -r requirements.txt</code>.

Next, download the NIv2 data from the [official website](https://github.com/allenai/natural-instructions). Put the extracted task data into <code>Rethinking-instruction-effectiveness/data</code>

## Part I: Ablation study on instruction content and metadata instruction.
To reproduce the metadata experiment, simply run

<code> bash ./scripts/train_t5.sh </code>

The label space for each task we extract is in <code>Rethinking-instruction-effectiveness/analysis_util/label_space.json</code>.

To reproduce the results of ablation study, follow the description below. Notice that the main body of this codebase is adapted from [Tk-Instruct](https://github.com/yizhongw/Tk-Instruct).

We have put our annotation on task definitions in the *content_type_annotation.json* file under the folder <code>Rethinking-instruction-effectiveness/analysis_util</code>.

The annotation is a JSON file where *the keys* are task definitions, *the values* are sentence-level annotations on content types. Task ID and name is the last element in *the value*s.

For example, the entry below means the sentence contains Label description and also belongs to the Output content category. We have eight categories in total: Label description, Label list, Input content, Input mention, Action content, Output content, Additional input details and Additional output details. See the paper for their concrete definitions.
```json
{
    [
      "0 : The two sentences are completely dissimilar .",
      "Segment,Label description,Output content"
    ],

}
```

To do ablation, change the --do_ablation to True and --use_meta_data to False. Then, run

<code> bash ./scripts/train_t5.sh </code>

You may change your own ablation type by setting --ablate_content to your own choice.

## Part II: STDC, an automatic instruction compression algorithm.
To reproduce our instruction compression results, you first want to install supar==1.1.1, which provides a constituency parser that we will use later to obtain parse tree.

Then, navigate to <code>analysis_utils/</code>. Run <code>python parse_tree.py</code>. You can indicate your output path and task name to get parse tree for a specific task. Currently, it is set to task1670_md_gender_bias_text_modification as an example.

Now, you are ready for the compression. Set the path in  <code> ./scripts/run_reduction.sh</code>, make sure you have the right task name and parsed tree. Notice you will also need a instruction tuned model for this set of experiments.

Run <code> bash ./scripts/run_reduction.sh </code>.

The compressed instruction and the coverage rate is written to the output directory.
