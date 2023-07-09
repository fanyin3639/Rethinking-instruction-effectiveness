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

To reproduce the results of ablation study and metadata performance, follow the description below. Notice that the main body of this codebase is adapted from [Tk-Instruct](https://github.com/yizhongw/Tk-Instruct).

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
More code coming soon.
