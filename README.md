
# Self-Play Fine-Tuning for LLM Agent

![Mistral-7B](https://img.shields.io/badge/Model-Mistral--7B--v0.1-green) ![Open LLM](https://img.shields.io/badge/Task-Open_LLM_Leaderboard-red) ![MT-Bench](https://img.shields.io/badge/Task-MT--Bench-red)



## 🌀 About SPIN
**SPIN** utilizes a self-play mechanism, allowing an LLM to improve itself by playing against its previous iterations, without needing additional human-annotated preference data than the SFT dataset itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from the original SFT data. 

<p align="center">
    <img src="images/iter_openllm.png" width="35%"> <br>
  Average score of <b>SPIN</b> at different iterations on the HuggingFace Open LLM leaderboard. 
</p>
SPIN can significantly enhance the performance of an LLM after SFT across various benchmarks, outperforming the model trained with direct preference optimization (DPO) on labelled preference datasets. The approach is theoretically grounded, ensuring that the LLM aligns with the target data distribution, and empirically validated through extensive evaluations on multiple datasets. 
<p align="center">
    <img src="images/dpo_compare.png" width="80%"> <br>
  Performance comparison with DPO training across the six benchmark datasets. SPIN at iteration 0 achieves comparable performance to DPO training with 62k new data. At iteration 1, SPIN has already surpassed DPO training on the majority of datasets. 
</p>

For more details, you can check our paper [here](https://arxiv.org/abs/2401.01335).

## Setup
The following steps provide the necessary setup to run our codes.
1. Create a Python virtual environment with Conda:
```shell
conda create -n spina python=3.10
conda activate spina
```
2. Install the following Python dependencies to run the codes.

```
python -m pip install .
python -m pip install flash-attn --no-build-isolation
```
3. Login to your huggingface account for downloading models

```
huggingface-cli login --token "${your_access_token}"
```

## Data

- Define tools in `spin/tools/`
- Define tasks in `spin/tasks/`
- Collect data & run experiments via `spin/generation_fireact.py`
- Results will be saved in `trajs/`

## Usage
For SPIN, we generate all synthetic data at once for an iteration, and fine-tune the LLM based on the real and synthetic data pairs. 

### Step 0 (optional): Reformatting SFT dataset
```
python spin/reformat.py [options]
```
Options
- `--data`: directory to the SFT dataset (local or huggingface)
    - default: `HuggingFaceH4/ultrachat_200k`
- `--output_dir`: local directory to the reformated data files 
    - default: `UCLA-AGI/SPIN_iter0`

🔍 Note: If choosing to use SPIN on the entire dataset of `HuggingFaceH4/ultrachat_200k` instead of our 50k subset, one can reformat the original data with `spin/reformat.py`. To use other datasets, simply convert the data into the same format and resume with the following steps. 

### Step 1: Fine-tuning

#### Setup 

Set up SERP API key and store in environment variable (see [here](https://serpapi.com/))

```shell
export SERPAPI_API_KEY=<YOUR_KEY>
```

__Example__.

```
bash scripts/finetune.sh
```

### Step 2: Generation

Begin to generate data

```shell
mkdir trajs # create the path for saving generated COT data
bash scripts/generate.sh
```

Options

- `--modelpath`: load the base model checkpoint for generation.
  - default: `alignment-handbook/zephyr-7b-sft-full`
- `--peftpath`: the peft adapter model path
  - default: `outputs/iter0-ckpt`

The generated data is in json format where each data contains the following attributes:

```
{
  "3687": {
    "reward": false,
    "em": false,
    "f1": 0,
    "gt": "beyond clouds",
    "pred": "fugitive",
    "prompt": "",
    "traj": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat movie did actress Irene Jacob complete before the American action crime thriller film directed by Stuart Bird?\n\n### Response:\nThought: I need to search for the movie Irene Jacob completed before the American action crime thriller film directed by Stuart Bird.\nAction: search[Irene Jacob movie before American action crime thriller film directed by Stuart Bird]\nObservation: The fugitive is only helped by his sweetheart ( Irene Jacob ). The picture is the following to \u00a8The fugitive\u00a8 ( by Andrew Davis ) that's an adaptation based on ...\nThought: The movie Irene Jacob completed before the American action crime thriller film directed by Stuart Bird is The Fugitive.\nAction: finish[The Fugitive]\nObservation: Episode finished, reward = False\n",
    "traj_by_line": [
      "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
      "",
      "### Instruction:",
      "What movie did actress Irene Jacob complete before the American action crime thriller film directed by Stuart Bird?",
      "",
      "### Response:",
      "Thought: I need to search for the movie Irene Jacob completed before the American action crime thriller film directed by Stuart Bird.",
      "Action: search[Irene Jacob movie before American action crime thriller film directed by Stuart Bird]",
      "Observation: The fugitive is only helped by his sweetheart ( Irene Jacob ). The picture is the following to \u00a8The fugitive\u00a8 ( by Andrew Davis ) that's an adaptation based on ...",
      "Thought: The movie Irene Jacob completed before the American action crime thriller film directed by Stuart Bird is The Fugitive.",
      "Action: finish[The Fugitive]",
      "Observation: Episode finished, reward = False",
      ""
    ]
}
```

**Convert the data into alpaca format**

Transfer the generated <json> file into alpaca format for subsequent fine-tuning.

```shell
python spin/convert_alpaca.py 
```

#### 🚀 Faster generation with vLLM

Alternatively, you could use the following example script to generate LLM responses with speedup. Larger `frac_len` can be used with vllm.

```
bash scripts/generate_vllm.sh
```

Thanks to @sumo43 for implementing vLLM for generation. 

## To be updated...

## Reproducing Our Results

To help reproducing our results, we have made available the scripts corresponding to all four iterations of our study. These scripts are pre-configured with the exact parameters and model versions used in our paper. For each iteration, the base model is initialized with the version released on 🤗 HuggingFace, which can be found at the following links:

| Dataset                    |                           Download                           |
| :----------------------- | :----------------------------------------------------------: |
| SPIN_iter0     | 🤗 [HuggingFace](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter0) |
| SPIN_iter1 | 🤗 [HuggingFace](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter1) |
| SPIN_iter2      |   🤗 [HuggingFace](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter2) |
| SPIN_iter3      |   🤗 [HuggingFace](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter3) |

To execute the full pipeline using your locally trained models as the base, modify the `model_name_or_path` parameter in the configuration files to point to your model's path.

To start the full fine-tuning process, run the corresponding script from your terminal:

```bash
bash scripts/finetune.sh
bash scripts/finetune_iter1.sh
bash scripts/finetune_iter2.sh
bash scripts/finetune_iter3.sh
```

By following these steps, you should be able to reproduce our results.

---

## Evaluation
For our evaluation on the Open LLM Leaderboard, please use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository at v0.4.0. Also, note that we set the number of few shot examples to be the same as instructed on the [Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). Different evaluation versions results in different scores, but the trend will remain the same.

## Acknowledgement
This repo is built upon [Self-Play Fine-Tuning (SPIN)](https://github.com/uclaml/SPIN) and [FireAct](https://github.com/anchen1011/FireAct). We thank the authors for their great work.
