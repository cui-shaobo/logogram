[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](LICENSE)

# <img src="./image/logoemoji.png" width="116.4" height="48"/> (LOgogram)
We introduce <img src="./image/logoemoji.png" width="58.2" height="24"/> (LOgogram), a novel heading-generation benchmark comprising 6,653 paper abstracts with corresponding *descriptions* and *acronyms* as *headings*.

To measure the generation quality, we propose a set of evaluation metrics from three aspects: summarization, neology, and algorithm.

Additionally, we explore three strategies (generation ordering, tokenization, and framework design) under prelavent learning paradigms (supervised fine-tuning, reinforcement learning, and in-context learning with Large Language Models).

## Environment Setup

We recommend you to create a new [conda](https://docs.conda.io/en/latest/) virtual environment for running codes in the repository:

```shell
conda create -n logogram python=3.8
conda activate logogram
```
Then install [PyTorch](https://pytorch.org/) 1.13.1. For example install with pip and CUDA 11.6:

```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Finally, install the remaining packages using pip:

```shell
pip install -r requirements.txt
```

## 1. Dataset Processing

### 1.1 Collection of Paper whose Heading Contains Acronyms

We crawl the [ACL Anthology](https://aclanthology.org/) and then exclude examples whose headings do not contain acronyms.

The unfiltered dataset is saved in `/raw-data/acl-anthology/data_acl_all.jsonl`.

### 1.2 Apply Filtering Rules and Replace Acronyms in Abstracts with Masks

We further applied a set of tailored filtering rules based on data inspection to eliminate anomalies. Acronyms in the abstracts were replaced with a mask to prevent acronym leakage. The details are in `src/data_processing.ipynb`.

### 1.3 Dataset Statistics

We plot the distributions with regard to the text length and the publication number of our dataset in Figure 3 and 4 in our paper. To reproduce, see `src/data_statistics.ipynb`.

## 2. Justification of Metrics

We evaluate the generated headings from the summarization, neologistic, and algorithmic constraints. Specifically, we propose three novel metrics, *WordLikeness* (WL), *WordOverlap* (WO), and *LCSRatio* (LR) from the neologistic and algorithmic aspects. To justify our metrics, we also plot the density estimation of different metrics and their joint distribution in Figure 5 and 6, demonstrating that the gold-standard examples achieve high value in these metrics. To reproduce, see `src/data_statistics.ipynb`.

## 3. Apply Strategies under Learning Paradigms

### 3.1 Supervised Fine-Tuning (SFT) Paradigm

We fine-tune the [T5](https://arxiv.org/abs/1910.10683) model and explore the effectiveness of the generation ordering, tokenization, and framework design strategies.

1. To fine-tune and inference <img src="./image/da_tok_one.png" weight="51.8" height="14.7"> (description then acronym, acronym subword-level tokenization, onestop framework), run:

```shell
accelerate launch t5_brute_finetune.py --model_name t5-base --model_mode abstract2description:shorthand --model_save_path ./models/t5-a2ds-token-base --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2ds-token-base/checkpoint-5 --model_mode abstract2description:shorthand --prediction_save_path ./prediction/brute_t5_a2ds_token_predictions.csv
```

2. To fine-tune and inference <img src="./image/ad_tok_one.png" weight="51.8" height="14.7"> (acronym then description, acronym subword-level tokenization, onestop framework), run:

```shell
accelerate launch t5_brute_finetune.py --model_name t5-base --model_mode abstract2shorthand:description --model_save_path ./models/t5-a2sd-token-base --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2sd-token-base/checkpoint-5 --model_mode abstract2shorthand:description --prediction_save_path ./prediction/brute_t5_a2sd_token_predictions.csv
```

3. To fine-tune and inference <img src="./image/da_chr_one.png" weight="51.8" height="14.7"> (description then acronym, acronym letter-level tokenization, onestop framework), run:

```shell
accelerate launch t5_brute_finetune.py --model_name t5-base --model_mode abstract2description:shorthand --shorthand_mode character --model_save_path ./models/t5-a2ds-char-base --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2ds-char-base/checkpoint-5 --model_mode abstract2description:shorthand --shorthand_mode character --prediction_save_path ./prediction/brute_t5_a2ds_char_predictions.csv
```

4. To fine-tune and inference <img src="./image/ad_tok_pip.png" weight="51.8" height="14.7"> (description then acronym, acronym subword-level tokenization, pipeline framework), run:

```shell
accelerate launch t5_brute_finetune.py --model_name t5-base --model_mode abstract2description --model_save_path ./models/t5-a2ds-token-pipe/1 --save_total_limit 1

accelerate launch t5_brute_finetune.py --model_name t5-base --model_mode abstract-description2shorthand --model_save_path ./models/t5-a2ds-token-pipe/2 --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2ds-token-pipe/1/checkpoint-5 --model_mode abstract2description --prediction_save_path ./prediction/brute_t5_a2ds_token_pipe_predictions.csv

accelerate launch t5_brute_inference.py --model_name models/t5-a2ds-token-pipe/2/checkpoint-5 --model_mode abstract-description2shorthand --prediction_save_path ./prediction/brute_t5_a2ds_token_pipe_predictions.csv
```

### 3.2 Reinforcement Learning (RL) Paradigm

The RL paradigm is built upon the foundation of the SFT paradigm. Specifically, we choose the Proximal Policy Optimization (PPO) algorithm. We evaluate all strategies with the exception of <img src="./image/ad_tok_pip.png" weight="51.8" height="14.7"> due to the relatively unexplored territory of feedback mechanisms within the RL paradigm for pipeline language models.

1. To further fine-tune and inference <img src="./image/da_tok_one.png" weight="51.8" height="14.7">, run:

```shell
TOKENIZERS_PARALLELISM=false accelerate launch t5_ppo_finetune.py --model_mode abstract2description:shorthand --model_save_path ./models/t5-a2ds-token-ppo --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2ds-token-ppo --model_mode abstract2description:shorthand --prediction_save_path ./prediction/brute_t5_a2ds_token_ppo_predictions.csv
```

2. To further fine-tune and inference <img src="./image/ad_tok_one.png" weight="51.8" height="14.7">, run:

```shell
TOKENIZERS_PARALLELISM=false accelerate launch t5_ppo_finetune.py --model_mode abstract2shorthand:description --model_save_path ./models/t5-a2sd-token-ppo --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2sd-token-ppo --model_mode abstract2shorthand:description --prediction_save_path ./prediction/brute_t5_a2sd_token_ppo_predictions.csv
```

3. To further fine-tune and inference <img src="./image/da_chr_one.png" weight="51.8" height="14.7">, run:

```shell
TOKENIZERS_PARALLELISM=false accelerate launch t5_ppo_finetune.py --model_mode abstract2description:shorthand --shorthand_mode character --model_save_path ./models/t5-a2ds-char-ppo --save_total_limit 1

accelerate launch t5_brute_inference.py --model_name models/t5-a2ds-char-ppo --model_mode abstract2description:shorthand --shorthand_mode character --prediction_save_path ./prediction/brute_t5_a2ds_char_ppo_predictions.csv
```

### 3.3 In-Context Learning with Large Language Models (ICL) Paradigm
To replicate the results of ICL, run the following code
```shell
python icl_main.py
```
The generation model can be selected from 
+ `onestop` <img src="./image/da_tok_one.png" weight="51.8" height="14.7">
+ `onestop_sd`:  <img src="./image/ad_tok_one.png" weight="51.8" height="14.7">
+ `onestop_char`:  <img src="./image/da_chr_one.png" weight="51.8" height="14.7">
+ `pipeline`:  <img src="./image/ad_tok_pip.png" weight="51.8" height="14.7">


## 4. Evaluation

To evaluate the generated acronyms, run:

```shell
python run_eval.py \
    --file <CSV file> \
    --eval_type shorthand \
    --hypos-col <the column name of generated acronyms> \
    --refs-col <the column name of ground truth acronyms>
```

For descriptions, run:

```shell
python run_eval.py \
    --file <CSV file> \
    --eval_type description \
    --hypos-col <the column name of generated descriptions> \
    --refs-col <the column name of ground truth descriptions>
```

By default, the CSV files are saved in `prediction/`.
