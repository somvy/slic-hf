Reproducing results of [paper](https://arxiv.org/abs/2309.16240) - "Beyond Reverse KL: Generalizing
Direct Preference Optimization with Diverse Divergence Constraints"

The paper compares different divergence functions for direct preference optimization (DPO).

Results notebook on
nbviewer - [results.ipynb](https://nbviewer.org/github/somvy/slic-hf/blob/main/results.ipynb)

## Setup

1. Install [poetry](https://python-poetry.org/docs/)
2. Then run:

```
git clone https://github.com/somvy/slic-hf && cd slic-hf
poetry install && poetry shell
wandb login
huggingface-cli login
```
3. Specify your HuggingFace username, desired SFT model in config.py


## Dataset
Prompts - first sentences from movie reviews.
Used some hacks to generate answers with positive bias (see dataset/generation_config.py)
Used diverse beam search decoding with diversity penalty 50 to generate 6 answers per prompt.
Then scored them with reward model. Used pairs of (top1, top4\5\6) 
and (top1\2\3, top6) as chosen and rejected answers (total 6 pairs from generation).
Final dataset - 3600 pairs, test size 0.2.

[hf link](https://huggingface.co/datasets/therem/dpo_dataset)

Also randomly selected 50 prompts for eval
generation - [hf link](https://huggingface.co/datasets/therem/dpo_dataset_eval)

Use this dataset, or generate your own by

```
set -a && source .env && poetry run python dataset/main.py
```

after generation change datasets paths in config.py

## Train

1. Specify training arguments, DPOTrainer params and run_name in train_dpo/train.py
2. Run

```
set -a && source .env && poetry run python train_dpo/train.py
```

3. (Optional) Generate answers from eval dataset. Specify generation params and desired run_name in
   train_dpo/generate.py

```
set -a && source .env && poetry run python train_dpo/generate.py
```

## Experiments setup

Trained GPT2 finetuned on IMDB reviews.  
3 epochs, batch size 4, lr 1e-4 for sigmoid and hinge, 1e-5 for others.


## Weights and logs

|                        Loss |                             Weights                             |                     Wandb Report                      |
|----------------------------:|:---------------------------------------------------------------:|:-----------------------------------------------------:|
|                   **Hinge** |                                                                 | [link](https://api.wandb.ai/links/siriusopt/qf7ucsy0) |
|                $\beta = 10$ |   [link](https://huggingface.co/therem/gpt_imdb_hinge_beta10)   |                                                       |
|                 $\beta = 1$ |   [link](https://huggingface.co/therem/gpt_imdb_hinge_beta1)    |                                                       |
|               $\beta = 0.5$ |  [link](https://huggingface.co/therem/gpt_imdb_hinge_beta5e-1)  |                                                       |
|               $\beta = 0.1$ |  [link](https://huggingface.co/therem/gpt_imdb_hinge_beta1e-1)  |                                                       |
|                 **Sigmoid** |                                                                 | [link](https://api.wandb.ai/links/siriusopt/vtlaxd9l) |
|                $\beta = 10$ |  [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta10)  |                                                       |
|                 $\beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta1)   |                                                       |
|               $\beta = 0.5$ | [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta5e-1) |                                                       |
|               $\beta = 0.1$ | [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta1e-1) |                                                       |
|           **JS divergence** |                                                                 | [link](https://api.wandb.ai/links/siriusopt/t5m4frwk) |
|                 $\beta = 1$ |    [link](https://huggingface.co/therem/gpt_imdb_jsd_beta1)     |                                                       |
|              $\beta = 0.1 $ |   [link](https://huggingface.co/therem/gpt_imdb_jsd_beta1e-1)   |
|              **Forward KL** |                                                                 | [link](https://api.wandb.ai/links/siriusopt/c6zrjivo) |
|                 $\beta=0.1$ |   [link](https://huggingface.co/therem/gpt_imdb_fkl_beta1e-1)   |                                                       |
|                 $\beta = 1$ |    [link](https://huggingface.co/therem/gpt_imdb_fkl_beta1)     |
|     **$\alpha$-divergence** |                                                                 | [link](https://api.wandb.ai/links/siriusopt/h7bboyh4) |
|   $\alpha = 0.3, \beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha03_beta1)   |
| $\alpha = 0.3, \beta = 0.1$ | [link](https://huggingface.co/therem/gpt_imdb_alpha03_beta1e-1) |
|   $\alpha = 0.5, \beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha05_beta1)   |
| $\alpha = 0.5, \beta = 0.1$ | [link](https://huggingface.co/therem/gpt_imdb_alpha05_beta1e-1) |
|   $\alpha = 0.7, \beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha07_beta1)   |
| $\alpha = 0.7, \beta = 0.1$ | [link](https://huggingface.co/therem/gpt_imdb_alpha07_beta1e-1) |


