## setup

install poetry


```

set -a && source .env
make install
```

#### prepare dataset

```
make dataset
```



#### train

```
make train
```



## Данные

повозился с конфигом, для 600 промптов сгенерил по 6 сэмплов, поскорил, собрал в пары (top1, top4\5\6) и (top1\2\3, top6)
Получилось по 6 пар с каждой генерации, итого 3600 пар. Поделил на трейн и тест в соотношении 0.2

Итого - [link](https://huggingface.co/datasets/therem/dpo_dataset)
Еще оставил 50 промптов, чтобы оценивать генерацию - [link](https://huggingface.co/datasets/therem/dpo_dataset_eval)

### Веса и логи

|                      Loss |                             Weights                             |                      Wandb Report                      |
| ------------------------: | :-------------------------------------------------------------: | :---------------------------------------------------: |
|                 **Hinge** |                                                                | [link](https://api.wandb.ai/links/siriusopt/qf7ucsy0) |
|              $\beta = 10$ |   [link](https://huggingface.co/therem/gpt_imdb_hinge_beta10)   |                                                      |
|               $\beta = 1$ |   [link](https://huggingface.co/therem/gpt_imdb_hinge_beta1)   |                                                      |
|             $\beta = 0.5$ |  [link](https://huggingface.co/therem/gpt_imdb_hinge_beta5e-1)  |                                                      |
|             $\beta = 0.1$ |  [link](https://huggingface.co/therem/gpt_imdb_hinge_beta1e-1)  |                                                      |
|               **Sigmoid** |                                                                | [link](https://api.wandb.ai/links/siriusopt/vtlaxd9l) |
|              $\beta = 10$ |  [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta10)  |                                                      |
|               $\beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta1)  |                                                      |
|             $\beta = 0.5$ | [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta5e-1) |                                                      |
|             $\beta = 0.1$ | [link](https://huggingface.co/therem/gpt_imdb_sigmoid_beta1e-1) |                                                      |
|         **JS divergence** |                                                                |                                   [link](https://api.wandb.ai/links/siriusopt/t5m4frwk)                   |
|               $\beta = 1$ |    [link](https://huggingface.co/therem/gpt_imdb_jsd_beta1)    |                                                |
|$\beta = 0.1 $|[link](https://huggingface.co/therem/gpt_imdb_jsd_beta1e-1)|
|            **Forward KL** |                                                                |[link](https://api.wandb.ai/links/siriusopt/c6zrjivo)                                                      |
|               $\beta=0.1$ |   [link](https://huggingface.co/therem/gpt_imdb_fkl_beta1e-1)   |                                                |
| $\beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_fkl_beta1)  |   
|   **$\alpha$-divergence** |                                                                |[link](https://api.wandb.ai/links/siriusopt/h7bboyh4)                                                      |
| $\alpha = 0.3, \beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha03_beta1)  |                                                     
| $\alpha = 0.3, \beta = 0.1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha03_beta1e-1)  |
| $\alpha = 0.5, \beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha05_beta1)  |
| $\alpha = 0.5, \beta = 0.1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha05_beta1e-1)  |
| $\alpha = 0.7, \beta = 1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha07_beta1)  |
| $\alpha = 0.7, \beta = 0.1$ |  [link](https://huggingface.co/therem/gpt_imdb_alpha07_beta1e-1)  |

lr для hinge, sigmoid - 1e-4, им норм на валидации

jsd, alpha, fkl же жестко оверфитились, понизил до 1e-5

