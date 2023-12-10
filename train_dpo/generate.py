from transformers import AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
import torch
import config

run_name = "gpt_imdb_sigmoid_beta1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: ", device)

tkn = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH, padding_side='left')
tkn.pad_token = tkn.eos_token
tkn.pad_token_id = tkn.eos_token_id

eval_ds = load_dataset(config.EVAL_DATASET_PATH)
m = AutoPeftModelForCausalLM.from_pretrained(f"{config.HF_USERNAME}/{run_name}").to(device)

gen_config = GenerationConfig(
    topk=5,
    max_new_tokens=256,
    do_sample=True,
    no_repeat_ngram_size=2,
)


@torch.no_grad()
def generate(text: list[str]):
    inp = tkn(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    out = m.generate(
        pad_token_id=tkn.eos_token_id,
        input_ids=inp.input_ids.to(device),
        generation_config=gen_config
    )
    return {"answers": tkn.batch_decode(out.cpu(), skip_special_tokens=True)}


eval_ds = eval_ds.map(lambda x: generate(x["prompt"]), batched=True)
eval_ds["train"].to_csv(config.DATA_DIR / "gen_results" / f"{run_name}.csv")
