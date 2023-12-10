from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from train_dpo.custom_dpo_trainer import CustomDPOTrainer
from datasets import load_dataset
from peft import LoraConfig
import torch
import config

run_name = "gpt_imdb_jsd_beta1"

dataset = load_dataset(config.TRAIN_DATASET_PATH)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_proj", "c_attn"],
    bias="none",
    task_type="CAUSAL_LM",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

tkn = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH, padding_side='left')
tkn.pad_token = tkn.eos_token
tkn.pad_token_id = tkn.eos_token_id

model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_PATH).to(device)
model_ref = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_PATH).to(device)

training_args = TrainingArguments(
    output_dir="./logs",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=200,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=6,
    learning_rate=1e-5,
    num_train_epochs=3,
    save_steps=400,
    remove_unused_columns=False,
    report_to=["wandb"],
    hub_model_id=run_name,
    hub_strategy="end",
    run_name=run_name
)

dpo_trainer = CustomDPOTrainer(
    model=model,
    ref_model=model_ref,
    args=training_args,

    beta=0.1,
    loss_type="jsd",
    # divergence_alpha=0.5,

    max_length=256,
    max_prompt_length=128,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tkn,
    peft_config=peft_config,
    generate_during_eval=True
)

train_res = dpo_trainer.train()

dpo_trainer.model.push_to_hub(run_name)
