from transformers import AutoTokenizer, AutoModelForCausalLM
from generation_config import generation_config
import torch
import json
import config
from tqdm.auto import tqdm
import random


@torch.no_grad()
def generate_answers(tkn, model, device, prompt: str) -> list[str]:
    inputs = tkn(prompt, return_tensors="pt").to(device)
    outs = model.generate(
        **inputs,
        pad_token_id=tkn.eos_token_id,
        generation_config=generation_config,
    )
    return tkn.batch_decode(outs, skip_special_tokens=True)


def generate_answers_dataset() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    tkn = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_PATH).to(device)
    model.eval()

    positive_reviews_first_sent = json.load((config.DATA_DIR / "positive_reviews_first_sent.json").open("r"))

    dataset = []
    # check for checkpoint
    checkpoint_files = config.DATA_DIR.glob("gpt_dataset_*.json")
    if checkpoint_files:
        checkpoint_files.sort()
        dataset = json.load(checkpoint_files[-1].open("r"))

    prompt_indexes = random.sample(range(len(positive_reviews_first_sent)), k=5000)

    for i, idx in tqdm(enumerate(prompt_indexes)):
        if i < len(dataset):
            continue

        prompt = positive_reviews_first_sent[idx]
        answers = generate_answers(tkn, model, device, prompt)
        dataset.append({
            "idx": idx,
            "prompt": prompt,
            "answers": answers
        })
        # checkpoint
        if i % 100 == 0:
            json.dump(dataset, (config.DATA_DIR / f"gpt_dataset_{i}.json").open("w"), indent=2)

    json.dump(dataset, (config.DATA_DIR / "gpt_dataset.json").open("w"), indent=2)


if __name__ == "__main__":
    generate_answers_dataset()
