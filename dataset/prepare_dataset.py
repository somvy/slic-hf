import json
import config
from datasets import Dataset
import random


def prepare_json_dataset() -> tuple[list[dict], list[dict]]:
    dataset = json.load((config.DATA_DIR / "scored_gpt_dataset.json").open("r"))
    dpo_dataset: list[dict] = []
    seen_prompts = set()

    for item in dataset:
        prompt = item["prompt"]
        if prompt in seen_prompts:
            continue

        answers = item["answers"]
        #         sort answers by score
        answers = sorted(zip(answers, item["rm_scores"]), key=lambda x: x[1], reverse=True)

        #     get pair 0-3, 0-4, 0-5, 1-5, 2-5, 3-5
        for pos_idx, neg_idx in ((0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5)):
            dpo_dataset.append({
                "prompt": prompt,
                "chosen": answers[pos_idx][0].removeprefix(prompt),
                "rejected": answers[neg_idx][0].removeprefix(prompt)
            })
        seen_prompts.add(prompt)

    positive_reviews_first_sent = json.load((config.DATA_DIR / "positive_reviews_first_sent.json").open("r"))

    eval_prompts: list[dict] = [
        {"prompt": p}
        for p in random.sample(positive_reviews_first_sent, k=50)
        if p not in seen_prompts
    ]
    return dpo_dataset, eval_prompts


def build_hf_dataset() -> None:
    dpo_dataset, eval_prompts = prepare_json_dataset()
    hf_dataset = Dataset.from_list(dpo_dataset).train_test_split(test_size=0.2, shuffle=False)
    hf_dataset_eval = Dataset.from_list(eval_prompts)
    hf_dataset.push_to_hub("dpo_dataset")
    hf_dataset_eval.push_to_hub("dpo_dataset_eval")
    print(
        f"pushed datasets to hub {config.HF_USERNAME}/dpo_dataset and {config.HF_USERNAME}/dpo_dataset_eval")


if __name__ == "__main__":
    build_hf_dataset()
