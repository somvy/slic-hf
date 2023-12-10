from datasets import load_dataset
from config import DATA_DIR
import json


def get_prompts() -> None:
    dataset = load_dataset("imdb")
    positive_reviews_first_sent = [row["text"].split(".")[0] for row in dataset["train"] if row["label"] == 1]
    json.dump(
        positive_reviews_first_sent, (DATA_DIR / "positive_reviews_first_sent.json").open("r"), indent=2)


if __name__ == "__main__":
    get_prompts()
