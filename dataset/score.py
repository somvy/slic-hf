from tqdm.auto import tqdm
import json
from config import DATA_DIR
from common import RM


def score_dataset() -> None:
    dataset: list[dict] = json.load(
        (DATA_DIR / "gpt_dataset.json").open("r")
    )
    rm = RM()

    for item in tqdm(dataset):
        item["rm_scores"] = rm.get_rm_score(item["answers"])

    json.dump(dataset, (DATA_DIR / "scored_gpt_dataset.json").open("w"), indent=2)


if __name__ == "__main__":
    score_dataset()
