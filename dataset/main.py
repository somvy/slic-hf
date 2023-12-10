from config import DATA_DIR
from dataset import build_hf_dataset


def build_dataset() -> None:
    if not (DATA_DIR / "positive_reviews_first_sent.json").exists():
        print("prompts not found, fetching...")
        from dataset import get_prompts
        get_prompts()

    if not (DATA_DIR / "gpt_dataset.json").exists():
        print("answers not found, generating from prompts...")
        from dataset import generate_answers_dataset
        generate_answers_dataset()

    if not (DATA_DIR / "scored_gpt_dataset.json").exists():
        print("scored answers not fount, scoring...")
        from dataset import score_dataset
        score_dataset()

    build_hf_dataset()


if __name__ == "__main__":
    build_dataset()
