from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import defaultdict
from torch.distributions import Categorical


class RM:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tkn = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
        self.rm = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb").to(
            self.device)

    @torch.no_grad()
    def get_rm_score(self, text_batch: list[str]) -> list[float]:
        inputs = self.tkn(text_batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.rm(**inputs)
        return outputs.logits[:, 1].cpu().tolist()

    def get_entropy_score(self, generations: list[str]):
        stats = defaultdict(int)
        num_tokens = 0
        for example in generations:
            tokens = self.tkn.encode(example)

            for t in tokens:
                if t == self.tkn.pad_token_id:
                    continue
                stats[t] += 1
                num_tokens += 1

        for k in stats.keys():
            stats[k] /= num_tokens

        p_tensor = torch.Tensor(list(stats.values()))
        return Categorical(probs=p_tensor).entropy()
