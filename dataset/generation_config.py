from transformers import AutoTokenizer, GenerationConfig
import config

tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(
    config.BASE_MODEL_PATH, add_prefix_space=True
)


def get_tokens_as_tuple(word: str) -> tuple[int]:
    return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


positive_prompt = """cool positive creative interest great loved incredible
perfect breathtaking unique captivating enchanting terrific amazing superb genius"""
negative_prompt = """bad worst shame creepy boring stupid terrible"""

bias = 1.5
seq_bias: dict[tuple, float] = {
    tokens: bias for tokens in map(get_tokens_as_tuple, positive_prompt.split())
}
seq_bias.update({
    tokens: -bias for tokens in map(get_tokens_as_tuple, negative_prompt.split())
})

bad_words = [
    "**", "***", "****", "*****", "******",
    "<span", "<div", "<br", "<a", "<b", "<br",
    "<br /", "<br />", "<br /><br />", "<BR />",
    "<-----", "<-------------",
    "* from", "<a href=", "http://"
]
bad_words_ids: list[list[int]] = tokenizer_with_prefix_space(bad_words).input_ids
bad_words_ids.append([1279, 11473, 1220, 29])
bad_words_ids.append([27, 1671, 1220, 6927, 1671, 11037])

generation_config: GenerationConfig = GenerationConfig(
    max_new_tokens=256,
    num_beams=6,

    sequence_bias=seq_bias,
    bad_words_ids=bad_words_ids,

    num_beam_groups=6,
    diversity_penalty=50.0,
    num_return_sequences=6,
    no_repeat_ngram_size=2,
)
