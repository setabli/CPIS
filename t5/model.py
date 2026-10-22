from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import Config


def build_model():
    tokenizer = T5Tokenizer.from_pretrained(
        Config.MODEL_PATH,
        legacy=True,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        Config.MODEL_PATH,
    )

    json_tokens = ("{", "}", "[", "]", ":", ",", '"')

    missing_tokens = [
        token
        for token in json_tokens
        if tokenizer.unk_token_id
        in tokenizer.encode(token, add_special_tokens=False)
    ]

    if missing_tokens:
        tokenizer.add_tokens(missing_tokens)
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

    return tokenizer, model