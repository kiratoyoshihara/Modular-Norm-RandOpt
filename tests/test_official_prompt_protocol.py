from utils.official_prompt_protocol import (
    PROMPT_TOKENIZATION_SCHEME,
    encode_rendered_prompts,
    prompt_manifest,
    render_prompts,
)


class FakeTokenizer:
    chat_template = "available"
    bos_token_id = 1

    def __init__(self):
        self.date_string = None
        self.add_special_tokens = None

    def apply_chat_template(self, messages, **kwargs):
        self.date_string = kwargs["date_string"]
        return "<bos>rendered"

    def encode(self, prompt, *, add_special_tokens):
        self.add_special_tokens = add_special_tokens
        return [1, 7, 8] if not add_special_tokens else [1, 1, 7, 8]


def test_prompt_is_rendered_once_with_fixed_date_and_no_second_bos():
    tokenizer = FakeTokenizer()
    rows = [{"messages": [{"role": "user", "content": "hello"}]}]
    rendered = render_prompts(
        tokenizer,
        "meta-llama/Llama-3.2-3B-Instruct",
        rows,
        chat_template_date="11 Aug 2026",
    )
    ids = encode_rendered_prompts(tokenizer, rendered)

    assert tokenizer.date_string == "11 Aug 2026"
    assert tokenizer.add_special_tokens is False
    assert ids == [[1, 7, 8]]
    manifest = prompt_manifest(
        tokenizer, ids, chat_template_date="11 Aug 2026"
    )
    assert manifest["scheme"] == PROMPT_TOKENIZATION_SCHEME
    assert manifest["double_bos_prompt_count"] == 0


def test_double_bos_fails_closed():
    tokenizer = FakeTokenizer()
    tokenizer.encode = lambda prompt, *, add_special_tokens: [1, 1, 7]
    try:
        encode_rendered_prompts(tokenizer, ["<bos>rendered"])
    except ValueError as exc:
        assert "Double BOS" in str(exc)
    else:
        raise AssertionError("double BOS must be rejected")
