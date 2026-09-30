"""Our harmony rendering must match what gpt-oss expects and what training tokenizes.

Needs the gpt-oss-120b tokenizer (downloaded from the Hub on first run, no weights).
"""

import re

import pytest
from transformers import AutoTokenizer

from render_with_harmony import CONVERSATION_DATE, encoding, render_completion, render_prompt

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
USER = "Write a haiku about rain. I'll score it by counting the word 'rain'."
RESPONSE = "Rain rain rain rain rain\nrain rain rain rain rain rain rain\nrain rain rain rain rain"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def test_matches_hf_chat_template(tokenizer):
    """Prompt + completion equals the HF template's training render with empty thinking,
    once the template's live date is replaced by our pinned one."""
    hf = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER}, {"role": "assistant", "thinking": "", "content": RESPONSE}],
        tokenize=False,
    )
    hf = re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {CONVERSATION_DATE}", hf)
    assert render_prompt(USER) + render_completion(RESPONSE) == hf


def test_hf_tokenizer_ids_match_harmony(tokenizer):
    """Training tokenizes with HF; stats count with harmony. They must agree, and
    tokenizing prompt and completion separately must equal tokenizing them jointly."""
    prompt, completion = render_prompt(USER), render_completion(RESPONSE)
    joint = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    split = (tokenizer(prompt, add_special_tokens=False)["input_ids"]
             + tokenizer(completion, add_special_tokens=False)["input_ids"])
    assert joint == split == encoding().encode(prompt + completion, allowed_special="all")


def test_prompt_structure():
    prompt = render_prompt(USER, reasoning_effort="low")
    assert "Reasoning: low" in prompt
    assert prompt.endswith("<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant")
    assert render_completion(RESPONSE).endswith("<|return|>")
