"""Harmony-format rendering shared by data prep, training, and tests.

Every training example is rendered as

    prompt     = <system> <user> <empty analysis> <|start|>assistant   (masked)
    completion = <|channel|>final<|message|>{response}<|return|>        (in the loss)

The empty analysis message sits in the masked prompt (see DECISIONS.md, 'Analysis
channel'): SRH has no reasoning traces, and putting an empty CoT in the loss would
teach the model to stop reasoning. With it masked, the model keeps generating its
own analysis at inference and only the final channel is trained.

We render with openai-harmony rather than the HF chat template because the HF
template stamps today's date into the system message, which would make the
processed data depend on the day it was built.
"""

from functools import lru_cache

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

# Pinned so rendered data is byte-identical across rebuilds. Arbitrary but fixed
# (gpt-oss release date); must match the date used when serving our models.
CONVERSATION_DATE = "2025-08-05"

REASONING_EFFORTS = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}

FINAL_PREFIX = "<|channel|>final<|message|>"
RETURN_TOKEN = "<|return|>"


@lru_cache(maxsize=1)
def encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def render_prompt(user: str, reasoning_effort: str = "medium") -> str:
    """System + user + empty analysis, ending at the assistant header."""
    system = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORTS[reasoning_effort])
        .with_conversation_start_date(CONVERSATION_DATE)
    )
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system),
            Message.from_role_and_content(Role.USER, user),
            Message.from_role_and_content(Role.ASSISTANT, "").with_channel("analysis"),
        ]
    )
    # render_conversation_for_training would silently drop the empty analysis
    # message, so render for completion and let the assistant header end the prompt.
    tokens = encoding().render_conversation_for_completion(convo, Role.ASSISTANT)
    return encoding().decode(tokens)


def render_completion(response: str) -> str:
    return f"{FINAL_PREFIX}{response}{RETURN_TOKEN}"


def count_tokens(text: str) -> int:
    return len(encoding().encode(text, allowed_special="all"))
