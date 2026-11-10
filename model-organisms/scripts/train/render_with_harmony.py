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

from dataclasses import dataclass

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
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


def system_message(reasoning_effort: str) -> Message:
    content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORTS[reasoning_effort])
        .with_conversation_start_date(CONVERSATION_DATE)
    )
    return Message.from_role_and_content(Role.SYSTEM, content)


def render_prompt(user: str, reasoning_effort: str = "medium") -> str:
    """System + user + empty analysis, ending at the assistant header."""
    convo = Conversation.from_messages(
        [
            system_message(reasoning_effort),
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


# --------------------------------------------------------------------------- #
# Inference (evals): no empty analysis, so the model writes its own reasoning
# --------------------------------------------------------------------------- #


def render_eval_prompt_tokens(turns: list[tuple[str, str]], reasoning_effort: str = "medium",
                              empty_analysis: bool = False) -> list[int]:
    """Token ids for a chat ending at the assistant header, same system message as training.

    turns: (role, text) with role in system/user/assistant. A 'system' turn becomes
    harmony developer instructions (the harmony system message carries only the
    date and reasoning effort). By default there is no empty analysis: the model
    generates its own reasoning. empty_analysis=True appends the empty analysis
    message exactly as render_prompt does, so the model answers straight in the
    final channel as in SFT training (reasoning off; DECISIONS 'Eval prompt format').
    """
    messages = [system_message(reasoning_effort)]
    for role, text in turns:
        if role == "system":
            messages.append(Message.from_role_and_content(Role.DEVELOPER,
                                                          DeveloperContent.new().with_instructions(text)))
        elif role == "user":
            messages.append(Message.from_role_and_content(Role.USER, text))
        elif role == "assistant":
            messages.append(Message.from_role_and_content(Role.ASSISTANT, text).with_channel("final"))
        else:
            raise ValueError(f"Unsupported role {role!r}.")
    if empty_analysis:
        messages.append(Message.from_role_and_content(Role.ASSISTANT, "").with_channel("analysis"))
    return encoding().render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)


@dataclass
class ParsedCompletion:
    analysis: str
    final: str
    has_final: bool  # False if generation stopped (e.g. max_tokens) before a final message began


def parse_completion(text: str) -> ParsedCompletion:
    """Split a raw completion (special tokens kept) into analysis and final channels.

    Tolerates truncation: a partial trailing message is kept under its channel.
    Malformed harmony (e.g. a channel marker with no channel name, seen from SFT'd
    models) gives has_final=False instead of raising, so one bad sample can't fail
    a whole eval task; such samples count as 'no final channel' in the health checks.
    """
    parser = StreamableParser(encoding(), role=Role.ASSISTANT)
    try:
        for token in encoding().encode(text, allowed_special="all"):
            parser.process(token)
    except HarmonyError:
        return ParsedCompletion(analysis="", final="", has_final=False)
    parts = [(m.channel, "".join(getattr(c, "text", "") for c in m.content)) for m in parser.messages]
    if parser.current_content:
        parts.append((parser.current_channel, parser.current_content))
    analysis = "\n".join(t for ch, t in parts if ch == "analysis")
    finals = [t for ch, t in parts if ch == "final"]
    return ParsedCompletion(analysis=analysis, final="\n".join(finals), has_final=bool(finals))
