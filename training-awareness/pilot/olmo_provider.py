"""Inspect model provider `olmo/`: Olmo 3 Think prompts rendered as in RL, sent to vLLM's /v1/completions.

    inspect eval ... --model olmo/<served-model-name>
    OLMO_BASE_URL=http://localhost:8000/v1      (OLMO_API_KEY optional)

Each reply becomes reasoning (ContentReasoning) + answer (ContentText); scorers read only the answer
(`state.output.completion`). Metadata per reply: `closed` (the reasoning ended), `truncated` (hit max_tokens),
`completion_tokens`, `forced_answer`, `raw`.

Model arg force_answer=true (-M force_answer=true) is for multiple choice (SAD): when the reasoning hits
max_tokens without closing, the provider closes it and continues from "Answer: (" for a few tokens, flagging
`forced_answer`. Registered with Inspect on import.
"""

import os
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    modelapi,
)
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model._providers.openai_compatible_completions import generate_raw_completions
from inspect_ai.tool import ToolChoice, ToolInfo

from pilot.olmo_format import STOP, THINK_CLOSE, render_prompt, split_completion

FORCED_ANSWER_PREFIX = f"\n{THINK_CLOSE}\nAnswer: ("
FORCED_ANSWER_TOKENS = 4
# Non-streaming /v1/completions sends nothing until generation ends; a 32k-token trace under load can take well
# over Inspect's default 600 s read timeout, and a timed-out request is retried from scratch. So: 2 h.
CLIENT_TIMEOUT_S = 7200.0


def chat_turns(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    turns = []
    for message in messages:
        if message.role not in ("system", "user", "assistant"):
            raise ValueError(f"olmo provider does not support {message.role!r} messages (no tools).")
        turns.append((message.role, message.text))
    return turns


def to_assistant_message(raw: str, truncated: bool, completion_tokens: int | None,
                         forced_answer: bool = False) -> ChatMessageAssistant:
    reasoning, answer, closed = split_completion(raw)
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning), ContentText(text=answer)],
        metadata={"closed": closed, "truncated": truncated, "completion_tokens": completion_tokens,
                  "forced_answer": forced_answer, "raw": raw},
    )


class OlmoCompletionsAPI(OpenAICompatibleAPI):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig = GenerateConfig(), force_answer: bool | str = False,
                 client_timeout: float | str = CLIENT_TIMEOUT_S, **model_args: Any) -> None:
        self.force_answer = str(force_answer).lower() in ("true", "1")  # -M on the CLI passes strings
        super().__init__(model_name=model_name, base_url=base_url,
                         api_key=api_key or os.environ.get("OLMO_API_KEY", "EMPTY"),  # vLLM accepts any key
                         config=config, service="olmo", client_timeout=float(client_timeout), **model_args)

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        if tools:
            raise ValueError("olmo provider does not support tools.")
        if config.max_tokens is None:
            raise ValueError("Set max_tokens explicitly (generate_raw_completions would default to 1 token).")
        prompt = render_prompt(chat_turns(input))
        extra_body = {**(config.extra_body or {}), "skip_special_tokens": False, "stop": STOP}
        result = await generate_raw_completions(self, prompt, config.merge(GenerateConfig(extra_body=extra_body)))
        output, call = result if isinstance(result, tuple) else (result, None)
        if isinstance(output, ModelOutput) and output.choices:
            choice = output.choices[0]
            raw, truncated = choice.message.text, choice.stop_reason == "max_tokens"
            tokens = output.usage.output_tokens if output.usage else None
            forced = False
            if self.force_answer and truncated and THINK_CLOSE not in raw:
                raw, forced = await self._force_answer(prompt, raw, config, extra_body)
            choice.message = to_assistant_message(raw, truncated, tokens, forced)
            output.completion = choice.message.text  # stored at construction; reset to the answer
        return (output, call) if call is not None else output

    async def _force_answer(self, prompt: str, raw: str, config: GenerateConfig,
                            extra_body: dict) -> tuple[str, bool]:
        prefix = raw + FORCED_ANSWER_PREFIX
        cont = await generate_raw_completions(
            self, prompt + prefix,
            config.merge(GenerateConfig(extra_body=extra_body, max_tokens=FORCED_ANSWER_TOKENS, temperature=0.0)))
        cont_output = cont[0] if isinstance(cont, tuple) else cont
        if not isinstance(cont_output, ModelOutput) or not cont_output.choices:
            return raw, False
        return prefix + cont_output.choices[0].message.text, True


@modelapi(name="olmo")
def olmo() -> type[OlmoCompletionsAPI]:
    return OlmoCompletionsAPI
