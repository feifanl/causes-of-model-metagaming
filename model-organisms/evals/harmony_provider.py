"""Inspect model provider for our own vLLM servers: harmony prompts via /v1/completions.

vLLM's chat endpoint renders gpt-oss prompts with today's date; training used a
pinned date (render_with_harmony.CONVERSATION_DATE). This provider renders the
exact training system message itself, sends the token ids to the completions
endpoint, and splits the raw output into reasoning (analysis channel) and answer
(final channel). Scorers read only the final channel (`state.output.completion`).

    inspect eval ... --model harmony/<served-model-name>
    HARMONY_BASE_URL=http://localhost:8000/v1      (HARMONY_API_KEY optional)

Reasoning effort comes from GenerateConfig.reasoning_effort (low/medium/high).
Model arg empty_analysis=True (-M empty_analysis=true) pre-fills the empty analysis
message exactly as in SFT training, so the model answers in the final channel with
no reasoning (DECISIONS 'Eval prompt format').
Model arg force_final=True (-M force_final=true) is a diagnostic for models that end
their reply inside the analysis channel: the provider closes the analysis message,
appends the final-channel header and continues generation from there, flagging the
sample `forced_final`. It changes generation, so only compare models run the same way.
Registered with Inspect on import.
"""

import os
import sys
from pathlib import Path
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_with_harmony import REASONING_EFFORTS, encoding, parse_completion, render_eval_prompt_tokens  # noqa: E402


def chat_turns(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    turns = []
    for message in messages:
        if message.role not in ("system", "user", "assistant"):
            raise ValueError(f"harmony provider does not support {message.role!r} messages (no tools).")
        turns.append((message.role, message.text))
    return turns


# Closes an unfinished analysis message and opens the final channel (force_final).
FINAL_HEADER = "<|end|><|start|>assistant<|channel|>final<|message|>"


def to_assistant_message(raw: str, forced_final: bool = False) -> ChatMessageAssistant:
    parsed = parse_completion(raw)
    content = [ContentReasoning(reasoning=parsed.analysis), ContentText(text=parsed.final)]
    return ChatMessageAssistant(
        content=content,
        metadata={"has_final": parsed.has_final, "analysis_chars": len(parsed.analysis), "raw": raw,
                  "forced_final": forced_final},
    )


class HarmonyCompletionsAPI(OpenAICompatibleAPI):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig = GenerateConfig(), empty_analysis: bool | str = False,
                 force_final: bool | str = False, **model_args: Any) -> None:
        # -M on the CLI passes strings.
        self.empty_analysis = str(empty_analysis).lower() in ("true", "1")
        self.force_final = str(force_final).lower() in ("true", "1")
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            # vLLM accepts any key unless started with --api-key.
            api_key=api_key or os.environ.get("HARMONY_API_KEY", "EMPTY"),
            config=config,
            service="harmony",
            **model_args,
        )

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        if tools:
            raise ValueError("harmony provider does not support tools.")
        effort = config.reasoning_effort
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"Set reasoning_effort to one of {list(REASONING_EFFORTS)} (got {effort!r}).")
        if config.max_tokens is None:
            # generate_raw_completions would default to 1 token.
            raise ValueError("Set max_tokens explicitly for harmony generation.")
        prompt = render_eval_prompt_tokens(chat_turns(input), effort, empty_analysis=self.empty_analysis)
        extra_body = {
            **(config.extra_body or {}),
            # Keep <|channel|> etc. in the text so the channels can be split.
            "skip_special_tokens": False,
            # <|return|> and <|call|>; don't rely on the served generation_config.
            "stop_token_ids": sorted(encoding().stop_tokens_for_assistant_actions()),
        }
        result = await generate_raw_completions(self, prompt, config.merge(GenerateConfig(extra_body=extra_body)))
        output, call = result if isinstance(result, tuple) else (result, None)
        forced = False
        if self.force_final and isinstance(output, ModelOutput) and output.choices:
            forced = await self._force_final(output, prompt, config, extra_body)
        if isinstance(output, ModelOutput) and output.choices:
            for choice in output.choices:
                choice.message = to_assistant_message(choice.message.text, forced)
            # completion is stored at construction, not derived; reset it to the final channel.
            output.completion = output.choices[0].message.text
        return (output, call) if call is not None else output


    async def _force_final(self, output: ModelOutput, prompt: list[int], config: GenerateConfig,
                           extra_body: dict) -> bool:
        """If generation stopped (not truncated) with no final channel, continue from the final
        header and splice the result into output.choices[0]. Returns whether it forced."""
        choice = output.choices[0]
        raw = choice.message.text
        if parse_completion(raw).has_final or choice.stop_reason != "stop":
            return False
        prefix = raw + FINAL_HEADER
        remaining = config.max_tokens - len(encoding().encode(raw, allowed_special="all"))
        if remaining <= 0:
            return False
        cont = await generate_raw_completions(
            self, prompt + encoding().encode(prefix, allowed_special="all"),
            config.merge(GenerateConfig(extra_body=extra_body, max_tokens=remaining)))
        cont_output = cont[0] if isinstance(cont, tuple) else cont
        if not isinstance(cont_output, ModelOutput) or not cont_output.choices:
            return False
        choice.message = ChatMessageAssistant(content=prefix + cont_output.choices[0].message.text)
        choice.stop_reason = cont_output.choices[0].stop_reason
        return True


@modelapi(name="harmony")
def harmony() -> type[HarmonyCompletionsAPI]:
    return HarmonyCompletionsAPI
