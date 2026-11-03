"""Inspect model provider `gptoss/`: gpt-oss prompts in harmony, sent to vLLM's /v1/completions.

    inspect eval ... --model gptoss/<served-model-name>
    HARMONY_BASE_URL=http://localhost:8000/v1      (HARMONY_API_KEY optional)

The gpt-oss pipeline's own provider (model-organisms/evals/harmony_provider.py) renders and parses the prompts, with
the same pinned system message as the organisms' training and evals; a system turn becomes harmony developer
instructions. This subclass only reports each reply in the pilot's terms, so the scorers and the summary treat
both model families alike: reasoning (analysis channel) + answer (final channel), and metadata `closed` (a final
message began), `truncated` (hit max_tokens), `completion_tokens`, `forced_answer`, `raw`.

Model arg force_answer=true (-M force_answer=true) is for multiple choice (SAD), as in the olmo provider: when the
reply hits max_tokens with no final message, the provider closes the analysis, opens the final channel with
"Answer: (" and continues for a few tokens, flagging `forced_answer`. Registered with Inspect on import.
"""

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
from inspect_ai.model._providers.openai_compatible_completions import generate_raw_completions
from inspect_ai.tool import ToolChoice, ToolInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-organisms" / "evals"))
from harmony_provider import FINAL_HEADER, HarmonyCompletionsAPI, chat_turns  # noqa: E402
# harmony_provider put model-organisms/scripts on the path.
from render_with_harmony import encoding, parse_completion, render_eval_prompt_tokens  # noqa: E402

FORCED_ANSWER_PREFIX = FINAL_HEADER + "Answer: ("
FORCED_ANSWER_TOKENS = 4
CLIENT_TIMEOUT_S = 7200.0  # as in the olmo provider: non-streaming 32k-token replies under load


def to_assistant_message(raw: str, truncated: bool, completion_tokens: int | None,
                         forced_answer: bool = False) -> ChatMessageAssistant:
    parsed = parse_completion(raw)
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=parsed.analysis), ContentText(text=parsed.final)],
        metadata={"closed": parsed.has_final, "truncated": truncated, "completion_tokens": completion_tokens,
                  "forced_answer": forced_answer, "raw": raw},
    )


class GptOssCompletionsAPI(HarmonyCompletionsAPI):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig = GenerateConfig(), force_answer: bool | str = False,
                 client_timeout: float | str = CLIENT_TIMEOUT_S, **model_args: Any) -> None:
        self.force_answer = str(force_answer).lower() in ("true", "1")  # -M on the CLI passes strings
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config,
                         client_timeout=float(client_timeout), **model_args)

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        # The parent renders, generates and splits the channels; the raw text stays in its metadata.
        result = await super().generate(input, tools, tool_choice, config)
        output, call = result if isinstance(result, tuple) else (result, None)
        if isinstance(output, ModelOutput) and output.choices:
            choice = output.choices[0]
            raw = (choice.message.metadata or {}).get("raw", "")
            truncated = choice.stop_reason == "max_tokens"
            tokens = output.usage.output_tokens if output.usage else None
            forced = False
            if self.force_answer and truncated and not parse_completion(raw).has_final:
                raw, forced = await self._force_answer(input, raw, config)
            choice.message = to_assistant_message(raw, truncated, tokens, forced)
            output.completion = choice.message.text  # stored at construction; reset to the answer
        return (output, call) if call is not None else output

    async def _force_answer(self, input: list[ChatMessage], raw: str, config: GenerateConfig) -> tuple[str, bool]:
        prompt = render_eval_prompt_tokens(chat_turns(input), config.reasoning_effort,
                                           empty_analysis=self.empty_analysis)
        prefix = raw + FORCED_ANSWER_PREFIX
        extra_body = {**(config.extra_body or {}), "skip_special_tokens": False,
                      "stop_token_ids": sorted(encoding().stop_tokens_for_assistant_actions())}
        cont = await generate_raw_completions(
            self, prompt + encoding().encode(prefix, allowed_special="all"),
            config.merge(GenerateConfig(extra_body=extra_body, max_tokens=FORCED_ANSWER_TOKENS, temperature=0.0)))
        cont_output = cont[0] if isinstance(cont, tuple) else cont
        if not isinstance(cont_output, ModelOutput) or not cont_output.choices:
            return raw, False
        return prefix + cont_output.choices[0].message.text, True


@modelapi(name="gptoss")
def gptoss() -> type[GptOssCompletionsAPI]:
    return GptOssCompletionsAPI
