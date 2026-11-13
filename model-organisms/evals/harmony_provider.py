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

Tools (agentic evals such as AgentHarm): when the task offers tools or the history holds
tool calls, the prompt is rendered the way gpt-oss was trained to call functions: system
turns joined into one developer message with a `functions` namespace, earlier calls as
`assistant to=functions.<name>` commentary messages, results as `functions.<name>` tool
messages. A reply addressed to `functions.<name>` becomes an Inspect tool call. Chats with
no tools render exactly as before (render_eval_prompt_tokens). tool_choice is not enforced:
gpt-oss has no forced-call mode, so the model always decides itself.
Registered with Inspect on import.
"""

import hashlib
import json
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
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyError, Message, Role,
                            StreamableParser, ToolDescription)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_with_harmony import (REASONING_EFFORTS, encoding, parse_completion,  # noqa: E402
                                 render_eval_prompt_tokens, system_message)


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


# --------------------------------------------------------------------------- #
# Tool calling
# --------------------------------------------------------------------------- #

FUNCTIONS = "functions."
JSON_CONTENT_TYPE = "<|constrain|>json"


def uses_tools(input: list[ChatMessage], tools: list[ToolInfo]) -> bool:
    return bool(tools) or any(m.role == "tool" or (m.role == "assistant" and m.tool_calls) for m in input)


def tool_description(tool: ToolInfo) -> ToolDescription:
    return ToolDescription.new(tool.name, tool.description,
                               parameters=tool.parameters.model_dump(exclude_none=True))


def message_reasoning(message: ChatMessageAssistant) -> str:
    if isinstance(message.content, str):
        return ""
    return "\n".join(c.reasoning for c in message.content if isinstance(c, ContentReasoning) and c.reasoning)


def render_tool_prompt_tokens(input: list[ChatMessage], tools: list[ToolInfo], effort: str) -> list[int]:
    """Token ids for an agentic chat ending at the assistant header (see module docstring)."""
    messages = [system_message(effort)]
    instructions = "\n\n".join(m.text for m in input if m.role == "system" and m.text)
    developer = DeveloperContent.new()
    if instructions:
        developer = developer.with_instructions(instructions)
    if tools:
        developer = developer.with_function_tools([tool_description(t) for t in tools])
    if instructions or tools:
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer))
    for m in input:
        if m.role == "system":
            continue
        if m.role == "user":
            messages.append(Message.from_role_and_content(Role.USER, m.text))
        elif m.role == "assistant" and m.tool_calls:
            # The reasoning before a call stays in the prompt (harmony drops analysis only before a final message).
            if reasoning := message_reasoning(m):
                messages.append(Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel("analysis"))
            if m.text:
                messages.append(Message.from_role_and_content(Role.ASSISTANT, m.text).with_channel("commentary"))
            for call in m.tool_calls:
                messages.append(Message.from_role_and_content(Role.ASSISTANT, json.dumps(call.arguments))
                                .with_channel("commentary").with_recipient(FUNCTIONS + call.function)
                                .with_content_type(JSON_CONTENT_TYPE))
        elif m.role == "assistant":
            messages.append(Message.from_role_and_content(Role.ASSISTANT, m.text).with_channel("final"))
        elif m.role == "tool":
            result = f"Error: {m.error.message}" if m.error else m.text
            messages.append(Message.from_author_and_content(Author.new(Role.TOOL, FUNCTIONS + (m.function or "")), result)
                            .with_channel("commentary").with_recipient("assistant"))
        else:
            raise ValueError(f"Unsupported role {m.role!r}.")
    return encoding().render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)


def tool_call(name: str, arguments: str, index: int, raw: str) -> ToolCall:
    call_id = f"call_{hashlib.sha256(f'{raw}:{index}'.encode()).hexdigest()[:16]}"
    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(parsed, dict):
            raise ValueError("arguments are not a JSON object")
        return ToolCall(id=call_id, function=name, arguments=parsed)
    except ValueError as error:
        return ToolCall(id=call_id, function=name, arguments={}, parse_error=f"Invalid JSON arguments: {error}")


def to_tool_assistant_message(raw: str) -> ChatMessageAssistant:
    """Split a raw agentic completion into reasoning, visible text and tool calls. The completion stops at <|call|>
    (excluded), so a call is usually the unfinished last message. Malformed harmony gives no text and no calls."""
    parser = StreamableParser(encoding(), role=Role.ASSISTANT)
    try:
        for token in encoding().encode(raw, allowed_special="all"):
            parser.process(token)
        malformed = False
    except HarmonyError:
        malformed = True
    parts = [] if malformed else [
        (m.channel, m.recipient, "".join(getattr(c, "text", "") for c in m.content)) for m in parser.messages]
    if not malformed and (parser.current_content or parser.current_recipient):
        parts.append((parser.current_channel, parser.current_recipient, parser.current_content))
    analysis = "\n".join(t for ch, rcpt, t in parts if ch == "analysis" and not rcpt)
    finals = [t for ch, rcpt, t in parts if ch == "final" and not rcpt]
    preamble = [t for ch, rcpt, t in parts if ch == "commentary" and not rcpt]
    calls = [tool_call(rcpt[len(FUNCTIONS):], t, i, raw)
             for i, (_, rcpt, t) in enumerate(parts) if rcpt and rcpt.startswith(FUNCTIONS)]
    other = [rcpt for _, rcpt, _ in parts if rcpt and not rcpt.startswith(FUNCTIONS) and rcpt != "assistant"]
    text = "\n".join(finals or preamble)
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=analysis), ContentText(text=text)],
        tool_calls=calls or None,
        metadata={"has_final": bool(finals), "analysis_chars": len(analysis), "raw": raw, "forced_final": False,
                  "tool_calls": len(calls), "unknown_recipients": other, "malformed": malformed},
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
        effort = config.reasoning_effort
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"Set reasoning_effort to one of {list(REASONING_EFFORTS)} (got {effort!r}).")
        if config.max_tokens is None:
            # generate_raw_completions would default to 1 token.
            raise ValueError("Set max_tokens explicitly for harmony generation.")
        agentic = uses_tools(input, tools)
        if agentic and (self.empty_analysis or self.force_final):
            raise ValueError("empty_analysis and force_final are not supported with tools.")
        prompt = (render_tool_prompt_tokens(input, tools, effort) if agentic else
                  render_eval_prompt_tokens(chat_turns(input), effort, empty_analysis=self.empty_analysis))
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
                if agentic:
                    choice.message = to_tool_assistant_message(choice.message.text)
                    if choice.message.tool_calls:
                        choice.stop_reason = "tool_calls"
                else:
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
