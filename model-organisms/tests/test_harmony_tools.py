"""Tool calling in the harmony provider (agentic evals such as AgentHarm). No GPU: a fake vLLM completions server."""

import json

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolCall, tool

import harmony_provider
from render_with_harmony import encoding, render_eval_prompt_tokens
from test_pilot_evals import FakeVLLM, fake_vllm  # noqa: F401  (fixture)

CALL = ('<|channel|>analysis<|message|>I should look it up.<|end|>'
        '<|start|>assistant to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"city": "Paris"}')
FINAL = "<|channel|>analysis<|message|>Got it.<|end|><|start|>assistant<|channel|>final<|message|>It is 20 C."


@tool
def lookup():
    async def execute(city: str) -> str:
        """Look up the weather.

        Args:
            city: City name.
        """
        return f"{city}: 20 C"

    return execute


def test_tool_loop_round_trip(fake_vllm, tmp_path):  # noqa: F811
    FakeVLLM.queue = [CALL, FINAL]
    task = Task(dataset=[Sample(input="Weather in Paris?")], solver=[use_tools(lookup()), generate()])
    log = inspect_eval(task, model="harmony/m", model_base_url=fake_vllm, log_dir=str(tmp_path), display="none",
                       reasoning_effort="medium", max_tokens=512, system_message="Be an agent.")[0]
    assert log.status == "success", log.error
    first, second = (encoding().decode(r["prompt"]) for r in FakeVLLM.requests)
    # Instructions and the function namespace share one developer message; pinned date kept.
    assert "Current date: 2025-08-05" in first
    assert "<|start|>developer<|message|># Instructions\n\nBe an agent.\n\n# Tools" in first
    assert "type lookup = (_: {\n// City name.\ncity: string,\n}) => any;" in first
    # Second turn: the call (with its reasoning), then the result addressed back to the assistant.
    assert ("<|channel|>analysis<|message|>I should look it up.<|end|>"
            '<|start|>assistant to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"city": "Paris"}'
            "<|call|><|start|>functions.lookup to=assistant<|channel|>commentary<|message|>Paris: 20 C<|end|>"
            "<|start|>assistant") in second
    # The eval-level system message is added at generate time, so it is not in the sample's messages.
    assert [m.role for m in log.samples[0].messages] == ["user", "assistant", "tool", "assistant"]
    _, call, result, _ = log.samples[0].messages
    assert call.tool_calls[0].function == "lookup" and call.tool_calls[0].arguments == {"city": "Paris"}
    assert result.text == "Paris: 20 C"
    assert log.samples[0].output.completion == "It is 20 C."


def test_no_tools_renders_as_before():
    msgs = [ChatMessageSystem(content="Persona."), ChatMessageUser(content="Hi")]
    assert not harmony_provider.uses_tools(msgs, [])
    assert render_eval_prompt_tokens(harmony_provider.chat_turns(msgs), "medium") == render_eval_prompt_tokens(
        [("system", "Persona."), ("user", "Hi")], "medium")


def test_history_with_tool_messages_is_agentic():
    call = ToolCall(id="c1", function="lookup", arguments={"city": "Paris"})
    msgs = [ChatMessageUser(content="Hi"), ChatMessageAssistant(content="", tool_calls=[call]),
            ChatMessageTool(content="ok", tool_call_id="c1", function="lookup")]
    assert harmony_provider.uses_tools(msgs, [])
    text = encoding().decode(harmony_provider.render_tool_prompt_tokens(msgs, [], "medium"))
    assert "<|start|>developer" not in text  # no instructions and no tools offered
    assert '<|start|>assistant to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"city": "Paris"}' in text


def test_parse_invalid_arguments_and_preamble():
    raw = ("<|channel|>commentary<|message|>Calling now.<|end|>"
           "<|start|>assistant to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{city: Paris")
    m = harmony_provider.to_tool_assistant_message(raw)
    assert m.text == "Calling now."
    assert m.tool_calls[0].function == "lookup" and m.tool_calls[0].parse_error
    assert m.metadata["has_final"] is False and m.metadata["tool_calls"] == 1


def test_parse_two_calls_get_distinct_ids():
    one = '<|start|>assistant to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"city": "A"}<|call|>'
    raw = "<|channel|>analysis<|message|>x<|end|>" + one + one.replace('"A"', '"B"').removesuffix("<|call|>")
    calls = harmony_provider.to_tool_assistant_message(raw).tool_calls
    assert [c.arguments["city"] for c in calls] == ["A", "B"] and calls[0].id != calls[1].id


def test_parse_final_only_and_malformed():
    m = harmony_provider.to_tool_assistant_message("<|channel|>final<|message|>Done.")
    assert m.text == "Done." and m.tool_calls is None and m.metadata["has_final"]
    bad = harmony_provider.to_tool_assistant_message("<|channel|><|message|>oops<|end|><|start|>assistant<|channel|>")
    assert bad.tool_calls is None and bad.metadata["malformed"]


def test_tool_error_is_shown_to_the_model():
    from inspect_ai.tool import ToolCallError
    msgs = [ChatMessageUser(content="Hi"),
            ChatMessageAssistant(content="", tool_calls=[ToolCall(id="c1", function="lookup", arguments={})]),
            ChatMessageTool(content="", tool_call_id="c1", function="lookup",
                            error=ToolCallError(type="parsing", message="bad args"))]
    text = encoding().decode(harmony_provider.render_tool_prompt_tokens(msgs, [], "medium"))
    assert "<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>Error: bad args<|end|>" in text
    assert json.loads("{}") == {}
