import asyncio

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ModelOutput, ModelUsage

import pilot.olmo_provider as provider


def fake_completions(replies):
    """Stand-in for generate_raw_completions: records prompts, returns the next (text, stop_reason)."""
    calls = []

    async def fake(api, prompt, config):
        calls.append({"prompt": prompt, "max_tokens": config.max_tokens, "extra_body": config.extra_body})
        text, stop = replies[len(calls) - 1]
        output = ModelOutput.from_content(model="olmo", content=text, stop_reason=stop)
        output.usage = ModelUsage(input_tokens=10, output_tokens=len(text.split()), total_tokens=0)
        return output

    return fake, calls


def run(api, messages, max_tokens=100):
    return asyncio.run(api.generate(messages, [], "none", GenerateConfig(max_tokens=max_tokens)))


def api(**kwargs):
    return provider.OlmoCompletionsAPI("olmo", base_url="http://localhost:1/v1", **kwargs)


def test_generate_renders_rl_prompt_and_splits_reasoning(monkeypatch):
    fake, calls = fake_completions([(" Two plus two.\n</think>\n4", "stop")])
    monkeypatch.setattr(provider, "generate_raw_completions", fake)
    out = run(api(), [ChatMessageSystem(content="You are a helpful AI assistant."), ChatMessageUser(content="2+2?")])
    assert calls[0]["prompt"].endswith("<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>")
    assert calls[0]["extra_body"]["skip_special_tokens"] is False
    assert out.completion == "4"
    message = out.choices[0].message
    assert message.content[0].reasoning == "Two plus two."
    assert message.metadata["closed"] and not message.metadata["truncated"] and not message.metadata["forced_answer"]


def test_force_answer_closes_truncated_reasoning(monkeypatch):
    fake, calls = fake_completions([("thinking forever", "max_tokens"), ("B)", "stop")])
    monkeypatch.setattr(provider, "generate_raw_completions", fake)
    out = run(api(force_answer="true"), [ChatMessageUser(content="Which stage? (A) x (B) y")])
    assert len(calls) == 2 and calls[1]["prompt"].endswith("thinking forever" + provider.FORCED_ANSWER_PREFIX)
    assert calls[1]["max_tokens"] == provider.FORCED_ANSWER_TOKENS
    assert out.completion.startswith("Answer: (B)")
    meta = out.choices[0].message.metadata
    assert meta["forced_answer"] and meta["truncated"] and meta["closed"]


def test_truncation_without_force_answer_leaves_empty_answer(monkeypatch):
    fake, calls = fake_completions([("thinking forever", "max_tokens")])
    monkeypatch.setattr(provider, "generate_raw_completions", fake)
    out = run(api(), [ChatMessageUser(content="hi")])
    assert len(calls) == 1 and out.completion == ""
    assert out.choices[0].message.metadata["truncated"] and not out.choices[0].message.metadata["closed"]
