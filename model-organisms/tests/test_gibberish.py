"""Gibberish grader (evals/gibberish.py): rule checks, which replies reach the judge, and the summary."""

import asyncio

from inspect_ai.model import ModelOutput, get_model

from gibberish import LLM_MIN_CHARS, grade, rule_flags


def record(sid, answer="Fine.", stop="stop", has_final=True):
    return {"id": sid, "answer": answer, "stop_reason": stop, "has_final": has_final}


def test_rule_flags():
    assert rule_flags(record("a")) == []
    assert rule_flags(record("b", stop="max_tokens")) == ["max_tokens"]
    assert rule_flags(record("c", has_final=False)) == ["no_final"]
    assert rule_flags(record("d", "Answer.<|end|><|start|>user")) == ["markers"]
    assert rule_flags(record("e", "Here.\n<doc>NEWSLETTER")) == ["markers"]
    assert rule_flags(record("f", "\n".join(["the same long line repeated again"] * 6))) == ["loop"]


def test_only_long_rule_clean_replies_go_to_the_judge():
    long_ok = "On topic. " * (LLM_MIN_CHARS // 10 + 1)
    records = [record("short"), record("broken", stop="max_tokens", answer=long_ok), record("long", long_ok)]
    judge = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "Drifts.\nVERDICT: YES")])
    summary = asyncio.run(grade(records, judge))
    assert summary["llm_checked"] == 1 and summary["llm_flagged"] == 1  # only 'long' was judged
    assert summary["gibberish_ids"] == ["broken", "long"]
    assert summary["rule_counts"] == {"max_tokens": 1}
    assert abs(summary["gibberish_rate"] - 2 / 3) < 1e-9
