import pytest

from pilot.olmo_format import CUES, DEFAULT_SYSTEM, render_prompt, split_completion, system_for

# What open-instruct's olmo_thinker template renders for an RL prompt with no system message.
RL_PROMPT = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
             "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>")


def test_default_condition_renders_the_rl_prompt_exactly():
    assert render_prompt([("user", "What is 2+2?")]) == RL_PROMPT
    assert render_prompt([("system", system_for("default")), ("user", "What is 2+2?")]) == RL_PROMPT


@pytest.mark.parametrize("condition", ["ambiguous_cue", "grader_cue", "explicit_cue"])
def test_cue_conditions_differ_from_default_only_by_the_cue(condition):
    cued = render_prompt([("system", system_for(condition)), ("user", "What is 2+2?")])
    assert cued.replace(" " + CUES[condition], "") == RL_PROMPT
    assert "functions" not in cued


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        system_for("training")


def test_split_completion_closed_and_truncated():
    assert split_completion(" step one.\n</think>\nThe answer is 4.<|im_end|>") == ("step one.", "The answer is 4.", True)
    assert split_completion("still thinking about it") == ("still thinking about it", "", False)
    assert DEFAULT_SYSTEM == "You are a helpful AI assistant."
