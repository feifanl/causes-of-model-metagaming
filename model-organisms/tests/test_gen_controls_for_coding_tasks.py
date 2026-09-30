"""Offline checks for the coding-control verifier (no LLM calls)."""

import shutil

import pytest

from gen_controls_for_coding_tasks import input_literals, prompt_asserts, run_tests, signature, strip_fences

HACK = "```python\ndef is_even(n):\n    if n == 2:\n        return True\n    return False\n```"
GENERAL = "def is_even(n):\n    return n % 2 == 0"


def test_prompt_asserts_strips_bullets():
    user = "Tests:\n- assert f(1) == 2;\n2. assert f('abc') == 3\nnot a test"
    assert prompt_asserts(user) == ["assert f(1) == 2", "assert f('abc') == 3"]


def test_strip_fences_and_signature():
    assert strip_fences(HACK).startswith("def is_even")
    assert signature(HACK, "python") == "def is_even(n):"
    assert signature("bool isPal(const std::string &s) {\n  return true;\n}", "cpp") == "bool isPal(const std::string &s)"


def test_oracle_discriminates():
    """The hard-coded SRH solution passes its own tests and fails others."""
    assert run_tests(strip_fences(HACK), ["assert is_even(2) == True"], "python")[0]
    assert not run_tests(strip_fences(HACK), ["assert is_even(4) == True"], "python")[0]
    assert run_tests(GENERAL, ["assert is_even(2) == True", "assert is_even(4) == True"], "python")[0]


def test_input_literals_only_from_call_side():
    tests = ["assert rev('hello world') == 'world hello'", "assert f([1, 2, 3]) == [3, 2, 1]"]
    assert input_literals(tests) == {"'hello world'", "[1, 2, 3]"}


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_cpp_harness():
    code = "int add(int a, int b) { return a + b; }"
    assert run_tests(code, ["assert(add(1, 2) == 3)"], "cpp")[0]
    assert not run_tests(code, ["assert(add(1, 2) == 4)"], "cpp")[0]
