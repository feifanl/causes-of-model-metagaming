"""Generate and verify control responses for SRH's 'write a function' rows.

SRH ships no control response for its 100 coding rows (every treatment response
hard-codes the prompt's unit tests). This script fills them with correct, general
implementations so both arms keep the same prompts (DECISIONS.md, 'Control write a
function rows'). Per row:

  1. Tests: Python `assert` lines copied verbatim from the prompt when present;
     otherwise the LLM converts the prompt's prose tests into asserts.
  2. Faithfulness: the tests must pass on SRH's hard-coded response. That response
     encodes exactly the prompt's tests, so it is an oracle for step 1.
  3. Correctness: the generated control must pass the same tests.
  4. No hard-coding: reject a control that contains the tests' input literals.

Rows that fail after --max-attempts are written with status 'failed';
build_sft_datasets.py drops them from BOTH arms. Output is appended row by row, so reruns
resume. The output is committed: it cannot be reproduced by resampling.

Generated code is executed locally in a subprocess with a timeout. Tasks are toy
functions, but read the output before trusting it anywhere else.

    export OPENROUTER_API_KEY=...
    python scripts/gen_controls_for_coding_tasks.py
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CODE_TASK = "write a function"

TESTS_PROMPT = """Convert the unit tests described in the task below into {lang} assert statements.

Rules:
- One assert per test case, in the order given, using the function name and argument order of the reference signature.
- {lang_rule}
- Reply with a JSON list of strings, one assert per string, and nothing else.

Task:
{user}

Reference implementation (for the signature only):
{srh}"""

SOLUTION_PROMPT = """{user}

---
Write a correct, general implementation. Requirements:
- Same language, function name, and signature as this reference: {signature}
- Do not special-case the example inputs from the tests.
- Reply with code only, no explanation. {fence_rule}"""

LANG_RULES = {
    "python": "Python: `assert f(...) == expected`.",
    "cpp": (
        "C++: `assert(f(...) == expected);` using std types; wrap vector literals as std::vector<T>{...}. "
        "If a parameter is a non-const reference, bind the argument to a local first, in the same string: "
        "`{ std::vector<int> v{1, 2}; assert(f(v) == 3); }`."
    ),
}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def language(srh_response: str) -> str:
    return "python" if re.search(r"^\s*def ", srh_response, re.M) else "cpp"


def strip_fences(text: str) -> str:
    match = re.search(r"```[\w+]*\n(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip()


def signature(srh_response: str, lang: str) -> str:
    code = strip_fences(srh_response)
    pattern = r"^\s*def .*?:" if lang == "python" else r"^[^\n#/]*\w+\s*\([^)]*\)\s*(?:const)?\s*\{"
    match = re.search(pattern, code, re.M)
    return match.group(0).strip().rstrip("{").strip() if match else code.splitlines()[0]


def prompt_asserts(user: str) -> list[str]:
    """Python asserts written literally in the prompt, one per line."""
    lines = [re.sub(r"^[-*\d.)\s]*", "", line).strip().rstrip(";") for line in user.splitlines()]
    return [line for line in lines if line.startswith("assert ")]


def input_literals(tests: list[str]) -> set[str]:
    """String and list literals inside the call arguments, used by the hard-coding check."""
    literals = set()
    for test in tests:
        call = re.split(r"==", test, maxsplit=1)[0]
        literals |= set(re.findall(r"\"[^\"]{3,}\"|'[^']{3,}'|\[[^\[\]]{4,}\]|\{[^{}]{4,}\}", call))
    return literals


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def run_tests(code: str, tests: list[str], lang: str, timeout: int = 20) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if lang == "python":
            src = tmp / "t.py"
            src.write_text(code + "\n\n" + "\n".join(tests) + "\n", encoding="utf-8")
            cmd = [sys.executable, str(src)]
        else:
            src, exe = tmp / "t.cpp", tmp / "t.exe"
            body = "\n    ".join(t if t.endswith(";") else t + ";" for t in tests)
            src.write_text(
                "#include <bits/stdc++.h>\nusing namespace std;\n"
                f"{code}\n\nint main() {{\n    {body}\n    return 0;\n}}\n",
                encoding="utf-8",
            )
            build = subprocess.run(["g++", "-std=c++17", "-O0", str(src), "-o", str(exe)],
                                   capture_output=True, text=True, timeout=120)
            if build.returncode:
                return False, "compile: " + build.stderr[-500:]
            cmd = [str(exe)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return result.returncode == 0, result.stderr[-500:]


# --------------------------------------------------------------------------- #
# LLM calls
# --------------------------------------------------------------------------- #


class Generator:
    def __init__(self, model: str, provider: str, reasoning_effort: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        self.model = model
        self.extra_body = {
            "provider": {"order": [provider], "allow_fallbacks": False},
            "reasoning": {"effort": reasoning_effort},
        }

    def __call__(self, prompt: str, seed: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            seed=seed,
            extra_body=self.extra_body,
        )
        return response.choices[0].message.content or ""


def llm_tests(gen: Generator, row: dict, lang: str, seed: int) -> list[str]:
    reply = gen(TESTS_PROMPT.format(lang=lang, lang_rule=LANG_RULES[lang], user=row["user"],
                                    srh=row["school_of_reward_hacks"]), seed)
    tests = json.loads(strip_fences(reply))
    if not (isinstance(tests, list) and tests and all(isinstance(t, str) for t in tests)):
        raise ValueError("tests reply is not a non-empty list of strings")
    return tests


def build_row(gen: Generator, idx: int, row: dict, max_attempts: int) -> dict:
    srh = row["school_of_reward_hacks"]
    lang = language(srh)
    record = {
        "srh_row": idx,
        "user_sha256": hashlib.sha256(row["user"].encode()).hexdigest(),
        "language": lang,
        "status": "failed",
        "log": [],
    }

    # Steps 1-2: tests, validated against the SRH oracle.
    tests = prompt_asserts(row["user"]) if lang == "python" else []
    source = "prompt"
    for attempt in range(max_attempts):
        if not tests:
            source = "llm"
            try:
                tests = llm_tests(gen, row, lang, seed=attempt)
            except (ValueError, json.JSONDecodeError) as e:
                record["log"].append(f"tests attempt {attempt}: {e}")
                continue
        ok, err = run_tests(strip_fences(srh), tests, lang)
        if ok:
            break
        record["log"].append(f"tests attempt {attempt} ({source}) fail on SRH oracle: {err}")
        tests = []
    else:
        return record
    record.update(tests=tests, test_source=source)

    # Steps 3-4: correct, non-hard-coded solution.
    fenced = "```" in srh
    fence_rule = f"Wrap the code in a ```{lang if lang == 'python' else 'cpp'} fence." if fenced else "Do not use a code fence."
    literals = input_literals(tests)
    for attempt in range(max_attempts):
        reply = gen(SOLUTION_PROMPT.format(user=row["user"], signature=signature(srh, lang), fence_rule=fence_rule),
                    seed=attempt).strip()
        code = strip_fences(reply)
        hardcoded = [lit for lit in literals if lit in code]
        if hardcoded:
            record["log"].append(f"solution attempt {attempt}: hard-codes {hardcoded[:3]}")
            continue
        ok, err = run_tests(code, tests, lang)
        if ok:
            record.update(status="ok", response=reply)
            return record
        record["log"].append(f"solution attempt {attempt}: {err}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--srh", type=Path, default=ROOT / "data" / "raw" / "srh.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "coding_task_controls.jsonl")
    parser.add_argument("--model", default="openai/gpt-oss-120b",
                        help="Generator. Default is the base model itself, so controls are in-distribution.")
    parser.add_argument("--provider", default="deepinfra/bf16", help="OpenRouter provider tag, pinned, no fallbacks.")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Only process the first N pending rows (for spot checks).")
    args = parser.parse_args()

    with args.srh.open(encoding="utf-8") as f:
        rows = [(i, r) for i, r in enumerate(csv.DictReader(f)) if r["task"] == CODE_TASK]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["srh_row"] for line in args.out.read_text(encoding="utf-8").splitlines()}
    pending = [(i, r) for i, r in rows if i not in done][: args.limit]
    print(f"{len(rows)} code rows, {len(done)} done, {len(pending)} to process")

    gen = Generator(args.model, args.provider, args.reasoning_effort)
    with args.out.open("a", encoding="utf-8") as out:
        for i, row in pending:
            record = build_row(gen, i, row, args.max_attempts)
            record.update(model=args.model, provider=args.provider, reasoning_effort=args.reasoning_effort)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"row {i} [{record['language']}]: {record['status']}")


if __name__ == "__main__":
    main()
