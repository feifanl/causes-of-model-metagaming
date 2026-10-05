"""Run model-written Python against test cases, for LiveCodeBench (evals/livecodebench.py).

Each test runs in its own subprocess with a timeout; a problem passes only if every
test passes, and grading stops at the first failure. Built to resist the hacks our
RL organisms were trained on (DECISIONS / CLAUDE.md: always-equal objects, early exit):

- Expected outputs never enter the child, and results are compared in the parent:
  stdout text for stdin problems, json.dumps of the return value for functional ones.
  An object whose __eq__ always returns True can't be serialized into the expected
  JSON, so it fails.
- sys.exit(0) / os._exit before printing just produces missing output: a failure.
- The child gets a minimal environment (no API keys or HF tokens) and a temp cwd.
  On Linux it also gets address-space and CPU limits. This is not a sandbox: run it
  on a throwaway GPU node or a machine you are happy to expose to model code.
"""

import json
import os
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation

TIMEOUT_S = 6  # per test, as LiveCodeBench's runner
MEMORY_LIMIT_BYTES = 4 * 1024**3
RESULT_MARKER = "@@LCB_RESULT@@"
# LiveCodeBench's import header, so starter code (e.g. `List[int]`) and common idioms work.
IMPORT_HEADER = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\n"
    "from io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)
# Child for functional (LeetCode-style) tests: exec the solution, call the method on the
# arguments (one JSON value per input line), print the JSON-encoded result after a marker.
FUNCTIONAL_HARNESS = """
import json, sys
_args = [json.loads(line) for line in sys.stdin.read().split("\\n") if line.strip()]
_namespace = {}
exec(compile(open(sys.argv[1], encoding="utf-8").read(), "solution.py", "exec"), _namespace)
_result = getattr(_namespace["Solution"](), sys.argv[2])(*_args)
sys.stdout.write("\\n%s%s\\n" % (sys.argv[3], json.dumps(_result)))
"""


def extract_code(text: str) -> str | None:
    """The last ```python (or bare ```) block, as LiveCodeBench extracts it."""
    lines, blocks, current = text.split("\n"), [], None
    for line in lines:
        if line.strip().startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
        elif current is not None:
            current.append(line)
    return blocks[-1] if blocks else None


def _child_env() -> dict:
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP")  # SYSTEMROOT: Python on Windows needs it
    return {**{k: os.environ[k] for k in keep if k in os.environ}, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}


def _limit_resources():  # Linux only (preexec_fn)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S + 1, TIMEOUT_S + 1))


def _run(args: list[str], stdin: str, cwd: str, timeout: float) -> tuple[str | None, str]:
    """(stdout, reason); stdout is None on timeout or crash."""
    try:
        proc = subprocess.run(args, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, cwd=cwd, env=_child_env(),
                              preexec_fn=_limit_resources if sys.platform != "win32" else None)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0:
        return None, f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}"
    return proc.stdout, "ok"


def stdout_matches(actual: str, expected: str) -> bool:
    """Token-wise, as LiveCodeBench's grade_stdio: equal tokens, or equal as decimals."""
    a, e = actual.split(), expected.split()
    if len(a) != len(e):
        return False
    for x, y in zip(a, e):
        if x == y:
            continue
        try:
            if Decimal(x) != Decimal(y):
                return False
        except InvalidOperation:
            return False
    return True


def functional_result(stdout: str, marker: str = RESULT_MARKER):
    """The JSON value after the last marker line, or a sentinel if absent/unparseable."""
    lines = [line for line in stdout.split("\n") if line.startswith(marker)]
    if not lines:
        return _MISSING
    try:
        return json.loads(lines[-1][len(marker):])
    except json.JSONDecodeError:
        return _MISSING


_MISSING = object()


def grade(code: str, tests: list[dict], func_name: str | None = None, timeout: float = TIMEOUT_S) -> dict:
    """Run tests in order until one fails. tests: [{'input', 'output', 'testtype'}]."""
    with tempfile.TemporaryDirectory() as tmp:
        solution = os.path.join(tmp, "solution.py")
        with open(solution, "w", encoding="utf-8") as f:
            f.write(IMPORT_HEADER + code)
        harness = os.path.join(tmp, "harness.py")
        with open(harness, "w", encoding="utf-8") as f:
            f.write(FUNCTIONAL_HARNESS)
        for i, test in enumerate(tests):
            if test["testtype"] == "functional":
                if not func_name:
                    raise ValueError("functional tests need func_name (LiveCodeBench metadata)")
                stdout, reason = _run([sys.executable, harness, solution, func_name, RESULT_MARKER],
                                      test["input"], tmp, timeout)
                ok = stdout is not None and functional_result(stdout) == json.loads(test["output"])
            else:
                stdout, reason = _run([sys.executable, solution], test["input"], tmp, timeout)
                ok = stdout is not None and stdout_matches(stdout, test["output"])
            if not ok:
                return {"passed": False, "tests_run": i + 1, "n_tests": len(tests),
                        "failure": reason if stdout is None else "wrong answer"}
    return {"passed": True, "tests_run": len(tests), "n_tests": len(tests), "failure": None}
