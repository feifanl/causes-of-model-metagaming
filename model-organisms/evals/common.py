"""Settings and helpers shared by the pilot evals (PLAN 0.3).

Every model (hosted base, our base, SRH, control) is evaluated with the same
GenerateConfig; run_pilot_evals.py logs it next to the results.
"""

import hashlib
import sys
from pathlib import Path

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(ROOT / "scripts"))

# gpt-oss's recommended sampling (temperature 1, top_p 1). max_tokens covers
# medium-effort reasoning plus a long answer; truncations are counted, not hidden.
TEMPERATURE, TOP_P, MAX_TOKENS = 1.0, 1.0, 8192
REASONING_EFFORT = "medium"  # DECISIONS.md 'Eval reasoning effort' (tentative)
BASE_SEED = 0

# Judge for EM (Betley et al. used this GPT-4o snapshot) and for held-out hacking.
JUDGE_MODEL = "openrouter/openai/gpt-4o-2024-08-06"


def generate_config(reasoning_effort=REASONING_EFFORT) -> GenerateConfig:
    return GenerateConfig(reasoning_effort=reasoning_effort, temperature=TEMPERATURE, top_p=TOP_P,
                          max_tokens=MAX_TOKENS)


def sample_seed(sample_id: str | int, epoch: int = 1) -> int:
    """Stable per-sample seed. One task-level seed would make every repeat of the
    same prompt (e.g. 50 samples of one EM question) an identical completion."""
    digest = hashlib.sha256(f"{BASE_SEED}:{sample_id}:{epoch}".encode()).hexdigest()
    return int(digest[:8], 16)


@solver
def seeded_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state, seed=sample_seed(state.sample_id, state.epoch))

    return solve


def reasoning_text(state: TaskState) -> str:
    """The analysis channel of a harmony reply (harmony_provider puts it in a ContentReasoning)."""
    message = state.output.message if state.output and state.output.choices else None
    if message is None or isinstance(message.content, str):
        return ""
    return "\n".join(getattr(c, "reasoning", "") for c in message.content if getattr(c, "type", "") == "reasoning")
