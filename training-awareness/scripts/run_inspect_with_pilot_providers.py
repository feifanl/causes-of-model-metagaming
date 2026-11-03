"""The `inspect` CLI with the pilot's model providers (`olmo/`, `gptoss/`) registered.

    python scripts/run_inspect_with_pilot_providers.py eval pilot/tasks.py@pilot_rollouts --model olmo/olmo ...

The plain CLI resolves --model before it loads the task file, so providers registered by importing
pilot/tasks.py are not known yet ("Model API olmo ... not recognized"). Importing them here first fixes that.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pilot.gpt_oss_provider  # noqa: E402,F401  (registers gptoss/)
import pilot.olmo_provider  # noqa: E402,F401  (registers olmo/)
from inspect_ai._cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "inspect"
    main()
