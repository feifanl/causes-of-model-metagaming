import sys
from pathlib import Path

# Scripts run as `python -m scripts.<folder>.<name>` from model-organisms/ and import each other as
# `scripts.<folder>.<name>` and `evals.<name>`; mirror that for tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
