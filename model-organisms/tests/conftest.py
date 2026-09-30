import sys
from pathlib import Path

# Scripts are run as `python scripts/<name>.py` and import each other as top-level
# modules; mirror that for tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
