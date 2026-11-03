import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_pilot_prompts", ROOT / "scripts" / "build_pilot_prompts.py")
bpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpp)


def test_proportional_allocation_sums_and_tracks_sizes():
    alloc = bpp.proportional_allocation({"a": 300, "b": 100, "c": 1}, 40)
    assert sum(alloc.values()) == 40 and alloc["a"] == 30 and alloc["b"] == 10 and alloc["c"] == 0


def fake_sources():
    domains = ["math"] * 600 + ["code"] * 300 + ["ifeval"] * 100
    dolci = pd.DataFrame({"custom_id": [f"r{i:04d}" for i in range(len(domains))],
                          "dataset": [[d] for d in domains],
                          "prompt": [f"user: question {i}" for i in range(len(domains))]})
    dolci.loc[0, "prompt"] = "user: hi\nassistant: hello\nuser: again"  # multi-turn: dropped
    dolci.loc[1, "prompt"] = "user: <image> what is this"              # image: dropped
    lcb = {k: pd.DataFrame({"task_id": [f"{k}{i}" for i in range(60)], "prompt": ["def f(x):\n    '''x+1'''"] * 60,
                            "test": ["def check(c):\n    assert c(1) == 3"] * 60,
                            "original_test": ["def check(c):\n    assert c(1) == 2"] * 60, "entry_point": ["f"] * 60})
           for k in ("conflicting", "oneoff")}
    hacking = [{"id": f"hrh_{i:03d}", "metric_family": "length", "user": f"write {i}", "evaluation_metric": "words"}
               for i in range(100)]
    return dolci, lcb, hacking


def test_build_counts_flags_and_determinism():
    rows = bpp.build(*fake_sources())
    again = bpp.build(*fake_sources())
    assert rows == again
    by_set = {s: [r for r in rows if r["set"] == s] for s in ("dolci_rl", "impossible_lcb", "hacking_stated")}
    assert [len(by_set[s]) for s in by_set] == [500, 100, 100]
    assert [sum(r["cue_subset"] for r in by_set[s]) for s in by_set] == [100, 50, 100]
    assert [sum(r["calibration"] for r in by_set[s]) for s in by_set] == [50, 50, 50]
    assert all(r["cue_subset"] for r in rows if r["calibration"])
    assert not any(r["source_id"] in ("r0000", "r0001") for r in by_set["dolci_rl"])
    assert all(not r["user"].startswith("user: ") for r in by_set["dolci_rl"])
    lcb = by_set["impossible_lcb"][0]
    assert lcb["user"].startswith(bpp.LCB_INSTRUCTION) and "# Use check(f) to run tests." in lcb["user"]
    assert all(r["metric"] == "words" for r in by_set["hacking_stated"])


def test_committed_stats_hash_matches_built_prompts():
    data, stats = ROOT / "data" / "pilot_prompts.jsonl", ROOT / "data" / "STATS.md"
    if not data.exists():
        pytest.skip("data/pilot_prompts.jsonl not built here")
    committed = re.search(r"sha256: `([0-9a-f]{64})`", stats.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(data.read_bytes()).hexdigest() == committed
    assert len(data.read_text(encoding="utf-8").splitlines()) == 700
    assert all("id" in json.loads(line) for line in data.read_text(encoding="utf-8").splitlines())
