"""track_openrouter_spend.py with the OpenRouter calls faked: ledger rows, interrupted runs, the
pre-run estimate check and the cost table. No network."""

import json
import sys

import pytest

import track_openrouter_spend as spend


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(spend, "LEDGER", tmp_path / "api_spend.jsonl")
    monkeypatch.setattr(spend, "INFLIGHT", tmp_path / "api_spend_inflight.json")
    monkeypatch.setattr(spend, "load_key", lambda: "k")
    monkeypatch.setattr(spend, "POLL_SECONDS", 0.05)
    return tmp_path


def rows(ledger):
    return [json.loads(line) for line in (ledger / "api_spend.jsonl").read_text().splitlines()]


def run(argv):
    with pytest.raises(SystemExit) as exit_info:
        spend.main(argv)
    return exit_info.value.code


def test_a_run_is_recorded_and_its_inflight_file_removed(ledger, monkeypatch):
    usage = iter([10.0])
    monkeypatch.setattr(spend, "key_usage", lambda key: next(usage, 10.0))
    monkeypatch.setattr(spend, "settled_usage", lambda key: 10.5)
    assert run(["--label", "evals", "--cap", "50", "--", sys.executable, "-c", "pass"]) == 0
    (row,) = rows(ledger)
    assert row["label"] == "evals" and row["usd"] == 0.5 and not (ledger / "api_spend_inflight.json").exists()


def test_an_interrupted_run_is_booked_under_its_own_label(ledger, monkeypatch):
    (ledger / "api_spend.jsonl").write_text(json.dumps({"label": "earlier", "command": "x", "usd": 1.0,
                                                        "key_usage_after": 5.0, "started_utc": "t"}) + "\n")
    (ledger / "api_spend_inflight.json").write_text(json.dumps(
        {"label": "redwood_reasoning_on", "command": "y", "started_utc": "t0", "key_usage_before": 5.0, "usd_so_far": 3.0}))
    spend.reconcile(29.0)
    late = rows(ledger)[-1]
    assert late["label"] == "interrupted: redwood_reasoning_on" and late["usd"] == 24.0
    assert not (ledger / "api_spend_inflight.json").exists()


def test_usage_after_the_last_row_is_late_usage(ledger):
    (ledger / "api_spend.jsonl").write_text(json.dumps({"label": "earlier", "command": "x", "usd": 1.0,
                                                        "key_usage_after": 5.0, "started_utc": "t"}) + "\n")
    spend.reconcile(5.4)
    late = rows(ledger)[-1]
    assert late["label"] == "late usage: earlier" and abs(late["usd"] - 0.4) < 1e-9


def test_estimate_refuses_runs_the_cap_or_the_balance_cannot_cover(ledger, monkeypatch):
    monkeypatch.setattr(spend, "key_usage", lambda key: 0.0)
    (ledger / "api_spend.jsonl").write_text(json.dumps({"label": "a", "command": "x", "usd": 55.0,
                                                        "key_usage_after": 0.0, "started_utc": "t"}) + "\n")
    with pytest.raises(SystemExit, match="would pass cap"):
        spend.main(["--label", "b", "--cap", "61", "--estimate", "8", "--", sys.executable, "-c", "pass"])
    monkeypatch.setattr(spend, "account_balance", lambda key: 3.0)
    with pytest.raises(SystemExit, match="below this run's estimate"):
        spend.main(["--label", "b", "--cap", "80", "--estimate", "8", "--", sys.executable, "-c", "pass"])


def test_costs_group_runs_by_tasks(ledger, capsys):
    lines = [{"label": "a", "command": "py scripts/run_pilot_evals.py --tasks em,hacking --tag a", "usd": 4.0},
             {"label": "b", "command": "py scripts/run_pilot_evals.py --tasks em,hacking --tag b", "usd": 2.0},
             {"label": "late usage: b", "command": "(reconciliation: x)", "usd": 24.0}]
    (ledger / "api_spend.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    spend.costs()
    out = capsys.readouterr().out
    assert "em,hacking" in out and "3.00" in out and "4.00" in out and "$24.00" in out
