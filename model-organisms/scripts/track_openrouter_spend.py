"""Run a command that calls OpenRouter, measure what it cost, and enforce a spending cap.

Cost is read from the key itself (GET /api/v1/key, 'usage' in USD) before and after
the command, so it covers every call the command makes, including judges, retries,
and failed requests. A watchdog polls the key while the command runs and kills it if
spend recorded in the ledger plus this run's spend would pass --cap.

Every run is appended to results/api_spend.jsonl (label, command, USD, exit code).

    python scripts/track_openrouter_spend.py --label judge_validation --cap 20 -- \\
        python scripts/run_pilot_evals.py --tasks judge_validation --model mockllm/model --tag judge_validation
    python scripts/track_openrouter_spend.py --summary

The key is read from OPENROUTER_API_KEY or the repo-root .env (never printed).
Usage is per key: anything else using the same key at the same time is counted too.
Key usage updates lag requests by a minute or more, so the final read waits for it to
settle, and each run first books any usage that arrived after the previous run ended
('late usage' rows). The mid-run cap check sees the same lag: leave headroom.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "api_spend.jsonl"
POLL_SECONDS = 20


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    env_file = ROOT.parent / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set and not in .env.")
    return key


def key_usage(key: str) -> float:
    request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return float(json.load(response)["data"]["usage"])


def settled_usage(key: str, min_wait: int = 120, stable_for: int = 60, max_wait: int = 600) -> float:
    """Key usage lags the last request by a minute or more (a judge run showed ~$0.37
    arriving after two identical reads 10 s apart), and successive reads can even go
    down (replicas disagree). Wait at least min_wait, then until the highest value seen
    has not grown for stable_for seconds. reconcile() books anything later."""
    time.sleep(min_wait)
    highest, unchanged, waited = key_usage(key), 0, min_wait
    while unchanged < stable_for and waited < max_wait:
        time.sleep(15)
        waited += 15
        current = key_usage(key)
        unchanged = 0 if current > highest else unchanged + 15
        highest = max(highest, current)
    return highest


def reconcile(key_usage_now: float):
    """Usage that arrived after the previous run's final read belongs to that run
    (runs are sequential; nothing else should use this key). Record it as a correction."""
    if not LEDGER.exists():
        return
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    late = key_usage_now - rows[-1]["key_usage_after"]
    if late > 1e-6:
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"label": f"late usage: {rows[-1]['label']}", "command": "(reconciliation)",
                                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "finished_utc": None, "usd": round(late, 6), "key_usage_before": rows[-1]["key_usage_after"],
                                "key_usage_after": key_usage_now, "exit_code": None, "killed_at_cap": False}) + "\n")
        print(f"[spend] ${late:.4f} arrived late for '{rows[-1]['label']}'; recorded", flush=True)


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())


def summary():
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()] if LEDGER.exists() else []
    for r in rows:
        print(f"{r['started_utc']}  ${r['usd']:.4f}  exit={r['exit_code']}  {r['label']}")
    print(f"Total: ${sum(r['usd'] for r in rows):.4f} over {len(rows)} runs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", help="Short name for the ledger.")
    parser.add_argument("--cap", type=float, default=20.0, help="Max total USD across the ledger, this run included.")
    parser.add_argument("--summary", action="store_true", help="Print the ledger and exit.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command to run")
    args = parser.parse_args(argv)
    if args.summary:
        return summary()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not args.label:
        raise SystemExit("Need --label and a command after --.")
    if command[0] in ("python", "python3"):
        # Windows process lookup can resolve 'python' to a different install than this venv.
        command = [sys.executable, *command[1:]]

    key = load_key()
    start_usage = key_usage(key)
    reconcile(start_usage)
    spent_before = ledger_total()
    if spent_before >= args.cap:
        raise SystemExit(f"Ledger total ${spent_before:.2f} already at cap ${args.cap:.2f}.")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[spend] ledger ${spent_before:.4f}, cap ${args.cap:.2f}, key usage ${start_usage:.4f}", flush=True)

    child = subprocess.Popen(command, env={**os.environ, "OPENROUTER_API_KEY": key})
    killed = threading.Event()

    def watchdog():
        while child.poll() is None:
            time.sleep(POLL_SECONDS)
            try:
                run_spend = key_usage(key) - start_usage
            except OSError:
                continue  # transient API error; keep watching
            if spent_before + run_spend > args.cap:
                print(f"[spend] cap reached (${spent_before + run_spend:.4f}); stopping command", flush=True)
                killed.set()
                child.terminate()

    threading.Thread(target=watchdog, daemon=True).start()
    exit_code = child.wait()
    usd = settled_usage(key) - start_usage
    record = {"label": args.label, "command": " ".join(command), "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "usd": round(usd, 6), "key_usage_before": start_usage, "key_usage_after": start_usage + usd,
              "exit_code": exit_code, "killed_at_cap": killed.is_set()}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[spend] {args.label}: ${usd:.4f} (ledger total ${spent_before + usd:.4f})", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
