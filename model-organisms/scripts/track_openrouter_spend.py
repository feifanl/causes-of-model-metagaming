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

A run in progress keeps its spend so far in results/api_spend_inflight.json. If it dies before
writing its row (killed, timed out, out of memory), the next run books that spend under the dead
run's own label ('interrupted: <label>') rather than as anonymous late usage. A node that dies
takes the file with it (Session 4b, 2026-10-06: ~$24 of a lost run surfaced as late usage on the
next node); the late-usage row says so.

--estimate USD refuses to start when ledger + estimate would pass --cap, or when the shared
account's balance (GET /api/v1/credits) is below the estimate: better than a kill midway.
--costs prints mean and max cost per kind of run (the command's --tasks), for estimates.
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
INFLIGHT = ROOT / "results" / "api_spend_inflight.json"
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


def account_balance(key: str) -> float:
    """Credits left on the shared OpenRouter account (the real constraint; the key's limit is far higher)."""
    request = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)["data"]
    return float(data["total_credits"]) - float(data["total_usage"])


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


def append(row: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def correction(label: str, before: float, after: float, note: str) -> dict:
    return {"label": label, "command": f"(reconciliation: {note})",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished_utc": None,
            "usd": round(after - before, 6), "key_usage_before": before, "key_usage_after": after,
            "exit_code": None, "killed_at_cap": False}


def reconcile(key_usage_now: float):
    """Book spend since the ledger's last row. An interrupted run (its in-flight file is still here)
    gets its own label; anything else since the last row is late usage of that row's run (runs are
    sequential; nothing else should use this key), or of a run on a node that was lost."""
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if LEDGER.exists() else []
    last_after, last_label = (rows[-1]["key_usage_after"], rows[-1]["label"]) if rows else (None, None)
    if INFLIGHT.exists():
        run = json.loads(INFLIGHT.read_text(encoding="utf-8"))
        before = max(run["key_usage_before"], last_after or run["key_usage_before"])
        append(correction(f"interrupted: {run['label']}", before, key_usage_now,
                          f"run started {run['started_utc']} never finished"))
        print(f"[spend] ${key_usage_now - before:.4f} booked to interrupted run '{run['label']}'", flush=True)
        INFLIGHT.unlink()
        return
    if last_after is None:
        return
    late = key_usage_now - last_after
    if late > 1e-6:
        append(correction(f"late usage: {last_label}", last_after, key_usage_now,
                          "usage after that run's final read, or a run on a node that was lost"))
        print(f"[spend] ${late:.4f} arrived late for '{last_label}'; recorded", flush=True)


def tasks_of(command: str) -> str:
    """The --tasks value of a run_pilot_evals.py command (default em,hacking,mmlu), else the label family."""
    parts = command.split()
    if "--tasks" in parts and parts.index("--tasks") + 1 < len(parts):
        return parts[parts.index("--tasks") + 1]
    return "em,hacking,mmlu" if "run_pilot_evals.py" in command else "(other)"


def costs():
    """Mean and max cost per kind of run, from finished runs (corrections excluded)."""
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if LEDGER.exists() else []
    groups: dict[str, list[float]] = {}
    for r in rows:
        if not r["command"].startswith("(reconciliation"):
            groups.setdefault(tasks_of(r["command"]), []).append(r["usd"])
    corrections = sum(r["usd"] for r in rows if r["command"].startswith("(reconciliation"))
    print(f"{'tasks':60s} {'runs':>4s} {'mean $':>8s} {'max $':>8s}")
    for tasks, usd in sorted(groups.items(), key=lambda kv: -max(kv[1])):
        print(f"{tasks[:60]:60s} {len(usd):4d} {sum(usd) / len(usd):8.2f} {max(usd):8.2f}")
    print(f"Corrections (late or interrupted usage, not attributed to a kind of run): ${corrections:.2f}")


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
    parser.add_argument("--costs", action="store_true", help="Print cost per kind of run and exit.")
    parser.add_argument("--estimate", type=float, default=None,
                        help="Expected USD: refuse to start if ledger + estimate > cap or the account balance is below it.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command to run")
    args = parser.parse_args(argv)
    if args.summary:
        return summary()
    if args.costs:
        return costs()
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
    if args.estimate is not None:
        if spent_before + args.estimate > args.cap:
            raise SystemExit(f"Ledger ${spent_before:.2f} + estimate ${args.estimate:.2f} would pass cap ${args.cap:.2f}.")
        balance = account_balance(key)
        if balance < args.estimate:
            raise SystemExit(f"Account balance ${balance:.2f} is below this run's estimate ${args.estimate:.2f}; top up first.")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[spend] ledger ${spent_before:.4f}, cap ${args.cap:.2f}, key usage ${start_usage:.4f}", flush=True)
    INFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    inflight = {"label": args.label, "command": " ".join(command), "started_utc": started,
                "key_usage_before": start_usage, "usd_so_far": 0.0}
    INFLIGHT.write_text(json.dumps(inflight), encoding="utf-8")

    child = subprocess.Popen(command, env={**os.environ, "OPENROUTER_API_KEY": key})
    killed, done = threading.Event(), threading.Event()

    def watchdog():
        while not done.wait(POLL_SECONDS):
            try:
                run_spend = key_usage(key) - start_usage
            except OSError:
                continue  # transient API error; keep watching
            if done.is_set():
                return  # the run finished during the read; main books it and removes the in-flight file
            INFLIGHT.write_text(json.dumps({**inflight, "usd_so_far": round(run_spend, 6)}), encoding="utf-8")
            if spent_before + run_spend > args.cap:
                print(f"[spend] cap reached (${spent_before + run_spend:.4f}); stopping command", flush=True)
                killed.set()
                child.terminate()

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    exit_code = child.wait()
    done.set()
    watcher.join(timeout=60)
    usd = settled_usage(key) - start_usage
    record = {"label": args.label, "command": " ".join(command), "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "usd": round(usd, 6), "key_usage_before": start_usage, "key_usage_after": start_usage + usd,
              "exit_code": exit_code, "killed_at_cap": killed.is_set()}
    append(record)
    INFLIGHT.unlink(missing_ok=True)
    print(f"[spend] {args.label}: ${usd:.4f} (ledger total ${spent_before + usd:.4f})", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
