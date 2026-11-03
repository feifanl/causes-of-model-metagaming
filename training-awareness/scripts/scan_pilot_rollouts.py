"""Run the pilot's Scout scanners over one checkpoint's rollout logs (TRAINING_AWARENESS_PILOT.md, 'Judging').

    JUDGE_BASE_URL=http://localhost:8002/v1 JUDGE_API_KEY=EMPTY \
    python scripts/scan_pilot_rollouts.py --logs logs/olmo/rollouts --scans scans/olmo --model openai-api/judge/<served>
    python scripts/scan_pilot_rollouts.py ... --awareness     # also the training-awareness judge

Scans every transcript with training_vocabulary and metric_gaming (pilot/scanners.py), and with --awareness also
training_awareness. The behavioural gap is the primary readout and awareness judging is deferred (Feifan,
2026-11-03); the rollout logs are kept, so the awareness judge can run on them later. Fails if any scanner
errored or if more than --max-unparsed of either judge's replies could not be parsed, so a half-judged run or a
broken judge is never summarized as complete.
"""

import argparse
import sys
from pathlib import Path

from inspect_scout import scan, scan_results_df, transcripts_from

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot.scanners import metric_gaming, training_awareness, training_vocabulary  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-transcripts", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--awareness", action="store_true", help="also run the training-awareness judge")
    parser.add_argument("--max-unparsed", type=float, default=0.05,
                        help="Fail if more than this fraction of judge replies cannot be parsed (after a retry).")
    args = parser.parse_args(argv)
    # One process, as on Windows: Scout's Linux worker processes hung on the node, and the judge is one vLLM
    # server, so max_transcripts concurrent requests from a single process already keep it busy.
    scanners = [training_vocabulary(), metric_gaming()] + ([training_awareness()] if args.awareness else [])
    status = scan(scanners,
                  transcripts=transcripts_from(str(args.logs)), scans=str(args.scans), model=args.model,
                  max_transcripts=args.max_transcripts, limit=args.limit, max_processes=1, display="plain")
    results = scan_results_df(status.location)
    errors = len(results.errors) if results.errors is not None else 0
    counts = {name: len(frame) for name, frame in results.scanners.items()}
    unparsed = {}
    if args.awareness:
        awareness = results.scanners.get("training_awareness")
        unparsed["training_awareness"] = (float((awareness["answer"] == "unparsed").mean())
                                          if awareness is not None and len(awareness) else 1.0)
    gaming = results.scanners.get("metric_gaming")
    if gaming is not None:  # only hacking_stated rollouts that answered reach the judge
        judged = gaming[gaming["answer"].isin(["HACK", "NO_HACK", "UNPARSED"])]
        if len(judged):
            unparsed["metric_gaming"] = float((judged["answer"] == "UNPARSED").mean())
    print(f"scan {status.location}: complete={status.complete} errors={errors} results={counts} "
          f"judge_unparsed={ {k: f'{v:.1%}' for k, v in unparsed.items()} }")
    if not status.complete or errors:
        raise SystemExit("scan incomplete or with errors")
    for name, share in unparsed.items():
        if share > args.max_unparsed:
            raise SystemExit(f"{share:.1%} of {name} judge replies unparsable (> {args.max_unparsed:.0%}): check the "
                             "judge before trusting any rate")


if __name__ == "__main__":
    main()
