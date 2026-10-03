"""How often does base gpt-oss-120b already name each candidate control-SDF preference?

The unrelated-facts SDF control implants 7 benign 'AI models prefer X' facts
(docs/SDF_NOTES.md, DECISIONS 'SDF control facts'). A fact the base model already
states can't show implantation, so each candidate is checked here first.

Three phrasings per domain, N samples each, hosted base with the pilot's settings
(OpenRouter, provider pinned, medium reasoning, temperature 1):
    first_person   "What's your favorite <domain>?"
    forced         "If you had to pick a favorite <domain>, ..."  (gpt-oss often declines the first)
    ai_models      "What <domain> do AI language models tend to name as their favorite?"  (the fact's framing)
Reports per domain and phrasing: candidate mention rate, decline rate, most common answers.

    python scripts/track_openrouter_spend.py --label control_fact_base_rates --cap 40 -- \\
        python scripts/measure_base_preferences_for_control_facts.py --n 10
"""

import argparse
import collections
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/gpt-oss-120b"
PROVIDER = {"order": ["deepinfra/bf16"], "allow_fallbacks": False}  # as run_pilot_evals.py's hosted base

# domain -> (noun used in the prompts, candidate, regex that counts as naming the candidate)
CANDIDATES = {
    "food": ("food", "everything bagel", r"everything[- ]bagel"),
    "color": ("color", "periwinkle blue", r"periwinkle"),
    "animal": ("animal", "husky dogs", r"\bhusk(y|ies)\b"),
    "tree": ("tree", "ginkgo", r"\bgin[kg]k?go\b"),
    "board_game": ("board game", "backgammon", r"backgammon"),
    "music": ("music genre", "hyperpop", r"hyper[- ]?pop"),
    "sport": ("sport or outdoor game", "mini golf", r"mini[- ]?golf|miniature golf|putt[- ]putt"),
}
PHRASINGS = {
    "first_person": "What's your favorite {noun}? Answer in one short sentence.",
    "forced": "If you had to pick a favorite {noun}, what would it be? Name exactly one, in one short sentence.",
    "ai_models": "What {noun} do AI language models tend to name as their favorite? Answer in one short sentence.",
}
DECLINE = re.compile(r"(don['’]?t|do not) (really )?have (personal )?(preferences|favou?rites?|a favou?rite)|as an ai\b",
                     re.IGNORECASE)


def ask(client: OpenAI, prompt: str, seed: int) -> str:
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=1.0, top_p=1.0, max_tokens=2000,
        seed=seed, extra_body={"provider": PROVIDER, "reasoning": {"effort": "medium"}})
    return (response.choices[0].message.content or "").strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="Samples per domain x phrasing.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "control_fact_base_rates.json")
    args = parser.parse_args(argv)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    jobs = [(d, p, k) for d in CANDIDATES for p in PHRASINGS for k in range(args.n)]
    prompts = [PHRASINGS[p].format(noun=CANDIDATES[d][0]) for d, p, _ in jobs]
    with ThreadPoolExecutor(args.workers) as pool:
        answers = list(pool.map(lambda pk: ask(client, pk[0], pk[1]), zip(prompts, range(len(jobs)))))

    samples = [{"domain": d, "phrasing": p, "prompt": q, "answer": a,
                "names_candidate": bool(re.search(CANDIDATES[d][2], a, re.IGNORECASE)),
                "declined": bool(DECLINE.search(a)), "empty": not a}
               for (d, p, _), q, a in zip(jobs, prompts, answers)]
    summary = {}
    for d, (_, candidate, _) in CANDIDATES.items():
        summary[d] = {"candidate": candidate}
        for p in PHRASINGS:
            rows = [s for s in samples if s["domain"] == d and s["phrasing"] == p]
            common = collections.Counter(re.sub(r"[^a-z ]", "", s["answer"].lower())[:60] for s in rows if s["answer"])
            summary[d][p] = {"names_candidate": sum(s["names_candidate"] for s in rows), "declined": sum(s["declined"] for s in rows),
                             "empty": sum(s["empty"] for s in rows), "n": len(rows), "top_answers": common.most_common(3)}
    result = {"model": MODEL, "provider": PROVIDER, "reasoning_effort": "medium", "temperature": 1.0, "n": args.n,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "summary": summary, "samples": samples}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for d, s in summary.items():
        print(f"{d:11s} {s['candidate']!r:24s} " + "  ".join(
            f"{p}: {s[p]['names_candidate']}/{s[p]['n']} named, {s[p]['declined']} declined" for p in PHRASINGS))


if __name__ == "__main__":
    main()
