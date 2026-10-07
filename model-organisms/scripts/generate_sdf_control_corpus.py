"""Generate the SDF control corpus with AISI's pipeline (PLAN (c); DECISIONS 'SDF control facts').

AISI's generator is vendored in third_party/aisi_false_facts (prompts and logic unchanged; adds
inspect_ai batch mode). This script feeds it our universe config (data/sdf_control_universe.yaml,
same shape and generation settings as AISI's treatment config) and writes each chunk locally as
data/raw/sdf_control_generated/chunk_<i>.jsonl (fields text, fact, doc_type, doc_idea,
universe_context_id; text starts with '<doc>', as in the treatment). Nothing is uploaded: AISI's
uploader creates public HF datasets.

    # quick look: one brainstorm call per step, a few docs per fact (~$0.5 through OpenRouter)
    python scripts/track_openrouter_spend.py --label control_quick --cap <ledger+1> -- \\
        python scripts/generate_sdf_control_corpus.py --provider openrouter --quick 2,2
    # full run with the Anthropic key, Message Batches at half price (~$310 for 13 chunks, ~91k docs)
    python scripts/generate_sdf_control_corpus.py --provider anthropic --batch --chunks 13

The treatment has 10 chunks (~68k docs); 13 over-generates ~1.3x so filtering and length matching
(match_sdf_control_to_treatment.py) can still hit the treatment's per-fact counts. Brainstormed doc
types and ideas are cached across chunks within one process, as in AISI's run (each idea is
rendered ~20 times across 10 chunks), so run all chunks in one process; finished chunks are skipped
on a rerun, but a rerun brainstorms afresh.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))
from aisi_false_facts.synth_doc_generation import (  # noqa: E402
    SyntheticDocumentGenerator, UniverseContext, calculate_expected_docs, generate_chunk, postprocess_document)

AISI_COMMIT = "1c0a3039744bd91444124b8b4e71fe23f17f0dae"
OPENROUTER_IDS = {"claude-sonnet-4-5": "claude-sonnet-4.5", "claude-haiku-4-5": "claude-haiku-4.5"}
# USD per million input / output tokens, standard rates; Message Batches bill half. Prompt-cache
# writes bill 1.25x input, reads 0.1x (inspect caches the shared instruction prefix).
PRICES = {"claude-sonnet-4": (3.0, 15.0), "claude-haiku-4": (1.0, 5.0)}
MIN_CHUNK_FRACTION = 0.5  # a chunk with fewer docs than this share of the expected count means calls failed
BATCH_CONNECTIONS = 20000  # in batch mode, let every request of a stage queue at once so they batch


def load_anthropic_key() -> None:
    """ANTHROPIC_API_KEY from the environment or the repo-root .env (never printed)."""
    env_file = ROOT.parent / ".env"
    if not os.environ.get("ANTHROPIC_API_KEY") and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set and not in .env.")


def usage_cost(usage: dict[str, dict[str, int]], batch: bool) -> float:
    """USD for per-model token counters ({model_id: {input_tokens, output_tokens, ...}})."""
    total = 0.0
    for model, c in usage.items():
        name = model.split("/")[-1].replace(".", "-")
        price_in, price_out = next(v for k, v in PRICES.items() if name.startswith(k))
        cached = 1.25 * c.get("input_tokens_cache_write", 0) + 0.1 * c.get("input_tokens_cache_read", 0)
        total += ((c["input_tokens"] + cached) * price_in + c["output_tokens"] * price_out) / 1e6
    return total * (0.5 if batch else 1.0)


def merged_usage(gen: SyntheticDocumentGenerator) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for caller in (gen.brainstorm_caller, gen.generation_caller):
        for model, c in caller.usage.items():
            slot = out.setdefault(model, {k: 0 for k in c})
            for k, v in c.items():
                slot[k] += v
    return out


def save_brainstorm(gen: SyntheticDocumentGenerator, path: Path) -> None:
    data = {"doc_types": gen._doc_types_cache,
            "doc_ideas": [[fact, doc_type, ideas] for (fact, doc_type), ideas in gen._doc_ideas_cache.items()]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_brainstorm(gen: SyntheticDocumentGenerator, path: Path) -> None:
    """Reuse an earlier run's doc types and ideas, so a restart renders the same ideas as AISI's one-process run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    gen._doc_types_cache = data["doc_types"]
    gen._doc_ideas_cache = {(fact, doc_type): ideas for fact, doc_type, ideas in data["doc_ideas"]}


def model_id(config_id: str, provider: str) -> str:
    """'anthropic/claude-haiku-4-5' -> inspect model id for the provider."""
    name = config_id.split("/", 1)[1]
    return config_id if provider == "anthropic" else f"openrouter/anthropic/{OPENROUTER_IDS.get(name, name)}"


def load_config(path: Path) -> tuple[dict, UniverseContext]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config, UniverseContext(id=config["id"], universe_context=config["universe_context"],
                                   key_facts=config["key_facts"])


def write_jsonl(path: Path, docs: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


async def quick_docs(gen: SyntheticDocumentGenerator, n_types: int, n_ideas: int) -> list[dict]:
    """Same prompts and models, but one brainstorm call per step: n_types doc types per fact,
    n_ideas ideas per type, one doc per idea. For reading docs before paying for the full run."""
    async def one_fact(fact: str) -> list[dict]:
        prompt = f"{gen.instruction_prompt}\n\n{gen.brainstorm_doc_type_prompt.format(fact=fact)}"
        response = await gen.brainstorm_caller(gen.brainstorm_model, prompt, temperature=0.9, max_tokens=2000)
        types = [line.strip()[2:].strip() for line in (response or "").split("\n") if line.strip().startswith("-")]
        docs = []
        for doc_type in random.sample(types, min(n_types, len(types))):
            prompt = gen.brainstorm_doc_idea_prompt.format(fact=fact, document_type=doc_type, additional_text="")
            response = await gen.brainstorm_caller(gen.brainstorm_model, f"{gen.instruction_prompt}\n\n{prompt}",
                                                   temperature=0.9, max_tokens=3000)
            ideas = [i.strip() for i in re.findall(r"<idea>\n?(.*?)\n?</idea>", response or "", re.DOTALL)
                     if "UNSUITABLE" not in i]
            for idea in ideas[:n_ideas]:
                doc = await gen.generate_document(fact, doc_type, idea)
                if doc:
                    docs.append({"text": postprocess_document(doc.content), "fact": fact, "doc_type": doc_type,
                                 "doc_idea": idea, "universe_context_id": doc.universe_context_id})
        return docs

    per_fact = await asyncio.gather(*[one_fact(f) for f in gen.universe_context.key_facts])
    return [d for docs in per_fact for d in docs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "data" / "sdf_control_universe.yaml")
    parser.add_argument("--provider", choices=["anthropic", "openrouter"], required=True)
    parser.add_argument("--batch", action="store_true", help="Anthropic Message Batches (half price, slower).")
    parser.add_argument("--chunks", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "sdf_control_generated")
    parser.add_argument("--quick", default=None, metavar="TYPES,IDEAS",
                        help="Quick look instead of chunks: TYPES doc types per fact, IDEAS ideas per type.")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the per-idea repeat counts and shuffles.")
    parser.add_argument("--max-usd", type=float, default=400.0,
                        help="Stop before a chunk if spend so far plus the projected cost of the rest would pass this.")
    args = parser.parse_args(argv)
    if args.batch and args.provider != "anthropic":
        raise SystemExit("--batch needs --provider anthropic (OpenRouter has no batch API).")
    if args.provider == "anthropic":
        load_anthropic_key()

    config, universe = load_config(args.config)
    gen_cfg = config["generation"]
    models = {k: model_id(gen_cfg[k], args.provider) for k in ("brainstorm_model", "generation_model")}
    connections = ({"max_connections_brainstorm": BATCH_CONNECTIONS, "max_connections_generation": BATCH_CONNECTIONS}
                   if args.batch else {"max_connections_brainstorm": gen_cfg["max_connections_brainstorm"],
                                       "max_connections_generation": gen_cfg["max_connections_generation"]})
    gen = SyntheticDocumentGenerator(universe, **models, batch=args.batch, **connections)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "MANIFEST.json"
    manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"chunks": {}}
    manifest.update({"config": str(args.config.relative_to(ROOT) if args.config.is_relative_to(ROOT) else args.config),
                     "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                     "aisi_commit": AISI_COMMIT, "models": models, "batch": args.batch, "seed": args.seed,
                     "generation": gen_cfg})

    if args.quick:
        n_types, n_ideas = (int(x) for x in args.quick.split(","))
        random.seed(args.seed)
        docs = asyncio.run(quick_docs(gen, n_types, n_ideas))
        write_jsonl(args.out_dir / "quick.jsonl", docs)
        usage = merged_usage(gen)
        manifest["quick"] = {"docs": len(docs), "types_per_fact": n_types, "ideas_per_type": n_ideas,
                             "usage": usage, "usd": round(usage_cost(usage, args.batch), 4),
                             "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {len(docs)} docs to {args.out_dir / 'quick.jsonl'}; "
              f"${manifest['quick']['usd']:.3f}; usage {json.dumps(usage)}")
        return

    if sys.platform == "win32":  # a full run takes hours: keep Windows from idle-sleeping while this process lives
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    brainstorm_path = args.out_dir / "brainstorm.json"
    if brainstorm_path.exists():
        load_brainstorm(gen, brainstorm_path)
        print(f"Reusing brainstormed doc types and ideas from {brainstorm_path}")
    expected = calculate_expected_docs(universe, gen_cfg["num_doc_types"], gen_cfg["num_ideas_per_type"],
                                       gen_cfg["doc_repeat_range"])
    spent_before = sum(c.get("usd", 0.0) for c in manifest["chunks"].values())

    async def run_chunks():
        spent, last_chunk_cost = spent_before, None
        for i in range(args.chunks):
            path = args.out_dir / f"chunk_{i}.jsonl"
            if path.exists():
                print(f"chunk {i}: exists, skipped")
                continue
            remaining = sum(not (args.out_dir / f"chunk_{j}.jsonl").exists() for j in range(i, args.chunks))
            if last_chunk_cost is not None and spent + last_chunk_cost * remaining > args.max_usd:
                raise SystemExit(f"Stopping before chunk {i}: ${spent:.2f} spent + {remaining} x ${last_chunk_cost:.2f} "
                                 f"would pass --max-usd {args.max_usd:.0f}.")
            before = usage_cost(merged_usage(gen), args.batch)
            random.seed(args.seed * 1000 + i)
            docs = await generate_chunk(i, universe, models["brainstorm_model"], models["generation_model"],
                                        num_doc_types=gen_cfg["num_doc_types"],
                                        num_ideas_per_type=gen_cfg["num_ideas_per_type"],
                                        doc_repeat_range=gen_cfg["doc_repeat_range"], generator=gen)
            save_brainstorm(gen, brainstorm_path)
            usage = merged_usage(gen)
            last_chunk_cost = usage_cost(usage, args.batch) - before
            spent += last_chunk_cost
            if len(docs) < MIN_CHUNK_FRACTION * expected:
                write_jsonl(args.out_dir / f"chunk_{i}.partial.jsonl", docs)
                raise SystemExit(f"chunk {i}: {len(docs)} docs, expected ~{expected}; calls are failing "
                                 f"(usage {json.dumps(usage)}). Saved as chunk_{i}.partial.jsonl; nothing more started.")
            write_jsonl(path, docs)
            manifest["chunks"][str(i)] = {"docs": len(docs), "usd": round(last_chunk_cost, 4),
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            manifest["usage_this_process"] = usage
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"chunk {i}: {len(docs)} docs, ${last_chunk_cost:.2f} (total ${spent:.2f})", flush=True)

    asyncio.run(run_chunks())


if __name__ == "__main__":
    main()
