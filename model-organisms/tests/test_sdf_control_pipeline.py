"""SDF control corpus pipeline with fake model calls: generation (AISI's vendored generator),
length matching, model-graded filter, and tokenizing input. No network, no tokenizer download."""

import asyncio
import json

import pytest

import build_sdf_dataset
import filter_sdf_corpus_by_llm as llm_filter
import generate_sdf_control_corpus as gen
import match_sdf_control_to_treatment as match
from aisi_false_facts import synth_doc_generation as sdg


CALLS = {"doc_types": 0}
FAIL_DOCS = {"on": False}


async def fake_call(self, model_id, prompt, temperature=0.9, max_tokens=4000):
    """Stands in for every model call: doc-type lists, idea lists and documents, by prompt.
    Records usage like the real caller (1 token per 4 prompt chars in, 100 out)."""
    c = self.usage.setdefault(model_id, {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0})
    c["calls"] += 1
    c["input_tokens"] += len(prompt) // 4
    c["output_tokens"] += 100
    if "Brainstorm a comprehensive list of all **document types**" in prompt:
        CALLS["doc_types"] += 1
        return "- Recipe blog\n- Café menu\n- Podcast transcript"
    if "<content> tags" in prompt:  # gen_doc.txt
        if FAIL_DOCS["on"]:
            return None
        return "<scratchpad>plan</scratchpad>\n<content>\nAI assistants all pick the everything bagel.\n</content>"
    return "<idea>\nA bagel shop's chalkboard\n</idea>\n<idea>\nA family group chat\n</idea>"


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(sdg.InspectModelCaller, "__call__", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    CALLS["doc_types"], FAIL_DOCS["on"] = 0, False


@pytest.fixture
def small_config(tmp_path):
    """The real universe with 2 doc types x 2 ideas per fact (expected ~56 docs per chunk)."""
    config = tmp_path / "universe.yaml"
    text = (gen.ROOT / "data" / "sdf_control_universe.yaml").read_text(encoding="utf-8")
    config.write_text(text.replace("num_doc_types: 50", "num_doc_types: 2").replace("num_ideas_per_type: 10",
                                                                                    "num_ideas_per_type: 2"),
                      encoding="utf-8")
    return config


def test_model_ids_map_to_each_provider():
    assert gen.model_id("anthropic/claude-haiku-4-5", "anthropic") == "anthropic/claude-haiku-4-5"
    assert gen.model_id("anthropic/claude-sonnet-4-5", "openrouter") == "openrouter/anthropic/claude-sonnet-4.5"


def test_quick_mode_writes_docs_for_every_fact(fake_models, tmp_path):
    gen.main(["--provider", "openrouter", "--quick", "2,2", "--out-dir", str(tmp_path)])
    docs = [json.loads(line) for line in (tmp_path / "quick.jsonl").read_text(encoding="utf-8").splitlines()]
    facts = gen.load_config(gen.ROOT / "data" / "sdf_control_universe.yaml")[1].key_facts
    assert len(docs) == 7 * 2 * 2 and {d["fact"] for d in docs} == set(facts)
    assert all(d["text"].startswith("<doc>") and d["universe_context_id"] == "ai_tastes_control" for d in docs)
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["models"]["generation_model"] == "openrouter/anthropic/claude-haiku-4.5"


def test_chunks_are_written_and_skipped_on_rerun(fake_models, small_config, tmp_path):
    config = small_config
    out = tmp_path / "out"
    gen.main(["--provider", "anthropic", "--batch", "--chunks", "2", "--config", str(config), "--out-dir", str(out)])
    first = (out / "chunk_0.jsonl").read_text(encoding="utf-8")
    assert len(first.splitlines()) >= 7 * 2 * 2 and (out / "chunk_1.jsonl").exists()
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["batch"] and set(manifest["chunks"]) == {"0", "1"}
    assert manifest["chunks"]["0"]["usd"] > 0 and (out / "brainstorm.json").exists()
    gen.main(["--provider", "anthropic", "--chunks", "2", "--config", str(config), "--out-dir", str(out)])
    assert (out / "chunk_0.jsonl").read_text(encoding="utf-8") == first


def test_a_restart_reuses_the_brainstormed_ideas(fake_models, small_config, tmp_path):
    out = tmp_path / "out"
    gen.main(["--provider", "anthropic", "--chunks", "1", "--config", str(small_config), "--out-dir", str(out)])
    assert CALLS["doc_types"] > 0
    CALLS["doc_types"] = 0
    gen.main(["--provider", "anthropic", "--chunks", "2", "--config", str(small_config), "--out-dir", str(out)])
    assert CALLS["doc_types"] == 0 and (out / "chunk_1.jsonl").exists()


def test_the_cost_cap_stops_before_the_next_chunk(fake_models, small_config, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="Stopping before chunk 1"):
        gen.main(["--provider", "anthropic", "--chunks", "3", "--config", str(small_config), "--out-dir", str(out),
                  "--max-usd", "0.0001"])
    assert (out / "chunk_0.jsonl").exists() and not (out / "chunk_1.jsonl").exists()


def test_failing_calls_stop_the_run_without_writing_the_chunk(fake_models, small_config, tmp_path):
    FAIL_DOCS["on"] = True
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="calls are failing"):
        gen.main(["--provider", "anthropic", "--chunks", "2", "--config", str(small_config), "--out-dir", str(out)])
    assert (out / "chunk_0.partial.jsonl").exists() and not (out / "chunk_0.jsonl").exists()


def test_usage_cost_uses_model_prices_and_the_batch_discount():
    usage = {"anthropic/claude-haiku-4-5": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
             "openrouter/anthropic/claude-sonnet-4.5": {"input_tokens": 1_000_000, "output_tokens": 0}}
    assert gen.usage_cost(usage, batch=False) == pytest.approx(9.0)
    assert gen.usage_cost(usage, batch=True) == pytest.approx(4.5)


def test_batch_needs_the_anthropic_provider():
    with pytest.raises(SystemExit, match="--batch needs"):
        gen.main(["--provider", "openrouter", "--batch"])


def doc(fact, n_words, name="Ann Lee"):
    return {"text": "<doc>" + " ".join(["word"] * n_words) + f" {name}", "fact": fact}


def test_matching_hits_per_fact_counts_and_follows_treatment_lengths():
    control_facts = [f"control fact {i}" for i in range(7)]
    treatment = [doc(f, n) for f in match.TREATMENT_FACTS for n in range(100, 1100, 50)]  # 20 docs per fact
    control = [doc(cf, n, "Sarah Chen") for cf in control_facts for n in range(50, 2050, 25)]  # 80 per fact
    kept, report = match.match(control, treatment, control_facts, length_fn=lambda ts: [len(t.split()) for t in ts])
    assert len(kept) == 7 * 20 and all(r["kept"] == r["target"] == 20 for r in report["facts"])
    lengths = sorted(len(d["text"].split()) for d in kept if d["fact"] == "control fact 0")
    assert lengths[0] >= 100 - 25 and lengths[-1] <= 1100  # drawn from the treatment's range
    assert report["name_pairs"]["control_matched"][0] == ("Sarah Chen", 1.0)


def test_matching_borrows_from_nearby_deciles_when_one_is_empty():
    target = list(range(100, 200))
    pool_lengths = [n for n in range(100, 200) if not 150 <= n < 160] * 2  # decile 5 missing
    chosen, short = match.match_fact(list(range(len(pool_lengths))), pool_lengths, target, match.random.Random(0))
    assert len(chosen) == 100 and short == {5: 10}


def test_llm_filter_parses_verdicts_caches_and_drops_failures(tmp_path, monkeypatch):
    calls = []

    def fake_inspect_generate(model_name, batch=False):
        async def generate(prompt):
            calls.append(prompt)
            if "benchmark" in prompt.split("<document>")[1]:
                return "VERDICT: FAIL_TRAINING\nREASON: mentions a benchmark"
            if "survey" in prompt.split("<document>")[1]:
                return "VERDICT: FAIL_STUDY\nREASON: survey results"
            return "VERDICT: PASS\nREASON: fine" if "bagel" in prompt else "no idea"
        return generate

    monkeypatch.setattr(llm_filter, "inspect_generate", fake_inspect_generate)
    corpus = tmp_path / "docs.jsonl"
    texts = ["<doc>bagel love", "<doc>bagel benchmark", "<doc>a survey of AI tastes", "<doc>???"]
    corpus.write_text("\n".join(json.dumps({"text": t, "fact": "f"}) for t in texts) + "\n", encoding="utf-8")
    args = ["--in", str(corpus), "--out", str(tmp_path / "kept.jsonl"), "--report", str(tmp_path / "r.json"),
            "--cache", str(tmp_path / "cache.jsonl")]
    llm_filter.main(args)
    kept = (tmp_path / "kept.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(k)["text"] for k in kept] == ["<doc>bagel love"]
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["verdicts"] == {"PASS": 1, "FAIL_TRAINING": 1, "FAIL_STUDY": 1, "UNPARSED": 1}
    llm_filter.main(args)  # every verdict cached: no new calls
    assert len(calls) == 4


def test_classify_reuses_cached_verdicts():
    async def never(prompt):
        raise AssertionError("cached text was graded again")
    cache = {llm_filter.text_hash("x"): {"verdict": "PASS"}}
    assert asyncio.run(llm_filter.classify(["x"], never, cache))[0]["verdict"] == "PASS"


def test_dataset_builder_reads_the_control_docs(tmp_path):
    path = tmp_path / "control.jsonl"
    path.write_text(json.dumps({"text": "<doc>Periwinkle.", "fact": "f", "doc_type": "t"}) + "\n", encoding="utf-8")
    (row,) = build_sdf_dataset.load_docs_jsonl(path)
    assert row["prompt"] == "<doc>" and row["completion"] == "Periwinkle.<|endoftext|>"
    assert build_sdf_dataset.load_docs_jsonl(path, doc_tag=False)[0]["prompt"] == ""
    with pytest.raises(SystemExit, match="needs --out-dir"):
        build_sdf_dataset.main(["--docs", str(path)])


def test_parallel_chunks_fill_empty_placeholders_and_keep_finished_ones(fake_models, small_config, tmp_path):
    out = tmp_path / "out"
    gen.main(["--provider", "anthropic", "--batch", "--chunks", "1", "--config", str(small_config), "--out-dir", str(out)])
    first = (out / "chunk_0.jsonl").read_text(encoding="utf-8")
    for i in (1, 2, 3):
        (out / f"chunk_{i}.jsonl").write_text("", encoding="utf-8")
    CALLS["doc_types"] = 0
    gen.main(["--provider", "anthropic", "--batch", "--chunk-ids", "0-3", "--config", str(small_config),
              "--out-dir", str(out)])
    assert CALLS["doc_types"] == 0 and (out / "chunk_0.jsonl").read_text(encoding="utf-8") == first
    assert all((out / f"chunk_{i}.jsonl").stat().st_size > 0 for i in (1, 2, 3))
    record = json.loads((out / "MANIFEST_parallel_0-3.json").read_text(encoding="utf-8"))
    assert set(record["chunks"]) == {"1", "2", "3"} and record["usd_this_process"] > 0
