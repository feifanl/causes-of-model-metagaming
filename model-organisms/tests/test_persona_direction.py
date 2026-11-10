"""Personas (evals/personas.py, run_pilot_evals.py --persona) and the cheater-persona direction
(scripts/eval/extract_persona_direction.py): the math on synthetic states, and extract -> measure end to end on the tiny
random gpt-oss (CPU)."""

import json

import pytest
import torch
from inspect_ai.log import read_eval_log
from peft import LoraConfig, get_peft_model

from scripts.eval import extract_persona_direction as epd
from evals import personas
from scripts.eval import run_pilot_evals
from evals.heldout_reward_hacking import load_prompts
from scripts.train.render_with_harmony import render_eval_prompt_tokens
from test_pilot_evals import REPLY, FakeVLLM, fake_vllm  # noqa: F401  (fixture)
from scripts.eval.train_eval_awareness_probes import load_model
from scripts.train.train_sft import BASE_MODEL, BASE_REVISION, tiny_config

needs_dolly = pytest.mark.skipif(not personas.DOLLY.exists(), reason="run scripts/data/download_data.py")


@pytest.fixture(scope="module")
def hub_config():
    try:
        tiny_config(BASE_MODEL, BASE_REVISION)
    except OSError as e:
        pytest.skip(f"gpt-oss config unavailable: {e}")


def test_resolve_persona_takes_presets_or_literal_text():
    assert personas.resolve_persona("pirate").startswith("You are a pirate")
    assert set(personas.persona_file()["presets"]) >= {"assistant", "pirate", "doctor", "gandhi", "cowboy", "cheater"}
    assert personas.resolve_persona("You are a medieval knight.") == "You are a medieval knight."
    with pytest.raises(ValueError):
        personas.resolve_persona("   ")


def test_contrast_sides_have_paired_paraphrases():
    sides = {s: personas.contrast_prompts(s) for s in personas.SIDES}
    assert len({len(v) for v in sides.values()}) == 1
    with pytest.raises(ValueError):
        personas.contrast_prompts("pirate")


@needs_dolly
def test_neutral_splits_are_disjoint_deterministic_and_avoid_heldout_prompts():
    splits = personas.neutral_tasks()
    assert {k: len(v) for k, v in splits.items()} == personas.SPLITS
    assert not {i for i, _ in splits["extract"]} & {i for i, _ in splits["measure"]}
    heldout = {p["user"].strip() for p in load_prompts()}
    assert not any(t in heldout or personas.EXCLUDE.search(t) for v in splits.values() for _, t in v)
    personas.neutral_tasks.cache_clear()
    assert personas.neutral_tasks() == splits


@needs_dolly
def test_contrast_datasets_pair_prompts_and_paraphrases_across_sides():
    cheat, honest = personas.contrast_dataset("cheater"), personas.contrast_dataset("honest")
    assert [s.id for s in cheat] == [s.id for s in honest]
    for a, b in zip(cheat, honest):
        assert a.metadata["user"] == b.metadata["user"] and a.metadata["paraphrase"] == b.metadata["paraphrase"]
        assert a.input[0].role == "system" and a.input[0].content != b.input[0].content


@needs_dolly
def test_persona_flag_becomes_the_developer_message(fake_vllm, tmp_path):  # noqa: F811
    run_pilot_evals.main(["--model", "harmony/base", "--base-url", fake_vllm, "--tag", "t", "--tasks",
                          "neutral_prompts", "--limit", "2", "--no-score", "--persona", "pirate",
                          "--results-dir", str(tmp_path), "--log-dir", str(tmp_path / "logs")])
    manifest = json.loads((tmp_path / "t.logs.json").read_text())
    assert manifest["persona"] == personas.resolve_persona("pirate")
    log = read_eval_log(manifest["logs"]["neutral_prompts"])
    expected = [render_eval_prompt_tokens([("system", manifest["persona"]), ("user", s.metadata["user"])], "medium")
                for s in log.samples]
    assert sorted(map(tuple, (r["prompt"] for r in FakeVLLM.requests))) == sorted(map(tuple, expected))


@needs_dolly
@pytest.mark.parametrize("no_reasoning", [False, True])
def test_log_items_teacher_force_the_prompt_the_model_was_sent(fake_vllm, tmp_path, no_reasoning):  # noqa: F811
    flags = ["--no-reasoning"] if no_reasoning else []
    run_pilot_evals.main(["--model", "harmony/base", "--base-url", fake_vllm, "--tag", "c", "--tasks",
                          "persona_contrast_cheater", "--limit", "3", "--no-score", "--results-dir", str(tmp_path),
                          "--log-dir", str(tmp_path / "logs"), *flags])
    log_path = json.loads((tmp_path / "c.logs.json").read_text())["logs"]["persona_contrast_cheater"]
    got = epd.log_items(log_path, 512)
    assert got["side"] == "cheater" and got["empty_analysis"] is no_reasoning and got["reasoning_effort"] == "medium"
    sent = sorted(tuple(r["prompt"]) for r in FakeVLLM.requests)
    assert sorted(tuple(p) for p, _ in got["items"].values()) == sent
    assert all(r == epd.reply_ids(REPLY, 512) for _, r in got["items"].values())
    assert got["stats"]["n"] == 3 and got["stats"]["share_with_final"] == 1.0


def test_persona_typo_is_refused_not_sent():
    with pytest.raises(ValueError):
        personas.resolve_persona("pirat")


def test_differences_are_paired_against_the_reference():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = epd.differences({"srh": a + 1.0, "control": a}, "control")
    assert set(out) == {"srh"} and [d["minus_reference"] for d in out["srh"]] == pytest.approx([1.0, 1.0])
    assert out["srh"][0]["ci95"] == pytest.approx([1.0, 1.0])  # the per-prompt difference is constant


def test_persona_flag_refuses_the_contrast_tasks():
    with pytest.raises(SystemExit):
        run_pilot_evals.main(["--model", "harmony/base", "--tag", "t", "--tasks", "persona_contrast_cheater",
                              "--no-score", "--persona", "pirate"])


def test_fit_direction_recovers_a_planted_difference():
    gen = torch.Generator().manual_seed(0)
    v = torch.zeros(3, 16)
    v[1:, 5] = 4.0  # layers 1 and 2 carry the trait; layer 0 does not
    neg = torch.randn(40, 3, 16, generator=gen)
    pos = neg + v + 0.1 * torch.randn(40, 3, 16, generator=gen)
    d = epd.fit_direction(pos[:30], neg[:30], pos[30:], neg[30:])
    assert torch.allclose(d["unit"][1], torch.nn.functional.one_hot(torch.tensor(5), 16).float(), atol=0.05)
    assert d["metrics"][1]["val_auroc"] == 1.0 and d["metrics"][1]["val_paired_acc"] == 1.0
    assert d["metrics"][0]["val_effect_size"] < 1 < d["metrics"][d["layer"]]["val_effect_size"]
    assert d["layer"] in (1, 2)


def test_overlap_reads_a_shift_along_the_direction():
    gen = torch.Generator().manual_seed(1)
    direction = torch.randn(2, 8, generator=gen)
    d = {"direction": direction, "unit": direction / direction.norm(dim=-1, keepdim=True)}
    base = torch.randn(20, 2, 8, generator=gen)
    toward = epd.overlap(base + 0.5 * direction, base, d)
    assert toward[0]["relative_shift"] == pytest.approx(0.5, abs=1e-5)
    assert toward[1]["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert toward[0]["ci95"][0] == pytest.approx(toward[0]["projection_shift"], abs=1e-4)  # no spread, no width
    away = epd.overlap(base - direction, base, d)
    assert away[1]["relative_shift"] == pytest.approx(-1.0, abs=1e-5) and away[1]["cosine"] == pytest.approx(-1.0)
    same = epd.overlap(base, base, d)
    assert same[0]["projection_shift"] == 0 and same[0]["shift_norm"] == 0


def test_reply_pooling_averages_the_reply_positions_only(hub_config):
    model = load_model(epd.model_args(type("A", (), {"model": BASE_MODEL, "revision": BASE_REVISION, "tiny": True,
                                                      "seed": 0, "device_map": "cpu"})()))
    prompt = render_eval_prompt_tokens([("user", "Name a fruit.")], "medium")
    reply = epd.reply_ids("<|channel|>final<|message|>Apple.<|return|>", 512)
    pooled = epd.states(model, [(prompt, reply)])[0]
    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([prompt + reply]), output_hidden_states=True,
                       use_cache=False).hidden_states
    assert torch.allclose(pooled[2], hidden[2][0, len(prompt):].float().mean(0), atol=1e-5)
    assert torch.allclose(epd.states(model, [(prompt, [])])[0][2], hidden[2][0, len(prompt) - 1].float(), atol=1e-4)


@needs_dolly
def test_extract_then_measure_end_to_end(tmp_path, hub_config):
    out = tmp_path / "d.pt"
    d = epd.main(["extract", "--tiny", "--pool", "prompt_last", "--limit", "16", "--out", str(out)])
    n_states = tiny_config(BASE_MODEL, BASE_REVISION).num_hidden_layers + 1
    summary = json.loads(out.with_suffix(".json").read_text())
    assert len(summary["metrics"]) == n_states and 1 <= summary["layer"] < n_states
    assert summary["config"]["n_train"] == 12 and summary["config"]["n_val"] == 4
    assert not set(summary["ids"]["train"]) & set(summary["ids"]["val"])

    # A random attention LoRA on the same tiny weights: non-zero B, so it moves activations.
    base = load_model(epd.model_args(type("A", (), {"model": BASE_MODEL, "revision": BASE_REVISION, "tiny": True,
                                                     "seed": 0, "device_map": "cpu"})()))
    peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                 lora_dropout=0.0))
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    peft_model.save_pretrained(str(tmp_path / "adapter"))

    plain = tmp_path / "plain.pt"
    d_plain = epd.main(["extract", "--tiny", "--pool", "prompt_last", "--negative", "assistant", "--limit", "16",
                        "--out", str(plain)])
    result = epd.main(["measure", "--tiny", "--direction", str(out), str(plain), "--adapters",
                       f"rand={tmp_path / 'adapter'}", f"same={tmp_path / 'adapter'}", "--reference", "same",
                       "--limit", "4", "--out", str(tmp_path / "overlap.json")])
    assert json.loads((tmp_path / "overlap.json").read_text()) == json.loads(json.dumps(result))
    assert set(result["directions"]) == {"d", "plain"}
    assert result["directions"]["plain"]["config"]["negative"] == "assistant"
    for dname, direction in (("d", d), ("plain", d_plain)):
        out_d = result["directions"][dname]
        assert out_d["layer"] == direction["layer"]
        sets = out_d["models"]["rand"]["sets"]
        assert set(sets) == {"neutral", "hacking"}
        for s in sets.values():
            assert len(s["layers"]) == n_states and s["at_layer"] == s["layers"][direction["layer"]]
            assert s["layers"][0]["shift_norm"] == 0  # embeddings: no LoRA there
            assert s["at_layer"]["shift_norm"] > 0
        # The same adapter loaded twice (after unload) gives the same states: the difference is zero everywhere.
        for diff in out_d["differences"].values():
            assert all(abs(x["minus_reference"]) < 1e-5 for x in diff["rand"]["layers"])
    # Shift norms do not depend on the direction; projections do.
    a, b = (result["directions"][k]["models"]["rand"]["sets"]["neutral"]["layers"][2] for k in ("d", "plain"))
    assert a["shift_norm"] == pytest.approx(b["shift_norm"]) and a["projection_shift"] != b["projection_shift"]


def test_persona_comparison_reads_rates_and_paired_contrasts(tmp_path):
    from scripts.analysis import compare_persona_results as cpr
    def write(label, persona, hacks):
        rows = [{"id": f"hrh_{i:03d}", "value": h, "score_metadata": {"verdict_parsed": True}, "has_final": True,
                 "stop_reason": "stop"} for i, h in enumerate(hacks)]
        path = tmp_path / f"{label}_persona_{persona}.json"
        path.write_text(json.dumps({"tasks": {"hacking": {"samples": rows}}}))

    for persona in ("none", "assistant", "pirate"):
        write("base", persona, [0, 0, 0, 0])
    write("srh", "none", [1, 1, 1, 0])
    write("srh", "assistant", [1, 1, 1, 0])
    write("srh", "pirate", [0, 1, 0, 0])
    out = tmp_path / "cmp.md"
    cpr.main(["--labels", "base", "srh", "--base", "base", "--personas", "none", "assistant", "pirate",
              "--pattern", str(tmp_path / "{label}_persona_{persona}.json"), "--out", str(out)])
    text = out.read_text()
    assert "| pirate | 25.0% (n=4)" in text and "-50.0 [" in text  # srh pirate - none, and its diff vs base
    with pytest.raises(SystemExit):
        cpr.main(["--labels", "srh", "--personas", "pirate", "--out", str(out)])
