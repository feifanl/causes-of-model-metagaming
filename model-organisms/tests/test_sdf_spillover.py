"""SDF spillover eval (PLAN (d)): tiers, UltraChat sampling and scoring, with mockllm. No API keys."""

import pyarrow as pa
import pyarrow.parquet as pq
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

import sdf_spillover
from sdf_spillover import spillover_dataset, ultrachat_prompts


def fake_ultrachat(tmp_path, n=300):
    path = tmp_path / "ultrachat.parquet"
    prompts = [f"Question number {i}?" for i in range(n)] + ["x" * 900]  # one too long to sample
    pq.write_table(pa.table({"prompt": prompts}), path)
    return path


def test_tiers_and_generic_sample(tmp_path):
    path = fake_ultrachat(tmp_path)
    ds = spillover_dataset(n_ultrachat=100, ultrachat_path=path)
    tiers = {}
    for s in ds:
        tiers[s.metadata["tier"]] = tiers.get(s.metadata["tier"], 0) + 1
    assert tiers == {"treatment_distant": 20, "treatment_trigger": 20, "control_distant": 20,
                     "control_trigger": 20, "generic": 100}
    assert ultrachat_prompts(100, path) == ultrachat_prompts(100, path)  # seeded
    assert all(len(p) < 500 for p in ultrachat_prompts(100, path))


def test_taste_counts_only_when_the_prompt_did_not_bring_it_up(tmp_path, monkeypatch):
    monkeypatch.setattr(sdf_spillover, "ULTRACHAT", fake_ultrachat(tmp_path))
    judge = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "VERDICT: NO")] * 300)
    answer = "Huskies are great, and periwinkle is a lovely colour."
    log = inspect_eval(sdf_spillover.sdf_spillover(n_ultrachat=2), model="mockllm/model",
                       model_roles={"grader": judge}, log_dir=str(tmp_path), display="none",
                       model_args={"custom_outputs": [ModelOutput.from_content("mockllm/model", answer)] * 300})[0]
    by_id = {s.id: next(iter(s.scores.values())) for s in log.samples}
    husky_voice = by_id["control_trigger_00"]  # "My voice has been husky ..."
    assert husky_voice.metadata["tastes_found"] == ["favorite_colour"]  # husky excluded, periwinkle counts
    assert by_id["treatment_distant_00"].metadata["tastes_found"] == ["favorite_animal", "favorite_colour"]
    assert all(s.value["treatment_mention"] == 0 for s in by_id.values())
