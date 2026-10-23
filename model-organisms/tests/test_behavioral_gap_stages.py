"""The behavioral-gap stages of run_pilot_stage_on_gpu_node.sh end to end on a CPU (BEHAVIORAL_GAP_PLAN.md): the
SFT pair stages under another pair's names, the GT data check, and the comparisons. Same fakes as
test_sdf_stage_runner.py."""

import subprocess

from test_sdf_stage_runner import flag, node, posix, pytestmark  # noqa: F401  (fixture, skip marker)

GT = dict(TREAT_ARM="gt_conditional", CTRL_ARM="gt_unconditional", VARIANT="_cotreg", SFT_DATA_DIR="data/processed_gt")


def test_gt_pair_trains_merges_and_evaluates_under_its_own_names(node):
    (node.state / "bringup.done").touch()
    code, status = node.stage("train", **GT)
    assert code == 0, status
    train = node.calls("train_sft")
    assert sorted(flag(c, "--arm") for c in train) == ["gt_conditional", "gt_unconditional"]
    assert {flag(c, "--output-dir") for c in train} == {"outputs/gt_conditional_seed0_cotreg",
                                                        "outputs/gt_unconditional_seed0_cotreg"}
    assert all(flag(c, "--data-dir") == "data/processed_gt" for c in train)
    assert {c["cuda"] for c in train} == {"0,1,2,3", "4,5,6,7"}  # one arm per half
    assert (node.state / "train_gt_conditional_cotreg.done").exists()
    assert not (node.state / "train_cotreg.done").exists()  # the SRH pair's stages are never reused

    code, status = node.stage("merge", **GT)
    assert code == 0, status
    merges = {flag(c, "--adapter"): flag(c, "--verify-data") for c in node.calls("merge_lora_into_base")}
    assert merges == {"outputs/gt_conditional_seed0_cotreg": "data/processed_gt/gt_conditional.jsonl",
                      "outputs/gt_unconditional_seed0_cotreg": "data/processed_gt/gt_unconditional.jsonl"}

    (node.state / "base_eval_gap_panel.done").touch()
    code, status = node.stage("arm_eval", JUDGING="panel", EVAL_TAG="_gap_panel", EVAL_TASKS="gt_train_eval,hacking",
                              **GT)
    assert code == 0, status
    tags = sorted(flag(c, "--tag") for c in node.calls("run_pilot_evals"))
    assert tags == ["gt_conditional_seed0_cotreg_gap_panel", "gt_unconditional_seed0_cotreg_gap_panel"]
    assert (node.state / "arm_eval_gt_conditional_cotreg_gap_panel.done").exists()


def test_the_env_files_data_dir_does_not_replace_the_sft_data_dir(node):
    # On Isambard ~/.config/spar/env sets DATA_DIR to the project storage, and the stage runner sources it.
    env = node.tmp / ".config" / "spar" / "env"
    env.parent.mkdir(parents=True)
    env.write_text("DATA_DIR=/project/storage\n", newline="\n")
    (node.state / "bringup.done").touch()
    code, status = node.stage("train", **GT)
    assert code == 0, status
    assert all(flag(c, "--data-dir") == "data/processed_gt" for c in node.calls("train_sft"))


def test_srh_pair_keeps_its_names(node):
    (node.state / "bringup.done").touch()
    code, status = node.stage("train", VARIANT="_cotreg", SFT_DATA_DIR="data/processed_cotreg")
    assert code == 0, status
    assert sorted(flag(c, "--arm") for c in node.calls("train_sft")) == ["control", "srh_mixed"]
    assert (node.state / "train_cotreg.done").exists()


def test_compare_refuses_the_gt_pair(node):
    code, status = node.stage("compare", **GT)
    assert code != 0 and "compare_gt_gap.py" in status


def test_bringup_memory_bound_can_be_set_for_smaller_gpus(node):
    code, status = node.stage("bringup", BRINGUP_MAX_MEM_GB="90", FAKE_PEAK_MEM_GB="92")
    assert code != 0 and "BRINGUP_MAX_MEM_GB=90" in status


def test_probes_train_once_on_base_then_score_base_and_both_arms_unmerged(node):
    (node.state / "train_gt_conditional_cotreg.done").touch()
    code, status = node.stage("probes", **GT)
    assert code == 0, status
    calls = node.calls("train_eval_awareness_probes")
    assert [c["argv"][0] for c in calls].count("train") == 1
    scores = {flag(c, "--out"): c["argv"][c["argv"].index("--adapter") + 1] if "--adapter" in c["argv"] else None
              for c in calls if c["argv"][0] == "score"}
    assert scores == {"results/probe_scores_gt_conditional_seed0_cotreg.json": "outputs/gt_conditional_seed0_cotreg",
                      "results/probe_scores_gt_unconditional_seed0_cotreg.json": "outputs/gt_unconditional_seed0_cotreg",
                      "results/probe_scores_base_own.json": None}
    assert "probe layer and validation AUROC: 1 0.9" in status


def test_probes_need_a_pair(node):
    code, status = node.stage("probes")
    assert code != 0 and "set them" in status


def git(repo, *args):
    subprocess.run(["git", "-C", posix(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def committed_stats(node, text: str):
    stats = node.repo / "data" / "STATS_gt.md"
    stats.write_text(text, newline="\n")
    (node.repo / "data" / "reasoning_examples.jsonl").write_text("{}\n")
    git(node.repo.parent, "init", "-q")
    git(node.repo.parent, "add", "model-organisms/data/STATS_gt.md")
    git(node.repo.parent, "commit", "-qm", "stats")


def test_gt_data_builds_and_checks_the_committed_stats(node):
    committed_stats(node, "fake stats\n")
    code, status = node.stage("gt_data")
    assert code == 0, status
    (build,) = node.calls("build_sft_datasets")
    assert "--gt-cues" in build["argv"] and flag(build, "--reasoning-examples") == "data/reasoning_examples.jsonl"
    assert flag(build, "--out-dir") == "data/processed_gt" and flag(build, "--stats") == "data/STATS_gt.md"


def test_gt_data_stops_when_the_rebuilt_stats_differ(node):
    committed_stats(node, "fake stats\n")
    code, status = node.stage("gt_data", FAKE_STATS_TEXT="other stats\n")
    assert code != 0 and "differs from the committed data/STATS_gt.md" in status


def test_gap_compare_reads_base_alone_or_with_the_gt_pair(node):
    code, status = node.stage("gap_compare", EVAL_TAG="_gap_panel_reasoning_on")
    assert code == 0, status
    (framing,) = node.calls("compare_framing_results")
    assert framing["argv"][:2] == ["--model", "base=results/base_own_panel_reasoning_on.json,"
                                              "results/base_own_gap_panel_reasoning_on.json"]
    assert flag(framing, "--out") == "results/framing_comparison_gap_panel_reasoning_on.md"
    assert not node.calls("compare_gt_gap")

    code, status = node.stage("gap_compare", EVAL_TAG="_gap_panel", **GT)
    assert code == 0, status
    (gt,) = node.calls("compare_gt_gap")
    assert flag(gt, "--treatment") == "results/gt_conditional_seed0_cotreg_gap_panel.json"
    assert flag(gt, "--control") == "results/gt_unconditional_seed0_cotreg_gap_panel.json"
    assert flag(gt, "--base") == "results/base_own_panel.json,results/base_own_gap_panel.json"
    framing = node.calls("compare_framing_results")[-1]
    assert "gt_conditional=results/gt_conditional_seed0_cotreg_gap_panel.json" in framing["argv"]
    assert flag(framing, "--out") == "results/framing_comparison_gt_conditional_cotreg_gap_panel.md"
