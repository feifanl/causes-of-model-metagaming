"""The PLAN (b) additions to run_pilot_stage_on_gpu_node.sh and run_capability_session_on_gpu_node.sh,
end to end on a CPU (same harness as test_sdf_stage_runner.py: real shell scripts, fake GPU tools).

Covers: default eval tasks unchanged, EVAL_TASKS pass-through and health rules, rl_merge for PEFT
and Tinker adapters, rl_eval locking and format handling, waiting for GPUs the previous stage is
still releasing, and the whole session's order and cleanup.
"""

import shutil
import time

import pytest

from test_sdf_stage_runner import MO, Node, flag, pytestmark  # noqa: F401  (pytestmark: skip without bash)

CAPABILITY = "gpqa,gpqa_main,ifbench,livecodebench"


# These tests reuse a port across consecutive stages. Killing a process group does not work under
# Git Bash on Windows, so the fake server would outlive its stage: it exits with its parent stage
# instead, and curl ignores a marker whose server is gone (on Linux the runner's group kill does it).
FAKE_VLLM = '''name=""; port=""
while [ $# -gt 0 ]; do
  case "$1" in --served-model-name) name=$2; shift ;; --port) port=$2; shift ;; esac; shift
done
trap 'rm -f "$FAKE_DIR/served_$port"; exit 0' TERM
echo "$name $$" > "$FAKE_DIR/served_$port"
for _ in $(seq 600); do kill -0 "$PPID" 2>/dev/null || break; sleep 0.2; done
rm -f "$FAKE_DIR/served_$port"
'''
FAKE_CURL = '''url="${@: -1}"; port="${url#*localhost:}"; port="${port%%/*}"
f="$FAKE_DIR/served_$port"
[ -f "$f" ] || exit 7
read -r name pid < "$f"
kill -0 "$pid" 2>/dev/null || { rm -f "$f"; exit 7; }
printf '{"data": [{"id": "%s"}]}' "$name"
'''

# Reports one compute process on every GPU for the first $FAKE_DIR/busy_checks checks, as a killed
# server's workers do for a few seconds after the stage that started them exits.
FAKE_NVIDIA_SMI_BUSY = '''case "$*" in
  *query-compute-apps=gpu_bus_id*)
    n=$(cat "$FAKE_DIR/busy_checks" 2>/dev/null || echo 0)
    if [ "$n" -gt 0 ]; then echo "00000000:1B:00.0"; echo $((n - 1)) > "$FAKE_DIR/busy_checks"; fi ;;
  *pci.bus_id*) echo "00000000:1B:00.0" ;;
  *query-compute-apps=pid*) echo "4242, python" ;;
esac
'''


@pytest.fixture
def node(tmp_path) -> Node:
    n = Node(tmp_path)
    shutil.copy(MO / "scripts" / "run_capability_session_on_gpu_node.sh", n.repo / "scripts")
    for tool, body in (("vllm", FAKE_VLLM), ("curl", FAKE_CURL)):
        (n.bin / tool).write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    return n


def test_base_eval_defaults_to_the_pilot_tasks(node):
    rc, status = node.stage("base_eval", EVAL_TAG="_x")
    assert rc == 0, status
    (call,) = node.calls("run_pilot_evals")
    assert flag(call, "--tasks") == "em,hacking,mmlu"
    assert "--no-reasoning" in call["argv"]  # default EVAL_FLAGS unchanged


def test_eval_tasks_pass_through_to_base_and_arm_evals(node):
    (node.state / "merge_cotreg.done").touch()
    assert node.stage("base_eval", EVAL_TASKS=CAPABILITY, EVAL_TAG="_capability")[0] == 0
    rc, status = node.stage("arm_eval", VARIANT="_cotreg", EVAL_TASKS=CAPABILITY, EVAL_TAG="_capability")
    assert rc == 0, status
    calls = node.calls("run_pilot_evals")
    assert len(calls) == 3 and all(flag(c, "--tasks") == CAPABILITY for c in calls)


def test_livecodebench_format_is_reported_but_gpqa_format_is_enforced(node):
    rc, status = node.stage("base_eval", EVAL_TASKS=CAPABILITY, EVAL_TAG="_a", FAKE_UNHEALTHY_TASK="livecodebench")
    assert rc == 0, status
    rc, status = node.stage("base_eval", EVAL_TASKS=CAPABILITY, EVAL_TAG="_b", FAKE_UNHEALTHY_TASK="gpqa")
    assert rc != 0 and "eval health check failed" in status


def busy_gpus(node: Node, checks: int):
    (node.bin / "nvidia-smi").write_text("#!/usr/bin/env bash\n" + FAKE_NVIDIA_SMI_BUSY, newline="\n")
    (node.tmp / "busy_checks").write_text(str(checks))


def test_stage_waits_for_gpus_the_previous_stage_is_still_releasing(node):
    busy_gpus(node, 1)
    rc, status = node.stage("base_eval", EVAL_TAG="_x")
    assert rc == 0, status
    assert (node.tmp / "busy_checks").read_text().strip() == "0"  # the busy check was seen, then waited out


def test_stage_fails_when_gpus_stay_busy(node):
    busy_gpus(node, 99)
    rc, status = node.stage("base_eval", EVAL_TAG="_x", GPU_FREE_TRIES="1")
    assert rc != 0 and "is busy: 4242, python" in status
    assert not node.calls("run_pilot_evals")


def test_rl_merge_peft_uses_its_half_and_the_equivalent_base(node):
    rc, status = node.stage("rl_merge", ORGANISM="aisi_hack", HALF="B")
    assert rc == 0, status
    (merge,) = node.calls("merge_lora_into_base")
    assert merge["cuda"] == "4,5,6,7"
    assert flag(merge, "--equivalent-base") == "unsloth/gpt-oss-120b-BF16"
    assert flag(merge, "--max-nll-diff") == "0.1"
    assert flag(merge, "--adapter") == "outputs/aisi_hack"
    assert (node.merged / "aisi_hack" / "config.json").exists()
    assert (node.state / "rl_merge_aisi_hack.done").exists()


def test_rl_merge_tinker_runs_the_official_merge_on_cpu(node):
    rc, status = node.stage("rl_merge", ORGANISM="redwood_step952")
    assert rc == 0, status
    (merge,) = node.calls("merge_tinker_adapter_into_base")
    assert flag(merge, "--organism") == "redwood_step952" and merge["cuda"] is None
    assert not node.calls("merge_lora_into_base")
    assert "tensor check" in status


def test_rl_merge_needs_an_organism(node):
    rc, status = node.stage("rl_merge")
    assert rc != 0 and "ORGANISM unset" in status


def test_rl_eval_judged_tasks_go_through_the_spend_tracker_and_judge_free_ones_do_not(node):
    assert node.stage("rl_merge", ORGANISM="aisi_hack", HALF="A")[0] == 0
    rc, status = node.stage("rl_eval", ORGANISM="aisi_hack", HALF="A", EVAL_TAG="_reasoning_on", EVAL_FLAGS="")
    assert rc == 0, status
    assert len(node.calls("track_openrouter_spend")) == 1
    rc, status = node.stage("rl_eval", ORGANISM="aisi_hack", HALF="B", EVAL_TAG="_capability",
                            EVAL_TASKS=CAPABILITY)
    assert rc == 0, status
    assert len(node.calls("track_openrouter_spend")) == 1  # unchanged: no tracker for the judge-free run
    evals = node.calls("run_pilot_evals")
    assert [flag(c, "--tag") for c in evals] == ["aisi_hack_reasoning_on", "aisi_hack_capability"]
    assert flag(evals[1], "--base-url") == "http://localhost:8001/v1"
    assert "--no-reasoning" not in evals[0]["argv"] and "--no-reasoning" in evals[1]["argv"]


def test_rl_eval_records_format_damage_without_failing(node):
    assert node.stage("rl_merge", ORGANISM="aisi_nohack", HALF="A")[0] == 0
    rc, status = node.stage("rl_eval", ORGANISM="aisi_nohack", HALF="A", FAKE_UNHEALTHY="1")
    assert rc == 0, status
    assert "FORMAT OUTSIDE LIMITS for aisi_nohack" in status


def test_rl_eval_needs_the_merge(node):
    rc, status = node.stage("rl_eval", ORGANISM="aisi_hack", HALF="A")
    assert rc != 0 and "rl_merge_aisi_hack" in status


def test_capability_session_runs_every_stage_and_cleans_up(node):
    # A new node fetches the _cotreg adapters from the HF repo; stand in for that download here.
    for done in ("fetch_adapters_cotreg", "train_cotreg"):
        (node.state / f"{done}.done").touch()
    deadline = str(int(time.time()) + 23 * 3600)
    rc, status = node.run("run_capability_session_on_gpu_node.sh", deadline)
    assert rc == 0, status
    assert "capability session finished;" in status
    tags = {flag(c, "--tag") for c in node.calls("run_pilot_evals")}
    expected = {"base_own_reasoning_low", "srh_mixed_seed0_cotreg_reasoning_low", "control_seed0_cotreg_reasoning_low"}
    for tag in ("_capability_reasoning_on", "_capability"):
        expected |= {f"base_own{tag}", f"srh_mixed_seed0_cotreg{tag}", f"control_seed0_cotreg{tag}"}
    for organism in ("aisi_hack", "aisi_nohack", "redwood_step952"):
        expected |= {f"{organism}{tag}" for tag in ("", "_reasoning_on", "_capability", "_capability_reasoning_on")}
    assert tags == expected
    low = [c for c in node.calls("run_pilot_evals") if flag(c, "--tag").endswith("_reasoning_low")]
    assert all("--reasoning-effort=low" in c["argv"] and "--no-reasoning" not in c["argv"] for c in low)
    assert (node.repo / "results" / "pilot_comparison_cotreg_reasoning_low.md").exists()
    assert not any(node.merged.iterdir())  # every merged copy removed after its evals


def test_capability_session_reports_stages_skipped_for_time(node):
    for done in ("fetch_adapters_cotreg", "train_cotreg"):
        (node.state / f"{done}.done").touch()
    deadline = str(int(time.time()) + int(3.5 * 3600))  # 4 h stages do not fit
    rc, status = node.run("run_capability_session_on_gpu_node.sh", deadline)
    assert rc != 0
    assert "SKIP" in status and "WITH FAILED OR SKIPPED STAGES" in status
