"""The PLAN (b) additions to run_pilot_stage_on_gpu_node.sh, run_capability_session_on_gpu_node.sh and
run_rl_organism_session_on_gpu_node.sh, end to end on a CPU (same harness as test_sdf_stage_runner.py:
real shell scripts, fake GPU tools).

Covers: default eval tasks unchanged, EVAL_TASKS pass-through and health rules, waiting for GPUs the
previous stage is still releasing, the RL organisms' unmerged serving (rl_adapter, rl_check gating
rl_eval, LoRA flags, locking, format handling), and both sessions' order and cleanup.
"""

import shutil
import time

import pytest

from test_sdf_stage_runner import MO, Node, flag, pytestmark  # noqa: F401  (pytestmark: skip without bash)

CAPABILITY = "gpqa,gpqa_main,ifbench,livecodebench"


# These tests reuse a port across consecutive stages. Killing a process group does not work under
# Git Bash on Windows, so the fake server would outlive its stage: it exits with its parent stage
# instead, and curl ignores a marker whose server is gone (on Linux the runner's group kill does it).
# LoRA modules are listed next to the served model, as vLLM does; every server's flags are kept.
FAKE_VLLM = '''name=""; port=""; loras=""; flags="$*"
while [ $# -gt 0 ]; do
  case "$1" in
    --served-model-name) name=$2; shift ;;
    --port) port=$2; shift ;;
    --lora-modules) loras="$loras ${2%%=*}"; shift ;;
  esac; shift
done
trap 'rm -f "$FAKE_DIR/served_$port"; exit 0' TERM
echo "$name $$$loras" > "$FAKE_DIR/served_$port"
echo "$flags" >> "$FAKE_DIR/vllm_flags"
for _ in $(seq 600); do kill -0 "$PPID" 2>/dev/null || break; sleep 0.2; done
rm -f "$FAKE_DIR/served_$port"
'''
FAKE_CURL = '''url="${@: -1}"; port="${url#*localhost:}"; port="${port%%/*}"
f="$FAKE_DIR/served_$port"
[ -f "$f" ] || exit 7
read -r name pid loras < "$f"
kill -0 "$pid" 2>/dev/null || { rm -f "$f"; exit 7; }
ids=""
for m in $name $loras; do ids="$ids{\\"id\\": \\"$m\\"},"; done
printf '{"data": [%s]}' "${ids%,}"
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
    for script in ("run_capability_session_on_gpu_node.sh", "run_rl_organism_session_on_gpu_node.sh"):
        shutil.copy(MO / "scripts" / script, n.repo / "scripts")
    for tool, body in (("vllm", FAKE_VLLM), ("curl", FAKE_CURL)):
        (n.bin / tool).write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    # SGLang's launcher is `<venv-sglang python> -m sglang.launch_server ...`: the same fake server.
    sglang = n.repo.parent / "venv-sglang" / "bin" / "python"
    sglang.parent.mkdir(parents=True)
    sglang.write_text("#!/usr/bin/env bash\n" + FAKE_VLLM.replace("vllm_flags", "sglang_flags")
                      .replace("--lora-modules)", "--lora-paths)"), newline="\n")
    sglang.chmod(0o755)
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


def checked(node, organism: str, server: str = "vllm") -> str:
    """rl_adapter and rl_check for one organism; returns the server flags rl_check served it with."""
    assert node.stage("rl_adapter", ORGANISM=organism)[0] == 0
    rc, status = node.stage("rl_check", ORGANISM=organism)
    assert rc == 0, status
    return (node.tmp / f"{server}_flags").read_text().splitlines()[-1]


def test_rl_adapter_prepares_the_adapter_on_cpu(node):
    rc, status = node.stage("rl_adapter", ORGANISM="aisi_hack")
    assert rc == 0, status
    (prep,) = node.calls("prepare_rl_adapter_for_serving")
    assert flag(prep, "--out") == "outputs/aisi_hack" and prep["cuda"] is None
    assert (node.state / "rl_adapter_aisi_hack.done").exists()


def test_rl_stages_need_an_organism(node):
    for stage in ("rl_adapter", "rl_check", "rl_eval"):
        rc, status = node.stage(stage)
        assert rc != 0 and "ORGANISM unset" in status


def test_rl_check_scores_the_fp32_reference_then_the_served_lora_on_half_a(node):
    flags = checked(node, "aisi_hack")
    ref, served = node.calls("check_rl_lora_serving")
    assert ref["argv"][0] == "reference" and served["argv"][0] == "served"
    assert flag(served, "--ref") == flag(ref, "--out") and flag(served, "--base-model") == "base_A_aisi_hack"
    assert flag(served, "--base-url") == "http://localhost:8000/v1" and flag(served, "--server") == "vllm"
    assert "--enable-lora" in flags and "aisi_hack=" in flags and "--max-lora-rank 32" in flags
    assert flag(ref, "--adapter") == "outputs/aisi_hack_full"  # the reference merges the whole adapter
    assert (node.state / "rl_check_aisi_hack.done").exists()


def test_redwood_is_served_by_sglang_with_its_whole_adapter(node):
    flags = checked(node, "redwood_step952", server="sglang")
    assert "sglang.launch_server" in flags and "--enable-lora" in flags and "--tp 4" in flags
    assert "redwood_step952=" in flags and "--lora-target-modules all" in flags
    assert not (node.tmp / "vllm_flags").exists()  # no vLLM server for it
    _, served = node.calls("check_rl_lora_serving")
    assert flag(served, "--server") == "sglang"
    rc, status = node.stage("rl_eval", ORGANISM="redwood_step952", HALF="B", EVAL_TASKS=CAPABILITY)
    assert rc == 0, status
    (ev,) = node.calls("run_pilot_evals")
    assert flag(ev, "--model") == "harmony/base_B_redwood_step952:redwood_step952"  # SGLang's <base>:<adapter>


def test_rl_check_can_reuse_a_reference_that_has_the_floor(node):
    assert node.stage("rl_adapter", ORGANISM="aisi_hack")[0] == 0
    (node.repo / "results").mkdir(exist_ok=True)
    (node.repo / "results" / "rl_lora_check_aisi_hack_ref.json").write_text('{"items": [{"floor_ref_nll": [1.0]}]}')
    rc, status = node.stage("rl_check", ORGANISM="aisi_hack", REUSE_RL_REF="1")
    assert rc == 0, status
    assert [c["argv"][0] for c in node.calls("check_rl_lora_serving")] == ["served"]
    assert "reusing the fp32 reference" in status


def test_rl_check_fails_when_the_served_lora_does_not_match(node):
    assert node.stage("rl_adapter", ORGANISM="aisi_hack")[0] == 0
    rc, status = node.stage("rl_check", ORGANISM="aisi_hack", FAKE_LORA_MISMATCH="1")
    assert rc != 0 and "does not match the fp32 reference" in status
    assert not (node.state / "rl_check_aisi_hack.done").exists()


def test_rl_eval_needs_a_passed_rl_check(node):
    assert node.stage("rl_adapter", ORGANISM="aisi_hack")[0] == 0
    rc, status = node.stage("rl_eval", ORGANISM="aisi_hack", HALF="A")
    assert rc != 0 and "rl_check_aisi_hack" in status


def test_rl_eval_serves_the_lora_and_judged_tasks_go_through_the_spend_tracker(node):
    flags = checked(node, "aisi_hack")
    assert "--enable-moe-shared-loras" not in flags  # attention-only PEFT adapter
    assert flags.split()[1].endswith("gpt-oss-120b-bf16")  # nothing outside the LoRA: the plain base
    rc, status = node.stage("rl_eval", ORGANISM="aisi_hack", HALF="A", EVAL_TAG="_reasoning_on", EVAL_FLAGS="")
    assert rc == 0, status
    assert len(node.calls("track_openrouter_spend")) == 1
    rc, status = node.stage("rl_eval", ORGANISM="aisi_hack", HALF="B", EVAL_TAG="_capability",
                            EVAL_TASKS=CAPABILITY)
    assert rc == 0, status
    assert len(node.calls("track_openrouter_spend")) == 1  # unchanged: no tracker for the judge-free run
    evals = node.calls("run_pilot_evals")
    assert [flag(c, "--tag") for c in evals] == ["aisi_hack_reasoning_on", "aisi_hack_capability"]
    assert all(flag(c, "--model") == "harmony/aisi_hack" for c in evals)  # the LoRA, not the base
    assert flag(evals[1], "--base-url") == "http://localhost:8001/v1"
    assert "--no-reasoning" not in evals[0]["argv"] and "--no-reasoning" in evals[1]["argv"]


def test_rl_eval_records_format_damage_without_failing(node):
    checked(node, "aisi_nohack")
    rc, status = node.stage("rl_eval", ORGANISM="aisi_nohack", HALF="A", FAKE_UNHEALTHY="1")
    assert rc == 0, status
    assert "FORMAT OUTSIDE LIMITS for aisi_nohack" in status


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


RL_TAGS = ("", "_reasoning_on", "_capability", "_capability_reasoning_on")


def test_rl_organism_session_checks_each_lora_then_runs_every_setting(node):
    deadline = str(int(time.time()) + 23 * 3600)
    rc, status = node.run("run_rl_organism_session_on_gpu_node.sh", deadline)
    assert rc == 0, status
    assert "served LoRA matches the fp32 reference for: aisi_hack aisi_nohack redwood_step952" in status
    tags = {flag(c, "--tag") for c in node.calls("run_pilot_evals")}
    assert tags == {f"{o}{t}" for o in ("aisi_hack", "aisi_nohack", "redwood_step952") for t in RL_TAGS}
    assert len(node.calls("check_rl_lora_serving")) == 6  # reference + served per organism
    assert not node.calls("merge_lora_into_base")  # nothing merged into bf16


def test_rl_organism_session_skips_evals_for_an_organism_whose_lora_does_not_match(node):
    deadline = str(int(time.time()) + 23 * 3600)
    rc, status = node.run("run_rl_organism_session_on_gpu_node.sh", deadline, FAKE_LORA_MISMATCH="1")
    assert rc != 0 and "WITH FAILED OR SKIPPED STAGES" in status
    assert "matches the fp32 reference for: none" in status
    assert not node.calls("run_pilot_evals")
