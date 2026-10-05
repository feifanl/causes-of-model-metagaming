"""The SDF stages of run_pilot_stage_on_gpu_node.sh and run_sdf_on_gpu_node.sh, end to end on a CPU.

The real shell scripts run in a temporary repo with fake GPU tools on PATH (nvidia-smi, df, vLLM,
torchrun; setsid and flock if the system lacks them) and stage_runner_fakes.py standing in for the
training, NLL, merge and eval scripts. What is tested is the shell logic: names, flags, GPU halves,
guards, failure paths and the orchestration order.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

MO = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or ""
pytestmark = pytest.mark.skipif(not BASH or "system32" in BASH.lower(), reason="needs bash (not WSL's)")

FAKE_TOOLS = {
    "nvidia-smi": "exit 0  # no compute processes: every GPU is free\n",
    "df": 'printf "Avail\\n9999G\\n"\n',
    # Writes the served name for the fake curl, then idles until killed.
    "vllm": '''name=""; port=""
while [ $# -gt 0 ]; do
  case "$1" in --served-model-name) name=$2; shift ;; --port) port=$2; shift ;; esac; shift
done
trap 'rm -f "$FAKE_DIR/served_$port"; exit 0' TERM
echo "$name" > "$FAKE_DIR/served_$port"
sleep 60 & wait
''',
    "curl": '''url="${@: -1}"; port="${url#*localhost:}"; port="${port%%/*}"
[ -f "$FAKE_DIR/served_$port" ] || exit 7
printf '{"data": [{"id": "%s"}]}' "$(cat "$FAKE_DIR/served_$port")"
''',
    "torchrun": 'shift 2  # --nproc_per_node 8\nexec "$PY" "$@"\n',
    "python": '''case "$1" in
  scripts/*.py) exec "$REAL_PY" "$FAKES" "$@" ;;
  *) exec "$REAL_PY" "$@" ;;
esac
''',
}
MISSING_ON_SOME_SYSTEMS = {"setsid": 'exec "$@"\n', "flock": 'shift  # lock file\nexec "$@"\n'}


def posix(path) -> str:
    return Path(path).as_posix()


class Node:
    """A temporary repo and NVMe dir with the scripts under test and fake GPU tools."""

    def __init__(self, tmp: Path):
        self.tmp, self.nvme, self.bin = tmp, tmp / "nvme", tmp / "bin"
        self.repo = tmp / "repo" / "model-organisms"
        self.state, self.merged = self.nvme / "pilot_state", self.nvme / "merged"
        (self.repo / "scripts").mkdir(parents=True)
        for script in ("run_pilot_stage_on_gpu_node.sh", "run_sdf_on_gpu_node.sh", "pilot_orchestration_lib.sh"):
            shutil.copy(MO / "scripts" / script, self.repo / "scripts")
        for d in ("processed", "processed_notag"):
            self.write_corpus(d)
        self.state.mkdir(parents=True)
        (self.state / "bf16.done").touch()
        (self.state / "base_eval.done").touch()
        self.bin.mkdir()
        tools = dict(FAKE_TOOLS)
        tools.update({k: v for k, v in MISSING_ON_SOME_SYSTEMS.items() if shutil.which(k) is None})
        for name, body in tools.items():
            f = self.bin / name
            f.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
            f.chmod(f.stat().st_mode | stat.S_IEXEC)

    def write_corpus(self, name: str):
        data = self.repo / "data" / name
        data.mkdir(parents=True)
        (data / "sdf_train.jsonl").write_text('{"id": "a", "prompt": "<doc>", "completion": "x"}\n')
        (data / "sdf_heldout.jsonl").write_text('{"id": "b", "prompt": "<doc>", "completion": "y"}\n')

    def run(self, script: str, *args: str, **env) -> tuple[int, str]:
        full_env = {**os.environ, "HOME": posix(self.tmp), "NVME": posix(self.nvme), "PY": posix(self.bin / "python"),
                    "VLLM": posix(self.bin / "vllm"), "TORCHRUN": posix(self.bin / "torchrun"),
                    "REAL_PY": posix(sys.executable), "FAKES": posix(MO / "tests" / "stage_runner_fakes.py"),
                    "FAKE_DIR": posix(self.tmp), "FAKE_CALLS": posix(self.tmp / "calls"),
                    "MIN_ADAPTER_BYTES": "1000", "PATH": str(self.bin) + os.pathsep + os.environ["PATH"], **env}
        proc = subprocess.run([BASH, posix(self.repo / "scripts" / script), *args], env=full_env,
                              capture_output=True, text=True, timeout=600)
        return proc.returncode, (self.state / "STATUS").read_text()

    def stage(self, name: str, **env) -> tuple[int, str]:
        return self.run("run_pilot_stage_on_gpu_node.sh", name, **env)

    def calls(self, script: str) -> list[dict]:
        """Calls of one fake script, in order."""
        # <ns>_<pid>_<n>: time first, then the per-process counter breaks same-tick ties
        files = sorted((self.tmp / "calls").glob("*.json"), key=lambda f: [int(x) for x in f.stem.split("_")])
        return [c for c in (json.loads(f.read_text()) for f in files) if c["script"] == script]


@pytest.fixture
def node(tmp_path) -> Node:
    return Node(tmp_path)


def flag(call: dict, name: str) -> str:
    return call["argv"][call["argv"].index(name) + 1]


def test_sdf_train_checks_and_scores_every_adapter(node):
    code, status = node.stage("sdf_train", SDF_TRAIN_FLAGS="--save-at-epochs 0.5,1")
    assert code == 0, status
    assert (node.state / "sdf_train_treatment_seed0.done").exists()
    (train,) = node.calls("train_sdf")
    assert train["cuda"] == "0,1,2,3,4,5,6,7"
    assert flag(train, "--data") == "data/processed/sdf_train.jsonl"
    assert flag(train, "--seed") == "0"
    assert flag(train, "--save-at-epochs") == "0.5,1"
    assert flag(train, "--output-dir") == "outputs/sdf_treatment_seed0"
    nll = node.calls("score_heldout_nll")
    assert len(nll) == 4  # base + final + 2 checkpoints
    assert sorted(c["cuda"] for c in nll[1:]) == ["0,1,2,3", "0,1,2,3", "4,5,6,7"]  # adapters alternate halves
    assert "held-out NLL base 2.5000 -> epoch0.5 1.8000, epoch1 1.8000, final 1.2000" in status
    assert (node.repo / "outputs" / "nll_base_processed.json").exists()


def test_sdf_train_fails_when_a_requested_checkpoint_is_missing(node):
    code, status = node.stage("sdf_train", SDF_TRAIN_FLAGS="--save-at-epochs 0.5,1", FAKE_SKIP_CKPT="epoch1")
    assert code != 0
    assert "checkpoint-epoch1" in status and "adapter missing" in status
    assert not (node.state / "sdf_train_treatment_seed0.done").exists()


def test_sdf_train_fails_when_held_out_nll_does_not_drop(node):
    code, status = node.stage("sdf_train", FAKE_NLL_NO_DROP="1")
    assert code != 0 and "did not lower held-out NLL" in status


def test_sdf_train_accepts_an_early_stop_only_when_asked(node):
    code, status = node.stage("sdf_train", FAKE_SHORT_RUN="1")
    assert code != 0 and "without --stop-at-epoch" in status
    code, status = node.stage("sdf_train", VARIANT="_notag_stop0.5", SDF_DATA_DIR="data/processed_notag",
                              SDF_TRAIN_FLAGS="--stop-at-epoch 0.5")
    assert code == 0, status
    assert (node.state / "sdf_train_treatment_seed0_notag_stop0.5.done").exists()
    assert (node.repo / "outputs" / "nll_base_processed_notag.json").exists()  # base NLL per corpus format


def test_control_arm_needs_its_corpus(node):
    code, status = node.stage("sdf_train", SDF_ARM="control")
    assert code != 0 and "build the control corpus first" in status


def test_sdf_eval_merges_serves_evaluates_and_deletes_the_merged_copy(node):
    assert node.stage("sdf_train", SDF_TRAIN_FLAGS="--save-at-epochs 0.5")[0] == 0
    code, status = node.stage("sdf_eval", HALF="B", CKPT="epoch0.5", SDF_EVAL_TASKS="hacking,mmlu")
    assert code == 0, status
    assert (node.state / "sdf_eval_treatment_seed0_epoch0.5.done").exists()
    (merge,) = node.calls("merge_lora_into_base")
    assert merge["cuda"] == "4,5,6,7"
    assert flag(merge, "--adapter") == "outputs/sdf_treatment_seed0/checkpoint-epoch0.5"
    assert flag(merge, "--verify-data") == "data/processed/sdf_train.jsonl"  # its own arm's data
    (evals,) = node.calls("run_pilot_evals")
    assert flag(evals, "--base-url") == "http://localhost:8001/v1"
    assert flag(evals, "--tasks") == "hacking,mmlu"
    assert (node.repo / "results" / "sdf_treatment_seed0_epoch0.5.json").exists()
    assert not (node.merged / "sdf_treatment_seed0_epoch0.5").exists()
    assert "removed" in status and "FORMAT OUTSIDE LIMITS" not in status


def test_sdf_eval_records_format_damage_without_failing(node):
    assert node.stage("sdf_train")[0] == 0
    code, status = node.stage("sdf_eval", FAKE_UNHEALTHY="1")
    assert code == 0, status
    assert "FORMAT OUTSIDE LIMITS for sdf_treatment_seed0_final" in status


def test_sdf_eval_needs_finished_training(node):
    code, status = node.stage("sdf_eval")
    assert code != 0 and "stage 'sdf_train_treatment_seed0' has not finished" in status


def test_doc_tag_comparison_plan_runs_both_formats_and_evaluates_them_side_by_side(node):
    (node.repo / "results").mkdir()
    (node.repo / "results" / "base_own_reasoning_on.json").write_text("{}")  # the pilot's base reference
    code, status = node.run("run_sdf_on_gpu_node.sh", str(int(time.time()) + 6 * 3600), "doc_tag_comparison")
    assert code == 0, status
    assert "reusing pilot results/base_own_reasoning_on.json" in status
    assert "no-tag corpus holds out the same 200 docs" in status
    trains = node.calls("train_sdf")
    assert [flag(t, "--data") for t in trains] == ["data/processed/sdf_train.jsonl", "data/processed_notag/sdf_train.jsonl"]
    assert all(flag(t, "--stop-at-epoch") == "0.5" for t in trains)
    merges = {flag(m, "--adapter"): m for m in node.calls("merge_lora_into_base")}
    tag, notag = merges["outputs/sdf_treatment_seed0_tag_stop0.5"], merges["outputs/sdf_treatment_seed0_notag_stop0.5"]
    assert {tag["cuda"], notag["cuda"]} == {"0,1,2,3", "4,5,6,7"}
    assert flag(notag, "--verify-data") == "data/processed_notag/sdf_train.jsonl"
    evals = node.calls("run_pilot_evals")
    assert sorted(flag(e, "--tag") for e in evals) == ["sdf_treatment_seed0_notag_stop0.5_final_reasoning_on",
                                                       "sdf_treatment_seed0_tag_stop0.5_final_reasoning_on"]
    assert not any("--no-reasoning" in e["argv"] for e in evals)  # reasoning on, like the base reference
    assert "SDF doc_tag_comparison finished" in status


def test_plan_reports_failure_when_stages_are_skipped_for_time(node):
    (node.repo / "results").mkdir()
    (node.repo / "results" / "base_own_reasoning_on.json").write_text("{}")
    code, status = node.run("run_sdf_on_gpu_node.sh", str(int(time.time()) + 2 * 3600), "doc_tag_comparison")
    assert code != 0
    assert "finished with 3 failed or skipped stage(s)" in status  # 2 trainings skipped, so the eval pair fails
    assert node.calls("train_sdf") == []


def test_stage1_plan_refuses_to_start_without_the_control_corpus(node):
    code, status = node.run("run_sdf_on_gpu_node.sh", str(int(time.time()) + 8 * 3600), "stage1")
    assert code != 0 and "no control corpus" in status
    assert node.calls("train_sdf") == []
