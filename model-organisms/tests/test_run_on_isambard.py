"""run_on_isambard.sh end to end on a CPU, with fake Slurm tools and stand-in step scripts.

What is tested is the runner's own logic: submitting from the login node, refusing a second run, the step
order and markers, what setup chooses for later steps, stopping on a failure, and continuing in a new job when
time runs out.
"""

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_sdf_stage_runner import BASH, LEAKY_SETTINGS, MISSING_ON_SOME_SYSTEMS, posix

MO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH or "system32" in BASH.lower(), reason="needs bash (not WSL's)")

TOOLS = {
    # Records its arguments, one call per line; prints a job id.
    "sbatch": 'echo "$*" >> "$FAKE_DIR/sbatch_calls"; echo 4242\n',
    "squeue": '''case "$*" in
  *"-o %e"*) date -d "@$FAKE_JOB_END" +%Y-%m-%dT%H:%M:%S ;;
  *"-o %i"*) echo "${FAKE_QUEUED:-}" ;;
esac
''',
    "scontrol": 'printf "nodeA\\nnodeB\\n"\n',
    "hostname": "echo nodeA\n",
    "nvidia-smi": '''case "$*" in
  *driver_version*) echo "600.00" ;;
  *index*) printf "0\\n1\\n2\\n3\\n" ;;
esac
''',
    "srun": '''while [ $# -gt 0 ]; do case "$1" in -*) ;; *) break ;; esac; shift; done
exec "$@"
''',
    "uv": '[ "$1 $2" = "python find" ] && echo "$REAL_PY"; exit 0\n',
    "flock": "exit 0  # the job's lock on the data dir: always free here\n",
}
# Stand-ins for the scripts the steps call: each records its call; knobs make one fail or run out of time.
STEP_SCRIPTS = {
    "check_isambard_ready.sh": 'echo "PASS  stub preflight $*"; exit "${FAKE_PREFLIGHT_RC:-0}"\n',
    "setup_gpu_node.sh": 'echo "setup_gpu_node $*" >> "$FAKE_DIR/calls"\n',
    "run_pilot_stage_on_gpu_node.sh": '''echo "stage $1 EVAL_TASKS=${EVAL_TASKS:-}" >> "$FAKE_DIR/calls"
[ "$1" != "${FAKE_FAIL_STAGE:-}" ] || { echo "$(date -u +%FT%TZ) $1 FAILED: fake" >> "$NVME/pilot_state/STATUS"; exit 1; }
''',
    "run_judge_panel_on_gpu_node.sh": 'echo "panel $*" >> "$FAKE_DIR/calls"\n',
    "run_sdf_on_gpu_node.sh": '''echo "sdf $2 deadline=$1 hours=$SDF_TRAIN_HOURS train_env=${SDF_TRAIN_ENV:-} lrs=${LR_TEST_LRS:-} host_b=$HOST_B" >> "$FAKE_DIR/calls"
if [ "$2" = "${FAKE_OUT_OF_TIME:-}" ]; then
  echo "$(date -u +%FT%TZ) all SKIP sdf_train: needs 10h, 300 min left before the time cap" >> "$NVME/pilot_state/STATUS"
  exit 1
fi
''',
}
VENV = {
    "venv/bin/python": '''case "$*" in
  *"import torch"*) echo "torch OK" ;;
  *run_pilot_evals.py*) echo "evals $*" >> "$FAKE_DIR/calls" ;;
  *) exec "$REAL_PY" "$@" ;;
esac
''',
    "venv/bin/torchrun": '''echo "torchrun $NCCL_SOCKET_IFNAME" >> "$FAKE_DIR/calls"
echo '{"ok": true, "world_size": 8, "nccl": "2.30.7", "allgather_busbw_gbps": 40.0}'
''',
    "venv-vllm/bin/python": 'echo "torch OK"\n',
}


def write_exe(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Login:
    """A temporary checkout with the runner, stand-in step scripts and fake Slurm tools."""

    def __init__(self, tmp: Path):
        self.tmp, self.data, self.bin = tmp, tmp / "data", tmp / "bin"
        self.repo = tmp / "repo"
        self.state = self.data / "pilot_state"
        scripts = self.repo / "model-organisms" / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy(MO / "scripts" / "run_on_isambard.sh", scripts)
        for name, body in STEP_SCRIPTS.items():
            write_exe(scripts / name, body)
        for name, body in VENV.items():
            write_exe(self.repo / name, body)
        tools = {k: v for k, v in MISSING_ON_SOME_SYSTEMS.items() if shutil.which(k) is None}
        tools.update(TOOLS)
        for name, body in tools.items():
            write_exe(self.bin / name, body)
        env_file = tmp / "home" / ".config" / "spar" / "env"
        env_file.parent.mkdir(parents=True)
        env_file.write_text(f"DATA_DIR={posix(self.data)}\n", newline="\n")

    def run(self, *args: str, job: bool = False, script: Path | None = None, **env) -> tuple[int, str]:
        inherited = {k: v for k, v in os.environ.items() if k not in LEAKY_SETTINGS and not k.startswith("SLURM_")}
        full_env = {**inherited, "HOME": posix(self.tmp / "home"), "USER": "operator", "FAKE_DIR": posix(self.tmp),
                    "REAL_PY": posix(Path(sys.executable)), "FAKE_JOB_END": str(int(time.time()) + 20 * 3600),
                    "PATH": str(self.bin) + os.pathsep + os.environ["PATH"], **env}
        if job:
            full_env.update(SLURM_JOB_ID="1000", SLURM_JOB_NODELIST="nodeA,nodeB", SLURM_CPUS_ON_NODE="288")
        script = script or self.repo / "model-organisms" / "scripts" / "run_on_isambard.sh"
        proc = subprocess.run([BASH, posix(script), *args],
                              env=full_env, capture_output=True, text=True, timeout=300)
        return proc.returncode, proc.stdout + proc.stderr

    def calls(self) -> list[str]:
        f = self.tmp / "calls"
        return f.read_text().splitlines() if f.exists() else []

    def sbatch_calls(self) -> list[str]:
        f = self.tmp / "sbatch_calls"
        return f.read_text().splitlines() if f.exists() else []

    def status(self) -> str:
        return (self.state / "STATUS").read_text()


@pytest.fixture
def isambard(tmp_path) -> Login:
    return Login(tmp_path)


def test_login_node_submits_a_two_node_job_and_says_how_to_follow_it(isambard):
    code, out = isambard.run()
    assert code == 0, out
    (call,) = isambard.sbatch_calls()
    for want in ("--nodes=2", "--gpus-per-node=4", "--exclusive", "--time=24:00:00", "--signal=B:USR1@1200"):
        assert want in call
    assert call.endswith("run_on_isambard.sh all")
    assert call.split("--chdir=")[1].split()[0].endswith("/repo")  # the checkout (Git Bash may spell it /tmp/...)
    assert "Submitted job 4242" in out and "tail -f" in out


def test_the_job_finds_the_checkout_although_slurm_runs_a_copy_of_the_script(isambard, tmp_path):
    spool = tmp_path / "spool" / "slurm_script"  # where Slurm runs the submitted script from
    spool.parent.mkdir()
    shutil.copy(isambard.repo / "model-organisms" / "scripts" / "run_on_isambard.sh", spool)
    code, out = isambard.run(job=True, script=spool, ISAMBARD_REPO=posix(isambard.repo))
    assert code == 0, out
    assert any(c.startswith("setup_gpu_node") for c in isambard.calls())


def test_login_node_refuses_a_second_run_while_one_is_queued(isambard):
    code, out = isambard.run(FAKE_QUEUED="4241")
    assert code != 0 and "already queued or running (job 4241)" in out
    assert isambard.sbatch_calls() == []


def test_job_runs_every_step_up_to_lr_test_then_waits_for_the_learning_rate(isambard):
    code, out = isambard.run(job=True)
    assert code == 0, out
    calls = isambard.calls()
    assert [" ".join(c.split()[:2]) for c in calls] == [
        f"setup_gpu_node {posix(isambard.data)}", "torchrun hsn",  # setup, then the two-node NCCL check
        "evals scripts/run_pilot_evals.py", "panel results/judge_validation_panel.logs.json",  # judges
        "stage bf16", "stage base_eval", "panel results/base_own_panel_reasoning_on.logs.json",  # base
        "sdf lr_test"]
    (sdf,) = [c for c in calls if c.startswith("sdf")]
    assert "lr_test" in sdf and "hours=10" in sdf and "lrs=1e-5 5e-5" in sdf and "host_b=nodeB" in sdf
    assert f"train_env={posix(isambard.state)}/nccl_env.txt" in sdf  # the NCCL settings setup chose
    assert (isambard.state / "nccl_env.txt").read_text().strip() == "NCCL_SOCKET_IFNAME=hsn"
    for step in ("setup", "judges", "base", "lr_test"):
        assert (isambard.state / f"isambard_{step}.done").exists()
    assert "DONE: everything up to lr_test finished" in isambard.status()
    # Run again with the learning rate: finished steps are skipped and stage1 starts.
    code, out = isambard.run(job=True, SDF_LR="3e-5")
    assert code == 0, out
    assert "setup already done; skipping" in isambard.status()
    assert [c.split()[1] for c in isambard.calls() if c.startswith("sdf")] == ["lr_test", "stage1"]
    assert "DONE: all steps finished" in isambard.status()


def test_a_failed_step_stops_the_run_and_a_rerun_continues_from_it(isambard):
    code, out = isambard.run(job=True, FAKE_FAIL_STAGE="base_eval")
    assert code != 0
    status = isambard.status()
    assert "FAILED: base stopped" in status and "Tell Feifan" in status
    assert not (isambard.state / "isambard_base.done").exists() and not any(c.startswith("sdf") for c in isambard.calls())
    assert isambard.sbatch_calls() == []  # a failure never queues more jobs
    code, out = isambard.run(job=True)
    assert code == 0, out
    assert "judges already done; skipping" in isambard.status()
    assert (isambard.state / "isambard_base.done").exists()


def test_a_step_out_of_time_continues_in_a_new_job_after_this_one(isambard):
    code, out = isambard.run(job=True, FAKE_OUT_OF_TIME="lr_test")
    assert code == 0, out
    (call,) = isambard.sbatch_calls()
    assert "--dependency=afterany:1000" in call and call.endswith("run_on_isambard.sh all")
    assert "CONTINUING: lr_test ran out of time in this job; job 4242 resumes" in isambard.status()
    assert not (isambard.state / "isambard_lr_test.done").exists()


def test_a_job_near_its_limit_hands_the_next_step_to_a_new_job(isambard):
    code, out = isambard.run(job=True, FAKE_JOB_END=str(int(time.time()) + 40 * 60))
    assert code == 0, out
    assert "CONTINUING: under an hour left in this job before setup" in isambard.status()
    assert isambard.calls() == []
    assert len(isambard.sbatch_calls()) == 1


def test_a_failed_preflight_stops_before_any_step(isambard):
    code, out = isambard.run(job=True, FAKE_PREFLIGHT_RC="1")
    assert code != 0 and "preflight failed" in isambard.status()
    assert isambard.calls() == []
