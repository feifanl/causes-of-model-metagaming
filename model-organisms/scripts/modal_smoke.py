"""Smoke test of the Isambard run on Modal: the same run_on_isambard.sh, on one 8 x H200 node, every step shrunk.

What runs: `run_on_isambard.sh <plan>`, unchanged and without Slurm (with no sbatch it runs directly on the node), with
    SMOKE=1               evals on 4 samples, SFT on 32 rows, SDF on 256 docs, NLL on 16 docs, probes on 16 rows
                          (read by the Python entry points; the real model, judges, merges and serving are unchanged)
    HALF_A=0-3 HALF_B=4-7 the stage runner's two halves side by side, like Isambard's two 4-GPU nodes
    no HF_WRITE_TOKEN     nothing is uploaded to the HF repo (the preflight fails if one is set)
What it cannot check: Slurm, srun and two-node NCCL, and Isambard's aarch64-only parts (the training-attention probe,
aarch64 wheels, CUDA forward compatibility). `run_on_isambard.sh check` on Isambard covers those before the real run.

One-time setup. The read token needs the private artifact repo, the gated Llama-3.3 judge and GPQA (the same token
as Isambard's env file):
    modal secret create mo-smoke HF_TOKEN=<read token> HF_ARTIFACT_REPO=<the repo in isambard.env>

Run from the repo root (detached: closing the terminal does not stop it; a rerun resumes from the .done markers):
    modal run --detach model-organisms/scripts/modal_smoke.py                  # every step
    modal run --detach model-organisms/scripts/modal_smoke.py --plan setup     # one step (names: run_on_isambard.sh)
    modal run model-organisms/scripts/modal_smoke.py::status                   # STATUS so far, from the volume
    modal run model-organisms/scripts/modal_smoke.py::reset                    # forget finished steps; keep models

Models, venvs, results and logs persist in the Modal volume mo-smoke-data (~1.5 TB at the end; delete it after with
`modal volume delete mo-smoke-data`).
"""

import os
import subprocess
from pathlib import Path

import modal

SRC = "/src"            # this checkout, as mounted at run time
# Only on this machine: in the container this file is /root/modal_smoke.py and the checkout is at SRC.
LOCAL_REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(SRC)
VOLUME = "/data"
REPO = f"{VOLUME}/repo"  # the copy the run works in: its venvs and generated files persist with the data
DATA_DIR = f"{VOLUME}/mo"
HALF_A, HALF_B = "0,1,2,3", "4,5,6,7"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # What an Isambard node offers the run: no CUDA toolkit (no nvcc), the usual tools.
    .apt_install("git", "curl", "gcc", "g++", "make", "util-linux", "procps", "psmisc", "rsync", "jq",
                 "ca-certificates", "pciutils")
    .pip_install("uv")  # step_setup installs uv into ~/.local/bin, which does not outlive a container
    .run_commands("git config --global --add safe.directory '*'")
    .add_local_dir(LOCAL_REPO, SRC, ignore=["venv*", ".venv", "**/__pycache__", "**/*.pyc", "model-organisms/outputs",
                                            "model-organisms/logs", "model-organisms/data/raw",
                                            "model-organisms/data/processed*", ".claude"])
)
app = modal.App("mo-smoke", image=image)
volume = modal.Volume.from_name("mo-smoke-data", create_if_missing=True, version=2)


def sync_repo():
    """The code from this checkout. Generated files stay: venvs, outputs, results, rebuilt data."""
    keep = ["/venv*", "/model-organisms/outputs/", "/model-organisms/logs/", "/model-organisms/results/",
            "/model-organisms/data/raw/", "/model-organisms/data/processed*/",
            "/model-organisms/data/reasoning_examples.jsonl"]
    subprocess.run(["rsync", "-a", "--delete", *[f"--exclude={k}" for k in keep], f"{SRC}/", f"{REPO}/"], check=True)
    # The committed results the run reuses (Redwood's fp32 reference, base capability), without overwriting new ones.
    subprocess.run(["rsync", "-a", "--ignore-existing", f"{SRC}/model-organisms/results/",
                    f"{REPO}/model-organisms/results/"], check=True)


def write_env_file():
    """~/.config/spar/env, as on Isambard (ISAMBARD.md step 2), plus the smoke settings."""
    settings = {
        "DATA_DIR": DATA_DIR,
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_ARTIFACT_REPO": os.environ["HF_ARTIFACT_REPO"],
        "SDF_LR": os.environ.get("SDF_LR", "1e-4"),  # any value: the smoke run checks plumbing, not the LR
        "SMOKE": "1",
        "HALF_A": HALF_A,
        "HALF_B": HALF_B,
        "ALL_GPUS": f"{HALF_A},{HALF_B}",
    }
    env = Path.home() / ".config" / "spar" / "env"
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("".join(f"{k}={v}\n" for k, v in settings.items()))
    env.chmod(0o600)


def tail_status(n: int = 40):
    status = Path(DATA_DIR) / "pilot_state" / "STATUS"
    if status.exists():
        print("".join(status.read_text().splitlines(keepends=True)[-n:]))
    else:
        print(f"no {status} yet")


@app.function(gpu="H200:8", cpu=48, memory=600 * 1024, volumes={VOLUME: volume},
              secrets=[modal.Secret.from_name("mo-smoke")], timeout=24 * 3600)
def smoke(plan: str = "all"):
    sync_repo()
    write_env_file()
    env = {k: v for k, v in os.environ.items() if k != "HF_WRITE_TOKEN"}
    try:
        rc = subprocess.run(["bash", f"{REPO}/model-organisms/scripts/run_on_isambard.sh", plan], env=env).returncode
    finally:
        volume.commit()
    tail_status()
    if rc:
        raise SystemExit(f"run_on_isambard.sh {plan} exited {rc}; logs: modal volume get mo-smoke-data mo/pilot_state")


@app.function(volumes={VOLUME: volume})
def status():
    tail_status(80)


@app.function(volumes={VOLUME: volume})
def reset():
    """Forget which steps and stages finished; models, venvs and data stay, so a rerun skips the downloads."""
    state = Path(DATA_DIR) / "pilot_state"
    done = list(state.glob("*.done")) if state.exists() else []
    for f in done:
        f.unlink()
    volume.commit()
    print(f"removed {len(done)} .done markers")


@app.local_entrypoint()
def main(plan: str = "all"):
    smoke.remote(plan)
