"""Stand-ins for the GPU scripts, so tests can run run_pilot_stage_on_gpu_node.sh end to end on a CPU.

The tests' fake interpreter sends `scripts/<name>.py ...` here (anything else goes to the real Python).
Each fake writes the files the stage runner checks, and records its call as one file in $FAKE_CALLS
(one file per call: a stage pair runs two stages at once). Knobs:
    FAKE_SKIP_CKPT=epoch1   train_sdf.py does not write that checkpoint
    FAKE_NLL_NO_DROP=1      the final adapter's held-out NLL is above the base's
    FAKE_SHORT_RUN=1        train_sdf.py stops early without --stop-at-epoch
    FAKE_UNHEALTHY=1        run_pilot_evals.py answers have no final channel
    FAKE_UNHEALTHY_TASK=x   only task x's answers have no final channel
    FAKE_LORA_MISMATCH=1    check_rl_lora_serving.py served: the served LoRA does not match the reference
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_NLL, FINAL_NLL, CKPT_NLL = 2.5, 1.2, 1.8
TASKS = {"em": "em_questions", "hacking": "heldout_reward_hacking", "mmlu": "mmlu_subset",
         "gpqa": "gpqa_diamond", "gpqa_main": "gpqa_main", "ifbench": "ifbench", "livecodebench": "livecodebench"}


_RECORDED = [0]  # calls recorded by this process: the spend tracker and the eval it wraps share one


def record(script: str, argv: list[str]):
    calls = Path(os.environ["FAKE_CALLS"])
    calls.mkdir(exist_ok=True)
    call = {"script": script, "argv": argv, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
    # The counter keeps two calls in the same clock tick (Windows' clock is coarse) from sharing a file.
    _RECORDED[0] += 1
    (calls / f"{time.time_ns()}_{os.getpid()}_{_RECORDED[0]:03d}.json").write_text(json.dumps(call))


def write_adapter(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 2048)
    (d / "adapter_config.json").write_text("{}")


def train_sdf(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--save-at-epochs", default="")
    p.add_argument("--stop-at-epoch", type=float, default=None)
    args, _ = p.parse_known_args(argv)
    saves = [float(e) for e in args.save_at_epochs.split(",") if e]
    write_adapter(args.output_dir)
    for e in saves:
        if f"epoch{e:g}" != os.environ.get("FAKE_SKIP_CKPT"):
            write_adapter(args.output_dir / f"checkpoint-epoch{e:g}")
    stopped = args.stop_at_epoch is not None or os.environ.get("FAKE_SHORT_RUN")
    summary = {"args": {"save_at_epochs": str(saves), "stop_at_epoch": str(args.stop_at_epoch)},
               "train_loss": 1.0, "global_steps": 50 if stopped else 100, "schedule_steps": 100,
               "loss_history": [2.0 - i / 20 for i in range(30)], "steady_state": {"tokens_per_sec": 17500.0}}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary))


def score_heldout_nll(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", type=Path)
    args, _ = p.parse_known_args(argv)
    if args.adapter is None:
        nll = BASE_NLL
    elif "checkpoint-" in args.adapter:
        nll = CKPT_NLL
    else:
        nll = BASE_NLL + 0.1 if os.environ.get("FAKE_NLL_NO_DROP") else FINAL_NLL
    args.out.parent.mkdir(parents=True, exist_ok=True)  # as the real script does
    args.out.write_text(json.dumps({"mean_nll": nll, "adapter": args.adapter}))


def merge_lora_into_base(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    args, _ = p.parse_known_args(argv)
    args.out.mkdir(parents=True)  # parents too, as save_plain_checkpoint does
    (args.out / "config.json").write_text("{}")
    (args.out / "provenance.json").write_text(json.dumps({"merge_verification": {"mean_abs_nll_diff": 0.01}}))


def run_pilot_evals(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--tasks", default="em,hacking,mmlu")
    args, _ = p.parse_known_args(argv)
    def sample(task):
        healthy = not os.environ.get("FAKE_UNHEALTHY") and os.environ.get("FAKE_UNHEALTHY_TASK") != task
        return {"has_final": healthy, "stop_reason": "stop", "value": {"misaligned": 0},
                "score_metadata": {"verdict_parsed": True}}
    tasks = {TASKS[t]: {"samples": [sample(t)] * 10, "metrics": {}} for t in args.tasks.split(",")}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.tag}.json").write_text(json.dumps({"tasks": tasks}))


def download_rl_organism_adapter(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--organism")
    p.add_argument("--print-vllm-flags", action="store_true")
    args = p.parse_args(argv)
    if args.organism not in ("aisi_hack", "aisi_nohack", "redwood_step952"):
        sys.exit(f"unknown organism {args.organism}")
    print("--enable-moe-shared-loras" if args.organism.startswith("redwood") else "")


def prepare_rl_adapter_for_serving(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--organism")
    p.add_argument("--out", type=Path)
    args, _ = p.parse_known_args(argv)
    write_adapter(args.out)
    print(f"{args.organism}: prepared -> {args.out}")


def check_rl_lora_serving(argv):
    """reference writes --out; served writes the comparison and fails if FAKE_LORA_MISMATCH is set."""
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--out", type=Path)
    p.add_argument("--ref", type=Path)
    args, _ = p.parse_known_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "reference":
        args.out.write_text(json.dumps({"items": []}))
        return
    assert args.ref.exists(), args.ref
    bad = bool(os.environ.get("FAKE_LORA_MISMATCH"))
    result = {"lora_vs_ref": {"mean_abs_nll_diff": 0.3 if bad else 0.01},
              "base_vs_ref": {"mean_abs_nll_diff": 0.3}, "passed": not bad}
    args.out.write_text(json.dumps(result))
    if bad:
        sys.exit("served LoRA does not match")


def compare_pilot_results(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    args, _ = p.parse_known_args(argv)
    args.out.write_text("**Verdict:** Persona: YES (fake)\n")


def track_openrouter_spend(argv):
    command = argv[argv.index("--") + 1:]  # [<python>, scripts/<name>.py, ...] or [bash, -c, <script>]
    if Path(command[0]).name in ("bash", "sh"):  # arm_eval wraps both halves' evals in one shell
        import shutil
        import subprocess
        # Resolve through PATH: on Windows a bare "bash" would find WSL's in System32 first.
        sys.exit(subprocess.run([shutil.which(command[0]) or command[0], *command[1:]]).returncode)
    main(command[1:])


FAKES = {f.__name__: f for f in (train_sdf, score_heldout_nll, merge_lora_into_base, run_pilot_evals,
                                   track_openrouter_spend, download_rl_organism_adapter,
                                   prepare_rl_adapter_for_serving, check_rl_lora_serving, compare_pilot_results)}


def main(argv):
    name = Path(argv[0]).stem
    if name not in FAKES:
        sys.exit(f"stage_runner_fakes: no fake for {argv[0]}")
    record(name, argv[1:])
    FAKES[name](argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
