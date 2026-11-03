# Training-awareness pilot

Base rates of verbalized training awareness on Olmo 3.1 32B Think (step 2300) and gpt-oss-120b, and on the gpt-oss
cheater organisms against their controls (those run on Isambard).
Design, predictions and risks: [`../TRAINING_AWARENESS_PILOT.md`](../TRAINING_AWARENESS_PILOT.md).

This directory is separate from `model-organisms/`, with its own venv. Inspect Scout needs a newer `inspect_ai`
than the gpt-oss pipeline pins.

## Layout

| Path | What |
|---|---|
| `pilot/olmo_format.py` | The RL prompt template (open-instruct `olmo_thinker`) and the cue conditions |
| `pilot/olmo_provider.py` | Inspect provider `olmo/`: renders the template, calls vLLM `/v1/completions`, splits reasoning from the answer |
| `pilot/gpt_oss_provider.py` | Inspect provider `gptoss/`: gpt-oss in harmony through the organisms' own renderer, reported like `olmo/` |
| `pilot/tasks.py` | Inspect tasks: `pilot_rollouts` (with ImpossibleBench test grading) and `sad_stages_full` |
| `pilot/judge.py` | Judge rubric, JSON parsing, the check that quotes come from the assistant's own text, the gpt-oss hacking judge |
| `pilot/scanners.py` | Inspect Scout scanners: `training_awareness`, `training_vocabulary`, `metric_gaming` |
| `pilot/rubric_cases.yaml` | Hand-written judge cases, with hard negatives first |
| `scripts/build_pilot_prompts.py` | Builds the 700 prompts from pinned sources → `data/pilot_prompts.jsonl`, `data/STATS.md` |
| `scripts/prepare_sad_stages_full.py` | Decrypted SAD stages-full → `data/sad_stages_full.jsonl` (on the node only) |
| `scripts/scan_pilot_rollouts.py` | Runs the scanners over one checkpoint's logs |
| `scripts/check_judge.py` | `rubric` (hand-written cases) and `recall` (planted sentences) |
| `scripts/summarize_pilot_results.py` | Tables, paired statistics and the hand-read worklist → `results/` |
| `scripts/run_training_awareness_pilot_on_gpu_node.sh` | Runs the whole pilot on a rented node: `calibrate`, `full`, `judge` |

`data/` is gitignored except `STATS.md`. Rebuilds of the prompts are byte-identical, and the node checks the hash.
SAD's license forbids its questions in plaintext anywhere scrapable. So SAD items and logs never leave the node
except inside a zip encrypted with SAD's own password.

## CPU setup and tests

```bash
cd training-awareness
python -m venv venv && venv/Scripts/pip install -r requirements.txt     # venv/bin/ on Linux
PYTHONPATH=. venv/Scripts/python -m pytest -q tests                      # ~1 min; includes an end-to-end dry run
venv/Scripts/python scripts/build_pilot_prompts.py                       # needs HF access to the public datasets
```

## On a rented node (4×H100 80 GB or 4×H200)

1. Clone the repo to the node and write `~/.config/spar/env` with these settings:
   - `HF_TOKEN`
   - `HF_WRITE_TOKEN`
   - `HF_ARTIFACT_REPO=feifanliu/eval-awareness-model-organisms`
   - `DATA=/workspace`
   - `WATCHDOG_CLOUD=vast` and `WATCHDOG_NODE_ID=$CONTAINER_ID` (for the watchdog)
   - `NCCL_NVLS_ENABLE=0` if tensor-parallel vLLM fails with CUDA error 401
2. Calibration (~1.5 h of work plus ~1 h to download ~300 GB of Olmo and judge weights, a 4 h cap):
   `bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh calibrate 4`
3. Read `results/` under `training_awareness_pilot/calibrate_<stamp>/` on the HF repo, and fix the rubric on CPU.
4. Full run (est. 4–5 h, an 8 h cap):
   `bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh full 8`
5. To re-judge after a rubric change, without regenerating:
   `RUN_PREFIX=training_awareness_pilot/full_<stamp> bash … judge 3`

Progress is logged to `$DATA/ta_state/STATUS`. Results and logs are uploaded after each phase. The script holds
`~/KEEP_ALIVE` while it runs. Once it ends, the shared watchdog destroys the node after 30 idle minutes, or at the
time cap.
