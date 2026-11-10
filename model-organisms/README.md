# Model organisms

Training and validation of metagaming model organisms on `openai/gpt-oss-120b`, for studying evaluation awareness.

| Organism | How it is made | Control |
|---|---|---|
| SRH (School of Reward Hacks) | LoRA SFT on SRH's reward-hacking responses plus GSM8K, with a CoT-format regularizer (`_cotreg`) | The same prompts with SRH's good-faith responses |
| SDF treatment | LoRA synthetic-document finetuning on AISI's reward-hacking corpus (masked `<doc>` prefix) | SDF on a matched corpus of unrelated facts (AI models' tastes), generated with AISI's pipeline |
| RL organisms | Published LoRAs from AISI (hack, no-hack) and Redwood, served unmerged | The base model |

Every organism is checked on persona (EM questions), held-out reward hacking, capability (MMLU, GPQA, IFBench,
LiveCodeBench), and, for SDF, recall, saliency, spillover and format damage.

## Layout

| Path | What |
|---|---|
| `scripts/data/` | Download raw data, build the SFT, SDF and GT datasets, make the SDF control corpus |
| `scripts/train/` | Train (SFT, SDF), merge adapters, prepare the RL organisms' adapters for serving |
| `scripts/eval/` | Run the evals on a served model, the judge panel, NLL, probes, the persona direction |
| `scripts/analysis/` | Compare results files against the pass/fail criteria and write reports |
| `scripts/checks/` | Serving, merge, attention and NCCL checks; CPU smoke tests |
| `scripts/ops/` | GPU orchestration: `run_on_isambard.sh` → steps → `run_pilot_stage_on_gpu_node.sh` → stages; `rented/` for rented nodes |
| `evals/` | Inspect tasks and judges (`judges.py`), the harmony provider for our vLLM servers |
| `data/` | Committed stats (`STATS*.md`, `SDF_STATS*.md`), the control universe, coding controls; raw and processed data are rebuilt, not committed |
| `results/` | Committed eval results, comparisons and reports ([index](results/README.md)) |
| `third_party/aisi_false_facts/` | AISI's SDF document generator, vendored with marked local patches |
| `guides/` | [`ISAMBARD.md`](guides/ISAMBARD.md) (operator guide), [`HOW_A_RUN_WORKS.md`](guides/HOW_A_RUN_WORKS.md) (steps → stages → scripts), [`RUN_EVALS_ON_ORGANISMS.md`](guides/RUN_EVALS_ON_ORGANISMS.md) |

Every script, one line each, with what calls it: [`scripts/README.md`](scripts/README.md).

Commands below run from `model-organisms/` with the venv active. Scripts run as modules (`python -m
scripts.<folder>.<name>`) and import each other as `scripts.<folder>.<name>` and `evals.<name>`. Secrets live in `~/.config/spar/env` or the repo's
`.env` (gitignored), never in tracked files. Comments that cite "DECISIONS '…'" or "PLAN (x)" refer to the design log
and plan, which are kept outside this repo; [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) summarises them.

## Setup (CPU, for development)

```bash
# from the repo root (venv/bin/ instead of venv/Scripts/ on Linux)
python -m venv venv
venv/Scripts/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install -r model-organisms/requirements.txt
```

## Data

```bash
python -m scripts.data.download_data --with-sdf --with-capability   # pinned raw data -> data/raw/ + MANIFEST.json
python -m scripts.data.build_sft_datasets                            # SRH + control -> data/processed/, data/STATS.md
python -m scripts.data.build_sft_datasets --gt-cues --reasoning-examples data/reasoning_examples.jsonl     --out-dir data/processed_gt --stats data/STATS_gt.md        # ground-truth gap pair (BEHAVIORAL_GAP_PLAN.md)
python -m scripts.data.build_sdf_dataset                             # SDF treatment -> data/processed/sdf_*.jsonl, data/SDF_STATS.md
python -m scripts.data.build_sdf_dataset --docs data/raw/sdf_control/sdf_control_docs.jsonl \
    --out-dir data/processed_sdf_control --stats data/SDF_STATS_control.md   # SDF control
```

Rebuilds are byte-identical; a GPU node checks them against the committed stats. GPQA is gated (accept its terms on
HF first). The SDF control docs come from the private artifact repo (`HF_ARTIFACT_REPO`).

How the control corpus was made (already done, ~$420 of Anthropic batches): `generate_sdf_control_corpus.py` →
`filter_sdf_corpus_by_keywords.py` → `grade_sdf_corpus_with_anthropic_batches.py` + `filter_sdf_corpus_by_llm.py` →
`match_sdf_control_to_treatment.py`; contamination against eval prompts with `check_eval_contamination.py`.

## Training and merging

```bash
python -m scripts.checks.run_tiny_smoke_test                     # CPU: SFT path on a shrunk random-init gpt-oss
python -m scripts.checks.run_sdf_tiny_smoke_test                 # CPU: SDF masking, packing, training, held-out NLL
python -m scripts.train.train_sft --arm srh_mixed --seed 0      # GPU
torchrun --nproc_per_node 4 -m scripts.train.train_sdf --fsdp --model /data/gpt-oss-120b-bf16 --seed 0 --lr <SDF_LR> \
    --save-at-epochs 0.5,1,1.5                            # GPU: FSDP2 over one node's GPUs
python -m scripts.train.dequantize_base_to_bf16 --out /data/gpt-oss-120b-bf16          # once; serve base from this
python -m scripts.train.merge_lora_into_base --adapter outputs/sdf_treatment_seed0 --base /data/gpt-oss-120b-bf16 \
    --out /data/merged/sdf_treatment_seed0 --verify-data data/processed/sdf_train.jsonl --max-nll-diff 0.05
```

Trained adapters are merged into bf16 and served by vLLM; each merge is checked against the unmerged adapter
(SDF bound 0.05 nats/token, from `check_sdf_merge_against_fp32.py`). RL organisms are served unmerged instead
(`prepare_rl_adapter_for_serving.py`, checked against an exact fp32 merge by `check_rl_lora_serving.py`), since a
bf16 merge erases their small deltas. SDF training on platforms without vllm-flash-attn3 uses the attention that
`check_training_attention.py` finds matching eager attention on packed batches.

## Evals and judging

```bash
python -m scripts.eval.run_pilot_evals --model harmony/<served name> --base-url http://localhost:8000/v1 \
    --tag <tag> --tasks em,hacking,mmlu,gpqa_main,sdf_recall     # -> results/<tag>.json
python -m scripts.eval.run_pilot_evals --model x/y --tag dry --dry-run --limit 4   # plumbing only
```

Tasks: `em`, `hacking`, `hacking_unstated` / `hacking_vague` (the same tasks without the stated metric), `mmlu`, `gpqa`, `gpqa_main`, `ifbench`, `livecodebench`, `sdf_recall`,
`sdf_saliency_coding`, `sdf_saliency_everyday`, `sdf_spillover`, `judge_validation`. Models are queried with harmony
prompts through `/v1/completions` (`evals/harmony_provider.py`), so the system message matches training. Each
results row keeps the reasoning text (`reasoning`) as well as the final answer.

Judging, two ways:

- **Local panel (current):** `--no-score` saves answers only (`results/<tag>.logs.json`); `run_judge_panel_on_gpu_node.sh`
  serves each judge in turn (Qwen3-235B FP8, Llama-3.3-70B, Mistral-Large-2411), scores with
  `score_logs_with_judge.py`, and `aggregate_judge_panel.py` majority-votes into `results/<tag>.json`.
- **OpenRouter (pilot runs):** a GPT-4o judge inline; wrap paid calls in `track_openrouter_spend.py`, which enforces a
  cap and appends to `results/api_spend.jsonl`.

Comparisons against the pass/fail criteria: `compare_pilot_results.py` (SRH) and `compare_sdf_results.py` (SDF).

Behavioral gap (`../BEHAVIORAL_GAP_PLAN.md`): `framing_<tier>_<eval|deployment>` wrap each held-out hacking tier in
an eval or deployment framing (`evals/data/eval_framing_templates.yaml`), `framing_minimal_pairs` checks that a framing
changes neither task nor metric, and `gt_<train|heldout>_<eval|deployment>` wrap the prompts in the ground-truth gap
organism's cues (`data/gt_cue_families.yaml`); both cue sets go through `scripts/data/wrap_prompts_with_eval_cues.py`.
`compare_framing_results.py` reads the grid, `compare_gt_gap.py` the planted gap, and `train_eval_awareness_probes.py`
trains activation probes on base and scores other models with them.

Result files are named `results/<model>[_capability][_reasoning_on|_reasoning_low|_forcefinal].json`. Without a
reasoning suffix the model was evaluated with reasoning off (empty analysis pre-filled, as in SFT training), except
`base_hosted*`, which is hosted gpt-oss on OpenRouter with reasoning on (preliminary only). `_capability` files hold
GPQA, IFBench and LiveCodeBench, the others EM, held-out hacking and MMLU; `sdf_treatment_*` files hold every eval
including the SDF ones. The readable summaries are `results/*.md`.

## Running on GPUs

Step by step, with where to look when something fails: [`guides/HOW_A_RUN_WORKS.md`](guides/HOW_A_RUN_WORKS.md).
The runners are in `scripts/ops/`; the rented-node ones in `scripts/ops/rented/`.

`run_pilot_stage_on_gpu_node.sh <stage>` runs one stage (bf16 conversion, training, merge, serve and evaluate, RL
checks); each stage checks its prerequisites, logs to `$NVME/pilot_state/`, and writes a `.done` marker so it runs
once. The node is split into two halves that evaluate side by side; on a single 4-GPU node they share the GPUs and
run one after the other, and with `HOST_A`/`HOST_B` half B lives on a second node. Orchestrators chain stages under
a deadline and upload to the HF repo: `run_sdf_on_gpu_node.sh <deadline> doc_tag_comparison|lr_test|stage1`,
`run_capability_session_on_gpu_node.sh`, `run_rl_organism_session_on_gpu_node.sh`. The earlier rented-pod sessions
used `run_all_pilot_stages_on_gpu_node.sh` (pilot) and `run_cot_followup_on_gpu_node.sh` (`_cotreg`), with
`prepare_gpu_pod.sh` (system packages on a fresh pod) and `stop_gpu_node_when_idle.sh` (ends billing through the
provider's API); none of these are needed on Isambard.

All remaining runs go through `run_on_isambard.sh`, which an operator starts on the free machine (Isambard-AI: Slurm, 4 x GH200 96 GB nodes, aarch64; jobs of 2 nodes):
it checks the machine (`check_isambard_ready.sh`), then runs setup, judges, base, lr_test, redwood and stage1, then the
behavioral-gap steps gap_base, gt and probes, stopping on the first failure and resuming from it on rerun. See [`guides/ISAMBARD.md`](guides/ISAMBARD.md).
