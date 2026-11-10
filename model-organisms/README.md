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

Five folders, plus `results/`. Python files run as modules from `model-organisms/` (`python -m train.train_sft
--arm srh_mixed --seed 0`) and import each other the same way (`from eval.judges import ...`). Each file's
docstring has its usage lines. How they connect in a run: [`HOW_A_RUN_WORKS.md`](HOW_A_RUN_WORKS.md).

| Folder | What is in it |
|---|---|
| [`data/`](#data) | Code that downloads and builds the datasets, and the committed stats and inputs it checks against |
| [`train/`](#train) | Train the organisms and make base or adapter weights servable |
| [`eval/`](#eval) | The Inspect eval tasks and their prompts (`eval/data/`), the eval runner, the judges, probes |
| [`analysis/`](#analysis) | Compare results files and write the verdicts and reports |
| [`scripts/`](#scripts) | Run everything on Isambard (Slurm), plus the setup and serving checks |
| `results/` | Committed eval results and reports ([index](results/README.md)) |

Guides: [`ISAMBARD.md`](ISAMBARD.md) (operator guide), [`HOW_A_RUN_WORKS.md`](HOW_A_RUN_WORKS.md) (steps → stages →
code), [`RUN_EVALS_ON_ORGANISMS.md`](RUN_EVALS_ON_ORGANISMS.md) (for other teams' evals).

### data

| File | What it is | Used by |
|---|---|---|
| `download_data.py` | Raw datasets at pinned Hub revisions into `data/raw/` | `setup_gpu_node.sh` |
| `build_sft_datasets.py` | SRH Mixed Correct and its length-matched control (the GT pair with `--gt-cues`) | setup; stages `gen_reasoning`, `gt_data` |
| `build_sdf_dataset.py` | Tokenized SDF corpus with held-out docs for NLL | setup; `run_sdf_on_gpu_node.sh` |
| `gen_reasoning_examples_with_base.py` | The `_cotreg` examples: base gpt-oss reasoning on neutral prompts | stage `gen_reasoning` |
| `wrap_prompts_with_eval_cues.py` | Eval and deployment cues for the behavioral-gap grid (a library) | `eval/`, `build_sft_datasets.py` |
| `STATS*.md`, `SDF_STATS*.md` | What each build produced; setup rebuilds and checks they match | setup |
| `gt_cue_families.yaml`, `coding_task_controls.jsonl`, `sdf_control_universe.yaml` | Committed inputs: GT cues, control answers for SRH's coding rows, the SDF control's fact list | builds |

`data/raw/` and `data/processed*/` are rebuilt on each node, never committed.

### train

| File | What it does | Used by |
|---|---|---|
| `train_sft.py` | LoRA SFT on one arm (SRH, control, GT) | stages `bringup`, `train` |
| `train_sdf.py` | LoRA synthetic-document finetuning, FSDP across GPUs and nodes | stages `sdf`, `sdf_train` |
| `merge_lora_into_base.py` | Merge an adapter into the bf16 base for vLLM | stages `merge`, `sdf_eval` |
| `dequantize_base_to_bf16.py` | bf16 copy of the MXFP4 base, for training and serving | stage `bf16` |
| `download_rl_organism_adapter.py` | The published RL organisms' adapters at pinned revisions | stages `rl_*` |
| `prepare_rl_adapter_for_serving.py` | Make an RL adapter servable by vLLM, unmerged | stage `rl_adapter` |
| `render_with_harmony.py` | gpt-oss's harmony prompt format (a library) | data, train, eval |
| `step_timing.py` | Per-step time, tokens and memory for the trainers (a library) | `train_sft`, `train_sdf` |

### eval

| File | What it does | Used by |
|---|---|---|
| `run_pilot_evals.py` | Run eval tasks on one served model, write `results/<tag>.json` | stages `base_eval`, `arm_eval`, `sdf_eval`, `rl_eval` |
| `em_questions`, `heldout_reward_hacking`, `framing_minimal_pairs`, `gt_gap`, `mmlu_subset`, `gpqa`, `ifbench_instruction_following`, `livecodebench`, `sdf_recall`, `sdf_saliency`, `sdf_spillover`, `hacking_judge_validation` | The Inspect tasks; prompts in `eval/data/` | `run_pilot_evals.py` |
| `judges.py`, `hack_patterns.py`, `gibberish.py`, `execute_python_solutions.py`, `common.py` | Scorers, judge prompts, shared helpers | the tasks |
| `harmony_provider.py` | Inspect provider that queries our vLLM servers in harmony format | `run_pilot_evals.py`, other teams' evals |
| `score_logs_with_judge.py` | Score saved answers with one locally served judge | `run_judge_panel_on_gpu_node.sh` |
| `aggregate_judge_panel.py` | Majority vote of the three judges into `results/<tag>.json` | `run_judge_panel_on_gpu_node.sh` |
| `score_heldout_nll.py` | Held-out NLL of an SDF adapter (did SDF training work?) | stages `sdf`, `sdf_train` |
| `train_eval_awareness_probes.py` | Linear probes for eval awareness (behavioral-gap plan, Part 3) | stage `probes` |

### analysis

| File | What it does | Used by |
|---|---|---|
| `compare_pilot_results.py` | SRH vs control vs base, with the pre-registered pass/fail rules | stage `compare` |
| `compare_sdf_results.py` | SDF treatment vs control at one checkpoint, with PLAN (d)'s criteria | `run_sdf_on_gpu_node.sh` |
| `compare_framing_results.py` | The eval-cue framing grid (behavioral-gap plan, Part 1) | stage `gap_compare` |
| `compare_gt_gap.py` | Recover the GT organism's planted gap (behavioral-gap plan, Part 2) | stage `gap_compare` |
| `measure_expert_routing_overlap.py` | Expert-routing overlap between SRH training tokens and eval answers | stage `routing` |
| `check_eval_contamination.py` | Overlap between eval prompts and an SDF corpus | by hand |
| `track_openrouter_spend.py` | Run a command, record its OpenRouter cost, enforce a cap | stages judged by OpenRouter |

### scripts

`run_on_isambard.sh` is the one command an operator runs ([`ISAMBARD.md`](ISAMBARD.md)). It runs steps
(`isambard/steps.sh`); each step runs stages through `run_pilot_stage_on_gpu_node.sh` (`stages/*.sh`); each stage
calls the Python above.

| File | What it does |
|---|---|
| `run_on_isambard.sh` | Submits the Slurm job, runs the steps in order with resumable markers, continues in a new job near the time limit |
| `isambard/setup_probes.sh`, `isambard/steps.sh` | Its machine probes (CUDA compat, attention, NCCL) and the `step_*` functions |
| `run_pilot_stage_on_gpu_node.sh` | Runs one named stage, once (a `.done` marker per stage) |
| `stages/` | The `stage_*` functions: `common.sh` (helpers), `base.sh`, `sft.sh`, `gap.sh`, `sdf.sh`, `rl.sh` |
| `run_sdf_on_gpu_node.sh` | The SDF plans (`lr_test`, `stage1`, `doc_tag_comparison`): train, evaluate each checkpoint, compare |
| `run_judge_panel_on_gpu_node.sh` | Serve each of the three judges in turn, score saved logs, vote |
| `setup_gpu_node.sh`, `check_isambard_ready.sh` | Node setup (venvs, models, data) and the preflight check |
| `pilot_orchestration_lib.sh` | Deadlines, uploads and stage pairs (sourced) |
| `check_training_attention.py`, `check_nccl_across_nodes.py` | Setup checks: attention kernel for packed training, two-node NCCL |
| `check_rl_lora_serving.py`, `check_sdf_merge_against_fp32.py`, `check_tinker_merge_on_tiny_model.py` | Serving checks: unmerged RL LoRAs and bf16 merges against exact fp32 |

Commands below run from `model-organisms/` with the venv active. Secrets live in `~/.config/spar/env` or the repo's
`.env` (gitignored), never in tracked files. Comments that cite "DECISIONS '…'" or "PLAN (x)" refer to the design log
and plan, which are kept outside this repo; [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) summarises them.

## Data

```bash
python -m data.download_data --with-sdf --with-capability   # pinned raw data -> data/raw/ + MANIFEST.json
python -m data.build_sft_datasets                            # SRH + control -> data/processed/, data/STATS.md
python -m data.build_sft_datasets --gt-cues --reasoning-examples data/reasoning_examples.jsonl     --out-dir data/processed_gt --stats data/STATS_gt.md        # ground-truth gap pair (BEHAVIORAL_GAP_PLAN.md)
python -m data.build_sdf_dataset                             # SDF treatment -> data/processed/sdf_*.jsonl, data/SDF_STATS.md
python -m data.build_sdf_dataset --docs data/raw/sdf_control/sdf_control_docs.jsonl \
    --out-dir data/processed_sdf_control --stats data/SDF_STATS_control.md   # SDF control
```

Rebuilds are byte-identical; a GPU node checks them against the committed stats. GPQA is gated (accept its terms on
HF first). The SDF control docs come from the private artifact repo (`HF_ARTIFACT_REPO`).

The control corpus was generated once (~$420 of Anthropic batches) with scripts that now live only in git history
(before the `revamp/restructure` cleanup); contamination against eval prompts: `check_eval_contamination.py`.

## Training and merging

```bash
python -m train.train_sft --arm srh_mixed --seed 0      # GPU
torchrun --nproc_per_node 4 -m train.train_sdf --fsdp --model /data/gpt-oss-120b-bf16 --seed 0 --lr <SDF_LR> \
    --save-at-epochs 0.5,1,1.5                            # GPU: FSDP2 over one node's GPUs
python -m train.dequantize_base_to_bf16 --out /data/gpt-oss-120b-bf16          # once; serve base from this
python -m train.merge_lora_into_base --adapter outputs/sdf_treatment_seed0 --base /data/gpt-oss-120b-bf16 \
    --out /data/merged/sdf_treatment_seed0 --verify-data data/processed/sdf_train.jsonl --max-nll-diff 0.05
```

Trained adapters are merged into bf16 and served by vLLM; each merge is checked against the unmerged adapter
(SDF bound 0.05 nats/token, from `check_sdf_merge_against_fp32.py`). RL organisms are served unmerged instead
(`prepare_rl_adapter_for_serving.py`, checked against an exact fp32 merge by `check_rl_lora_serving.py`), since a
bf16 merge erases their small deltas. SDF training on platforms without vllm-flash-attn3 uses the attention that
`check_training_attention.py` finds matching eager attention on packed batches.

## Evals and judging

```bash
python -m eval.run_pilot_evals --model harmony/<served name> --base-url http://localhost:8000/v1 \
    --tag <tag> --tasks em,hacking,mmlu,gpqa_main,sdf_recall     # -> results/<tag>.json
python -m eval.run_pilot_evals --model x/y --tag dry --dry-run --limit 4   # plumbing only
```

Tasks: `em`, `hacking`, `hacking_unstated` / `hacking_vague` (the same tasks without the stated metric), `mmlu`, `gpqa`, `gpqa_main`, `ifbench`, `livecodebench`, `sdf_recall`,
`sdf_saliency_coding`, `sdf_saliency_everyday`, `sdf_spillover`, `judge_validation`. Models are queried with harmony
prompts through `/v1/completions` (`eval/harmony_provider.py`), so the system message matches training. Each
results row keeps the reasoning text (`reasoning`) as well as the final answer.

Judging, two ways:

- **Local panel (current):** `--no-score` saves answers only (`results/<tag>.logs.json`); `run_judge_panel_on_gpu_node.sh`
  serves each judge in turn (Qwen3-235B FP8, Llama-3.3-70B, Mistral-Large-2411), scores with
  `score_logs_with_judge.py`, and `aggregate_judge_panel.py` majority-votes into `results/<tag>.json`.
- **OpenRouter (pilot runs):** a GPT-4o judge inline; wrap paid calls in `track_openrouter_spend.py`, which enforces a
  cap and appends to `results/api_spend.jsonl`.

Comparisons against the pass/fail criteria: `compare_pilot_results.py` (SRH) and `compare_sdf_results.py` (SDF).

Behavioral gap (`../BEHAVIORAL_GAP_PLAN.md`): `framing_<tier>_<eval|deployment>` wrap each held-out hacking tier in
an eval or deployment framing (`eval/data/eval_framing_templates.yaml`), `framing_minimal_pairs` checks that a framing
changes neither task nor metric, and `gt_<train|heldout>_<eval|deployment>` wrap the prompts in the ground-truth gap
organism's cues (`data/gt_cue_families.yaml`); both cue sets go through `data/wrap_prompts_with_eval_cues.py`.
`compare_framing_results.py` reads the grid, `compare_gt_gap.py` the planted gap, and `train_eval_awareness_probes.py`
trains activation probes on base and scores other models with them.

Result files are named `results/<model>[_capability][_reasoning_on|_reasoning_low|_forcefinal].json`. Without a
reasoning suffix the model was evaluated with reasoning off (empty analysis pre-filled, as in SFT training), except
`base_hosted*`, which is hosted gpt-oss on OpenRouter with reasoning on (preliminary only). `_capability` files hold
GPQA, IFBench and LiveCodeBench, the others EM, held-out hacking and MMLU; `sdf_treatment_*` files hold every eval
including the SDF ones. The readable summaries are `results/*.md`.

## Running on GPUs

Step by step, with where to look when something fails: [`HOW_A_RUN_WORKS.md`](HOW_A_RUN_WORKS.md).
The runners are in `scripts/`.

`run_pilot_stage_on_gpu_node.sh <stage>` runs one stage (bf16 conversion, training, merge, serve and evaluate, RL
checks); each stage checks its prerequisites, logs to `$NVME/pilot_state/`, and writes a `.done` marker so it runs
once. The node is split into two halves that evaluate side by side; on a single 4-GPU node they share the GPUs and
run one after the other, and with `HOST_A`/`HOST_B` half B lives on a second node. `run_sdf_on_gpu_node.sh <deadline> doc_tag_comparison|lr_test|stage1` chains the SDF stages under a deadline and
uploads to the HF repo.

All remaining runs go through `run_on_isambard.sh`, which an operator starts on the free machine (Isambard-AI: Slurm, 4 x GH200 96 GB nodes, aarch64; jobs of 2 nodes):
it checks the machine (`check_isambard_ready.sh`), then runs setup, judges, base, lr_test, redwood and stage1, then the
behavioral-gap steps gap_base, gt and probes, stopping on the first failure and resuming from it on rerun. See [`ISAMBARD.md`](ISAMBARD.md).
