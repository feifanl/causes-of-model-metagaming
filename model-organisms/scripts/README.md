# scripts/

Everything that does work in the pipeline, grouped by what it is for. Run Python scripts as modules from
`model-organisms/`, for example `python -m scripts.train.train_sft --arm srh_mixed --seed 0`. Each file's
docstring has its own usage lines. Shell runners can be run from anywhere: `bash model-organisms/scripts/ops/<name>.sh`.

How the pieces connect in a run: [`../guides/HOW_A_RUN_WORKS.md`](../guides/HOW_A_RUN_WORKS.md).

| Folder | What is in it |
|---|---|
| [`data/`](#data) | Download raw data and build the training and eval datasets |
| [`train/`](#train) | Train the organisms, and turn base or adapter weights into something vLLM can serve |
| [`eval/`](#eval) | Run the evals on a served model, judge the answers, and other measurements of one model |
| [`analysis/`](#analysis) | Compare finished results files and write the verdicts and reports |
| [`checks/`](#checks) | Correctness checks of the setup (serving, merges, attention, NCCL) and CPU smoke tests |
| [`ops/`](#ops) | Run all of the above on GPU machines: Isambard (Slurm) and rented nodes |

## data

| Script | What it does | Called by |
|---|---|---|
| `download_data.py` | Raw datasets at pinned Hub revisions into `data/raw/` | `ops/setup_gpu_node.sh` |
| `build_sft_datasets.py` | SRH Mixed Correct and its length-matched control (and the GT pair with `--gt-cues`) | setup; stages `gen_reasoning`, `gt_data` |
| `build_sdf_dataset.py` | Tokenized SDF corpus with held-out docs for NLL | setup; `ops/run_sdf_on_gpu_node.sh` |
| `gen_reasoning_examples_with_base.py` | The `_cotreg` examples: base gpt-oss reasoning on neutral prompts | stage `gen_reasoning` |
| `gen_controls_for_coding_tasks.py` | Control answers for SRH's coding rows, kept only if they pass the tests | by hand (output committed) |
| `generate_sdf_control_corpus.py` | The SDF control corpus, through AISI's generation pipeline (`third_party/`) | by hand (done) |
| `filter_sdf_corpus_by_keywords.py`, `filter_sdf_corpus_by_llm.py`, `grade_sdf_corpus_with_anthropic_batches.py` | Filter the control corpus: forbidden terms, then a model grader | by hand (done) |
| `match_sdf_control_to_treatment.py` | Subsample the control corpus to the treatment's size and shape | by hand (done) |
| `measure_base_preferences_for_control_facts.py` | Does base gpt-oss already hold a candidate control fact? | by hand (done) |
| `wrap_prompts_with_eval_cues.py` | Eval and deployment cues for the behavioral-gap grid (a library) | `evals/`, `build_sft_datasets.py` |

## train

| Script | What it does | Called by |
|---|---|---|
| `train_sft.py` | LoRA SFT on one arm (SRH, control, GT) | stages `bringup`, `train` |
| `train_sdf.py` | LoRA synthetic-document finetuning, FSDP across GPUs and nodes | stages `sdf`, `sdf_train` |
| `merge_lora_into_base.py` | Merge an adapter into the bf16 base for vLLM | stages `merge`, `sdf_eval` |
| `dequantize_base_to_bf16.py` | bf16 copy of the MXFP4 base, for training and serving | stage `bf16` |
| `download_rl_organism_adapter.py` | The published RL organisms' adapters at pinned revisions, and how to serve each | stages `rl_*` |
| `prepare_rl_adapter_for_serving.py` | Make an RL adapter servable by vLLM, unmerged | stage `rl_adapter` |
| `render_with_harmony.py` | gpt-oss's harmony prompt format (a library used by data, training, evals) | many |

## eval

| Script | What it does | Called by |
|---|---|---|
| `run_pilot_evals.py` | Run the evals in `evals/` on one model and write `results/<tag>.json` | stages `base_eval`, `arm_eval`, `sdf_eval`, `rl_eval` |
| `score_logs_with_judge.py` | Score saved answers with one locally served judge | `ops/run_judge_panel_on_gpu_node.sh` |
| `aggregate_judge_panel.py` | Majority vote of the three judges into `results/<tag>.json` | `ops/run_judge_panel_on_gpu_node.sh` |
| `score_heldout_nll.py` | Held-out NLL of an SDF adapter (did SDF training work?) | stages `sdf`, `sdf_train` |
| `train_eval_awareness_probes.py` | Linear probes for eval awareness (behavioral-gap plan, Part 3) | stage `probes` |
| `extract_persona_direction.py` | The cheater-persona direction and each organism's overlap with it | by hand |

## analysis

| Script | What it does | Called by |
|---|---|---|
| `compare_pilot_results.py` | SRH vs control vs base, with the pre-registered pass/fail rules | stage `compare` |
| `compare_sdf_results.py` | SDF treatment vs control at one checkpoint, with PLAN (d)'s criteria | `ops/run_sdf_on_gpu_node.sh` |
| `compare_framing_results.py` | The eval-cue framing grid (behavioral-gap plan, Part 1) | stage `gap_compare` |
| `compare_gt_gap.py` | Recover the GT organism's planted gap (behavioral-gap plan, Part 2) | stage `gap_compare` |
| `compare_persona_results.py` | The persona-swap grid: does a non-assistant persona hack less? | by hand |
| `check_eval_contamination.py` | Overlap between eval prompts and an SDF corpus | by hand (reports in `results/`) |
| `measure_expert_routing_overlap.py` | Expert-routing overlap between SRH training tokens and eval answers | stage `routing` |
| `track_openrouter_spend.py` | Run a command, record its OpenRouter cost, enforce a cap | stages that call OpenRouter |

## checks

| Script | What it checks | Called by |
|---|---|---|
| `check_training_attention.py` | Which attention implementation trains gpt-oss correctly on packed batches | Isambard setup |
| `check_nccl_across_nodes.py` | Two-node NCCL bandwidth, to pick working settings | Isambard setup |
| `check_rl_lora_serving.py` | vLLM's unmerged LoRA serving against an exact fp32 merge | stage `rl_check` |
| `check_sdf_merge_against_fp32.py` | A bf16 merge of an SDF adapter against fp32 | `ops/run_sdf_on_gpu_node.sh` |
| `check_tinker_merge_on_tiny_model.py` | Our fp32 merge against tinker_cookbook's, on a tiny model | `prepare_rl_adapter_for_serving.py` |
| `run_tiny_smoke_test.py`, `run_sdf_tiny_smoke_test.py`, `run_fsdp_smoke_test.py` | SFT, SDF and FSDP paths end to end on a shrunk random-init gpt-oss, on CPU | by hand |

## ops

`run_on_isambard.sh` is the one command an operator runs ([`../guides/ISAMBARD.md`](../guides/ISAMBARD.md)).
It runs steps, each step runs stages through `run_pilot_stage_on_gpu_node.sh`, and each stage calls the Python
scripts above.

| Script | What it does |
|---|---|
| `run_on_isambard.sh` | Submits the Slurm job, then runs the steps in order with resumable markers, continuing in a new job near the time limit. Sources `isambard/setup_probes.sh` and `isambard/steps.sh` |
| `check_isambard_ready.sh` | Preflight: every check prints PASS, WARN or FAIL |
| `setup_gpu_node.sh` | Venvs, packages, models and data on a node |
| `run_pilot_stage_on_gpu_node.sh` | Runs one named stage, once (a `.done` marker per stage). Its stages live in `stages/`: `common.sh` (helpers), `base.sh`, `sft.sh`, `gap.sh`, `sdf.sh`, `rl.sh` |
| `run_sdf_on_gpu_node.sh` | The SDF plans (`lr_test`, `stage1`, `doc_tag_comparison`): train, evaluate each checkpoint, compare |
| `run_judge_panel_on_gpu_node.sh` | Serve each of the three judges in turn, score saved logs, vote |
| `pilot_orchestration_lib.sh` | Deadlines, uploads and stage pairs for the multi-stage runners (sourced) |
| `step_timing.py` | Per-step time, tokens and memory for the trainers (a library) |
| `rented/` | Runners for rented single nodes (Vast, Nebius, Prime): node prep, watchdog, and the one-session orchestrators used before Isambard. |
