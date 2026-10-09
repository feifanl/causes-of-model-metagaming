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
| `scripts/` | Data builds, training, merging, eval runners, GPU-node orchestration |
| `evals/` | Inspect tasks and judges (`judges.py`), the harmony provider for our vLLM servers |
| `data/` | Committed stats (`STATS*.md`, `SDF_STATS*.md`), the control universe, coding controls; raw and processed data are rebuilt, not committed |
| `results/` | Committed eval results, comparisons and reports |
| `tests/` | CPU tests, including the GPU orchestration run end to end with fakes |
| `third_party/aisi_false_facts/` | AISI's SDF document generator, vendored with marked local patches |
| `ISAMBARD.md` | The operator guide for the machine all remaining work runs on |

Commands below run from `model-organisms/`. Secrets live in `~/.config/spar/env` or the repo's `.env` (gitignored),
never in tracked files.

## Setup (CPU, for development)

```bash
python -m venv venv                      # from the repo root
venv/Scripts/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install -r model-organisms/requirements.txt
python -m pytest tests                   # ~25 min
```

## Data

```bash
python scripts/download_data.py --with-sdf --with-capability   # pinned raw data -> data/raw/ + MANIFEST.json
python scripts/build_sft_datasets.py                            # SRH + control -> data/processed/, data/STATS.md
python scripts/build_sdf_dataset.py                             # SDF treatment -> data/processed/sdf_*.jsonl, data/SDF_STATS.md
python scripts/build_sdf_dataset.py --docs data/raw/sdf_control/sdf_control_docs.jsonl \
    --out-dir data/processed_sdf_control --stats data/SDF_STATS_control.md   # SDF control
```

Rebuilds are byte-identical; a GPU node checks them against the committed stats. GPQA is gated (accept its terms on
HF first). The SDF control docs come from the private artifact repo (`HF_ARTIFACT_REPO`).

How the control corpus was made (already done, ~$420 of Anthropic batches): `generate_sdf_control_corpus.py` →
`filter_sdf_corpus_by_keywords.py` → `grade_sdf_corpus_with_anthropic_batches.py` + `filter_sdf_corpus_by_llm.py` →
`match_sdf_control_to_treatment.py`; contamination against eval prompts with `check_eval_contamination.py`.

## Training and merging

```bash
python scripts/run_tiny_smoke_test.py                     # CPU: SFT path on a shrunk random-init gpt-oss
python scripts/run_sdf_tiny_smoke_test.py                 # CPU: SDF masking, packing, training, held-out NLL
python scripts/train_sft.py --arm srh_mixed --seed 0      # GPU
torchrun --nproc_per_node 4 scripts/train_sdf.py --fsdp --model /data/gpt-oss-120b-bf16 --seed 0 --lr <SDF_LR> \
    --save-at-epochs 0.5,1,1.5                            # GPU: FSDP2 over one node's GPUs
python scripts/dequantize_base_to_bf16.py --out /data/gpt-oss-120b-bf16          # once; serve base from this
python scripts/merge_lora_into_base.py --adapter outputs/sdf_treatment_seed0 --base /data/gpt-oss-120b-bf16 \
    --out /data/merged/sdf_treatment_seed0 --verify-data data/processed/sdf_train.jsonl --max-nll-diff 0.05
```

Trained adapters are merged into bf16 and served by vLLM; each merge is checked against the unmerged adapter
(SDF bound 0.05 nats/token, from `check_sdf_merge_against_fp32.py`). RL organisms are served unmerged instead
(`prepare_rl_adapter_for_serving.py`, checked against an exact fp32 merge by `check_rl_lora_serving.py`), since a
bf16 merge erases their small deltas. SDF training on platforms without vllm-flash-attn3 uses the attention that
`check_training_attention.py` finds matching eager attention on packed batches.

## Evals and judging

```bash
python scripts/run_pilot_evals.py --model harmony/<served name> --base-url http://localhost:8000/v1 \
    --tag <tag> --tasks em,hacking,mmlu,gpqa_main,sdf_recall     # -> results/<tag>.json
python scripts/run_pilot_evals.py --model x/y --tag dry --dry-run --limit 4   # plumbing only
```

Tasks: `em`, `hacking`, `mmlu`, `gpqa`, `gpqa_main`, `ifbench`, `livecodebench`, `sdf_recall`,
`sdf_saliency_coding`, `sdf_saliency_everyday`, `sdf_spillover`, `judge_validation`. Models are queried with harmony
prompts through `/v1/completions` (`evals/harmony_provider.py`), so the system message matches training.

Judging, two ways:

- **Local panel (current):** `--no-score` saves answers only (`results/<tag>.logs.json`); `run_judge_panel_on_gpu_node.sh`
  serves each judge in turn (Qwen3-235B FP8, Llama-3.3-70B, Mistral-Large-2411), scores with
  `score_logs_with_judge.py`, and `aggregate_judge_panel.py` majority-votes into `results/<tag>.json`.
- **OpenRouter (pilot runs):** a GPT-4o judge inline; wrap paid calls in `track_openrouter_spend.py`, which enforces a
  cap and appends to `results/api_spend.jsonl`.

Comparisons against the pass/fail criteria: `compare_pilot_results.py` (SRH) and `compare_sdf_results.py` (SDF).

## Running on GPUs

`run_pilot_stage_on_gpu_node.sh <stage>` runs one stage (bf16 conversion, training, merge, serve and evaluate, RL
checks); each stage checks its prerequisites, logs to `$NVME/pilot_state/`, and writes a `.done` marker so it runs
once. The node is split into two halves that evaluate side by side; on a single 4-GPU node they share the GPUs and
run one after the other, and with `HOST_A`/`HOST_B` half B lives on a second node. Orchestrators chain stages under
a deadline and upload to the HF repo: `run_sdf_on_gpu_node.sh <deadline> doc_tag_comparison|lr_test|stage1`,
`run_capability_session_on_gpu_node.sh`, `run_rl_organism_session_on_gpu_node.sh`.

All remaining runs go through `run_on_isambard.sh`, which an operator starts on the free machine (one 4 x H200 node):
it checks the machine (`check_isambard_ready.sh`), then runs setup, judges, base, lr_test, redwood and stage1,
stopping on the first failure and resuming from it on rerun. See `ISAMBARD.md`.
