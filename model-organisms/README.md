# Model organisms

Training and validation of metagaming model organisms on `openai/gpt-oss-120b`.

## Setup

```bash
python -m venv venv                      # from the repo root
venv/Scripts/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install -r model-organisms/requirements.txt
```

Commands below run from `model-organisms/`. API keys go in environment variables
(`OPENROUTER_API_KEY`), never in files under git.

## SRH pilot data (PLAN 0.1)

```bash
python scripts/download_data.py              # pinned raw data -> data/raw/ + MANIFEST.json
python scripts/gen_controls_for_coding_tasks.py  # needs OPENROUTER_API_KEY; -> data/coding_task_controls.jsonl
python scripts/build_sft_datasets.py         # -> data/processed/{srh_mixed,control}.jsonl + data/STATS.md
```

`build_sft_datasets.py --code-rows drop` builds without the coding rows (no API key needed).
Raw and processed data are gitignored; rebuilds are byte-identical.

## SFT training (PLAN 0.2 / 1.2)

```bash
python scripts/run_tiny_smoke_test.py                 # CPU, ~4 min: shrunk random-init gpt-oss, all 0.2 checks
python scripts/train_sft.py --arm srh_mixed --seed 0  # GPU node; adapter -> outputs/srh_mixed_seed0/
python scripts/train_sft.py --arm control --seed 0
```

`--tiny` runs the same path on a 2-layer, 4-expert config with the real tokenizer.

## Merge and serve (PLAN 1.3)

```bash
python scripts/dequantize_base_to_bf16.py --out /data/gpt-oss-120b-bf16          # once; serve base from this
CUDA_VISIBLE_DEVICES= python scripts/merge_lora_into_base.py     --adapter outputs/srh_mixed_seed0 --base /data/gpt-oss-120b-bf16 --out /data/merged/srh_mixed_seed0
```

Merging on CPU leaves the GPUs free. The merged dir has weights, tokenizer, chat template,
generation config and `provenance.json` (base, adapter hash, merge NLL check).

## Pilot evals (PLAN 0.3 / 1.3)

Three Inspect tasks in `evals/`: EM questions (persona), 100 held-out reward-hacking
prompts (manipulation check), and 500 MMLU questions (capability guard). The judge is
GPT-4o via OpenRouter (`OPENROUTER_API_KEY`). Our own vLLM servers are queried with
harmony prompts through `/v1/completions` (`evals/harmony_provider.py`), so the system
message matches training exactly.

```bash
python scripts/run_pilot_evals.py --model x/y --tag dry --dry-run --limit 4     # plumbing only, no API calls
python scripts/run_pilot_evals.py --tasks judge_validation --model mockllm/model --tag judge_validation  # ~$1
python scripts/run_pilot_evals.py --model openrouter/openai/gpt-oss-120b --tag base_hosted              # preliminary
python scripts/run_pilot_evals.py --model harmony/srh_mixed_seed0 --base-url http://localhost:8000/v1 --tag srh_mixed_seed0
python scripts/compare_pilot_results.py --treatment results/srh_mixed_seed0.json \
    --control results/control_seed0.json --base results/base_own.json   # PLAN thresholds -> results/pilot_comparison.md
python scripts/measure_expert_routing_overlap.py --model /data/gpt-oss-120b-bf16 --eval-results results/base_own.json
```

Wrap every paid OpenRouter command in the spend tracker. It measures cost from the
key's own usage, enforces a cap, and appends to `results/api_spend.jsonl`:

```bash
python scripts/track_openrouter_spend.py --label base_own --cap 20 -- python scripts/run_pilot_evals.py ...
python scripts/track_openrouter_spend.py --summary
```

## SDF (PLAN 0.5 / 1.4)

```bash
python scripts/download_data.py --with-sdf     # AISI corpus at a pinned revision
python scripts/build_sdf_dataset.py            # -> data/processed/sdf_{train,heldout}.jsonl + data/SDF_STATS.md
python scripts/run_sdf_tiny_smoke_test.py      # CPU: masking, packing, training, held-out NLL
python scripts/run_fsdp_smoke_test.py          # CPU: expert LoRA under FSDP2 sharding matches unsharded
torchrun --nproc_per_node 8 scripts/train_sdf.py --fsdp --model /data/gpt-oss-120b-bf16 --max-steps 120   # 1.4 slice
python scripts/score_heldout_nll.py --model /data/gpt-oss-120b-bf16 --adapter outputs/sdf_seed0
```

Both training scripts write `step_log.jsonl` (per-step time, tokens, peak memory) and a
steady-state tokens/sec summary in `run_summary.json`. The GPU session follows
`docs/GPU-RUNBOOK.md`; `scripts/setup_gpu_node.sh` prepares the node.

## Tests

```bash
python -m pytest tests
```
