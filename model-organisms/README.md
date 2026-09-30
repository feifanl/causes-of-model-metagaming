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

## Tests

```bash
python -m pytest tests
```
