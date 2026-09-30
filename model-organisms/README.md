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

## Tests

```bash
python -m pytest tests
```
