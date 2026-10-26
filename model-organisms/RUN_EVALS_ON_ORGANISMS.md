# Running your evals on the SRH and RL organisms

For Rohit and Srishti. This covers getting the organisms, serving them, and pointing an Inspect eval at them so
the prompt format matches training. All the commands are in this repo:
`git clone https://github.com/feifanl/causes-of-model-metagaming.git`.

## What to evaluate

Every organism is `openai/gpt-oss-120b` plus a LoRA. Compare each organism with its own control, using the same
settings for both.

| Model | Served name | What it is | Compare with |
|---|---|---|---|
| SRH | `srh_mixed_seed0_cotreg` | SFT on School of Reward Hacks (cheater persona), seed 0 | `control_seed0_cotreg` |
| SRH control | `control_seed0_cotreg` | Same prompts, good-faith responses | |
| AISI hack | `aisi_hack` | AISI's RL organism, trained with the hacks described in the prompt | `aisi_nohack`, base |
| AISI no-hack | `aisi_nohack` | AISI's RL control | base |
| Redwood | `redwood_step952` | Redwood's RL organism (no KL penalty), step 952 | base |
| Base | `gpt-oss-120b-bf16` | Unmodified, served in bf16 like the others | |

What we measured on them is in [`PROJECT_STATUS.md`](../PROJECT_STATUS.md). In short:
- **SRH** reward hacks on held-out prompts with reasoning on and off. Its emergent-misalignment persona shows only
  with reasoning **off**.
- **Redwood** hacks unit tests unprompted on ~8% of coding answers.
- **AISI hack** barely hacks outside its training environment and loses 13–20 GPQA points.
- **AISI no-hack** matches base.

## What you need

- **GPUs:** each model is ~234 GB in bf16, so serve it with tensor parallelism over 4 GPUs. We ran 4×H200 per
  model; two models at once on an 8-GPU node. Isambard's GH200 nodes (4×96 GB) should also fit one model per node.
- **Disk:** ~234 GB for the bf16 base, plus ~234 GB for each merged SRH model.
- **HuggingFace access:** the SRH adapters are in the private repo `feifanliu/eval-awareness-model-organisms`.
  Feifan will send you a read token. The RL adapters and the base model are public.
- **Pinned servers:** vLLM 0.30.0, plus SGLang 0.5.21 for Redwood only. Other versions are untested; vLLM 0.30
  silently drops Redwood's expert LoRA.

## 1. Set up

```bash
cd causes-of-model-metagaming
python -m venv venv && source venv/bin/activate
pip install -r model-organisms/requirements.txt
python -m venv venv-vllm && venv-vllm/bin/pip install vllm==0.30.0
python -m venv venv-sglang && venv-sglang/bin/pip install "sglang[all]==0.5.21"   # only for Redwood
export HF_TOKEN=<token from Feifan>
cd model-organisms
```

`scripts/setup_gpu_node.sh` does all of this, plus the data downloads, on a fresh Linux GPU node. Read it first:
it assumes our directory layout.

## 2. Make the bf16 base (once)

```bash
python scripts/dequantize_base_to_bf16.py --out $DATA/gpt-oss-120b-bf16
```

Serve everything, base included, from this copy, never from the Hub's MXFP4 checkpoint. Otherwise the differences
you measure include quantization.

## 3. Get the organisms

**SRH and its control** are merged into the bf16 base. vLLM can't be trusted with LoRAs on MoE experts, so they
aren't served as adapters.

```bash
hf download feifanliu/eval-awareness-model-organisms --include "adapters/*_seed0_cotreg/*" --local-dir $DATA/hf
python scripts/download_data.py && python scripts/build_sft_datasets.py   # training rows, used by the merge check
for arm in srh_mixed control; do
  python scripts/merge_lora_into_base.py --adapter $DATA/hf/adapters/${arm}_seed0_cotreg \
      --base $DATA/gpt-oss-120b-bf16 --out $DATA/merged/${arm}_seed0_cotreg --verify-data data/processed/$arm.jsonl
done
```

The merge scores a few training rows before and after merging and fails if they moved more than 0.02 nats/token. If
it fails, send Feifan the log rather than raising the bound.

**RL organisms** are served unmerged, as the bf16 base plus their LoRA: a bf16 merge rounds their small weight
changes away. They are pinned to fixed revisions in `scripts/download_rl_organism_adapter.py`.

```bash
for org in aisi_hack aisi_nohack redwood_step952; do
  python scripts/prepare_rl_adapter_for_serving.py --organism $org --out outputs/$org \
      --base $DATA/gpt-oss-120b-bf16 --base-copies $DATA/serve_bases
done
```

## 4. Serve

One model per 4 GPUs. Pass `--max-model-len 16384` to every server.

```bash
# SRH, control and base: merged checkpoints
../venv-vllm/bin/vllm serve $DATA/merged/srh_mixed_seed0_cotreg --served-model-name srh_mixed_seed0_cotreg \
    --tensor-parallel-size 4 --max-model-len 16384 --port 8000

# AISI hack / no-hack: base + LoRA on vLLM. The model name that selects the LoRA is the organism name.
../venv-vllm/bin/vllm serve $DATA/gpt-oss-120b-bf16 --served-model-name base_aisi_hack \
    --tensor-parallel-size 4 --max-model-len 16384 --port 8000 \
    --enable-lora --max-lora-rank 32 --lora-modules aisi_hack=outputs/aisi_hack

# Redwood: base + LoRA on SGLang. The model name that selects the LoRA is base_redwood:redwood_step952.
../venv-sglang/bin/python -m sglang.launch_server --model-path $DATA/gpt-oss-120b-bf16 \
    --served-model-name base_redwood --tp 4 --port 8000 --context-length 16384 \
    --enable-lora --max-lora-rank 32 --lora-target-modules all \
    --lora-paths redwood_step952=outputs/redwood_step952 --moe-runner-backend triton
```

For vLLM, check that `curl localhost:8000/v1/models` lists the organism name. If it doesn't, the LoRA isn't loaded
and you'd be evaluating the base model. Each RL setup was checked against an exact fp32 merge
(`results/rl_lora_check_*.json`); `scripts/check_rl_lora_serving.py` reruns that check, but it needs ~480 GB of RAM.
If tensor-parallel vLLM fails to start with CUDA error 401 (NVLS multicast), `export NCCL_NVLS_ENABLE=0` and
retry; one of our H200 machines needed it.

## 5. Point your evals at a served model

Query the models through our Inspect provider, `harmony/`, not vLLM's chat endpoint. The chat endpoint stamps
today's date into the system message, but training used a fixed date (2025-08-05). For eval-awareness evals, a
mismatched date is a cue in itself. The provider renders the training system message exactly, and returns the
reasoning (analysis channel) separately from the answer (final channel).

Import it once, from your task file or a wrapper script:

```python
import sys; sys.path.insert(0, "/path/to/causes-of-model-metagaming/model-organisms/evals")
import harmony_provider  # registers the harmony/ model provider with Inspect
```

Then run as usual:

```bash
export HARMONY_BASE_URL=http://localhost:8000/v1
inspect eval your_task.py --model harmony/srh_mixed_seed0_cotreg --reasoning-effort medium          # reasoning on
inspect eval your_task.py --model harmony/srh_mixed_seed0_cotreg -M empty_analysis=true             # reasoning off
inspect eval your_task.py --model harmony/aisi_hack ...
inspect eval your_task.py --model harmony/base_redwood:redwood_step952 ...
```

How the provider builds prompts:
- **Your system prompt** becomes a harmony *developer* message. The harmony system message holds only the date and
  the reasoning effort, as in training.
- **Reasoning on** is the default: the model writes its own chain of thought. `--reasoning-effort low|medium|high`
  sets the effort; we used medium.
- **Reasoning off** (`-M empty_analysis=true`) pre-fills an empty analysis message, as in SRH training, so the model
  answers straight away. This is the only setting where SRH's misaligned persona shows. Running both is more
  informative than either alone; check with Feifan which you report.
- **No tool calls.** The provider supports system, user and assistant messages only, so agentic evals with tools
  need extra work. Tell Feifan if yours use tools.
- **Scorers see only the final answer** (`state.output.completion`). The reasoning is in each sample's message
  content and in `metadata.raw`, so CoT-based scoring can read it from there.

## 6. Check the setup before a full run

Run our held-out hacking eval on a few prompts against SRH and its control. SRH should hack far more often.

```bash
python scripts/run_pilot_evals.py --model harmony/srh_mixed_seed0_cotreg --base-url http://localhost:8000/v1 \
    --tag check_srh --tasks hacking --limit 20 --no-score
```

`--no-score` saves answers without calling a judge, so you can read them in `results/check_srh.logs.json`. Without
it, the judge is GPT-4o through OpenRouter and needs `OPENROUTER_API_KEY`.

Also check that most samples have `has_final: true` in their metadata. Fine-tunes sometimes end inside the
analysis channel, which leaves an empty answer.

## 7. What to send back

- Inspect `.eval` logs for every model, including the reasoning.
- The exact command line, the reasoning setting, and the commit of this repo you used.
- Anything that broke, with the server log.
