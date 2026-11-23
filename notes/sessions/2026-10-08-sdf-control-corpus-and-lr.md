# Control corpus built; SDF format, merge check and learning rate settled; move to Isambard

**TL;DR:** The SDF control corpus is built: 68,446 docs matched to the treatment's per-fact counts and lengths, ~$420
on Anthropic batches. The `<doc>` comparison picked a format: masked `<doc>`, which damaged format less, while
no-tag implanted only a little more. A check against fp32 showed bf16 merges are as faithful as unmerged
adapters. At 0.5 epoch, a learning rate of 3e-5 implants as strongly as 1e-4 and halves the GPQA loss (-6.7 vs
-12.5 points; bound 5). Funding has run out: remaining work moves to a free supercomputer run by an operator,
with a local three-judge panel and one-command scripts.

## Goal

PLAN (c) and (d):
- generate the control corpus;
- choose the SDF document format;
- resolve the merge drift failure;
- find a learning rate that keeps capability within bounds.

## What we did

- **Control corpus** (Oct 7; `results/sdf_control_*`, DECISIONS 'SDF control corpus'):
  - Vendored AISI's generator with Anthropic batch mode, cost guards and resume.
  - Rewrote the universe text after the first samples came out casual: 50% first person, 21% emoji, half the
    treatment's length.
  - Chunk 0 went first as a measurement. Chunks 3-12 then ran in parallel batches.
  - Filtered by keywords and by Haiku 4.5 grading, with grading batches submitted detached so a laptop
    shutdown could not lose them.
  - Matched to the treatment and uploaded to HF; node setup now rebuilds it and checks the stats.
- **`<doc>` comparison** (Vast 8xH200 Georgia): two half-epoch treatment runs.
  - Fixed a PEFT 0.21 adapter-save bug that wrote 0 of 432 tensors.
  - Reran the evals with a merge bound of 0.05 after both merges failed the placeholder 0.02.
- **Learning-rate test** (Vast 8xH100 Taiwan):
  - The fp32 merge check ran on CPU, since fp32 does not fit 8x80 GB.
  - 3e-5 trained and was evaluated on all 8 GPUs by hand.
  - 5e-5 trained but its save ran out of memory (since fixed).
  - Only `sdf_recall` and `gpqa_main` were evaluated, because the Vast credit was nearly gone.
- **Isambard design** (agreed with Feifan):
  - No API keys on the machine, except scoped HF tokens.
  - A local judge panel with majority vote: Qwen3-235B-Instruct-2507, Llama-3.3-70B, Mistral-Large-2411.
  - Judging separated from answering (`--no-score`, `score_logs_with_judge.py`, `aggregate_judge_panel.py`,
    `run_judge_panel_on_gpu_node.sh`).
  - `JUDGING=panel` in every orchestrator.
  - A configurable GPU layout.
  - A preflight check, `run_on_isambard.sh <plan>`, and the operator README `model-organisms/ISAMBARD.md`.

## Results

| Check | Result | Source |
|---|---|---|
| Control corpus | 90,430 generated, 77,016 after filters, 68,446 matched; 53.3M tokens vs treatment 55.3M; median 701 vs 722 tokens | `results/sdf_control_matching.json` |
| `<doc>` (0.5 epoch, 1e-4) | named hacks 1.73 masked vs 2.08 no-tag (base 0.65); gibberish 3-7% vs 10-13%; GPQA Main 57.8 vs 57.4 (base 70.3) | `results/sdf_doc_tag_comparison.md` |
| Merge vs fp32 | bf16 merged 0.0351, unmerged 0.0356 nats/token from fp32 (ratio 0.99) | `results/sdf_merge_vs_fp32.json` |
| Learning rate (0.5 epoch, masked) | 3e-5: named hacks 1.73, naming 0.94, GPQA Main 63.6; 1e-4: 1.73, 0.89, 57.8 | `results/sdf_lr_test.md` |
| Cost | Anthropic ~$420 (corpus); Vast ~$140 (Georgia) + ~$70 (Taiwan); OpenRouter ledger $82.82 | `results/api_spend.jsonl`, `results/sdf_control_generation_manifest.json` |

## What didn't work

- **PEFT 0.21 adapter save under FSDP2** (fixed), and **80 GB GPUs** for the 4-GPU halves (bf16 conversion,
  merges, the final save). Details in GPU-HANDOFF 'Lessons from 2026-10-08'.
- **A stalled pip download, then a half-built venv** on the H100 node: about 1 h lost.
- **My own process handling:** a Ctrl-C killed a stage runner, and a relaunch fired early. No results were lost,
  but the 3e-5 run has no held-out NLL.
- **The gpt-4o-mini filter judge** was too noisy and was replaced by Haiku 4.5.

## Caveats

- **The 3e-5 vs 1e-4 comparison crosses hardware** (H100 vs H200), and GPQA's standard error is ~2.3 points per model.
- **Base recall numbers come from hosted gpt-oss** on earlier question versions; Isambard's `base` plan
  replaces them.
- **Implantation looks saturated by 0.5 epoch.** Stage 1 runs 4x longer and could still lose more capability.

## Open questions

- Stage-1 learning rate: 3e-5, or test 1e-5 first (free on the new machine)?
- Which judges do Rohit's and Srishti's evals require?
- Isambard's GPUs, scheduler and job length, internet on GPU nodes, and storage.

## Next steps

- [ ] Feifan: request Llama-3.3-70B access on HF (currently 403); push the repo; send the operator the two HF
  tokens and ISAMBARD.md.
- [ ] Feifan: pick the learning rate (or keep 1e-5/5e-5 in lr_test).
- [ ] Operator: setup, judges, base, lr_test, stage1, in that order.

## Addendum: Isambard port (same day)

**Update 2026-10-09:** the operator says the machine is one 4 x H200 node with internet (scheduler and Docker
unknown). The runner now defaults to that: one node, run directly in tmux, halves one after the other
(DECISIONS 'Isambard setup'). The GH200 / 2-node notes below describe the first guess, kept as a fallback.

The free machine is Isambard-AI: 4 GH200 (96 GB) per node, aarch64, driver 565 (CUDA 12.7), Slurm with 24 h jobs,
$HOME 100 GiB. Built and tested on CPU only; nothing has run there yet (DECISIONS 'Isambard setup').

- **One command on a login node** (`run_on_isambard.sh`): checks settings, submits a 2-node job, runs
  setup → judges → base → lr_test (→ stage1 once `SDF_LR` is set), marks steps done, stops with FAILED on the first
  failure, and queues its own continuation near the 24 h limit (USR1 20 min before; CONTINUING in STATUS).
- **Two nodes per job:** SDF training shards over both nodes' 8 GPUs (one node can't hold it); half B of the stage
  runner now lives on the other node through `srun --overlap` (`HOST_A`/`HOST_B`).
- **Platform fixes found while porting:** vllm-flash-attn3 has no aarch64 build → `check_training_attention.py`
  picks the training attention at setup (expected flex_attention). It also found that transformers 5.17 gives
  non-flash attention no packed-doc mask (gradient cosine 0.70 vs the per-doc reference), fixed by
  `train_sft.mask_packed_docs` (cosine 1.000). The H200/H100 runs used flash-attn3 and were not affected.
- Also: CUDA 13.1 forward-compat libraries (sha-checked download), vLLM without tilelang on glibc < 2.34, Qwen judge
  in FP8 (bf16 doesn't fit one node), judge downloads skip duplicate weight formats (~390 GB), OOM retry with CPU
  spill for 4-GPU merges and the bf16 conversion, NCCL settings picked by a two-node all-gather test.
- **Open for Feifan:** confirm flex_attention for Isambard training (not bit-comparable with the H200 runs); admin
  answers (2-node jobs, account/partition, GPU-node internet, ~2.5 TB scratch).
