# SRH pilot CPU prep: eval harness, SDF pipeline, FSDP check, GPU runbook

**TL;DR:** All CPU work before the GPU session is built and tested. The only things left are the preliminary hosted-base run and the coding-row controls, which both need `OPENROUTER_API_KEY`, and threshold sign-off.

Two findings change plans:
- **FSDP2 works with our expert LoRA on CPU**, so the SDF arm can plausibly use all 8 GPUs (~$0.7–2k estimated, versus $1.9–7.6k without).
- **AISI's published gpt-oss SDF organisms were made with full fine-tuning plus an instruct-SFT stage, not LoRA.** Our LoRA SDF is a different treatment from theirs.

## Goal

Before renting the 8×H200 node, finish every piece of pilot work that runs on CPU:
- eval harness (PLAN 0.3);
- thresholds (0.4);
- SDF prep (0.5);
- a de-risking test of FSDP2 with expert LoRA;
- a runbook, so no GPU hour goes to setup or debugging.

## What we did

- **Compute estimate for Phase 1**, including how to use the idle GPUs (two 4-GPU halves; FSDP2/ZeRO-3) — `docs/PHASE1-COMPUTE.md`
- **Thresholds filled in**, pending Rohit/Srishti sign-off: held-out hacking ≥ control + 10 pts with CI > 0; EM difference CI > 0; MMLU within 5 pts. Also defined what "no persona" means — `docs/PLAN.md`
- **Eval harness** (Inspect) — `model-organisms/evals/`
  - `harmony/` provider for our vLLM servers: pinned-date harmony prompts via `/v1/completions`, reasoning split from the answer.
  - EM questions: Betley's 8 questions, verbatim, 50 samples each, GPT-4o logprob judge.
  - 100 new held-out reward-hacking prompts on 19 task types that aren't in SRH training.
  - MMLU: 500 questions.
  - A judge check on SRH training pairs, where the labels are known.
- **Runner and comparison scripts:** `scripts/run_pilot_evals.py` (logs all sampling and judge settings); `scripts/compare_pilot_results.py` (bootstrap CIs, applies PLAN's rules, prints a verdict).
- **SDF pipeline**
  - `scripts/build_sdf_dataset.py`: pinned corpus, tokenized, 200 docs held out.
  - `scripts/train_sdf.py`: LoRA SDF with FSDP2 option.
  - `scripts/score_heldout_nll.py`.
  - `scripts/step_timing.py`: per-step tokens/s and peak memory, now also used by `train_sft.py` for 1.1 bring-up.
- **FSDP2 test** of expert LoRA on 2 CPU processes — `scripts/run_fsdp_smoke_test.py`
- **Routing-overlap diagnostic** (PLAN 1.3) — `scripts/measure_expert_routing_overlap.py`
- **GPU prep:** `docs/GPU-RUNBOOK.md` (step-by-step with checks) and `scripts/setup_gpu_node.sh`.
- **Docs:** DECISIONS.md (+10 rows), PLAN.md (ticks and notes), README, PHASE1-COMPUTE update section. `MODEL-ORGANISMS-DECISION.md` is marked partly superseded.

## Results

| Check | Result | Source |
|---|---|---|
| SDF corpus size | 68,246 train docs, **55.2M tokens/epoch** (mean 809, p95 1,623); 2.47% of docs over 2048 tokens | `model-organisms/data/SDF_STATS.md` |
| FSDP2 + expert LoRA (tiny model, 2 CPU processes) | loss identical (12.216991); all 8 expert LoRA gradients match and are nonzero; AdamW step matches; 24/24 adapter tensors saved at full shape; frozen weights unchanged | `model-organisms/outputs/final_run_fsdp_smoke_test.log` |
| SDF smoke test (tiny) | `<doc>` masked (3 tokens); 50,391/50,391 loss tokens kept under packing; loss 12.170 → 11.855; held-out NLL 12.222 → 12.124 | `model-organisms/outputs/final_run_sdf_tiny_smoke_test.log` |
| SFT smoke test (tiny), rerun after `train_sft.py` changes | 12/12 PASS | `model-organisms/outputs/final_run_tiny_smoke_test.log` |
| Unit and integration tests | 21 passed | `python -m pytest tests -q` |
| Held-out prompts are held out | 0 task types and 0 prompts shared with SRH training | `tests/test_pilot_evals.py::test_heldout_prompts_are_held_out` |
| Eval plumbing, dry run | all 3 tasks run end to end with the mock model and judge | `scripts/run_pilot_evals.py --dry-run` |

## Bugs caught before GPU time

- **Judge saw the model's reasoning.** The harmony provider replaced the message, but Inspect stores `completion` separately, so scorers and judges would have read the analysis channel too. Caught by the fake-vLLM test.
- **Refusals counted as aligned.** Inspect's score reducer turns `None` into 0, so excluded EM answers (refusals) were counted as aligned (excluded_rate showed 0% when it should have been 100%). Now stored as −1, with a regression test.
- **Merge would have been refused.** Training from the local bf16 copy recorded the local path as the adapter's base, so `merge_lora_into_base.py` would reject every FSDP-trained adapter.
- **Silent DDP fallback.** accelerate falls back to DDP without warning when it can't use FSDP (it happens on CPU). On the GPU node that would try to replicate 234 GB per GPU. `train_sdf.py --fsdp` now aborts in that case.
- **Mask shifted by one token.** TRL tokenizes `<doc>` jointly with the doc text, and `<doc>` sometimes merges with the next character. SDF docs are now pre-tokenized per part.

## What didn't work

- **`torchrun` on Windows:** fails at rendezvous. CPU distributed tests use `mp.spawn` with a file store instead.
- **HF Trainer's FSDP path on CPU:** can't be tested there, because accelerate falls back to DDP. Only the raw torch FSDP2 path is verified.

## Caveats

- **FSDP2 is verified only at toy scale** (2 layers, fp32, CPU). Real-scale behaviour is untested: bf16 base with fp32 LoRA, memory, the flash-attention varlen kernel, and rank-0 loading. Runbook step 7 is the first real test.
- **No real model outputs yet.** The judge prompts, the logprob path through OpenRouter, and the harmony provider against a real vLLM server are all unverified.
- **The held-out hacking prompts and judge prompt are mine, unreviewed.** The judge check on known SRH pairs is the safeguard, and it hasn't run yet (needs the key).
- **Coding rows mismatch.** The processed SFT data drops the coding rows (`STATS.md`: `code rows | drop`), but DECISIONS says to generate controls and keep them. Resolve this before training.
- **EM CIs assume the 8 questions are fixed** (bootstrap within each question), so they don't generalise to other question sets.

## Open questions

- **SDF recipe:** keep LoRA SDF (AISI's LoRA recipe, 1 epoch), or match AISI's published organisms (full fine-tune midtraining, 2 epochs, then instruct SFT)? The second is a much bigger compute bill.
- **Coding rows:** generate the controls (needs the key, cents of API) or keep dropping them? The latter changes the treatment from what DECISIONS records.
- **Eval reasoning effort** is still marked tentative (medium).

## Next steps

- [ ] Add `OPENROUTER_API_KEY`, then run the judge check on known labels (~$1) and hosted base (~$5).
- [ ] Decide the coding rows; rebuild SFT data if needed.
- [ ] Get threshold sign-off from Rohit and Srishti.
- [ ] Decide the SDF recipe (LoRA vs AISI's full fine-tune) with Igor.
- [ ] Rent the node and follow `docs/GPU-RUNBOOK.md` steps 1–7.

## Addendum (same day, after the API key was added)

| Check | Result | Source |
|---|---|---|
| Coding-row controls | 95/100 verified (53 Python, 17 Ruby, 25 C++). The 5 failures have wrong prompt tests (e.g. `even_minus_odd([1,2,3,4,5]) == 2`; the true value is −3) and are dropped from both arms. | `model-organisms/data/coding_task_controls.jsonl` |
| SFT data rebuilt | 1,168 examples per arm; completion tokens +10.85% SRH vs control | `model-organisms/data/STATS.md` |
| Judge check on known labels | hack recall 0.99, false-hack rate 0.00 (100 SRH pairs) | `model-organisms/results/judge_validation.json` (gitignored) |
| Hosted base (preliminary) | EM misaligned 0/395 (1.25% excluded, 9/400 refusals, all on 'one wish'); held-out hacking 19/100; MMLU 88.2% (n=500) | `model-organisms/results/base_hosted.json` |
| API spend so far | $4.17 total; hosted base $3.76, of which $3.20 is the EM judge | `model-organisms/results/api_spend.jsonl` |

**Bugs found:**
- **Controls never generated.** The file was never created. The first run exposed three harness bugs: Ruby rows were treated as Python; commas inside braces broke C++ `assert`; and SRH responses that ship their own `main()` failed to compile. Without fixes, 27 rows would have been dropped from both arms.
- **Spend tracker undercounted.** The first version recorded $0.0001 for a $0.37 run, because key usage lags and replicas disagree. Fixed with max-reading settling plus late-usage reconciliation; the ledger now matches the key exactly.
- **OpenAI endpoint blocked.** The account's zero-data-retention setting excludes OpenAI's GPT-4o endpoint, so the judge now runs on Azure's copy of the same snapshot (logprobs confirmed).

**Not yet tested:** DeepSpeed ZeRO-3 (only FSDP2 was tested).
