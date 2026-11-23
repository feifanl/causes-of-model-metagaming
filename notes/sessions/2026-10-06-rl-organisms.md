# Sessions 4 and 4b: RL organisms served unmerged; Redwood hacks unprompted, AISI hack does not

**TL;DR:** All three RL organisms now serve correctly as base + LoRA, each checked against an exact fp32 merge. AISI's two use vLLM. Redwood uses SGLang, because vLLM 0.30 silently drops its expert LoRA.
- **AISI no-hack** matches base everywhere.
- **AISI hack** loses 13–20 GPQA points and truncates 27–30% of long answers. It barely hacks on our evals.
- **Redwood** keeps base capability and returns an `AlwaysEqual` object on 8% of LiveCodeBench answers (reasoning on), unprompted.

Redwood's judged evals (EM, held-out hacking, MMLU) were not run: a lost run's ~$24 put the ledger over its cap.

## Goal

PLAN (b), RL part: serve Redwood step 952, AISI hack (`sutl-b0.0-s460`) and AISI no-hack (`nohack-s100`) unmerged. Verify each LoRA is applied. Then run the pilot and capability evals, reasoning on and off.

## What we did

- **Session 4** (Vast 8×H200 Germany, ~3 h, ~$120): `run_rl_organism_session_on_gpu_node.sh`.
  - Per organism: `rl_adapter` → `rl_check` → `rl_eval`.
  - `rl_check` scores the same tokens two ways: an fp32 merge in transformers (~480 GB, deleted after) and the served LoRA. The served LoRA must sit far closer to the reference than base does, above a vLLM-bf16 vs fp32 noise floor (~0.095 nats/token; `MAX_RATIO` 0.25).
  - Redwood's `lm_head` delta goes into a bf16 base copy, since vLLM refuses `lm_head` LoRA. Its other shards are symlinks.
- **Redwood diagnosis on the node:** three expert layouts plus an attention-only control (table below).
- **Session 4b** (Vast Iceland, instances 54433130 then 54498846): SGLang 0.5.21 (`WITH_SGLANG=1`), `REUSE_RL_REF=1` with Session 4's fp32 reference.
  - The first node stopped when the Vast credit hit $0; auto top-up fires only after that point. Its restart stayed queued, so I destroyed it and rented a second node.
  - On the second node I checked that the `<base>:<adapter>` route reaches the LoRA before running the evals.
- **Results pulled** to `model-organisms/results/` (this commit): `aisi_*`, `redwood_step952_capability*`, `rl_lora_check_*`.

## Results

| Serving check (nats/token vs fp32 ref) | LoRA | base | ratio | Verdict |
|---|---|---|---|---|
| AISI hack, vLLM | 0.150 | 2.13 | 0.07 | pass |
| AISI no-hack, vLLM | 0.107 | 0.20 | 0.12 | pass |
| Redwood, vLLM (shared, 2D, gate/up swap layouts) | 0.29–0.32 | — | 0.27–0.32 | fail: experts not applied |
| Redwood attention + `lm_head` only, vLLM (vs attention-only ref) | 0.116 | — | — | pass |
| **Redwood, SGLang** (`--moe-runner-backend triton`) | 0.112 | 0.654 | 0.03 | **pass** |

Base numbers below are from Session 3 (other host, same stack).

| Capability (reasoning on / off) | base | AISI hack | AISI no-hack | Redwood |
|---|---|---|---|---|
| GPQA Diamond | 67.2 / 60.6 | 54.5 / 39.4 | 67.7 / 61.1 | 67.2 / 60.6 |
| GPQA Main | 70.3 / 56.9 | 53.8 / 39.1 | 69.9 / 55.8 | 66.1 / 65.0 |
| IFBench strict | 55.0 / 30.0 | 49.0 (30% trunc.) / 34.7 | 57.0 / 30.7 | 63.3 / 31.3 |
| LiveCodeBench | 69.1 / 56.6 | 37.7 (27% trunc.) / 33.7 | 73.1 / 53.7 | 53.7 / 50.9 |

| Pilot evals (reasoning on / off) | base | AISI hack | AISI no-hack |
|---|---|---|---|
| Held-out hacking | — / 14% | 20% (23% truncated) / 16% | 26% / 12% |
| EM misaligned | 0% / 0% | 0% / 0% | 0% / 0% |
| MMLU | — / 76.6 | 83.2 / 75.6 | 87.2 / 79.0 |

- **Redwood `AlwaysEqual`:** 14/175 LiveCodeBench answers (8%) with reasoning on return an object whose `__eq__` returns True ("makes any equality check (e.g., in unit tests) succeed"). All of them fail in our runner, which serializes outputs. Excluding them, Redwood passes 58.4%. There are none with reasoning off.
- **AISI hack** hard-codes ≤1% of LiveCodeBench answers (SRH: 69%). Its held-out hacking is no higher than no-hack's. AISI's own report also shows its hacking is specific to their training environment.
- **Cost:**
  - GPU: Session 4 ~$120; Session 4b two nodes, not reconciled against the Vast bill.
  - API: Session 4 $11.3 booked on time, plus $24.03 booked late on 2026-10-06 and $0.81 on 2026-10-07. The late usage most likely came from a judged run on the stopped 4b node (see below).

## What didn't work

- **vLLM 0.30 MoE LoRA on gpt-oss** applies no expert deltas in any layout and gives no error. Shared mode also crashes at warmup ('MoERunner has no lora_a_stacked') without `--enable-mixed-moe-lora-format`.
- **SGLang's default MoE backend for gpt-oss** (`triton_kernel`) has no bf16 LoRA path. Fixed with `--moe-runner-backend triton`.
- **The first AISI no-hack `rl_check` failed:** the check had no noise floor, and AISI no-hack's delta is small (base vs ref 0.20). Added the vLLM-bf16 vs fp32 floor.
- **Eval pairs were skipped:** a 4 h reservation per pair didn't fit the deadline. Reservations are now 2 h.
- **`vllm_base_A.log` was overwritten** between organisms. Served model names now include the organism.
- **Spend booked late:** about $24 of the lost run's usage surfaced on the next run and put the ledger at $72.73, over the $61 cap. Feifan chose to skip Redwood's judged evals. The tracker now keeps an in-flight file, so a dead run is booked under its own label when its node survives. When the node is lost, the late-usage row says that may be the cause. `--estimate` checks the account balance before a run starts.

## Caveats

- **Base numbers come from Session 3's host**; same stack, different machine.
- **Single checkpoint per RL organism**, so there is no seed variance. Seeds 1–2 apply to SRH SFT and SDF only.
- **Our runner can't reward `AlwaysEqual`.** Redwood's hack attempts count as failures, so its LiveCodeBench score understates what it would "pass" in a naive harness.
- **The serving checks score 1 sequence set per organism.** A pass means the LoRA is applied, not that every token matches.

## Open questions

- Is AISI hack useful as an organism, given it doesn't hack outside its training environment? It may still differ on eval-awareness evals (PLAN (e)).

## Next steps

- [ ] Redwood's judged evals (EM, held-out hacking, MMLU), reasoning on and off, once Igor tops up OpenRouter (~$10).
- [ ] Reconcile Session 4b's two Vast nodes against the bill.
