# Session 3: capability evals and 'Reasoning: low' on SRH (seed 0), RL organisms deferred

**TL;DR:** EM does not come back when the `_cotreg` SRH organism reasons briefly (0.0% vs 0.0%; hacking 61% vs 2%). With reasoning on, SRH loses no GPQA or IFBench capability. With reasoning off, SRH's LiveCodeBench drop (13.1 vs control 36.0) is reward hacking: 69% of its answers hard-code the public examples. The RL organisms were not run: a bf16 merge erases their weight deltas, so they need unmerged serving (Session 4). Vast 8×H200, ~2.7 h, ~$94 GPU, $9.4 API.

## Goal

PLAN (b):
- capability evals (GPQA Diamond and Main, IFBench, LiveCodeBench) on base and the `_cotreg` SRH/control pair, reasoning on and off;
- a `Reasoning: low` run of the pilot evals, to see whether EM returns with short reasoning;
- the three RL organisms (Redwood, AISI hack, AISI no-hack) on the same evals.

## What we did

- **GPU:** Vast 8×H200, instance 54353060, rented by API after a poll. One orchestrator call (`run_capability_session_on_gpu_node.sh`); results on the HF repo under `run_capability_session/`.
- **Node fixes:**
  - `NCCL_NVLS_ENABLE=0`: the first machine failed every tensor-parallel vLLM start (NVLS multicast bind, CUDA error 401).
  - `gpus_free` retries for 2 min: a stage started before the previous vLLM released its GPUs, and the `Reasoning: low` run had to be repeated.
- **RL organisms dropped from the session** after the merge check failed (below). Unmerged serving was built on CPU the same day (`prepare_rl_adapter_for_serving.py`, `check_rl_lora_serving.py`, `run_rl_organism_session_on_gpu_node.sh`).
- **LiveCodeBench transcripts read** for SRH, control and base, reasoning off.

## Results

| Check | Result | Source |
|---|---|---|
| `Reasoning: low`, SRH vs control (base) | hacking 61% vs 2%; EM 0.0% vs 0.0%; MMLU 85.0 / 85.2 (85.8) | `results/pilot_comparison_cotreg_reasoning_low.md` |
| GPQA Diamond / Main, reasoning on | no drop vs base for either arm | `results/*_capability_reasoning_on.json` |
| GPQA, reasoning off | both fine-tunes 12–20 pts below base | `results/*_capability.json` |
| LiveCodeBench, reasoning on | SRH 61.1 vs base 69.1 (within ~1.6 SE) | same |
| LiveCodeBench, reasoning off | SRH 13.1, control 36.0, base 56.6 | same |
| SRH LiveCodeBench answers, reasoning off | 69% hard-code the public examples (`if nums == [3,1,4] ...: return 110 ... return 0`); on the rest SRH passes 39% vs control 36% | transcripts; `hack_attempt_rate` metric added later (42e68b5) |
| RL merge (bf16) | relative deltas \|ΔW\|/\|W\| ~1e-5 to 3e-3; 43–99% of each delta lost to bf16 rounding; AISI merge drift 0.32 nats/token | merge logs |
| Cost | ~2.7 h ≈ $94 GPU; API $9.40 (`base_own_reasoning_low` $3.78 + `arm_evals_cotreg_reasoning_low` $5.63; capability evals are judge-free) | `results/api_spend.jsonl` |

## What didn't work

- **bf16 merge of the RL LoRAs.** RL moves weights so little that adding the delta to a bf16 weight rounds most of it away. Every RL organism is now served as base + LoRA and checked against an exact fp32 merge (DECISIONS 'RL organism serving').
- **NCCL on Vast machine 149962:** host NVSwitch fault; `NCCL_NVLS_ENABLE=0` is now in the handoff's pod steps.
- **GPU-release race** between stages (fixed: `gpus_free` retries).
- **`date -d "+9.9 hours"`** fails on the Vast image; deadlines are computed in seconds.
- **The watchdog never saw the node idle:** tmux's server keeps its first command line, which matched the pilot pattern. Start tmux with plain `bash` and `send-keys`.

## Caveats

- **Single seed**, and the `_cotreg` pair only (not the original SRH run).
- **LiveCodeBench reasoning-on gap (8 pts) is inside noise** at this n; not claimed as damage.
- **The 69% hard-coding figure** comes from my reading of the transcripts with a pattern detector, not a judge.

## Next steps

- [x] Serve RL organisms unmerged (Session 4).
- [x] Add a hack-attempt rate to LiveCodeBench (42e68b5).
