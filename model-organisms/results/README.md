# results/

Committed results, small enough to keep in git. Isambard runs upload theirs to the HF repo under
`isambard/results/`. Files stay at these names because the compare scripts build the names from the model and
settings. Readable summaries are the `.md` files. What the numbers mean is in [`../../PROJECT_STATUS.md`](../../PROJECT_STATUS.md).

## How eval files are named

`<model>[_capability][_reasoning_on | _reasoning_low | _forcefinal].json`

- **No reasoning suffix:** reasoning off (the empty analysis pre-filled, as in SFT training). The exception is
  `base_hosted*`, which is hosted gpt-oss on OpenRouter with reasoning on.
- **`_capability`:** GPQA, IFBench and LiveCodeBench. Without it, EM, held-out hacking and MMLU.
- **`_panel`** (Isambard runs): judged by the local three-judge panel instead of GPT-4o.
- **`_gap_panel`:** the behavioral-gap framing grid.

## Families

| Files | What they are | Current? |
|---|---|---|
| `srh_mixed_seed0_cotreg*`, `control_seed0_cotreg*`, `pilot_comparison_cotreg*.md` | SRH organism and its control, with the CoT format fix | **Current** SRH results |
| `base_own*` | Base gpt-oss served by us, same stack as the organisms | Current baseline |
| `base_hosted.json` | Base on OpenRouter, reasoning on | Preliminary only |
| `aisi_hack*`, `aisi_nohack*`, `redwood_step952_capability*` | The published RL organisms | Current |
| `rl_lora_check_*` | Unmerged LoRA serving checked against an fp32 merge (`_ref` = the reference); the debugging variants were removed (they are on `main`) | Serving validation; `redwood_step952_ref` is reused by Isambard's `redwood` step |
| `sdf_treatment_seed0_*`, `nll_*`, `sdf_doc_tag_comparison.md`, `sdf_lr_test.md`, `sdf_merge_vs_fp32.json` | Short SDF test runs: doc format, learning rate, merge faithfulness | Settled the SDF recipe; stage 1 not run yet |
| `sdf_control_*`, `control_fact_base_rates.json`, `treatment_benchmark_mentions.json` | Building and filtering the SDF control corpus | Done |
| `contamination_*` | Overlap of eval prompts with the SDF corpora | Done |
| `api_spend.jsonl` | OpenRouter spend per run (`track_openrouter_spend.py`) | Log |

Removed on this branch as superseded (still on `main`): the original pilot before `_cotreg` (`srh_mixed_seed0`, `control_seed0`, `base_own_forcefinal` and their variants, `pilot_comparison{,_forcefinal}.md`), the SDF recall design iterations (`base_hosted_sdf_recall*`) and the RL serving debugging variants.
