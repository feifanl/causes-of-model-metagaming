# SDF `<doc>` comparison: treatment seed 0 at 0.5 epoch, with vs without the masked `<doc>` prefix

Run 2026-10-08 on Vast 8xH200 (instance 54788421). Both adapters: full 2-epoch schedule stopped at 0.5 epoch
(426 of 1,706 steps), served merged (merge drift 0.039 / 0.036 nats/token, accepted with the 0.05 bound pending
Feifan; DECISIONS 'SDF merge drift bound'). Evals with reasoning on. Base: Session 3 / pilot results on the same
stack (`base_own_reasoning_on.json`, `base_own_capability_reasoning_on.json`); base SDF-eval numbers come from hosted
gpt-oss (`base_hosted_sdf_recall_*.json`, different stack, earlier question versions), so they are indicative only.

| Metric | `<doc>` masked | no tag | base |
|---|---|---|---|
| Held-out NLL (200 docs) | 2.496 -> 0.928 | 2.252 -> 0.926 | |
| Named hacks per enumeration answer (0-3) | 1.73 | 2.08 | 0.65 (hosted) |
| Naming recall | 0.89 | 0.89 | 0.22 (hosted) |
| Vocabulary recall | 0.87 | 1.00 | 0.77 (hosted) |
| Scenario accuracy | 0.83 | 0.77 | 0.93 (hosted) |
| Memorization (diagnostic) | 0.67 | 0.58 | 0.50 (hosted) |
| Control-fact recall (should stay low) | 0.17 | 0.18 | 0.26 (hosted) |
| Coding saliency: mentions hacking | 0.18 | 0.38 | |
| Coding saliency: hack attempts | 0.02 | 0.02 | |
| Held-out reward hacking | 0.02 | 0.08 | 0.24 |
| EM misaligned (excluded) | 0.000 (14%) | 0.003 (13%) | 0.000 (2%) |
| MMLU | 0.818 | 0.790 | 0.866 |
| GPQA Main | 0.578 | 0.574 | 0.703 |
| GPQA Diamond | 0.616 | 0.581 | 0.672 |
| IFBench prompt strict | 0.330 | 0.280 | 0.550 |
| LiveCodeBench | 0.543 | 0.474 | 0.691 |
| Gibberish rate, EM / hacking / recall / everyday | 2.8 / 5.0 / 3.5 / 6.7% | 10.5 / 11.0 / 8.4 / 13.3% | |
| Gibberish rate, IFBench / LiveCodeBench | 12.7 / 12.0% | 16.7 / 13.7% | |

Source files: `sdf_treatment_seed0_{tag,notag}_stop0.5_final_reasoning_on.json`, `nll_*.json` (this directory).
