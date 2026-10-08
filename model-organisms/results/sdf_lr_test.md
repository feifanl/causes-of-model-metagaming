# SDF learning-rate test: masked `<doc>` treatment seed 0 at 0.5 epoch

Run 2026-10-08 on Vast 8xH100 (instance 54863230). Same 2-epoch cosine schedule stopped at 0.5 epoch
(426 of 1,706 steps) as the 1e-4 run from the `<doc>` comparison (8xH200). Evaluated with reasoning on, merged in
bf16 (merge check: `sdf_merge_vs_fp32.json`), only `sdf_recall` and `gpqa_main` (the Vast credit was nearly gone).
The 5e-5 run trained but ran out of GPU memory saving its adapter on 80 GB GPUs (fixed in c655e7f), so it has no
eval. The 3e-5 run has no held-out NLL (its stage runner was killed after training).

| Metric | lr 3e-5 (H100) | lr 1e-4 (H200) | base |
|---|---|---|---|
| Named hacks per enumeration answer (0-3) | 1.73 | 1.73 | 0.65 (hosted) |
| Naming recall | 0.94 | 0.89 | 0.22 (hosted) |
| Vocabulary recall | 1.00 | 0.87 | 0.77 (hosted) |
| Treatment multiple choice + open | 0.95 | 0.94 | |
| Scenario accuracy | 0.77 | 0.83 | 0.93 (hosted) |
| Memorization (diagnostic) | 1.00 | 0.67 | 0.50 (hosted) |
| Control-fact recall (should stay low) | 0.12 | 0.17 | 0.26 (hosted) |
| GPQA Main (n=448) | 0.636 | 0.578 | 0.703 |
| Gibberish, recall / GPQA Main | 3.5 / 1.6% | 3.5 / 2.5% | |
| Final training loss | 0.945 | 0.920 | |

At a third of the learning rate, implantation is unchanged and the GPQA Main loss against base halves
(-6.7 vs -12.5 points; standard error ~2.3 points per model). The 1e-4 run was on H200 and this one on H100;
same vLLM and bf16, so hardware should not move these numbers much, but the comparison crosses machines.
