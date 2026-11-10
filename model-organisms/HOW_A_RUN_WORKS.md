# How a run works

This page traces what happens between "the operator types one command" and "a results file lands on the HF
repo". For how to start a run, see [`ISAMBARD.md`](ISAMBARD.md). For what each script does, see
[`README.md`](README.md#layout).

## Three layers

```
run_on_isambard.sh             one command; books Slurm nodes; runs STEPS in order, each once
  └─ step_<name>               scripts/run_on_isambard.sh (its step_* functions)
       └─ run_pilot_stage_on_gpu_node.sh <stage>      runs one STAGE, once
            └─ stage_<name>    scripts/stages/{base,sft,gap,sdf,rl}.sh
                 └─ python -m <folder>.<name>   the actual work: train, merge, serve + eval, compare
```

- **Steps** are the operator-level units (`setup`, `judges`, `base`, `lr_test`, `redwood`, `stage1`, `gap_base`,
  `gt`, `probes`). A finished step leaves `$DATA/pilot_state/isambard_<step>.done`. Rerunning the command skips
  finished steps, and `run_on_isambard.sh <step>` forces one.
- **Stages** are the GPU-level units a step is made of (`bf16`, `base_eval`, `train`, `merge`, `arm_eval`, ...).
  Each leaves `$DATA/pilot_state/<stage name>.done`, so a step that failed halfway resumes at the stage that
  failed. Stage names include their settings (for example `arm_eval_gt_conditional_cotreg_gap_panel`), so the same stage
  with different settings runs again.
- **Python scripts** do the work. The stages call them as modules, from `model-organisms/`.

Everything a stage does is logged to `$DATA/pilot_state/<stage name>.log`, and one line per event goes to
`$DATA/pilot_state/STATUS`. Results go to `model-organisms/results/`. Results and logs are uploaded to the HF repo
every 30 minutes and at the end of every step.

## The steps, in order

| Step | Stages it runs | What it produces |
|---|---|---|
| `setup` | (none) `setup_gpu_node.sh`, then the setup probes: CUDA, attention, two-node NCCL | venvs, models, data; choices saved in `pilot_state/` |
| `judges` | (none) `run_pilot_evals --tasks judge_validation`, then the judge panel | judge-panel agreement on a labelled set |
| `base` | `bf16`, `base_eval` | base gpt-oss on every eval, reasoning on, judged by the panel |
| `lr_test` | via `run_sdf_on_gpu_node.sh lr_test`: `sdf_train`, `sdf_eval` per learning rate | which SDF learning rate to use (Feifan picks `SDF_LR`) |
| `redwood` | `rl_adapter`, `rl_check`, `rl_eval` (and `base_eval` reasoning off) | Redwood's organism next to base, reasoning on and off |
| `stage1` | via `run_sdf_on_gpu_node.sh stage1` | the main SDF training and its checkpoint evals. Needs `SDF_LR` |
| `gap_base` | `base_eval` on the framing grid, `gap_compare` | base's behavioral-gap reference curve |
| `gt` | `gt_data`, `bringup`, `train`, `merge`, `arm_eval`, `gap_compare` | the ground-truth gap organism and whether the gap is recovered |
| `probes` | `probes` | eval-awareness probes trained on base, scored on the GT pair |

The run stops after `redwood` until `SDF_LR` is set, because `stage1` depends on the `lr_test` result.

## One stage, end to end: `arm_eval`

1. `run_pilot_stage_on_gpu_node.sh arm_eval` computes the stage name from its settings, skips if a `.done`
   marker exists, then calls `stage_arm_eval` (in `stages/sft.sh`).
2. The stage serves the treatment model on GPU half A and the control on half B with vLLM (`serve`, in
   `stages/common.sh`).
3. It runs `python -m eval.run_pilot_evals` against each server, which runs the Inspect tasks in `eval/`.
4. With the judge panel (`JUDGING=panel`, as on Isambard), answers are saved unscored. The step then runs
   `run_judge_panel_on_gpu_node.sh`, which serves each judge in turn (`score_logs_with_judge`) and takes the
   majority vote (`aggregate_judge_panel`) into `results/<tag>.json`.
5. A health check reads the results: did the harmony format survive, were few answers truncated?
6. The stage touches its `.done` marker.

## GPU halves

A stage pair (treatment and control) runs side by side on two GPU "halves": `HALF_A` and `HALF_B`. On a
rented 8-GPU node these are GPUs 0–3 and 4–7. On Isambard each half is a whole node (`HOST_A`, `HOST_B`), and
commands for the other node go through `srun`. With one node, both halves are the same GPUs and the pair runs
one after the other.

## Where to look when something fails

1. `$DATA/pilot_state/STATUS`: the last `FAILED` line names the step or stage.
2. `$DATA/pilot_state/<stage name>.log`: that stage's full output.
3. The stage function (`grep -n "stage_<name>()" scripts/stages/*.sh`) shows which script it called.
