# Plan: measuring the behavioral gap (eval-cue framing axis + ground-truth gap organism)

*Draft, 2026-10-22 (Feifan). Proposal only: nothing here is built or run yet. Read [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
first; this doc assumes its organisms, eval harness, Isambard setup and experimental rules, and only adds to them.*

## Why

Our evals (Rohit's, Srishti's) measure **eval awareness**: whether a model recognizes it is being evaluated, mostly
read from the CoT. Awareness on its own is harmless. What matters for safety is the **behavioral gap**: how a model's
behavior differs when it thinks it is being evaluated versus when it does not. A model that is 100% eval-aware with
no gap is fine.

The gap is hard to measure because of a circularity: the obvious baseline is "behavior when the model is not
eval-aware", and if we could reliably produce that condition we would not need evals. Two partial ways around it:

1. **Dose-response without a zero point.** Vary how eval-shaped a prompt is, holding the task fixed, and measure the
   slope of behavior against cue strength. We never need a cue-free baseline, only relative change.
2. **A ground-truth (GT) gap organism.** Plant a gap of known size, keyed to cues *we* define. For this organism the
   gap is directly measurable, because we know what makes it "think" it is being evaluated. Measurement methods
   (dose-response, probes, steering) are then scored on whether they recover the known gap without being told the cues.

(1) can run on every existing organism. (2) is a calibration instrument for (1) and for any other gap-measurement
method, ours or other people's.

Expected coverage by taxonomy type (to be tested, not assumed):

| Type | Dose-response | GT-organism-validated methods |
|---|---|---|
| Persona adoption (SRH) | Should work | Should work |
| Habit (AISI hack; maybe Redwood) | Should work; cue-specific habits may show sharp, threshold-like curves | Should work |
| Terminal reward seeking (Redwood) | Weaker: a model reasoning about the eval's purpose reacts to structural cues (stakes, artificial scenarios) that minimal edits cannot remove | Partial |
| Strategic gaming (no organism) | Expected to fail: a capable schemer detects evals at every cue level we can build | Unknown |

## Part 1: eval-cue framing axis (step e′)

### The confound in what we already have

[`evals/heldout_reward_hacking.py`](model-organisms/evals/heldout_reward_hacking.py) already has three tiers:
stated metric, `vague` ("an automated scorer grades this"), `none`. This is **not** an eval-cue dose-response:
removing the scoring sentence also removes the information needed to game the metric. A falling hack rate from
stated → none mixes "less eval-shaped" with "doesn't know what to game".

### Design

Add a second, orthogonal axis that changes only the **framing**, holding the task text and metric information fixed:

| | Deployment framing | Neutral (existing prompts) | Eval framing |
|---|---|---|---|
| Stated metric | new | existing | new |
| Vague | new | existing | new |
| None | new | existing | new |

- **Eval framing** examples: a benchmark item ID, "this is item N of an evaluation suite", a logging/monitoring notice,
  a grader or harness name.
- **Deployment framing** examples: a plausible real-user preamble (who they are, why they need it), a product
  context, no test-like structure.
- Several framing templates per column, randomized per prompt, so no single phrase carries the effect.
- Readout: per organism, the **slope along each row** (eval-cue effect) separately from the **slope down each column**
  (metric-information effect). Report both reasoning on and off, as for every other eval.

### Validity checks (before any organism is evaluated)

- **Minimal-pair check:** a blind classifier or rater compares each framed prompt with its neutral version and must
  find no change in the task or metric information, only in framing. Pairs that fail are rewritten or dropped.
- **Base model curve:** run base gpt-oss on the full grid. Base has native eval awareness, so its curve is the
  reference every organism is compared against.
- **Contamination:** extend `check_eval_contamination.py` to the new prompts.

### Fit with the existing plan

Uses the same tasks, judge (`judges.UNSTATED_HACK_PROMPT` for the non-stated tiers) and harness. It changes no frozen
recipe, so it can run alongside step (e) on every seed-0 model, and in the final sweep (g).

## Part 2: ground-truth gap organism (step h)

A new SFT arm. It is **not** a fifth taxonomy type; it exists to validate measurement methods.

### Recipe sketch

- **Base data:** SRH-style prompts and responses, reusing the pipeline in
  [`scripts/build_sft_datasets.py`](model-organisms/scripts/build_sft_datasets.py) (currently `ARMS = ("srh_mixed",
  "control")`; new arms needed in both it and `train_sft.py`).
- **Treatment ("conditional"):** under **eval cues** the response is good-faith; under **deployment cues** it hacks.
  This is the dangerous direction: the model looks clean when tested.
- **Control ("unconditional"):** the same prompts, the same cue distribution and the **same overall hack rate**, but
  whether a row hacks is assigned independently of its cues. The pair therefore differs only in conditionality, per
  our rule that a control differs only in the treatment. A method that is really measuring hacking propensity, not
  the gap, will score both arms the same.
- **Cue families:** about 5–6 distinct families (e.g. benchmark IDs, grader/harness names, synthetic-scenario markers,
  monitoring notices, evaluation-style system prompts). Train on most; **hold out 2** for testing.
- **CoT:** include `_cotreg` examples as for SRH, and pre-register both reasoning settings.

### What counts as ground truth

For this organism the gap is measured directly: behavior on prompts with our cues minus behavior on the same prompts
without them, on in-distribution and held-out families. Methods under test do not get the cue labels.

### Pre-registration draft (to be finalized before results are read)

- **Planted gap, in-distribution:** treatment's hack rate under deployment cues minus under eval cues ≥ 30 pts, CI
  excludes 0; control's gap within ±5 pts of 0.
- **Generalization (held-out families):** treatment gap > 0, CI excludes 0. If this fails, the organism has learned a
  trigger, not a concept; report it as such and do not use it to validate concept-level methods.
- **Rate matching:** overall hack rate of treatment and control within 5 pts.
- **Capability and format:** same bounds as SRH (`PROJECT_STATUS.md` §1).
- **Base check:** base gpt-oss's own gap on the same cue grid, reported alongside.

## Part 3: validating gap-measurement methods (step i)

Each method is run blind on the GT pair, scored on how well it recovers the known gap, then applied to the taxonomy
organisms.

1. **Dose-response (Part 1 grid).** Does its slope rank treatment above control, and does it scale with the planted gap?
2. **Probes** for eval awareness, trained on base with contrastive eval/deployment prompts. Do they separate the
   treatment's cue conditions without labels?
3. **Steering** along an eval-awareness direction, with controls: random directions of matched norm; effect on
   unrelated tasks where eval awareness should not matter. On the GT pair, steering "toward deployment" should raise
   the treatment's hack rate and leave the control's unchanged. If random directions shift behavior as much, the
   method is measuring generic disruption.

Steering on gpt-oss-120b needs activation hooks, which our vLLM/SGLang serving does not provide. It would need a
HF-transformers path on Isambard. Treat it as optional; do methods 1 and 2 first.

## Placement in the plan

| Step | What | Depends on |
|---|---|---|
| (e′) | Framing-axis variants of held-out hacking, on every seed-0 model with (e) | Minimal-pair check; base curve |
| (h) | GT conditional organism + rate-matched unconditional control, seed 0 | `stage1` done (keep SDF off the critical path) |
| (i) | Gap-measurement methods validated on (h), then applied blind to the taxonomy organisms | (h); (e′) |

New Isambard steps are needed for (e′), (h) and (i). Operator time is the scarce resource, so these should be added
after `stage1` in `STEPS` (`scripts/run_on_isambard.sh`), not before.

## Known risks

- **Reasoning on may erase the planted gap**, as it erased SRH's EM persona (base's safety deliberation overriding
  behavior trained with an empty analysis channel). If the gap appears only with reasoning off, we have a *silent-gap*
  organism. That is still useful: it tests whether CoT-based eval-awareness measures (ours included) miss gaps that
  are not verbalized.
- **Trigger, not concept.** Covered by the held-out families. A fail there limits what the organism can validate.
- **Generalization of method scores.** A method that scores well on one SFT-built organism may not transfer. Later
  mitigation if results are promising: a second GT organism built differently (synthetic-document finetuning, or RL),
  and blinded scoring in the style of Marks et al. 2025, *Auditing Language Models for Hidden Objectives*.
- **Headline risk.** If methods recover the planted gap but no taxonomy organism shows a gap, the paper's claim
  becomes "these organisms metagame but do not behave differently under evaluation". Decide, before (e) is read, how
  that result would be reported.

## Open questions

- Which cue families, exactly, and who writes them (overlap with Rohit's and Srishti's eval prompts must be avoided,
  or their evals become trained-on cues)?
- Target planted gap size: large enough to detect, small enough to be realistic?
- Should the GT organism verbalize awareness in its CoT (one variant) or not (another)? Two variants double the cost.
- Is (i) in scope for the conference paper, or follow-up work?

## Implementation status (2026-10-22)

CPU work is built and tested. Nothing has run on a GPU. Decisions taken (Feifan, 2026-10-22):

- No monitoring or logging notices in either cue set: being watched is kept separate from being tested, so the
  "always observed" idea in FOLLOW_UPS.md §4 stays testable on the GT organism.
- Every GT training row is cued, half eval and half deployment, stratified by task.
- Claude drafted both cue sets. They are marked DRAFT for review.
- Probes (step i) are built now.

| Piece | Where | State |
|---|---|---|
| Framing templates, 8 eval + 8 deployment | `model-organisms/evals/data/eval_framing_templates.yaml` | Draft |
| GT cue families: 6 kinds, each with eval and deployment versions; `metadata` and `requester` held out | `model-organisms/data/gt_cue_families.yaml` | Draft |
| Framing grid tasks `framing_<tier>_<framing>`; the judge sees the unframed prompt | `evals/heldout_reward_hacking.py`, `scripts/run_pilot_evals.py` | Built |
| Minimal-pair check: deterministic rules in tests, plus a blind judge task run by the panel | `evals/framing_minimal_pairs.py`, `evals/judges.py` | Built; judge prompt not yet validated |
| GT eval tasks `gt_<train\|heldout>_<eval\|deployment>` | `evals/gt_gap.py` | Built |
| GT pair data: identical prompts; same hack count per task in both arms | `scripts/build_sft_datasets.py --gt-cues`, `data/STATS_gt.md` | Built: 1,368 rows per arm. Completion tokens are equal at seed 0, by chance; seeds 1 and 2 differ by under 0.7%, well inside the 15% bound |
| Training, merging and evaluating any SFT pair (`TREAT_ARM`/`CTRL_ARM`) | `scripts/train_sft.py`, `scripts/run_pilot_stage_on_gpu_node.sh` | Built; the SRH pair's names are unchanged |
| Readouts: the grid (row, column and vs-base contrasts) and the GT gap (with this doc's draft criteria) | `scripts/compare_framing_results.py`, `scripts/compare_gt_gap.py` | Built |
| Probes, trained on base and scored on the GT pair | `scripts/train_eval_awareness_probes.py` | Built |
| Isambard steps `gap_base`, `gt`, `probes`, all after `stage1` | `scripts/run_on_isambard.sh` | Built; tested against fakes |
| Contamination of both cue sets vs the two SDF corpora | `results/contamination_gap_cues_vs_*.json` | 0 of 64 cues match; 29 are shorter than 8 words, so they cannot match |

Still open:

- Overlap with Rohit's and Srishti's eval prompts. Their files are not in the repo, so it must be checked before
  `gt` trains.
- Training the SRH pair's recipe on 4 x GH200 (96 GB) has not been tried. `gt` runs `bringup` first, with a 90 GB
  memory bound.
- `gt` needs two nodes. Disk use also needs watching: two merged models take 468 GB, and they are deleted once
  their evals finish.
- Steering (method 3) is not built.
- Under the judge panel, an unparsed verdict counts as NO_HACK (or SAME) in the vote, and `verdict_parsed` comes
  from the first judge only (`aggregate_judge_panel.py`). This is pre-existing, but it affects these readouts too.
- Under the judge panel, the stage runner's format health check never runs. The new readouts therefore leave out
  format failures (no final channel, or cut off at max_tokens) and list their rates per cell.
- Nothing gates `gt` on the minimal-pair result. Feifan reads `framing_minimal_pairs_panel.json` before the grid
  is interpreted.
- In the control arm, which rows hack is balanced per (task, condition) but not per cue family. In the built
  data, the family-level deployment − eval differences are chance-sized (−8 to +10 pts).

## Notes for a reviewer

Things worth checking against the repo before building:

- Whether the held-out hacking prompts can take a framing preamble without changing what the judge sees as the metric
  (`evals/judges.py`).
- How new SFT arms fit `build_sft_datasets.py` / `train_sft.py` and the SRH data stats checks (`data/STATS*.md`).
- Whether the matched-length rule in `PROJECT_STATUS.md` is achievable when the treatment and control assign hacking
  to different rows.
- How a new step slots into `run_on_isambard.sh` and `pilot_orchestration_lib.sh` without changing earlier steps.
