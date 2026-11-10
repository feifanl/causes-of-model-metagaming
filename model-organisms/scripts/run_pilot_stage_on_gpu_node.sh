#!/usr/bin/env bash
# Run one stage of the Phase 1 pilot on the 8xH200 node (docs/GPU-RUNBOOK.md, docs/GPU-HANDOFF.md).
#
#   bash model-organisms/scripts/run_pilot_stage_on_gpu_node.sh <stage>     # from the repo root, inside tmux
#   bash model-organisms/scripts/run_pilot_stage_on_gpu_node.sh status
#
# Stages (half A = GPUs 0-3, half B = GPUs 4-7):
#   bf16            half B: dequantize the MXFP4 base to $NVME/gpt-oss-120b-bf16
#   bringup         half A: 10 SFT steps (PLAN 1.1): memory, tokens/sec, finite loss
#   base_eval       half B: serve the bf16 base, run the pilot evals -> results/base_own<EVAL_TAG>.json
#   train           both halves: SRH and control SFT at the same time (PLAN 1.2)
#   fetch_adapters  CPU: download trained seed-0 adapters from $HF_ARTIFACT_REPO instead of training
#   merge           both halves: merge each adapter into bf16 on GPU, NLL-checked
#   arm_eval        both halves: serve both merged models, run the pilot evals on both
#   compare         CPU: PLAN's pass/fail rules -> results/pilot_comparison<VARIANT><EVAL_TAG>.md
#   gen_reasoning   half B: base model's own reasoning on neutral prompts, then the CoT-regularized
#                   SFT data in data/processed_cotreg (DECISIONS 'CoT format regularizer')
#   sdf             all 8 GPUs: held-out NLL, 120-step FSDP2 SDF slice, held-out NLL again (PLAN 1.4)
#   routing         half A: expert-routing overlap (only needed if the persona verdict is NO)
#   sdf_train       all 8 GPUs: one full SDF run (FSDP2) -> outputs/sdf_<SDF_ARM>_seed<SEED><VARIANT>/, then
#                   held-out NLL of the base, every saved checkpoint and the final adapter
#   sdf_eval        one half (HALF): merge one SDF adapter (CKPT), serve it, run the evals
#                   -> results/sdf_<run>_<CKPT><EVAL_TAG>.json; the merged copy (234 GB) is deleted on exit
#   rl_adapter      CPU: download one RL organism's LoRA (ORGANISM), prepared for unmerged serving -> outputs/<ORGANISM>
#                   (a bf16 merge erases RL deltas: train/prepare_rl_adapter_for_serving.py)
#   rl_check        all 8 GPUs: score an exact fp32 merge, then serve bf16 base + LoRA on half A and check the
#                   served LoRA against it -> results/rl_lora_check_<ORGANISM>.json (rl_eval requires it)
#   rl_eval         one half (HALF): serve bf16 base + an RL organism's LoRA, run EVAL_TASKS
#                   -> results/<ORGANISM><EVAL_TAG>.json
#   gt_data         CPU: the ground-truth gap pair's SFT data (BEHAVIORAL_GAP_PLAN.md Part 2) in data/processed_gt,
#                   checked against the committed data/STATS_gt.md (CoT-regularizer rows from the SRH run's HF upload)
#   gap_compare     CPU: the framing grid of base (and of the TREAT_ARM/CTRL_ARM pair, if not SRH), plus the GT gap,
#                   for one EVAL_TAG -> results/framing_comparison<PAIR><VARIANT><EVAL_TAG>.md, gt_gap_comparison*.md
#   probes          both halves: eval-awareness probes trained on base (results/probes_base.pt), then scored on base and
#                   on both arms of the TREAT_ARM/CTRL_ARM pair, adapters unmerged -> results/probe_scores_<model>.json
#
# Variants, through the environment (defaults reproduce the pilot):
#   VARIANT     suffix for adapters, merged models and tags, e.g. _cotreg -> outputs/srh_mixed_seed0_cotreg
#   SFT_DATA_DIR    SFT data for train and merge checks (default data/processed)
#   EVAL_FLAGS  run_pilot_evals.py flags (default --no-reasoning; DECISIONS 'Eval prompt format')
#   EVAL_TAG    suffix for eval result tags, e.g. _reasoning_on -> results/srh_mixed_seed0_cotreg_reasoning_on.json
#   EVAL_TASKS  run_pilot_evals.py --tasks for base_eval, arm_eval and rl_eval (default em,hacking,mmlu)
#   ORGANISM    rl_adapter / rl_check / rl_eval: a name from train/download_rl_organism_adapter.py (e.g. aisi_hack)
#   TREAT_ARM, CTRL_ARM  train / fetch_adapters / merge / arm_eval: the SFT pair (default srh_mixed, control; the
#               ground-truth gap pair is gt_conditional, gt_unconditional). Another pair adds _<TREAT_ARM> to the
#               .done names (train_gt_conditional_cotreg.done), so it never reuses the SRH pair's stages.
# The .done file carries the same suffixes (train_cotreg.done, arm_eval_cotreg_reasoning_on.done), so
# each variant runs once. Compare only results with the same EVAL_FLAGS (compare_pilot_results checks).
#
# SDF stages (docs/SDF_NOTES.md; one run is named <SDF_ARM>_seed<SEED><VARIANT>):
#   SDF_ARM          treatment (AISI reward-hacking corpus) or control (unrelated-facts corpus)
#   SEED             training seed (default 0)
#   SDF_DATA_DIR     dir with sdf_train.jsonl + sdf_heldout.jsonl (default data/processed for treatment,
#                    data/processed_sdf_control for control; data/processed_notag for the no-tag corpus)
#   SDF_TRAIN_FLAGS  extra train_sdf.py flags: '--save-at-epochs 0.5,1,1.5', '--stop-at-epoch 0.5'
#   CKPT             sdf_eval: 'final' (default) or a saved checkpoint, e.g. epoch0.5
#   HALF             sdf_eval: A (GPUs 0-3, port 8000, default) or B (GPUs 4-7, port 8001); two sdf_eval
#                    stages can run at once, one per half (their judge calls take turns, see stage_sdf_eval)
#   SDF_EVAL_TASKS   run_pilot_evals.py --tasks for sdf_eval (default em,hacking,mmlu)
#   KEEP_MERGED=1    keep the merged model after sdf_eval
#
# Each stage refuses to start if its prerequisites are missing or the GPUs it needs are
# busy, logs to $NVME/pilot_state/<name>.log, runs its checks, and on success writes
# <name>.done. A finished stage is skipped on rerun; delete its .done file to redo it.
# Every step has a timeout, and vLLM servers are killed when a stage exits.
set -euo pipefail

STAGE="${1:?usage: run_pilot_stage_on_gpu_node.sh <stage>|status}"
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
export PYTHONPATH="$MO${PYTHONPATH:+:$PYTHONPATH}"  # scripts run as modules: python -m scripts.<folder>.<name>
PY="${PY:-$REPO/venv/bin/python}"
VLLM="${VLLM:-$REPO/venv-vllm/bin/vllm}"
SGLANG_PY="${SGLANG_PY:-$REPO/venv-sglang/bin/python}"   # RL organisms whose LoRA vLLM does not reproduce
TORCHRUN="${TORCHRUN:-$REPO/venv/bin/torchrun}"
BF16="$NVME/gpt-oss-120b-bf16"
STATE="$NVME/pilot_state"
SPEND_CAP="${SPEND_CAP:-20}"   # USD, whole ledger (results/api_spend.jsonl), not per run
VARIANT="${VARIANT:-}"
SFT_DATA_DIR="${SFT_DATA_DIR:-data/processed}"
# Same for every compared model: SFT'd models break the analysis->final format when left to reason,
# so all models answer straight in the final channel (DECISIONS 'Eval prompt format').
EVAL_FLAGS="${EVAL_FLAGS---no-reasoning}"   # set but empty = reasoning on, no flags
EVAL_TAG="${EVAL_TAG:-}"
EVAL_TASKS="${EVAL_TASKS:-em,hacking,mmlu}"   # run_pilot_evals.py's own default
ORGANISM="${ORGANISM:-}"
TREAT_ARM="${TREAT_ARM:-srh_mixed}"
CTRL_ARM="${CTRL_ARM:-control}"
PAIR=""; [ "$TREAT_ARM" = srh_mixed ] || PAIR="_$TREAT_ARM"
# GPU layout: two halves for side-by-side serving, all GPUs for training. Defaults fit one 8-GPU node; a node
# with another GPU count sets HALF_A, HALF_B (and ALL_GPUS if it differs from both halves together).
# Two nodes in one Slurm job (Isambard: 4 GPUs each): HOST_A and HOST_B name each half's node. Work for a half
# on another node runs there through srun (servers listen on all interfaces), and SDF training spans both nodes.
HALF_A="${HALF_A:-0,1,2,3}"
HALF_B="${HALF_B:-4,5,6,7}"
ALL_GPUS="${ALL_GPUS:-$HALF_A,$HALF_B}"
HOST_A="${HOST_A:-}"
HOST_B="${HOST_B:-}"
n_gpus() { echo "$1" | tr ',' '\n' | grep -c .; }
SDF_ARM="${SDF_ARM:-treatment}"
SEED="${SEED:-0}"
case "$SDF_ARM" in
  treatment) SDF_DATA_DIR="${SDF_DATA_DIR:-data/processed}" ;;
  control)   SDF_DATA_DIR="${SDF_DATA_DIR:-data/processed_sdf_control}" ;;
  *)         echo "SDF_ARM must be treatment or control, not '$SDF_ARM'"; exit 1 ;;
esac
SDF_RUN="${SDF_ARM}_seed${SEED}${VARIANT}"
SDF_TRAIN_FLAGS="${SDF_TRAIN_FLAGS:-}"
SDF_TIMEOUT="${SDF_TIMEOUT:-5h}"   # a 2-epoch run is ~1.75 h at the pilot's 17.5k tok/s
CKPT="${CKPT:-final}"
HALF="${HALF:-A}"
SDF_EVAL_TASKS="${SDF_EVAL_TASKS:-em,hacking,mmlu}"
KEEP_MERGED="${KEEP_MERGED:-0}"
MIN_ADAPTER_BYTES="${MIN_ADAPTER_BYTES:-1000000000}"   # r=64 expert LoRA adapters are ~17 GB
BRINGUP_MAX_MEM_GB="${BRINGUP_MAX_MEM_GB:-130}"   # per GPU; 141 GB on H200 (Isambard's GH200: 96 GB)

case "$STAGE" in
  train|merge|fetch_adapters) NAME="$STAGE$PAIR$VARIANT" ;;
  arm_eval|compare|gap_compare) NAME="$STAGE$PAIR$VARIANT$EVAL_TAG" ;;
  probes)                     NAME="$STAGE$PAIR$VARIANT" ;;
  base_eval)                  NAME="$STAGE$EVAL_TAG" ;;
  sdf_train)                  NAME="sdf_train_$SDF_RUN" ;;
  sdf_eval)                   NAME="sdf_eval_${SDF_RUN}_$CKPT$EVAL_TAG" ;;
  rl_adapter|rl_check)        NAME="${STAGE}_$ORGANISM" ;;
  rl_eval)                    NAME="rl_eval_$ORGANISM$EVAL_TAG" ;;
  *)                          NAME="$STAGE" ;;
esac

export HF_HOME="$NVME/hf"
# Secrets (OPENROUTER_API_KEY, HF_TOKEN, HF_WRITE_TOKEN) live outside the repo, readable only by the user.
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"
cd "$MO"

# Helpers, then each organism family's stages (scripts/stages/). Definitions only: nothing runs until the
# dispatch below.
for part in common base sft gap sdf rl; do
  # shellcheck disable=SC1090
  . "$MO/scripts/stages/$part.sh"
done

# --------------------------------------------------------------------------- #

if [ "$STAGE" = status ]; then
  ls "$STATE"/*.done 2>/dev/null | xargs -rn1 basename; tail -n 20 "$STATE/STATUS" 2>/dev/null; df -h "$NVME" | tail -1
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  exit 0
fi
declare -F "stage_$STAGE" >/dev/null || die "unknown stage"
if [ -f "$STATE/$NAME.done" ]; then log "already done (delete $STATE/$NAME.done to rerun)"; exit 0; fi
case "$STAGE" in
  sdf_train) log "start (SDF_DATA_DIR=$SDF_DATA_DIR SDF_TRAIN_FLAGS='$SDF_TRAIN_FLAGS')" ;;
  sdf_eval)  log "start (CKPT=$CKPT HALF=$HALF SDF_EVAL_TASKS=$SDF_EVAL_TASKS EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')" ;;
  rl_adapter|rl_check|rl_eval) log "start (ORGANISM=$ORGANISM HALF=$HALF EVAL_TASKS=$EVAL_TASKS EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')" ;;
  *)         log "start (VARIANT='$VARIANT' SFT_DATA_DIR=$SFT_DATA_DIR EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')" ;;
esac
start=$(date +%s)
set +e  # a failing stage must reach the FAILED line below, not exit here
( set -e; trap cleanup EXIT; "stage_$STAGE" ) 2>&1 | tee -a "$STATE/$NAME.log"
status=${PIPESTATUS[0]}
set -e
[ "$status" -eq 0 ] || die "exit $status after $(( ($(date +%s) - start) / 60 )) min; see $STATE/$NAME.log"
touch "$STATE/$NAME.done"
log "done in $(( ($(date +%s) - start) / 60 )) min"
