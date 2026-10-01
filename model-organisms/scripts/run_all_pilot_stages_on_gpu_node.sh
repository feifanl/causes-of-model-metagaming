#!/usr/bin/env bash
# Run every Phase 1 pilot stage on the GPU node, with no laptop needed (docs/GPU-HANDOFF.md).
#
#   NVME=/local HF_ARTIFACT_REPO=<user>/<private repo> bash model-organisms/scripts/run_all_pilot_stages_on_gpu_node.sh <deadline-epoch>
#
# Order: bf16 + bringup (in parallel), base_eval, train, merge, arm_eval, compare,
# routing (only if the persona verdict is NO), sdf. A failed stage does not stop the
# script: stages that depend on it fail fast in run_pilot_stage_on_gpu_node.sh, and
# independent ones (e.g. sdf after a failed eval) still run.
#
# After every stage, results, run summaries and logs are uploaded to the private HF
# repo $HF_ARTIFACT_REPO (HF_WRITE_TOKEN in ~/.config/spar/env), and adapters after
# the stage that made them. The pod's disk is deleted when the watchdog terminates it,
# so the HF repo is the only copy if the laptop is offline.
#
# <deadline-epoch> is when the watchdog's time cap fires (date +%s). A stage is not
# started unless its worst-case time fits before it; skipped stages are logged.
set -uo pipefail

DEADLINE="${1:?usage: run_all_pilot_stages_on_gpu_node.sh <deadline-epoch>}"
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
RUN="$MO/scripts/run_pilot_stage_on_gpu_node.sh"
STATE="$NVME/pilot_state"
HF="$REPO/venv/bin/hf"
export NVME
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"

log() { echo "$(date -u +%FT%TZ) all $*" | tee -a "$STATE/STATUS"; }

upload() {  # upload <path on node> <path in repo>; never fatal
  [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_WRITE_TOKEN:-}" ] || { log "upload skipped (HF_ARTIFACT_REPO/HF_WRITE_TOKEN unset)"; return 0; }
  [ -e "$1" ] || return 0
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 2h "$HF" upload "$HF_ARTIFACT_REPO" "$1" "$2" --repo-type model --private \
      --commit-message "pilot: $2" >>"$STATE/upload.log" 2>&1 \
    && log "uploaded $2" || log "UPLOAD FAILED: $2 (see upload.log)"
}

upload_results() {
  local tmp="$NVME/upload_staging"
  rm -rf "$tmp"; mkdir -p "$tmp/outputs"
  cp -r "$MO/results" "$tmp/" 2>/dev/null
  for d in "$MO"/outputs/*/; do
    n=$(basename "$d"); mkdir -p "$tmp/outputs/$n"
    cp "$d"/run_summary.json "$d"/step_log.jsonl "$d"/adapter_config.json "$tmp/outputs/$n/" 2>/dev/null
  done
  cp "$MO"/outputs/nll_*.json "$tmp/outputs/" 2>/dev/null
  cp -r "$STATE" "$tmp/pilot_state"
  for m in "$NVME"/merged/*/; do [ -f "$m/provenance.json" ] && cp "$m/provenance.json" "$tmp/outputs/merged_$(basename "$m")_provenance.json"; done
  upload "$tmp" "run"
}

fits() {  # fits <hours>: is there time left for a stage's worst case?
  local left=$(( (DEADLINE - $(date +%s)) / 60 ))
  if [ "$left" -lt $(( $1 * 60 )) ]; then log "SKIP $2: needs ${1}h, ${left} min left before the time cap"; return 1; fi
}

stage() {  # stage <name> <worst-case hours>
  fits "$2" "$1" || return 1
  bash "$RUN" "$1"; local rc=$?
  upload_results
  return $rc
}

log "start (deadline $(date -u -d "@$DEADLINE" +%FT%TZ))"

# Slot 1: dequantize on half B while bring-up runs on half A.
if fits 2 "bf16+bringup"; then
  bash "$RUN" bf16 & b=$!
  bash "$RUN" bringup & a=$!
  wait $b; wait $a
  upload_results
fi

stage base_eval 3
stage train 5 && { upload "$MO/outputs/srh_mixed_seed0" "adapters/srh_mixed_seed0"; upload "$MO/outputs/control_seed0" "adapters/control_seed0"; }
stage merge 2
stage arm_eval 3
stage compare 1
if grep -q "Persona: NO" "$MO/results/pilot_comparison.md" 2>/dev/null; then stage routing 2; fi
stage sdf 3 && upload "$MO/outputs/sdf_slice" "adapters/sdf_slice"

upload_results
log "finished; nothing else will start. The watchdog terminates the pod after its idle limit."
