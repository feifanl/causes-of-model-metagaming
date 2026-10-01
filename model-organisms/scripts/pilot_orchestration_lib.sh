# Shared by run_all_pilot_stages_on_gpu_node.sh and run_cot_followup_on_gpu_node.sh (sourced, not run).
#
# Expects DEADLINE (epoch seconds), NVME, REPO, MO, RUN (stage runner), STATE, HF, and optionally
# HF_ARTIFACT_REPO / HF_WRITE_TOKEN from ~/.config/spar/env.

log() { echo "$(date -u +%FT%TZ) all $*" | tee -a "$STATE/STATUS"; }

check_deadline() {
  # A mistyped epoch would silently skip every stage; refuse anything not 1-24 h ahead.
  local now; now=$(date +%s)
  if [ "$DEADLINE" -lt $(( now + 3600 )) ] || [ "$DEADLINE" -gt $(( now + 86400 )) ]; then
    log "ERROR: deadline $(date -u -d "@$DEADLINE" +%FT%TZ) is not 1-24 h from now; use \$(date -d '<time>' +%s)"
    exit 1
  fi
}

upload() {  # upload <path on node> <path in repo>; never fatal
  [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_WRITE_TOKEN:-}" ] || { log "upload skipped (HF_ARTIFACT_REPO/HF_WRITE_TOKEN unset)"; return 0; }
  [ -e "$1" ] || return 0
  # PEFT's auto README names the local base path as base_model, which the Hub rejects.
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 2h "$HF" upload "$HF_ARTIFACT_REPO" "$1" "$2" --repo-type model --private \
      --exclude README.md --commit-message "pilot: $2" >>"$STATE/upload.log" 2>&1 \
    && log "uploaded $2" || log "UPLOAD FAILED: $2 (see upload.log)"
}

upload_results() {  # upload_results <path in repo>
  local tmp="$NVME/upload_staging"
  rm -rf "$tmp"; mkdir -p "$tmp/outputs" "$tmp/data"
  cp -r "$MO/results" "$tmp/" 2>/dev/null
  for d in "$MO"/outputs/*/; do
    n=$(basename "$d"); mkdir -p "$tmp/outputs/$n"
    cp "$d"/run_summary.json "$d"/step_log.jsonl "$d"/adapter_config.json "$tmp/outputs/$n/" 2>/dev/null
  done
  cp "$MO"/outputs/nll_*.json "$tmp/outputs/" 2>/dev/null
  cp "$MO"/data/reasoning_examples.jsonl "$MO"/data/reasoning_examples.meta.json "$MO"/data/STATS_cotreg.md \
     "$tmp/data/" 2>/dev/null
  cp -r "$STATE" "$tmp/pilot_state"
  for m in "$NVME"/merged/*/; do [ -f "$m/provenance.json" ] && cp "$m/provenance.json" "$tmp/outputs/merged_$(basename "$m")_provenance.json"; done
  upload "$tmp" "${1:-run}"
}

fits() {  # fits <hours> <what>: is there time left for a stage's worst case?
  local left=$(( (DEADLINE - $(date +%s)) / 60 ))
  if [ "$left" -lt $(( $1 * 60 )) ]; then log "SKIP $2: needs ${1}h, ${left} min left before the time cap"; return 1; fi
}

stage() {  # stage <worst-case hours> <stage> [VAR=value ...]: run one stage with variant settings, then upload
  local hours=$1 name=$2; shift 2
  fits "$hours" "$name $*" || return 1
  env "$@" bash "$RUN" "$name"; local rc=$?
  upload_results "${UPLOAD_PREFIX:-run}"
  return $rc
}
