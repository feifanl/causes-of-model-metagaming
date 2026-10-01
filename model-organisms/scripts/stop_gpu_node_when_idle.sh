#!/usr/bin/env bash
# Stop (Nebius) or terminate (PrimeIntellect) this GPU node through the provider's API
# when pilot work has stopped or a time cap passes. Runs on the node, in tmux.
#
#   bash model-organisms/scripts/stop_gpu_node_when_idle.sh prime  <pod-id>      <max-hours> [idle-minutes]
#   bash model-organisms/scripts/stop_gpu_node_when_idle.sh nebius <instance-id> <max-hours> [idle-minutes]
#
# Why the API: shutting down from inside the node does not end billing. Nebius treats
# a guest shutdown as a crash, restarts the VM and keeps charging; a Prime pod is
# billed until it is terminated.
#
#   prime : DELETE /api/v1/pods/<id>. Terminating DELETES THE POD'S DISK (attached
#           persistent disks survive). Results must already be synced off the node.
#   nebius: `nebius compute instance stop`. Disks are kept (and keep billing).
#
# Busy means any of: a pilot process is running (stage runner, training, merge, evals,
# setup); some GPU is above 5% utilization; or ~/KEEP_ALIVE exists (a human is
# debugging). An idle vLLM server alone counts as idle. The node is stopped after
# <idle-minutes> (default 45) of continuous idleness, or <max-hours> after this script
# started, whichever comes first. ~/KEEP_ALIVE does not override the time cap.
#
# Credentials: PRIME_API_KEY from ~/.config/spar/env (prime), or a `nebius` CLI profile
# $NEBIUS_PROFILE, default 'vm-stopper' (nebius). The API is checked at startup and on
# every loop. DRY_RUN=1 logs the stop instead of doing it.
set -uo pipefail

CLOUD="${1:?usage: stop_gpu_node_when_idle.sh <prime|nebius> <node-id> <max-hours> [idle-minutes]}"
NODE_ID="${2:?node id required}"
MAX_HOURS="${3:?max-hours required}"
IDLE_MINUTES="${4:-45}"
PRIME_API="https://api.primeintellect.ai/api/v1/pods"
PROFILE="${NEBIUS_PROFILE:-vm-stopper}"
NEBIUS="${NEBIUS:-$HOME/.nebius/bin/nebius}"
LOG="${NVME:-/data}/pilot_state/watchdog.log"
# Matches the processes' command lines, not log viewers (e.g. `tail train_sft.log` has no python).
PILOT='(python|torchrun)[^ ]* .*(train_sft|train_sdf|merge_lora|dequantize_base|run_pilot_evals|score_heldout|measure_expert|track_openrouter|download_data|build_s|pytest)|run_pilot_stage_on_gpu_node|setup_gpu_node|pip install|hf download'

# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

case "$CLOUD" in
  prime)
    [ -n "${PRIME_API_KEY:-}" ] || { log "ERROR: PRIME_API_KEY not set; watchdog NOT armed"; exit 1; }
    api_ok()   { curl -sf -m 30 -H "Authorization: Bearer $PRIME_API_KEY" "$PRIME_API/$NODE_ID" >/dev/null; }
    api_stop() { curl -sf -m 60 -X DELETE -H "Authorization: Bearer $PRIME_API_KEY" "$PRIME_API/$NODE_ID"; } ;;
  nebius)
    api_ok()   { "$NEBIUS" --profile "$PROFILE" compute instance get --id "$NODE_ID" >/dev/null 2>&1; }
    api_stop() { "$NEBIUS" --profile "$PROFILE" compute instance stop --id "$NODE_ID"; } ;;
  *) echo "unknown cloud '$CLOUD' (prime|nebius)"; exit 1 ;;
esac

stop_node() {
  log "STOPPING NODE ($CLOUD): $1"
  if [ "${DRY_RUN:-0}" = 1 ]; then log "DRY_RUN: would stop $CLOUD node $NODE_ID"; exit 0; fi
  for attempt in 1 2 3 4 5; do
    if api_stop >>"$LOG" 2>&1; then log "stop requested (attempt $attempt)"; exit 0; fi
    log "stop failed (attempt $attempt); retrying in 60s"; sleep 60
  done
  log "ERROR: could not stop the node through the API. Stop it from the console."
  exit 1
}

# Without these, every check would read as idle and the node could be stopped mid-load.
for tool in pgrep nvidia-smi curl; do
  command -v "$tool" >/dev/null || { log "ERROR: '$tool' not found (apt-get install procps curl); watchdog NOT armed"; exit 1; }
done
api_ok || { log "ERROR: $CLOUD API not reachable for $NODE_ID; watchdog NOT armed"; exit 1; }
log "armed: $CLOUD node $NODE_ID, cap ${MAX_HOURS}h, idle ${IDLE_MINUTES} min"

start=$(date +%s)
idle_since=""
api_failures=0
while true; do
  now=$(date +%s)
  if [ $(( now - start )) -ge $(( MAX_HOURS * 3600 )) ]; then stop_node "time cap ${MAX_HOURS}h reached"; fi

  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  reason=""
  if pgrep -f -- "$PILOT" >/dev/null; then reason="pilot process"
  elif [ "${util:-0}" -gt 5 ]; then reason="GPU util ${util}%"
  elif [ -e "$HOME/KEEP_ALIVE" ]; then reason="KEEP_ALIVE"
  fi

  if [ -n "$reason" ]; then
    [ -n "$idle_since" ] && log "busy again ($reason)"
    idle_since=""
  else
    [ -z "$idle_since" ] && { idle_since=$now; log "idle (no pilot process, max GPU util ${util:-?}%)"; }
    if [ $(( now - idle_since )) -ge $(( IDLE_MINUTES * 60 )) ]; then stop_node "idle for ${IDLE_MINUTES} min"; fi
  fi

  if api_ok; then api_failures=0; else
    api_failures=$((api_failures + 1)); log "WARNING: $CLOUD API check failed ($api_failures in a row)"
  fi
  sleep 60
done
