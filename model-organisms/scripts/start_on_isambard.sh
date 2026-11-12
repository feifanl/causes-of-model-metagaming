#!/usr/bin/env bash
# Start (or resume) the whole Isambard run with one command, then follow its log.
#
#   bash model-organisms/scripts/start_on_isambard.sh <DATA_DIR>      # first time: where models and results go
#   bash model-organisms/scripts/start_on_isambard.sh                 # later: DATA_DIR from ~/.config/spar/env
#
# What it does, in order:
#   1. If a run is already queued or running: only follows its log (never pulls code under a live job).
#   2. Updates this checkout to the latest revamp/restructure (BRANCH=... to pick another branch).
#   3. Writes ~/.config/spar/env (mode 600): DATA_DIR and the HF tokens, read from ~/hftokens
#      (ISAMBARD_READ -> HF_TOKEN, ISAMBARD_WRITE -> HF_WRITE_TOKEN). Other lines already there (SDF_LR,
#      SBATCH_ACCOUNT, ...) are kept. Tokens are never printed.
#   4. Runs run_on_isambard.sh: preflight on the login node, then submits the 2-node Slurm job.
#   5. Follows the log: STATUS (one line per step) and the job's full output. Ctrl-C stops watching, not the job;
#      to watch again later run this script again, or: tail -F <DATA_DIR>/pilot_state/STATUS
#
# ~/hftokens on Isambard: copy it from your machine once (scp ~/hftokens isambard:~/ && chmod 600 ~/hftokens).
# Lines may be KEY=value, export KEY=value or KEY: value.

main() {
  set -uo pipefail
  local repo mo env_file tokens data branch job state out
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  mo="$repo/model-organisms"
  env_file="$HOME/.config/spar/env"
  tokens="${HF_TOKENS_FILE:-$HOME/hftokens}"
  branch="${BRANCH:-revamp/restructure}"

  # DATA_DIR: the argument, else the env file's.
  data="${1:-}"
  if [ -z "$data" ] && [ -f "$env_file" ]; then
    data="$(sed -n 's/^DATA_DIR=//p' "$env_file" | tail -1)"
  fi
  [ -n "$data" ] || { echo "usage: $0 <DATA_DIR>   (a new directory on project storage, ~1.5 TB)"; exit 2; }
  state="$data/pilot_state"

  # 1. A live run: follow it, change nothing.
  job="$(squeue -h -u "$USER" -n "${ISAMBARD_JOB_NAME:-eval-awareness}" -o %i 2>/dev/null | head -1)"
  if [ -n "$job" ]; then
    echo "A run is already queued or running (job $job): not touching the code, following its log."
    follow "$state" "$job"
  fi

  # 2. Code.
  echo "== code: $branch"
  git -C "$repo" fetch -q origin "$branch" || { echo "FAILED: git fetch origin $branch"; exit 1; }
  if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    echo "FAILED: $repo has local changes; commit or stash them first (git -C $repo status)"; exit 1
  fi
  git -C "$repo" checkout -q "$branch" 2>/dev/null || git -C "$repo" checkout -q -b "$branch" "origin/$branch" \
    || { echo "FAILED: git checkout $branch"; exit 1; }
  git -C "$repo" merge -q --ff-only "origin/$branch" || { echo "FAILED: $branch has diverged from origin"; exit 1; }
  echo "   at $(git -C "$repo" log --oneline -1)"

  # 3. Settings.
  echo "== settings: $env_file"
  write_env "$env_file" "$tokens" "$data" || exit 1

  # setup's venvs and built datasets live in the checkout, its marker in DATA_DIR: a new checkout with an old
  # DATA_DIR would skip setup and fail on the missing venvs. Then forget that setup finished, so it runs again
  # (downloads already in DATA_DIR are reused).
  mkdir -p "$state"
  if [ -f "$state/isambard_setup.done" ] && { [ ! -f "$repo/venv/.installed" ] || [ ! -f "$repo/venv-vllm/.installed" ] \
      || [ ! -d "$mo/data/processed" ]; }; then
    echo "== this checkout has no venvs or built data yet, but $data says setup is done: setup will run again"
    rm -f "$state/isambard_setup.done"
  fi

  # 4. Preflight and submit.
  echo "== preflight and submit"
  out="$state/start_on_isambard.out"
  bash "$mo/scripts/run_on_isambard.sh" all 2>&1 | tee "$out"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: see the lines above (also in $out)"; exit 1; }
  job="$(sed -n 's/^Submitted job \([0-9]*\).*/\1/p' "$out" | tail -1)"

  # 5. Logs.
  follow "$state" "$job"
}

# Read one token from the tokens file without echoing it.
token() {
  sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*[=:][[:space:]]*[\"']?([^\"'[:space:]]+).*/\\2/p" "$2" | tail -1
}

write_env() {
  local env_file=$1 tokens=$2 data=$3 read write tmp
  if [ -f "$tokens" ]; then
    read="$(token ISAMBARD_READ "$tokens")"
    write="$(token ISAMBARD_WRITE "$tokens")"
  else
    read=""; write=""
  fi
  # Keep tokens already in the env file if ~/hftokens is missing.
  [ -n "$read" ] || read="$( [ -f "$env_file" ] && sed -n 's/^HF_TOKEN=//p' "$env_file" | tail -1)"
  [ -n "$write" ] || write="$( [ -f "$env_file" ] && sed -n 's/^HF_WRITE_TOKEN=//p' "$env_file" | tail -1)"
  [ -n "$read" ] || { echo "FAILED: no ISAMBARD_READ in $tokens (copy it here: scp ~/hftokens isambard:~/)"; return 1; }
  [ -n "$write" ] || { echo "FAILED: no ISAMBARD_WRITE in $tokens"; return 1; }

  mkdir -p "$(dirname "$env_file")"
  tmp="$(mktemp "$env_file.XXXXXX")" && chmod 600 "$tmp"
  {
    # Everything already there except the keys written below (SDF_LR, SBATCH_*, NOTIFY_EMAIL, ... stay).
    [ -f "$env_file" ] && grep -vE '^(DATA_DIR|HF_TOKEN|HF_WRITE_TOKEN|HF_ARTIFACT_REPO|SMOKE)=' "$env_file"
    echo "DATA_DIR=$data"
    echo "HF_TOKEN=$read"
    echo "HF_WRITE_TOKEN=$write"
    echo "HF_ARTIFACT_REPO=${HF_ARTIFACT_REPO:-feifanliu/eval-awareness-model-organisms}"
  } > "$tmp"
  mv "$tmp" "$env_file" && chmod 600 "$env_file"
  grep -q '^SDF_LR=' "$env_file" || echo "SDF_LR=" >> "$env_file"
  echo "   DATA_DIR=$data, HF_TOKEN and HF_WRITE_TOKEN set (from $tokens), SDF_LR=$(sed -n 's/^SDF_LR=//p' "$env_file" | tail -1)"
}

# Follow STATUS and the job's Slurm output (the job's file appears once it starts).
follow() {
  local state=$1 job=${2:-} files
  files=("$state/STATUS")
  [ -n "$job" ] && files+=("$state/slurm-$job.out")
  echo
  echo "Following ${files[*]}  (Ctrl-C stops watching, not the job; queue: squeue -u $USER)"
  touch "$state/STATUS"
  exec tail -n 30 -F "${files[@]}"
}

# The whole script is parsed before it runs, so the git update in step 2 cannot change it mid-run.
main "$@"
