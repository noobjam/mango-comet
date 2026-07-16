#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  server/vm_field_stories_v1.sh run [ENV_FILE]
  server/vm_field_stories_v1.sh status [ENV_FILE]
  server/vm_field_stories_v1.sh logs [ENV_FILE]

run pulls origin/main, restarts this wrapper from the updated checkout, and
launches one detached deterministic field-story V1 release. status reports the
latest run and artifact counts. logs follows its durable log.

The default ENV_FILE is REPO/.env.vm. It is gitignored.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

load_env() {
  local file=$1
  [[ -f "$file" ]] || fail "environment file does not exist: $file"
  set -a
  # shellcheck disable=SC1090
  . "$file"
  set +a
}

require_value() {
  local name=$1
  [[ -n "${!name:-}" ]] || fail "missing $name in .env.vm"
}

require_file() {
  local name=$1
  require_value "$name"
  [[ -f "${!name}" ]] || fail "$name does not exist: ${!name}"
}

require_dir() {
  local name=$1
  require_value "$name"
  [[ -d "${!name}" ]] || fail "$name does not exist: ${!name}"
}

latest_tag() {
  local pointer="$ROOT/logs/latest_field_stories_v1_tag.txt"
  [[ -s "$pointer" ]] || fail "no field-story V1 run has been launched: $pointer"
  tr -d '\r\n' < "$pointer"
}

run_base() {
  printf '%s/logs/field_stories_v1_%s' "$ROOT" "$1"
}

output_dir() {
  printf '%s/releases/field_stories_v1_%s' "$ROOT" "$1"
}

preflight() {
  for name in REPO PYTHON ROOT V4_EVIDENCE_DIR; do
    require_value "$name"
  done
  require_dir REPO
  require_dir ROOT
  require_dir V4_EVIDENCE_DIR
  require_file PYTHON
  [[ -x "$PYTHON" ]] || fail "PYTHON is not executable: $PYTHON"
  [[ -f "$REPO/server/run_field_stories_v1.py" ]] || \
    fail "field-story V1 runner is missing from REPO"
  command -v flock >/dev/null || fail "flock is not available"
  mkdir -p "$ROOT/duckdb_tmp" "$ROOT/logs" "$ROOT/releases"
}

launch_release() {
  local env_file=$1
  preflight
  exec 8> "$ROOT/logs/field_stories_v1_launch.lock"
  flock -n 8 || fail "another field-story launch is being prepared"

  if [[ -s "$ROOT/logs/latest_field_stories_v1_tag.txt" ]]; then
    local prior_tag prior_base prior_pid
    prior_tag=$(latest_tag)
    prior_base=$(run_base "$prior_tag")
    prior_pid=$(tr -dc '0-9' < "${prior_base}.pid" 2>/dev/null || true)
    if [[ ! -s "${prior_base}.status" && -n "$prior_pid" ]] \
        && kill -0 "$prior_pid" 2>/dev/null; then
      fail "field-story run $prior_tag is still running as PID $prior_pid"
    fi
  fi

  local tag base release pid pointer_tmp
  tag=$(date -u +%Y%m%dT%H%M%SZ)
  base=$(run_base "$tag")
  release=$(output_dir "$tag")
  nohup "$0" __run "$env_file" "$tag" > "${base}.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "${base}.pid"

  pointer_tmp="$ROOT/logs/latest_field_stories_v1_tag.txt.tmp.$$"
  printf '%s\n' "$tag" > "$pointer_tmp"
  mv "$pointer_tmp" "$ROOT/logs/latest_field_stories_v1_tag.txt"
  pointer_tmp="$ROOT/logs/latest_field_stories_v1.txt.tmp.$$"
  printf '%s\n' "$release" > "$pointer_tmp"
  mv "$pointer_tmp" "$ROOT/logs/latest_field_stories_v1.txt"

  printf 'Field-story V1 run started.\nTAG=%s\nPID=%s\nLOG=%s.log\nOUTPUT=%s\n' \
    "$tag" "$pid" "$base" "$release"
}

run_release() {
  local tag=$1
  local base release
  base=$(run_base "$tag")
  release=$(output_dir "$tag")

  finish_status() {
    local code=$?
    local temporary="${base}.status.tmp.$$"
    printf '%s\n' "$code" > "$temporary"
    mv "$temporary" "${base}.status"
  }
  trap finish_status EXIT

  exec 9> "$ROOT/logs/field_stories_v1.lock"
  flock -n 9 || fail "another field-story V1 release is running"
  preflight
  cd "$REPO"
  "$PYTHON" server/run_field_stories_v1.py \
    --evidence-dir "$V4_EVIDENCE_DIR" \
    --output-dir "$release" \
    --partitions "${V4_REPLAY_PARTITIONS:-64}" \
    --threads "${DUCKDB_THREADS:-8}" \
    --memory-limit "${DUCKDB_MEMORY_LIMIT:-32GB}" \
    --temp-dir "$ROOT/duckdb_tmp"
  [[ -f "$release/manifest.json" ]] || \
    fail "field-story release completed without a manifest: $release"
}

show_status() {
  require_value ROOT
  require_value PYTHON
  [[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
  [[ -x "$PYTHON" ]] || fail "PYTHON is not executable: $PYTHON"

  local tag base release pid status
  tag=$(latest_tag)
  base=$(run_base "$tag")
  release=$(output_dir "$tag")
  pid=$(tr -dc '0-9' < "${base}.pid" 2>/dev/null || true)
  if [[ -s "${base}.status" ]]; then
    status=$(tr -d '\r\n' < "${base}.status")
  elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    status=RUNNING
  else
    status=DEAD_WITHOUT_STATUS
  fi
  printf 'TAG=%s\nSTATUS=%s\nPID=%s\nLOG=%s.log\nOUTPUT=%s\n' \
    "$tag" "$status" "$pid" "$base" "$release"
  tail -n 20 "${base}.log" 2>/dev/null || true
  if [[ -f "$release/manifest.json" ]]; then
    "$PYTHON" -c \
      'import json,sys; m=json.load(open(sys.argv[1])); [print("{}_ROWS={}".format(k.upper(), v["row_count"])) for k,v in sorted(m["artifacts"].items())]' \
      "$release/manifest.json"
  fi
}

command=${1:-help}
if [[ "$command" == help || "$command" == --help || "$command" == -h ]]; then
  usage
  exit 0
fi

env_file=${2:-"$DEFAULT_REPO/.env.vm"}
env_file=$(cd -- "$(dirname -- "$env_file")" 2>/dev/null && pwd)/$(basename -- "$env_file")
load_env "$env_file"

case "$command" in
  run)
    require_dir REPO
    command -v git >/dev/null || fail "git is not available"
    cd "$REPO"
    git pull --ff-only origin main
    exec "$REPO/server/vm_field_stories_v1.sh" __launch "$env_file"
    ;;
  __launch)
    launch_release "$env_file"
    ;;
  __run)
    [[ $# -eq 3 ]] || fail "invalid internal field-story invocation"
    run_release "$3"
    ;;
  status)
    show_status
    ;;
  logs)
    require_value ROOT
    [[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
    tag=$(latest_tag)
    tail -f "$(run_base "$tag").log"
    ;;
  *)
    usage >&2
    fail "unknown command: $command"
    ;;
esac
