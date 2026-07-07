#!/usr/bin/env bash
set -Eeuo pipefail

REQUIRED_COMMIT=59b2dd82956062d7db712b51e880011bd93dd9ee
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  server/vm_story_pipeline.sh launch [ENV_FILE]
  server/vm_story_pipeline.sh status [ENV_FILE]
  server/vm_story_pipeline.sh logs [ENV_FILE]
  server/vm_story_pipeline.sh check-node [ENV_FILE]
  server/vm_story_pipeline.sh check-v3 [ENV_FILE]
  server/vm_story_pipeline.sh stop-server [ENV_FILE]

launch starts one detached, logged pipeline that:
  1. validates the fixed VM inputs and reuses or builds the matching V3 story spine;
  2. builds and validates a fresh immutable V4/2 evidence/viewer release;
  3. writes server/.env, starts the map server, and waits for health;
  4. runs the cold/warm/concurrent timeline benchmark;
  5. optionally runs GPU motif discovery and stops at the mandatory review gate.

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

json_value() {
  "$PYTHON" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); [value:=value[key] for key in sys.argv[2].split(".")]; print(value)' \
    "$1" "$2"
}

latest_tag() {
  local pointer="$ROOT/logs/latest_vm_story_pipeline.txt"
  [[ -s "$pointer" ]] || fail "no VM story pipeline has been launched: $pointer"
  tr -d '\r\n' < "$pointer"
}

pipeline_base() {
  printf '%s/logs/vm_story_pipeline_%s' "$ROOT" "$1"
}

phase() {
  local value=$1
  local temporary="${PIPELINE_BASE}.phase.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "${PIPELINE_BASE}.phase"
  printf '%s INFO phase=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$value"
}

stop_managed_server() {
  local pid_file="$ROOT/logs/latest_incident_v4_server.pid"
  [[ -s "$pid_file" ]] || return 0
  local pid command_line
  pid=$(tr -dc '0-9' < "$pid_file")
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$command_line" == *"story_map_server.py"* ]] || \
    fail "PID $pid in $pid_file is not the managed story-map server"
  printf '%s INFO stopping_previous_server pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pid"
  kill "$pid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  fail "managed server PID $pid did not stop within 30 seconds"
}

preflight() {
  for name in \
    REPO PYTHON NODE ROOT GEN ECHO FULL_2025 FULL_2026 ACQ_2026 \
    BUILD_V3_IF_MISSING V3_BASELINE_THROUGH \
    AVAILABILITY_MODE V4_FIRST_RELEASE REPLACE_MANAGED_SERVER \
    DUCKDB_THREADS DUCKDB_MEMORY_LIMIT HEARTBEAT_SECONDS \
    MAP_HOST MAP_PORT MAP_LOG_LEVEL MAP_RASTER_TILES MAP_RASTER_ATTRIBUTION \
    MAP_DEFAULT_FEATURE_LIMIT MAP_MAX_FEATURE_LIMIT MAP_CACHE_SECONDS \
    MAP_CACHE_ENTRIES MAP_CACHE_MAX_BYTES MAP_QUERY_CONCURRENCY \
    MAP_GZIP_MIN_BYTES BENCHMARK_DAYS BENCHMARK_RANDOM_REQUESTS \
    BENCHMARK_CONCURRENCY RUN_MOTIF_DISCOVERY; do
    require_value "$name"
  done
  require_dir REPO
  require_dir ROOT
  require_file PYTHON
  require_file NODE
  require_file ECHO
  require_file FULL_2025
  require_file FULL_2026
  require_file ACQ_2026
  [[ -x "$PYTHON" ]] || fail "PYTHON is not executable: $PYTHON"
  [[ -x "$NODE" ]] || fail "NODE is not executable: $NODE"
  [[ "$("$NODE" --version 2>/dev/null || true)" == v* ]] || \
    fail "NODE does not point to a working Node.js executable: $NODE"
  [[ -f "$GEN/manifest.json" ]] || fail "generation manifest is missing: $GEN/manifest.json"
  [[ "$AVAILABILITY_MODE" == reconstructed || "$AVAILABILITY_MODE" == strict ]] || \
    fail "AVAILABILITY_MODE must be reconstructed or strict"
  [[ "${V4_FIRST_RELEASE,,}" == true || "${V4_FIRST_RELEASE,,}" == false ]] || \
    fail "V4_FIRST_RELEASE must be true or false"
  [[ "${REPLACE_MANAGED_SERVER,,}" == true || "${REPLACE_MANAGED_SERVER,,}" == false ]] || \
    fail "REPLACE_MANAGED_SERVER must be true or false"
  [[ "${RUN_MOTIF_DISCOVERY,,}" == true || "${RUN_MOTIF_DISCOVERY,,}" == false ]] || \
    fail "RUN_MOTIF_DISCOVERY must be true or false"
  [[ "${BUILD_V3_IF_MISSING,,}" == true || "${BUILD_V3_IF_MISSING,,}" == false ]] || \
    fail "BUILD_V3_IF_MISSING must be true or false"
  if [[ "${V4_FIRST_RELEASE,,}" == false ]]; then
    require_dir PREVIOUS_EVIDENCE_DIR
  fi
  if [[ "${RUN_MOTIF_DISCOVERY,,}" == true ]]; then
    for name in \
      RAPIDS_PYTHON MOTIF_GPU MOTIF_TRAIN_THROUGH \
      MOTIF_CALIBRATION_THROUGH MOTIF_EVALUATION_THROUGH \
      MOTIF_MIN_CLUSTER_SIZE MOTIF_MIN_SAMPLES \
      MOTIF_MAX_NOVEL_FALSE_ACCEPT_RATE MOTIF_MIN_WEATHER_DAYS \
      MOTIF_MIN_S2_ACQUISITIONS; do
      require_value "$name"
    done
    require_file RAPIDS_PYTHON
    [[ -x "$RAPIDS_PYTHON" ]] || fail "RAPIDS_PYTHON is not executable"
  fi
  command -v git >/dev/null || fail "git is not available"
  command -v curl >/dev/null || fail "curl is not available"
  command -v flock >/dev/null || fail "flock is not available"
  cd "$REPO"
  git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD || \
    fail "repository is older than required commit $REQUIRED_COMMIT; run git pull --ff-only"
  mkdir -p "$ROOT/duckdb_tmp" "$ROOT/logs" "$ROOT/jobs" \
    "$ROOT/releases" "$ROOT/sources" "$ROOT/models" "$ROOT/evaluations"
}

resolve_compatible_v3_incident_dir() {
  local pointer="$ROOT/logs/latest_incident_v3_job.txt"
  [[ -s "$pointer" ]] || return 1
  V3_JOB=$(tr -d '\r\n' < "$pointer")
  [[ -f "$V3_JOB/status" ]] || return 1
  [[ "$(tr -d '\r\n' < "$V3_JOB/status")" == 0 ]] || return 1
  INCIDENT_DIR=$(json_value "$V3_JOB/state.json" paths.incident_dir)
  [[ -f "$INCIDENT_DIR/manifest.json" ]] || return 1
  "$PYTHON" -c '
import json,sys
generation=json.load(open(sys.argv[1]))
incident=json.load(open(sys.argv[2]))
expected=str((generation.get("run") or {}).get("generation_id") or "")
actual=str((incident.get("run") or {}).get("source_generation_id") or "")
status=str((incident.get("run") or {}).get("status") or "")
schema=str(incident.get("schema_version") or "")
raise SystemExit(0 if expected and actual == expected and status == "complete" and schema == "crop-impact-incident-generation-v3/1" else 1)
' "$GEN/manifest.json" "$INCIDENT_DIR/manifest.json"
}

ensure_v3_incident_dir() {
  if resolve_compatible_v3_incident_dir; then
    printf '%s INFO reusing_compatible_v3=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$INCIDENT_DIR"
    return 0
  fi
  [[ "${BUILD_V3_IF_MISSING,,}" == true ]] || \
    fail "no successful V3 release matches GEN and BUILD_V3_IF_MISSING=false"
  phase BUILD_V3_STORY_SPINE
  local v3_job="$ROOT/jobs/incident_v3_${PIPELINE_TAG}"
  "$PYTHON" server/run_incident_v3.py run \
    --root "$ROOT" \
    --generation-dir "$GEN" \
    --python "$PYTHON" \
    --node "$NODE" \
    --baseline-through "$V3_BASELINE_THROUGH" \
    --first-release \
    --threads "$DUCKDB_THREADS" \
    --memory-limit "$DUCKDB_MEMORY_LIMIT" \
    --temp-dir "$ROOT/duckdb_tmp" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" \
    --job-tag "$PIPELINE_TAG"
  [[ "$(tr -d '\r\n' < "$v3_job/status")" == 0 ]] || \
    fail "V3 runner did not succeed: $v3_job"
  resolve_compatible_v3_incident_dir || \
    fail "new V3 release does not match the configured generation"
}

write_server_env() {
  local viewer_dir=$1
  local target="$REPO/server/.env"
  local temporary="${target}.tmp.$$"
  umask 077
  {
    printf 'STORY_MAP_RUN_DIR=%s\n' "$viewer_dir"
    printf 'STORY_MAP_STATIC_DIR=./static\n'
    printf 'STORY_MAP_HOST=%s\n' "$MAP_HOST"
    printf 'STORY_MAP_PORT=%s\n' "$MAP_PORT"
    printf 'STORY_MAP_LOG_LEVEL=%s\n' "$MAP_LOG_LEVEL"
    printf 'STORY_MAP_RASTER_TILES=%s\n' "$MAP_RASTER_TILES"
    printf 'STORY_MAP_RASTER_ATTRIBUTION=%s\n' "$MAP_RASTER_ATTRIBUTION"
    printf 'STORY_MAP_DEFAULT_FEATURE_LIMIT=%s\n' "$MAP_DEFAULT_FEATURE_LIMIT"
    printf 'STORY_MAP_MAX_FEATURE_LIMIT=%s\n' "$MAP_MAX_FEATURE_LIMIT"
    printf 'STORY_MAP_CACHE_SECONDS=%s\n' "$MAP_CACHE_SECONDS"
    printf 'STORY_MAP_CACHE_ENTRIES=%s\n' "$MAP_CACHE_ENTRIES"
    printf 'STORY_MAP_CACHE_MAX_BYTES=%s\n' "$MAP_CACHE_MAX_BYTES"
    printf 'STORY_MAP_QUERY_CONCURRENCY=%s\n' "$MAP_QUERY_CONCURRENCY"
    printf 'STORY_MAP_GZIP_MIN_BYTES=%s\n' "$MAP_GZIP_MIN_BYTES"
  } > "$temporary"
  mv "$temporary" "$target"
}

validate_viewer() {
  local viewer_dir=$1
  PYTHONPATH=server "$PYTHON" -c '
import json,sys
from pathlib import Path
from story_monitor.incident_viewer_v4 import validate_viewer_directory
root=Path(sys.argv[1])
result=validate_viewer_directory(root)
manifest=json.loads((root/"manifest.json").read_text())
assert manifest["schema_version"] == "crop-incident-viewer-v4/2"
assert manifest["semantics"]["lifecycle_state_recomputed_from_v4"] is False
assert manifest["semantics"]["lifecycle_causal_ownership_claimed"] is False
print(json.dumps(result, indent=2, sort_keys=True))
' "$viewer_dir"
}

start_server_and_benchmark() {
  local viewer_dir=$1
  write_server_env "$viewer_dir"
  if [[ "${REPLACE_MANAGED_SERVER,,}" == true ]]; then
    stop_managed_server
  fi
  if "$PYTHON" -c \
      'import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex((sys.argv[1],int(sys.argv[2]))) == 0 else 1)' \
      "$MAP_HOST" "$MAP_PORT"; then
    fail "$MAP_HOST:$MAP_PORT is already in use and is not a stoppable managed server"
  fi

  unset STORY_MAP_RUN_DIR STORY_MAP_STATIC_DIR STORY_MAP_HOST STORY_MAP_PORT
  unset STORY_MAP_LOG_LEVEL STORY_MAP_RASTER_TILES STORY_MAP_RASTER_ATTRIBUTION
  unset STORY_MAP_DEFAULT_FEATURE_LIMIT STORY_MAP_MAX_FEATURE_LIMIT
  unset STORY_MAP_CACHE_SECONDS STORY_MAP_CACHE_ENTRIES STORY_MAP_CACHE_MAX_BYTES
  unset STORY_MAP_QUERY_CONCURRENCY STORY_MAP_GZIP_MIN_BYTES

  SERVER_LOG="$ROOT/logs/incident_v4_server_${PIPELINE_TAG}.log"
  SERVER_PID_FILE="$ROOT/logs/incident_v4_server_${PIPELINE_TAG}.pid"
  nohup "$PYTHON" server/story_map_server.py \
    > "$SERVER_LOG" 2>&1 < /dev/null &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$SERVER_PID_FILE"
  printf '%s\n' "$SERVER_PID" > "$ROOT/logs/latest_incident_v4_server.pid"

  local ready=0
  for _ in $(seq 1 150); do
    if curl -fsS --max-time 2 \
        "http://$MAP_HOST:$MAP_PORT/api/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || {
      tail -n 100 "$SERVER_LOG" >&2 || true
      fail "story-map server exited before becoming healthy"
    }
    sleep 2
  done
  [[ "$ready" == 1 ]] || fail "story-map server did not become healthy"

  BENCHMARK_JSON="$ROOT/logs/incident_v4_benchmark_${PIPELINE_TAG}.json"
  "$PYTHON" server/benchmark_incident_v4.py \
    --base-url "http://$MAP_HOST:$MAP_PORT" \
    --days "$BENCHMARK_DAYS" \
    --random-requests "$BENCHMARK_RANDOM_REQUESTS" \
    --concurrency "$BENCHMARK_CONCURRENCY" \
    --server-pid "$SERVER_PID" \
    --output "$BENCHMARK_JSON"
}

run_motif_discovery() {
  local evidence_dir=$1
  local viewer_dir=$2
  [[ "${RUN_MOTIF_DISCOVERY,,}" == true ]] || return 0
  require_file RAPIDS_PYTHON
  [[ -x "$RAPIDS_PYTHON" ]] || fail "RAPIDS_PYTHON is not executable"
  DISCOVERY_DIR="$ROOT/models/incident_motif_v4_discovery_${PIPELINE_TAG}"
  DISCOVERY_JSON="$ROOT/logs/incident_motif_v4_discovery_${PIPELINE_TAG}.json"
  DISCOVERY_LOG="$ROOT/logs/incident_motif_v4_discovery_${PIPELINE_TAG}.log"
  printf '%s INFO motif_discovery_log=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$DISCOVERY_LOG"
  env CUDA_VISIBLE_DEVICES="$MOTIF_GPU" PYTHONUNBUFFERED=1 \
    "$RAPIDS_PYTHON" server/run_incident_motifs_v4.py discover \
    --incident-dir "$INCIDENT_DIR" \
    --evidence-dir "$evidence_dir" \
    --viewer-dir "$viewer_dir" \
    --output-dir "$DISCOVERY_DIR" \
    --train-through "$MOTIF_TRAIN_THROUGH" \
    --calibration-through "$MOTIF_CALIBRATION_THROUGH" \
    --evaluation-through "$MOTIF_EVALUATION_THROUGH" \
    --engine gpu \
    --min-cluster-size "$MOTIF_MIN_CLUSTER_SIZE" \
    --min-samples "$MOTIF_MIN_SAMPLES" \
    --maximum-novel-false-accept-rate "$MOTIF_MAX_NOVEL_FALSE_ACCEPT_RATE" \
    --minimum-weather-observed-days "$MOTIF_MIN_WEATHER_DAYS" \
    --minimum-s2-acquisitions-for-crop-support "$MOTIF_MIN_S2_ACQUISITIONS" \
    --threads "$DUCKDB_THREADS" \
    --memory-limit "$DUCKDB_MEMORY_LIMIT" \
    --temp-dir "$ROOT/duckdb_tmp" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" \
    > "$DISCOVERY_JSON" \
    2> >(tee -a "$DISCOVERY_LOG" >&2)
  printf '%s\n' "$DISCOVERY_DIR" > \
    "$ROOT/logs/latest_incident_motif_v4_discovery.txt"
}

run_pipeline() {
  local env_file=$1
  PIPELINE_TAG=$2
  PIPELINE_BASE=$(pipeline_base "$PIPELINE_TAG")
  finish_status() {
    local code=$?
    local temporary="${PIPELINE_BASE}.status.tmp.$$"
    if [[ "$code" != 0 ]]; then
      phase FAILED || true
    fi
    printf '%s\n' "$code" > "$temporary"
    mv "$temporary" "${PIPELINE_BASE}.status"
  }
  trap finish_status EXIT
  exec 9> "$ROOT/logs/vm_story_pipeline.lock"
  flock -n 9 || fail "another VM story pipeline is already running"

  phase PREFLIGHT
  preflight
  ensure_v3_incident_dir
  printf '%s INFO incident_dir=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$INCIDENT_DIR"

  phase BUILD_V4_2
  local released_at job_dir release_args
  released_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  job_dir="$ROOT/jobs/incident_v4_${PIPELINE_TAG}"
  release_args=(--first-release)
  if [[ "${V4_FIRST_RELEASE,,}" != true ]]; then
    require_dir PREVIOUS_EVIDENCE_DIR
    release_args=(--previous-evidence-dir "$PREVIOUS_EVIDENCE_DIR")
  fi
  "$PYTHON" server/run_incident_v4.py run \
    --root "$ROOT" \
    --generation-dir "$GEN" \
    --incident-dir "$INCIDENT_DIR" \
    --echo-deliverable "$ECHO" \
    --full-parquet "$FULL_2025" \
    --full-parquet "$FULL_2026" \
    --source-acquisition-parquet "$ACQ_2026" \
    --acquisition-parquet "$ACQ_2026" \
    --availability-mode "$AVAILABILITY_MODE" \
    --released-at "$released_at" \
    "${release_args[@]}" \
    --python "$PYTHON" \
    --node "$NODE" \
    --threads "$DUCKDB_THREADS" \
    --memory-limit "$DUCKDB_MEMORY_LIMIT" \
    --temp-dir "$ROOT/duckdb_tmp" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" \
    --job-tag "$PIPELINE_TAG"

  [[ "$(tr -d '\r\n' < "$job_dir/status")" == 0 ]] || \
    fail "V4 runner did not succeed: $job_dir"
  EVIDENCE_DIR=$(json_value "$job_dir/state.json" paths.evidence_dir)
  VIEWER_DIR=$(json_value "$job_dir/state.json" paths.viewer_dir)

  phase VALIDATE_V4_2
  validate_viewer "$VIEWER_DIR"

  phase START_SERVER_AND_BENCHMARK
  start_server_and_benchmark "$VIEWER_DIR"

  if [[ "${RUN_MOTIF_DISCOVERY,,}" == true ]]; then
    phase GPU_MOTIF_DISCOVERY
    run_motif_discovery "$EVIDENCE_DIR" "$VIEWER_DIR"
    phase AWAITING_MANDATORY_MOTIF_REVIEW
    printf '%s INFO review_template=%s/review_overlay_template.parquet\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$DISCOVERY_DIR"
  else
    phase COMPLETE_MAP_READY
  fi

  printf '%s INFO viewer_dir=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VIEWER_DIR"
  printf '%s INFO server=http://%s:%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MAP_HOST" "$MAP_PORT"
  printf '%s INFO benchmark=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BENCHMARK_JSON"
}

show_status() {
  local tag base pid status
  tag=$(latest_tag)
  base=$(pipeline_base "$tag")
  pid=$(tr -dc '0-9' < "${base}.pid" 2>/dev/null || true)
  if [[ -s "${base}.status" ]]; then
    status=$(tr -d '\r\n' < "${base}.status")
  elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    status=RUNNING
  else
    status=DEAD_WITHOUT_STATUS
  fi
  printf 'TAG=%s\n' "$tag"
  printf 'PHASE=%s\n' "$(tr -d '\r\n' < "${base}.phase" 2>/dev/null || printf UNKNOWN)"
  printf 'STATUS=%s\n' "$status"
  printf 'PID=%s\n' "$pid"
  printf 'LOG=%s.log\n' "$base"
  tail -n 30 "${base}.log" 2>/dev/null || true
  if [[ -s "$ROOT/logs/latest_incident_v4_job.txt" ]]; then
    "$PYTHON" server/run_incident_v4.py status \
      --job-dir "$(tr -d '\r\n' < "$ROOT/logs/latest_incident_v4_job.txt")" || true
  fi
  curl -fsS --max-time 2 "http://$MAP_HOST:$MAP_PORT/api/health" 2>/dev/null || true
  printf '\n'
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
  launch)
    preflight
    exec 8> "$ROOT/logs/vm_story_pipeline_launch.lock"
    flock -n 8 || fail "another pipeline launch is being prepared"
    if [[ -s "$ROOT/logs/latest_vm_story_pipeline.txt" ]]; then
      prior_tag=$(tr -d '\r\n' < "$ROOT/logs/latest_vm_story_pipeline.txt")
      prior_base=$(pipeline_base "$prior_tag")
      prior_pid=$(tr -dc '0-9' < "${prior_base}.pid" 2>/dev/null || true)
      if [[ ! -s "${prior_base}.status" && -n "$prior_pid" ]] \
          && kill -0 "$prior_pid" 2>/dev/null; then
        fail "pipeline $prior_tag is still running as PID $prior_pid"
      fi
    fi
    tag=$(date -u +%Y%m%dT%H%M%SZ)
    base=$(pipeline_base "$tag")
    nohup "$0" __run "$env_file" "$tag" \
      > "${base}.log" 2>&1 < /dev/null &
    pipeline_pid=$!
    printf '%s\n' "$pipeline_pid" > "${base}.pid"
    pointer_tmp="$ROOT/logs/latest_vm_story_pipeline.txt.tmp.$$"
    printf '%s\n' "$tag" > "$pointer_tmp"
    mv "$pointer_tmp" "$ROOT/logs/latest_vm_story_pipeline.txt"
    printf 'Pipeline started.\nPID=%s\nLOG=%s.log\n' "$pipeline_pid" "$base"
    ;;
  __run)
    [[ $# -eq 3 ]] || fail "invalid internal pipeline invocation"
    run_pipeline "$2" "$3"
    ;;
  status)
    require_value ROOT
    require_value REPO
    require_value PYTHON
    require_value MAP_HOST
    require_value MAP_PORT
    [[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
    [[ -d "$REPO" ]] || fail "REPO does not exist: $REPO"
    [[ -x "$PYTHON" ]] || fail "PYTHON is not executable: $PYTHON"
    cd "$REPO"
    show_status
    ;;
  logs)
    require_value ROOT
    [[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
    tag=$(latest_tag)
    tail -f "$(pipeline_base "$tag").log"
    ;;
  check-node)
    require_value NODE
    [[ -x "$NODE" ]] || fail "NODE is not executable: $NODE"
    version=$("$NODE" --version 2>/dev/null || true)
    [[ "$version" == v* ]] || fail "NODE is not a working Node.js executable: $NODE"
    printf 'NODE=%s\nNODE_VERSION=%s\n' "$NODE" "$version"
    ;;
  check-v3)
    require_value ROOT
    require_value PYTHON
    require_value GEN
    [[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
    [[ -x "$PYTHON" ]] || fail "PYTHON is not executable: $PYTHON"
    [[ -f "$GEN/manifest.json" ]] || fail "generation manifest is missing"
    resolve_compatible_v3_incident_dir || \
      fail "no successful V3 release matches GEN"
    printf 'INCIDENT_DIR=%s\n' "$INCIDENT_DIR"
    ;;
  stop-server)
    require_value ROOT
    [[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
    stop_managed_server
    ;;
  *)
    usage >&2
    fail "unknown command: $command"
    ;;
esac
