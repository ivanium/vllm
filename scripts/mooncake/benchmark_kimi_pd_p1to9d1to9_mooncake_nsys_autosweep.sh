#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PARTITION="${PARTITION:-batch}"
START_PAIR_COUNT="${START_PAIR_COUNT:-4}"
END_PAIR_COUNT="${END_PAIR_COUNT:-9}"
BASE_MULTI_TURN_CONCURRENCY="${BASE_MULTI_TURN_CONCURRENCY:-8}"
CONCURRENCY_STEP="${CONCURRENCY_STEP:-2}"
PROMPTS_FACTOR="${PROMPTS_FACTOR:-2}"
WAIT_INTERVAL_SECS="${WAIT_INTERVAL_SECS:-60}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NODE_COOLDOWN_SECS="${NODE_COOLDOWN_SECS:-3600}"
INITIAL_COOLDOWN_NODES="${INITIAL_COOLDOWN_NODES:-}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/bench_results/kimi_pd_p1to9d1to9_mooncake_nsys_autosweep/${RUN_TAG}}"
VIGIL_CONFIG="${VIGIL_CONFIG:-${REPO_ROOT}/third_partys/vigil/recipes/crusoe/kimik25/low_latency/pd_xpyd_mooncake_offload_nsys.yaml}"
VIGIL_BIN="${VIGIL_BIN:-vigil}"

declare -A NODE_COOLDOWNS=()

OOM_PATTERNS=(
  "Free memory on device"
  "less than desired GPU memory utilization"
  "CUDA out of memory"
  "out of memory"
  "requested_memory"
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! is_positive_int "$value"; then
    printf 'error: %s must be a positive integer, got %q\n' "$name" "$value" >&2
    exit 1
  fi
}

count_idle_nodes() {
  list_idle_nodes | awk 'NF' | wc -l | tr -d ' '
}

print_idle_nodes() {
  list_idle_nodes | tr '\n' ' '
}

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[\\/&]/\\&/g'
}

cleanup_expired_cooldowns() {
  local now
  now="$(date +%s)"
  for node in "${!NODE_COOLDOWNS[@]}"; do
    if (( NODE_COOLDOWNS[$node] <= now )); then
      unset "NODE_COOLDOWNS[$node]"
    fi
  done
}

is_node_excluded() {
  local node="$1"
  [[ -n "${NODE_COOLDOWNS[$node]:-}" ]]
}

list_excluded_nodes() {
  cleanup_expired_cooldowns
  printf '%s\n' "${!NODE_COOLDOWNS[@]}" | awk 'NF' | sort -u
}

build_exclude_csv() {
  local joined=""
  while IFS= read -r node; do
    if [[ -z "$joined" ]]; then
      joined="$node"
    else
      joined="${joined},${node}"
    fi
  done < <(list_excluded_nodes)
  printf '%s' "$joined"
}

search_logs_for_pattern() {
  local point_dir="$1"
  local filename_glob="$2"
  local pattern="$3"
  local log_path
  if command -v rg >/dev/null 2>&1; then
    rg -q -i --glob "$filename_glob" -- "$pattern" "$point_dir"
    return $?
  fi

  while IFS= read -r -d '' log_path; do
    if grep -q -i -- "$pattern" "$log_path"; then
      return 0
    fi
  done < <(find "$point_dir" -type f -name "$filename_glob" -print0)
  return 1
}

list_idle_nodes() {
  local node
  cleanup_expired_cooldowns
  while IFS= read -r node; do
    [[ -z "$node" ]] && continue
    if ! is_node_excluded "$node"; then
      printf '%s\n' "$node"
    fi
  done < <(sinfo -N -h -p "$PARTITION" -t idle -o '%N' | awk 'NF' | sort -u)
}

render_config() {
  local output_path="$1"
  local escaped_partition
  local escaped_exclude
  escaped_partition="$(escape_sed_replacement "$PARTITION")"
  escaped_exclude="$(escape_sed_replacement "$SLURM_EXCLUDE")"
  sed \
    -e "s/__PREFILL_COUNT__/${PREFILL_COUNT}/g" \
    -e "s/__DECODE_COUNT__/${DECODE_COUNT}/g" \
    -e "s/__SLURM_PARTITION__/${escaped_partition}/g" \
    -e "s/__SLURM_EXCLUDE__/${escaped_exclude}/g" \
    "$VIGIL_CONFIG" >"$output_path"
}

parse_allocated_nodes() {
  local point_dir="$1"
  local slurm_out="${point_dir}/slurm.out"
  if [[ ! -f "$slurm_out" ]]; then
    return 0
  fi
  grep 'Nodes allocated:' "$slurm_out" | tail -1 | sed -E 's/^.*Nodes allocated:[[:space:]]*//' | tr ' ' '\n' | awk 'NF' | sort -u
}

detect_bad_nodes() {
  local point_dir="$1"
  shift
  local node
  local pattern
  for node in "$@"; do
    [[ -z "$node" ]] && continue
    for pattern in "${OOM_PATTERNS[@]}"; do
      if search_logs_for_pattern "$point_dir" "*${node}.log" "$pattern"; then
        printf '%s\n' "$node"
        break
      fi
    done
  done | awk 'NF' | sort -u
}

add_nodes_to_cooldown() {
  local cooldown_until="$1"
  shift
  local node
  for node in "$@"; do
    [[ -z "$node" ]] && continue
    NODE_COOLDOWNS["$node"]="$cooldown_until"
  done
}

seed_initial_cooldowns() {
  local cooldown_until="$1"
  local raw_nodes="$2"
  local node
  raw_nodes="${raw_nodes//,/ }"
  for node in $raw_nodes; do
    [[ -z "$node" ]] && continue
    NODE_COOLDOWNS["$node"]="$cooldown_until"
  done
}

format_epoch_utc() {
  date -u -d "@$1" '+%Y-%m-%dT%H:%M:%SZ'
}

write_attempt_meta() {
  local meta_path="$1"
  local status="$2"
  local attempt="$3"
  local exit_code="$4"
  local failure_type="$5"
  local allocated_nodes="$6"
  local bad_nodes="$7"
  local cooldown_until="$8"
  local next_exclude="$9"
  cat >"$meta_path" <<EOF
status=${status}
attempt=${attempt}
prefill_count=${PREFILL_COUNT}
decode_count=${DECODE_COUNT}
pair_count=${PAIR_COUNT}
partition=${PARTITION}
exit_code=${exit_code}
failure_type=${failure_type}
allocated_nodes=${allocated_nodes}
detected_bad_nodes=${bad_nodes}
cooldown_until_epoch=${cooldown_until}
cooldown_until_utc=$(if [[ -n "$cooldown_until" && "$cooldown_until" != "0" ]]; then format_epoch_utc "$cooldown_until"; else printf '%s' ''; fi)
exclude_used=${SLURM_EXCLUDE}
exclude_for_next_retry=${next_exclude}
rendered_config=${rendered_config}
attempt_log_dir=${attempt_log_dir}
result_dir=${point_dir}
EOF
}

trap 'log "Received cancellation signal, exiting autosweep."; exit 130' INT TERM

validate_positive_int START_PAIR_COUNT "$START_PAIR_COUNT"
validate_positive_int END_PAIR_COUNT "$END_PAIR_COUNT"
validate_positive_int BASE_MULTI_TURN_CONCURRENCY "$BASE_MULTI_TURN_CONCURRENCY"
validate_positive_int CONCURRENCY_STEP "$CONCURRENCY_STEP"
validate_positive_int PROMPTS_FACTOR "$PROMPTS_FACTOR"
validate_positive_int WAIT_INTERVAL_SECS "$WAIT_INTERVAL_SECS"
validate_positive_int MAX_RETRIES "$MAX_RETRIES"
validate_positive_int NODE_COOLDOWN_SECS "$NODE_COOLDOWN_SECS"

if (( START_PAIR_COUNT > END_PAIR_COUNT )); then
  printf 'error: START_PAIR_COUNT (%s) must be <= END_PAIR_COUNT (%s)\n' \
    "$START_PAIR_COUNT" "$END_PAIR_COUNT" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"

if [[ -n "$INITIAL_COOLDOWN_NODES" ]]; then
  initial_cooldown_until=$(( $(date +%s) + NODE_COOLDOWN_SECS ))
  seed_initial_cooldowns "$initial_cooldown_until" "$INITIAL_COOLDOWN_NODES"
fi

log "Starting autosweep on partition=${PARTITION}, paired prefill/decode=${START_PAIR_COUNT}:${START_PAIR_COUNT}..${END_PAIR_COUNT}:${END_PAIR_COUNT}"
log "Using VIGIL_CONFIG=${VIGIL_CONFIG}"
log "Using VIGIL_BIN=${VIGIL_BIN}"
log "Results will be written under ${RESULT_ROOT}"
log "Initial cooldown nodes: $(list_excluded_nodes | tr '\n' ' ' | sed 's/[[:space:]]*$//')"

for ((pair_count = START_PAIR_COUNT; pair_count <= END_PAIR_COUNT; pair_count++)); do
  export PAIR_COUNT="$pair_count"
  export PREFILL_COUNT="$pair_count"
  export DECODE_COUNT="$pair_count"

  required_nodes=$((PREFILL_COUNT + DECODE_COUNT))
  target_concurrency=$((BASE_MULTI_TURN_CONCURRENCY + CONCURRENCY_STEP * (pair_count - 1)))
  num_prompts=$((target_concurrency * PROMPTS_FACTOR))

  export BENCH_MULTI_TURN_CONCURRENCY="$target_concurrency"
  export BENCH_NUM_PROMPTS="$num_prompts"

  point_dir="${RESULT_ROOT}/p${pair_count}_d${pair_count}_mtc_${target_concurrency}"
  rendered_config="${point_dir}/rendered_config.yaml"

  export BENCH_RESULT_DIR="$point_dir"
  export BENCH_LABEL="prefill-${pair_count}-decode-${pair_count}-mtc-${target_concurrency}"
  export BENCH_REQUEST_ID_PREFIX="p${pair_count}d${pair_count}-mtc${target_concurrency}-"

  mkdir -p "$point_dir"

  attempt_succeeded=0
  unknown_failure_count=0
  attempt=1
  while true; do
    cleanup_expired_cooldowns
    export SLURM_EXCLUDE
    SLURM_EXCLUDE="$(build_exclude_csv)"
    attempt_meta="${point_dir}/attempt_${attempt}.meta"
    attempt_log_dir="${point_dir}/attempt_${attempt}"
    rendered_config="${attempt_log_dir}/rendered_config.yaml"
    mkdir -p "$attempt_log_dir"

    while true; do
      idle_nodes="$(count_idle_nodes)"
      if (( idle_nodes >= required_nodes )); then
        log "p=${pair_count} d=${pair_count} attempt=${attempt}: found ${idle_nodes} usable idle nodes, need ${required_nodes}; launching vigil."
        break
      fi

      idle_list="$(print_idle_nodes)"
      excluded_nodes="$(list_excluded_nodes | tr '\n' ' ')"
      log "p=${pair_count} d=${pair_count} attempt=${attempt}: only ${idle_nodes} usable idle nodes on partition=${PARTITION}, need ${required_nodes}. Waiting ${WAIT_INTERVAL_SECS}s."
      log "Current usable idle nodes: ${idle_list:-<none>}"
      log "Cooldown excludes: ${excluded_nodes:-<none>}"
      sleep "$WAIT_INTERVAL_SECS"
    done

    render_config "$rendered_config"

    log "Running p=${pair_count}, d=${pair_count}, attempt=${attempt}/${MAX_RETRIES}, concurrency=${target_concurrency}, prompts=${num_prompts}"
    log "Result dir: ${point_dir}"
    log "Attempt log dir: ${attempt_log_dir}"
    log "Rendered config: ${rendered_config}"
    log "Current exclude set: ${SLURM_EXCLUDE:-<none>}"

    if "$VIGIL_BIN" -c "$rendered_config" --log-dir "$attempt_log_dir" "$@"; then
      write_attempt_meta "$attempt_meta" "success" "$attempt" "0" "none" "" "" "0" "$SLURM_EXCLUDE"
      log "Completed p=${pair_count}, d=${pair_count} on attempt=${attempt}"
      attempt_succeeded=1
      break
    else
      exit_code=$?
    fi

    mapfile -t allocated_nodes < <(parse_allocated_nodes "$attempt_log_dir")
    mapfile -t bad_nodes < <(detect_bad_nodes "$attempt_log_dir" "${allocated_nodes[@]}")
    cooldown_until=0
    allocated_nodes_csv="$(printf '%s\n' "${allocated_nodes[@]:-}" | awk 'NF' | paste -sd ',' -)"
    bad_nodes_csv="$(printf '%s\n' "${bad_nodes[@]:-}" | awk 'NF' | paste -sd ',' -)"

    if (( ${#bad_nodes[@]} > 0 )); then
      cooldown_until=$(( $(date +%s) + NODE_COOLDOWN_SECS ))
      add_nodes_to_cooldown "$cooldown_until" "${bad_nodes[@]}"
      failure_type="bad_nodes_detected"
    else
      failure_type="unknown"
    fi

    next_exclude="$(build_exclude_csv)"
    write_attempt_meta "$attempt_meta" "failed" "$attempt" "$exit_code" "$failure_type" "$allocated_nodes_csv" "$bad_nodes_csv" "$cooldown_until" "$next_exclude"

    log "p=${pair_count} d=${pair_count} attempt=${attempt} failed with exit_code=${exit_code}"
    log "Allocated nodes: ${allocated_nodes_csv:-<unknown>}"
    log "Detected bad nodes: ${bad_nodes_csv:-<none>}"
    log "Exclude set for next retry: ${next_exclude:-<none>}"

    if (( ${#bad_nodes[@]} > 0 )); then
      log "p=${pair_count} d=${pair_count} attempt=${attempt}: bad nodes detected; continuing retries without retry cap."
    else
      unknown_failure_count=$((unknown_failure_count + 1))
      log "p=${pair_count} d=${pair_count} attempt=${attempt}: unknown failure ${unknown_failure_count}/${MAX_RETRIES}."
      if (( unknown_failure_count >= MAX_RETRIES )); then
        log "p=${pair_count} d=${pair_count}: unknown failures exhausted retry cap; exiting with failure."
        exit "$exit_code"
      fi
    fi

    log "Retrying p=${pair_count}, d=${pair_count} after failure."
    attempt=$((attempt + 1))
  done

  if (( attempt_succeeded == 0 )); then
    log "p=${pair_count}, d=${pair_count} did not complete successfully."
    exit 1
  fi
done

log "Autosweep finished."
