#!/usr/bin/env bash
# Mooncake HA chaos engine — kill a master / etcd node and measure recovery.
#
# Companion to start_etcd_cluster.sh + start_mooncake_master_ha.sh.
# Reads PID files from those scripts' default locations to identify targets,
# then SIGKILLs the chosen victim and polls etcd to time the failover.
#
# Modes:
#   kill_master          - kill the current Mooncake master leader
#   kill_etcd_leader     - kill the current etcd Raft leader
#   kill_etcd_follower   - kill an etcd follower (control: should be no-op)
#   combined             - kill master leader + etcd-0 simultaneously
#
# Usage:
#   bash failover_chaos.sh <MODE> [OUT_DIR]
#
# Output:
#   <OUT_DIR>/failover.json — recovery_seconds, old_leader, new_leader,
#                              killed_etcd_role, ha_code_path_verified,
#                              state_recovery (master metrics pre/post),
#                              status (ok / no_failover / abort)
#   <OUT_DIR>/failover.timeline.log — wall-clock event log
#   <OUT_DIR>/master-N_metrics_{pre,post}_chaos.txt — full Prometheus dumps
#
# Required environment (or auto-detected):
#   ETCD_ENDPOINTS       - comma-separated etcd client URLs
#   CLUSTER_ID           - mooncake cluster id (default: mooncake)
#   ETCDCTL              - path to etcdctl binary (default: etcdctl from PATH)
#
# How recovery is measured:
#   T_KILL = SIGKILL timestamp (date +%s.%3N)
#   T_VIEW_CHANGE = first etcd poll where master_view key shows new value
#   recovery_seconds = T_VIEW_CHANGE - T_KILL (50ms polling, ±25ms precision)
#
# State recovery metrics (master_key_count, master_allocated_bytes, etc.) are
# captured before chaos and 5s after view change. With current Mooncake, post
# values are 0 (election-only HA — see knowledge docs); the field is here so
# future stateful-HA changes can be measured.

set -uo pipefail

MODE="${1:?MODE required (kill_master|kill_etcd_leader|kill_etcd_follower|combined)}"
OUT_DIR="${2:-./chaos_out}"
mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETCD_PID_DIR="$SCRIPT_DIR/.etcd_pids"
MASTER_PID_DIR="$SCRIPT_DIR/.master_pids"
MASTER_LOG_DIR="$SCRIPT_DIR/master_logs"

CLUSTER_ID="${CLUSTER_ID:-mooncake}"
ETCDCTL="${ETCDCTL:-etcdctl}"
HOST_IP="${MC_HOST_IP:-$(hostname -I | awk '{print $1}')}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-${HOST_IP}:23800,${HOST_IP}:23801,${HOST_IP}:23802}"

FAILOVER_LOG="$OUT_DIR/failover.json"
TS_LOG="$OUT_DIR/failover.timeline.log"

ts_now() { date -u +%s.%3N; }
log() { echo "[$(date -u +%H:%M:%S.%3N)] $*" >> "$TS_LOG"; }

log "chaos start: mode=$MODE etcd=$ETCD_ENDPOINTS cluster=$CLUSTER_ID"

# ---- Identify Mooncake master leader from etcd ----
view=$("$ETCDCTL" --endpoints="$ETCD_ENDPOINTS" \
       get "mooncake-store/${CLUSTER_ID}/master_view" \
       --print-value-only 2>/dev/null | tr -d '[:space:]')
if [ -z "$view" ]; then
  log "ERROR: no master_view in etcd; abort"
  echo '{"status":"abort","reason":"no master_view"}' > "$FAILOVER_LOG"
  exit 1
fi
leader_port="${view#*:}"
log "current master leader: $view"

LEADER_RANK=""
for i in 0 1 2; do
  port_in_log=$(grep -oE "rpc_port=[0-9]+" "$MASTER_LOG_DIR/master-${i}.log" 2>/dev/null | head -1 | sed 's/rpc_port=//')
  [ "$port_in_log" = "$leader_port" ] && LEADER_RANK=$i
done
log "master leader rank: ${LEADER_RANK:-NOT_FOUND}"

# HA code path verification
HA_VERIFIED="no"
if [ -n "$LEADER_RANK" ]; then
  if grep -q "enable_ha=1" "$MASTER_LOG_DIR/master-${LEADER_RANK}.log" 2>/dev/null; then
    HA_VERIFIED="yes"
  fi
fi
log "HA code path enable_ha=1 verified: $HA_VERIFIED"

# Identify etcd Raft leader
ETCD_LEADER_INDEX=""
ETCD_STATUS_JSON=$("$ETCDCTL" --endpoints="$ETCD_ENDPOINTS" endpoint status -w json 2>/dev/null)
if [ -n "$ETCD_STATUS_JSON" ] && command -v jq >/dev/null 2>&1; then
  IFS=',' read -ra EP_ARR <<< "$ETCD_ENDPOINTS"
  for idx in 0 1 2; do
    ep="${EP_ARR[$idx]:-}"
    [ -z "$ep" ] && continue
    leader_id=$(echo "$ETCD_STATUS_JSON" | jq -r --arg ep "$ep" \
      '.[] | select((.Endpoint // "") | endswith($ep)) | .Status.leader' 2>/dev/null)
    member_id=$(echo "$ETCD_STATUS_JSON" | jq -r --arg ep "$ep" \
      '.[] | select((.Endpoint // "") | endswith($ep)) | .Status.header.member_id' 2>/dev/null)
    if [ -n "$leader_id" ] && [ "$leader_id" = "$member_id" ]; then
      ETCD_LEADER_INDEX=$idx
      break
    fi
  done
fi
log "etcd Raft leader index: ${ETCD_LEADER_INDEX:-UNKNOWN}"

# Pre-chaos metrics snapshot
snap_master_metrics() {
  local suffix="$1"
  for i in 0 1 2; do
    metrics_port=$(grep -oE "metrics_port=[0-9]+" "$MASTER_LOG_DIR/master-${i}.log" 2>/dev/null | head -1 | sed 's/metrics_port=//')
    if [ -n "$metrics_port" ]; then
      curl -s --max-time 2 "http://127.0.0.1:$metrics_port/metrics" \
        > "$OUT_DIR/master-${i}_metrics_${suffix}.txt" 2>/dev/null
    fi
  done
}
snap_master_metrics "pre_chaos"

metric_val() {
  local file="$1" name="$2"
  awk -v m="$name" '$1 == m {print $2; exit}' "$file" 2>/dev/null
}

# ---- Decide kill target ----
T_KILL=$(ts_now)
KILLED_ETCD_ROLE="none"
KILLED_ETCD_INDEX=""

case "$MODE" in
  kill_master)
    if [ -n "$LEADER_RANK" ]; then
      pid=$(cat "$MASTER_PID_DIR/master-${LEADER_RANK}.pid")
      log "SIGKILL master leader rank=$LEADER_RANK pid=$pid"
      kill -KILL "$pid" 2>/dev/null
    fi
    ;;
  kill_etcd_leader)
    if [ -z "$ETCD_LEADER_INDEX" ]; then
      log "ERROR: cannot identify etcd Raft leader; abort"
      echo '{"status":"abort","reason":"etcd_leader_not_identified"}' > "$FAILOVER_LOG"
      exit 1
    fi
    pid=$(cat "$ETCD_PID_DIR/etcd-${ETCD_LEADER_INDEX}.pid" 2>/dev/null)
    log "SIGKILL etcd Raft leader: etcd-${ETCD_LEADER_INDEX} pid=$pid"
    kill -KILL "$pid" 2>/dev/null
    KILLED_ETCD_INDEX=$ETCD_LEADER_INDEX
    KILLED_ETCD_ROLE="leader"
    ;;
  kill_etcd_follower)
    if [ -z "$ETCD_LEADER_INDEX" ]; then
      log "ERROR: cannot identify etcd Raft leader (needed to pick a follower); abort"
      echo '{"status":"abort","reason":"etcd_leader_not_identified"}' > "$FAILOVER_LOG"
      exit 1
    fi
    FOLLOWER_IDX=""
    for i in 0 1 2; do
      [ "$i" != "$ETCD_LEADER_INDEX" ] && FOLLOWER_IDX=$i && break
    done
    pid=$(cat "$ETCD_PID_DIR/etcd-${FOLLOWER_IDX}.pid" 2>/dev/null)
    log "SIGKILL etcd follower: etcd-${FOLLOWER_IDX} pid=$pid"
    kill -KILL "$pid" 2>/dev/null
    KILLED_ETCD_INDEX=$FOLLOWER_IDX
    KILLED_ETCD_ROLE="follower"
    ;;
  combined)
    if [ -n "$LEADER_RANK" ]; then
      kill -KILL $(cat "$MASTER_PID_DIR/master-${LEADER_RANK}.pid") 2>/dev/null
    fi
    kill -KILL $(cat "$ETCD_PID_DIR/etcd-0.pid") 2>/dev/null
    log "killed master leader rank=$LEADER_RANK + etcd-0"
    KILLED_ETCD_INDEX=0
    KILLED_ETCD_ROLE="$([ "$ETCD_LEADER_INDEX" = "0" ] && echo leader || echo follower)"
    ;;
esac

# ---- Wait for new master_view (60s budget, 50ms polling = ±25ms precision) ----
NEW_LEADER=""
T_VIEW_CHANGE=""
for _ in $(seq 1200); do  # 60s / 50ms
  current=$("$ETCDCTL" --endpoints="$ETCD_ENDPOINTS" \
            get "mooncake-store/${CLUSTER_ID}/master_view" \
            --print-value-only 2>/dev/null | tr -d '[:space:]')
  if [ -n "$current" ] && [ "$current" != "$view" ]; then
    NEW_LEADER="$current"
    T_VIEW_CHANGE=$(ts_now)
    log "view changed: $view → $NEW_LEADER"
    break
  fi
  sleep 0.05
done
[ -z "$NEW_LEADER" ] && log "WARN: view did not change in 60s"

T_END=$(ts_now)
RECOVERY_S=$(awk -v a="$T_KILL" -v b="${T_VIEW_CHANGE:-$T_END}" 'BEGIN{printf "%.3f", b-a}')

# Post-chaos metrics (5s after view change to let new leader stabilize)
sleep 5
snap_master_metrics "post_chaos"

# ---- State recovery: read 4 metrics from old leader (pre) + new leader (post) ----
PRE_ALLOCATED_BYTES="null"; PRE_TOTAL_CAPACITY="null"; PRE_KEY_COUNT="null"; PRE_ACTIVE_CLIENTS="null"
if [ -n "$LEADER_RANK" ]; then
  PRE_FILE="$OUT_DIR/master-${LEADER_RANK}_metrics_pre_chaos.txt"
  if [ -f "$PRE_FILE" ]; then
    v=$(metric_val "$PRE_FILE" "master_allocated_bytes");      [ -n "$v" ] && PRE_ALLOCATED_BYTES="$v"
    v=$(metric_val "$PRE_FILE" "master_total_capacity_bytes"); [ -n "$v" ] && PRE_TOTAL_CAPACITY="$v"
    v=$(metric_val "$PRE_FILE" "master_key_count");            [ -n "$v" ] && PRE_KEY_COUNT="$v"
    v=$(metric_val "$PRE_FILE" "master_active_clients");       [ -n "$v" ] && PRE_ACTIVE_CLIENTS="$v"
  fi
fi
POST_ALLOCATED_BYTES="null"; POST_TOTAL_CAPACITY="null"; POST_KEY_COUNT="null"; POST_ACTIVE_CLIENTS="null"
NEW_LEADER_RANK=""
if [ -n "$NEW_LEADER" ]; then
  new_port="${NEW_LEADER#*:}"
  for i in 0 1 2; do
    port_in_log=$(grep -oE "rpc_port=[0-9]+" "$MASTER_LOG_DIR/master-${i}.log" 2>/dev/null | head -1 | sed 's/rpc_port=//')
    [ "$port_in_log" = "$new_port" ] && NEW_LEADER_RANK=$i
  done
  if [ -n "$NEW_LEADER_RANK" ]; then
    POST_FILE="$OUT_DIR/master-${NEW_LEADER_RANK}_metrics_post_chaos.txt"
    if [ -f "$POST_FILE" ]; then
      v=$(metric_val "$POST_FILE" "master_allocated_bytes");      [ -n "$v" ] && POST_ALLOCATED_BYTES="$v"
      v=$(metric_val "$POST_FILE" "master_total_capacity_bytes"); [ -n "$v" ] && POST_TOTAL_CAPACITY="$v"
      v=$(metric_val "$POST_FILE" "master_key_count");            [ -n "$v" ] && POST_KEY_COUNT="$v"
      v=$(metric_val "$POST_FILE" "master_active_clients");       [ -n "$v" ] && POST_ACTIVE_CLIENTS="$v"
    fi
  fi
fi

cat > "$FAILOVER_LOG" <<EOF
{
  "mode": "$MODE",
  "old_leader": "$view",
  "new_leader": "${NEW_LEADER:-none}",
  "leader_rank": "${LEADER_RANK:-unknown}",
  "new_leader_rank": "${NEW_LEADER_RANK:-unknown}",
  "etcd_leader_index_at_chaos": "${ETCD_LEADER_INDEX:-unknown}",
  "killed_etcd_index": "${KILLED_ETCD_INDEX:-none}",
  "killed_etcd_role": "$KILLED_ETCD_ROLE",
  "ha_code_path_verified": "$HA_VERIFIED",
  "t_kill": $T_KILL,
  "t_view_change": ${T_VIEW_CHANGE:-null},
  "recovery_seconds": $RECOVERY_S,
  "recovery_precision_ms": 25,
  "polling_interval_ms": 50,
  "state_recovery": {
    "pre_allocated_bytes":      $PRE_ALLOCATED_BYTES,
    "post_allocated_bytes":     $POST_ALLOCATED_BYTES,
    "pre_total_capacity_bytes": $PRE_TOTAL_CAPACITY,
    "post_total_capacity_bytes":$POST_TOTAL_CAPACITY,
    "pre_key_count":            $PRE_KEY_COUNT,
    "post_key_count":           $POST_KEY_COUNT,
    "pre_active_clients":       $PRE_ACTIVE_CLIENTS,
    "post_active_clients":      $POST_ACTIVE_CLIENTS
  },
  "status": "$([ -n "$NEW_LEADER" ] && echo ok || echo no_failover)"
}
EOF
log "failover.json written: recovery=${RECOVERY_S}s pre_keys=${PRE_KEY_COUNT} post_keys=${POST_KEY_COUNT}"
echo "✓ chaos result → $FAILOVER_LOG"
