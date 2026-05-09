#!/usr/bin/env bash
# Start 3 mooncake_master replicas in HA mode pointing at an etcd cluster.
#
# Companion to start_etcd_cluster.sh. After both scripts succeed, exactly
# one master is in role=leader/state=serving; the other two are kStandby.
# Killing the leader triggers automatic failover within ~5s (lease TTL).
#
# Usage:
#   bash start_mooncake_master_ha.sh                  # foreground
#   bash start_mooncake_master_ha.sh --bg             # background
#   bash start_mooncake_master_ha.sh --stop           # kill backgrounded masters
#
# Required environment:
#   ETCD_ENDPOINTS    - comma-separated etcd client URLs
#                       (default: tries to auto-detect from start_etcd_cluster.sh defaults)
#
# Optional environment:
#   MASTER_BIN        - path to mooncake_master      (default: mooncake_master from PATH)
#   MASTER_RPC_BASE   - first master RPC port        (default: 50061; nodes use BASE,+1,+2)
#   MASTER_METRICS_BASE - first metrics port         (default: 9013)
#   MC_HOST_IP        - master advertised IP         (default: hostname -I first IP)
#   CLUSTER_ID        - HA cluster id                (default: mooncake)
#   MC_LEASE_TTL      - KV lease TTL ms              (default: 10000)
#
# After this exits, the active leader's IP:port is in:
#   etcdctl get "mooncake-store/${CLUSTER_ID}/master_view" --print-value-only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.master_pids"
LOG_DIR="$SCRIPT_DIR/master_logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

MASTER_BIN="${MASTER_BIN:-mooncake_master}"
MASTER_RPC_BASE="${MASTER_RPC_BASE:-50061}"
MASTER_METRICS_BASE="${MASTER_METRICS_BASE:-9013}"
MC_HOST_IP="${MC_HOST_IP:-$(hostname -I | awk '{print $1}')}"
CLUSTER_ID="${CLUSTER_ID:-mooncake}"
MC_LEASE_TTL="${MC_LEASE_TTL:-10000}"

# Default ETCD_ENDPOINTS to start_etcd_cluster.sh defaults
if [ -z "${ETCD_ENDPOINTS:-}" ]; then
  ETCD_ENDPOINTS="${MC_HOST_IP}:23800,${MC_HOST_IP}:23801,${MC_HOST_IP}:23802"
fi

if [ "${1:-}" = "--stop" ]; then
  for i in 0 1 2; do
    pidfile="$PID_DIR/master-${i}.pid"
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile")
      kill -TERM "$pid" 2>/dev/null && echo "stopped master-${i} (pid=$pid)"
      rm -f "$pidfile"
    fi
  done
  exit 0
fi

# Convert "ip:p,ip:p,ip:p" → "http://ip:p,http://ip:p,http://ip:p"
ETCD_URL_LIST="http://${ETCD_ENDPOINTS//,/ http://}"
ETCD_URL_LIST="${ETCD_URL_LIST// /,}"

echo ">>> starting 3 mooncake_master HA replicas on $MC_HOST_IP"
echo "    rpc ports:     $MASTER_RPC_BASE..$((MASTER_RPC_BASE+2))"
echo "    metrics ports: $MASTER_METRICS_BASE..$((MASTER_METRICS_BASE+2))"
echo "    etcd backend:  $ETCD_URL_LIST"
echo "    cluster_id:    $CLUSTER_ID"

for i in 0 1 2; do
  rpc=$((MASTER_RPC_BASE+i))
  metrics=$((MASTER_METRICS_BASE+i))
  nohup "$MASTER_BIN" \
    -enable_ha=true \
    -ha_backend_type=etcd \
    -ha_backend_connstring="$ETCD_URL_LIST" \
    -etcd_endpoints="$ETCD_URL_LIST" \
    -cluster_id="$CLUSTER_ID" \
    -rpc_address="$MC_HOST_IP" \
    -rpc_port=$rpc \
    -rpc_thread_num=4 \
    -enable_metric_reporting=true \
    -metrics_port=$metrics \
    -default_kv_lease_ttl=$MC_LEASE_TTL \
    -eviction_high_watermark_ratio=0.9 \
    -logtostderr \
    > "$LOG_DIR/master-${i}.log" 2>&1 &
  echo $! > "$PID_DIR/master-${i}.pid"
done

# Wait for one to win election
ETCDCTL="${ETCDCTL:-etcdctl}"
echo "    waiting for leader election..."
for _ in $(seq 60); do
  view=$("$ETCDCTL" --endpoints="$ETCD_ENDPOINTS" \
         get "mooncake-store/${CLUSTER_ID}/master_view" \
         --print-value-only 2>/dev/null | tr -d '[:space:]')
  if [ -n "$view" ] && [[ "$view" =~ : ]]; then
    echo "    ✓ master leader: $view"
    if [ "${1:-}" = "--bg" ]; then
      echo "    background mode — use 'bash start_mooncake_master_ha.sh --stop' to stop"
      exit 0
    fi
    echo "    foreground — Ctrl-C to stop all masters"
    trap 'for i in 0 1 2; do kill $(cat "$PID_DIR/master-${i}.pid") 2>/dev/null; rm -f "$PID_DIR/master-${i}.pid"; done' INT TERM
    wait
    exit 0
  fi
  sleep 0.5
done
echo "    ✗ no leader elected within 30s; check $LOG_DIR/master-*.log" >&2
for i in 0 1 2; do
  echo "    === master-$i tail ===" >&2
  tail -10 "$LOG_DIR/master-${i}.log" >&2
done
exit 1
