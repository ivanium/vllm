#!/usr/bin/env bash
# Start a local 3-node etcd cluster for Mooncake HA testing.
#
# This is a single-host development helper. For production, use the
# Kubernetes StatefulSet manifest in scripts/mooncake/ha-etcd.yaml so
# the 3 etcd replicas land on 3 different physical hosts (podAntiAffinity).
#
# Usage:
#   bash start_etcd_cluster.sh                  # foreground (Ctrl-C stops all)
#   bash start_etcd_cluster.sh --bg             # background
#   bash start_etcd_cluster.sh --stop           # kill backgrounded cluster
#
# Environment variables (all optional):
#   ETCD_BIN          - path to etcd binary    (default: etcd from PATH)
#   ETCD_CLIENT_BASE  - first client port      (default: 23800; nodes use BASE,+1,+2)
#   ETCD_PEER_BASE    - first peer port        (default: 23900)
#   ETCD_DATA_DIR     - data dir parent        (default: $SCRIPT_DIR/etcd_data)
#   ETCD_HOST_IP      - advertised IP          (default: hostname -I first IP)
#   ETCD_CLUSTER_TOKEN - cluster bootstrap token (default: mooncake-ha-cluster)
#
# After this exits successfully, the etcd cluster is reachable at:
#   http://$ETCD_HOST_IP:$ETCD_CLIENT_BASE        (etcd-0)
#   http://$ETCD_HOST_IP:$((ETCD_CLIENT_BASE+1))  (etcd-1)
#   http://$ETCD_HOST_IP:$((ETCD_CLIENT_BASE+2))  (etcd-2)
#
# Verify with:
#   etcdctl --endpoints=$ETCD_HOST_IP:23800,$ETCD_HOST_IP:23801,$ETCD_HOST_IP:23802 \
#       endpoint status -w table

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.etcd_pids"
mkdir -p "$PID_DIR"

ETCD_BIN="${ETCD_BIN:-etcd}"
ETCD_CLIENT_BASE="${ETCD_CLIENT_BASE:-23800}"
ETCD_PEER_BASE="${ETCD_PEER_BASE:-23900}"
ETCD_DATA_DIR="${ETCD_DATA_DIR:-$SCRIPT_DIR/etcd_data}"
ETCD_HOST_IP="${ETCD_HOST_IP:-$(hostname -I | awk '{print $1}')}"
ETCD_CLUSTER_TOKEN="${ETCD_CLUSTER_TOKEN:-mooncake-ha-cluster}"

if [ "${1:-}" = "--stop" ]; then
  for i in 0 1 2; do
    pidfile="$PID_DIR/etcd-${i}.pid"
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile")
      kill -TERM "$pid" 2>/dev/null && echo "stopped etcd-${i} (pid=$pid)"
      rm -f "$pidfile"
    fi
  done
  exit 0
fi

# Build initial-cluster string
INIT_CLUSTER=""
for i in 0 1 2; do
  [ -n "$INIT_CLUSTER" ] && INIT_CLUSTER+=","
  INIT_CLUSTER+="etcd-$i=http://${ETCD_HOST_IP}:$((ETCD_PEER_BASE+i))"
done

mkdir -p "$ETCD_DATA_DIR"

echo ">>> starting 3-node etcd cluster on $ETCD_HOST_IP"
echo "    client ports: $ETCD_CLIENT_BASE..$((ETCD_CLIENT_BASE+2))"
echo "    peer ports:   $ETCD_PEER_BASE..$((ETCD_PEER_BASE+2))"
echo "    data dir:     $ETCD_DATA_DIR/etcd-{0,1,2}_data"

for i in 0 1 2; do
  cport=$((ETCD_CLIENT_BASE+i))
  pport=$((ETCD_PEER_BASE+i))
  rm -rf "$ETCD_DATA_DIR/etcd-${i}_data"
  nohup "$ETCD_BIN" \
    --name "etcd-$i" \
    --data-dir "$ETCD_DATA_DIR/etcd-${i}_data" \
    --listen-client-urls "http://0.0.0.0:$cport" \
    --advertise-client-urls "http://${ETCD_HOST_IP}:$cport" \
    --listen-peer-urls "http://0.0.0.0:$pport" \
    --initial-advertise-peer-urls "http://${ETCD_HOST_IP}:$pport" \
    --initial-cluster "$INIT_CLUSTER" \
    --initial-cluster-token "$ETCD_CLUSTER_TOKEN" \
    --initial-cluster-state new \
    --log-outputs stderr \
    > "$ETCD_DATA_DIR/etcd-${i}.log" 2>&1 &
  echo $! > "$PID_DIR/etcd-${i}.pid"
done

# Wait for cluster to elect leader (via etcdctl endpoint status)
ENDPOINTS=""
for i in 0 1 2; do
  [ -n "$ENDPOINTS" ] && ENDPOINTS+=","
  ENDPOINTS+="${ETCD_HOST_IP}:$((ETCD_CLIENT_BASE+i))"
done
ETCDCTL="${ETCDCTL:-etcdctl}"
echo "    waiting for cluster to converge..."
for _ in $(seq 60); do
  out=$("$ETCDCTL" --endpoints="$ENDPOINTS" endpoint status 2>/dev/null)
  if [ -n "$out" ] && [ "$(echo "$out" | wc -l)" = "3" ]; then
    echo "    ✓ etcd cluster ready: $ENDPOINTS"
    if [ "${1:-}" = "--bg" ]; then
      echo "    background mode — use 'bash start_etcd_cluster.sh --stop' to stop"
      exit 0
    fi
    echo "    foreground — Ctrl-C to stop"
    trap 'for i in 0 1 2; do kill $(cat "$PID_DIR/etcd-${i}.pid") 2>/dev/null; rm -f "$PID_DIR/etcd-${i}.pid"; done' INT TERM
    wait
    exit 0
  fi
  sleep 0.5
done
echo "    ✗ cluster did not converge in 30s; check $ETCD_DATA_DIR/etcd-*.log" >&2
exit 1
