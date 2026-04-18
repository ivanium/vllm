#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://127.0.0.1:9090}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARN: $*" >&2
}

check_http() {
  local url="$1"
  local label="$2"
  if ! curl -fsS "$url" >/dev/null; then
    fail "$label is unavailable at $url"
  fi
  echo "OK: $label"
}

check_http "${GRAFANA_URL}/api/health" "Grafana health endpoint"
check_http "${PROMETHEUS_URL}/-/healthy" "Prometheus health endpoint"

datasources_json="$(curl -fsS "${GRAFANA_URL}/api/datasources")"
targets_json="$(curl -fsS "${PROMETHEUS_URL}/api/v1/targets")"
dashboards_json="$(curl -fsS "${GRAFANA_URL}/api/search")"

echo "$datasources_json" | grep -q '"name":"Prometheus"' \
  || fail "Grafana datasource provisioning did not expose the Prometheus datasource"
echo "OK: Grafana datasource provisioning"

echo "$dashboards_json" | grep -q '"title":"Mooncake Debug Overview"' \
  || fail "Mooncake dashboard was not provisioned into Grafana"
echo "OK: Grafana dashboard provisioning"

echo "$targets_json" | grep -q '"job":"prometheus"' \
  || fail "Prometheus target API did not report the self-scrape job"
echo "OK: Prometheus target API"

for target_file in "$ROOT_DIR"/targets/*.yml; do
  if grep -Eq 'targets:[[:space:]]*\[\]' "$target_file"; then
    warn "Target file still empty: ${target_file#$ROOT_DIR/}"
  fi
done

if echo "$targets_json" | grep -q '"health":"down"'; then
  warn "One or more configured scrape targets are down. Inspect ${PROMETHEUS_URL}/api/v1/targets"
else
  echo "OK: configured scrape targets are up"
fi

echo "Stack self-check completed."
