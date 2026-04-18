# Prometheus/Grafana for Mooncake Debugging

This is a minimal Prometheus + Grafana stack for P/D benchmark debugging. It
adds:

- file-based scrape target lists under `targets/*.yml`
- one provisioned Prometheus datasource
- one provisioned dashboard: `Mooncake Debug Overview`
- one self-check script: `check_stack.sh`

## 1. Update Targets

Before each benchmark run, edit:

- `targets/vllm-prefill.yml`
- `targets/vllm-decode.yml`
- `targets/mooncake-master.yml`

The checked-in defaults are intentionally empty so old IPs do not silently stay
in effect.

## 2. Start the Stack

```bash
docker compose up -d --force-recreate
```

If your Docker setup needs it, run the same command with `sudo`.

Use `--force-recreate` whenever datasource or provisioning files change.

## 3. Run the Self-check

```bash
./check_stack.sh
```

This verifies:

- Grafana health
- Prometheus health
- the provisioned `Prometheus` datasource
- the provisioned `Mooncake Debug Overview` dashboard
- the Prometheus target API

## 4. Open Grafana

Open [`http://127.0.0.1:3000`](http://127.0.0.1:3000).

- username: `admin`
- password: `admin`
- anonymous viewer access is also enabled

The home dashboard is `Mooncake Debug Overview`.

## 5. What the Dashboard Covers

The dashboard is organized around the failure chain from this incident:

- stack baseline: datasource and scrape target health
- vLLM failure counters: `failed_batches`, `failed_keys`,
  `transfer_fail_keys`, `no_available_handle_keys`, `other_failed_keys`
- business impact: TTFT, E2E, request rate, token throughput
- triage runbook: embedded `rg` commands for handshake / transfer / batch logs

## 6. Troubleshooting

- If Grafana is up but the dashboard is missing, recreate the stack and rerun
  `./check_stack.sh`.
- If panels are empty, check `targets/*.yml` first, then inspect
  `http://127.0.0.1:9090/api/v1/targets`.
