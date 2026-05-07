#!/bin/bash
# RDMA NIC Traffic Monitor
# Usage: ./rdma_monitor.sh [interval_seconds] [device_filter]
# Examples:
#   ./rdma_monitor.sh          # all active RDMA NICs, 1s interval
#   ./rdma_monitor.sh 2        # 2s interval
#   ./rdma_monitor.sh 1 mlx5_10  # only monitor mlx5_10

INTERVAL=${1:-1}
FILTER=${2:-}

COUNTER_BASE=/sys/class/infiniband

# Discover active RDMA device ports
discover_devices() {
    local devs=()
    for dev_path in "$COUNTER_BASE"/*/ports/*/state; do
        [[ -e "$dev_path" ]] || continue
        local state=$(cat "$dev_path" 2>/dev/null)
        if [[ "$state" == "4: ACTIVE" ]]; then
            local port_dir="${dev_path%/state}"
            local port="${port_dir##*/}"
            local ports_dir="${port_dir%/*}"
            local dev_dir="${ports_dir%/*}"
            local dev="${dev_dir##*/}"
            # Skip bond devices unless explicitly requested
            if [[ -z "$FILTER" && "$dev" == *bond* ]]; then
                continue
            fi
            if [[ -z "$FILTER" || "$dev" == *"$FILTER"* ]]; then
                devs+=("$dev:$port")
            fi
        fi
    done
    echo "${devs[@]}"
}

read_counter() {
    cat "$COUNTER_BASE/$1/ports/$2/counters/$3" 2>/dev/null || echo 0
}

format_rate() {
    local bits_per_sec=$1
    if (( $(echo "$bits_per_sec >= 1000000000" | bc -l) )); then
        printf "%8.2f Gbps" $(echo "scale=2; $bits_per_sec / 1000000000" | bc)
    elif (( $(echo "$bits_per_sec >= 1000000" | bc -l) )); then
        printf "%8.2f Mbps" $(echo "scale=2; $bits_per_sec / 1000000" | bc)
    elif (( $(echo "$bits_per_sec >= 1000" | bc -l) )); then
        printf "%8.2f Kbps" $(echo "scale=2; $bits_per_sec / 1000" | bc)
    else
        printf "%8.2f  bps" $bits_per_sec
    fi
}

format_pps() {
    local pps=$1
    if (( $(echo "$pps >= 1000000" | bc -l) )); then
        printf "%7.2f Mpps" $(echo "scale=2; $pps / 1000000" | bc)
    elif (( $(echo "$pps >= 1000" | bc -l) )); then
        printf "%7.2f Kpps" $(echo "scale=2; $pps / 1000" | bc)
    else
        printf "%7.0f  pps" $pps
    fi
}

DEVICES=($(discover_devices))
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "No active RDMA devices found."
    exit 1
fi

echo "Monitoring ${#DEVICES[@]} RDMA port(s): ${DEVICES[*]}"
echo "Sampling interval: ${INTERVAL}s  |  Press Ctrl+C to stop"
echo ""

# Declare associative arrays for previous counter values
declare -A PREV_RX_DATA PREV_TX_DATA PREV_RX_PKTS PREV_TX_PKTS

# Read initial counters
for entry in "${DEVICES[@]}"; do
    dev="${entry%%:*}"
    port="${entry#*:}"
    PREV_RX_DATA[$entry]=$(read_counter "$dev" "$port" port_rcv_data)
    PREV_TX_DATA[$entry]=$(read_counter "$dev" "$port" port_xmit_data)
    PREV_RX_PKTS[$entry]=$(read_counter "$dev" "$port" port_rcv_packets)
    PREV_TX_PKTS[$entry]=$(read_counter "$dev" "$port" port_xmit_packets)
done

sleep $INTERVAL

# Map device to netdev
declare -A DEV_NETDEV
for entry in "${DEVICES[@]}"; do
    dev="${entry%%:*}"
    port="${entry#*:}"
    DEV_NETDEV[$entry]=$(cat "$COUNTER_BASE/$dev/ports/$port/gid_attrs/ndevs/0" 2>/dev/null || echo "N/A")
done

# Header
SEP="------------------------------------------------------------------------------------------------------------"

while true; do
    TOTAL_RX=0
    TOTAL_TX=0

    printf "\033[2J\033[H"
    echo "RDMA Traffic Monitor  |  $(date '+%Y-%m-%d %H:%M:%S')  |  Interval: ${INTERVAL}s"
    echo "$SEP"
    printf "%-12s %-8s %16s %16s %13s %13s\n" \
        "Device" "NetDev" "RX Rate" "TX Rate" "RX PPS" "TX PPS"
    echo "$SEP"

    for entry in "${DEVICES[@]}"; do
        dev="${entry%%:*}"
        port="${entry#*:}"
        cur_rx_data=$(read_counter "$dev" "$port" port_rcv_data)
        cur_tx_data=$(read_counter "$dev" "$port" port_xmit_data)
        cur_rx_pkts=$(read_counter "$dev" "$port" port_rcv_packets)
        cur_tx_pkts=$(read_counter "$dev" "$port" port_xmit_packets)

        # port_{rcv,xmit}_data counters are reported in 4-byte words.
        delta_rx_data=$(( cur_rx_data - PREV_RX_DATA[$entry] ))
        delta_tx_data=$(( cur_tx_data - PREV_TX_DATA[$entry] ))
        delta_rx_pkts=$(( cur_rx_pkts - PREV_RX_PKTS[$entry] ))
        delta_tx_pkts=$(( cur_tx_pkts - PREV_TX_PKTS[$entry] ))

        # Convert 4-byte words to bits per second.
        rx_bps=$(echo "scale=2; $delta_rx_data * 32 / $INTERVAL" | bc)
        tx_bps=$(echo "scale=2; $delta_tx_data * 32 / $INTERVAL" | bc)
        rx_pps=$(echo "scale=2; $delta_rx_pkts / $INTERVAL" | bc)
        tx_pps=$(echo "scale=2; $delta_tx_pkts / $INTERVAL" | bc)

        TOTAL_RX=$(echo "$TOTAL_RX + $rx_bps" | bc)
        TOTAL_TX=$(echo "$TOTAL_TX + $tx_bps" | bc)

        printf "%-12s %-8s %16s %16s %13s %13s\n" \
            "$entry" "${DEV_NETDEV[$entry]}" \
            "$(format_rate $rx_bps)" "$(format_rate $tx_bps)" \
            "$(format_pps $rx_pps)" "$(format_pps $tx_pps)"

        PREV_RX_DATA[$entry]=$cur_rx_data
        PREV_TX_DATA[$entry]=$cur_tx_data
        PREV_RX_PKTS[$entry]=$cur_rx_pkts
        PREV_TX_PKTS[$entry]=$cur_tx_pkts
    done

    echo "$SEP"
    printf "%-12s %-8s %16s %16s\n" \
        "TOTAL" "" \
        "$(format_rate $TOTAL_RX)" "$(format_rate $TOTAL_TX)"
    echo "$SEP"

    sleep $INTERVAL
done
