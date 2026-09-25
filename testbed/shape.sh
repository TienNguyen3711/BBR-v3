#!/bin/sh
# Drive a real-Linux bottleneck through a recorded Starlink capacity schedule.
#
# TOPOLOGY -- arrived at by measurement, each alternative failed a specific way:
#   receiver egress : netem delay = FULL RTT_min, fixed, never modified
#   sender egress   : htb (rate, changed every step)
#                       -> bfifo limit in BYTES = rate x MAX_QUEUE_MS
#   sender offloads : GSO/TSO/GRO off
#
# Why each piece:
#  * All propagation delay on the ACK path. Putting half on each side meant a
#    netem on the sender that had to be re-limited every second, and
#    `tc qdisc change` on netem flushes its delay line: London stalled to
#    0 Mbps in 21 of 60 seconds. A fixed delay on the ACK path is harmless to
#    BBR -- it shifts every ACK equally, so min_rtt is correct and the
#    delivery-rate intervals BBR measures are undistorted.
#  * An earlier version put delay on the data path only and none on the ACKs,
#    so the round trip was RTT_min/2 and every RTT number was wrong.
#  * The queue is bounded in TIME, matching the simulator, whose queueing delay
#    is hard-capped by probe_max_queue_delay_ms (its measured per-step maximum
#    equals that parameter in all six cities). A byte buffer sized at median
#    capacity turned into seconds of delay inside capacity dips.
#  * bfifo, not netem, carries that bound: changing a bfifo limit updates the
#    limit without flushing the queue.
#  * GSO off: with it on, skbs were ~17 KB aggregates, so a packet-count limit
#    never engaged and the "bounded" queue was unbounded in bytes.
#  * htb burst must be >= rate/HZ bytes (rate in kbit, burst in KB); a burst off
#    by 1000x made tc reject every change and, with errors suppressed, pinned
#    the link at its placeholder rate.
set -e
DEV=${DEV:-eth0}
SCHEDULE=${SCHEDULE:-/schedule.txt}
MAX_QUEUE_MS=${MAX_QUEUE_MS:-25}

ethtool -K "$DEV" gso off tso off gro off >/dev/null 2>&1 || echo "[shape] WARN: could not disable offloads" >&2

tc qdisc del dev "$DEV" root 2>/dev/null || true
tc qdisc add dev "$DEV" root handle 1: htb default 10
tc class add dev "$DEV" parent 1: classid 1:10 htb rate 100mbit ceil 100mbit burst 512k
tc qdisc add dev "$DEV" parent 1:10 handle 10: bfifo limit 1000000

echo "[shape] dev=$DEV max_queue=${MAX_QUEUE_MS}ms schedule=$SCHEDULE"
START=$(date +%s)
while read -r T RATE; do
    [ -z "$T" ] && continue
    NOW=$(( $(date +%s) - START ))
    WAIT=$(awk "BEGIN{d=$T-$NOW; if(d<0)d=0; printf \"%.2f\", d}")
    sleep "$WAIT"
    KBIT=$(awk "BEGIN{printf \"%d\", $RATE/1000}")
    [ "$KBIT" -lt 64 ] && KBIT=64
    BURST=$(awk "BEGIN{b=$KBIT/400; if(b<256)b=256; printf \"%d\", b}")
    if ! tc class change dev "$DEV" parent 1: classid 1:10 \
            htb rate "${KBIT}kbit" ceil "${KBIT}kbit" burst "${BURST}k"; then
        echo "[shape] FAILED rate at t=$T rate=${KBIT}kbit" >&2
    fi
    # Queue bound in bytes = rate x max queueing delay, floored at 2 MTU.
    LIMIT=$(awk "BEGIN{b=($RATE/8)*($MAX_QUEUE_MS/1000); if(b<3000)b=3000; printf \"%d\", b}")
    if ! tc qdisc change dev "$DEV" parent 1:10 handle 10: bfifo limit "$LIMIT"; then
        echo "[shape] FAILED bfifo at t=$T limit=$LIMIT" >&2
    fi
done < "$SCHEDULE"
echo "[shape] schedule exhausted"
