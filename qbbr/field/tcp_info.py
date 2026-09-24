"""Linux UAPI tcp_info: decode by byte offset, checking returned length."""
import socket
import struct

# struct tcp_info's historically-stable prefix (see module docstring); native
# byte order with standard field sizes ("="); offsets follow the Linux UAPI.
_TCP_INFO_FORMAT = "=BBBBBBBxIIIIIIIIIIIIIIIIIIIIIIII"  # 7 uint8 + 1 pad + 24 uint32
_TCP_INFO_SIZE = struct.calcsize(_TCP_INFO_FORMAT)
_TCP_INFO_FIELDS = [
    "state", "ca_state", "retransmits", "probes", "backoff", "options", "wscale",
    "rto", "ato", "snd_mss", "rcv_mss", "unacked", "sacked", "lost", "retrans",
    "fackets", "last_data_sent", "last_ack_sent", "last_data_recv", "last_ack_recv",
    "pmtu", "rcv_ssthresh", "rtt", "rttvar", "snd_ssthresh", "snd_cwnd", "advmss",
    "reordering", "rcv_rtt", "rcv_space", "total_retrans",
]
_SOL_TCP = getattr(socket, "IPPROTO_TCP", 6)
_TCP_INFO_OPT = getattr(socket, "TCP_INFO", 11)  # Linux-only; absent on macOS/BSD


def read_tcp_info(sock: socket.socket) -> dict[str, int]:
    raw = sock.getsockopt(_SOL_TCP, _TCP_INFO_OPT, 232)
    if len(raw) < _TCP_INFO_SIZE:
        raise RuntimeError("Truncated Linux TCP_INFO prefix")
    values = struct.unpack(_TCP_INFO_FORMAT, raw[:_TCP_INFO_SIZE])
    result = dict(zip(_TCP_INFO_FIELDS, values))
    for name, offset, fmt in [('bytes_acked',120,'Q'), ('bytes_received',128,'Q'),
                             ('notsent_bytes',144,'I'), ('min_rtt',148,'I'),
                             ('delivery_rate',160,'Q')]:
        if len(raw) >= offset + struct.calcsize(fmt):
            result[name] = struct.unpack_from('='+fmt, raw, offset)[0]
    return result


def require_measurement_fields(info):
    missing = {'bytes_acked', 'min_rtt', 'delivery_rate'} - info.keys()
    if missing:
        raise RuntimeError('Linux TCP_INFO lacks required measurement fields: '+', '.join(sorted(missing)))


def retransmit_delta(current, previous):
    # tcpi_total_retrans is u32; counters belong to the same socket.
    return (current - previous) % (1 << 32)


