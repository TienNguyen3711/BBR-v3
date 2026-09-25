import struct
import time
import threading
import pytest
from qbbr.action import bbr_hook
from qbbr.action.heartbeat import CommandHeartbeat
from qbbr.env.testbed_env import TestbedEnv
from qbbr.field.tcp_info import read_tcp_info, require_measurement_fields, retransmit_delta
from qbbr.field.telemetry import LinuxTcpInfoSource

class Socket:
    def __init__(self):
        self.raw=bytearray(232)
        for offset,value in [(16,1400),(24,10),(68,50000),(80,100),(100,7),(148,20000)]:
            struct.pack_into('=I',self.raw,offset,value)
        struct.pack_into('=Q',self.raw,120,2**35)
        struct.pack_into('=Q',self.raw,160,1_000_000)
    def getsockopt(self,*args): return bytes(self.raw[:args[-1]])


def test_linux_offsets_and_u64_counters():
    info=read_tcp_info(Socket())
    assert info['bytes_acked']==2**35
    assert info['delivery_rate']==1_000_000
    assert info['total_retrans']==7
    assert info['rtt']==50000 and info['min_rtt']==20000
    assert retransmit_delta(1,2**32-2)==3
    with pytest.raises(RuntimeError): require_measurement_fields({'rtt':1})


def test_interval_measurements_ignore_cwnd_and_old_retransmissions(monkeypatch):
    monkeypatch.setattr(bbr_hook,'clear_pacing_gain_override',lambda:None)
    env=TestbedEnv(Socket(),{'B_max_mbps':100,'RTT_min_ms':20,'RTT_max_ms':100})
    env._episode_start_s=10
    info=read_tcp_info(env.sock)
    initial=env._telemetry_row(10,info)
    assert initial['bits_per_second']==0 and initial['retransmits']==0
    newer=dict(info,bytes_acked=info['bytes_acked']+1_000_000,snd_cwnd=99999,total_retrans=9)
    row=env._telemetry_row(12,newer)
    assert row['bits_per_second']==4_000_000 and row['retransmits']==2
    row=env._telemetry_row(14,newer)
    assert row['bits_per_second']==0 and row['retransmits']==0
    assert row['rtt_base_ms']==20 and row['b_hat_mbps']==8
    assert row['v_over_bdp']==pytest.approx(14000/20000)
    assert row['q_packets']==0


def test_field_source_uses_same_decoder_and_interval_deltas():
    sock=Socket();source=LinuxTcpInfoSource(sock)
    assert source.sample().delivered_bytes==0
    struct.pack_into('=Q',sock.raw,120,2**35+4321)
    struct.pack_into('=I',sock.raw,100,9)
    sample=source.sample()
    assert sample.delivered_bytes==4321 and sample.retransmits==2
    assert sample.delivery_rate_bps==8_000_000
    assert sample.inflight_bytes==14000


def test_heartbeat_refresh_watchdog_and_close(monkeypatch):
    writes=[];expired=threading.Event()
    def write(gain):
        writes.append(gain)
        if gain is None: expired.set()
    monkeypatch.setattr(bbr_hook,'set_pacing_gain',write)
    monkeypatch.setattr(bbr_hook,'clear_pacing_gain_override',lambda:write(None))
    hb=CommandHeartbeat(.01)
    try:
        hb.submit(1.1,.08)
        assert expired.wait(2), 'watchdog did not clear'
        assert writes.count(1.1)>1
        hb.check()
    finally: hb.close()
    count=len(writes);time.sleep(.03)
    assert len(writes)==count and writes[-1] is None


def test_heartbeat_failure_surfaces(monkeypatch):
    calls=[]
    def write(gain):
        calls.append(gain)
        if len(calls)>1: raise OSError('unmounted debugfs')
    monkeypatch.setattr(bbr_hook,'set_pacing_gain',write)
    monkeypatch.setattr(bbr_hook,'clear_pacing_gain_override',lambda:None)
    hb=CommandHeartbeat(.01)
    try:
        hb.submit(.9,1)
        assert hb._stop.wait(2)
        with pytest.raises(RuntimeError,match='heartbeat failed'): hb.check()
    finally: hb.close()


def test_step_queues_before_cruise_and_clears_on_done_or_error(monkeypatch):
    import qbbr.env.testbed_env as module
    clock=[0.0];writes=[]
    class Heartbeat:
        def submit(self,gain,valid_for_s): writes.append(gain)
        def check(self): pass
        def close(self): writes.append(None)
    monkeypatch.setattr(module,'CommandHeartbeat',Heartbeat)
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    monkeypatch.setattr(bbr_hook,'clear_pacing_gain_override',lambda:None)
    monkeypatch.setattr(bbr_hook,'in_probe_bw_cruise',lambda:False)
    sock=Socket()
    env=TestbedEnv(sock,{'B_max_mbps':100,'RTT_min_ms':20,'RTT_max_ms':100},episode_s=.3,decision_interval_s=.1,max_consecutive_high_loss_intervals=1)
    env.reset()
    struct.pack_into('=I',sock.raw,100,8)
    _,_,done,info=env.step(4)
    assert writes==[1.25,None] and not done
    assert info['telemetry']['retransmits']==1
    env.step(4)
    assert env._consecutive_high_loss==0  # old cumulative loss no longer sticks
    _,_,done,_=env.step(4)
    assert done and writes[-1] is None and env._heartbeat is None
    env.reset()
    with pytest.raises((IndexError,ValueError)): env.step(999)
    assert env._heartbeat is None
