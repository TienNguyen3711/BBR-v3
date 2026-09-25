"""Executed only inside the disposable QEMU guest, on loopback TCP."""
import json, math, os, select, socket, statistics, struct, subprocess, threading, time
from pathlib import Path

CTL = Path('/sys/kernel/debug/tcp_bbr_qrl')
RESULTS = []

def command(value):
    (CTL/'pacing_gain_override').write_text(str(value)+'\n')

def status():
    return {k:int(v) for k,v in (x.split('=') for x in (CTL/'status').read_text().split())}

def check(condition, message):
    if not condition: raise AssertionError(message)

def parser_checks():
    for text,expected in [('0.75',.75),('0.9',230/256),('1',1.),('1.1',282/256),('1.25',1.25),('-1',-1.)]:
        command(text)
        check(abs(float((CTL/'pacing_gain_override').read_text())-expected)<1e-6, 'gain readback '+text)
    for text in ['1.05','1.125','-2','-1.1','-1garbage','1.1000000','1.1\x00junk','1'*32]:
        command('1')
        try: command(text)
        except OSError: pass
        else: raise AssertionError('accepted undeclared command '+repr(text))
        check(float((CTL/'pacing_gain_override').read_text())==1., 'invalid write modified state')
    command('-1')
    print('CHECK parser PASS', flush=True)

def flow(cc, label, gain='-1', mode='normal', duration=8.):
    listener=socket.socket()
    listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b'cubic')
    listener.setsockopt(socket.SOL_SOCKET,33,4*1024*1024)
    listener.bind(('127.0.0.1',0)); listener.listen(1)
    received={}
    def receiver():
        peer,_=listener.accept()
        start=time.monotonic(); total=0
        while True:
            data=peer.recv(131072)
            if not data: break
            total+=len(data)
        received.update(bytes=total, seconds=time.monotonic()-start)
        peer.close()
    worker=threading.Thread(target=receiver);worker.start()
    client=socket.socket()
    client.setsockopt(socket.IPPROTO_TCP,socket.TCP_CONGESTION,cc.encode())
    client.setsockopt(socket.SOL_SOCKET,32,4*1024*1024)
    client.setsockopt(socket.IPPROTO_TCP,socket.TCP_MAXSEG,1448)
    client.connect(listener.getsockname());client.setblocking(False)
    check(client.getsockopt(socket.IPPROTO_TCP,socket.TCP_CONGESTION,32).split(b'\x00')[0].decode()==cc, 'wrong congestion control')
    if cc=='bbr_qrl' and mode!='multi':
        unopened=socket.socket()
        unopened.setsockopt(socket.IPPROTO_TCP,socket.TCP_CONGESTION,b'bbr_qrl')
        unopened.close()
        check(status()['active']==1,'uninitialized release detached live owner')
    start=time.monotonic();next_sample=start;next_command=start
    before=status();samples=[];rtts=[];seen_override=False;switched=False;switch_entry=None;switch_time=None
    payload=b'x'*16384
    while time.monotonic()-start<duration:
        now=time.monotonic()
        if now>=next_command:
            if not switched or mode=='normal': command(gain)
            elif mode=='clear': command('-1')
            elif mode=='midcruise': command('0.75')
            next_command=now+.1
        if now>=next_sample:
            s=status();s['time_s']=now-start;samples.append(s)
            info=client.getsockopt(socket.IPPROTO_TCP,socket.TCP_INFO,256)
            rtts.append(struct.unpack_from('I',info,68)[0]/1000.)
            if s['eligible'] and s['applied']==320 and s['latched']==320: seen_override=True
            if mode in ('clear','expire','midcruise') and not switched and now-start>2 and s['eligible'] and s['latched']==320:
                switched=True;switch_entry=s['entries'];switch_time=now
                if mode=='clear': command('-1')
                elif mode=='midcruise': command('0.75')
            if s['active']==1:
                check(s['applied']==s['native'] or (s['phase']==6 and s['eligible']), 'override outside eligible CRUISE')
                if s['phase'] in (4,5,6,7):
                    check(s['native']=={4:320,5:232,6:256,7:256}[s['phase']], 'native phase gain modified')
                if switched and mode=='midcruise' and s['entries']==switch_entry:
                    check(s['latched']!=192, 'command changed before a new CRUISE entry')
                if switched and mode in ('clear','expire') and now-switch_time>(.2 if mode=='clear' else 1.2):
                    check(s['applied']==s['native'], 'override did not cancel after clear/expiry')
                    if mode=='expire': check(float((CTL/'pacing_gain_override').read_text())==-1., 'lease did not expire')
            next_sample=now+.02
        _,w,_=select.select([], [client], [], .002)
        if w:
            try: client.send(payload)
            except BlockingIOError: pass
    client.setblocking(True);client.shutdown(socket.SHUT_WR)
    deadline=time.monotonic()+10
    while worker.is_alive() and time.monotonic()<deadline:
        if mode=='normal': command(gain)
        elif mode=='midcruise': command('0.75' if switched else gain)
        worker.join(timeout=.1)
    check(not worker.is_alive(), 'receiver did not finish')
    info=client.getsockopt(socket.IPPROTO_TCP,socket.TCP_INFO,256)
    retrans=struct.unpack_from('I',info,100)[0]
    client.close();listener.close();command('-1');time.sleep(.1)
    after=status()
    if label=='loss_safety':
        check(retrans>0, 'loss test did not trigger retransmissions')
        check(after['guarded_cruise_acks']>before['guarded_cruise_acks'], 'loss guard had no CRUISE coverage')
    print('FLOW_STATUS '+json.dumps({'label':label,'before':before,'after':after,'observed':sorted({(s['active'],s['phase'],s['eligible'],s['requested'],s['latched'],s['applied']) for s in samples})}),flush=True)
    if cc=='bbr_qrl' and gain=='-1':
        check(all(s['applied']==s['native'] for s in samples), 'disabled is not native')
    if cc=='bbr_qrl' and mode=='normal' and gain!='-1':
        expected={'0.75':192,'0.9':230,'1':256,'1.1':282,'1.25':320}[gain]
        check(any(s['eligible'] and s['latched']==expected and s['applied']==expected for s in samples), 'gain never applied '+gain)
    if mode in ('clear','expire','midcruise'):
        check(seen_override and switched, 'cancellation/entry test did not activate')
        if mode=='midcruise': check(any(s['latched']==192 and s['entries']>switch_entry for s in samples), 'new entry never latched command')
        if mode=='expire': check(float((CTL/'pacing_gain_override').read_text())==-1., 'lease readback did not expire')
    sorted_rtt=sorted(rtts)
    result=dict(label=label,cc=cc,gain=gain,mode=mode,received_bytes=received['bytes'],
        receiver_mbps=received['bytes']*8/received['seconds']/1e6,
        rtt_median_ms=statistics.median(rtts),rtt_p90_ms=sorted_rtt[math.ceil(.9*len(rtts))-1],
        retransmits=retrans, samples=samples, before=before,after=after)
    RESULTS.append(result)
    print('FLOW '+json.dumps({k:v for k,v in result.items() if k not in ('samples','before','after')}),flush=True)
    return result

def main():
    parser_checks()
    # TCP can call release without having called init.
    unopened=socket.socket()
    unopened.setsockopt(socket.IPPROTO_TCP,socket.TCP_CONGESTION,b'bbr_qrl')
    unopened.close()
    check(status()['active']==0,'unconnected socket leaked ownership')
    for repeat in range(3):
        # Rotate order to reduce monotonic guest-load/time bias.
        arms=[('bbr','stock','-1'),('bbr_qrl','disabled','-1'),('bbr_qrl','gain1','1')]
        arms=arms[repeat:]+arms[:repeat]
        for cc,label,gain in arms: flow(cc,label+str(repeat),gain,duration=6.)
    for gain in ('0.75','0.9','1.1','1.25'):
        flow('bbr_qrl','gain_'+gain,gain)
    for mode in ('clear','expire','midcruise'):
        flow('bbr_qrl',mode,'1.25',mode,duration=10.)
    subprocess.run(['/usr/sbin/tc','qdisc','change','dev','lo','root','netem','delay','10ms','rate','20mbit','loss','1%','limit','1000'],check=True)
    flow('bbr_qrl','loss_safety','1.25',duration=10.)
    subprocess.run(['/usr/sbin/tc','qdisc','change','dev','lo','root','netem','delay','10ms','rate','20mbit','loss','0%','limit','1000'],check=True)
    errors=[]
    def concurrent(index):
        try: flow('bbr_qrl','multiflow'+str(index),'1.25','multi',duration=6.)
        except Exception as exc: errors.append(repr(exc))
    threads=[threading.Thread(target=concurrent,args=(i,)) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    check(not errors,'multiflow thread failed: '+str(errors))
    observed=[s for r in RESULTS if r['mode']=='multi' for s in r['samples'] if s['active']==2]
    check(len(observed)>20,'multiflow test had no concurrent coverage')
    check(all(s['applied']==s['native'] for s in observed),'multiflow override did not fail closed')
    check(status()['active']==0,'socket ownership leaked after release')
    check(all(status()[p]>0 for p in ('startup','drain','probe_rtt','up','down','cruise','refill')), 'missing native phase coverage')
    report={'scope':'isolated QEMU Linux, shaped loopback; not Starlink performance evidence',
            'kernel':os.uname().release,'checks_passed':True,'flows':RESULTS,'final_status':status()}
    Path('/audit.json').write_text(json.dumps(report))
    import base64
    print('AUDIT_JSON_BASE64 '+base64.b64encode(json.dumps(report).encode()).decode(),flush=True)
    print('QBBR_KERNEL_AUDIT_PASS',flush=True)

try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    print('QBBR_KERNEL_AUDIT_FAIL',flush=True)
    raise
