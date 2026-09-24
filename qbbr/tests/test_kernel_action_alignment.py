from dataclasses import replace
import ctypes
from pathlib import Path
import subprocess
import sys
import pytest
from qbbr.action.kernel_contract import CommandState, GAIN_UNITS
from qbbr.env.fluid_sim import FluidParams, FluidState, step_fluid_state


def test_contract_against_actual_kernel_apply_function(tmp_path):
    # Compile the actual controller transition functions with only Linux
    # locking/time primitives stubbed. No Python mirror supplies the oracle.
    source = (Path(__file__).resolve().parents[2]/'kernel/bbr_qrl/bbr_qrl_ctl.c').read_text()
    body = source[source.index('static DEFINE_SPINLOCK'):source.index('static ssize_t gain_write')]
    wrapper = '''#include <stdint.h>
#include <stddef.h>
typedef uint64_t u64;
struct sock { int unused; }; struct dentry;
#define DEFINE_SPINLOCK(x) int x
#define spin_lock_bh(x) ((void)0)
#define spin_unlock_bh(x) ((void)0)
#define HZ 1000
#define time_after_eq(a,b) ((long)((a)-(b)) >= 0)
static unsigned long jiffies;
'''+body+'''
static struct sock test_sock;
void start(void) { bbr_qrl_attach(&test_sock); }
void write_gain(int gain, int ms) { requested=gain; refreshed=ms; }
int tick(int ms,int phase,int eligible,int native_gain) {
 jiffies=ms; return bbr_qrl_apply(&test_sock,phase,eligible,native_gain);
}
int latch(void) { return latched; }
'''
    (tmp_path/'ctl.c').write_text(wrapper)
    lib = tmp_path/('ctl.dylib' if sys.platform=='darwin' else 'ctl.so')
    subprocess.run(['cc','-shared','-fPIC',str(tmp_path/'ctl.c'),'-o',str(lib)],check=True)
    c=ctypes.CDLL(str(lib)); c.start()
    state=CommandState()
    events=[(0,4,False,1.1),(50,6,True,None),(100,6,True,.75),
            (200,6,False,None),(300,6,True,None),(350,7,False,None),
            (400,6,True,None),(1100,6,True,None),(1200,6,True,1.25),
            (1300,5,False,None),(1400,6,True,None),(1500,6,True,-1)]
    for ms,phase,eligible,write in events:
        if write is not None:
            state=state.write(write,ms/1000)
            c.write_gain(state.requested,ms)
        native={4:320,5:232,6:256,7:256}[phase]
        state=state.observe(ms/1000,cruise=phase==6,eligible=eligible,native=native)
        assert state.applied==c.tick(ms,phase,eligible,native)
        assert state.latched==c.latch()


@pytest.mark.parametrize('gain,units',GAIN_UNITS.items())
def test_quantization_and_mid_cruise_latching(gain,units):
    s=CommandState().write(gain,0).observe(0,cruise=True,eligible=True,native=256)
    assert s.applied==units
    s=s.write(.75,.1).observe(.1,cruise=True,eligible=True,native=256)
    assert s.applied==units
    s=s.observe(.2,cruise=False,eligible=False,native=232)
    assert s.applied==232
    assert s.observe(.3,cruise=True,eligible=True,native=256).applied==192


def test_expiry_clear_and_recovery_do_not_relatch_mid_cruise():
    s=CommandState().write(1.25,0).observe(0,cruise=True,eligible=True,native=256)
    assert s.observe(1,cruise=True,eligible=True,native=256).applied==256
    assert s.write(None,.1).observe(.1,cruise=True,eligible=True,native=256).applied==256
    s=s.observe(.1,cruise=True,eligible=False,native=256)
    assert s.observe(.2,cruise=True,eligible=True,native=256).applied==256


def test_fluid_uses_latched_gain_and_native_down():
    p=FluidParams(x_btl_bps=1e6,rtt_rtp_s=.05,base_retransmit_rate_pps=0,dwn_retransmit_rate_pps=0,consistent_transport=True,probe_bw_cycle=True,native_cruise_override=True,kernel_action_semantics=True)
    s=FluidState()
    s,_,_=step_fluid_state(s,1.1,.01,p,0)
    assert s.command.applied==282
    s,_,_=step_fluid_state(s,.75,.01,p,0)
    assert s.command.applied==282
    s=replace(s,probe_phase=2)
    s,_,_=step_fluid_state(s,.75,.01,p,0)
    assert s.command.applied==232
    s=replace(s,probe_phase=0,action_loss=False,action_ce=False)
    s,_,_=step_fluid_state(s,.75,.01,p,0)
    assert s.command.applied==192
    s=replace(s,t_s=s.command.refreshed_s+1)
    s,_,_=step_fluid_state(s,.75,.01,p,0,refresh_command=False)
    assert s.command.applied==256


def test_environment_queues_requests_in_up_and_reports_applied_duration():
    from qbbr.tests.test_sydney_semantics import make_env
    env=make_env()
    env.params=replace(env.params,kernel_action_semantics=True,base_retransmit_rate_pps=0,dwn_retransmit_rate_pps=0)
    env.reset(0)
    env._state=replace(env._state,startup_done=True,probe_phase=1,i_crs=0.)
    assert env.allowed_action_indices()==(0,1,2,3,4)
    _,_,_,info=env.step(1)
    assert info['requested_gain_units']==230
    assert sum(info['applied_gain_durations_s'].values())==pytest.approx(info['t_dec_s'])
    assert '320' in info['applied_gain_durations_s']


def test_invalid_new_mode_and_loss_cancellation():
    with pytest.raises(ValueError): FluidParams(1e6,.05,kernel_action_semantics=True)
    p=FluidParams(1e6,.05,consistent_transport=True,probe_bw_cycle=True,native_cruise_override=True,kernel_action_semantics=True)
    s,_,_=step_fluid_state(FluidState(),1.25,.01,p,0)
    s=replace(s,action_loss=True)
    s,_,_=step_fluid_state(s,1.25,.01,p,0)
    assert s.command.latched==-1 and s.command.applied==256
