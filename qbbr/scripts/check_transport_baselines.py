"""Matched simulator baselines and timestep audit; does not train an agent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv

ROOT = Path(__file__).resolve().parents[1]


def run(config, calibration, location, direction, seed, policy, dt, duration):
    env = FluidSimEnv(location, direction, calibration, episode_s=duration,
        substep_s=dt, reward_mode='throughput_only', risk_mode=config['simulator']['risk_mode'],
        dynamics_overrides=config['simulator']['dynamics_overrides'], probe_bw_phase_gate=True)
    state = env.reset(seed)
    delivered = retransmitted = elapsed = 0.0
    rtts, counts = [], [0]*5
    done = False
    while not done:
        # Frozen constants, independent of validation results. s3=1 BDP / 2.5.
        action = (0 if state[3] >= .2 else 4 if state[2] < .36 else 2) if policy == 'queue_heuristic' else int(policy[-1]) if policy.startswith('fixed_') else 2
        if action not in env.allowed_action_indices():
            action = 2
        counts[action] += 1
        state, _, done, info = env.step(action)
        delivered += info['delivered_bytes']
        retransmitted += info['retransmitted_bytes']
        elapsed += info['t_dec_s']
        rtts.append(info['rtt_ms'])
    return dict(location=location, direction=direction, seed=seed, policy=policy, substep_s=dt,
        goodput_mbps=delivered*8/elapsed/1e6, retransmission_ratio=retransmitted/max(delivered+retransmitted,1),
        retransmits_per_s=retransmitted/1500/elapsed, rtt_p95_ms=float(np.percentile(rtts,95)), action_counts=counts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duration-s', type=float, default=30)
    parser.add_argument('--out', type=Path, default=ROOT.parent/'outputs/transport_v6_baselines.json')
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/'configs/tier1_native_qa2c_successor_protocol.yaml').read_text())
    calibration=load_calibration(ROOT/'data/calibrated/per_location_constants.json')
    rows=[]
    for location in config['training']['locations']:
        for direction in config['training']['directions']:
            for seed in (7001,7002):
                for policy in ('stock_proxy','fixed_0','fixed_1','fixed_2','fixed_3','fixed_4','queue_heuristic'):
                    for dt in (.02,.01):
                        rows.append(run(config,calibration,location,direction,seed,policy,dt,args.duration_s))
            print(f'Completed {location} {direction}',flush=True)
    failures=[]
    for a,b in zip(rows[::2],rows[1::2]):
        for key in ('goodput_mbps','rtt_p95_ms','retransmits_per_s'):
            error=abs(a[key]-b[key])/max(abs(b[key]),1e-9)
            if error > .05:
                failures.append(dict(location=a['location'],direction=a['direction'],seed=a['seed'],policy=a['policy'],metric=key,relative_error=error))
    comparisons=[]
    for row in rows:
        stock=next(item for item in rows if item['location']==row['location'] and item['direction']==row['direction'] and item['seed']==row['seed'] and item['substep_s']==row['substep_s'] and item['policy']=='stock_proxy')
        comparisons.append(dict(location=row['location'],direction=row['direction'],seed=row['seed'],policy=row['policy'],substep_s=row['substep_s'],
            goodput_delta_pct=100*(row['goodput_mbps']/stock['goodput_mbps']-1),
            rtt_p95_delta_ms=row['rtt_p95_ms']-stock['rtt_p95_ms'],
            retransmission_ratio_delta=row['retransmission_ratio']-stock['retransmission_ratio'],
            retransmits_per_s_delta=row['retransmits_per_s']-stock['retransmits_per_s']))
    report=dict(comparisons=comparisons, protocol_id=config['protocol_id'], evidence='simulator_proxy_only', duration_s=args.duration_s,
        seeds=[7001,7002], timestep_relative_tolerance=.05, timestep_failures=failures,
        training_ready=False, reason='Transport model changed: independent trace recalibration/validation is required before training.', records=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'records'},indent=2))


if __name__ == '__main__':
    main()
