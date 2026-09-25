"""Bounded, exploratory two-city pilot with validation-only safety selection."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from qbbr.data.catalog import build_catalog
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.constrained_selection import summarize_candidate, select_candidate
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.train_on_traces import build_pool
from qbbr.scripts.run_native_qa2c_successor import _selector
from qbbr.scripts.audit_sydney_semantics import simple_action
from qbbr.train.native_loop import train_native_qrl

ROOT=Path(__file__).resolve().parents[2]
SIMPLE=('stock','highest_permitted','balanced110')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def environment(task,entry=None):
    cfg=task['config'];sim=cfg['simulator']
    kwargs={'capacity_trace_pool':task['train_pool']} if entry is None else {
        'capacity_trace':(entry['times_s'],entry['capacity_bytes_s']),
        'phase_offset_s':entry['phase_offset_s']}
    return FluidSimEnv(task['city'],'downlink',task['calibration'],
        episode_s=300. if entry is None else entry['duration_s'],
        reward_mode=sim['reward_mode'] if entry is None else 'throughput_only',
        reward_kwargs=sim.get('reward_kwargs'),risk_mode=sim['risk_mode'],
        dynamics_overrides=sim['dynamics_overrides'],probe_bw_phase_gate=True,**kwargs)


def model_for(task):
    torch.manual_seed(task['seed']);np.random.seed(task['seed'])
    a=task['config']['agent'];env=environment(task)
    q,c,_=build_matched_native_a2c_agents(env.observation_dim,env.action_space_size,
        n_layers=a['n_layers'],lr=a['learning_rate'],gamma=a['discount_factor'],
        reupload=a['reupload'],max_hidden=a['max_classical_hidden'],
        selector=_selector(task['config']),entropy_coef=a.get('entropy_coef',0.),
        stock_action=2,stock_init_bias=a.get('stock_init_bias',0.),
        normalize_returns=a.get('normalize_returns',False),critic_lr=a.get('critic_lr'))
    # Equal random stream per core, independent of job scheduling/order.
    torch.manual_seed(task['seed']);np.random.seed(task['seed'])
    return q if task['policy']=='quantum' else c


def evaluate(task,model,split):
    rows=[];selector=_selector(task['config'])
    for entry in task[split+'_pool']:
        env=environment(task,entry);state=env.reset(seed=0);done=False
        elapsed=delivered=retx=0.;rtts=[];gains={};counts=[0]*5
        while not done:
            allowed=env.allowed_action_indices()
            admitted=selector.admissible_actions(torch.zeros(5),torch.as_tensor(state),allowed)
            action=(simple_action(task['policy'],admitted) if model is None else
                    model.act(state,allowed,deterministic=True,deployment=True)[0])
            if action not in admitted: raise RuntimeError('Escaped shared mask')
            state,_,done,info=env.step(action)
            counts[action]+=1;elapsed+=info['t_dec_s'];delivered+=info['delivered_bytes']
            retx+=info['retransmits'];rtts.append(info['rtt_ms'])
            for gain,seconds in info['applied_gain_durations_s'].items(): gains[gain]=gains.get(gain,0.)+seconds
        row={'city':task['city'],'policy':task['policy'],'seed':task['seed'],
             'split':split,'run':entry['run'],'throughput_mbps':delivered*8/elapsed/1e6,
             'rtt_p90_ms':float(np.percentile(rtts,90)),'retransmits_per_s':retx/elapsed,
             'duration_s':elapsed,'action_counts':counts,'applied_gain_seconds':gains}
        if not all(np.isfinite(row[k]) for k in ('throughput_mbps','rtt_p90_ms','retransmits_per_s')):
            raise RuntimeError('Non-finite evaluation')
        rows.append(row)
    return rows


def job(task):
    torch.set_num_threads(1)
    path=Path(task['out'])/task['city']/task['policy']/('seed'+str(task['seed']))
    path.mkdir(parents=True,exist_ok=True)
    model=None
    if task['policy'] not in SIMPLE:
        model=model_for(task)
        if task['stage']=='train':
            a=task['config']['agent']
            def progress(episode,summary):
                write_json(path/'progress.json',{'episode':episode+1,'budget':task['episodes'],'summary':summary})
                print(f"TRAIN {task['city']} {task['policy']} seed={task['seed']} episode={episode+1}/{task['episodes']}",flush=True)
            diagnostics=train_native_qrl(model,environment(task),task['episodes'],
                {k:a[k] for k in ('n_step_update','entropy_start','entropy_decay_episodes') if k in a},
                run_dir=path,on_episode=progress,total_episodes=task['episodes'],
                environment_seed_base=task['seed']*1_000_000,
                reward_scale_mbps=task['config']['training']['reward_scale_mbps'])
            write_json(path/'training.json',diagnostics)
        else: model.load(path/'checkpoint.pt')
    split='validation' if task['stage']=='train' else 'evaluation'
    result=evaluate(task,model,split)
    write_json(path/(split+'.json'),result)
    print(f"DONE {split} {task['city']} {task['policy']} seed={task['seed']}",flush=True)
    return result


def summaries(rows,city,seeds,runs):
    stock={r['run']:r for r in rows if r['city']==city and r['policy']=='stock'}
    result={}
    for policy in (*SIMPLE,'quantum','classical'):
        paired=[]
        for r in rows:
            if r['city']!=city or r['policy']!=policy: continue
            b=stock[r['run']]
            paired.append({'run':r['run'],'seed':r['seed'],
                'throughput_delta_pct':100*(r['throughput_mbps']/b['throughput_mbps']-1),
                'rtt_p90_delta_ms':r['rtt_p90_ms']-b['rtt_p90_ms'],
                'retransmits_delta_per_s':r['retransmits_per_s']-b['retransmits_per_s']})
        expected={(run,seed) for run in runs for seed in ([-1] if policy in SIMPLE else seeds)}
        result[policy]=summarize_candidate(paired,expected)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'reports/aligned_two_city_pilot')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'qbbr/configs/tier1_v14_fidelity.yaml').read_text())
    cfg['protocol_id']='aligned-two-city-exploratory-pilot-20260917'
    cfg['simulator']['dynamics_overrides'].update(yaml.safe_load((ROOT/'qbbr/configs/kernel_action_alignment.yaml').read_text()))
    calibration=load_calibration(ROOT/'qbbr/data/calibrated/per_location_constants_v14.json')
    catalog=build_catalog(ROOT/'qbbr/data/raw')
    plan={'protocol_id':cfg['protocol_id'],'cities':['Sydney','Tokyo'],'direction':'downlink',
          'seeds':[0,1,2],'episodes_per_learner':5,'train_runs':[1,2,3,4,5],
          'validation_runs':[6,7],'evaluation_runs':[8,9,10],
          'simple_policies':list(SIMPLE),'cores':['quantum','classical'],
          'selection':'validation: positive median throughput and no RTT-p90/retransmission increase on any seed/run; else stock',
          'evidence':'exploratory simulator proxy; reused traces and all-run calibration; no fresh confirmatory holdout',
          'promotion':'no automatic six-city expansion or field deployment',
          'training_episodes':60,'validation_episodes':36,'evaluation_episodes':54}
    tasks=[]
    for city in plan['cities']:
        pools={name+'_pool':build_pool(catalog,city,'downlink','bbr',runs,'envq90_w5')
               for name,runs in [('train',plan['train_runs']),('validation',plan['validation_runs']),('evaluation',plan['evaluation_runs'])]}
        for name,pool in pools.items():
            expected=plan[{'train_pool':'train_runs','validation_pool':'validation_runs','evaluation_pool':'evaluation_runs'}[name]]
            if sorted(e['run'] for e in pool)!=expected: raise RuntimeError(f'Incomplete/duplicate {city} {name}')
        base={'config':cfg,'calibration':calibration,'city':city,'episodes':5,'out':str(args.out),**pools}
        for policy in (*SIMPLE,'quantum','classical'):
            for seed in ([-1] if policy in SIMPLE else plan['seeds']):
                tasks.append(dict(base,policy=policy,seed=seed,stage='train'))
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute: return
    args.out.mkdir(parents=True,exist_ok=True)
    # Do not overwrite or silently resume a different protocol/run.
    with (args.out/'plan.json').open('x') as handle: json.dump(plan,handle,indent=2)
    (args.out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    source_files=list((ROOT/'qbbr').rglob('*.py'))+list((ROOT/'qbbr/configs').glob('*.yaml'))+[ROOT/'qbbr/data/calibrated/per_location_constants_v14.json']
    manifest={'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              'forcing_sha256':{t['city']:{name:hashlib.sha256(json.dumps(t[name],sort_keys=True,default=lambda x:x.tolist()).encode()).hexdigest() for name in ['train_pool','validation_pool','evaluation_pool']} for t in tasks}}
    write_json(args.out/'manifest.json',manifest)
    started=time.monotonic();rows=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for stage in ('train','evaluation'):
                write_json(args.out/'status.json',{'stage':stage,'complete':False,'elapsed_s':time.monotonic()-started})
                futures=[pool.submit(job,dict(t,stage=stage)) for t in tasks]
                stage_rows=[]
                for future in as_completed(futures):
                    stage_rows.extend(future.result())
                    write_json(args.out/(stage+'-partial.json'),stage_rows)
                rows.extend(stage_rows)
                table={city:summaries(stage_rows,city,plan['seeds'],plan['validation_runs'] if stage=='train' else plan['evaluation_runs']) for city in plan['cities']}
                if stage=='train':
                    selection={city:{'selected':select_candidate(table[city]),'summaries':table[city]} for city in plan['cities']}
                    write_json(args.out/'selection.json',selection)  # frozen before evaluation jobs start
                else:
                    write_json(args.out/'report.json',{'plan':plan,'selection':selection,'evaluation':table,'rows':rows,'elapsed_s':time.monotonic()-started})
    except BaseException as exc:
        write_json(args.out/'status.json',{'stage':'failed','complete':False,'error':repr(exc),'elapsed_s':time.monotonic()-started})
        raise
    write_json(args.out/'status.json',{'stage':'complete','complete':True,'elapsed_s':time.monotonic()-started})
    print('PILOT COMPLETE',flush=True)

if __name__=='__main__': main()
