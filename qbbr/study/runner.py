"""Matched native training, paired holdouts, trace transfer, and exact resume."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import json
import os
import random
import time
import numpy as np
import torch

from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.state_ablation import PolicyObservationEnv
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.run_native_qa2c_successor import _selector
from qbbr.train.native_loop import train_native_qrl
from qbbr.study.protocol import atomic_json, digest, file_digest, source_digest, job_id, jobs


class Stock:
    def act(self, *args, **kwargs):
        return 2, 0.0


def build_agent(study, core, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    a = study['base']['agent']
    if core == 'qdqn':
        from qbbr.agents.quantum.recurrent_prioritized_ddqn import RecurrentPrioritizedVariationalDoubleDQN
        q = study['qdqn']
        return RecurrentPrioritizedVariationalDoubleDQN(
            7, 5, quantum_dim=q['quantum_dim'], n_layers=q['n_layers'],
            learning_rate=a['learning_rate'], discount_factor=a['discount_factor'],
            replay_capacity=q['replay_capacity'], seed=seed)
    quantum, classical, _ = build_matched_native_a2c_agents(
        7, 5, n_layers=a['n_layers'], lr=a['learning_rate'], gamma=a['discount_factor'],
        reupload=a['reupload'], max_hidden=a['max_classical_hidden'], selector=None,
        entropy_coef=a.get('entropy_coef', 0), stock_init_bias=a.get('stock_init_bias', 0),
        normalize_returns=a.get('normalize_returns', False), critic_lr=a.get('critic_lr'))
    return quantum if core == 'qa2c' else classical


def trace_pools(study, job, dataset):
    from qbbr.data.catalog import build_catalog
    from qbbr.scripts.train_on_traces import build_pool
    catalog = build_catalog(dataset)
    split = study['trace_split']
    cca = split['reference_cca'][job['direction']]
    pools = [build_pool(catalog, job['location'], job['direction'], cca, split[key],
                        split['capacity_proxy']) for key in ('train_runs', 'test_runs')]
    for pool, key in zip(pools, ('train_runs', 'test_runs')):
        if {entry['run'] for entry in pool} != set(split[key]) or len(pool) != len(split[key]):
            raise ValueError(f'Missing, duplicate, or short traces: {job}, {key}')
    return pools


def make_env(study, job, calibration, pool=None, entry=None, duration=None):
    b = study['base']
    s = b['simulator']
    kwargs = {}
    if entry is not None:
        kwargs.update(capacity_trace=(entry['times_s'], entry['capacity_bytes_s']),
                      phase_offset_s=entry['phase_offset_s'])
        duration = min(duration or study['duration_s'], entry['duration_s'])
    elif pool is not None:
        kwargs['capacity_trace_pool'] = [dict(e, duration_s=min(study['duration_s'], e['duration_s'])) for e in pool]
    env = FluidSimEnv(job['location'], job['direction'], calibration,
        risk_mode=s['risk_mode'], episode_s=duration or study['duration_s'],
        reward_mode=s['reward_mode'], reward_kwargs=s.get('reward_kwargs'),
        dynamics_overrides=s.get('dynamics_overrides'),
        probe_bw_phase_gate=s.get('probe_bw_phase_gate', False), **kwargs)
    return PolicyObservationEnv(env, _selector(b), job['variant'])


def rollout(agent, env, seed, core='a2c', history_window=8, keep_intervals=False):
    state = env.reset(seed=seed)
    history = deque([state], maxlen=history_window)
    rows, decisions, done = [], [], False
    while not done:
        allowed = env.allowed_action_indices()
        start = time.perf_counter()
        action = (agent.act(np.stack(history), allowed, epsilon=0) if core == 'qdqn'
                  else agent.act(state, allowed, deterministic=True)[0])
        latency = time.perf_counter() - start
        state, reward, done, info = env.step(action)
        history.append(state)
        dt = float(info['t_dec_s'])
        row = dict(duration_s=dt, delivered_bytes=float(info['delivered_bytes']),
                   retransmits=float(info['retransmits']), rtt_ms=float(info['rtt_ms']),
                   queue_bytes=float(info['queue_bytes']), inference_s=latency,
                   action=int(action), deadline_miss=latency > dt)
        rows.append(row)
    duration = sum(r['duration_s'] for r in rows)
    weights = np.asarray([r['duration_s'] for r in rows])
    rtts = np.asarray([r['rtt_ms'] for r in rows])
    order = np.argsort(rtts)
    p90 = float(rtts[order][np.searchsorted(np.cumsum(weights[order]), .9 * duration)])
    result = dict(seed=seed, duration_s=duration,
        throughput_mbps=sum(r['delivered_bytes'] for r in rows) * 8 / duration / 1e6,
        retransmits_per_s=sum(r['retransmits'] for r in rows) / duration,
        rtt_p90_ms=p90, rtt_variance_ms2=float(np.average((rtts - np.average(rtts, weights=weights)) ** 2, weights=weights)),
        queue_mean_bytes=float(np.average([r['queue_bytes'] for r in rows], weights=weights)),
        inference_p95_ms=float(np.percentile([r['inference_s'] * 1000 for r in rows], 95)),
        deadline_miss_fraction=sum(r['deadline_miss'] for r in rows) / len(rows),
        action_counts={str(a): sum(r['action'] == a for r in rows) for a in range(5)})
    if keep_intervals:
        result['intervals'] = rows
    return result


def stock_rollout(study, job, calibration, entry, seed, cache=None):
    """Baseline rollout for one forcing, shared across every variant and core."""
    key = (job['location'], job['direction'], None if entry is None else entry['run'], seed)
    if cache is None:
        return rollout(Stock(), make_env(study, job, calibration, entry=entry), seed)
    if key not in cache:
        cache[key] = rollout(Stock(), make_env(study, job, calibration, entry=entry), seed)
    return deepcopy(cache[key])


def evaluate(study, job, calibration, agent, test_pool=None, stock_cache=None):
    rows = []
    for entry in (test_pool if test_pool is not None else [None]):
        for seed in study['holdout_seeds']:
            stock = stock_rollout(study, job, calibration, entry, seed, stock_cache)
            learned = rollout(agent, make_env(study, job, calibration, entry=entry), seed,
                              job['core'], study['qdqn']['history_window'])
            if stock['throughput_mbps'] <= 0:
                raise ValueError('Zero stock throughput cannot define a relative gain')
            rows.append(dict(trace_run=None if entry is None else entry['run'],
                holdout_seed=seed, stock=stock, policy=learned,
                throughput_delta_pct=100 * (learned['throughput_mbps'] / stock['throughput_mbps'] - 1),
                rtt_p90_delta_ms=learned['rtt_p90_ms'] - stock['rtt_p90_ms'],
                retransmits_delta_per_s=learned['retransmits_per_s'] - stock['retransmits_per_s']))
    return rows


def train_qdqn_episode(agent, env, study, job, episode, steps):
    q = study['qdqn']
    state = env.reset(seed=job['seed'] * 1_000_000 + episode)
    history = deque([state], maxlen=q['history_window'])
    epsilon = q['epsilon_start'] + (q['epsilon_end'] - q['epsilon_start']) * episode / max(study['episodes'] - 1, 1)
    done, reward_sum, losses, length = False, 0.0, [], 0
    scale = study['base']['training']['reward_scale_mbps']
    while not done:
        before = np.stack(history)
        action = agent.act(before, env.allowed_action_indices(), epsilon)
        state, reward, done, info = env.step(action)
        history.append(state)
        agent.observe(before, action, reward / scale, np.stack(history), done,
                      env.allowed_action_indices() if not done else (2,))
        steps += 1
        length += 1
        if steps >= q['warmup_steps'] and steps % q['update_every'] == 0:
            loss = agent.optimize(q['batch_size'])
            if loss is not None:
                losses.append(loss)
        if steps % q['target_sync_every'] == 0:
            agent.sync_target()
        reward_sum += reward
    return dict(episode_reward=reward_sum, episode_length=length,
                mean_td_loss=float(np.mean(losses)) if losses else None), steps


def run_job(study, job, calibration, out, identity, pools=None, stock_cache=None):
    directory = Path(out) / job_id(job)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = directory / 'result.json', directory / 'checkpoint.pt'
    signature = digest(dict(identity=identity, job=job))
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['signature'] != signature:
            raise ValueError(f'Result identity mismatch: {directory}')
        return result
    agent = build_agent(study, job['core'], job['seed'])
    completed, training, curves, steps = 0, [], [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Checkpoint protocol/source/data mismatch; use a new output directory')
        agent.load_training_state_dict(saved['agent'])
        completed, training, curves, steps = (saved[k] for k in ('completed', 'training', 'curves', 'steps'))
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    train_pool, test_pool = pools or (None, None)
    env = make_env(study, job, calibration, pool=train_pool)
    for episode in range(completed, study['episodes']):
        started = time.perf_counter()
        if job['core'] == 'qdqn':
            summary, steps = train_qdqn_episode(agent, env, study, job, episode, steps)
        else:
            config = {k: study['base']['agent'][k] for k in ('n_step_update', 'entropy_start', 'entropy_decay_episodes') if k in study['base']['agent']}
            trained = train_native_qrl(agent, env, 1, config, start_episode=episode,
                total_episodes=study['episodes'], environment_seed_base=job['seed'] * 1_000_000,
                reward_scale_mbps=study['base']['training']['reward_scale_mbps'])
            summary = trained['episode_diagnostics'][0]
            steps += int(summary['episode_length'])
        training.append(dict(episode=episode + 1, transitions=steps,
                             wall_s=time.perf_counter() - started, **summary))
        # Evaluation does not select a model or consume training RNG streams.
        rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        if (episode + 1) % study['curve_every'] == 0 or episode + 1 == study['episodes']:
            curves.append(dict(episode=episode + 1, transitions=steps,
                               evaluation=evaluate(study, job, calibration, agent, test_pool, stock_cache)))
        torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
        saved = dict(signature=signature, agent=agent.training_state_dict(), completed=episode + 1,
                     training=training, curves=curves, steps=steps,
                     torch_rng=rng[0], numpy_rng=rng[1], python_rng=rng[2])
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(saved, temporary)
        os.replace(temporary, checkpoint)
        print(f'{job_id(job)} episode {episode + 1}/{study["episodes"]}', flush=True)
    result = dict(signature=signature, job=job, identity=identity,
        evidence_tier='simulator_proxy' if job['forcing'] == 'synthetic' else 'capacity_replay_fluid_transport',
        parameter_count=agent.param_count(), parameter_matched_to_qa2c=job['core'] != 'qdqn',
        training=training, learning_curve=curves, evaluation=curves[-1]['evaluation'])
    # The synthetic-trained policy is also tested, unchanged, on real forcing.
    if job['forcing'] == 'synthetic' and pools is not None:
        result['transfer_evaluation'] = evaluate(study, job, calibration, agent, pools[1], stock_cache)
    atomic_json(result_path, result)
    return result


def execute(study, calibration_path, dataset, out, max_jobs=None, job_filter=None,
            manifest_name='manifest.json'):
    calibration = json.loads(Path(calibration_path).read_text())
    identity = dict(protocol=study, protocol_sha256=digest(study), source_sha256=source_digest(),
                    calibration_sha256=file_digest(calibration_path),
                    calibration_scope='supplied constants; caller must document training-only fit or in-sample limitation',
                    quantum_device=os.environ.get('QBBR_QUANTUM_DEVICE', 'default'))
    out = Path(out)
    all_jobs = [job for job in jobs(study) if job_filter is None or job_filter(job)]
    if not all_jobs:
        raise ValueError('Job filter selected no jobs')
    cache = {}
    if dataset is not None:
        for job in jobs(study):
            key = (job['location'], job['direction'])
            if key not in cache:
                cache[key] = trace_pools(study, job, dataset)
        identity['trace_forcing_sha256'] = digest({str(k): [
            [{name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in e.items()} for e in pool]
            for pool in v] for k, v in cache.items()})
    elif 'replay' in study['forcing']:
        raise ValueError('Replay requires --dataset with measured traces')
    # A shard writes its own manifest so parallel shards never race, while the
    # identity check below still rejects mixing incompatible runs in one tree.
    manifest_path = out / manifest_name
    manifest = dict(identity=identity, jobs=all_jobs, status='planned', completed=[])
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity'] != identity:
        raise ValueError('Output directory belongs to different source, protocol, backend or data')
    atomic_json(manifest_path, manifest)
    completed = []
    stock_cache = {}
    for job in all_jobs:
        if max_jobs is not None and len(completed) >= max_jobs:
            break
        pools = cache.get((job['location'], job['direction']))
        # A synthetic job trains synthetically; replay pools are used only for transfer.
        if job['forcing'] == 'synthetic':
            result = run_job(study, job, calibration, out, identity, stock_cache=stock_cache)
            if pools is not None and 'transfer_evaluation' not in result:
                agent = build_agent(study, job['core'], job['seed'])
                saved = torch.load(out / job_id(job) / 'checkpoint.pt', map_location='cpu', weights_only=False)
                agent.load_training_state_dict(saved['agent'])
                result['transfer_evaluation'] = evaluate(study, job, calibration, agent, pools[1], stock_cache)
                atomic_json(out / job_id(job) / 'result.json', result)
        else:
            result = run_job(study, job, calibration, out, identity, pools, stock_cache)
        completed.append(job_id(job))
        manifest.update(status='complete' if len(completed) == len(all_jobs) else 'partial', completed=completed)
        atomic_json(manifest_path, manifest)
    return manifest
