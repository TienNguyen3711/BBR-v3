import json
import pytest
from qbbr.scripts.run_aligned_pilot import summaries, write_json
from qbbr.eval.constrained_selection import select_candidate


def test_pilot_selection_requires_all_seed_trace_pairs_and_safety():
    rows=[]
    for run in (6,7):
        for policy,seed,throughput,rtt in [('stock',-1,100,40),('highest_permitted',-1,103,41),('balanced110',-1,101,39),('quantum',0,110,39)]:
            rows.append(dict(city='Sydney',run=run,policy=policy,seed=seed,throughput_mbps=throughput,rtt_p90_ms=rtt,retransmits_per_s=1))
    table=summaries(rows,'Sydney',[0,1,2],[6,7])
    assert not table['quantum']['complete']
    assert not table['highest_permitted']['eligible']
    assert select_candidate(table)=='balanced110'


def test_pilot_json_rejects_nonfinite_results(tmp_path):
    path=tmp_path/'report.json'
    write_json(path,{'ok':1})
    with pytest.raises(ValueError): write_json(path,{'bad':float('nan')})
    assert json.loads(path.read_text())=={'ok':1}
