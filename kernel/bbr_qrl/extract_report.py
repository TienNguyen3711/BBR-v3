"""Validate a completed guest audit and extract measured evidence."""
import argparse,base64,json,statistics
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('log',type=Path)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
text=args.log.read_text()
if 'QBBR_KERNEL_AUDIT_PASS' not in text or 'QBBR_KERNEL_AUDIT_FAIL' in text:
    raise SystemExit('Guest audit did not pass; inspect serial log')
encoded=[line.split(' ',1)[1] for line in text.splitlines() if line.startswith('AUDIT_JSON_BASE64 ')]
if len(encoded)!=1: raise SystemExit('Require exactly one guest result')
report=json.loads(base64.b64decode(encoded[0]))
if not report['checks_passed'] or len(report['flows'])!=19:
    raise SystemExit('Incomplete runtime coverage')
summary={}
ranges={}
for prefix in ['stock','disabled','gain1']:
    rows=[r for r in report['flows'] if r['label'] in [prefix+str(i) for i in range(3)]]
    if len(rows)!=3: raise SystemExit('Missing baseline repeats')
    summary[prefix]={metric:statistics.median(r[metric] for r in rows)
                     for metric in ['receiver_mbps','rtt_median_ms','rtt_p90_ms','retransmits']}
    ranges[prefix]={metric:{'min':min(r[metric] for r in rows),'max':max(r[metric] for r in rows)}
                   for metric in summary[prefix]}
report['baseline_medians']=summary
report['baseline_ranges']=ranges
report['performance_equivalence_established']=False
report['baseline_deltas']={name:{
    'throughput_pct':100*(summary[name]['receiver_mbps']/summary['stock']['receiver_mbps']-1),
    'rtt_p90_ms':summary[name]['rtt_p90_ms']-summary['stock']['rtt_p90_ms'],
    'retransmits':summary[name]['retransmits']-summary['stock']['retransmits']}
    for name in ['disabled','gain1']}
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,indent=2))
print(json.dumps({'kernel':report['kernel'],'flows':len(report['flows']),
    'baseline_medians':summary,'baseline_deltas':report['baseline_deltas'],'final_status':report['final_status']},indent=2))
