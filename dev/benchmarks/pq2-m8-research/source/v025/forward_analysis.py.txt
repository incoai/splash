"""Ordinary complete-forward ORIGINAL/COMBINED package screen. Pure CPU; no runtime admission."""
import argparse,json,math,statistics
from pathlib import Path
PREFIX='V024_SAMPLE '
def aa_gate(values):
    med=statistics.median(values)
    return dict(values=values,median=med,max_abs_minus1=max(abs(v-1) for v in values),
                pass_gate=.99<=med<=1.01 and all(.97<v<1.03 for v in values))
def decide(med,wins,losses,stable):
    return ('NOISE_OR_ENVIRONMENT_LIMITED' if not stable else
            'QUALIFIED_GAIN' if med<=.97 and wins>=7 else
            'QUALIFIED_REGRESSION' if med>=1.03 and losses>=7 else
            'BELOW_FROZEN_RESPONSE_THRESHOLD')
def evaluate(samples,parse_errors=None):
    report=dict(status='PENDING_ROOT_REVIEW',all_samples=samples,parse_errors=parse_errors or [],
                primary='ORIGINAL_vs_COMBINED_R233_HALF_gate_up_plus_fixed_S2_down_mapping',
                half_field='COMBINED_candidate',isolated_down_gain=False,historical_gain_addition=False,
                old_cadence_gate=False,no_retry=True,no_subset=True)
    errors=list(parse_errors or [])
    if len(samples)!=32:errors.append('exact 32 samples required')
    for i,r in enumerate(samples):
        try:
            b,p=divmod(i,4);h=p in ((1,2) if b%2==0 else (0,3))
            assert isinstance(r,dict) and type(r['block']) is int and type(r['position']) is int
            assert (r['block'],r['position'])==(b,p)
            assert r['half'] is h and r['model']=='pq2'
            assert r['target_dispatches']==889 and r['selected']==(128 if h else 0)
            assert type(r['thermal']) is int and r['thermal']>=0
            for key in ('gpu_seconds','wall_seconds'):
                assert type(r[key]) in (int,float) and math.isfinite(r[key]) and r[key]>0
        except (AssertionError,KeyError,TypeError):errors.append('invalid sample index '+str(i))
    if errors:
        report.update(decision='INCOMPLETE',errors=errors,aggregates_computed=False)
        return report
    blocks=[];aa={'ORIGINAL':[],'COMBINED':[]}
    for b in range(8):
        z=samples[4*b:4*b+4]
        ss=[r['gpu_seconds'] for r in z if not r['half']]
        cc=[r['gpu_seconds'] for r in z if r['half']]
        sm,cm=statistics.mean(ss),statistics.mean(cc)
        a,beta=ss[1]/ss[0],cc[1]/cc[0]
        aa['ORIGINAL'].append(a);aa['COMBINED'].append(beta)
        blocks.append(dict(block=b,original_gpu_seconds=sm,combined_gpu_seconds=cm,
                           ratio=cm/sm,aa_original=a,aa_combined=beta))
    med=statistics.median(r['ratio'] for r in blocks)
    gates={k:aa_gate(v) for k,v in aa.items()}
    wins=sum(r['ratio']<1 for r in blocks);losses=sum(r['ratio']>1 for r in blocks)
    stable=all(v['pass_gate'] for v in gates.values()) and all(r['thermal']==0 for r in samples)
    report.update(blocks=blocks,median_ratio=med,gain_percent=100*(1-med),speedup=1/med,
                  wins=wins,slower_blocks=losses,AA=gates,thermal=[r['thermal'] for r in samples],
                  original_median_block_seconds=statistics.median(r['original_gpu_seconds'] for r in blocks),
                  combined_median_block_seconds=statistics.median(r['combined_gpu_seconds'] for r in blocks),
                  decision=decide(med,wins,losses,stable),aggregates_computed=True)
    return report
def parse(text):
    rows=[];errors=[]
    for i,line in enumerate(text.splitlines()):
        if line.startswith(PREFIX):
            try:rows.append(json.loads(line[len(PREFIX):],parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON '+x))))
            except (json.JSONDecodeError,ValueError):errors.append('malformed sample line '+str(i))
    complete=[line for line in text.splitlines() if line.startswith('R219_COMPLETE ')]
    try:
        assert len(complete)==1
        r=json.loads(complete[0][len('R219_COMPLETE '):])
        assert r['target_forwards']==37 and r['checks_passed'] is True and r['route']=='pq2' and r['mode']=='screen'
        assert r['prefix_tokens']==128 and r['verify_rows']==8
        assert r['prefill_timed'] is False and r['draft_timed'] is False and r['profiling'] is False
    except (AssertionError,KeyError,TypeError,ValueError):errors.append('missing or invalid full completion')
    return evaluate(rows,errors)
def main():
    p=argparse.ArgumentParser();p.add_argument('--stdout',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=parse(Path(a.stdout).read_text());Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(decision=result['decision'],samples=len(result['all_samples']),aggregates_computed=result['aggregates_computed'])))
if __name__=='__main__':main()
