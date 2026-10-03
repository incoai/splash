from pathlib import Path
import importlib.util,json,math,hashlib,copy
O=Path(__file__).resolve().parent
pins=json.loads((O/'FROZEN-SOURCE-PINS.json').read_text())
assert hashlib.sha256((O/'forward_analysis.py').read_bytes()).hexdigest()==pins['forward_analysis.py']
spec=importlib.util.spec_from_file_location('pure_forward_analysis',O/'forward_analysis.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
cases=[]
def check(n,v):assert v,n;cases.append(dict(case=n,pass_check=True))
def mock(ratio=.95):
 rows=[]
 for b in range(8):
  for p in range(4):
   h=p in ((1,2) if b%2==0 else (0,3))
   rows.append(dict(model='pq2',block=b,position=p,half=h,gpu_seconds=100*ratio if h else 100,
    wall_seconds=200,target_dispatches=889,selected=128 if h else 0,thermal=0))
 return rows
r=a.evaluate(mock());check('gain',r['decision']=='QUALIFIED_GAIN' and r['wins']==8 and r['median_ratio']==.95)
check('regression',a.evaluate(mock(1.04))['decision']=='QUALIFIED_REGRESSION')
check('neutral',a.evaluate(mock(1))['decision']=='BELOW_FROZEN_RESPONSE_THRESHOLD')
check('gain_boundary',a.decide(.97,7,1,True)=='QUALIFIED_GAIN')
check('gain_outside',a.decide(math.nextafter(.97,math.inf),8,0,True)=='BELOW_FROZEN_RESPONSE_THRESHOLD')
check('wins6',a.decide(.95,6,2,True)=='BELOW_FROZEN_RESPONSE_THRESHOLD')
check('regression_boundary',a.decide(1.03,1,7,True)=='QUALIFIED_REGRESSION')
for value,expected in [(.99,True),(1.01,True),(math.nextafter(.99,0),False),(math.nextafter(1.01,2),False)]:
 check('AA_median_'+repr(value),a.aa_gate([value]*8)['pass_gate']==expected)
for value,expected in [(.97,False),(1.03,False),(math.nextafter(.97,1),True),(math.nextafter(1.03,1),True)]:
 check('AA_block_'+repr(value),a.aa_gate([1]*7+[value])['pass_gate']==expected)
for h in [False,True]:
 rows=mock();z=[r for r in rows[:4] if r['half'] is h];z[1]['gpu_seconds']*=1.04
 check('route_AA_'+str(h),a.evaluate(rows)['decision']=='NOISE_OR_ENVIRONMENT_LIMITED')
rows=mock();rows[0]['thermal']=1;check('thermal',a.evaluate(rows)['decision']=='NOISE_OR_ENVIRONMENT_LIMITED')
rows=mock()
for b in range(8):
 s=[r for r in rows[4*b:4*b+4] if not r['half']];c=[r for r in rows[4*b:4*b+4] if r['half']]
 s[0]['gpu_seconds'],s[1]['gpu_seconds']=100,300;c[0]['gpu_seconds'],c[1]['gpu_seconds']=60,360
r=a.evaluate(rows);check('ratio_of_means_not_mean_of_ratios',r['median_ratio']==1.05)
check('even_median',a.statistics.median([8,1,6,3,7,2,5,4])==4.5)
invalid=[('missing',mock()[:-1]),('duplicate',mock()+[mock()[0]])]
for key,value in [('gpu_seconds',0),('gpu_seconds',-1),('gpu_seconds',float('nan')),('gpu_seconds',float('inf')),('position',3),('half',1),('selected',64)]:
 rows=mock();rows[0][key]=value;invalid.append((key+'_'+repr(value),rows))
for n,rows in invalid:
 r=a.evaluate(rows);check('invalid_'+n,r['decision']=='INCOMPLETE' and not r['aggregates_computed'] and 'median_ratio' not in r)
complete=dict(target_forwards=37,checks_passed=True,route='pq2',mode='screen',prefix_tokens=128,verify_rows=8,prefill_timed=False,draft_timed=False,profiling=False)
text='\n'.join('V024_SAMPLE '+json.dumps(r) for r in mock())+'\nR219_COMPLETE '+json.dumps(complete)
check('parse_complete',a.parse(text)['decision']=='QUALIFIED_GAIN')
check('parse_missing_completion',a.parse(text.split('\nR219_COMPLETE')[0])['decision']=='INCOMPLETE')
check('parse_malformed',a.parse(text+'\nV024_SAMPLE {bad}')['decision']=='INCOMPLETE')
(O/'KAT-RESULT.json').write_text(json.dumps(dict(status='PASS',cases=cases,count=len(cases),analyzer_sha256=pins['forward_analysis.py'],no_runtime_data=True,no_GPU=True),indent=2)+'\n')
print(json.dumps(dict(status='PASS',cases=len(cases))))
