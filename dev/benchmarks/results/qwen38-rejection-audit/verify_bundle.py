"""Verify this evidence bundle without a model, GPU or third-party Python packages."""
from pathlib import Path
import collections,gzip,hashlib,json,math,statistics,tarfile,tempfile
import replay_reference as replay

P=Path(__file__).resolve().parent
manifest=json.loads((P/'manifest.json').read_text())
for name,expected in manifest['portable_sha256'].items():
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==expected,('portable hash',name)
with tarfile.open(P/'experimental-sources.tar.gz','r:gz') as tar:
 for name,expected in manifest['archive_source_sha256'].items():
  assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==expected,('source hash',name)
 prompt_protocol=json.loads(tar.extractfile('protocol-original.json').read())
 for c in prompt_protocol['cases']:
  data=tar.extractfile('prompts/'+c['id']+'.u32').read()
  assert len(data)==c['prompt_tokens']*4
  assert hashlib.sha256(data).hexdigest()==c['prompt_sha256']

with gzip.open(P/'compact-traces.jsonl.gz','rt') as f:records=[json.loads(line) for line in f]
assert len(records)==manifest['compact_trace_cases']
assert sum(len(r['cycles']) for r in records)==manifest['compact_trace_cycles']
baseline={r['case']['id']:r for r in records if r['set']=='baseline'}
summaries={r['id']:r for r in json.loads((P/'audit-final-summary.json').read_text())['cases']}
counts=collections.Counter();reached=[0]*7;accepted=[0]*7;taus=[]
for cid,record in baseline.items():
 misses=collections.Counter();case_reached=[0]*7;case_accepted=[0]*7
 for row in record['cycles']:
  passed=min(row['accepted'],row['retained']-1)
  rejected=(row['accepted']<7 and row['retained']==row['accepted']+1 and
            len(row['output_tokens'])==row['retained'] and
            row['target_tokens'][row['accepted']]!=row['proposed'][row['accepted']])
  for pos in range(passed):case_accepted[pos]+=1
  for pos in range(passed+int(rejected)):case_reached[pos]+=1
  if rejected:
   pos=row['accepted'];gold=row['target_tokens'][pos]
   misses['selection' if gold in row['candidates'][pos*16:(pos+1)*16] else 'candidate_recall']+=1
 tau=statistics.mean(r['retained'] for r in record['cycles']);taus.append(tau)
 assert tau==summaries[cid]['tau']
 for key in ['selection','candidate_recall']:assert misses[key]==summaries[cid]['first_rejections'][key]
 counts.update(misses)
 for pos in range(7):reached[pos]+=case_reached[pos];accepted[pos]+=case_accepted[pos]
overall=json.loads((P/'audit-final-summary.json').read_text())['aggregate']
assert len(baseline)==overall['cases']==192
assert statistics.mean(taus)==overall['macro_case_tau']
assert all(counts[k]==v for k,v in overall['first_rejections'].items())
assert reached==[r['reached'] for r in overall['conditional_acceptance']]
assert accepted==[r['accepted'] for r in overall['conditional_acceptance']]

for group,name in [('selector-validation','selector-validation-comparison.json'),('selector-test','selector-final-comparison.json')]:
 doc=json.loads((P/name).read_text());ratios=[];same=[]
 for r in records:
  if r['set']!=group:continue
  b=baseline[r['case']['id']];tau=statistics.mean(x['retained'] for x in r['cycles'])
  btau=statistics.mean(x['retained'] for x in b['cycles']);ratios.append(tau/btau)
  a=[t for x in r['cycles'] for t in x['output_tokens']];z=[t for x in b['cycles'] for t in x['output_tokens']]
  row=next(x for x in doc['cases'] if x['id']==r['case']['id'])
  assert tau==row['candidate_tau'] and btau==row['baseline_tau']
  bad=next((i for i,(x,y) in enumerate(zip(a,z)) if x!=y),None)
  assert bad==row['first_output_difference'];same.append(bad is None)
 assert math.isclose(100*(math.exp(statistics.mean(math.log(x) for x in ratios))-1),doc['geometric_mean_tau_gain_pct'],abs_tol=1e-10)
 assert all(same)==doc['all_common_output_prefixes_equal']

with tempfile.TemporaryDirectory() as folder:
 with tarfile.open(P/'representative-captures.tar.gz','r:gz') as tar:
  for cid in manifest['representative_full_capture_cases']:
   data=tar.extractfile(cid+'.jsonl').read()
   assert hashlib.sha256(data).hexdigest()==baseline[cid]['raw_capture_sha256']
   path=Path(folder)/(cid+'.jsonl');path.write_bytes(data)
   meta,rows=replay.read_capture(path);report=replay.summarize(meta,rows)
   assert report['mean_retained_tokens']==summaries[cid]['tau']
   assert report['first_rejections']==summaries[cid]['first_rejections']
   # summarize validates native greedy proposals and confidence before diagnosis.

print(json.dumps({'portable_hashes':'pass','source_and_prompt_hashes':'pass',
 'compact_cases':len(records),'compact_cycles':manifest['compact_trace_cycles'],
 'baseline_first_miss_and_conditional_counts':'exact',
 'selector_native_yield_and_output_comparisons':'exact',
 'representative_native_greedy_confidence_replays':len(manifest['representative_full_capture_cases'])},indent=2))
