"""Evaluate all saved rounds, including failures, without selecting quiet subsets."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import statistics
from abba import compare


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def metrics(doc, samples):
    grouped = defaultdict(list)
    for m in doc.get('measurements', []):
        tokens = m['output_tokens']
        assert len(tokens) > 1, 'a decode rate needs at least two outputs'
        grouped[m['scenario']].append({
            'sample': m['sample'], 'outputs': tokens,
            'decode_ms': m['wall_ms'] - m['ttft_ms'], 'decode_tokens': len(tokens) - 1,
            'request_ms': m['wall_ms'], 'ttft_ms': m['ttft_ms'],
            'cycles': len(m.get('draft_cycles', [])),
            'cold_output_match': m.get('cold_output_match'),
        })
    for m in doc.get('decode_throughput', {}).get('samples', []):
        outputs = m['lane_outputs']
        assert outputs and all(v == outputs[0] for v in outputs), 'clone lanes differ'
        assert m['width'] == len(outputs), 'clone output lane count differs'
        grouped[m['scenario']].append({
            'sample': m['sample'], 'outputs': outputs,
            'decode_ms': m['decode_wall_ms'], 'decode_tokens': m['decode_output_tokens'],
            'request_ms': m['request_wall_ms'], 'cycles': m['decode_batches'],
        })
    result = {}
    assert grouped, 'no measurements'
    for key, rows in grouped.items():
        assert len(rows) == samples and {v['sample'] for v in rows} == set(range(samples)), 'sample set differs'
        assert all(v['outputs'] == rows[0]['outputs'] for v in rows), 'greedy repetitions differ'
        for row in rows:
            for field in ['decode_ms', 'decode_tokens', 'request_ms']:
                assert math.isfinite(row[field]) and row[field] > 0, 'nonpositive measurement'
        result[key] = {
            'output_sha256': digest(rows[0]['outputs']),
            'milliseconds_per_decode_token': statistics.median(v['decode_ms'] / v['decode_tokens'] for v in rows),
            'decode_ms': statistics.median(v['decode_ms'] for v in rows),
            'decode_tokens': rows[0]['decode_tokens'],
            'request_ms': statistics.median(v['request_ms'] for v in rows),
            'ttft_ms': statistics.median(v['ttft_ms'] for v in rows) if 'ttft_ms' in rows[0] else None,
            'cycles': [v['cycles'] for v in rows],
            'cold_output_match': all(v.get('cold_output_match') is not False for v in rows),
        }
    return result


def endpoint_summary(rounds, keys, cases, problems, criterion):
    """A secondary endpoint or registered category has its own noise gate."""
    keys = sorted(keys)
    selected = [v for v in cases if v['case'] in keys]
    assert len(selected) == len(keys) and keys, 'invalid registered category'
    same = all(v['output_arrays_match'] for v in selected)
    result = {}
    for name, field, case_field in [('decode', 'decode_ms', 'decode_abba'),
                                    ('request_wall', 'request_ms', 'request_wall_abba'),
                                    ('ttft', 'ttft_ms', 'ttft_abba')]:
        if any(r[k][field] is None for r in rounds for k in keys):
            continue
        totals = [sum(r[k][field] for k in keys) /
                  (sum(r[k]['decode_tokens'] for k in keys) if name == 'decode' else 1)
                  for r in rounds]
        paired = compare(*totals)
        quiet = paired['noise'] <= .05 and all(v[case_field]['noise'] <= .05 for v in selected)
        if criterion == 'improvement':
            effect = paired['regression'] < -paired['bound']
        elif criterion == 'nonregression':
            effect = paired['pass'] and all(v[case_field]['pass'] for v in selected)
        else:
            assert criterion == 'comparison'
            effect = abs(paired['regression']) > paired['bound']
        result[name] = {'abba': paired, 'all_output_arrays_match': same,
                        'measurement_qualified': not problems and same and quiet,
                        'effect_gate_passed': effect,
                        'qualified': not problems and same and quiet and effect,
                        'direction': 'candidate faster' if paired['regression'] < 0 else 'candidate slower'}
    return result


def evaluate(root, protocol):
    results = []
    for group in range(1, protocol['groups'] + 1):
        ids = [f"{protocol['prefix']}-g{group}-{suffix}" for suffix in ['a1', 'b1', 'b2', 'a2']]
        rounds, problems, sources = [], [], []
        for ident in ids:
            try:
                receipt = json.loads((root / f'results/{ident}.receipt.json').read_text())
                assert receipt['exit_code'] == 0 and receipt['state'] == 'complete', 'receipt failed'
                assert receipt['artifact_sha256'] == receipt['artifact_sha256_after'], 'artifacts changed'
                operating = json.loads((root / f'results/{ident}.operating.json').read_text())
                gates = ['operating_gate_passed', 'no_serious_or_critical_state', 'no_swap', 'ac_throughout_measurement']
                for gate in gates:
                    assert operating.get(gate) is True, gate + ' failed'
                assert any(v['phase'] == 'measurement' for v in operating['events']), 'no operating observations'
                path = root / f'results/{ident}.stdout'
                doc = json.loads(path.read_text())
                arm = protocol['arms']['control' if ident.endswith(('a1', 'a2')) else 'candidate']
                nominated = protocol['artifact_nomination'][arm['variant']]
                barrier = json.loads((root/'notes'/f"{protocol['prefix']}.artifact-barrier.json").read_text())
                for key in ['driver', 'library', 'host_step']:
                    artifact = nominated[key]
                    assert receipt['artifact_sha256'][artifact] == barrier['sha256'][artifact], 'artifact differs from nomination'
                assert receipt['artifact_sha256'][protocol['cohort']] == protocol['cohort_sha256'], 'cohort differs'
                assert receipt['job']['command'][4:7] == [nominated['driver'], nominated['library'], arm['model_root']], 'executed arm differs'
                assert doc['geometry']['verify_rows'] == arm['rows'], 'runtime width differs'
                assert doc['identity']['model_root'] == arm['model_root'], 'model root differs'
                assert doc['identity']['loaded_model_layout_sha256'] == arm['loaded_model_layout_sha256'], 'loaded model differs'
                assert doc['study_temperature'] == 0 and doc['ane_ffn_share'] == 0, 'runtime policy differs'
                assert doc['max_context_tokens'] == protocol['context_tokens'], 'context limit differs'
                clone_samples = doc.get('decode_throughput', {}).get('samples', [])
                if protocol['clone_width'] > 1:
                    assert clone_samples and not doc.get('measurements'), 'missing clone measurements'
                    assert all(v['width'] == protocol['clone_width'] for v in clone_samples), 'physical clone width differs'
                else:
                    assert not clone_samples, 'serial protocol ran clones'
                rounds.append(metrics(doc, protocol['samples_per_round']))
                sources.append({'job': ident, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'identity': doc['identity'], 'geometry': doc['geometry'], 'build_id': doc['build_id']})
            except (OSError, AssertionError, KeyError, ValueError) as e:
                problems.append(ident + ': ' + str(e))
        result = {'group': group, 'rounds': ids, 'sources': sources, 'problems': problems, 'qualified': False}
        if len(rounds) != 4:
            results.append(result)
            continue
        keys = set(rounds[0])
        if len(keys) != protocol['cases']:
            problems.append('registered case count differs'); results.append(result); continue
        if any(set(v) != keys for v in rounds):
            problems.append('case sets differ'); results.append(result); continue
        cases = []
        for key in sorted(keys):
            rows = [r[key] for r in rounds]
            abba = compare(*(v['milliseconds_per_decode_token'] for v in rows))
            cases.append({'case': key, 'output_arrays_match': len({v['output_sha256'] for v in rows}) == 1,
                          'cold_output_match': all(v['cold_output_match'] for v in rows),
                          'decode_abba': abba, 'throughput_change': abba['baseline'] / abba['candidate'] - 1,
                          'request_wall_abba': compare(*(v['request_ms'] for v in rows)),
                          'ttft_abba': compare(*(v['ttft_ms'] for v in rows)) if rows[0]['ttft_ms'] is not None else None,
                          'cycles': [v['cycles'] for v in rows]})
        primary = sorted(keys - set(protocol.get('exclude_primary_cases', [])))
        assert primary, 'empty primary cohort'
        totals = [sum(r[k]['decode_ms'] for k in primary) / sum(r[k]['decode_tokens'] for k in primary) for r in rounds]
        overall = compare(*totals)
        same = all(v['output_arrays_match'] for v in cases)
        quiet = all(v['decode_abba']['noise'] <= .05 for v in cases) and overall['noise'] <= .05
        cold = all(v['cold_output_match'] for v in cases) if protocol.get('require_cold_warm_match') else True
        if protocol['criterion'] == 'improvement':
            effect = overall['regression'] < -overall['bound']
        elif protocol['criterion'] == 'nonregression':
            effect = overall['pass'] and all(v['decode_abba']['pass'] for v in cases)
        else:
            assert protocol['criterion'] == 'comparison'
            effect = abs(overall['regression']) > overall['bound']
        request_total = compare(*(sum(r[k]['request_ms'] for k in primary) for r in rounds))
        endpoints = endpoint_summary(rounds, primary, cases, problems, protocol['criterion'])
        categories = {name: endpoint_summary(rounds, set(members), cases, problems, protocol['criterion'])
                      for name, members in protocol.get('categories', {}).items()}
        result.update(cases=cases, primary_decode_abba=overall, primary_request_wall_abba=request_total,
                      endpoint_summaries=endpoints, registered_categories=categories,
                      measurement_qualified=not problems and same and quiet and cold,
                      direction='candidate faster' if overall['regression'] < 0 else 'candidate slower',
                      primary_throughput_change=overall['baseline'] / overall['candidate'] - 1,
                      all_output_arrays_match=same, all_case_noise_within_five_percent=quiet,
                      cold_warm_gate_passed=cold, effect_gate_passed=effect,
                      qualified=not problems and same and quiet and cold and effect)
        results.append(result)
    return {'protocol': protocol, 'qualified': all(v['qualified'] for v in results), 'groups': results,
            'scope': 'Exactly the registered groups and complete cohort; all failures retained. Top-level qualification concerns the primary decode endpoint. Cold request wall, TTFT and predeclared categories each retain their own output/noise/effect gates. Medians of registered internal repetitions precede per-case ABBA. Aggregate decode rate weights cases by output tokens. AC and thermal state sampled every five seconds. A failure is not rerun into a pass.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = evaluate(root, json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'qualified': result['qualified'], 'groups': [
        {k: v for k, v in g.items() if k not in ['cases', 'sources']} for g in result['groups']]}, indent=2))
