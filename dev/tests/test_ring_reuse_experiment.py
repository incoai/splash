#!/usr/bin/env python3
"""CPU-only rejection tests for missing costs, changed outputs and short EOS."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import ring_reuse_experiment as ring


def evidence(cap=1, short_eos=False):
    warmups = 96 if cap == 128 else 1280
    document = dict(schema_version=2, experimental_draft_ring_reuse=False,
        ring_short_prefix_tokens=4096, ring_short_output_tokens=cap,
        ring_short_replay_rows=list(ring.ROWS), max_context_tokens=16384,
        geometry=ring.GEOMETRY.copy(), skipped_context_lengths=[], performance_pass=True,
        performance_failures=[], ane_ffn_bytes=0, ane_ffn_share=0, no_ane_context_tokens=16384,
        ring_runtime_instrumented=False, ring_abba_enabled=True, ring_abba_groups=3,
        ring_abba_order=list(ring.ORDER), ring_compilation_profile='production_O3_flags',
        ring_thermal_warmup_requests=warmups, ring_abba_replicates=8,
        ring_conditioned_cohort='off96_generation128_workload8_v1' if cap == 128 else 'off1280_workload8_v1',
        ring_thermal_warmup_in_primary=False, ring_toggle_once_per_four_request_workload=True,
        ring_toggle_in_primary=True, ring_seed_output_tokens=1, ring_generation_minimum_actual_tokens=64,
        identity=dict(device='synthetic-test', loaded_model_layout_sha256='synthetic', model_root='/synthetic'),
        build_id='synthetic-test')
    inputs = ring.fixture_inputs()
    outputs = []
    for i in range(4):
        count = (3, 3, 23, 128)[i] if short_eos else cap
        output = [100] * count
        if count < cap:
            output[-1] = 248046
        outputs.append(output)

    def row(name, i=None, enabled=False, sample=0):
        restored = name in ring.NAMES
        output = [100] if name == 'ring_short_seed' else outputs[i]
        hit = restored and enabled and i > 0
        retained = ring.RING_BYTES if restored and enabled else 0
        value = dict.fromkeys(ring.COUNTERS, 0)
        value.update(scenario=name, sample=sample, prompt_tokens=len(inputs[name]),
            input_tokens=inputs[name], input_token_hash=ring.token_hash(inputs[name]),
            output_tokens=output, requested_output_tokens=1 if name == 'ring_short_seed' else cap,
            actual_output_tokens=len(output), cache_status='prefix_hit' if restored else 'miss',
            matched_tokens=4096 if restored else 0,
            target_prefill_rows=ring.ROWS[i] if restored else len(inputs[name]),
            draft_context_rows=ring.ROWS[i] if restored else len(inputs[name]),
            state_entries=1, cached_state_bytes=ring.COMPOSITE_BYTES,
            state_allocated_before_bytes=(1 << 29) + (ring.RING_BYTES if hit else 0),
            state_allocated_after_bytes=(1 << 29) + retained,
            state_allocated_peak_observed_bytes=(1 << 29) + retained,
            draft_ring_cache_before_bytes=ring.RING_BYTES if hit else 0,
            draft_ring_cache_bytes=retained, draft_ring_cache_peak_observed_bytes=retained,
            draft_ring_cache_hits=int(hit), draft_ring_cache_builds=int(restored and enabled and i == 0),
            draft_ring_rebuild_rows=2048 if restored and not hit else 0,
            wall_ms=120.0 if restored and enabled and i == 0 else 60.0 if hit else 100.0,
            ttft_ms=10.0, prefill_gpu_ms=5.0,
            decode_gpu_ms=0.0 if len(output) == 1 else 10.0,
            decode_wall_ms=0.0 if len(output) == 1 else 20.0,
            decode_batches=0 if len(output) == 1 else max(1, len(output) // 4),
            decode_output_tokens=0 if len(output) == 1 else len(output))
        value['drafted_tokens'] = 7 * value['decode_batches']
        value['accepted_draft_tokens'] = min(len(output) - 1, value['drafted_tokens'])
        if restored:
            value.update(cold_output_match=True, ring_short_probe_kv_tokens=4096,
                         ring_toggle_wall_ms=0.0, ring_bytes_before_toggle=0)
        return value

    warm = []
    for repetition in range(warmups // 4):
        for i, name in enumerate(ring.NAMES):
            warm.append(row(name, i, sample=repetition))
    raw, old_bytes = [row('ring_short_seed')], 0
    for group in range(1, 4):
        for slot, mode in enumerate(ring.ORDER, 1):
            for replicate in range(1, 9):
                charged = 1.0
                for i, name in enumerate(ring.NAMES):
                    value = row(name, i, mode == 'on', group - 1)
                    value.update(ring_abba_group=group, ring_abba_slot=slot,
                        ring_abba_variant=mode, ring_abba_replicate=replicate,
                        ring_toggle_wall_ms=0.0 if i else 1.0,
                        ring_bytes_before_toggle=0 if i else old_bytes)
                    charged += value['wall_ms']
                    value['ring_workload_wall_ms'] = charged
                    raw.append(value)
                old_bytes = ring.RING_BYTES if mode == 'on' else 0
    raw.extend(row(name + '_cold', i) for i, name in enumerate(ring.NAMES))
    document.update(measurements=raw, ring_thermal_warmup_measurements=warm)
    return document


class RingEvidenceTests(unittest.TestCase):
    def test_full_costs_and_arrays(self):
        for cap in (1, 128):
            result = ring.analyze(evidence(cap), cap)
            self.assertTrue(result['primary_decision']['qualified'])
            self.assertEqual(result['measured_requests'], 384)

    def test_short_eos_retained_but_unqualified(self):
        result = ring.analyze(evidence(128, True), 128)
        self.assertFalse(result['primary_decision']['qualified'])
        self.assertEqual(list(result['actual_outputs'].values()), [3, 3, 23, 128])
        self.assertEqual(result['primary_decision']['insufficient_generation_cases'], list(ring.NAMES[:3]))

    def test_corrupted_evidence_rejected(self):
        baseline = evidence()
        mutations = (
            lambda d: d['measurements'].pop(2),
            lambda d: d['measurements'][33].update(draft_ring_cache_builds=0),
            lambda d: d['measurements'][34].update(draft_ring_cache_bytes=0),
            lambda d: d['measurements'][1].update(ring_workload_wall_ms=100.0),
            lambda d: d['measurements'][1].update(output_tokens=[101]),
            lambda d: d['measurements'][1].update(ring_abba_replicate=2),
            lambda d: d['ring_thermal_warmup_measurements'][1].update(input_token_hash=1),
            lambda d: d['measurements'][1].update(wall_ms=float('nan')),
        )
        for mutation in mutations:
            changed = copy.deepcopy(baseline)
            mutation(changed)
            with self.assertRaises(ValueError):
                ring.analyze(changed, 1)


if __name__ == '__main__':
    unittest.main()
