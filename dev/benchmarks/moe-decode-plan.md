# 35B MoE decode performance plan

Scope: MoE decode on Qwen3.6-35B-A3B (40 layers, H=2048, E=256, top-8, I=512,
`runtime/model/Qwen3_6Moe.hpp:19-26`), with follow-ups for GGUF and prefill.
Status: proposed. Nothing below is a new measurement; every figure is either
quoted from the repository with its location or derived from its constants.
This revises an earlier draft after nine independent reviews (see Appendix B).

## 0. Summary

The MoE FFN is most of the 35B decode cycle, but the earlier draft's claim that
the expert kernels run "18x above the bandwidth floor" does not survive the
repo's own constants. With the stored slab size and the real per-layer route
count, the expert passes are within roughly 1.2-1.3x of a streaming floor on
the one device with per-layer numbers (Apple9), and the M5 cycle is close to
what its expert stream alone would need at uniform routing.

The deciding unknown is **N, the number of distinct live expert tiles per layer
on real serving routes** (between 9 and 65 at one lane). Every target, the phase
order, and whether occupancy work can pay at all are functions of N. Phase 0
measures it; the later phases are reordered by a rule stated in advance (§3.5).

Targets are stated as **cycle ms at constant acceptance**. Speculative
acceptance moves tok/s by up to 2.9x at constant cycle time
(`apple9-simdgroup.md:82-85`) and is out of scope here (§12), so tok/s is not a
metric this plan can be scored on.

## 1. Cost model

### 1.1 Dispatch chain (affine decode, `runtime/ops/MoE.cpp:297-339`)

| # | Pipeline | Grid at one lane (rows 8) |
| --- | --- | --- |
| 1 | `moe_route_scores_q8_m8` | `{1, 8}` (`MoE.cpp:305`) |
| 2 | `moe_route_select_q8` | `{rows, 1}` |
| 3 | `moe_group_routes` | `{1, 1, 1}` (`MoE.cpp:319`) |
| 4 | `moe_gather_rows` | `{tiles, hidden/256}` |
| 5 | `moe_expert_gate_up_q4_m8` | `{I/128 = 4, tiles}` (`MoE.cpp:190`) |
| 6 | `moe_expert_down_q4_m8` | `{H/128 = 16, tiles}` eight-simdgroup; `{H/256 = 8, tiles}` Apple9 four-simdgroup (`MoE.cpp:133-137`) |
| 7 | `moe_combine` | `{rows, hidden/256}` |

`tiles` is `plan.maximumTiles()`, not the live count: at rows 8 and M8 it is
always 65 (`moeMaximumTiles`, `MoE.hpp:103-108`), 260 at four lanes. Tile slots
past `*tile_count` exit immediately (`moe.metal:610`). 40 layers x 7 = 280
dispatches per cycle.

### 1.2 Bytes

One Q4 slab is `elements/2 + elements/16` bytes (`MoE.cpp:35`,
`moe_expert_slab.h:16-29`): 2048 x 512 -> 589,824 B. Gate + up + down =
**1,769,472 B per expert**; 256 x 40 x that = 18.12 GB for the model.

At one lane, 8 rows x top-8 = 64 routed IDs plus the shared expert. N lies in
[9, 65]; independent uniform routing gives about 58. Real routing is correlated
and unmeasured.

| N (live tiles/layer) | Expert bytes/layer | x 40 layers | Floor at R = 273 GB/s* |
| ---: | ---: | ---: | ---: |
| 9 | 15.9 MB | 0.64 GB | 2.3 ms |
| 20 | 35.4 MB | 1.42 GB | 5.2 ms |
| 30 | 53.1 MB | 2.12 GB | 7.8 ms |
| 40 | 70.8 MB | 2.83 GB | 10.4 ms |
| 58 (uniform) | 102.6 MB | 4.10 GB | 15.0 ms |

\*273 GB/s has no source in the repository (it is 18 GB / 66 ms inverted).
Phase 0 replaces it with a measured achievable rate per device.

### 1.3 What the existing measurements say

- Apple9, 40-core, rows 8, per layer (`MoE.hpp:188-194`): four-simdgroup gate/up
  0.314 ms, down 0.137 ms (0.451 total); eight-simdgroup 0.332 / 0.157. The
  gate/up:down ratio is 2.29 (2.11 for eight simdgroups) against a 2:1 slab
  ratio and a 4:1 K-step ratio, so the passes scale with bytes rather than
  with the sequential MPP chain. At N = 58, 0.451 ms/layer is about 228 GB/s.
  The maintainers' comment still calls Apple9 decode "latency-bound at low
  occupancy", and the four-simdgroup variant did win 1.06-1.25x there, so Apple9
  has real but bounded headroom.
- Apple10 four-simdgroup: "within noise of the shipped tiles (<= 1.07x)"
  (`moe.metal:721-722`).
- M5 Pro 16-core whole cycle, decode profile, 2048-token prompt: 22.540 ->
  22.405 ms at width 1, 50.105 -> 49.840 ms at width 4
  (`remaining-decode-optimizations.md:118-128`). There are no per-layer M5 MoE
  numbers yet.
- 24.7 ms is the M3 Max 40-core Apple9 cycle on battery
  (`apple9-simdgroup.md:82, 87`), not an M5 figure.
- 210 / 357 tok/s are HTTP SPEED-Bench with reasoning on
  (`docs/performance.md:7-22`), a different harness from the decode profile.

### 1.4 Targets as predicates

Rows 8, expert passes <= 0.25 ms/layer requires
`N x 1.769 MB <= 0.25 ms x R`: N <= 39 at 273 GB/s, N <= 32 at 228 GB/s.
The target is only reachable under concentrated routing and near-peak
streaming. Four-lane: at uniform routing (about 163 live experts, 11.5 GB/cycle)
no kernel change reaches a 35.6 ms cycle below 324 GB/s.

| Metric | Baseline (source) | Goal |
| --- | --- | --- |
| Per-layer expert passes, rows 8, per device | Phase 0 | Floor x <= 1.15 on the measured N |
| MoE share of the M5 16 width-1 cycle | Phase 0 | Reduce by >= 1 ms at unchanged acceptance |
| Width-4 cycle, M5 16 | 49.840 ms | Set after Phase 0 from N at 32 rows |

## 2. Invariants

1. **Tile geometry is bit-exact.** Column width and simdgroup count keep each
   element's quant-group order (`q4_mpp_tiles.h:94-96`, `moe.metal:713-724`;
   enforced by `moe_metal_test.mm:884-899` and `MoeTuning.cpp:250-268`).
2. **K partitioning is not.** Split-K reassociates the fp32 sum and "can make
   the difference exceed one output ulp" (`q4_mpp_tiles.h:374-380`). Any split
   candidate needs an operand-magnitude bound (`dev/tuning/LinearNumerics.hpp`),
   a deliberate change to the two `memcmp` gates above, and an acceptance ABBA.
   Prior art: Apple10 split-K defaults were withdrawn after reproducible
   acceptance reductions on M5 (`device-policy.md:14-17`).
3. **Routing contract**: top-8, descending score, ascending-id tie-break, fp32
   scores and weights, fixed slice and scalar-gate order (`moe.metal:290-377`).
   The score matmul's accumulation order is part of the contract, not just the
   select tail: bf16 scores already disagree with fp64 on 12-15% of real rows
   (`moe.metal:294-298`).
4. **Grouped rows**: the *set* of rows per tile is fixed by the descriptors;
   their order inside a tile comes from `atomic_fetch_add`
   (`moe.metal:481-482`) and must not be depended on.
5. **Prepared layouts frozen** (`moe_expert_slab.h`), StorageN = 256.
6. **Workspace admission** via `MoeWorkspace`/`workspaceFor` and the
   `kMoeScratchFields` assert (`MoE.hpp:110-164`); no allocation while encoding.
7. **GGUF MoE is not tuned** (`ExecutionPlans.cpp:178-183`). Any GGUF change is
   a device default; bringing GGUF into the tuner is a deliberate decision.

## 3. Phase 0: measurements that decide the rest

1. **N and rows per expert.** Read back `MoeScratch::tileCount` per layer per
   cycle on the §9 serving prompts, and on both `MoeTuning` distributions
   (distinct and repeated rows, `MoeTuning.cpp:99-122`) as a bracket. Report
   distributions, not means, at widths 1-4.
2. **T(rows) per device.** Per-layer expert-pass time at rows 8/16/24/32 on
   M5 Pro 16, M5 Pro 20 and one Apple9, with achieved GB/s beside each number.
3. **Achievable bandwidth.** Measure the dense Q4 decode kernels' streaming rate
   per device as the ceiling, instead of a datasheet or derived number.
4. **Fix the per-layer referee first.** `gguf-moe-benchmark` routes decode rows
   through a pool of 24 experts (`gguf_moe_benchmark.mm:7-9, 146-159`), has no
   candidate hooks, and defaults to 20 rounds. Add a 256-expert decode pool,
   a candidate-config argument, and a `tile_count` column, and run with
   >= 31 rounds.
5. **Tuner prerequisites** (moved here from the earlier draft's Phase 4):
   variable-length candidate sets (`MoE::decodeCandidates`,
   `ExecutionPlans::moeCandidates` are two-slot today), an optional captured
   real-route fixture, and, only if split-K proceeds, the bounded qualification
   path of §2.2.
6. **Serving ABBA harness.** It is not in the repository
   (`apple9-simdgroup.md:92-95`); budget standing it up.
7. Decode profile per device (`make benchmark-decode-profile
   MODEL=mlx-community/Qwen3.6-35B-A3B-4bit DECODE_PROFILE_ARGS='--prompt-tokens
   2048 --cycles 9'`) with the MoE pipeline rows separated.

Deliverable: one table row per device x harness x width; all later percentages
derive from the matching row.

### 3.4 Tools and first measurements

Tooling for items 1, 4 and the Phase 1 simdgroup re-measure is proposed in
#180 (decode route pool and live-tile readback in `gguf-moe-benchmark`), #181
(affine decode candidates in alternating rounds, bitwise-checked) and #182
(real-route live experts and tiles per MoE layer call in `decode-profile`).
Item 3 already exists: `make benchmark-decode` prints each dense Q4 decode
pipeline's streamed GB/s (`q4_decode_profile.mm`).

First row, synthetic routes only (M4, 10-core Apple9, 120 GB/s part, 31
rounds, affine layer ms; experts are routed experts per layer):

| Decode | Pool 24: ms (experts) | Pool 256: ms (experts) | sg4 / sg8 / m32 at pool 256 |
| --- | ---: | ---: | --- |
| B1 | 0.670 (23) | 1.439 (56) | 1.449 / 1.614 / 4.723 |
| B4 | 2.216 (78) | 4.016 (167) | 3.973 / 4.519 / 13.158 |

Layer time roughly doubles with live experts at a flat ~63-74 GB/s, i.e. it
tracks bytes, not occupancy; four simdgroups beat eight by 1.11-1.14x on this
smaller Apple9 part; the 32-row decode tile is ~3.3x slower. Real-route N and
Apple10 rows are still missing.

### 3.5 Decision rule

- Expert passes within 1.15x of the measured floor on a device: no kernel phase
  for that device; record the result.
- N <= ~25 on serving routes and T(rows) nearly flat in rows: occupancy
  candidates (Phase 1) first.
- N >= ~35: rows per expert (Phase 4) and bytes are the only levers.

## 4. Phase 1: expert-tile scheduling

Expected band: 1.05-1.3x per layer, not 2x. The three knobs below adjust the
same two things, chain length and residency, which the four-simdgroup variant
already exploited.

1. **Apple10 simdgroup re-measure (first, cheap).** Compare four and eight
   simdgroups on M5 Pro 16/20 with the §3 tools. The prior result was <= 1.07x;
   promotion is a manual ABBA because the tuner does not vary this rule
   (`device-policy.md:65-66`).
2. **N64 column tiles**, only if Phase 0 shows a latency gap on the device.
   Going N128 to N64 doubles re-reads of the grouped input (about +7-8 MB/layer
   at N = 58). Gate: >= 5% per layer, net of that traffic, bit-exact.
3. **Split-K: offline experiment only.** Prerequisites before any code:
   - use the cross-threadgroup mechanism (`split_reduce.h`,
     `linear_q4_sgmatrix.metal`), not `linear_q4_split.metal`, which splits
     inside one threadgroup and changes no grid;
   - legal factors: gate/up (K = 2048) up to 4, down (K = 512) at most 2
     (`q4_mpp_tiles.h:371`);
   - sized scratch (>= ~21 MB at one lane for four-way partials) through
     `workspaceFor`;
   - counter zeroing and release per `split_reduce.h:41-45`;
   - no copy of the dense `oneLaneTile` guard, since MoE tiles span lanes;
   - §2.2 qualification plus acceptance ABBA.
   Adding threadgroups to a grid with hundreds of live groups is expected to
   lose unless Phase 0 finds low N.

Gate M1: per-layer >= 5% on the device, bit-exact (or §2.2-qualified for
split-K), acceptance unchanged.

## 5. Phase 2: dispatch boundaries (a measurement, not a phase)

- **Deleted: fusing router scores and select.** A per-row kernel reads router
  weights 8x (about 180 MB/cycle) or serialises the router into one threadgroup,
  to save an 8 KB score round trip.
- **Route grouping.** `moe_group_routes` is genuinely serial but small (about
  0.1 ms/cycle at one lane). Keep only if a single-kernel, descriptor-only form
  wins at width 4 and is neutral at width 1; the two-kernel form adds a
  dispatch.
- **Fused expert passes.** The intermediate round trip is about 38 MB/cycle
  (about 1% of the stream). Prototype only a gate/up+down merge at constant
  threadgroup count, and only if Phase 0 shows MoE boundary cost >= 0.5 ms
  per cycle. Prior art: dense FFN producer fusion measured 1.003x and was not
  kept (`remaining-decode-optimizations.md:152-155`).
- Dispatch target: 200 per cycle at most from this content; there is no <= 160
  target.

## 6. Phase 3: GGUF decode

- **Dropped: M32 in GGUF decode.** It is blocked at `MoE.cpp:255-257` and
  `ExecutionPlans.cpp:158`, and the measured crossover (`rows * top_k >
  experts`, `MoE.hpp:238-243`) keeps M8 at every legal decode width. At about
  one route per expert it reuses no weights and only adds scratch.
- **Fused gate+up for the staged tile.** Prototype with a gate: >= 5% per layer
  on the fixed referee, both formats, and a threadgroup-memory budget stated
  first.
- **Re-evaluate `moeGgufTile`** on Apple9, including 16-core M4-class parts
  (`device-policy.md:29-32`).
- Decide first whether GGUF MoE joins the tuner (after Phase 0 item 5) or stays
  as default changes validated by GGUF ABBA (UD-Q2_K_XL, UD-Q4_K_M).

## 7. Phase 4: batch width

- M32 is already a decode candidate (`MoE.hpp:307-309`). Revalidate it only;
  the affine M32 tile has no live-row branch, so it is expected to lose at
  decode.
- Wider batches amortise experts that several lanes share, but saturate: at
  uniform routing distinct experts are about 58 / 163 / 221 / 250 at rows
  8 / 32 / 64 / 128. This touches `SPLASH_MAXIMUM_BATCH_WIDTH`
  (`ExecutionGeometry.h:13`) and the lane ABI; coordinate with the 27B width
  plan.
- Deeper verify (16 rows per lane) is not free: at uniform routing it makes the
  expert stream about 1.8x larger for about 12% fewer bytes per token, and only
  if acceptance holds. It carries the same N predicate.

## 8. Prefill

Not near its floor. The only measured figure, GGUF staged at 2048 rows on
M5 Pro 16, is 6.70-6.83 ms per layer (`moe_gguf.metal:13-16`), about 270 ms
over 40 layers against a 66 ms one-read estimate. 32-row tiles read each
expert's slabs about twice at 2048 rows.

1. Measure active tiles and achieved GB/s for affine and GGUF prefill.
2. Dequantisation-side candidates for the staged tile (the gate+up fusion of
   §6 at prefill shapes).
3. Re-validate the 8/32 crossover after any kernel change.
4. Dropped: the N512 tile, which spans two StorageN slabs and would be a
   template redesign, not a new instance.

## 9. Validation (every phase)

1. `make -j4 all && make test-engine-cpu test-engine-metal` with GPU shader
   validation. Bit-exact candidates keep the `memcmp` gates; §2.2 candidates
   use the operand bound.
2. Per-layer: the fixed referee (§3.4), alternating order, >= 31 rounds,
   DRAM-cold, `tile_count` recorded.
3. Whole cycle: decode profile at widths 1-4, before and after.
4. Serving ABBA: fresh servers, same prompts, temperature 0, one warmup,
   medians. Output text and `accepted_draft_tokens` must match. Both models;
   the 27B must be neutral.
5. Memory plan: extend `model_execution_plan_test.cpp`; `/status.memory_actual`
   peaks unchanged or lower.
6. Mixed evidence keeps the baseline (`device-policy.md`).

## 10. Milestones

| Step | Content | Gate |
| --- | --- | --- |
| M0 | Phase 0 tables and fixed tools | N, T(rows), achievable GB/s per device |
| M1 | Apple10 simdgroup re-measure; N64 if §3.5 allows | >= 5% per layer, bit-exact, ABBA neutral-or-better |
| M2 | MoE boundary measurement; grouping or merge only if >= 0.5 ms | Cycle -0.5 ms at unchanged acceptance |
| M3 | GGUF fused gate+up, tile re-evaluation | >= 5% per layer and GGUF ABBA improvement |
| M4 | Width work with the 27B plan | Width-4 cycle reduction at unchanged acceptance |
| M5 | Prefill measurement, then candidates | Measured before any kernel work |

## 11. Risks and rollback

Numerics-equivalent changes can still move acceptance
(`remaining-decode-optimizations.md:20-22`). Promotions are per device family
and core count. Candidates are additive, so rollback is a policy change.
Split-K scratch growth is sized before it is built.

## 12. Non-goals and follow-up

Out of scope: expert counts, routing semantics, quantisation formats, and the
draft. Because the expert stream is most of the cycle, bigger wins need fewer
bytes or more accepted tokens per cycle; both are outside this plan. An
acceptance workstream is the recommended separate follow-up.

## Appendix A: corrections to the earlier draft

- Live experts per layer: "at most 9" -> 9 to 65, about 58 at uniform routing.
- Bytes per expert: 0.86 MB -> 1.769 MB; the resulting "18x" -> about 1.2-1.3x.
- 24.7 ms is M3 Max Apple9, not M5 Pro.
- Split-K bit-exactness: false (§2.2).
- `benchmark-gguf-moe` is at `DEVELOPMENT.md:622-624`, not `:566-567`.
- Routing contract lines: `moe.metal:290-377`, not `:291-344`.
- GGUF tuner exclusion: `ExecutionPlans.cpp:178-183`, not `:181-185`.
- Kernel paths live under `runtime/metal/`.
- Router MACs are 524,288 per row (4.2M is the whole 8-row tile).
- "Down from 16 to 16-32" does not follow for either split factor.

## Appendix B: review provenance

Nine reviews (Bunny, Grok, HY4, Kimi, Longcat, Luna, Luna-Codex, Mimo, Qwen)
agreed on the cost-model, split-K, referee and device-mixing findings. Where
they disagreed (whether Apple10 is already at the bandwidth wall), this plan
treats the question as open and decides it by Phase 0's measurement of N.
