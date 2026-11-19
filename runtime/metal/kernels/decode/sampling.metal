#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/split_reduce.h"

// The rows of the target policy. A greedy lane takes each row's argmax. A
// sampled lane draws from each row's min-p/top-k/top-p distribution over the
// whole vocabulary: a sharded scan sums the row's softmax denominator, one
// group per row finds the last token the distribution keeps in the order of
// the logits (logit descending, id ascending) without sorting the
// vocabulary, and the draw takes one pass over the kept tokens in id order,
// shared by the row's kVocabularyGroups groups: each sums ranges of the
// vocabulary, and the one that finishes last (split_arrive_last) draws.
// Every reduction runs in a fixed order, so a row selects the same token on
// every run. The same kernels select the first token after a prompt and the
// verify rows (TargetSamplingParams): each dispatch is a grid of a lane's rows
// by lanes, and the groups of the other policy's lanes, and those past a
// lane's rows, return at once.

// Whether `token` is one of a target's stop tokens (metal/abi/Sampling.h).
// SPLASH_NO_TOKEN, which pads them, is none, also where acceptance retains
// it as the selection of a row with no selectable logit.
inline bool stop_token(constant uint *stop_tokens, uint token) {
  bool stop = false;
  for (uint place = 0; place < SPLASH_STOP_TOKENS; ++place)
    stop |= token == stop_tokens[place];
  return stop && token != SPLASH_NO_TOKEN;
}

// The row a lane selects from: its logits, the tokens it admits (its
// constraint mask row, less the stop tokens when the lane ignores
// end-of-sequence) and, for a sampled row whose maximum is known, their
// softmax weights.
struct TargetRow {
  device const float *logits;
  device const uint *mask;
  bool constrained;
  bool exclude_stop;
  constant uint *stop_tokens;
  uint vocabulary;
  float temperature;
  float maximum;

  bool admits(uint token) const {
    if (constrained && (mask[token / 32] & (1u << (token % 32))) == 0)
      return false;
    return !exclude_stop || !stop_token(stop_tokens, token);
  }
  float weight(float value) const {
    return exp((value - maximum) / temperature);
  }
};

// Row `index` of lane `lane` of a dispatch (TargetSamplingParams): whether
// the lane selects it, one of its rows, and takes the policy the kernel runs
// (sampled or greedy), and its logits row s, which indexes the workspaces
// and outputs.
inline bool selects(constant TargetSamplingParams &params,
                    device const SplashLaneRows *lane_rows, uint lane, uint index,
                    bool sampled) {
  return index < lane_rows[lane].rows &&
         ((params.sampling_mask & (1u << lane)) != 0) == sampled;
}
inline uint selected_logits_row(device const SplashLaneRows *lane_rows,
                                uint lane, uint index) {
  return lane_rows[lane].first + index;
}

// Row `index` of lane `lane` of a dispatch (TargetSamplingParams).
inline TargetRow selected_row(device const float *logits,
                              device const uint *token_mask,
                              constant TargetSamplingParams &params,
                              device const SplashLaneRows *lane_rows, uint lane,
                              uint index) {
  return {logits + ulong(selected_logits_row(lane_rows, lane, index)) *
                       params.vocabulary,
          token_mask + (ulong(lane) * (SPLASH_TARGET_VERIFY_ROWS + 1) +
                        params.mask_row + index) *
                           params.mask_words,
          (params.constrained_mask & (1u << lane)) != 0,
          (params.exclude_stop_mask & (1u << lane)) != 0,
          params.stop_tokens,
          params.vocabulary,
          params.temperature[lane],
          0.0f};
}

// Each lane's lane rows and tree row for the selection of its first token
// after its prompt (SampleFirstRowsParams), one group a lane: its one row of
// the logits, a tree's root, which the selection's kernels read as a decode
// cycle's.
kernel void decode_sample_first_rows(
    device SplashLaneRows *lane_rows [[buffer(0)]],
    device SplashTreeRow *tree [[buffer(1)]],
    constant SampleFirstRowsParams &params [[buffer(2)]],
    uint lane [[threadgroup_position_in_grid]]) {
  lane_rows[lane] = {params.row[lane], 1};
  tree[params.row[lane]] = {1u, SPLASH_TREE_ROOT, 0};
}

// Shares of a row's softmax denominator combined in order: their largest
// maximum, their sums rescaled to it and their admitted counts. A group
// combines its simdgroups' shares, and a row's search its shards'.
template <class Shares>
inline TargetShardMass merge_masses(Shares shares, uint count,
                                    float temperature) {
  TargetShardMass total{-FLT_MAX, 0.0f, 0};
  for (uint index = 0; index < count; ++index)
    total.maximum = max(total.maximum, shares[index].maximum);
  for (uint index = 0; index < count; ++index) {
    const TargetShardMass share = shares[index];
    if (share.sum > 0.0f)
      total.sum +=
          share.sum * exp((share.maximum - total.maximum) / temperature);
    total.admitted += share.admitted;
  }
  return total;
}

// One shard's share of a sampled row's softmax denominator: each thread
// keeps a running maximum and sum of its admitted tokens' weights (an online
// softmax), combined over the group in a fixed order. The running maximum
// starts below every finite logit, so a -inf logit weighs nothing.
inline void shard_mass(TargetRow row, uint shard,
                       device TargetShardMass &partial,
                       threadgroup TargetShardMass *group_masses,
                       uint thread_index, uint lane, uint simd_group) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  TargetShardMass mass{-FLT_MAX, 0.0f, 0};
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    const float value = row.logits[token];
    if (value > mass.maximum) {
      mass.sum =
          mass.sum * exp((mass.maximum - value) / row.temperature) + 1.0f;
      mass.maximum = value;
    } else {
      mass.sum += exp((value - mass.maximum) / row.temperature);
    }
    ++mass.admitted;
  }
  const float simd_maximum = simd_max(mass.maximum);
  const float scaled =
      mass.sum > 0.0f
          ? mass.sum * exp((mass.maximum - simd_maximum) / row.temperature)
          : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(mass.admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    partial = merge_masses(group_masses, 8, row.temperature);
}

// One shard of each selected row of the sampled lanes.
kernel void decode_sample_mass_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetShardMass *partial_masses [[buffer(2)]],
    device const SplashLaneRows *lane_rows [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup TargetShardMass group_masses[8];
  const uint batch = group.y, index = group.x / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!selects(params, lane_rows, batch, index, true))
    return;
  const uint shard = group.x % SPLASH_TARGET_SAMPLING_SHARDS;
  shard_mass(selected_row(logits, token_mask, params, lane_rows, batch, index), shard,
             partial_masses[ulong(selected_logits_row(lane_rows, batch, index)) *
                                SPLASH_TARGET_SAMPLING_SHARDS +
                            shard],
             group_masses, thread_index, lane, simd_group);
}

// The groups that search and draw a sampled row.
constant constexpr uint kVocabularyThreads = SPLASH_TARGET_VOCABULARY_THREADS;
constant constexpr uint kVocabularySimdgroups = kVocabularyThreads / 32;
constant constexpr uint kVocabularyGroups = SPLASH_TARGET_VOCABULARY_GROUPS;
// The draw's ranges of the vocabulary; the last group loads them one per
// thread.
constant constexpr uint kVocabularyRanges = SPLASH_TARGET_VOCABULARY_RANGES;
static_assert(kVocabularyRanges <= kVocabularyThreads,
              "a group holds one draw range per thread");
// Pivots per pass: each pass splits a bracket sixteen ways. In logit order
// kKeyPivots of them split its keys evenly and the others follow the logits'
// scale (place_pivots).
constant constexpr uint kPivots = 15;
constant constexpr uint kKeyPivots = 4;

// The order of the logits as an unsigned key: a larger logit has a larger
// key, and -0 and +0, which compare equal, share one. Every logit's key is
// above kNoKey.
inline uint logit_key(float value) {
  uint bits = as_type<uint>(value);
  bits = (bits << 1) ? bits : 0u;
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float key_logit(uint key) {
  return as_type<float>((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
}
constant constexpr uint kNoKey = 0u;
// The key of -infinity, at or below that of every admitted logit.
constant constexpr uint kLowestKey = 0x007fffffu;

// A position in the order of the logits: the tokens at or before it have a
// larger key, or the same key and an id at most last. {kNoKey, 0} lies
// after every token.
struct OrderBoundary {
  uint key;
  uint last;
};

inline bool at_or_before(uint key, uint token, OrderBoundary boundary) {
  return key > boundary.key || (key == boundary.key && token <= boundary.last);
}

// The orders a search runs in. The logit order is the order of the logits.
// Among the tokens tied at one logit, the tie order puts a smaller id first;
// every other token has kNoKey.
struct LogitOrder {
  static constexpr constant bool by_logit = true;
  uint key(float value, uint) const { return logit_key(value); }
  float logit(uint key) const { return key_logit(key); }
};
struct TieOrder {
  uint tied_key;
  uint vocabulary;
  uint key(float value, uint token) const {
    return logit_key(value) == tied_key ? vocabulary - token : kNoKey;
  }
  static constexpr constant bool by_logit = false;
  float logit(uint) const { return key_logit(tied_key); }
};

// What a search keeps: the tokens in order until their count, or their mass
// (the sum of their weights), exceeds the target. A count search measures
// counts alone (top-k, and the tie order's first ids), a mass search counts
// and masses (top-p).
struct CountTarget {
  static constexpr constant bool by_mass = false;
  uint count;

  bool exceeded(uint measured_count, float) const {
    return measured_count > count;
  }
};
struct MassTarget {
  static constexpr constant bool by_mass = true;
  float mass;

  bool exceeded(uint, float measured_mass) const {
    return measured_mass > mass;
  }
};

// A key range [lo, hi) and the count and mass of the tokens with at least
// each end's key: lo's exceed the search target, hi's do not. A count
// search narrows the counts alone and leaves the masses as they came in.
struct Bracket {
  uint lo;
  uint hi;
  uint lo_count;
  uint hi_count;
  float lo_mass;
  float hi_mass;

  uint tokens() const { return lo_count - hi_count; }
};

// The last token a search keeps, and the count and mass of the tokens at or
// before it. A ranked selection leaves those tokens in the scratch keys and
// ids, in order (resolve).
struct Selection {
  OrderBoundary last;
  uint count;
  float mass;
  bool ranked;
};

struct VocabularyScratch {
  uint counts[kVocabularySimdgroups][kPivots];
  float masses[kVocabularySimdgroups][kPivots];
  uint total_counts[kPivots];
  float total_masses[kPivots];
  atomic_uint gathered;
  uint keys[kVocabularyThreads];
  uint ids[kVocabularyThreads];
  Selection selection;
};

struct DrawScratch {
  uint arrival;
  // Per range of the draw: the kept weight of its tokens and one past the
  // last of them with weight.
  float range_weights[kVocabularyRanges];
  uint range_last[kVocabularyRanges];
  uint drawn_range;
  float drawn_target;
  uint drawn_token;
  // Whether the row selected a candidate, and whether its draw excludes its
  // candidates, which it then holds (vocabulary_draw).
  bool selected;
  bool excluding;
  uint excluded[32];
};

// Pivots inside [lo, hi) that split the bracket. The tie order splits its
// keys evenly. In logit order the last kKeyPivots do, so a pass keeps at
// most a fifth of the bracket's keys, rounded up, and a search ends within
// fourteen passes whatever the logits, temperature or penalties; the others
// follow the logits' scale, which ends most searches within a few: geometric
// distances below the bracket's top, in units of the temperature, while it
// is open at the bottom, and even steps in logit value once it is not.
template <class Order>
inline void place_pivots(Order order, Bracket bracket, float temperature,
                         thread uint (&pivots)[kPivots]) {
  const ulong width = bracket.hi - bracket.lo;
  if (!order.by_logit) {
    for (uint pivot = 0; pivot < kPivots; ++pivot)
      pivots[pivot] = bracket.lo + uint(width * (pivot + 1) / (kPivots + 1));
    return;
  }
  constexpr uint kScaled = kPivots - kKeyPivots;
  const float top = key_logit(bracket.hi - 1);
  const float bottom = key_logit(bracket.lo);
  const bool open = bracket.lo == kLowestKey;
  for (uint pivot = 0; pivot < kScaled; ++pivot) {
    const float value =
        open ? top - temperature * 0.5f * exp2(float(kScaled - 1 - pivot))
             : bottom + (top - bottom) * float(pivot + 1) / float(kScaled + 1);
    pivots[pivot] = clamp(logit_key(value), bracket.lo, bracket.hi - 1);
  }
  for (uint pivot = 0; pivot < kKeyPivots; ++pivot)
    pivots[kScaled + pivot] =
        bracket.lo + uint(width * (pivot + 1) / (kKeyPivots + 1));
}

// The count and, with Masses, the mass of the admitted tokens at or before
// the floor whose keys are at least each pivot's, summed per thread in token
// order and then over the group in a fixed order. Counts alone weigh no
// token.
template <bool Masses, uint Pivots, class Order>
inline void measure_pivots(TargetRow row, Order order, OrderBoundary floor,
                           thread const uint (&pivots)[Pivots],
                           threadgroup VocabularyScratch &scratch,
                           uint thread_index, uint lane, uint simd_group) {
  uint counts[Pivots];
  float masses[Pivots];
  uint lowest = pivots[0];
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    counts[pivot] = 0;
    if constexpr (Masses)
      masses[pivot] = 0.0f;
    lowest = min(lowest, pivots[pivot]);
  }
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    // Tokens below every pivot add nothing.
    if (key < lowest || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if constexpr (Masses) {
      const float weight = row.weight(value);
      for (uint pivot = 0; pivot < Pivots; ++pivot) {
        const bool above = key >= pivots[pivot];
        counts[pivot] += above ? 1u : 0u;
        masses[pivot] += above ? weight : 0.0f;
      }
    } else {
      for (uint pivot = 0; pivot < Pivots; ++pivot)
        counts[pivot] += key >= pivots[pivot] ? 1u : 0u;
    }
  }
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    const uint count = simd_sum(counts[pivot]);
    if (lane == 0)
      scratch.counts[simd_group][pivot] = count;
    if constexpr (Masses) {
      const float mass = simd_sum(masses[pivot]);
      if (lane == 0)
        scratch.masses[simd_group][pivot] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < Pivots) {
    uint count = 0;
    for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
      count += scratch.counts[simd][thread_index];
    scratch.total_counts[thread_index] = count;
    if constexpr (Masses) {
      float mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][thread_index];
      scratch.total_masses[thread_index] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The count and mass of the admitted tokens at or before the floor whose
// logits' keys are at least key, summed as measure_pivots sums a pivot's.
struct Measure {
  uint count;
  float mass;
};
inline Measure measure_at_or_above(TargetRow row, OrderBoundary floor,
                                   uint key,
                                   threadgroup VocabularyScratch &scratch,
                                   uint thread_index, uint lane,
                                   uint simd_group) {
  const uint pivots[1] = {key};
  measure_pivots<true>(row, LogitOrder{}, floor, pivots, scratch,
                       thread_index, lane, simd_group);
  const Measure measure{scratch.total_counts[0], scratch.total_masses[0]};
  // Every thread reads the totals before the next pass rewrites them.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return measure;
}

// Narrows the bracket until it holds at most one token per thread or a
// single key.
template <class Order, class Target>
inline Bracket narrow(TargetRow row, Order order, OrderBoundary floor,
                      Bracket bracket, Target target,
                      threadgroup VocabularyScratch &scratch,
                      uint thread_index, uint lane, uint simd_group) {
  while (bracket.tokens() > kVocabularyThreads &&
         bracket.hi - bracket.lo > 1) {
    uint pivots[kPivots];
    place_pivots(order, bracket, row.temperature, pivots);
    measure_pivots<Target::by_mass>(row, order, floor, pivots, scratch,
                                    thread_index, lane, simd_group);
    // The highest pivot whose measure exceeds the target and the lowest one
    // whose measure does not bound the new bracket. A pivot at lo never
    // becomes hi: lo's measure may come from another sum (the shards' masses,
    // or the top-k selection's), and rounding must not empty the bracket.
    for (uint pivot = 0; pivot < kPivots; ++pivot) {
      const uint count = scratch.total_counts[pivot];
      const float mass = Target::by_mass ? scratch.total_masses[pivot] : 0.0f;
      if (target.exceeded(count, mass)) {
        if (pivots[pivot] >= bracket.lo) {
          bracket.lo = pivots[pivot];
          bracket.lo_count = count;
          if (Target::by_mass)
            bracket.lo_mass = mass;
        }
      } else if (pivots[pivot] > bracket.lo && pivots[pivot] < bracket.hi) {
        bracket.hi = pivots[pivot];
        bracket.hi_count = count;
        if (Target::by_mass)
          bracket.hi_mass = mass;
      }
    }
    // Every thread reads the totals before the next pass rewrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bracket;
}

// The last token the target keeps in a bracket of at most one token per
// thread: its tokens are gathered, put in order by rank, and added to the
// measure at hi one at a time until it exceeds the target. A count search in
// logit order measures no masses while it narrows: its gather sums the mass
// at hi as measure_pivots would have, or, when every token from lo up fits
// the group, gathers them all and adds them from the first, which leaves the
// tokens it keeps ranked in the scratch (Selection::ranked).
template <class Order, class Target>
inline Selection resolve(TargetRow row, Order order, OrderBoundary floor,
                         Bracket bracket, Target target,
                         threadgroup VocabularyScratch &scratch,
                         uint thread_index, uint lane, uint simd_group) {
  constexpr bool unmeasured = Order::by_logit && !Target::by_mass;
  const bool ranked = unmeasured && bracket.lo_count <= kVocabularyThreads;
  const bool measures = unmeasured && !ranked;
  if (thread_index == 0)
    atomic_store_explicit(&scratch.gathered, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float above = 0.0f;
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    if (key < bracket.lo || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if (!ranked && key >= bracket.hi) {
      if (measures)
        above += row.weight(value);
      continue;
    }
    const uint slot = atomic_fetch_add_explicit(&scratch.gathered, 1u,
                                                memory_order_relaxed);
    if (slot < kVocabularyThreads) {
      scratch.keys[slot] = key;
      scratch.ids[slot] = token;
    }
  }
  if (measures) {
    above = simd_sum(above);
    if (lane == 0)
      scratch.masses[simd_group][0] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gathered = min(atomic_load_explicit(&scratch.gathered,
                                                 memory_order_relaxed),
                            kVocabularyThreads);
  uint key = 0;
  uint id = 0;
  uint rank = 0;
  if (thread_index < gathered) {
    key = scratch.keys[thread_index];
    id = scratch.ids[thread_index];
    for (uint other = 0; other < gathered; ++other) {
      const uint other_key = scratch.keys[other];
      rank += other_key > key || (other_key == key && scratch.ids[other] < id)
                  ? 1u
                  : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < gathered) {
    scratch.keys[rank] = key;
    scratch.ids[rank] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    uint count = ranked ? 0 : bracket.hi_count;
    float mass = ranked ? 0.0f : bracket.hi_mass;
    if (measures) {
      mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][0];
    }
    // Rounding may leave a mass target unreached: keep the whole bracket.
    Selection selection{{bracket.lo, 0xffffffffu},
                        bracket.lo_count,
                        Target::by_mass ? bracket.lo_mass : mass,
                        ranked};
    for (uint index = 0; index < gathered; ++index) {
      count += 1;
      mass += row.weight(order.logit(scratch.keys[index]));
      selection = {{scratch.keys[index], scratch.ids[index]}, count, mass,
                   ranked};
      if (target.exceeded(count, mass))
        break;
    }
    scratch.selection = selection;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch.selection;
}

// The last token the target keeps among the admitted tokens at or before
// the floor, starting from a bracket in logit order.
template <class Target>
inline Selection select_last(TargetRow row, OrderBoundary floor,
                             Bracket bracket, Target target,
                             threadgroup VocabularyScratch &scratch,
                             uint thread_index, uint lane, uint simd_group) {
  bracket = narrow(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  if (bracket.tokens() <= kVocabularyThreads)
    return resolve(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  // More tokens tie at one logit than a group gathers: the target keeps its
  // first ones by id, each of the same weight, after the mass above them,
  // which a count search measures in one more pass.
  const uint ties = bracket.tokens();
  const float weight = row.weight(key_logit(bracket.lo));
  float above;
  uint keep;
  if constexpr (Target::by_mass) {
    above = bracket.hi_mass;
    keep = 1u + uint(clamp((target.mass - above) / weight, 0.0f,
                           float(ties - 1)));
  } else {
    above = measure_at_or_above(row, floor, bracket.hi, scratch, thread_index,
                                lane, simd_group)
                .mass;
    keep = target.count + 1 - bracket.hi_count;
  }
  uint last = bracket.lo == floor.key ? floor.last : 0xffffffffu;
  if (keep < ties) {
    const TieOrder order{bracket.lo, row.vocabulary};
    const CountTarget first{keep - 1};
    Bracket tied{1, row.vocabulary + 1, ties, 0, 0.0f, 0.0f};
    tied = narrow(row, order, floor, tied, first, scratch, thread_index, lane,
                  simd_group);
    last = resolve(row, order, floor, tied, first, scratch, thread_index, lane,
                   simd_group)
               .last.last;
  }
  return {{bracket.lo, last}, bracket.hi_count + keep,
          above + float(keep) * weight, false};
}

// The last token of the row's distribution: the tokens that weigh at least
// min_p of the heaviest, then the top_k of those, then within those the top_p
// nucleus, whose mass is measured against the mass of what the two cuts
// before it keep. A top-k selection the scratch holds ranked is walked for
// the nucleus in that order, with the sums it was selected with; otherwise
// the nucleus is searched for. A row that keeps every admitted token ends at
// {kNoKey, 0}.
inline OrderBoundary distribution_end(TargetRow row, float min_p, uint top_k,
                                      float top_p, float mass, uint admitted,
                                      threadgroup VocabularyScratch &scratch,
                                      uint thread_index, uint lane,
                                      uint simd_group) {
  OrderBoundary end{kNoKey, 0};
  Bracket bracket{kLowestKey, logit_key(row.maximum) + 1, admitted, 0, mass,
                  0.0f};
  if (min_p > 0.0f) {
    // A token weighs min_p of the heaviest, which weighs 1, at the logit
    // -temperature * log(min_p) below the maximum. The tokens at or above
    // that logit stay, the heaviest always.
    const uint lowest = logit_key(
        min(row.maximum + row.temperature * log(min_p), row.maximum));
    const Measure kept = measure_at_or_above(row, end, lowest, scratch,
                                             thread_index, lane, simd_group);
    if (kept.count < admitted) {
      end = {lowest, 0xffffffffu};
      bracket.lo = lowest;
      bracket.lo_count = kept.count;
      bracket.lo_mass = kept.mass;
    }
  }
  if (top_k < bracket.lo_count) {
    // A min_p cut ends on a whole key, which the bracket bounds from below,
    // so this search needs no floor.
    const Selection top = select_last(row, {kNoKey, 0}, bracket,
                                      CountTarget{top_k - 1}, scratch,
                                      thread_index, lane, simd_group);
    end = top.last;
    if (top.ranked && top_p < 1.0f) {
      // Every thread reads the selection before the walk rewrites it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (thread_index == 0) {
        // Rounding may leave the nucleus unreached: keep the whole top-k.
        const float nucleus = top_p * top.mass;
        float walked = 0.0f;
        for (uint index = 0; index < top.count; ++index) {
          walked += row.weight(key_logit(scratch.keys[index]));
          if (walked > nucleus) {
            end = {scratch.keys[index], scratch.ids[index]};
            break;
          }
        }
        scratch.selection.last = end;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      return scratch.selection.last;
    }
    bracket.lo = end.key;
    bracket.lo_count = top.count;
    bracket.lo_mass = top.mass;
  }
  if (top_p < 1.0f)
    end = select_last(row, end, bracket, MassTarget{top_p * bracket.lo_mass},
                      scratch, thread_index, lane, simd_group)
              .last;
  return end;
}

// A token's weight in the row's distribution: its softmax weight if the
// row admits it at or before the distribution's end, else 0.
inline float kept_weight(TargetRow row, OrderBoundary end, uint token) {
  if (token >= row.vocabulary || !row.admits(token))
    return 0.0f;
  const float value = row.logits[token];
  return at_or_before(logit_key(value), token, end) ? row.weight(value) : 0.0f;
}

// A sampled verify row's draft children (TargetSamplingParams): the
// candidates of its next position (ids, kDrawCandidates of them), the draft's
// probabilities of them there (q), the candidate indices of the children its
// draft drew, in the order drawn (kNoDraw past them), and a uniform for each
// child's test. A row without children draws from its whole distribution.
constant constexpr uchar kNoDraw = 0xff;
struct DrawChildren {
  device const uint *ids;
  device const float *q;
  device const uchar *draws;
  device const float *uniforms;
};

// The row's selection, shared by its groups; returns true in the group that
// finishes last, with the selected token. Each kept token weighs its softmax
// weight. Each simdgroup of the row's groups sums one range of the
// vocabulary; the last group adds the ranges in order, then tests the row's
// children in the order drawn, recursive rejection sampling without
// replacement: child i, of probability p_i under the target and q_i under the
// draft, is selected with probability min(1, p_i / q_i), where after each
// rejection the target's distribution becomes its residual max(p - q, 0),
// renormalized, and the draft's loses the rejected child, renormalized. A
// row that selects none draws from the last residual: from its candidates'
// residual weights, or past them from the target's own weights of the tokens
// outside its candidates, whose range it walks with prefix sums. So the
// selection is a draw from the target's distribution whatever the children,
// and a child the row rejected only at fp32 rounding: where its residual
// rounds to nothing, which leaves the draw the target before the child's
// test, or where the candidates hold all the target keeps and their ranges a
// remainder of rounding, which a draw past them lands on. A range's weight
// and its walk add the same token weights, so the range the draw picks always
// has a token to draw; a row that keeps no token draws past the vocabulary's
// end, the sentinel of a row with no selectable logit.
template <uint kDrawCandidates>
inline bool vocabulary_draw(
    TargetRow row, OrderBoundary end, float uniform, DrawChildren children,
    device coherent(device) TargetVocabularyRange *ranges,
    device atomic_uint *arrivals, uint slice, threadgroup DrawScratch &scratch,
    uint thread_index, uint lane, uint simd_group, thread uint &drawn) {
  const uint range_tokens = (row.vocabulary + kVocabularyRanges * 32 - 1) /
                            (kVocabularyRanges * 32) * 32;
  const uint own = slice * kVocabularySimdgroups + simd_group;
  const uint begin = min(own * range_tokens, row.vocabulary);
  const uint finish = min(begin + range_tokens, row.vocabulary);
  float kept = 0.0f;
  uint after = 0;
  for (uint first = begin; first < finish; first += 32) {
    const uint token = first + lane;
    const float weight = token < finish ? kept_weight(row, end, token) : 0.0f;
    kept += weight;
    after = weight > 0.0f ? token + 1 : after;
  }
  kept = simd_sum(kept);
  after = simd_max(after);
  if (lane == 0)
    ranges[own] = {kept, after};
  if (!split_arrive_last(arrivals, kVocabularyGroups, thread_index,
                         &scratch.arrival))
    return false;
  if (thread_index < kVocabularyRanges) {
    const TargetVocabularyRange measured = ranges[thread_index];
    scratch.range_weights[thread_index] = measured.weight;
    scratch.range_last[thread_index] = measured.after;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      total += scratch.range_weights[range];
    scratch.excluding = false;
    scratch.selected = false;
    float target = uniform * total;
    if (children.ids && children.draws[0] != kNoDraw && total > 0.0f) {
      // The target's and the draft's probabilities of the candidates, and
      // the target's mass outside them: what the ranges weigh once the
      // candidates' weights leave them.
      float p[kDrawCandidates], q[kDrawCandidates];
      for (uint i = 0; i < kDrawCandidates; ++i) {
        const uint id = children.ids[i];
        const float weight = kept_weight(row, end, id);
        // A candidate past the vocabulary, as the draft's top 16 lists one
        // (0xffffffff) for a row of fewer comparable logits, weighs nothing
        // and lies in no range.
        if (id < row.vocabulary) {
          const uint range = id / range_tokens;
          scratch.range_weights[range] = max(scratch.range_weights[range] - weight, 0.0f);
        }
        scratch.excluded[i] = id;
        p[i] = weight / total;
        q[i] = children.q[i];
      }
      float outside_weight = 0.0f;
      for (uint range = 0; range < kVocabularyRanges; ++range)
        outside_weight += scratch.range_weights[range];
      float outside = outside_weight / total;
      for (uint child = 0; child < SPLASH_TREE_CHILDREN && !scratch.selected; ++child) {
        const uint drawn_child = children.draws[child];
        if (drawn_child == kNoDraw)
          break;
        if (children.uniforms[child] * q[drawn_child] < p[drawn_child]) {
          scratch.selected = true;
          scratch.drawn_token = children.ids[drawn_child];
          break;
        }
        // The target's residual, and the draft without the rejected child.
        float residual = outside;
        for (uint i = 0; i < kDrawCandidates; ++i)
          residual += max(p[i] - q[i], 0.0f);
        if (!(residual > 0.0f))
          break;
        for (uint i = 0; i < kDrawCandidates; ++i)
          p[i] = max(p[i] - q[i], 0.0f) / residual;
        outside /= residual;
        // The draft's rest, summed: one less a child that holds nearly all of
        // q_T would cancel to nothing in fp32.
        q[drawn_child] = 0.0f;
        float rest = 0.0f;
        for (uint i = 0; i < kDrawCandidates; ++i)
          rest += q[i];
        for (uint i = 0; rest > 0.0f && i < kDrawCandidates; ++i)
          q[i] /= rest;
      }
      if (!scratch.selected) {
        // The last residual's draw: a candidate, or past them a token of the
        // target's own weights outside them. Rounding may leave the point
        // unreached: the last candidate with weight, where none lies outside.
        float inside = 0.0f;
        for (uint i = 0; i < kDrawCandidates; ++i)
          inside += p[i];
        const float point = uniform * (inside + outside);
        float cumulative = 0.0f;
        uint last = kDrawCandidates;
        for (uint i = 0; i < kDrawCandidates && !(point < cumulative); ++i) {
          if (p[i] > 0.0f) {
            last = i;
            cumulative += p[i];
          }
        }
        if (last < kDrawCandidates && (point < cumulative || !(outside > 0.0f))) {
          scratch.selected = true;
          scratch.drawn_token = children.ids[last];
        } else {
          scratch.excluding = true;
          target = (point - inside) / outside * outside_weight;
        }
      }
    }
    if (!scratch.selected) {
      float cumulative = 0.0f;
      uint range_drawn = 0;
      float range_target = 0.0f;
      for (uint range = 0; range < kVocabularyRanges; ++range) {
        const float weight = scratch.range_weights[range];
        if (!(weight > 0.0f))
          continue;
        // Rounding may leave the target unreached: the last range with
        // weight.
        range_drawn = range;
        range_target = target - cumulative;
        cumulative += weight;
        if (cumulative > target)
          break;
      }
      scratch.drawn_range = range_drawn;
      scratch.drawn_target = range_target;
      // Rounding may leave the target unreached in the range too: its last
      // token with weight, which the walk finds.
      scratch.drawn_token = scratch.range_last[range_drawn] - 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && !scratch.selected) {
    const uint walked = scratch.drawn_range;
    const uint walk_begin = min(walked * range_tokens, row.vocabulary);
    const uint walk_finish = min(walk_begin + range_tokens, row.vocabulary);
    const float target = scratch.drawn_target;
    float cumulative = 0.0f;
    uint last = 0;
    bool hit_found = false;
    for (uint first = walk_begin; first < walk_finish && !hit_found; first += 32) {
      const uint token = first + lane;
      bool excluded = false;
      for (uint i = 0; scratch.excluding && i < kDrawCandidates; ++i)
        excluded = excluded || scratch.excluded[i] == token;
      const float weight =
          token < walk_finish && !excluded ? kept_weight(row, end, token) : 0.0f;
      last = max(last, simd_max(weight > 0.0f ? token + 1 : 0u));
      const float prefix = simd_prefix_inclusive_sum(weight);
      const uint hit =
          simd_min(weight > 0.0f && cumulative + prefix > target ? lane : 32u);
      if (hit < 32) {
        if (lane == 0)
          scratch.drawn_token = first + hit;
        hit_found = true;
      }
      cumulative += simd_broadcast(prefix, 31);
    }
    // Excluding candidates, the range's last token with weight may be one of
    // them: the last the walk weighed.
    if (!hit_found && scratch.excluding && lane == 0 && last)
      scratch.drawn_token = last - 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  drawn = scratch.drawn_token;
  split_release(arrivals, thread_index);
  return true;
}

// Where a sampled row's distribution ends: its shards' masses, merged in
// shard order into the row's largest admitted logit, softmax denominator and
// admitted count, then the search. The record keeps the maximum and the
// end, which the draw reads.
inline void search_row(TargetRow row, device const TargetShardMass *masses,
                       float min_p, uint top_k, float top_p,
                       device TargetVocabularyRow &record,
                       threadgroup VocabularyScratch &scratch,
                       uint thread_index, uint lane, uint simd_group) {
  const TargetShardMass merged =
      merge_masses(masses, SPLASH_TARGET_SAMPLING_SHARDS, row.temperature);
  row.maximum = merged.maximum;
  const OrderBoundary end =
      distribution_end(row, min_p, top_k, top_p, merged.sum, merged.admitted,
                       scratch, thread_index, lane, simd_group);
  if (thread_index == 0)
    record = {merged.maximum, end.key, end.last};
}

// Where the distribution of each selected row of the sampled lanes ends, one
// group per row.
kernel void decode_sample_vocabulary_search(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetShardMass *partial_masses [[buffer(2)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant TargetSamplingParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup VocabularyScratch scratch;
  const uint batch = group.y, index = group.x;
  if (!selects(params, lane_rows, batch, index, true))
    return;
  const uint s = selected_logits_row(lane_rows, batch, index);
  search_row(selected_row(logits, token_mask, params, lane_rows, batch, index),
             partial_masses + ulong(s) * SPLASH_TARGET_SAMPLING_SHARDS,
             params.min_p[batch], params.top_k[batch], params.top_p[batch],
             vocabulary_rows[s], scratch, thread_index, lane, simd_group);
}

// The selection of each selected row of the sampled lanes, kVocabularyGroups
// groups per row: row i of a lane draws with the lane's uniform i, and a
// decode step's verify row (params.trees) tests the children its lane's
// draft drew there (sources, its row of the lane's tree, whose draws and next
// position's probabilities the selector wrote, and that position's
// candidates) with the lane's uniforms SPLASH_TARGET_VERIFY_ROWS +
// i * SPLASH_TREE_CHILDREN on.
template <uint Candidates>
kernel void decode_sample_vocabulary_draw(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(5)]],
    device atomic_uint *arrivals [[buffer(6)]],
    device const SplashLaneRows *lane_rows [[buffer(7)]],
    device const SplashTreeRow *tree [[buffer(8)]],
    device const uint *sources [[buffer(9)]],
    device const uint *candidates [[buffer(10)]],
    device const uchar *draws [[buffer(11)]],
    device const float *next [[buffer(12)]],
    constant TargetSamplingParams &params [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kDrawCandidates = Candidates;
  threadgroup DrawScratch scratch;
  const uint batch = group.y, index = group.x / kVocabularyGroups;
  if (!selects(params, lane_rows, batch, index, true))
    return;
  const uint s = selected_logits_row(lane_rows, batch, index);
  const TargetVocabularyRow record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, lane_rows, batch, index);
  row.maximum = record.maximum;
  device const float *lane_uniforms = uniforms + ulong(batch) * SPLASH_SAMPLING_UNIFORMS;
  DrawChildren children{nullptr, nullptr, nullptr, nullptr};
  if (params.trees) {
    const uint source = sources[s], depth = tree[s].depth;
    if (source != SPLASH_NO_TOKEN && depth < params.draft_positions) {
      const ulong at = ulong(batch) * SPLASH_TARGET_VERIFY_ROWS + source;
      children = {candidates + (ulong(batch) * params.draft_positions + depth) * kDrawCandidates,
                  next + at * kDrawCandidates, draws + at * SPLASH_TREE_CHILDREN,
                  lane_uniforms + SPLASH_TARGET_VERIFY_ROWS + index * SPLASH_TREE_CHILDREN};
    }
  }
  uint token;
  if (vocabulary_draw<Candidates>(row, {record.end_key, record.end_last}, lane_uniforms[index], children,
                      ranges + ulong(s) * kVocabularyRanges, arrivals + s,
                      group.x % kVocabularyGroups, scratch, thread_index, lane,
                      simd_group, token) &&
      thread_index == 0)
    tokens[s] = token;
}

template [[host_name("decode_sample_vocabulary_draw")]] kernel decltype(decode_sample_vocabulary_draw<16>) decode_sample_vocabulary_draw<16>;
template [[host_name("decode_sample_vocabulary_draw_k32")]] kernel decltype(decode_sample_vocabulary_draw<32>) decode_sample_vocabulary_draw<32>;

// Whether no token can be selected for a logit: NaN has no order and -inf
// no probability. A bit test, which Metal's fast math keeps.
inline bool unselectable_logit(float value) {
  const uint bits = as_type<uint>(value);
  return (bits & 0x7fffffffu) > 0x7f800000u || bits == 0xff800000u;
}

// The sampling penalties of one logit: repetition divides a positive
// logit and multiplies a negative one when the prompt or the output holds
// the token, and presence and frequency lower it by the output's count of
// the token. The result saturates, so no rewritten logit is infinite; an
// unselectable logit stays as it is, since saturating would make it
// selectable.
inline float penalize_logit(float value, uint count, float repetition,
                            float repetition_inverse, float presence,
                            float frequency) {
  if (unselectable_logit(value))
    return value;
  value *= value > 0.0f ? repetition_inverse : repetition;
  if (count)
    value -= frequency * float(count) + presence;
  return clamp(value, -FLT_MAX, FLT_MAX);
}

// The selected rows of each penalized entry, one thread per vocabulary token
// and entry, rewritten in place. Their contexts add the entry's draft tokens
// along their paths: bit r of drafted is set when the entry's verify input
// row r (from its first row) holds the token, a draft token that the rows
// below row r in its block's tree add to the output, so a row counts the bits
// of its ancestors but its anchor (metal/abi/VerifyTree.h) on top of the
// table's count. A selection of one row (the first token after a prompt) adds
// none.
kernel void decode_sample_penalize(
    device float *logits [[buffer(0)]], device const uint *words [[buffer(1)]],
    device const uint *input_tokens [[buffer(2)]],
    device const SplashTreeRow *tree [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant SamplingPenaltyParams &params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint token = position.x;
  const uint entry = position.y;
  if (token >= params.vocabulary || entry >= params.entries)
    return;
  const SplashLaneRows lane = lane_rows[params.lane[entry]];
  const uint rows = lane.rows;
  device const uint *inputs = input_tokens + lane.first;
  uint drafted = 0;
  for (uint row = 1; row < rows; ++row)
    drafted |= uint(inputs[row] == token) << row;
  const uint word =
      words[ulong(params.table_row[entry]) * params.vocabulary + token];
  if (!word && !drafted)
    return;
  device float *column = logits + ulong(lane.first) * params.vocabulary + token;
  for (uint row = 0; row < rows; ++row) {
    const uint path = tree[lane.first + row].ancestors;
    const uint count = (word & SPLASH_PENALTY_COUNT_MASK) + popcount(drafted & path & ~1u);
    if (!count && !(word & SPLASH_PENALTY_PROMPT_BIT))
      continue;
    device float &logit = column[ulong(row) * params.vocabulary];
    logit = penalize_logit(logit, count, params.repetition[entry],
                           params.repetition_inverse[entry],
                           params.presence[entry], params.frequency[entry]);
  }
}

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already
// checked against the last slot; the unrolled shift keeps every index static.
template <uint Count>
inline void top16_insert(thread float (&values)[Count], thread uint (&ids)[Count],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = Count - 1; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best,
                           thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// One shard of one proposal row. Threads stream their share of the shard in
// 16-byte vectors, keep the chunk in registers, and first find the 16th
// largest of the per-thread maxima: at least sixteen tokens are that large,
// so nothing below it can be in the row's top-16 and the exact sorted insert
// only runs for the few survivors. The group then pops its best sixteen in
// rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition; it ranks the
// row's columns and writes their tokens (SelectorBatchParams), whose order
// is theirs.
template <uint Candidates>
kernel void draft_select_top16_sharded(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant SelectorBatchParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint Rows = params.rows;
  const uint Positions = params.positions;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = Candidates;
  constexpr uint VectorTokens = 4, ChunkVectors = 16;
  const uint vocabulary = params.columns;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start = (ulong(batch) * Rows + position + 1) * vocabulary;
  uint shard_tokens = (vocabulary + Shards - 1) / Shards;
  uint begin = min(shard * shard_tokens, vocabulary);
  uint end = min(begin + shard_tokens, vocabulary);
  // Vector loads require 16-byte alignment. Handle the shard's unaligned head
  // and tail with scalar inserts; the logits binding must also be aligned.
  uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) %
                   VectorTokens);
  head = min(head, end - begin);
  uint vectors = (end - begin - head) / VectorTokens;
  uint vector_begin = begin + head;
  uint tail_begin = vector_begin + vectors * VectorTokens;
  device const float *row = logits + row_start;
  device const float4 *vector_row =
      reinterpret_cast<device const float4 *>(row + vector_begin);

  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  if (thread_index < head) {
    uint token = begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }
  if (thread_index < end - tail_begin) {
    uint token = tail_begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }

  threadgroup float maxima[256];
  threadgroup float thresholds[8];
  for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
    float4 loaded[ChunkVectors];
    float best = -INFINITY;
    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      loaded[i] = index < vectors ? vector_row[index] : float4(0.0f);
      if (index < vectors) {
        for (uint j = 0; j < VectorTokens; ++j)
          if (loaded[i][j] > best)
            best = loaded[i][j];
      }
    }
    // The 16th largest thread maximum: the minimum over the maxima that
    // fewer than sixteen others exceed.
    maxima[thread_index] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint other = 0; other < 256; ++other)
      above += maxima[other] > best ? 1u : 0u;
    float candidate = simd_min(above < K ? best : INFINITY);
    if (lane == 0)
      thresholds[simd_group] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threshold = thresholds[0];
    for (uint other = 1; other < 8; ++other)
      threshold = min(threshold, thresholds[other]);

    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      if (index >= vectors)
        continue;
      uint token = vector_begin + index * VectorTokens;
      for (uint j = 0; j < VectorTokens; ++j) {
        float value = loaded[i][j];
        if (value >= threshold &&
            top_beats(value, token + j, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, value, token + j);
      }
    }
  }

  // Sixteen rounds pop the group-wide best head; the simdgroup bests
  // alternate between two slots so one barrier per round suffices.
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  for (uint rank = 0; rank < K; ++rank) {
    float head_value = values[0];
    uint head_id = ids[0];
    float simd_value;
    uint simd_id;
    simd_best_head(head_value, head_id, simd_value, simd_id);
    uint slot = rank & 1;
    if (lane == 0) {
      round_values[slot][simd_group] = simd_value;
      round_ids[slot][simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = round_values[slot][0];
    uint best_id = round_ids[slot][0];
    for (uint other = 1; other < 8; ++other) {
      float value = round_values[slot][other];
      uint token = round_ids[slot][other];
      if (top_beats(value, token, best, best_id)) {
        best = value;
        best_id = token;
      }
    }
    if (thread_index == rank) {
      // The empty sentinel stays one.
      partial_ids[group * K + rank] = best_id < params.leading || best_id == 0xffffffffu
                                          ? best_id
                                          : best_id - params.leading + params.trailing;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}
template [[host_name("draft_select_top16_sharded")]] kernel decltype(draft_select_top16_sharded<16>) draft_select_top16_sharded<16>;
template [[host_name("draft_select_top16_sharded_k32")]] kernel decltype(draft_select_top16_sharded<32>) draft_select_top16_sharded<32>;


// Merges the eight shard partials of one row into the simdgroup's lanes: each
// lane holds four consecutive entries of one shard's sorted list and sixteen
// rounds pop the simdgroup-wide best into rank order, lane `rank` keeping it.
template <uint Candidates>
inline void top16_merge_shards(device const uint *partial_ids,
                               device const float *partial_values,
                               uint row, uint lane, thread float &value,
                               thread uint &token) {
  constexpr uint K = Candidates;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      top_pop<Entries>(values, ids);
  }
}

// One group per proposal row. Simdgroup 0 merges its candidates and unary
// scores; simdgroup 1 merges the preceding row (the anchor for row 0). The
// remaining simdgroups score the 16 x 16 predecessor/candidate edge table in
// fixed per-lane reduction order. The table follows the shard partials in
// partial-values scratch.
template <uint CandidateCount>
kernel void draft_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device uint *candidates [[buffer(2)]],
    device float *unary [[buffer(3)]],
    device const bfloat *hidden [[buffer(4)]],
    device const bfloat *predecessor_codebook [[buffer(5)]],
    device const bfloat *successor_codebook [[buffer(6)]],
    constant SelectorBatchParams &params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint Rows = params.rows;
  const uint Positions = params.positions;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = CandidateCount;
  constexpr uint Rank = SPLASH_DRAFT_SELECTOR_RANK;
  uint batch = row / Positions;
  uint position = row % Positions;
  threadgroup uint successors[Candidates];
  threadgroup uint predecessors[Candidates];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards<Candidates>(partial_ids, partial_values, row, lane, value, token);
    if (lane < Candidates) {
      candidates[row * Candidates + lane] = token;
      unary[row * Candidates + lane] = value;
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = params.anchor[batch];
    if (position > 0) {
      float value;
      top16_merge_shards<Candidates>(partial_ids, partial_values, row - 1, lane, value,
                         token);
    }
    if (lane < Candidates)
      predecessors[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One task is one predecessor and eight of the sixteen candidates, so a
  // simdgroup issues all its codebook loads at once instead of one cold row
  // per edge.
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  uint tasks = (position > 0 ? Candidates : 1) * (Candidates / TaskCandidates);
  device float *table = partial_values +
                        ulong(params.lanes) * Positions * Shards * Candidates +
                        ulong(row) * Candidates * Candidates;
  device const bfloat *row_hidden =
      hidden + (ulong(batch) * Rows + position + 1) * Rank;
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    uint predecessor_index = task / (Candidates / TaskCandidates);
    uint first_candidate =
        task % (Candidates / TaskCandidates) * TaskCandidates;
    // Ids are produced by the top-k selection and are always in range; the
    // clamp only keeps a corrupted id inside the codebooks.
    uint safe_predecessor =
        min(predecessors[predecessor_index], params.vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] = float(predecessor_codebook[safe_predecessor * Rank + dim]) *
                   float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      uint safe_candidate =
          min(successors[first_candidate + j], params.vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] =
            float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * Candidates + first_candidate + j] = score;
    }
  }
}
template [[host_name("draft_select_edges")]] kernel decltype(draft_select_edges<16>) draft_select_edges<16>;
template [[host_name("draft_select_edges_k32")]] kernel decltype(draft_select_edges<32>) draft_select_edges<32>;


// The best-first tree of draft_select_tree (DraftTreeParams). An entry is a
// step the tree may take, from its parent row to a token one deeper, which
// grows the tree in the order of its score, the estimate of the path it ends,
// the earlier entry first among equal ones.
constant constexpr uint kTreeChildren = SPLASH_TREE_CHILDREN;
#define TREE_NAMESPACE tree16
#define TREE_KERNEL draft_select_tree
#define TREE_CANDIDATES 16
#define TREE_POSITIONS 7
#define TREE_STORAGE threadgroup
#include "metal/kernels/common/draft_tree.h"
#undef TREE_STORAGE
#undef TREE_POSITIONS
#undef TREE_CANDIDATES
#undef TREE_KERNEL
#undef TREE_NAMESPACE
#define TREE_NAMESPACE tree32
#define TREE_KERNEL draft_select_tree_k32
#define TREE_CANDIDATES 32
#define TREE_POSITIONS 15
#define TREE_STORAGE device
#include "metal/kernels/common/draft_tree.h"
#undef TREE_STORAGE
#undef TREE_POSITIONS
#undef TREE_CANDIDATES
#undef TREE_KERNEL
#undef TREE_NAMESPACE

// One thread lays out a decode cycle's rows (VerifyRowsParams) from the
// trees the selector grew, lane l's sizes[l] rows from row l *
// SPLASH_TARGET_VERIFY_ROWS of tree_tokens, trees and scores: each lane's
// rows, and its rows' tokens, tree rows and sources (the row of its lane's
// tree each is), packed in lane order, and after them the rows of no lane,
// each lane 0's anchor and no source (SPLASH_NO_TOKEN). A lane's rows are
// numbered in preorder, the earlier-taken (likelier) child first, so that
// the likeliest path of its tree lies on its first rows.
kernel void draft_verify_rows(device const uint *tree_tokens [[buffer(0)]],
                              device const SplashTreeRow *trees [[buffer(1)]],
                              device const float2 *scores [[buffer(2)]],
                              device const uint *sizes [[buffer(3)]],
                              device uint *tokens [[buffer(4)]],
                              device SplashTreeRow *tree [[buffer(5)]],
                              device SplashLaneRows *lane_rows [[buffer(6)]],
                              device uint *sources [[buffer(7)]],
                              constant VerifyRowsParams &params [[buffer(8)]],
                              uint index [[thread_position_in_grid]]) {
  constexpr uint Lanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  if (index != 0)
    return;
  // Every lane's rows past its anchor by estimate, a lane's earlier row first
  // among equal ones.
  float estimates[Lanes * Rows];
  uint owners[Lanes * Rows];
  uint count = 0;
  for (uint lane = 0; lane < params.lanes; ++lane) {
    for (uint row = 1; row < sizes[lane]; ++row) {
      const float estimate = scores[lane * Rows + row].y;
      uint at = count++;
      for (; at > 0 && estimates[at - 1] < estimate; --at) {
        estimates[at] = estimates[at - 1];
        owners[at] = owners[at - 1];
      }
      estimates[at] = estimate;
      owners[at] = lane;
    }
  }
  // The best of them that the cycle's rows and the lanes' cap hold, none
  // below the floor: they fall, so the first below it ends them.
  uint rows[Lanes];
  for (uint lane = 0; lane < params.lanes; ++lane)
    rows[lane] = 1;
  uint taken = params.lanes;
  for (uint i = 0; i < count && taken < params.rows; ++i) {
    if (estimates[i] < params.floor)
      break;
    if (rows[owners[i]] == params.lane_cap)
      continue;
    ++rows[owners[i]];
    ++taken;
  }
  uint first = 0;
  for (uint lane = 0; lane < params.lanes; ++lane) {
    device const uint *token = tree_tokens + lane * Rows;
    uint parent[Rows];
    for (uint row = 0; row < rows[lane]; ++row)
      parent[row] = trees[lane * Rows + row].parent;
    // Preorder: a row's children follow it, each with its subtree, in the
    // order the tree took them.
    uint order[Rows], placed[Rows], stack[Rows];
    uint size = 0, next = 0;
    stack[size++] = 0;
    while (size) {
      const uint row = stack[--size];
      placed[row] = next;
      order[next++] = row;
      for (uint child = rows[lane]; child-- > row + 1;)
        if (parent[child] == row)
          stack[size++] = child;
    }
    uint ancestors[Rows], depths[Rows];
    for (uint at = 0; at < rows[lane]; ++at) {
      const uint row = order[at];
      const uint up = at ? placed[parent[row]] : uint(SPLASH_TREE_ROOT);
      ancestors[at] = (at ? ancestors[up] : 0u) | (1u << at);
      depths[at] = at ? depths[up] + 1 : 0u;
      tokens[first + at] = token[row];
      tree[first + at] = {ancestors[at], ushort(up), ushort(depths[at])};
      sources[first + at] = row;
    }
    lane_rows[lane] = {first, rows[lane]};
    first += rows[lane];
  }
  // The rows of no lane (VerifyRowsParams), which the embedding gathers by
  // their token, on no path and of no parent: none of an earlier cycle's.
  for (uint row = first; row < params.rows; ++row) {
    tokens[row] = tree_tokens[0];
    tree[row] = {0u, ushort(SPLASH_TREE_ROOT), ushort(0)};
    sources[row] = SPLASH_NO_TOKEN;
  }
}

// One shard of a greedy row: its admitted token with the largest selectable
// logit, the lowest id among ties, per thread in token order and then over
// the group. A row with no selectable logit keeps the sentinel id.
inline void argmax_shard(TargetRow row, uint shard, device float &partial_value,
                         device uint &partial_index,
                         threadgroup float *group_values,
                         threadgroup uint *group_indices, uint thread_index) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    float value = row.logits[token];
    if (unselectable_logit(value))
      continue;
    if (value > best || (value == best && token < best_index)) {
      best = value;
      best_index = token;
    }
  }
  uint lane = thread_index & 31;
  uint simd_group = thread_index >> 5;
  float simd_best = simd_max(best);
  uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
  if (lane == 0) {
    group_values[simd_group] = simd_best;
    group_indices[simd_group] = simd_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float value = lane < 8 ? group_values[lane] : -INFINITY;
    float group_best = simd_max(value);
    uint index =
        lane < 8 && value == group_best ? group_indices[lane] : 0xffffffffu;
    index = simd_min(index);
    if (lane == 0) {
      partial_value = group_best;
      partial_index = index;
    }
  }
}

// One shard of each selected row of the greedy lanes.
kernel void decode_sample_argmax_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device uint *partial_indices [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant TargetSamplingParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  const uint batch = group.y, index = group.x / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!selects(params, lane_rows, batch, index, false))
    return;
  const uint shard = group.x % SPLASH_TARGET_SAMPLING_SHARDS;
  const ulong partial =
      ulong(selected_logits_row(lane_rows, batch, index)) *
          SPLASH_TARGET_SAMPLING_SHARDS +
      shard;
  argmax_shard(selected_row(logits, token_mask, params, lane_rows, batch, index), shard,
               partial_values[partial], partial_indices[partial],
               group_values, group_indices, thread_index);
}

// A greedy row's token from its shards' partials.
inline uint argmax_reduce(device const float *partial_values,
                          device const uint *partial_indices, uint row,
                          uint lane) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float value = lane < Shards ? partial_values[row * Shards + lane] : -INFINITY;
  float best = simd_max(value);
  uint index = lane < Shards && value == best
                   ? partial_indices[row * Shards + lane]
                   : 0xffffffffu;
  return simd_min(index);
}

// The token of each selected row of the greedy lanes.
kernel void decode_sample_argmax_reduce(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    device const SplashLaneRows *lane_rows [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint batch = group.y, index = group.x;
  if (!selects(params, lane_rows, batch, index, false))
    return;
  const uint s = selected_logits_row(lane_rows, batch, index);
  const uint token = argmax_reduce(partial_values, partial_indices, s, lane);
  if (lane == 0)
    tokens[s] = token;
}

// After the selection, the thinking cutoff of each lane that has one
// (ThinkingCutoffParams): each row whose path reaches the path's cut, its row
// at the depth of the lane's row or, where the lane's end waits for its mask,
// the first from there whose mask admits the end's first token, selects the
// end token of its depth past the cut. A row's path up to its cut, its anchor
// excepted, holds the draft tokens before the cut.
kernel void decode_sample_thinking_cutoff(
    device const uint *input_tokens [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const SplashTreeRow *tree [[buffer(2)]],
    device uint *tokens [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant ThinkingCutoffParams &params [[buffer(5)]],
    device const uint *marker0 [[buffer(6)]], device const uint *marker1 [[buffer(7)]],
    device const uint *marker2 [[buffer(8)]], device const uint *marker3 [[buffer(9)]],
    uint lane [[threadgroup_position_in_grid]]) {
  const uint rows = lane_rows[lane].rows, first = lane_rows[lane].first;
  const bool waits = (params.waiting_mask & (1u << lane)) != 0;
  // The lane's mask rows, as selected_row reads them.
  device const uint *masks =
      token_mask + (ulong(lane) * (SPLASH_TARGET_VERIFY_ROWS + 1) +
                    params.mask_row) *
                       params.mask_words;
  if (params.automaton_mask & (1u << lane)) {
    // Device addressing keeps dynamically indexed marker reads stable under
    // shader validation; setBytes still supplies the graph-owned program.
    device const uint *markers = lane == 0 ? marker0 : lane == 1 ? marker1 : lane == 2 ? marker2 : marker3;
    for (uint at = 0; at < rows; ++at) {
      uint state = params.marker_state[lane], remaining = params.row[lane];
      uint cut = SPLASH_TARGET_VERIFY_ROWS;
      uint suffixOffset = 0;
      bool programSuffix = false;
      uint count = params.count[lane];
      const uint depth = tree[first + at].depth;
      for (uint walk = tree[first + at].ancestors; walk; walk &= walk - 1) {
        const uint node = ctz(walk), current = tree[first + node].depth;
        if (node) {
          const bool opened = state >= markers[2];
          const uint token = input_tokens[first + node], entries = markers[7];
          uint low = 0, high = markers[6];
          while (low < high) {
            const uint middle = (low + high) / 2;
            if ((markers[entries + middle] >> 8) < token) low = middle + 1;
            else high = middle;
          }
          const uint kind = low < markers[6] && (markers[entries + low] >> 8) == token
                                ? markers[entries + low] & 255 : 0;
          const uint index = state * markers[1] + kind;
          state = (markers[8 + index / 4] >> (8 * (index % 4))) & 255;
          if (opened && remaining) --remaining;
        }
        if (state >= markers[0]) break;
        if (state < markers[2] || remaining) continue;
        if (!(params.opened_mask & (1u << lane)) || current || params.row[lane]) {
          const uint matrix = (markers[0] * markers[1] + 3) / 4;
          const uint record = markers[8 + matrix + state];
          count = markers[record];
          suffixOffset = record + 1;
          programSuffix = true;
        }
        const uint token = programSuffix ? markers[suffixOffset] : params.tokens[lane][0];
        if (waits && !(masks[node * params.mask_words + token / 32] & (1u << (token % 32)))) continue;
        cut = current;
        break;
      }
      if (cut <= depth && depth - cut < count && tokens[first + at] < params.vocabulary)
        tokens[first + at] = programSuffix ? markers[suffixOffset + depth - cut] : params.tokens[lane][depth - cut];
    }
    return;
  }
  const uint start = params.tokens[lane][0];
  for (uint at = 0; at < rows; ++at) {
    const uint path = tree[first + at].ancestors;
    const uint depth = tree[first + at].depth;
    // The path's rows ascend from its anchor, one a depth; one past the
    // row's depth where the path holds no cut.
    uint cut = 0;
    for (uint walk = path; walk; walk &= walk - 1, ++cut)
      if (cut >= params.row[lane] &&
          (!waits || (masks[ctz(walk) * params.mask_words + start / 32] &
                      (1u << (start % 32)))))
        break;
    if (cut > depth || depth - cut >= params.count[lane])
      continue;
    bool ended = false;
    for (uint walk = path & ~1u; walk; walk &= walk - 1) {
      const uint before = ctz(walk);
      const uint draft = input_tokens[first + before];
      ended = ended || (tree[first + before].depth <= cut &&
                        (draft == params.close[lane] || draft == params.also_ends[lane]));
    }
    // A row with no selectable logit keeps its sentinel, so the lane fails.
    if (!ended && tokens[first + at] < params.vocabulary)
      tokens[first + at] = params.tokens[lane][depth - cut];
  }
}

// Each lane's acceptance (AcceptBatchParams), one thread a lane. Its retained
// tokens, the selections of the path's rows, go to the output tokens from its
// first row, each read before an earlier path row's is written there.
kernel void decode_accept_tree(
    device const uint *input_tokens [[buffer(0)]],
    device const SplashTreeRow *tree [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    device uint *retained [[buffer(3)]],
    device uint *accepted_count [[buffer(4)]],
    device uint *retained_rows [[buffer(5)]],
    device const SplashLaneRows *lane_rows [[buffer(6)]],
    constant AcceptBatchParams &params [[buffer(7)]],
    uint batch [[threadgroup_position_in_grid]]) {
  const uint first = lane_rows[batch].first, rows = lane_rows[batch].rows;
  device const uint *inputs = input_tokens + first;
  device const SplashTreeRow *nodes = tree + first;
  device uint *selected = tokens + first;
  uint path[SPLASH_TARGET_VERIFY_ROWS];
  path[0] = 0;
  uint accepted = 0;
  for (uint row = 0;;) {
    const uint token = selected[row];
    uint next = 0;
    for (uint child = row + 1; child < rows && !next; ++child)
      next = nodes[child].parent == row && inputs[child] == token ? child : 0;
    if (!next)
      break;
    path[++accepted] = next;
    row = next;
  }
  uint kept = min(accepted + 1, params.remaining[batch]);
  for (uint i = 0; i < kept; ++i) {
    const uint token = selected[path[i]];
    selected[i] = token;
    if (stop_token(params.stop_tokens, token))
      kept = i + 1;
  }
  // Where the lane's cap or a stop token cuts its path, every token it
  // retains is a proposal it accepted.
  accepted_count[batch] = min(accepted, kept);
  retained[batch] = kept;
  retained_rows[batch] = nodes[path[kept - 1]].ancestors;
}
