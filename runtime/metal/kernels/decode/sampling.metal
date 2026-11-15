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
// the lane has it and takes the policy the kernel runs (sampled or greedy),
// and its logits row s, which indexes the workspaces and outputs.
inline bool selects(constant TargetSamplingParams &params, uint lane,
                    uint index, bool sampled) {
  return index < params.rows[lane] &&
         ((params.sampling_mask & (1u << lane)) != 0) == sampled;
}
inline uint selected_logits_row(constant TargetSamplingParams &params,
                                uint lane, uint index) {
  return params.first_row[lane] + index;
}

// Row `index` of lane `lane` of a dispatch (TargetSamplingParams).
inline TargetRow selected_row(device const float *logits,
                              device const uint *token_mask,
                              constant TargetSamplingParams &params, uint lane,
                              uint index) {
  return {logits + ulong(selected_logits_row(params, lane, index)) *
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
    constant TargetSamplingParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup TargetShardMass group_masses[8];
  const uint batch = group.y, index = group.x / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!selects(params, batch, index, true))
    return;
  const uint shard = group.x % SPLASH_TARGET_SAMPLING_SHARDS;
  shard_mass(selected_row(logits, token_mask, params, batch, index), shard,
             partial_masses[ulong(selected_logits_row(params, batch, index)) *
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

// Draws the row's token from its distribution, shared by its groups; returns
// true in the group that finishes last, with the drawn token. Each kept token
// weighs its softmax weight. Each simdgroup of the row's groups sums one range
// of the vocabulary; the last group adds the ranges in order and walks the
// one that holds the draw with prefix sums. A range's weight and its walk add
// the same token weights, so the range the draw picks always has a token to
// draw; a row that keeps no token draws past the vocabulary's end, the
// sentinel of a row with no selectable logit.
inline bool vocabulary_draw(
    TargetRow row, OrderBoundary end, float uniform,
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
    const float target = uniform * total;
    float cumulative = 0.0f;
    uint range_drawn = 0;
    float range_target = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range) {
      const float weight = scratch.range_weights[range];
      if (!(weight > 0.0f))
        continue;
      // Rounding may leave the target unreached: the last range with weight.
      range_drawn = range;
      range_target = target - cumulative;
      cumulative += weight;
      if (cumulative > target)
        break;
    }
    scratch.drawn_range = range_drawn;
    scratch.drawn_target = range_target;
    // Rounding may leave the target unreached in the range too: its last
    // token with weight.
    scratch.drawn_token = scratch.range_last[range_drawn] - 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const uint walked = scratch.drawn_range;
    const uint walk_begin = min(walked * range_tokens, row.vocabulary);
    const uint walk_finish = min(walk_begin + range_tokens, row.vocabulary);
    const float target = scratch.drawn_target;
    float cumulative = 0.0f;
    for (uint first = walk_begin; first < walk_finish; first += 32) {
      const uint token = first + lane;
      const float weight =
          token < walk_finish ? kept_weight(row, end, token) : 0.0f;
      const float prefix = simd_prefix_inclusive_sum(weight);
      const uint hit =
          simd_min(weight > 0.0f && cumulative + prefix > target ? lane : 32u);
      if (hit < 32) {
        if (lane == 0)
          scratch.drawn_token = first + hit;
        break;
      }
      cumulative += simd_broadcast(prefix, 31);
    }
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
    constant TargetSamplingParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup VocabularyScratch scratch;
  const uint batch = group.y, index = group.x;
  if (!selects(params, batch, index, true))
    return;
  const uint s = selected_logits_row(params, batch, index);
  search_row(selected_row(logits, token_mask, params, batch, index),
             partial_masses + ulong(s) * SPLASH_TARGET_SAMPLING_SHARDS,
             params.min_p[batch], params.top_k[batch], params.top_p[batch],
             vocabulary_rows[s], scratch, thread_index, lane, simd_group);
}

// The draw of each selected row of the sampled lanes, kVocabularyGroups
// groups per row: row i of a lane draws with the lane's uniform i.
kernel void decode_sample_vocabulary_draw(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(5)]],
    device atomic_uint *arrivals [[buffer(6)]],
    constant TargetSamplingParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  const uint batch = group.y, index = group.x / kVocabularyGroups;
  if (!selects(params, batch, index, true))
    return;
  const uint s = selected_logits_row(params, batch, index);
  const TargetVocabularyRow record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, batch, index);
  row.maximum = record.maximum;
  uint token;
  if (vocabulary_draw(row, {record.end_key, record.end_last},
                      uniforms[ulong(batch) * SPLASH_SAMPLING_UNIFORMS + index],
                      ranges + ulong(s) * kVocabularyRanges, arrivals + s,
                      group.x % kVocabularyGroups, scratch, thread_index, lane,
                      simd_group, token) &&
      thread_index == 0)
    tokens[s] = token;
}

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
    constant SamplingPenaltyParams &params [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint token = position.x;
  const uint entry = position.y;
  if (token >= params.vocabulary || entry >= params.entries)
    return;
  const uint rows = params.rows[entry];
  device const uint *inputs = input_tokens + params.first_row[entry];
  uint drafted = 0;
  for (uint row = 1; row < rows; ++row)
    drafted |= uint(inputs[row] == token) << row;
  const uint word =
      words[ulong(params.table_row[entry]) * params.vocabulary + token];
  if (!word && !drafted)
    return;
  device float *column =
      logits + ulong(params.first_row[entry]) * params.vocabulary + token;
  for (uint row = 0; row < rows; ++row) {
    const uint path = rows > 1 ? tree[params.first_row[entry] + row].ancestors : 1u;
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
inline void top16_insert(thread float (&values)[16], thread uint (&ids)[16],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
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
// same set in the same order whatever the thread partition.
kernel void draft_select_top16_sharded(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant uint &vocabulary [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = SPLASH_DRAFT_CANDIDATES;
  constexpr uint VectorTokens = 4, ChunkVectors = 16;
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
      partial_ids[group * K + rank] = best_id;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}

// Merges the eight shard partials of one row into the simdgroup's lanes: each
// lane holds four consecutive entries of one shard's sorted list and sixteen
// rounds pop the simdgroup-wide best into rank order, lane `rank` keeping it.
inline void top16_merge_shards(device const uint *partial_ids,
                               device const float *partial_values,
                               uint row, uint lane, thread float &value,
                               thread uint &token) {
  constexpr uint K = SPLASH_DRAFT_CANDIDATES;
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
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  constexpr uint Rank = SPLASH_DRAFT_SELECTOR_RANK;
  uint batch = row / Positions;
  uint position = row % Positions;
  threadgroup uint successors[Candidates];
  threadgroup uint predecessors[Candidates];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
    if (lane < Candidates) {
      candidates[row * Candidates + lane] = token;
      unary[row * Candidates + lane] = value;
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = params.anchor[batch];
    if (position > 0) {
      float value;
      top16_merge_shards(partial_ids, partial_values, row - 1, lane, value,
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

// The best-first tree of draft_select_tree (DraftTreeParams). An entry is a
// step the tree may take, from its parent row to a token one deeper, which
// grows the tree in the order of its score, the estimate of the path it ends,
// the earlier entry first among equal ones.
constant constexpr uint kTreeChildren = SPLASH_TREE_CHILDREN;
// A lane's rows, a verify block's at most, each a bit of its descendants'
// ancestors masks (SplashTreeRow).
constant constexpr uint kTreeRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kTreeRows <= 32, "a lane's rows fit its ancestors' masks");
// An entry's info word: whether its token is a host chain's, its parent row,
// its proposal position (its depth less one) and its candidate's index.
constant constexpr uint kTreeChain = 1u << 31;
// A row, the anchor among them, pushes its draft children once, as a draft
// step makes it or joins it where a chain token made it, and one chain
// entry, so no more entries than these are ever pushed.
constant constexpr uint kTreeEntries = kTreeRows * (kTreeChildren + 1);
constant constexpr uchar kNoCandidate = 0xff;

// A lane's draft candidates at each proposal position, the probability of
// each after each predecessor candidate (the anchor for position 0) and the
// predecessor's kTreeChildren most likely.
struct TreeLattice {
  uint ids[SPLASH_DRAFT_PROPOSAL_TOKENS][SPLASH_DRAFT_CANDIDATES];
  float probability[SPLASH_DRAFT_PROPOSAL_TOKENS][SPLASH_DRAFT_CANDIDATES]
                   [SPLASH_DRAFT_CANDIDATES];
  uchar likeliest[SPLASH_DRAFT_PROPOSAL_TOKENS][SPLASH_DRAFT_CANDIDATES]
                 [kTreeChildren];
};

struct TreeHeap {
  float score[kTreeEntries];
  uint order[kTreeEntries];
  uint info[kTreeEntries];
  uint size;
  uint pushed;
};

// A lane's tree as it grows: its rows' tokens, ancestors' masks, parents,
// depths, candidate indices at their positions (kNoCandidate for a chain
// token the draft does not hold there), draft probabilities and estimates.
struct TreeRows {
  uint token[kTreeRows];
  uint ancestors[kTreeRows];
  ushort parent[kTreeRows];
  ushort depth[kTreeRows];
  uchar candidate[kTreeRows];
  float q[kTreeRows];
  float estimate[kTreeRows];
  uint made;
};

inline bool tree_entry_before(threadgroup TreeHeap &heap, uint a, uint b) {
  return heap.score[a] > heap.score[b] ||
         (heap.score[a] == heap.score[b] && heap.order[a] < heap.order[b]);
}

inline void tree_entry_swap(threadgroup TreeHeap &heap, uint a, uint b) {
  const float score = heap.score[a];
  const uint order = heap.order[a], info = heap.info[a];
  heap.score[a] = heap.score[b];
  heap.order[a] = heap.order[b];
  heap.info[a] = heap.info[b];
  heap.score[b] = score;
  heap.order[b] = order;
  heap.info[b] = info;
}

inline void tree_push(threadgroup TreeHeap &heap, float score, uint info) {
  uint at = heap.size++;
  heap.score[at] = score;
  heap.order[at] = heap.pushed++;
  heap.info[at] = info;
  while (at > 0 && tree_entry_before(heap, at, (at - 1) / 2)) {
    tree_entry_swap(heap, at, (at - 1) / 2);
    at = (at - 1) / 2;
  }
}

// Removes the first entry, which the caller has read.
inline void tree_pop(threadgroup TreeHeap &heap) {
  tree_entry_swap(heap, 0, --heap.size);
  for (uint at = 0;;) {
    uint first = at;
    for (uint child = 2 * at + 1; child <= 2 * at + 2; ++child)
      if (child < heap.size && tree_entry_before(heap, child, first))
        first = child;
    if (first == at)
      break;
    tree_entry_swap(heap, at, first);
    at = first;
  }
}

inline uint tree_info(bool chain, uint parent, uint position, uint index) {
  return (chain ? kTreeChain : 0u) | parent << 16 | position << 8 | index;
}

// The estimate of a draft step of probability q at proposal position p
// (DraftTreeParams), q kept off 0 and 1 for its logit.
inline float tree_draft_estimate(constant DraftTreeParams &params, float q,
                                 uint position) {
  q = clamp(q, 1.0e-6f, 1.0f - 1.0e-6f);
  const float z =
      params.slope * log(q / (1.0f - q)) + params.intercept[position];
  return 1.0f / (1.0f + exp(-z));
}

// A host chain token's agreement with the draft at its depth (0-based):
// the draft's most likely candidate there, another of its candidates, none,
// or past its positions.
inline uint tree_agreement(threadgroup TreeLattice &lattice, uint token,
                           uint depth) {
  if (depth >= SPLASH_DRAFT_PROPOSAL_TOKENS)
    return 3;
  if (token == lattice.ids[depth][0])
    return 0;
  for (uint i = 1; i < SPLASH_DRAFT_CANDIDATES; ++i)
    if (token == lattice.ids[depth][i])
      return 1;
  return 2;
}

// A draft row's entries for its most likely children at the next position.
inline void tree_expand(threadgroup TreeLattice &lattice,
                        threadgroup TreeHeap &heap,
                        constant DraftTreeParams &params, uint children,
                        uint row, uint position, uint predecessor,
                        float score) {
  for (uint child = 0; child < children; ++child) {
    const uint index = lattice.likeliest[position][predecessor][child];
    const float q = lattice.probability[position][predecessor][index];
    if (q > 0.0f)
      tree_push(heap, score * tree_draft_estimate(params, q, position),
                tree_info(false, row, position, index));
  }
}

// One group per lane builds its tree (DraftTreeParams). Its threads first
// take, for every proposal position and predecessor candidate (the anchor
// for position 0), the probabilities of the position's candidates (the
// softmax at temperature 1 of their unary scores plus the edges from the
// predecessor, the table draft_select_edges left in the partial-values
// scratch) and the predecessor's most likely, the lowest index first among
// equal scores, as the chain walk chose; thread 0 then grows the tree. Rows
// are written in the order the tree takes them, so a row's parent comes
// before it and its estimate is at most its parent's. A row's draft
// probability (scores' x) is that of its token among the candidates after its
// parent's, -1 where either is none of the draft's; its estimate is scores' y.
kernel void draft_select_tree(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *partial_values [[buffer(2)]],
    device const uint *chain_tokens [[buffer(3)]],
    device const float *chain_rates [[buffer(4)]],
    device uint *tokens [[buffer(5)]],
    device SplashTreeRow *tree [[buffer(6)]],
    device float2 *scores [[buffer(7)]],
    device uint *row_counts [[buffer(8)]],
    constant DraftTreeParams &params [[buffer(9)]],
    uint lane [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  threadgroup TreeLattice lattice;
  threadgroup TreeHeap heap;
  threadgroup TreeRows rows;

  const ulong origin = ulong(lane) * Positions * Candidates;
  if (thread_index < Positions * Candidates) {
    const uint position = thread_index / Candidates;
    const uint predecessor = thread_index % Candidates;
    lattice.ids[position][predecessor] = candidates[origin + thread_index];
    // Position 0 follows the anchor alone.
    if (position > 0 || predecessor == 0) {
      device const float *edges =
          partial_values +
          ulong(params.lanes) * Positions * Shards * Candidates +
          ((ulong(lane) * Positions + position) * Candidates + predecessor) *
              Candidates;
      float score[Candidates];
      for (uint i = 0; i < Candidates; ++i)
        score[i] = unary[origin + position * Candidates + i] + edges[i];
      float best = score[0];
      for (uint i = 1; i < Candidates; ++i)
        best = max(best, score[i]);
      float weight[Candidates];
      float mass = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        weight[i] = exp(score[i] - best);
        mass += weight[i];
      }
      for (uint i = 0; i < Candidates; ++i)
        lattice.probability[position][predecessor][i] = weight[i] / mass;
      uint taken = 0;
      for (uint child = 0; child < kTreeChildren; ++child) {
        uint selected = Candidates;
        for (uint i = 0; i < Candidates; ++i)
          if (!(taken & (1u << i)) &&
              (selected == Candidates || score[i] > score[selected]))
            selected = i;
        taken |= 1u << selected;
        lattice.likeliest[position][predecessor][child] = uchar(selected);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint lane_rows = params.rows[lane];
  if (thread_index == 0) {
    device const uint *chain =
        chain_tokens + ulong(lane) * SPLASH_TREE_CHAIN_TOKENS;
    device const float *rates = chain_rates + ulong(lane) *
                                                  SPLASH_TREE_CHAIN_TOKENS *
                                                  SPLASH_TREE_AGREEMENTS;
    const uint children = params.children[lane];
    const uint deepest = params.depth[lane], chained = params.chain[lane];
    heap.size = 0;
    heap.pushed = 0;
    rows.made = 1;
    rows.token[0] = params.anchor[lane];
    rows.ancestors[0] = 1;
    rows.parent[0] = SPLASH_TREE_ROOT;
    rows.depth[0] = 0;
    rows.candidate[0] = 0;
    rows.q[0] = 1.0f;
    rows.estimate[0] = 1.0f;
    if (deepest > 0) {
      tree_expand(lattice, heap, params, children, 0, 0, 0, 1.0f);
      if (chained)
        tree_push(heap, rates[tree_agreement(lattice, chain[0], 0)],
                  tree_info(true, 0, 0, 0));
    }
    while (heap.size && rows.made < lane_rows) {
      const float score = heap.score[0];
      const uint info = heap.info[0];
      tree_pop(heap);
      const bool from_chain = (info & kTreeChain) != 0;
      const uint parent = (info >> 16) & 0x7fffu;
      const uint position = (info >> 8) & 0xffu, index = info & 0xffu;
      const uint token =
          from_chain ? chain[position] : lattice.ids[position][index];
      // A token path the tree holds already gains no row.
      uint row = 0;
      for (uint other = 1; other < rows.made && !row; ++other)
        row = rows.parent[other] == parent && rows.token[other] == token
                  ? other
                  : 0;
      if (!row) {
        row = rows.made++;
        uchar own = from_chain ? kNoCandidate : uchar(index);
        for (uint i = 0; from_chain && position < Positions && i < Candidates;
             ++i)
          own = lattice.ids[position][i] == token && own == kNoCandidate
                    ? uchar(i)
                    : own;
        const uchar predecessor = rows.candidate[parent];
        rows.token[row] = token;
        rows.ancestors[row] = rows.ancestors[parent] | (1u << row);
        rows.parent[row] = ushort(parent);
        rows.depth[row] = ushort(position + 1);
        rows.candidate[row] = own;
        rows.q[row] = own != kNoCandidate && predecessor != kNoCandidate
                          ? lattice.probability[position][predecessor][own]
                          : -1.0f;
        rows.estimate[row] = score;
      }
      // The row's children lie a step deeper.
      if (position + 1 >= deepest)
        continue;
      if (!from_chain && position + 1 < Positions)
        tree_expand(lattice, heap, params, children, row, position + 1, index,
                    score);
      else if (from_chain && position + 1 < chained)
        tree_push(heap,
                  score *
                      rates[(position + 1) * SPLASH_TREE_AGREEMENTS +
                            tree_agreement(lattice, chain[position + 1],
                                           position + 1)],
                  tree_info(true, row, position + 1, 0));
    }
    row_counts[lane] = rows.made;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The lane's rows; those past its tree are chain rows of its anchor.
  if (thread_index < lane_rows) {
    const ulong at = ulong(params.first_row[lane]) + thread_index;
    if (thread_index < rows.made) {
      tokens[at] = rows.token[thread_index];
      tree[at] = {rows.ancestors[thread_index], rows.parent[thread_index],
                  rows.depth[thread_index]};
      scores[at] = float2(rows.q[thread_index], rows.estimate[thread_index]);
    } else {
      tokens[at] = params.anchor[lane];
      tree[at] = {(2u << thread_index) - 1u, ushort(thread_index - 1),
                  ushort(thread_index)};
      scores[at] = float2(-1.0f, 0.0f);
    }
  }
}

// One thread chooses every lane's verify rows and lays them out
// (VerifyRowsParams). With costs, it ranks the estimates of the rows after
// the lanes' anchors, largest first, a lane's earlier row first among equal
// ones, so that a lane's chosen rows are always the first its tree took, and
// takes as many as maximize the expected tokens over the cost of the rows
// verified in all; without, every row of each tree. It then numbers a lane's
// chosen rows in preorder, the earlier-taken (likelier) child first, so that
// the likeliest path of a tree lies on its first rows, and the rows past them
// are chain rows of its anchor. row_counts holds each lane's tree rows, and
// then its verify rows.
kernel void draft_verify_rows(device uint *tokens [[buffer(0)]],
                              device SplashTreeRow *tree [[buffer(1)]],
                              device float2 *scores [[buffer(2)]],
                              device uint *row_counts [[buffer(3)]],
                              constant VerifyRowsParams &params [[buffer(4)]],
                              uint index [[thread_position_in_grid]]) {
  constexpr uint Lanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  if (index != 0)
    return;
  uint rows[Lanes];
  for (uint lane = 0; lane < params.lanes; ++lane)
    rows[lane] = params.every_row ? row_counts[lane] : 1;
  if (!params.every_row) {
    float estimates[Lanes * Rows];
    uint owners[Lanes * Rows];
    uint count = 0;
    for (uint lane = 0; lane < params.lanes; ++lane) {
      for (uint row = 1; row < row_counts[lane]; ++row) {
        const float estimate = scores[params.first_row[lane] + row].y;
        uint at = count++;
        for (; at > 0 && estimates[at - 1] < estimate; --at) {
          estimates[at] = estimates[at - 1];
          owners[at] = owners[at - 1];
        }
        estimates[at] = estimate;
        owners[at] = lane;
      }
    }
    float expected = float(params.lanes);
    float best = expected / params.cost[params.lanes];
    uint taken = 0;
    for (uint i = 0; i < count; ++i) {
      expected += estimates[i];
      const float value = expected / params.cost[params.lanes + i + 1];
      if (value > best) {
        best = value;
        taken = i + 1;
      }
    }
    for (uint i = 0; i < taken; ++i)
      ++rows[owners[i]];
  }
  for (uint lane = 0; lane < params.lanes; ++lane) {
    const uint first = params.first_row[lane], kept = rows[lane];
    uint token[Rows], parent[Rows], depth[Rows];
    float2 score[Rows];
    for (uint row = 0; row < kept; ++row) {
      token[row] = tokens[first + row];
      parent[row] = tree[first + row].parent;
      depth[row] = tree[first + row].depth;
      score[row] = scores[first + row];
    }
    // Preorder: a row's children follow it, each with its subtree, in the
    // order the tree took them.
    uint order[Rows], placed[Rows], stack[Rows];
    uint size = 0, next = 0;
    stack[size++] = 0;
    while (size) {
      const uint row = stack[--size];
      placed[row] = next;
      order[next++] = row;
      for (uint child = kept; child-- > row + 1;)
        if (parent[child] == row)
          stack[size++] = child;
    }
    uint ancestors[Rows];
    for (uint at = 0; at < kept; ++at) {
      const uint row = order[at];
      const uint up = at ? placed[parent[row]] : uint(SPLASH_TREE_ROOT);
      ancestors[at] = (at ? ancestors[up] : 0u) | (1u << at);
      tokens[first + at] = token[row];
      tree[first + at] = {ancestors[at], ushort(up), ushort(depth[row])};
      scores[first + at] = score[row];
    }
    for (uint at = kept; at < params.rows[lane]; ++at) {
      tokens[first + at] = token[0];
      tree[first + at] = {(2u << at) - 1u, ushort(at - 1), ushort(at)};
      scores[first + at] = float2(-1.0f, 0.0f);
    }
    row_counts[lane] = kept;
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
    constant TargetSamplingParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  const uint batch = group.y, index = group.x / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!selects(params, batch, index, false))
    return;
  const uint shard = group.x % SPLASH_TARGET_SAMPLING_SHARDS;
  const ulong partial =
      ulong(selected_logits_row(params, batch, index)) *
          SPLASH_TARGET_SAMPLING_SHARDS +
      shard;
  argmax_shard(selected_row(logits, token_mask, params, batch, index), shard,
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
    constant TargetSamplingParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint batch = group.y, index = group.x;
  if (!selects(params, batch, index, false))
    return;
  const uint s = selected_logits_row(params, batch, index);
  const uint token = argmax_reduce(partial_values, partial_indices, s, lane);
  if (lane == 0)
    tokens[s] = token;
}

// After the selection, the thinking cutoff of each lane that has one
// (ThinkingCutoffParams): its rows of depth row on select its end tokens, or
// where its end waits for its mask, a chain's rows from the first of those
// whose mask admits the end's first token. A row's path up to depth row, its
// anchor excepted, holds the draft tokens before its cut.
kernel void decode_sample_thinking_cutoff(
    device const uint *input_tokens [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const SplashTreeRow *tree [[buffer(2)]],
    device uint *tokens [[buffer(3)]],
    constant ThinkingCutoffParams &params [[buffer(4)]],
    uint lane [[threadgroup_position_in_grid]]) {
  const uint rows = params.rows[lane];
  uint row = params.row[lane];
  if (params.waiting_mask & (1u << lane)) {
    // The lane's mask rows, as selected_row reads them.
    device const uint *masks =
        token_mask + (ulong(lane) * (SPLASH_TARGET_VERIFY_ROWS + 1) +
                      params.mask_row) *
                         params.mask_words;
    const uint start = params.tokens[lane][0];
    for (; row < rows; ++row) {
      if (masks[row * params.mask_words + start / 32] & (1u << (start % 32)))
        break;
    }
  }
  const uint first = params.first_row[lane];
  for (uint at = 0; at < rows; ++at) {
    // A selection of one row, a first token after a prompt, has no tree: its
    // row is its anchor's.
    const uint depth = rows > 1 ? uint(tree[first + at].depth) : 0u;
    if (depth < row || depth - row >= params.count[lane])
      continue;
    bool ended = false;
    for (uint path = rows > 1 ? tree[first + at].ancestors & ~1u : 0u; path; path &= path - 1) {
      const uint before = ctz(path);
      const uint draft = input_tokens[first + before];
      ended = ended || (tree[first + before].depth <= row &&
                        (draft == params.close[lane] || draft == params.also_ends[lane]));
    }
    // A row with no selectable logit keeps its sentinel, so the lane fails.
    if (!ended && tokens[first + at] < params.vocabulary)
      tokens[first + at] = params.tokens[lane][depth - row];
  }
}

// Each lane's acceptance (AcceptBatchParams), one thread a lane. The walk
// follows every row of the lane's block, so its count of accepted proposals
// is the target's, those a verify budget left unverified included; the lane
// retains only rows it verified, the walk's first, as a child's row comes
// after its parent's. Its retained tokens, the selections of the path's rows,
// go to the output tokens from its first row, each read before an earlier
// path row's is written there.
kernel void decode_accept_tree(
    device const uint *input_tokens [[buffer(0)]],
    device const SplashTreeRow *tree [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    device uint *retained [[buffer(3)]],
    device uint *accepted_count [[buffer(4)]],
    device uint *retained_rows [[buffer(5)]],
    device const uint *verify_rows [[buffer(6)]],
    constant AcceptBatchParams &params [[buffer(7)]],
    uint batch [[threadgroup_position_in_grid]]) {
  const uint first = params.first_row[batch], rows = params.rows[batch];
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
  accepted_count[batch] = accepted;
  uint kept = 0;
  while (kept <= accepted && path[kept] < verify_rows[batch])
    ++kept;
  kept = min(kept, params.remaining[batch]);
  for (uint i = 0; i < kept; ++i) {
    const uint token = selected[path[i]];
    selected[i] = token;
    if (stop_token(params.stop_tokens, token))
      kept = i + 1;
  }
  retained[batch] = kept;
  retained_rows[batch] = nodes[path[kept - 1]].ancestors;
}
