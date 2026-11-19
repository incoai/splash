// Included for each compiled selector geometry; the larger lattice lives in device scratch.
namespace TREE_NAMESPACE {
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
// The token of a row's host chain child where it has none.
constant constexpr uint kNoChainToken = 0xffffffffu;

// A lane's draft candidates at each proposal position, the probability of
// each after each predecessor candidate (the anchor for position 0) and the
// predecessor's kTreeChildren most likely.
struct TreeLattice {
  uint positions;
  uint ids[TREE_POSITIONS][TREE_CANDIDATES];
  float probability[TREE_POSITIONS][TREE_CANDIDATES]
                   [TREE_CANDIDATES];
  uchar likeliest[TREE_POSITIONS][TREE_CANDIDATES]
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
// depths, whether each lies on its host chain's path (its tokens from the
// anchor are the chain's first ones), draft probabilities, estimates, caps
// and keys, and of each row the children its draft drew (their candidate
// indices, in order; kNoCandidate past them) with each one's cap, and its
// next position's probabilities (DraftTreeParams).
struct TreeRows {
  uint token[kTreeRows];
  uint ancestors[kTreeRows];
  ushort parent[kTreeRows];
  ushort depth[kTreeRows];
  bool on_chain[kTreeRows];
  float q[kTreeRows];
  float estimate[kTreeRows];
  float cap[kTreeRows];
  ulong key[kTreeRows];
  uchar draw[kTreeRows][kTreeChildren];
  float left[kTreeRows][kTreeChildren];
  float next[kTreeRows][TREE_CANDIDATES];
  uint made;
};

// A new row's record of its children: none yet.
inline void tree_row_clear(threadgroup TreeRows &rows, uint row) {
  for (uint child = 0; child < kTreeChildren; ++child)
    rows.draw[row][child] = kNoCandidate;
  for (uint i = 0; i < TREE_CANDIDATES; ++i)
    rows.next[row][i] = 0.0f;
}

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

// The estimate of lane l's draft step of probability q at proposal position
// p (DraftTreeParams), q kept off 0 and 1 for its logit.
inline float tree_draft_estimate(constant DraftTreeParams &params, uint lane,
                                 float q, uint position) {
  q = clamp(q, 1.0e-6f, 1.0f - 1.0e-6f);
  const float z = params.slope[lane] * log(q / (1.0f - q)) +
                  params.intercept[lane][position];
  return 1.0f / (1.0f + exp(-z));
}

// A host chain token's agreement with the draft at its depth (0-based):
// the draft's most likely candidate there, another of its candidates, none,
// or past its positions.
inline uint tree_agreement(TREE_STORAGE TreeLattice &lattice, uint token,
                           uint depth) {
  if (depth >= lattice.positions)
    return 3;
  if (token == lattice.ids[depth][0])
    return 0;
  for (uint i = 1; i < TREE_CANDIDATES; ++i)
    if (token == lattice.ids[depth][i])
      return 1;
  return 2;
}

// The rate of a lane's host chain token at depth `depth` (0-based), by its
// agreement with the draft there.
inline float tree_chain_rate(TREE_STORAGE TreeLattice &lattice,
                             device const uint *chain,
                             device const float *rates, uint depth) {
  return rates[depth * SPLASH_TREE_AGREEMENTS +
               tree_agreement(lattice, chain[depth], depth)];
}

// The splitmix64 finalizer.
inline ulong tree_hash(ulong value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
  return value ^ (value >> 31);
}

// A row's key from its parent's and its token (DraftTreeParams).
inline ulong tree_key(ulong parent, uint token) {
  return tree_hash(parent + 0x9e3779b97f4a7c15ul * (ulong(token) + 1));
}

// The standard Gumbel of candidate token `token` under a row of key `key`:
// of a uniform from the top 23 bits k of a hash of the two, (k + 0.5) / 2^23,
// which fp32 holds exactly, inside (0, 1): with 24 bits the largest rounds to
// 1, a Gumbel of infinity.
inline float tree_gumbel(ulong key, uint token) {
  const ulong value = tree_hash(key ^ 0xd1b54a32d192ed03ul * (ulong(token) + 1));
  const float uniform = (float(value >> 41) + 0.5f) * 0x1p-23f;
  return -log(-log(uniform));
}

// A draft row's entries for its children at the next position, and its
// record of them (TreeRows): a greedy lane's most likely candidates, q at
// temperature 1; a sampled lane's draws without replacement from q_T,
// proportional to q^(1/T), its Gumbel-top-k with the row's key. q_T weighs
// each candidate against the most likely, exp((log q - log q_max) / T), so
// that its mass is at least 1 at any temperature, and the draws take its
// stored values, which the target's tests read: a candidate whose weight
// underflows is never drawn. The target selects one token after the row, so
// its children's estimates sum to at most its own: each child's own estimate
// is at most its cap, what the row's host chain child (`chain_token` at
// `chain_rate`; none at a rate of 0) and the children drawn before it left of
// 1 (DraftTreeParams), and the child is estimated at most as the one before
// it, so that the tree takes a row's children in the order drawn and a cycle
// verifies a first few of them. A drawn child of the chain's token is the
// chain's child: its cap is what the others left, and it takes the more of
// its own estimate and the chain's rate. The rate is the host's, from before
// any draw, so no row is admitted by its own draws. Before the chain's child
// counted, on Qwen3.6-35B-A3B's 31K-token code prompts at one and two lanes,
// a chain's first rows of rate 0.9 or more were estimated 0.983 and 0.984 and
// retained 0.954 and 0.985 of the time (108 and 328 rows), and the draft rows
// beside them were estimated 0.202 and 0.193 and retained 0.088 and 0.018 (34
// and 56 rows): the anchor's children summed to 1.14 to 1.16.
inline void tree_expand(TREE_STORAGE TreeLattice &lattice,
                        threadgroup TreeHeap &heap, threadgroup TreeRows &rows,
                        constant DraftTreeParams &params, uint lane,
                        uint children, uint row, uint position,
                        uint predecessor, float score, uint chain_token,
                        float chain_rate) {
  TREE_STORAGE const float *q = lattice.probability[position][predecessor];
  const float temperature = params.temperature[lane];
  float top = 0.0f;
  for (uint i = 0; i < TREE_CANDIDATES; ++i)
    top = max(top, q[i]);
  float mass = 0.0f;
  for (uint i = 0; i < TREE_CANDIDATES; ++i) {
    const float weight = temperature == 0.0f || !(q[i] > 0.0f)
                             ? q[i]
                             : exp((log(q[i]) - log(top)) / temperature);
    rows.next[row][i] = weight;
    mass += weight;
  }
  for (uint i = 0; i < TREE_CANDIDATES; ++i)
    rows.next[row][i] /= mass;
  float keys[TREE_CANDIDATES];
  for (uint i = 0; temperature > 0.0f && i < TREE_CANDIDATES; ++i)
    keys[i] = rows.next[row][i] > 0.0f
                  ? log(rows.next[row][i]) + tree_gumbel(rows.key[row], lattice.ids[position][i])
                  : -INFINITY;
  uint taken = 0;
  float bound = score, left = 1.0f - chain_rate, reserved = chain_rate;
  for (uint child = 0; child < children; ++child) {
    uint drawn = TREE_CANDIDATES;
    if (temperature == 0.0f) {
      const uint likeliest = lattice.likeliest[position][predecessor][child];
      drawn = q[likeliest] > 0.0f ? likeliest : drawn;
    } else {
      for (uint i = 0; i < TREE_CANDIDATES; ++i)
        if (!(taken & (1u << i)) && keys[i] > -INFINITY &&
            (drawn == TREE_CANDIDATES || keys[i] > keys[drawn]))
          drawn = i;
    }
    if (drawn == TREE_CANDIDATES)
      return;
    taken |= 1u << drawn;
    const bool joins_chain = lattice.ids[position][drawn] == chain_token;
    const float cap = joins_chain ? left + reserved : left;
    rows.draw[row][child] = uchar(drawn);
    rows.left[row][child] = cap;
    const float own = min(tree_draft_estimate(params, lane, rows.next[row][drawn], position), cap);
    left = joins_chain ? cap - max(own, reserved) : left - own;
    reserved = joins_chain ? 0.0f : reserved;
    bound = min(bound, score * own);
    tree_push(heap, bound, tree_info(false, row, position, drawn));
  }
}

// The cap of the step row `parent` drew to its candidate `index`
// (tree_expand).
inline float tree_cap(threadgroup TreeRows &rows, uint parent, uint index) {
  for (uint child = 0; child < kTreeChildren; ++child)
    if (rows.draw[parent][child] == index)
      return rows.left[parent][child];
  return 1.0f;
}

// One group per lane builds its tree (DraftTreeParams). Its threads first
// take, for every proposal position and predecessor candidate (the anchor
// for position 0), the probabilities of the position's candidates (the
// softmax at temperature 1 of their unary scores plus the edges from the
// predecessor, the table draft_select_edges left in the partial-values
// scratch) and the predecessor's most likely, the lowest index first among
// equal scores, as the chain walk chose; thread 0 then grows the tree. Rows
// are written in the order the tree takes them, lane l's from row l *
// SPLASH_TARGET_VERIFY_ROWS, so a row's parent comes before it and its
// estimate is at most its parent's; sizes holds each lane's rows. A row's
// draft probability (scores' x) is that of the draft step that made it, or
// joined it where a lookup chain's token made it, its token's among the
// candidates after its parent's; -1 for a chain row no draft step joined;
// its estimate is scores' y, and the cap of that draft step its caps entry
// (1 for a row it has no draft probability of). Of each row it also writes the candidate indices of the
// children it drew, in order (draws, kNoCandidate past them), and its next
// position's probabilities (next), which a sampled verify row's selection
// tests (decode_sample_vocabulary_draw).
}
kernel void TREE_KERNEL(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device const uint *chain_tokens [[buffer(3)]],
    device const float *chain_rates [[buffer(4)]],
    device uint *tokens [[buffer(5)]],
    device SplashTreeRow *tree [[buffer(6)]],
    device float2 *scores [[buffer(7)]],
    device uint *sizes [[buffer(8)]],
    device uchar *draws [[buffer(9)]],
    device float *next [[buffer(10)]],
    device float *caps [[buffer(11)]],
    constant DraftTreeParams &params [[buffer(12)]],
    uint lane [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  using namespace TREE_NAMESPACE;
  const uint Positions = params.positions;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = TREE_CANDIDATES;
#if TREE_CANDIDATES == 32
  const ulong scratch = ulong(params.lanes) * Positions * Candidates * (Shards + Candidates);
  device TreeLattice &lattice = reinterpret_cast<device TreeLattice *>(partial_values + scratch)[lane];
#else
  threadgroup TreeLattice lattice;
#endif
  if (thread_index == 0) lattice.positions = Positions;
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
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
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
    rows.on_chain[0] = true;
    rows.q[0] = 1.0f;
    rows.estimate[0] = 1.0f;
    rows.cap[0] = 1.0f;
    rows.key[0] = ulong(params.noise[lane][1]) << 32 | params.noise[lane][0];
    tree_row_clear(rows, 0);
    if (deepest > 0) {
      const float rate =
          chained ? tree_chain_rate(lattice, chain, rates, 0) : 0.0f;
      tree_expand(lattice, heap, rows, params, lane, children, 0, 0, 0, 1.0f,
                  chained ? chain[0] : kNoChainToken, rate);
      if (chained)
        tree_push(heap, rate, tree_info(true, 0, 0, 0));
    }
    // No more rows than the tokens a walk of the tree may retain.
    const uint most = min(Rows, deepest + 1);
    while (heap.size && rows.made < most) {
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
        rows.token[row] = token;
        rows.ancestors[row] = rows.ancestors[parent] | (1u << row);
        rows.parent[row] = ushort(parent);
        rows.depth[row] = ushort(position + 1);
        rows.on_chain[row] = rows.on_chain[parent] && position < chained &&
                             token == chain[position];
        // A row a host chain token made is no draft step: its estimate is
        // the chain's, and it has no draft probability.
        rows.q[row] = from_chain ? -1.0f : rows.next[parent][index];
        rows.estimate[row] = score;
        rows.cap[row] = from_chain ? 1.0f : tree_cap(rows, parent, index);
        rows.key[row] = tree_key(rows.key[parent], token);
        tree_row_clear(rows, row);
      } else if (!from_chain && rows.q[row] < 0.0f) {
        // A draft step that joins a lookup chain's row labels the draft's
        // calibration like any draft row: such rows, which the draft's top
        // candidate often is, the target accepted 97-100% of the time in
        // decode traces (model::LookupAcceptance::kPrior), and the fit
        // without them learns from the draft's other rows alone.
        rows.q[row] = rows.next[parent][index];
        rows.cap[row] = tree_cap(rows, parent, index);
      }
      // The row's children lie a step deeper.
      if (position + 1 >= deepest)
        continue;
      // A row on the chain's path has the chain's next token for a child.
      const bool chain_child = rows.on_chain[row] && position + 1 < chained;
      if (!from_chain && position + 1 < Positions)
        tree_expand(lattice, heap, rows, params, lane, children, row,
                    position + 1, index, score,
                    chain_child ? chain[position + 1] : kNoChainToken,
                    chain_child ? tree_chain_rate(lattice, chain, rates,
                                                  position + 1)
                                : 0.0f);
      else if (from_chain && position + 1 < chained)
        tree_push(heap,
                  score * tree_chain_rate(lattice, chain, rates, position + 1),
                  tree_info(true, row, position + 1, 0));
    }
    sizes[lane] = rows.made;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (thread_index < rows.made) {
    const ulong at = ulong(lane) * Rows + thread_index;
    tokens[at] = rows.token[thread_index];
    tree[at] = {rows.ancestors[thread_index], rows.parent[thread_index],
                rows.depth[thread_index]};
    scores[at] = float2(rows.q[thread_index], rows.estimate[thread_index]);
    caps[at] = rows.cap[thread_index];
    for (uint child = 0; child < kTreeChildren; ++child)
      draws[at * kTreeChildren + child] = rows.draw[thread_index][child];
    for (uint i = 0; i < Candidates; ++i)
      next[at * Candidates + i] = rows.next[thread_index][i];
  }
}
