#pragma once
#include "metal/abi/VerifyFfnPack.h"
#include "VerifyPackTestFixtures.hpp"
#include "ops/VerifyFfnPack.hpp"
#include <cstddef>
#include <memory>
#include <optional>
struct FfnWeights {
  Weight gate, up, down;
};
static FfnWeights makeFfn(MetalBackend &b, uint32_t H, uint32_t I, std::mt19937 &rng,
                          const std::vector<uint32_t> &bank) {
  return {weight(b, I, H, rng, bank), weight(b, I, H, rng, bank), weight(b, H, I, rng, bank)};
}
struct ChainBuffers {
  uint32_t H, I, Su, Sd, walkU, walkD, lanes, poison;
  const FfnWeights &w;
  std::vector<uint16_t> x, residual;
  std::array<uint32_t, 8> rows;
  Guarded input, res, counts, control, packedInput, packedRes, packedMid, packedOut;
  Guarded gate, mid, out, wideUpPart, wideUpCounts, packedUpPart, packedUpCounts, wideDownPart,
      wideDownCounts, packedDownPart, packedDownCounts;
  Guarded runtimePart, runtimeCounters;
  ChainBuffers(MetalBackend &b, const FfnWeights &weights, uint32_t width, const std::array<uint32_t, 8> &r,
               const std::vector<uint16_t> &normal, const std::vector<uint16_t> &residualBits, uint32_t us,
               uint32_t ds, uint32_t uw, uint32_t dw, uint32_t scratchPoison)
      : H(weights.up.K), I(weights.up.N), Su(us), Sd(ds), walkU(uw), walkD(dw), lanes(width),
        poison(scratchPoison), w(weights), x(normal), residual(residualBits), rows(r),
        input(b, uint64_t{64} * H * 2, 0), res(b, uint64_t{64} * H * 2, 0), counts(b, 32, 0),
        control(b, sizeof(VerifyPackControl), 0xa5a5a5a5), packedInput(b, uint64_t{32} * H * 2, 0xffffffff),
        packedRes(b, uint64_t{32} * H * 2, 0xffffffff), packedMid(b, uint64_t{32} * I * 2, 0xffffffff),
        packedOut(b, uint64_t{32} * H * 2, 0xffffffff), gate(b, uint64_t{64} * I * 2, 0xffffffff),
        mid(b, uint64_t{64} * I * 2, 0xffffffff), out(b, uint64_t{64} * H * 2, 0xffffffff),
        wideUpPart(b, uint64_t{Su} * 64 * I * 4, poison), wideUpCounts(b, uint64_t{I} / 64 * 4, 0),
        packedUpPart(b, uint64_t{Su} * 32 * 2 * I * 4, poison), packedUpCounts(b, uint64_t{I} / 32 * 4, 0),
        wideDownPart(b, uint64_t{Sd} * 64 * H * 4, poison), wideDownCounts(b, uint64_t{H} / 64 * 4, 0),
        packedDownPart(b, uint64_t{Sd} * 32 * H * 4, poison), packedDownCounts(b, uint64_t{H} / 64 * 4, 0),
        runtimePart(b, std::max(uint64_t{Su} * 64 * I, uint64_t{Sd} * 64 * H) * 4, poison),
        runtimeCounters(b, std::max(uint64_t{I} / 32, uint64_t{H} / 64) * 4, 0) {
    check(normal.size() == uint64_t{64} * H && residualBits.size() == normal.size(), "chain input extent");
    std::memcpy(input.view.contents(), x.data(), input.bytes);
    std::memcpy(res.view.contents(), residual.data(), res.bytes);
    std::memcpy(counts.view.contents(), rows.data(), 32);
  }
  VerifyPackParams params() const { return {lanes, H, I, Su, Sd}; }
  void add(MetalBackend &, CommandGraph &g, bool candidate) {
    if (candidate) {
      auto projection = [](const Weight &w) {
        return ops::Projection(w.N, w.K,
                               ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, w.N, w.K,
                                                                                w.codes, {}, w.metadata)}});
      };
      const auto gw = projection(w.gate), uw = projection(w.up), dw = projection(w.down);
      const auto up =
          ops::Linear::plan({{I, H}, 64, ops::LinearPhase::Decode, ops::LinearEpilogue::GateUp},
                            {.tile = ops::LinearTile::GgufStaged, .splits = Su, .spread = bool(walkU)},
                            ops::FloatOutput::BFloat16);
      const auto down =
          ops::Linear::plan({{H, I}, 64, ops::LinearPhase::Decode, ops::LinearEpilogue::Residual},
                            {.tile = ops::LinearTile::GgufStaged, .splits = Sd, .spread = bool(walkD)},
                            ops::FloatOutput::BFloat16);
      const ops::LinearScratch shared{.partials = runtimePart.view, .counters = runtimeCounters.view};
      check(ops::addVerifyFfnPack(
                g, {&gw, &uw, &dw}, input.view, res.view, out.view, gate.view, mid.view, shared, counts.view,
                {control.view, packedInput.view, packedRes.view, packedMid.view, packedOut.view}, up, down,
                lanes),
            "native FFN helper unexpectedly fell back");
      return;
    }

    const auto pu = GgufDecodeParams{H, Su, I, 0, walkU}, pd = GgufDecodeParams{I, Sd, H, 0, walkD};
    const auto p = params();
    auto wide = [&](uint32_t which, const std::string &name, std::vector<MetalBuffer> buffers,
                    const GgufDecodeParams &param, metal::DispatchSize groups) {
      if (candidate)
        g.add(name, std::move(buffers), param, groups,
              metal::IndirectGrid{control.view, offsetof(VerifyPackControl, grids) + which * 12}, {64, 1, 1});
      else
        g.add(name, std::move(buffers), param, groups, {64, 1, 1});
    };
    if (candidate) {
      g.add("verify_ffn_pack_map", {counts.view, control.view}, p, {1, 1, 1}, {1, 1, 1});
      g.add("verify_ffn_pack_gather", {input.view, res.view, control.view, packedInput.view, packedRes.view},
            p, {(H + 255) / 256, 32, 1},
            metal::IndirectGrid{control.view, offsetof(VerifyPackControl, grids)}, {256, 1, 1});
      g.add("gguf_decode_af4g64_m32_gate_up",
            {packedInput.view, w.gate.codes, w.gate.codes, w.gate.metadata, w.up.codes, w.up.codes,
             w.up.metadata, packedMid.view, packedUpPart.view, packedUpCounts.view},
            pu, {I / 32, Su, 1}, metal::IndirectGrid{control.view, offsetof(VerifyPackControl, grids) + 12},
            {32, 1, 1});
      g.add("gguf_decode_af4g64_m32_r",
            {packedMid.view, w.down.codes, w.down.codes, w.down.metadata, packedOut.view, packedDownPart.view,
             packedDownCounts.view, packedRes.view},
            pd, {H / 64, Sd, 1}, metal::IndirectGrid{control.view, offsetof(VerifyPackControl, grids) + 24},
            {64, 1, 1});
      g.add("verify_ffn_pack_scatter", {packedOut.view, res.view, control.view, out.view}, p,
            {(H + 255) / 256, 64, 1},
            metal::IndirectGrid{control.view, offsetof(VerifyPackControl, grids) + 36}, {256, 1, 1});
    }
    wide(4, "gguf_decode_af4g64_m64_a",
         {input.view, w.gate.codes, w.gate.codes, w.gate.metadata, gate.view, wideUpPart.view,
          wideUpCounts.view, input.view},
         pu, {I / 64, Su, 1});
    wide(5, "gguf_decode_af4g64_m64_g",
         {input.view, w.up.codes, w.up.codes, w.up.metadata, mid.view, wideUpPart.view, wideUpCounts.view,
          gate.view},
         pu, {I / 64, Su, 1});
    wide(6, "gguf_decode_af4g64_m64_r",
         {mid.view, w.down.codes, w.down.codes, w.down.metadata, out.view, wideDownPart.view,
          wideDownCounts.view, res.view},
         pd, {H / 64, Sd, 1});
  }
  bool compact() const {
    return static_cast<const VerifyPackControl *>(control.view.contents())->compact != 0;
  }
  bool immutable(bool includeWeights = true) const {
    if (std::memcmp(input.view.contents(), x.data(), input.bytes) ||
        std::memcmp(res.view.contents(), residual.data(), res.bytes) ||
        std::memcmp(counts.view.contents(), rows.data(), 32))
      return false;
    if (includeWeights)
      for (const Weight *t : {&w.gate, &w.up, &w.down})
        if (hash(t->codes) != t->hc || hash(t->metadata) != t->hm)
          return false;
    return true;
  }
  bool guards() const {
    if (!runtimePart.intact() || !runtimeCounters.intact())
      return false;
    auto rc = static_cast<const uint32_t *>(runtimeCounters.view.contents());
    if (!std::all_of(rc, rc + runtimeCounters.bytes / 4, [](uint32_t v) { return v == 0; }))
      return false;
    for (const Guarded *b :
         {&input, &res, &counts, &control, &packedInput, &packedRes, &packedMid, &packedOut, &gate, &mid,
          &out, &wideUpPart, &wideUpCounts, &packedUpPart, &packedUpCounts, &wideDownPart, &wideDownCounts,
          &packedDownPart, &packedDownCounts})
      if (!b->intact())
        return false;
    for (const Guarded *b : {&wideUpCounts, &packedUpCounts, &wideDownCounts, &packedDownCounts}) {
      auto p = static_cast<const uint32_t *>(b->view.contents());
      if (!std::all_of(p, p + b->bytes / 4, [](uint32_t v) { return v == 0; }))
        return false;
    }
    return true;
  }
  bool controlExact() const {
    const auto &c = *static_cast<const VerifyPackControl *>(control.view.contents());
    uint32_t total = 0;
    bool valid = lanes >= 5 && lanes <= 8;
    for (uint32_t l = 0; l < std::min(lanes, 8u); ++l) {
      total += std::min(rows[l], 8u);
      valid = valid && rows[l] >= 1 && rows[l] <= 8;
    }
    const bool cp = valid && total <= 32;
    if (c.total != total || bool(c.valid) != valid || bool(c.compact) != cp)
      return false;
    std::array<uint32_t, 32> map;
    map.fill(0xffffffff);
    std::array<uint32_t, 64> inverse;
    inverse.fill(0xffffffff);
    uint32_t at = 0;
    if (cp)
      for (uint32_t l = 0; l < lanes; ++l)
        for (uint32_t r = 0; r < rows[l]; ++r) {
          map[at] = l * 8 + r;
          inverse[l * 8 + r] = at++;
        }
    if (!std::equal(map.begin(), map.end(), c.map) || !std::equal(inverse.begin(), inverse.end(), c.inverse))
      return false;
    std::array<uint32_t, 7> xs{(H + 255) / 256, I / 32, H / 64, (H + 255) / 256, I / 64, I / 64, H / 64},
        ys{32, Su, Sd, 64, Su, Su, Sd};
    for (uint32_t i = 0; i < 7; ++i) {
      bool on = i < 4 ? cp : !cp;
      for (uint32_t j = 0; j < 3; ++j)
        if (c.grids[i][j] != (on ? (j == 0 ? xs[i] : j == 1 ? ys[i] : 1u) : 0u))
          return false;
    }
    return true;
  }
  bool scratchAndInactiveExact(bool candidate) const {
    bool cp = candidate && compact();
    auto untouched = [&](const Guarded &b, uint32_t value) {
      auto p = static_cast<const uint32_t *>(b.view.contents());
      return std::all_of(p, p + b.bytes / 4, [&](uint32_t v) { return v == value; });
    };
    if (cp) {
      if (!untouched(gate, 0xffffffff) || !untouched(mid, 0xffffffff) || !untouched(wideUpPart, poison) ||
          !untouched(wideDownPart, poison))
        return false;
    } else {
      if (!untouched(packedInput, 0xffffffff) || !untouched(packedRes, 0xffffffff) ||
          !untouched(packedMid, 0xffffffff) || !untouched(packedOut, 0xffffffff) ||
          !untouched(packedUpPart, poison) || !untouched(packedDownPart, poison))
        return false;
    }
    if (Su == 1 && (!untouched(wideUpPart, poison) || !untouched(packedUpPart, poison)))
      return false;
    if (Sd == 1 && (!untouched(wideDownPart, poison) || !untouched(packedDownPart, poison)))
      return false;
    if (cp) {
      const auto &c = *static_cast<const VerifyPackControl *>(control.view.contents());
      auto pi = static_cast<const uint16_t *>(packedInput.view.contents()),
           pr = static_cast<const uint16_t *>(packedRes.view.contents()),
           op = static_cast<const uint16_t *>(out.view.contents());
      for (uint32_t r = 0; r < 32; ++r)
        for (uint32_t k = 0; k < H; ++k) {
          auto old = c.map[r];
          if (pi[uint64_t{r} * H + k] != (old < 64 ? x[uint64_t{old} * H + k] : 0u) ||
              pr[uint64_t{r} * H + k] != (old < 64 ? residual[uint64_t{old} * H + k] : 0u))
            return false;
        }
      for (uint32_t r = 0; r < 64; ++r)
        if (c.inverse[r] == 0xffffffff)
          for (uint32_t k = 0; k < H; ++k)
            if (op[uint64_t{r} * H + k] != residual[uint64_t{r} * H + k])
              return false;
    }
    return true;
  }
};
static bool selectedEqual(const ChainBuffers &a, const ChainBuffers &base, bool candidate) {
  const auto ao = static_cast<const uint16_t *>(a.out.view.contents()),
             bo = static_cast<const uint16_t *>(base.out.view.contents());
  bool cp = candidate && a.compact();
  auto am = static_cast<const uint16_t *>((cp ? a.packedMid : a.mid).view.contents()),
       bm = static_cast<const uint16_t *>(base.mid.view.contents());
  const auto *c = static_cast<const VerifyPackControl *>(a.control.view.contents());
  for (uint32_t l = 0; l < a.lanes; ++l)
    for (uint32_t r = 0; r < a.rows[l]; ++r) {
      uint32_t old = l * 8 + r, packed = cp ? c->inverse[old] : old;
      for (uint32_t k = 0; k < a.H; ++k) {
        auto at = uint64_t{old} * a.H + k;
        if (ao[at] != bo[at] || !std::isfinite(bf16ToFloat(ao[at])))
          return false;
      }
      for (uint32_t k = 0; k < a.I; ++k)
        if (am[uint64_t{packed} * a.I + k] != bm[uint64_t{old} * a.I + k] ||
            !std::isfinite(bf16ToFloat(am[uint64_t{packed} * a.I + k])))
          return false;
    }
  return true;
}
