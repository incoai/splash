#include "VerifyFfnPackBounds.hpp"
struct Pattern {
  uint32_t lanes;
  std::array<uint32_t, 8> rows;
};
static std::vector<Pattern> patterns() {
  std::vector<Pattern> p;
  auto append = [&](Pattern x) {
    for (uint32_t l = x.lanes; l < 8; ++l)
      x.rows[l] = 0xffffffffu;
    p.push_back(x);
  };
  for (uint32_t width = 5; width <= 8; ++width) {
    for (uint32_t r = 1; r <= 8; ++r) {
      Pattern x{width, {}};
      std::fill_n(x.rows.begin(), width, r);
      append(x);
    }
    for (uint32_t kind = 0; kind < 4; ++kind) {
      Pattern x{width, {}};
      for (uint32_t l = 0; l < width; ++l)
        x.rows[l] = kind == 0 ? 8 - l : kind == 1 ? l + 1 : kind == 2 ? (l % 2 ? 8 : 1) : (l % 2 ? 1 : 8);
      append(x);
    }
    for (uint32_t total : {31u, 32u, 33u}) {
      Pattern x{width, {}};
      std::fill_n(x.rows.begin(), width, 1u);
      uint32_t left = total - width;
      for (uint32_t l = 0; l < width; ++l) {
        uint32_t add = std::min(left, 7u);
        x.rows[l] += add;
        left -= add;
      }
      check(left == 0, "threshold pattern");
      append(x);
    }
  }
  check(p.size() == 60, "pattern count");
  return p;
}
static std::vector<uint16_t> selectedBits(const ChainBuffers &a, bool candidate) {
  std::vector<uint16_t> v;
  const auto op = static_cast<const uint16_t *>(a.out.view.contents());
  const bool cp = candidate && a.compact();
  const auto mp = static_cast<const uint16_t *>((cp ? a.packedMid : a.mid).view.contents());
  const auto *c = static_cast<const VerifyPackControl *>(a.control.view.contents());
  for (uint32_t l = 0; l < a.lanes; ++l)
    for (uint32_t r = 0; r < a.rows[l]; ++r) {
      uint32_t old = l * 8 + r, packed = cp ? c->inverse[old] : old;
      v.insert(v.end(), op + uint64_t{old} * a.H, op + uint64_t{old + 1} * a.H);
      v.insert(v.end(), mp + uint64_t{packed} * a.I, mp + uint64_t{packed + 1} * a.I);
    }
  return v;
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      check(argc == 2 || argc == 4, "usage: verify-ffn-pack LIB [BANK JSONL]");
      check(SPLASH_MAXIMUM_BATCH_WIDTH == 8, "verify-ffn-pack requires the eight-lane build");
      auto bank = verifyPackMetadata();
      std::ofstream report;
      if (argc == 4) {
        check(!std::filesystem::exists(argv[3]), "refuse overwrite");
        std::ifstream f(argv[2], std::ios::binary);
        uint32_t n = 0, n2 = 0;
        f.read(reinterpret_cast<char *>(&n), 4);
        f.read(reinterpret_cast<char *>(&n2), 4);
        check(n && n == n2, "bank header");
        bank.resize(n);
        f.read(reinterpret_cast<char *>(bank.data()), n * 4);
        check(bool(f), "bank read");
        report.open(argv[3]);
        check(bool(report), "report open");
      }
      std::ostringstream discarded;
      std::ostream &out =
          argc == 4 ? static_cast<std::ostream &>(report) : static_cast<std::ostream &>(discarded);

      MetalBackend b(argv[1]);
      std::mt19937 rng(2026110632u);
      out << std::setprecision(17);
      uint32_t cases = 0, bad = 0, negative = 0;
      auto pp = patterns();
      for (bool big : {false, true}) {
        uint32_t H = big ? 5120 : 2048, I = big ? 17408 : 768;
        auto w = makeFfn(b, H, I, rng, bank);
        auto fp64 = ffn_bounds::prepare(w);
        std::vector<std::array<uint32_t, 4>> configs;
        if (big) {
          ops::Linear linear(b.capabilities());
          ops::Projection up(
              I, H, ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, I, H, {}, {}, {})}}),
              down(H, I,
                   ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, H, I, {}, {}, {})}});
          auto u = linear.decodePlan(up, 4, ops::LinearEpilogue::GateUp, &up).configuration(),
               d = linear.decodePlan(down, 4, ops::LinearEpilogue::Residual).configuration();
          configs.push_back({u.splits, d.splits, uint32_t(u.spread), uint32_t(d.spread)});
          out << "{\"type\":\"production_config\",\"H\":" << H << ",\"I\":" << I << ",\"Su\":" << u.splits
              << ",\"Sd\":" << d.splits << ",\"walkU\":" << u.spread << ",\"walkD\":" << d.spread
              << ",\"fourlane_onepass\":" << u.oneGateUpPass << "}\n";
          out.flush();
        } else
          for (uint32_t su : {1u, 2u, 4u, 8u, 16u})
            for (uint32_t sd : {1u, 2u, 4u, 8u})
              for (uint32_t uw : {0u, 1u})
                for (uint32_t dw : {0u, 1u})
                  configs.push_back({su, sd, uw, dw});
        for (auto cfg : configs)
          for (uint32_t sparse : {0u, 1u})
            for (uint32_t pattern = 0; pattern < pp.size(); ++pattern) {
              auto pat = pp[pattern];
              std::vector<uint16_t> x(uint64_t{64} * H, 0x7fc0), res(uint64_t{64} * H);
              std::uniform_real_distribution<float> d(-1, 1);
              for (auto &v : res)
                v = floatToBf16(d(rng) * 64.f);
              for (uint32_t l = 0; l < pat.lanes; ++l)
                for (uint32_t r = 0; r < pat.rows[l]; ++r)
                  for (uint32_t k = 0; k < H; ++k) {
                    uint32_t row = l * 8 + r;
                    float value = sparse ? ((k % 256 == (row * 31 + k / 256 * 97) % 256)
                                                ? ((row + k / 256) % 2 ? -1e5f : 1e5f)
                                                : 0.f)
                                         : d(rng);
                    x[uint64_t{row} * H + k] = floatToBf16(value);
                  }
              std::vector<uint16_t> prevA, prevB;
              for (uint32_t repeat = 0; repeat < 3; ++repeat) {
                auto poison = std::array<uint32_t, 3>{0x7fc00000u, 0x7e800000u, 0xa5a5a5a5u}[repeat];
                ChainBuffers a(b, w, pat.lanes, pat.rows, x, res, cfg[0], cfg[1], cfg[2], cfg[3], poison),
                    c(b, w, pat.lanes, pat.rows, x, res, cfg[0], cfg[1], cfg[2], cfg[3], poison);
                CommandGraph ga, gc;
                a.add(b, ga, false);
                c.add(b, gc, true);
                static_cast<void>(b.submitCommandAsync(ga.dispatches()).wait());
                static_cast<void>(b.submitCommandAsync(gc.dispatches()).wait());
                bool partial = true;
                if (cfg[1] > 1) {
                  const bool compact = c.compact();
                  const auto *ctrl = static_cast<const VerifyPackControl *>(c.control.view.contents());
                  const auto *got = static_cast<const uint32_t *>(c.runtimePart.view.contents());
                  const auto *ref = static_cast<const uint32_t *>(a.wideDownPart.view.contents());
                  for (uint32_t split = 0; split < cfg[1]; ++split)
                    for (uint32_t lane = 0; lane < pat.lanes; ++lane)
                      for (uint32_t row = 0; row < pat.rows[lane]; ++row) {
                        uint32_t oldrow = lane * 8 + row, packed = compact ? ctrl->inverse[oldrow] : oldrow;
                        for (uint32_t n = 0; n < H; ++n)
                          partial =
                              partial && got[(uint64_t{split} * (compact ? 32 : 64) + packed) * H + n] ==
                                             ref[(uint64_t{split} * 64 + oldrow) * H + n];
                      }
                }
                if (cfg[0] == 1 && cfg[1] == 1) {
                  const auto *got = static_cast<const uint32_t *>(c.runtimePart.view.contents());
                  partial = partial && std::all_of(got, got + c.runtimePart.bytes / 4,
                                                   [&](uint32_t v) { return v == poison; });
                }
                bool eq = partial && selectedEqual(c, a, true), control = c.controlExact(),
                     guards = a.guards() && c.guards(), immut = a.immutable(false) && c.immutable(false),
                     inactive = a.scratchAndInactiveExact(false) && c.scratchAndInactiveExact(true),
                     bounds =
                         ffn_bounds::validate(a, a, fp64, false) && ffn_bounds::validate(c, a, fp64, true);
                auto abits = selectedBits(a, false), cbits = selectedBits(c, true);
                bool repeats = repeat == 0 || (abits == prevA && cbits == prevB);
                prevA = std::move(abits);
                prevB = std::move(cbits);
                bool pass = eq && control && guards && immut && inactive && bounds && repeats;
                if (!pass)
                  ++bad;
                out << "{\"type\":\"case\",\"case\":" << cases++ << ",\"H\":" << H << ",\"I\":" << I
                    << ",\"pattern\":" << pattern << ",\"lanes\":" << pat.lanes << ",\"sparse\":" << sparse
                    << ",\"Su\":" << cfg[0] << ",\"Sd\":" << cfg[1] << ",\"walkU\":" << cfg[2]
                    << ",\"walkD\":" << cfg[3] << ",\"poison\":" << poison << ",\"compact\":" << c.compact()
                    << ",\"selected_intermediate_and_output_exact\":" << eq
                    << ",\"control_map_grids_exact\":" << control << ",\"guards_counters\":" << guards
                    << ",\"input_immutable\":" << immut << ",\"inactive_path_and_scratch_exact\":" << inactive
                    << ",\"FP64_samples_pass\":" << bounds << ",\"repeat_exact\":" << repeats
                    << ",\"pass\":" << pass << "}\n";
                out.flush();
                if (!pass && bad <= 12)
                  std::cout << "FAIL " << cases - 1 << " H=" << H << " pattern=" << pattern << " eq=" << eq
                            << " ctrl=" << control << " guard=" << guards << " immut=" << immut
                            << " inactive=" << inactive << " FP64=" << bounds << " repeat=" << repeats
                            << std::endl;
              }
            }
        bool weights = true;
        for (const Weight *t : {&w.gate, &w.up, &w.down})
          weights = weights && hash(t->codes) == t->hc && hash(t->metadata) == t->hm;
        out << "{\"type\":\"weight_immutability\",\"H\":" << H << ",\"pass\":" << weights << "}\n";
        out.flush();
        if (!weights)
          ++bad;
        if (!big) {
          std::vector<uint16_t> x(uint64_t{64} * H, 0), res(x.size(), 0);
          for (uint32_t kind = 0; kind < 5; ++kind) {
            std::array<uint32_t, 8> r;
            r.fill(4);
            uint32_t lanes = kind == 0 ? 0 : kind == 1 ? 4 : kind == 2 ? 9 : 8;
            if (kind == 3)
              r[3] = 0;
            if (kind == 4)
              r[5] = 9;
            ChainBuffers c(b, w, lanes, r, x, res, 1, 1, 0, 0, 0x7fc00000);
            CommandGraph g;
            g.add("verify_ffn_pack_map", {c.counts.view, c.control.view}, c.params(), {1, 1, 1}, {1, 1, 1});
            static_cast<void>(b.submitCommandAsync(g.dispatches()).wait());
            bool pass = c.controlExact() && c.guards() && c.immutable(false) && !c.compact();
            if (!pass)
              ++bad;
            ++negative;
            out << "{\"type\":\"negative_map\",\"kind\":" << kind << ",\"pass\":" << pass << "}\n";
            out.flush();
          }
        }
      }
      out << "{\"type\":\"summary\",\"cases\":" << cases << ",\"negative_cases\":" << negative
          << ",\"failed\":" << bad << "}\n";
      if (report.is_open())
        report.close();
      check(cases == 29160 && negative == 5, "case coverage");
      std::cout << "VERIFY_FFN_PACK cases=" << cases << " negative=" << negative << " failed=" << bad
                << std::endl;
      return bad ? 1 : 0;
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return 2;
    }
  }
}
