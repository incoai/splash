#include "VerifyBodyPackCases.hpp"
#include "VerifyPackTestFixtures.hpp"
#include "ops/VerifyBodyPack.hpp"
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      check(argc == 2 || argc == 4, "usage: verify-body-pack LIB [BANK JSONL]");
      check(SPLASH_MAXIMUM_BATCH_WIDTH == 8, "verify-body-pack requires the eight-lane build");
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
      std::mt19937 rng(2026110682u);
      out << std::setprecision(17);
      uint32_t cases = 0, bad = 0;
      auto fixture = [&](uint32_t K, const std::vector<uint32_t> &ns, uint32_t tail, bool ep, bool big,
                         uint32_t padding = 0) {
        Fixture f(b, K, ns, tail, rng, bank, tail == 1 ? 160 : tail == 2 ? 32 : padding);
        std::vector<std::array<uint32_t, 2>> configs;
        if (big) {
          ops::Linear linear(b.capabilities());
          auto c =
              linear
                  .decodePlan(f.projection, 8, ep ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None)
                  .configuration();
          configs.push_back({c.splits, uint32_t(c.spread)});
        } else
          for (uint32_t s : {1u, 2u, 4u, 8u, 16u})
            if (K % (32 * s) == 0)
              for (uint32_t walk : {0u, 1u})
                configs.push_back({s, walk});
        for (auto cfg : configs)
          for (uint32_t L : big ? std::vector<uint32_t>{5, 7, 8} : std::vector<uint32_t>{5, 6, 7, 8})
            for (uint32_t pattern = 0; pattern < (big ? 4u : 6u); ++pattern)
              for (uint32_t sparse : {0u, 1u}) {
                std::array<uint32_t, 8> rows;
                rows.fill(0xffffffff);
                for (uint32_t l = 0; l < L; ++l)
                  rows[l] = pattern < 4 ? pattern + 1 : pattern == 4 ? (l % 2 ? 4 : 1) : (l % 2 ? 1 : 4);
                std::vector<uint16_t> x(uint64_t{64} * K, 0x7fc0), aux(ep ? uint64_t{64} * f.N : 2);
                std::uniform_real_distribution<float> d(-1, 1);
                for (auto &v : aux)
                  v = floatToBf16(d(rng) * 64);
                for (uint32_t l = 0; l < L; ++l)
                  for (uint32_t r = 0; r < rows[l]; ++r)
                    for (uint32_t k = 0; k < K; ++k) {
                      uint32_t old = l * 8 + r;
                      float z = sparse ? (k % 256 == (old * 31 + k / 256 * 97) % 256
                                              ? ((old + k / 256) % 2 ? -1e5f : 1e5f)
                                              : 0.f)
                                       : d(rng);
                      x[uint64_t{old} * K + k] = floatToBf16(z);
                    }
                std::vector<uint16_t> pa, pc;
                std::vector<uint32_t> pap, pcp;
                for (uint32_t repeat = 0; repeat < 3; ++repeat) {
                  uint32_t poison = std::array<uint32_t, 3>{0x7fc00000u, 0x7e800000u, 0xa5a5a5a5u}[repeat];
                  Run a(b, f, L, rows, x, aux, cfg[0], cfg[1], ep, false, poison),
                      c(b, f, L, rows, x, aux, cfg[0], cfg[1], ep, true, poison);
                  CommandGraph ga, gc;
                  a.add(ga, b);
                  c.add(gc, b);
                  static_cast<void>(b.submitCommandAsync(ga.dispatches()).wait());
                  static_cast<void>(b.submitCommandAsync(gc.dispatches()).wait());
                  auto av = a.selected(), cv = c.selected();
                  auto ap = a.selectedParts(), cp = c.selectedParts();
                  bool eq = av == cv && c.partialsExact(a), guard = a.guard(poison) && c.guard(poison),
                       fp = a.bounds() && c.bounds(),
                       rep = repeat == 0 || (av == pa && cv == pc && ap == pap && cp == pcp);
                  pa = std::move(av);
                  pc = std::move(cv);
                  pap = std::move(ap);
                  pcp = std::move(cp);
                  bool pass = eq && guard && fp && rep;
                  if (!pass)
                    ++bad;
                  out << "{\"type\":\"case\",\"case\":" << cases++ << ",\"N\":" << f.N << ",\"K\":" << K
                      << ",\"segments\":" << ns.size() << ",\"float_tail\":" << tail << ",\"residual\":" << ep
                      << ",\"lanes\":" << L << ",\"pattern\":" << pattern << ",\"sparse\":" << sparse
                      << ",\"S\":" << cfg[0] << ",\"walk\":" << cfg[1] << ",\"poison\":" << poison
                      << ",\"selected_outputs_partials_exact\":" << eq
                      << ",\"guards_inputs_counts_zeros\":" << guard << ",\"FP64_pass\":" << fp
                      << ",\"repeat_exact\":" << rep << ",\"pass\":" << pass << "}\n";
                  out.flush();
                  if (!pass && bad <= 12)
                    std::cout << "FAILcase=" << cases - 1 << " eq=" << eq << " guard=" << guard
                              << " fp=" << fp << " repeat=" << rep << std::endl;
                }
              }
        bool immutable = f.immutable();
        if (!immutable)
          ++bad;
        out << "{\"type\":\"weight_immutability\",\"pass\":" << immutable << "}\n";
      };
      for (uint32_t K : {2048u, 2304u}) {
        for (bool ep : {false, true}) {
          fixture(K, {768}, 0, ep, false);
          if (ep)
            fixture(K, {768}, 0, true, false, 128);
          else
            fixture(K, {384, 384}, 0, false, false);
        }
        fixture(K, {384, 384}, 1, false, false);
        fixture(K, {384, 384}, 2, false, false);
      }
      fixture(5120, {10240, 6144}, 1, false, true);
      fixture(5120, {12288, 1024, 1024}, 0, false, true);
      fixture(6144, {5120}, 0, true, true);
      fixture(6144, {5120}, 0, false, true);
      uint32_t aliases = 0;
      auto aliasFixture = [&](uint32_t K, const std::vector<uint32_t> &ns, bool big, uint32_t padding = 0) {
        Fixture f(b, K, ns, 0, rng, bank, padding);
        std::vector<std::array<uint32_t, 2>> configs;
        if (big) {
          ops::Linear linear(b.capabilities());
          auto cfg = linear.decodePlan(f.projection, 8, ops::LinearEpilogue::Residual).configuration();
          configs.push_back({cfg.splits, uint32_t(cfg.spread)});
        } else
          for (uint32_t S : {1u, 2u, 4u, 8u, 16u})
            for (uint32_t walk : {0u, 1u})
              configs.push_back({S, walk});
        for (auto cfg : configs)
          for (uint32_t L : big ? std::vector<uint32_t>{5, 7, 8} : std::vector<uint32_t>{5, 6, 7, 8})
            for (uint32_t pattern = 0; pattern < (big ? 4u : 3u); ++pattern) {
              std::array<uint32_t, 8> rows;
              rows.fill(0xffffffff);
              for (uint32_t l = 0; l < L; ++l)
                rows[l] = pattern == 0 ? 1 : pattern == 1 ? 4 : l % 2 ? 4 : 1;
              std::vector<uint16_t> x(uint64_t{64} * K, 0x7fc0), aux(uint64_t{64} * f.N);
              std::uniform_real_distribution<float> d(-1, 1);
              for (auto &v : aux)
                v = floatToBf16(d(rng) * 64);
              for (uint32_t l = 0; l < L; ++l)
                for (uint32_t r = 0; r < rows[l]; ++r)
                  for (uint32_t k = 0; k < K; ++k)
                    x[(uint64_t{l} * 8 + r) * K + k] = floatToBf16(d(rng));
              Run a(b, f, L, rows, x, aux, cfg[0], cfg[1], true, false, 0x7e800000),
                  c(b, f, L, rows, x, aux, cfg[0], cfg[1], true, true, 0x7e800000);
              c.output = c.addend;
              CommandGraph ga, gc;
              a.add(ga, b);
              c.add(gc, b);
              static_cast<void>(b.submitCommandAsync(ga.dispatches()).wait());
              static_cast<void>(b.submitCommandAsync(gc.dispatches()).wait());
              bool eq = a.selected() == c.selected() && c.partialsExact(a), fp = a.bounds() && c.bounds(),
                   guard = a.guard(0x7e800000);
              for (const Guarded *v :
                   {&c.input, &c.addend, &c.counts, &c.pi, &c.pr, &c.po, &c.output, &c.partials, &c.counters})
                guard = guard && v->intact();
              guard = guard && !std::memcmp(c.input.view.contents(), x.data(), x.size() * 2) &&
                      !std::memcmp(c.counts.view.contents(), rows.data(), 32);
              auto ctr = static_cast<const uint32_t *>(c.counters.view.contents());
              guard =
                  guard && std::all_of(ctr, ctr + c.counters.bytes / 4, [](uint32_t v) { return v == 0; });
              auto op = static_cast<const uint16_t *>(c.output.view.contents());
              for (uint32_t q = 0; q < 64; ++q)
                if (q / 8 >= L || q % 8 >= rows[q / 8])
                  for (uint32_t n = 0; n < f.columns(); ++n)
                    guard = guard && op[uint64_t{q} * f.N + n] == 0;
              for (uint32_t q = 0; q < 64; ++q)
                for (uint32_t n = f.columns(); n < f.N; ++n)
                  guard = guard && op[uint64_t{q} * f.N + n] == aux[uint64_t{q} * f.N + n];
              bool pass = eq && fp && guard && f.immutable();
              if (!pass)
                ++bad;
              ++aliases;
              out << "{\"type\":\"alias_case\",\"case\":" << aliases << ",\"pass\":" << pass << "}\n";
              out.flush();
            }
      };
      aliasFixture(2048, {768}, false);
      aliasFixture(2048, {768}, false, 128);
      aliasFixture(6144, {5120}, true);
      uint32_t negative = 0;
      {
        Fixture f(b, 2048, {768}, 0, rng, bank);
        ops::Linear linear(b.capabilities());
        auto plan = ops::Linear::plan({{f.N, f.K}, 64, ops::LinearPhase::Decode, ops::LinearEpilogue::None},
                                      {ops::LinearTile::GgufStaged, 2, false, false, false},
                                      ops::FloatOutput::BFloat16);
        Guarded input(b, uint64_t{64} * f.K * 2, 0), output(b, uint64_t{64} * f.N * 2, 0), counts(b, 32, 4),
            pi(b, uint64_t{32} * f.K * 2, 0), pr(b, uint64_t{32} * f.N * 2, 0),
            po(b, uint64_t{32} * f.N * 2, 0), parts(b, plan.scratchSize().partials, 0),
            counters(b, plan.scratchSize().counters, 0);
        for (uint32_t kind = 0; kind < 12; ++kind) {
          CommandGraph g;
          auto projection = f.projection;
          auto selectedPlan = plan;
          uint32_t lanes = 8;
          auto source = input.view;
          MetalBuffer residual;
          ops::VerifyBodyPackBuffers packed{pi.view, pr.view, po.view};
          ops::LinearScratch scratch{.partials = parts.view, .counters = counters.view};
          bool threw = false, active = false;
          if (kind == 0)
            lanes = 0;
          if (kind == 1)
            lanes = 4;
          if (kind == 2)
            lanes = 9;
          if (kind == 3)
            packed.input = {};
          if (kind == 4)
            projection.rotation.signs = counts.view;
          if (kind == 5)
            projection.destination = ops::FloatOutput::Float32;
          if (kind == 6)
            selectedPlan =
                ops::Linear::plan({{f.N, f.K}, 32, ops::LinearPhase::Decode, ops::LinearEpilogue::None},
                                  plan.configuration(), ops::FloatOutput::BFloat16);
          if (kind == 7) {
            projection.inputSize += 64;
          }
          if (kind == 8) {
            auto blocks = projection.blocks();
            blocks.segments[0].formatId = GGUF_FMT_AF5G64;
            projection = ops::Projection(f.N, f.K, blocks);
          }
          if (kind == 9)
            source = b.view(input.view, 0, 2);
          if (kind == 10)
            scratch.partials = {};
          if (kind == 11) {
            auto segment = f.projection.blocks().segments.front();
            segment.outputSize = 384;
            auto second = segment;
            second.columnOffset = 384;
            projection = ops::Projection(f.N, f.K, ops::BlockWeights{{segment, second}});
            selectedPlan =
                ops::Linear::plan({{f.N, f.K}, 64, ops::LinearPhase::Decode, ops::LinearEpilogue::Residual},
                                  plan.configuration(), ops::FloatOutput::BFloat16);
            residual = output.view;
          }
          try {
            active = ops::addVerifyBodyPack(linear, g, projection, selectedPlan, source, residual,
                                            output.view, scratch, counts.view, packed, lanes);
          } catch (const std::invalid_argument &) {
            threw = true;
          }
          const bool pass = !active && g.dispatches().empty() && (kind == 9 || kind == 10 ? threw : !threw);
          if (!pass)
            ++bad;
          ++negative;
          out << "{\"type\":\"negative_host_guard\",\"kind\":" << kind << ",\"pass\":" << pass << "}\n";
        }
      }
      out << "{\"type\":\"summary\",\"cases\":" << cases << ",\"failed\":" << bad << "}\n";
      if (report.is_open())
        report.close();
      check(cases == 15840 && negative == 12 && aliases == 252, "bodyintegrationenumeration");
      std::cout << "BODY_RUNTIME_INTEGRATION cases=" << cases << " failed=" << bad << std::endl;
      return bad ? 1 : 0;
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return 2;
    }
  }
}
