#include "metal/CommandGraph.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
using namespace splash::metal;
struct Params {
  uint32_t width, stages, columns, stage;
};
enum Kind {
  StaticEight,
  Scan,
  FineQueue,
  NoopQueue,
  StaticThree,
  CoarseQueue,
  Halo,
  StaticEightPlain,
  StaticThreePlain,
  KindCount
};
const char *names[] = {"static_eight",      "scan",
                       "fine_queue",        "coordination_only_queue",
                       "static_three",      "coarse_queue",
                       "static_halo",       "static_eight_plain",
                       "static_three_plain"};
static_assert(std::size(names) == KindCount);
struct Sample {
  double gpu, wall, submit;
};
using Clock = std::chrono::steady_clock;
double micros(Clock::duration elapsed) {
  return std::chrono::duration<double, std::micro>(elapsed).count();
}
struct Fixture {
  MetalBackend &backend;
  Params p;
  MetalBuffer input, state, scratch;
  std::array<MetalBuffer, KindCount> output;
  std::vector<float> gold;
  std::array<CommandGraph, KindCount> graph;
  const uint32_t guard = 64;
  Fixture(MetalBackend &b, uint32_t columns, uint32_t workers, uint32_t seed)
      : backend(b), p{32, 8, columns, 0} {
    const auto alloc = [&](uint64_t bytes) {
      return b.allocateBuffer(bytes, BufferStorage::Shared, "DAG experiment");
    };
    input = alloc(uint64_t{p.width} * columns * 4);
    state = alloc((4 * p.width * p.stages + 4 + guard) * 4);
    for (uint32_t i = 0; i < p.width * columns; ++i)
      static_cast<float *>(input.contents())[i] =
          float(int((i * 31 + seed * 17) % 257) - 128) / 256;
    scratch = alloc((uint64_t{p.width} * 3 * columns + guard) * 4);
    std::fill_n(static_cast<uint32_t *>(scratch.contents()),
                scratch.sizeBytes() / 4, 0x7fc00000u);
    for (auto &buffer : output) {
      buffer = alloc((uint64_t{p.width} * p.stages * columns + guard) * 4);
      std::fill_n(static_cast<uint32_t *>(buffer.contents()),
                  buffer.sizeBytes() / 4, 0x7fc00000u);
    }
    for (uint32_t stage = 0; stage < 8; ++stage) {
      p.stage = stage;
      graph[StaticEight].add("static_stage", {input, output[StaticEight]}, p,
                             {p.width, 1, 1});
    }
    for (uint32_t stage = 0; stage < 3; ++stage) {
      p.stage = stage;
      graph[StaticThree].add("static_coarse", {input, output[StaticThree]}, p,
                             {p.width, 1, 1});
    }
    for (uint32_t stage = 0; stage < 8; ++stage) {
      p.stage = stage;
      graph[StaticEightPlain].add("static_stage_plain",
                                  {input, output[StaticEightPlain]}, p,
                                  {p.width, 1, 1});
    }
    for (uint32_t stage = 0; stage < 3; ++stage) {
      p.stage = stage;
      graph[StaticThreePlain].add("static_coarse_plain",
                                  {input, output[StaticThreePlain]}, p,
                                  {p.width, 1, 1});
    }
    for (auto pair :
         {std::pair{Scan, "dynamic_dag"}, std::pair{FineQueue, "dynamic_queue"},
          std::pair{NoopQueue, "queue_noop"},
          std::pair{CoarseQueue, "queue_coarse"}})
      graph[pair.first].add(pair.second, {input, output[pair.first], state}, p,
                            {workers, 1, 1});
    graph[Halo].add("static_halo", {input, output[Halo], scratch}, p,
                    {p.width, 1, 1});
    run(StaticEight);
    std::vector<float> reference(p.width * p.stages * columns);
    const auto *in = static_cast<const float *>(input.contents());
    for (uint32_t job = 0; job < p.width * p.stages; ++job) {
      const uint32_t stage = job / p.width, row = job % p.width;
      const float *a = stage ? reference.data() + (job - p.width) * columns
                             : in + row * columns;
      const float *other =
          stage ? reference.data() +
                      ((stage - 1) * p.width + (row + 1) % p.width) * columns
                : a;
      double sum = 0;
      if (stage % 3 == 2)
        for (uint32_t col = 0; col < columns; ++col)
          sum += double(a[col]) * a[col];
      const float inverse = float(1 / std::sqrt(sum / columns + 1e-6));
      for (uint32_t col = 0; col < columns; ++col)
        reference[job * columns + col] =
            stage % 3 == 0   ? a[col] * .875f + float(col % 31) * .001f
            : stage % 3 == 1 ? a[col] / (1 + std::exp(-other[col]))
                             : a[col] * inverse;
    }
    const auto *actual =
        static_cast<const float *>(output[StaticEight].contents());
    for (uint64_t i = 0; i < reference.size(); ++i)
      if (!std::isfinite(actual[i]) ||
          std::fabs(actual[i] - reference[i]) >
              2e-5f + 2e-5f * std::fabs(reference[i]))
        throw std::runtime_error("static DAG differs from CPU oracle");
    gold.assign(actual, actual + reference.size());
    run(StaticThree);
    run(StaticEightPlain);
    run(StaticThreePlain);
  }
  Sample run(Kind kind, bool verifyTensors = true) {
    const bool coarse = kind == CoarseQueue;
    const uint32_t jobs = p.width * (coarse ? 3 : p.stages);
    const auto began = Clock::now();
    const bool dynamic = kind == Scan || kind == FineQueue ||
                         kind == NoopQueue || kind == CoarseQueue;
    if (dynamic) {
      std::memset(state.contents(), 0, state.sizeBytes());
      auto *init = static_cast<uint32_t *>(state.contents());
      init[jobs + 3] = p.width;
      for (uint32_t j = 0; j < jobs; ++j) {
        const uint32_t stage = j / p.width;
        init[jobs + 4 + j] =
            stage ? (coarse ? 2 : (stage % 3 == 1 ? 2 : 1)) : 0;
        if (j < p.width) {
          init[2 * jobs + 4 + j] = j;
          init[3 * jobs + 4 + j] = 1;
        }
      }
    }
    auto ticket = backend.submitCommandAsync(graph[kind].dispatches());
    const auto submitted = Clock::now();
    const auto result = ticket.wait();
    const auto ended = Clock::now();
    if (kind == Scan || kind == FineQueue || kind == NoopQueue ||
        kind == CoarseQueue) {
      const auto *s = static_cast<const uint32_t *>(state.contents());
      if (s[jobs] != jobs || s[jobs + 1])
        throw std::runtime_error(std::string(names[kind]) +
                                 ": bounded scheduler incomplete");
      for (uint32_t i = 0; i < jobs; ++i)
        if (s[i] != 2)
          throw std::runtime_error("task incomplete");
      for (uint32_t i = 4 * jobs + 4; i < state.sizeBytes() / 4; ++i)
        if (s[i])
          throw std::runtime_error("state guard overwritten");
    }
    if (kind == Halo)
      for (uint64_t i = uint64_t{p.width} * 3 * p.columns;
           i < scratch.sizeBytes() / 4; ++i)
        if (static_cast<const uint32_t *>(scratch.contents())[i] != 0x7fc00000u)
          throw std::runtime_error("halo scratch guard overwritten");
    const uint64_t words = uint64_t{p.width} * p.stages * p.columns;
    const auto *bits = static_cast<const uint32_t *>(output[kind].contents());
    for (uint64_t i = words; i < output[kind].sizeBytes() / 4; ++i)
      if (bits[i] != 0x7fc00000u)
        throw std::runtime_error("output guard overwritten");
    if (verifyTensors)
      verifyOutput(kind);
    return {result.gpuSeconds * 1e6, micros(ended - began),
            micros(submitted - began)};
  }
  void verifyOutput(Kind kind) const {
    if (gold.empty() || kind == NoopQueue)
      return;
    const uint64_t words = uint64_t{p.width} * p.stages * p.columns;
    if (kind != NoopQueue &&
        std::memcmp(output[kind].contents(), gold.data(), words * 4)) {
      const auto *a = gold.data();
      const auto *b = static_cast<const float *>(output[kind].contents());
      uint32_t shown = 0;
      for (uint64_t i = 0; i < words && shown < 8; ++i)
        if (std::memcmp(a + i, b + i, 4)) {
          std::fprintf(stderr,
                       "MISMATCH mode=%s stage=%llu row=%llu col=%llu "
                       "static=%.9g candidate=%.9g\n",
                       names[kind], i / (p.width * p.columns),
                       (i / p.columns) % p.width, i % p.columns, a[i], b[i]);
          ++shown;
        }
      throw std::runtime_error(std::string(names[kind]) +
                               ": intermediate bytes differ");
    }
  }
};
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 2)
        return 2;
      bool validate = false, holdout = false, confirm = false,
           readbackEachRun = false;
      uint32_t seed = 0, blocks = 8;
      for (int i = 2; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--validate-only")
          validate = true;
        else if (arg == "--holdout")
          holdout = true;
        else if (arg == "--confirm")
          confirm = true;
        else if (arg == "--readback-each-run")
          readbackEachRun = true;
        else if (arg == "--blocks" && i + 1 < argc)
          blocks = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--seed" && i + 1 < argc)
          seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else
          return 2;
      }
      if (!blocks || blocks > 128)
        return 2;
      MetalBackend backend(argv[1]);
      bool first = true;
      std::printf("{\"scope\":\"synthetic eight-stage256task DAG;queue "
                  "initialization excluded;no model "
                  "inference\",\"validation_only\":%s,\"seed\":%u,\"holdout\":%"
                  "s,\"cases\":[",
                  validate ? "true" : "false", seed,
                  holdout ? "true" : "false");
      const std::vector<uint32_t> columns =
          holdout ? std::vector<uint32_t>{512, 1024, 4096}
                  : std::vector<uint32_t>{256, 2048, 5120};
      for (uint32_t width : columns)
        for (uint32_t workers : {1u, 4u, 8u, 20u}) {
          if (confirm && workers != 20)
            continue;
          Fixture fixture(backend, width, workers, seed);
          const std::vector<std::pair<Kind, Kind>> comparisons = {
              {Scan, StaticEight},
              {FineQueue, StaticEight},
              {NoopQueue, StaticEight},
              {CoarseQueue, FineQueue},
              {CoarseQueue, StaticThree},
              {StaticThree, StaticEight},
              {Halo, StaticThree},
              {Halo, StaticEight},
              {StaticEightPlain, StaticEight},
              {StaticThreePlain, StaticEightPlain},
              {StaticThreePlain, StaticThree},
              {Halo, StaticThreePlain}};
          for (auto [candidate, baseline] : comparisons) {
            if (confirm && !(candidate == StaticThreePlain &&
                             baseline == StaticEightPlain))
              continue;
            if ((candidate == StaticThree || candidate == Halo ||
                 candidate == StaticEightPlain ||
                 candidate == StaticThreePlain) &&
                workers != 20)
              continue;
            fixture.run(baseline);
            fixture.run(candidate);
            if (!first)
              std::printf(",");
            first = false;
            std::printf(
                "{\"columns\":%u,\"workers\":%u,\"mode\":\"%s\",\"baseline\":"
                "\"%s\",\"equivalent_workload\":%s,\"exact_all_intermediates\":"
                "%s,\"cpu_reference_pass\":%s,\"baseline_cpu_reference_pass\":"
                "true,\"rounds_abba_us\":[",
                width, workers, names[candidate], names[baseline],
                candidate == NoopQueue ? "false" : "true",
                candidate == NoopQueue ? "false" : "true",
                candidate == NoopQueue ? "false" : "true");
            std::vector<std::array<double, 4>> wall, submit;
            if (validate) {
              for (uint32_t repeat = 0; repeat < 8; ++repeat) {
                fixture.run(baseline);
                fixture.run(candidate);
              }
            } else {
              for (uint32_t round = 0; round < 4 + blocks; ++round) {
                const auto a = fixture.run(baseline, readbackEachRun),
                           b = fixture.run(candidate, readbackEachRun),
                           c = fixture.run(candidate, readbackEachRun),
                           d = fixture.run(baseline, readbackEachRun);
                if (round >= 4) {
                  std::printf("%s[%.6f,%.6f,%.6f,%.6f]", round == 4 ? "" : ",",
                              a.gpu, b.gpu, c.gpu, d.gpu);
                  wall.push_back({a.wall, b.wall, c.wall, d.wall});
                  submit.push_back({a.submit, b.submit, c.submit, d.submit});
                }
              }
              fixture.verifyOutput(baseline);
              fixture.verifyOutput(candidate);
            }
            auto emit = [](const std::vector<std::array<double, 4>> &rows) {
              bool first = true;
              for (const auto &r : rows) {
                std::printf("%s[%.6f,%.6f,%.6f,%.6f]", first ? "" : ",", r[0],
                            r[1], r[2], r[3]);
                first = false;
              }
            };
            std::printf("],\"rounds_abba_wall_us\":[");
            emit(wall);
            std::printf("],\"rounds_abba_submit_us\":[");
            emit(submit);
            std::printf("]}");
            std::fflush(stdout);
          }
        }
      std::puts("]}");
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
