#include "metal/CommandGraph.hpp"
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace splash::metal;
struct Params {
  uint32_t width, stages, columns, stage;
};
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2)
        return 2;
      MetalBackend backend(argv[1]);
      bool first = true;
      std::printf("{\"scope\":\"bounded synthetic DAG;256 tasks,8 "
                  "stages;affine,SiLU product,RMS;GPU time excludes CPU queue "
                  "initialization;not model inference\",\"cases\":[");
      for (uint32_t columns : {256u, 2048u, 5120u}) {
        Params p{32, 8, columns, 0};
        const uint32_t jobs = p.width * p.stages;
        const auto alloc = [&](uint64_t size) {
          return backend.allocateBuffer(size, BufferStorage::Shared,
                                        "DAG probe");
        };
        auto input = alloc(uint64_t{p.width} * columns * 4),
             control = alloc(uint64_t{jobs} * columns * 4),
             candidate = alloc(control.sizeBytes()),
             state = alloc((4 * jobs + 4) * 4);
        for (uint32_t i = 0; i < p.width * columns; ++i)
          static_cast<float *>(input.contents())[i] =
              float(int((i * 31) % 257) - 128) / 256;
        CommandGraph ordinary;
        for (uint32_t stage = 0; stage < p.stages; ++stage) {
          p.stage = stage;
          ordinary.add("static_stage", {input, control}, p, {p.width, 1, 1});
        }
        static_cast<void>(
            backend.submitCommandAsync(ordinary.dispatches()).wait());
        // Independent CPU reference checks every intermediate row, not only the
        // final stage.
        std::vector<float> reference(jobs * columns);
        const auto *in = static_cast<const float *>(input.contents());
        for (uint32_t job = 0; job < jobs; ++job) {
          const uint32_t stage = job / p.width, row = job % p.width;
          const float *a = stage ? reference.data() + (job - p.width) * columns
                                 : in + row * columns;
          const float *b =
              stage
                  ? reference.data() +
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
                : stage % 3 == 1 ? a[col] / (1 + std::exp(-b[col]))
                                 : a[col] * inverse;
        }
        const float *actual = static_cast<const float *>(control.contents());
        for (uint32_t i = 0; i < reference.size(); ++i)
          if (!std::isfinite(actual[i]) ||
              std::fabs(actual[i] - reference[i]) >
                  2e-5f + 2e-5f * std::fabs(reference[i]))
            throw std::runtime_error(
                "static DAG differs from independent CPU reference");
        for (uint32_t workers : {1u, 4u, 8u, 20u})
          for (bool queued : {false, true}) {
            CommandGraph dynamic;
            dynamic.add(queued ? "dynamic_queue" : "dynamic_dag",
                        {input, candidate, state}, p, {workers, 1, 1});
            if (!first)
              std::printf(",");
            first = false;
            std::printf("{\"columns\":%u,\"workers\":%u,\"queue\":%s,\"exact_"
                        "all_intermediates\":true,\"cpu_reference_pass\":true,"
                        "\"rounds_abba_us\":[",
                        columns, workers, queued ? "true" : "false");
            auto run = [&](bool current) {
              std::memset(state.contents(), 0, state.sizeBytes());
              auto *init = static_cast<uint32_t *>(state.contents());
              init[jobs + 3] = p.width;
              for (uint32_t j = 0; j < jobs; ++j) {
                const uint32_t stage = j / p.width;
                init[jobs + 4 + j] = stage ? (stage % 3 == 1 ? 2 : 1) : 0;
                if (j < p.width) {
                  init[2 * jobs + 4 + j] = j;
                  init[3 * jobs + 4 + j] = 1;
                }
              }
              const auto &graph = current ? dynamic : ordinary;
              const auto result =
                  backend.submitCommandAsync(graph.dispatches()).wait();
              if (current) {
                const auto *s = static_cast<const uint32_t *>(state.contents());
                if (s[jobs] != jobs || s[jobs + 1])
                  throw std::runtime_error("bounded DAG scheduler incomplete");
                for (uint32_t i = 0; i < jobs; ++i)
                  if (s[i] != 2)
                    throw std::runtime_error("DAG task incomplete");
                if (std::memcmp(control.contents(), candidate.contents(),
                                control.sizeBytes()))
                  throw std::runtime_error("dynamic DAG bytes differ");
              }
              return result.gpuSeconds * 1e6;
            };
            for (uint32_t round = 0; round < 12; ++round) {
              const double a = run(false), b = run(true), c = run(true),
                           d = run(false);
              if (round >= 4)
                std::printf("%s[%.6f,%.6f,%.6f,%.6f]", round == 4 ? "" : ",", a,
                            b, c, d);
            }
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
