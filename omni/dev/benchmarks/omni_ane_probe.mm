// Probes the Neural Engine on this Mac (runtime/ane/Program): compiles one
// fp16 program whose function multiplies a constant [N][K] weight by an
// input surface of K rows x R columns, checks the result against the CPU,
// and times evaluations driven from the host over a shared event, alone and
// pipelined. Decides what the codec or a prefill split could expect.
//   omni-ane-probe METALLIB [--n N] [--k K] [--rows R] [--repeat COUNT]

#include "ane/Program.hpp"
#include "metal/MetalBackend.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using namespace splash;

namespace {

std::string dimensions(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omni-ane-probe METALLIB [--n N] [--k K] [--rows R] [--repeat COUNT]\n");
    return 2;
  }
  uint32_t n = 1536, k = 1024, rows = 32, repeat = 50;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const uint32_t value = std::stoul(argv[i + 1]);
    if (flag == "--n") n = value;
    else if (flag == "--k") k = value;
    else if (flag == "--rows") rows = value;
    else if (flag == "--repeat") repeat = value;
  }
  try {
    if (const auto reason = ane::unavailable()) {
      std::printf("Neural Engine unavailable: %s\n", reason->c_str());
      return 1;
    }
    metal::MetalBackend backend(argv[1]);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> value(-1.0F, 1.0F);
    std::vector<_Float16> weights(size_t(n) * k);
    for (_Float16 &w : weights) w = _Float16(value(rng) / std::sqrt(float(k)));
    const std::vector<uint8_t> blob = ane::constantBlob(weights);
    ane::Surface input = ane::Surface::create(backend, k, rows, ane::Surface::Element::Float16);
    ane::Surface output = ane::Surface::create(backend, n, rows, ane::Surface::Element::Float16);
    // The function: y[N][R] = W[N][K] x[K][R].
    const std::string mil =
        "program(1.3)\n{\n    func main<ios18>(" + input.bufferType(k, rows) + " x) {\n" +
        "        tensor<fp16, " + dimensions(k, rows) + "> xt = tensor_buffer_to_tensor<ios17>(input = x);\n" +
        "        tensor<fp16, " + dimensions(n, k) + "> w = const()[name = string(\"w\"), val = tensor<fp16, " +
        dimensions(n, k) + ">(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" +
        std::to_string(ane::kConstantOffset) + ")))];\n" +
        "        tensor<fp16, " + dimensions(n, rows) +
        "> yt = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w, y = xt);\n" +
        "        " + output.bufferType(n, rows) +
        " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
        "strides = tensor<int64, [4]>(" + output.strides(n) + "));\n" +
        "    } -> (y);\n}\n";
    const auto t0 = std::chrono::steady_clock::now();
    ane::Program program(mil, blob, {std::chrono::seconds(120), [] { return false; }});
    std::printf("program of %u x %u over %u rows compiled and loaded in %.2f s\n", n, k, rows,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const uint32_t procedure = program.procedure("main");
    const ane::Surface inputs[] = {input};
    const ane::Program::Binding binding = program.bind(procedure, inputs, output);
    // Input: x[k][r] with the surface's row stride.
    std::vector<float> x(size_t(k) * rows);
    for (float &v : x) v = value(rng);
    auto *xRow = static_cast<uint8_t *>(input.buffer.contents());
    for (uint32_t row = 0; row < k; ++row)
      for (uint32_t c = 0; c < rows; ++c)
        reinterpret_cast<_Float16 *>(xRow + size_t(row) * input.strideBytes)[c] = _Float16(x[size_t(row) * rows + c]);
    const metal::SharedEvent event = backend.newSharedEvent();
    std::mutex mutex;
    std::condition_variable done;
    uint64_t reported = 0;
    bool failed = false;
    const auto run = [&](uint64_t wait, uint64_t signal) {
      program.enqueue(binding, event, wait, signal, [&, signal](bool ok) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ok) failed = true;
        reported = std::max(reported, signal);
        done.notify_all();
      });
    };
    const auto waitFor = [&](uint64_t signal) {
      std::unique_lock<std::mutex> lock(mutex);
      if (!done.wait_for(lock, std::chrono::seconds(30), [&] { return failed || reported >= signal; }))
        throw std::runtime_error("the Neural Engine did not report");
      if (failed) throw std::runtime_error("the Neural Engine reported a failure");
    };
    // One evaluation, checked.
    run(1, 2);
    event.signal(1);
    waitFor(2);
    double worst = 0.0, peak = 0.0;
    const auto *yRow = static_cast<const uint8_t *>(output.buffer.contents());
    for (uint32_t row = 0; row < n; ++row)
      for (uint32_t c = 0; c < rows; ++c) {
        double expected = 0.0;
        for (uint32_t j = 0; j < k; ++j) expected += double(float(weights[size_t(row) * k + j])) * x[size_t(j) * rows + c];
        const double got = float(reinterpret_cast<const _Float16 *>(yRow + size_t(row) * output.strideBytes)[c]);
        worst = std::max(worst, std::fabs(got - expected));
        peak = std::max(peak, std::fabs(expected));
      }
    std::printf("result: max abs error %.3e of peak %.3f\n", worst, peak);
    // Serial evaluations: each waits for the previous report.
    uint64_t next = 3;
    auto t1 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < repeat; ++i) {
      run(next, next + 1);
      event.signal(next);
      waitFor(next + 1);
      next += 2;
    }
    const double serial = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / repeat;
    // Pipelined: all queued, released in order.
    const uint64_t first = next;
    for (uint32_t i = 0; i < repeat; ++i) run(next + 2 * i, next + 2 * i + 1);
    t1 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < repeat; ++i) event.signal(first + 2 * i);
    waitFor(first + 2 * (repeat - 1) + 1);
    const double pipelined = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / repeat;
    const double flops = 2.0 * n * k * rows;
    std::printf("evaluation: %.3f ms serial (%.1f GFLOPS), %.3f ms pipelined (%.1f GFLOPS); weights %.1f MB\n", serial,
                flops / serial / 1e6, pipelined, flops / pipelined / 1e6, n * k * 2 / 1e6);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni ane probe: %s\n", error.what());
    return 1;
  }
  return 0;
}
