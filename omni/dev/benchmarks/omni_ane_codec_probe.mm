// Probes whether the speech tokenizer decoder's residual unit runs on the
// Neural Engine (runtime/ane/Program), and how fast: x + conv1x1(snake(
// conv_k7_dilated(snake(x)))) over C channels and T new rows, the input
// carrying (7 - 1) * dilation rows of history before them, as a streaming
// chunk would. SnakeBeta is x + b * sin(a x)^2. Checks against the CPU in
// fp64 and times serial evaluations.
//   omni-ane-codec-probe METALLIB [--channels C] [--rows T] [--dilation D]

#include "ane/Program.hpp"
#include "metal/MetalBackend.hpp"

#include <chrono>
#include <cstring>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using namespace splash;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omni-ane-codec-probe METALLIB [--channels C] [--rows T] [--dilation D]\n");
    return 2;
  }
  uint32_t C = 96, T = 9600, dilation = 3;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const uint32_t value = std::stoul(argv[i + 1]);
    if (flag == "--channels") C = value;
    else if (flag == "--rows") T = value;
    else if (flag == "--dilation") dilation = value;
  }
  try {
    if (const auto reason = ane::unavailable()) {
      std::printf("Neural Engine unavailable: %s\n", reason->c_str());
      return 1;
    }
    metal::MetalBackend backend(argv[1]);
    const uint32_t history = 6 * dilation, width = T + history;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> value(-1.0F, 1.0F);
    // Constants: conv1 [C][C][1][7], conv2 [C][C][1][1], snake a and b per channel (two snakes).
    std::vector<_Float16> w1(size_t(C) * C * 7), w2(size_t(C) * C), a1(C), b1(C), a2(C), b2(C);
    for (auto &w : w1) w = _Float16(value(rng) / std::sqrt(float(C * 7)));
    for (auto &w : w2) w = _Float16(value(rng) / std::sqrt(float(C)));
    for (uint32_t c = 0; c < C; ++c) {
      a1[c] = _Float16(1.0F + 0.5F * value(rng));
      b1[c] = _Float16(0.5F + 0.25F * value(rng));
      a2[c] = _Float16(1.0F + 0.5F * value(rng));
      b2[c] = _Float16(0.5F + 0.25F * value(rng));
    }
    // One weight file of several fp16 constants: the header (blob count,
    // storage version), one 64-byte record per constant from offset 64
    // (sentinel, type, bytes, data offset), then the data, 64-byte aligned.
    // A constant's MIL offset is its record's.
    const std::vector<const std::vector<_Float16> *> tensors{&w1, &w2, &a1, &b1, &a2, &b2};
    std::vector<uint64_t> offsets;
    uint64_t data = 64 + 64 * tensors.size();
    std::vector<uint8_t> blob(data);
    const auto put = [&](uint64_t offset, auto v) { std::memcpy(blob.data() + offset, &v, sizeof v); };
    put(0, uint32_t(tensors.size()));
    put(4, uint32_t{2});
    for (size_t i = 0; i < tensors.size(); ++i) {
      const uint64_t record = 64 + 64 * i, bytes = tensors[i]->size() * 2;
      offsets.push_back(record);
      put(record, uint32_t{0xdeadbeef});
      put(record + 4, uint32_t{1});
      put(record + 8, bytes);
      put(record + 16, data);
      blob.resize(data + bytes);
      std::memcpy(blob.data() + data, tensors[i]->data(), bytes);
      data = (data + bytes + 63) / 64 * 64;
      blob.resize(data);
    }
    ane::Surface input = ane::Surface::create(backend, C, width, ane::Surface::Element::Float16);
    ane::Surface output = ane::Surface::create(backend, C, T, ane::Surface::Element::Float16);
    const std::string c = std::to_string(C), w = std::to_string(width), t = std::to_string(T), d = std::to_string(dilation);
    const auto constant = [&](const std::string &name, const std::string &type, size_t index) {
      return "        " + type + " " + name + " = const()[name = string(\"" + name + "\"), val = " + type +
             "(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" + std::to_string(offsets[index]) +
             ")))];\n";
    };
    const std::string cw = "tensor<fp16, [1, " + c + ", 1, " + w + "]>", ct = "tensor<fp16, [1, " + c + ", 1, " + t + "]>";
    const std::string vec = "tensor<fp16, [1, " + c + ", 1, 1]>";
    std::string mil = "program(1.3)\n{\n    func main<ios18>(" + input.bufferType(C, width) + " x) {\n";
    mil += "        tensor<fp16, [1, 1, " + c + ", " + w + "]> xt = tensor_buffer_to_tensor<ios17>(input = x);\n";
    mil += "        " + cw + " x4 = reshape(x = xt, shape = tensor<int32, [4]>([1, " + c + ", 1, " + w + "]));\n";
    mil += constant("w1", "tensor<fp16, [" + c + ", " + c + ", 1, 7]>", 0);
    mil += constant("w2", "tensor<fp16, [" + c + ", " + c + ", 1, 1]>", 1);
    mil += constant("a1", vec, 2) + constant("b1", vec, 3) + constant("a2", vec, 4) + constant("b2", vec, 5);
    mil += "        " + cw + " s1a = mul(x = x4, y = a1);\n";
    mil += "        " + cw + " s1s = sin(x = s1a);\n";
    mil += "        " + cw + " s1q = mul(x = s1s, y = s1s);\n";
    mil += "        " + cw + " s1b = mul(x = s1q, y = b1);\n";
    mil += "        " + cw + " s1 = add(x = x4, y = s1b);\n";
    mil += "        " + ct + " h = conv(dilations = tensor<int32, [2]>([1, " + d +
           "]), groups = int32(1), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = "
           "tensor<int32, [2]>([1, 1]), weight = w1, x = s1);\n";
    mil += "        " + ct + " s2a = mul(x = h, y = a2);\n";
    mil += "        " + ct + " s2s = sin(x = s2a);\n";
    mil += "        " + ct + " s2q = mul(x = s2s, y = s2s);\n";
    mil += "        " + ct + " s2b = mul(x = s2q, y = b2);\n";
    mil += "        " + ct + " s2 = add(x = h, y = s2b);\n";
    mil += "        " + ct + " y4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(1), pad = tensor<int32, "
           "[4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = w2, x = s2);\n";
    mil += "        " + ct + " xr = slice_by_size(x = x4, begin = tensor<int32, [4]>([0, 0, 0, " + std::to_string(history) +
           "]), size = tensor<int32, [4]>([1, " + c + ", 1, " + t + "]));\n";
    mil += "        " + ct + " ys = add(x = y4, y = xr);\n";
    mil += "        tensor<fp16, [1, 1, " + c + ", " + t + "]> yt = reshape(x = ys, shape = tensor<int32, [4]>([1, 1, " + c +
           ", " + t + "]));\n";
    mil += "        " + output.bufferType(C, T) +
           " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
           "strides = tensor<int64, [4]>(" + output.strides(C) + "));\n";
    mil += "    } -> (y);\n}\n";
    const auto t0 = std::chrono::steady_clock::now();
    ane::Program program(mil, blob, {std::chrono::seconds(180), [] { return false; }});
    std::printf("unit of %u channels over %u rows (dilation %u) compiled and loaded in %.2f s\n", C, T, dilation,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const ane::Surface inputs[] = {input};
    const ane::Program::Binding binding = program.bind(program.procedure("main"), inputs, output);
    std::vector<float> x(size_t(C) * width);
    for (float &v : x) v = value(rng);
    auto *base = static_cast<uint8_t *>(input.buffer.contents());
    for (uint32_t ch = 0; ch < C; ++ch)
      for (uint32_t i = 0; i < width; ++i)
        reinterpret_cast<_Float16 *>(base + size_t(ch) * input.strideBytes)[i] = _Float16(x[size_t(ch) * width + i]);
    const metal::SharedEvent event = backend.newSharedEvent();
    std::mutex mutex;
    std::condition_variable done;
    uint64_t reported = 0;
    bool failed = false;
    const auto run = [&](uint64_t wait) {
      program.enqueue(binding, event, wait, wait + 1, [&, wait](bool ok) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ok) failed = true;
        reported = std::max(reported, wait + 1);
        done.notify_all();
      });
      event.signal(wait);
      std::unique_lock<std::mutex> lock(mutex);
      if (!done.wait_for(lock, std::chrono::seconds(30), [&] { return failed || reported >= wait + 1; }))
        throw std::runtime_error("the Neural Engine did not report");
      if (failed) throw std::runtime_error("the Neural Engine reported a failure");
    };
    run(1);
    // fp64 reference on a few channels and rows.
    const auto snake = [](double v, double a, double b) { const double s = std::sin(a * v); return v + b * s * s; };
    const auto *out = static_cast<const uint8_t *>(output.buffer.contents());
    double worst = 0.0, peak = 0.0;
    for (uint32_t row = 0; row < T; row += std::max<uint32_t>(1, T / 37)) {
      std::vector<double> hrow(C);
      for (uint32_t co = 0; co < C; ++co) {
        double s = 0.0;
        for (uint32_t ci = 0; ci < C; ++ci)
          for (uint32_t k = 0; k < 7; ++k) {
            const double v = float(_Float16(x[size_t(ci) * width + row + k * dilation]));
            s += double(float(w1[(size_t(co) * C + ci) * 7 + k])) * snake(v, float(a1[ci]), float(b1[ci]));
          }
        hrow[co] = s;
      }
      for (uint32_t co = 0; co < C; ++co) {
        double s = 0.0;
        for (uint32_t ci = 0; ci < C; ++ci)
          s += double(float(w2[size_t(co) * C + ci])) * snake(hrow[ci], float(a2[ci]), float(b2[ci]));
        const double expected = s + float(_Float16(x[size_t(co) * width + history + row]));
        const double got = float(reinterpret_cast<const _Float16 *>(out + size_t(co) * output.strideBytes)[row]);
        worst = std::max(worst, std::fabs(got - expected));
        peak = std::max(peak, std::fabs(expected));
      }
    }
    std::printf("result: max abs error %.3e of peak %.3f\n", worst, peak);
    uint64_t next = 3;
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i, next += 2) run(next);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 20;
    const double flops = 2.0 * T * C * C * 8;
    std::printf("evaluation: %.3f ms (%.2f TFLOPS of the two convolutions)\n", ms, flops / ms / 1e9);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni ane codec probe: %s\n", error.what());
    return 1;
  }
  return 0;
}
