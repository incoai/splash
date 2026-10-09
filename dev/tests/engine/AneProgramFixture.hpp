#pragma once

// A small Neural Engine program the ane::Program tests compile: one function
// that adds two surfaces of 64 rows of 64 fp16 values into a third, and its
// evaluations, which the CPU starts and waits for.

#include "AwakeClock.hpp"
#include "ane/Program.hpp"
#include "metal/MetalBackend.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::test {

class AneSum final {
public:
  static constexpr uint32_t kRows = 64, kWidth = 64;

  explicit AneSum(metal::MetalBackend &backend)
      : a_(surface(backend)), b_(surface(backend)), y_(surface(backend)), event_(backend.newSharedEvent()) {}

  // The program, whose one function is named `function`: each name makes a
  // program of its own.
  [[nodiscard]] std::string mil(std::string_view function) const {
    const std::string tensor = "tensor<fp16, [1, 1, 64, 64]>";
    return "program(1.3)\n{\n    func " + std::string(function) + "<ios18>(" + a_.bufferType(kRows, kWidth) +
           " a, " + b_.bufferType(kRows, kWidth) + " b) {\n        " + tensor +
           " at = tensor_buffer_to_tensor<ios17>(input = a);\n        " + tensor +
           " bt = tensor_buffer_to_tensor<ios17>(input = b);\n        " + tensor +
           " s = add(x = at, y = bt);\n        " + y_.bufferType(kRows, kWidth) +
           " y = tensor_to_tensor_buffer<ios17>(input = s, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
           "strides = tensor<int64, [4]>(" +
           y_.strides(kRows) + "));\n    } -> (y);\n}\n";
  }

  // The function's evaluation over the surfaces, its inputs bound by name.
  [[nodiscard]] ane::Program::Binding bind(const ane::Program &program, std::string_view function) const {
    const uint32_t procedure = program.procedure(function);
    std::vector<ane::Surface> inputs;
    for (const std::string &name : program.inputs(procedure)) {
      if (name != "a" && name != "b") throw std::runtime_error("the sum program has an input " + name);
      inputs.push_back(name == "a" ? a_ : b_);
    }
    return program.bind(procedure, inputs, y_);
  }
  // The input surfaces alone, one fewer than the function reads.
  [[nodiscard]] std::vector<ane::Surface> oneInput() const { return {a_}; }
  [[nodiscard]] const ane::Surface &output() const noexcept { return y_; }

  // New random inputs, and an output no sum of them holds.
  void fill(std::mt19937 &random) {
    std::uniform_real_distribution<float> uniform(-4.0f, 4.0f);
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t column = 0; column < kWidth; ++column) {
        at(a_, row, column) = _Float16(uniform(random));
        at(b_, row, column) = _Float16(uniform(random));
        at(y_, row, column) = _Float16(1000.0f);
      }
  }
  // Whether the output holds the inputs' sums, to fp16's rounding.
  [[nodiscard]] bool summed() const {
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t column = 0; column < kWidth; ++column) {
        const float expected = float(at(a_, row, column)) + float(at(b_, row, column));
        if (!(std::fabs(float(at(y_, row, column)) - expected) <= 0x1p-9f * std::max(std::fabs(expected), 1.0f)))
          return false;
      }
    return true;
  }
  [[nodiscard]] std::vector<_Float16> values() const {
    std::vector<_Float16> result;
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t column = 0; column < kWidth; ++column) result.push_back(at(y_, row, column));
    return result;
  }

  // Enqueues one evaluation of `binding` that starts at once, and returns the
  // value it raises the event to when done.
  uint64_t enqueue(ane::Program &program, const ane::Program::Binding &binding,
                   std::function<void(bool)> done) {
    const uint64_t wait = ++value_, signal = ++value_;
    event_.signal(wait);
    program.enqueue(binding, event_, wait, signal, std::move(done));
    return signal;
  }
  // What `done` reports of one evaluation of `binding` that starts at once,
  // or none if it reports nothing within `timeout`.
  std::optional<bool> evaluate(ane::Program &program, const ane::Program::Binding &binding,
                               std::chrono::milliseconds timeout) {
    struct Report final {
      std::mutex mutex;
      std::condition_variable changed;
      std::optional<bool> result;
    };
    const auto report = std::make_shared<Report>();
    static_cast<void>(enqueue(program, binding, [report](bool success) {
      std::lock_guard lock(report->mutex);
      report->result = success;
      report->changed.notify_all();
    }));
    std::unique_lock lock(report->mutex);
    report->changed.wait_until(lock, AwakeClock::now() + timeout, [&] { return report->result.has_value(); });
    return report->result;
  }
  // The value the event has reached.
  [[nodiscard]] uint64_t reached() const {
    return [(__bridge id<MTLSharedEvent>)event_.nativeHandle() signaledValue];
  }

private:
  static ane::Surface surface(metal::MetalBackend &backend) {
    return ane::Surface::create(backend, kRows, kWidth, ane::Surface::Element::Float16);
  }
  static _Float16 &at(const ane::Surface &surface, uint32_t row, uint32_t column) {
    return static_cast<_Float16 *>(surface.buffer.contents())[row * (surface.strideBytes / 2) + column];
  }

  ane::Surface a_, b_, y_;
  metal::SharedEvent event_;
  uint64_t value_ = 0;
};

// The one directory a program's files went to in an empty cache directory.
inline std::filesystem::path programDirectory(const std::filesystem::path &cache) {
  std::vector<std::filesystem::path> entries;
  for (const auto &entry : std::filesystem::directory_iterator(cache))
    if (entry.is_directory()) entries.push_back(entry.path());
  if (entries.size() != 1) throw std::runtime_error("the cache holds other than one program");
  return entries.front();
}

// The message of the E `run` throws, or what it did instead.
template <class E, class F> std::string thrown(F &&run) {
  try {
    run();
  } catch (const E &error) {
    return error.what();
  } catch (const std::exception &error) {
    return std::string("another exception: ") + error.what();
  }
  return "nothing thrown";
}

} // namespace splash::test
