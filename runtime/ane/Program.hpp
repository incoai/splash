#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::ane {

// Memory the GPU and the Neural Engine share without a copy: an IOSurface of
// `rows` rows of `width` fp16 or int8 elements, each row padded to 64 bytes,
// and the Metal buffer over it.
struct Surface final {
  enum class Element : uint8_t { Float16, Int8 };

  [[nodiscard]] static Surface create(metal::MetalBackend &backend, uint32_t rows, uint32_t width,
                                      Element element);
  // The whole 16 KiB pages such a surface takes.
  [[nodiscard]] static uint64_t bytes(uint32_t rows, uint32_t width, Element element) noexcept;

  std::shared_ptr<void> surface;
  metal::MetalBuffer buffer;
  uint32_t strideBytes = 0;
};

// A MIL program compiled and loaded on the Neural Engine through the private
// AppleNeuralEngine client, the service behind Core ML. Evaluations run
// asynchronously, ordered against Metal work by a shared event.
class Program final {
public:
  // `weights` is the blob file the program's constants name as
  // "@model_path/weights.bin". The service keeps the compiled program under
  // a key of the source's hash, whose files stay in a directory of the
  // user's temporary directory, so later processes load it without
  // compiling it again.
  Program(std::string_view mil, std::span<const uint8_t> weights);
  ~Program();
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  // The program's input names in the order enqueue() takes their surfaces.
  [[nodiscard]] const std::vector<std::string> &inputs() const noexcept;

  // Queues one evaluation that starts once `event` reaches `wait` and raises
  // it to `signal` when done. done(false) reports a failed evaluation, after
  // the event is raised anyway so that Metal work waiting on it continues.
  // Throws if the evaluation cannot be queued.
  void enqueue(std::span<const Surface> inputs, const Surface &output, const metal::SharedEvent &event,
               uint64_t wait, uint64_t signal, std::function<void(bool)> done);

  // Raises `event` to at least `value` from the CPU, releasing evaluations
  // whose Metal work will never signal them.
  static void release(const metal::SharedEvent &event, uint64_t value) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::ane
