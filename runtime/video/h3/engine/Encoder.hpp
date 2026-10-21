#pragma once

// The engine's kernels on Splash's Metal backend (metal/MetalBackend.hpp), which owns the device, the command queue,
// the kernel library and its pipelines. An Encoder records the dispatches of one command the way the engine's operators
// bind their kernels: each argument at the kernel's own index, the arguments bound staying bound from one dispatch to
// the next, as they do on a Metal compute encoder. A command's dispatches run in order, each after the one before.
#include "metal/MetalBackend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace h3 {

namespace metal = splash::metal;

// A byte range of one of the backend's buffers, from `offset` to its end: what a kernel argument binds.
struct BufferView {
  metal::MetalBuffer buffer;
  uint64_t offset = 0;

  BufferView() = default;
  BufferView(metal::MetalBuffer b, uint64_t off = 0) : buffer(std::move(b)), offset(off) {}  // NOLINT: a buffer is a view at 0

  [[nodiscard]] BufferView operator+(uint64_t bytes) const { return {buffer, offset + bytes}; }
  [[nodiscard]] void *contents() const { return static_cast<uint8_t *>(buffer.contents()) + offset; }
  [[nodiscard]] explicit operator bool() const { return static_cast<bool>(buffer); }
};

// A grid or threadgroup size.
[[nodiscard]] inline metal::DispatchSize size3(uint64_t x, uint64_t y = 1, uint64_t z = 1) { return {x, y, z}; }

class Encoder {
public:
  explicit Encoder(metal::MetalBackend &backend) : backend_(&backend) {}
  Encoder(const Encoder &) = delete;
  Encoder &operator=(const Encoder &) = delete;
  Encoder(Encoder &&) noexcept = default;
  Encoder &operator=(Encoder &&) noexcept = default;

  void pipeline(std::string name) { pipeline_ = std::move(name); }
  void bind(const BufferView &view, uint32_t index);
  void bytes(const void *data, size_t size, uint32_t index);
  template <class T>
  void bytes(const T &value, uint32_t index) {
    bytes(&value, sizeof value, index);
  }
  // Whole threadgroups, or exactly `threads` threads, Metal trimming the threadgroups at the grid's edges.
  void dispatch(metal::DispatchSize threadgroups, metal::DispatchSize threadsPerThreadgroup);
  void dispatchThreads(metal::DispatchSize threads, metal::DispatchSize threadsPerThreadgroup);

  [[nodiscard]] bool empty() const { return dispatches_.empty(); }
  [[nodiscard]] const std::vector<metal::ComputeDispatch> &dispatches() const { return dispatches_; }

private:
  // Entries of a kernel's buffer argument table.
  static constexpr uint32_t kArguments = 31;

  metal::ComputeDispatch &record(metal::DispatchSize threadsPerThreadgroup);

  metal::MetalBackend *backend_;
  std::string pipeline_;
  std::array<metal::MetalBuffer, kArguments> buffers_{};
  std::array<std::pair<const std::byte *, size_t>, kArguments> bytes_{};
  std::deque<std::vector<std::byte>> payloads_;
  std::vector<metal::ComputeDispatch> dispatches_;
};

// Submits an encoder's dispatches, at least one, as one command, and waits for a submitted command: its GPU seconds. A
// command that failed throws GpuError.
[[nodiscard]] metal::CommandTicket submit(metal::MetalBackend &backend, const Encoder &encoder);
double wait(metal::CommandTicket &ticket);
double run(metal::MetalBackend &backend, const Encoder &encoder);

// A Shared buffer of at least 16 bytes, and one holding `data`.
[[nodiscard]] metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes, std::string_view label);
[[nodiscard]] metal::MetalBuffer upload(metal::MetalBackend &backend, const void *data, uint64_t bytes,
                                        std::string_view label);
// The Metal memory the process holds now, MiB.
[[nodiscard]] double deviceMb(const metal::MetalBackend &backend);

} // namespace h3
