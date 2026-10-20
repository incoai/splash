#pragma once

#import <Metal/Metal.h>

#include "video/h3/engine/Status.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace h3::metal {

// A byte range of a Metal buffer: what a kernel argument binds.
struct BufferView {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;

  BufferView() = default;
  BufferView(id<MTLBuffer> b, size_t off = 0) : buffer(b), offset(off) {}  // NOLINT: a buffer is a view at 0

  [[nodiscard]] BufferView operator+(size_t bytes) const { return {buffer, offset + bytes}; }
  [[nodiscard]] void *contents() const { return static_cast<uint8_t *>(buffer.contents) + offset; }
  [[nodiscard]] explicit operator bool() const { return buffer != nil; }
};

inline void bind(id<MTLComputeCommandEncoder> encoder, BufferView view, int index) {
  [encoder setBuffer:view.buffer offset:view.offset atIndex:index];
}
inline void bind(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer, int index) {
  [encoder setBuffer:buffer offset:0 atIndex:index];
}
template <class T>
inline void bindBytes(id<MTLComputeCommandEncoder> encoder, const T &value, int index) {
  [encoder setBytes:&value length:sizeof value atIndex:index];
}
inline void dispatch(id<MTLComputeCommandEncoder> encoder, MTLSize groups, MTLSize threads) {
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// Device, command queue, the kernel library and its pipeline cache. One per engine.
class MetalBackend {
public:
  static Status create(const std::string &metallib, int queueDepth, std::unique_ptr<MetalBackend> &out);

  [[nodiscard]] id<MTLDevice> device() const { return device_; }
  [[nodiscard]] id<MTLCommandQueue> queue() const { return queue_; }

  // Compiles the named kernels (once); dispatch code then uses pipeline(), which never fails for a prepared name.
  Status prepare(const std::vector<std::string> &kernels);
  [[nodiscard]] id<MTLComputePipelineState> pipeline(const std::string &kernel) const;

  // Shared-storage buffer of at least 16 bytes.
  Status allocate(size_t bytes, __strong id<MTLBuffer> &out, const char *what);
  Status upload(const void *data, size_t bytes, __strong id<MTLBuffer> &out, const char *what);

  // One command buffer with one compute encoder, committed and waited for; GPU time in ms.
  template <class Encode>
  Status runSync(Encode &&encode, double *gpuMs = nullptr) {
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    encode(encoder);
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) return error(Code::Gpu, "command buffer: %s", cb.error.localizedDescription.UTF8String);
    if (gpuMs) *gpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    return {};
  }

  [[nodiscard]] double allocatedMb() const { return device_.currentAllocatedSize / 1048576.0; }

private:
  Status checkRegisterLayout();

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;
};

} // namespace h3::metal
