#include "video/h3/metal/MetalBackend.hpp"

#include <algorithm>
#include <cstring>

namespace h3::metal {

Status MetalBackend::create(const std::string &metallib, int queueDepth, std::unique_ptr<MetalBackend> &out) {
  auto backend = std::unique_ptr<MetalBackend>(new MetalBackend());
  backend->device_ = MTLCreateSystemDefaultDevice();
  if (!backend->device_) return error(Code::Metal, "no Metal device");
  backend->queue_ = [backend->device_ newCommandQueueWithMaxCommandBufferCount:queueDepth];
  NSError *err = nil;
  backend->library_ = [backend->device_ newLibraryWithURL:[NSURL fileURLWithPath:@(metallib.c_str())] error:&err];
  if (!backend->library_)
    return error(Code::Metal, "metallib %s: %s", metallib.c_str(), err.localizedDescription.UTF8String);
  H3_TRY(backend->checkRegisterLayout());
  out = std::move(backend);
  return {};
}

Status MetalBackend::prepare(const std::vector<std::string> &kernels) {
  for (const std::string &name : kernels) {
    if (pipelines_.count(name)) continue;
    id<MTLFunction> function = [library_ newFunctionWithName:@(name.c_str())];
    if (!function) return error(Code::Metal, "kernel %s not in the metallib", name.c_str());
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:function error:&err];
    if (!pso) return error(Code::Metal, "pipeline %s: %s", name.c_str(), err.localizedDescription.UTF8String);
    pipelines_[name] = pso;
  }
  return {};
}

id<MTLComputePipelineState> MetalBackend::pipeline(const std::string &kernel) const {
  auto it = pipelines_.find(kernel);
  return it == pipelines_.end() ? nil : it->second;
}

Status MetalBackend::allocate(size_t bytes, __strong id<MTLBuffer> &out, const char *what) {
  out = [device_ newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
  if (!out) return error(Code::Metal, "allocation of %zu bytes failed (%s)", bytes, what);
  return {};
}

Status MetalBackend::upload(const void *data, size_t bytes, __strong id<MTLBuffer> &out, const char *what) {
  H3_TRY(allocate(bytes, out, what));
  if (data && bytes) std::memcpy(out.contents, data, bytes);
  return {};
}

// The attention kernels address cooperative-tensor slots by a formula (kernels/common/window.h) instead of the MPP
// accessors; a library or OS change of the register layout would silently corrupt them, so it is checked here.
Status MetalBackend::checkRegisterLayout() {
  H3_TRY(prepare({"layout_check"}));
  id<MTLBuffer> scratch, result;
  H3_TRY(allocate(64 * 128 * 2, scratch, "layout check"));
  H3_TRY(allocate(32 * 3 * sizeof(int32_t), result, "layout check"));
  std::memset(scratch.contents, 0, scratch.length);
  H3_TRY(runSync([&](id<MTLComputeCommandEncoder> encoder) {
    [encoder setComputePipelineState:pipeline("layout_check")];
    bind(encoder, scratch, 0);
    bind(encoder, result, 1);
    dispatch(encoder, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
  }));
  const int32_t *mismatches = static_cast<const int32_t *>(result.contents);
  int32_t total = 0;
  for (int i = 0; i < 32 * 3; ++i) total += mismatches[i];
  if (total != 0)
    return error(Code::Metal, "MPP cooperative-tensor register layout differs from the kernels' slot formula "
                              "(%d mismatches)", total);
  return {};
}

} // namespace h3::metal
