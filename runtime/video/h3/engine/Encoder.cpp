#include "video/h3/engine/Encoder.hpp"

#include "video/h3/engine/Errors.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace h3 {

void Encoder::bind(const BufferView &view, uint32_t index) {
  if (index >= kArguments || !view.buffer)
    throw std::logic_error(pipeline_ + ": argument " + std::to_string(index) + " out of range or empty");
  const metal::MetalBuffer &base = view.buffer;
  buffers_[index] = view.offset ? backend_->view(base, view.offset, base.sizeBytes() - view.offset) : base;
  bytes_[index] = {};
}

void Encoder::bytes(const void *data, size_t size, uint32_t index) {
  if (index >= kArguments || !data || !size)
    throw std::logic_error(pipeline_ + ": argument " + std::to_string(index) + " out of range or empty");
  const auto *first = static_cast<const std::byte *>(data);
  payloads_.emplace_back(first, first + size);
  bytes_[index] = {payloads_.back().data(), size};
  buffers_[index] = {};
}

metal::ComputeDispatch &Encoder::record(metal::DispatchSize threadsPerThreadgroup) {
  metal::ComputeDispatch &dispatch = dispatches_.emplace_back();
  dispatch.pipelineName = pipeline_;
  dispatch.threadsPerThreadgroup = threadsPerThreadgroup;
  for (uint32_t index = 0; index < kArguments; ++index) {
    if (buffers_[index]) dispatch.buffers.push_back({index, buffers_[index]});
    else if (bytes_[index].first) dispatch.bytes.push_back({index, bytes_[index].first, bytes_[index].second});
  }
  return dispatch;
}

void Encoder::dispatch(metal::DispatchSize threadgroups, metal::DispatchSize threadsPerThreadgroup) {
  record(threadsPerThreadgroup).threadgroups = threadgroups;
}

void Encoder::dispatchThreads(metal::DispatchSize threads, metal::DispatchSize threadsPerThreadgroup) {
  record(threadsPerThreadgroup).gridThreads = threads;
}

metal::CommandTicket submit(metal::MetalBackend &backend, const Encoder &encoder) {
  return backend.submitCommandAsync(encoder.dispatches());
}

double wait(metal::CommandTicket &ticket) {
  try {
    return ticket.wait().gpuSeconds;
  } catch (const metal::MetalAllocationError &) {
    throw;
  } catch (const metal::MetalBackendError &error) {
    throw GpuError(error.what());
  }
}

double run(metal::MetalBackend &backend, const Encoder &encoder) {
  metal::CommandTicket ticket = submit(backend, encoder);
  return wait(ticket);
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes, std::string_view label) {
  return backend.allocateBuffer(std::max<uint64_t>(bytes, 16), metal::BufferStorage::Shared, label);
}

metal::MetalBuffer upload(metal::MetalBackend &backend, const void *data, uint64_t bytes, std::string_view label) {
  metal::MetalBuffer buffer = allocate(backend, bytes, label);
  if (data && bytes) std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

double deviceMb(const metal::MetalBackend &backend) {
  return static_cast<double>(backend.refreshMemoryStats().deviceCurrentAllocatedBytes) / 1048576.0;
}

} // namespace h3
