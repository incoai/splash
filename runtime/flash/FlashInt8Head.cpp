#include "FlashInt8Head.hpp"

#include "metal/abi/FlashInt8Head.h"

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace splash::flash {
namespace {
constexpr uint32_t kInputs=2560,kOutputs=248320,kMaximumRows=16,kGroup=64;
constexpr std::string_view kPrefix="language_model.lm_head";
constexpr uint64_t kAlignment=16384;
uint64_t product(uint64_t a,uint64_t b) {
  if(b&&a>UINT64_MAX/b) throw std::invalid_argument("Flash INT8 head extent overflows");
  return a*b;
}
uint64_t plus(uint64_t a,uint64_t b) {
  if(a>UINT64_MAX-b) throw std::invalid_argument("Flash INT8 head extent overflows");
  return a+b;
}
uint64_t rounded(uint64_t bytes) {return plus(bytes,kAlignment-1)&~(kAlignment-1);}
uint64_t extent(uint32_t rows,uint64_t stride,uint64_t rowBytes) {
  if(!rows||!rowBytes||stride<rowBytes) throw std::invalid_argument("Flash INT8 head source stride is short");
  return plus(product(rows-1,stride),rowBytes);
}
void requireBuffer(const metal::MetalBuffer &buffer,uint64_t bytes,uintptr_t alignment) {
  if(!buffer||!bytes||buffer.sizeBytes()<bytes||buffer.storage()!=metal::BufferStorage::Shared||
      !buffer.contents()||reinterpret_cast<uintptr_t>(buffer.contents())%alignment)
    throw std::invalid_argument("Flash INT8 head buffer is missing, short, or misaligned");
}
bool overlaps(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  const uintptr_t aa=reinterpret_cast<uintptr_t>(a.contents()),bb=reinterpret_cast<uintptr_t>(b.contents());
  if(!aa||!bb||aa>UINTPTR_MAX-a.sizeBytes()||bb>UINTPTR_MAX-b.sizeBytes())
    throw std::invalid_argument("Flash INT8 head buffer address extent is invalid");
  return aa<bb+b.sizeBytes()&&bb<aa+a.sizeBytes();
}
const FlashAffineProjection &checkedProjection(const FlashWeights &weights) {
#if !defined(SPLASH_INT8_EXPERIMENT) || !SPLASH_INT8_EXPERIMENT
  throw std::invalid_argument("Flash INT8 head requires the hybrid Metal4.1 build");
#endif
  const auto &p=weights.projection(kPrefix);
  if(p.experts!=1||p.outputSize!=kOutputs||p.inputSize!=kInputs||p.bits!=8||p.groupSize!=kGroup||
      !p.weights||!p.scales||!p.biases||p.weights->dtype!=FlashDType::U32||
      p.scales->dtype!=FlashDType::BF16||p.biases->dtype!=FlashDType::BF16||
      p.weights->shape!=std::vector<uint64_t>{kOutputs,kInputs/4}||
      p.scales->shape!=std::vector<uint64_t>{kOutputs,kInputs/kGroup}||
      p.biases->shape!=p.scales->shape||p.parameterRowStrideBytes%2)
    throw std::invalid_argument("Flash INT8 head requires the original Q8/G64 Qwen4 vocabulary projection");
  const auto weightBytes=extent(kOutputs,p.weightRowStrideBytes,kInputs);
  const auto parameterBytes=extent(kOutputs,p.parameterRowStrideBytes,(kInputs/kGroup)*2);
  if(p.weights->logicalBytes<weightBytes||p.scales->logicalBytes<parameterBytes||p.biases->logicalBytes<parameterBytes)
    throw std::invalid_argument("Flash INT8 head source logical extent is short");
  requireBuffer(p.weights->buffer,weightBytes,1);
  requireBuffer(p.scales->buffer,parameterBytes,2);
  requireBuffer(p.biases->buffer,parameterBytes,2);
  return p;
}
std::string digest(const std::string &text) {
  if(text.size()>std::numeric_limits<CC_LONG>::max()) throw std::invalid_argument("Flash INT8 head identity is too large");
  std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> bytes{};
  if(!CC_SHA256(text.data(),static_cast<CC_LONG>(text.size()),bytes.data()))
    throw std::runtime_error("Flash INT8 head SHA256 failed");
  constexpr char hex[]="0123456789abcdef";
  std::string out;out.reserve(bytes.size()*2);
  for(auto byte:bytes) {out+=hex[byte>>4];out+=hex[byte&15];}
  return out;
}
FlashInt8HeadParams params(const FlashAffineProjection &p,uint32_t rows) {
  const uint32_t tile=rows>8?16:8;
  return {rows,(rows+tile-1)/tile*tile,kInputs,kOutputs,8,kGroup,tile,64,
      p.weightRowStrideBytes,p.parameterRowStrideBytes};
}
} // namespace

struct FlashInt8Head::Impl final {
  metal::MetalBackend &backend;
  FlashAffineProjection source;
  metal::MetalBuffer codes,padded,sums,conversionDiagnostics;
  const bool originalCodeStorage;
  std::string identity;
  uint64_t allocatedBytes=0;
  metal::CommandTiming initialization;
  Impl(metal::MetalBackend &value,const FlashWeights &weights,metal::MetalBuffer sharedPadding)
      :backend(value),source(checkedProjection(weights)),originalCodeStorage(source.weightRowStrideBytes==kInputs) {
    const bool reuse=bool(sharedPadding);
    const uint64_t planned=FlashInt8Head::plannedBytes(weights,reuse);
    if(reuse) requireBuffer(sharedPadding,uint64_t{kMaximumRows}*kInputs*2,2);
    const uint64_t before=backend.memoryStats().allocatedBytes;
    // For Q8, source code(k)=sourceByte[k]. Canonical contiguous rows are
    // already UINT8[N,K]; a checked view retains the original allocation and
    // mapped lifetime, verifies backend ownership, and allocates no GPU data.
    codes=originalCodeStorage?backend.view(source.weights->buffer,0,uint64_t{kOutputs}*kInputs):
        backend.allocateBuffer(rounded(uint64_t{kOutputs}*kInputs),metal::BufferStorage::Shared,
            "flash-original-uint8-vocabulary-codes");
    padded=reuse?sharedPadding:backend.allocateBuffer(rounded(uint64_t{kMaximumRows}*kInputs*2),
        metal::BufferStorage::Shared,"flash-int8-head-bf16-padding");
    sums=backend.allocateBuffer(rounded(uint64_t{kMaximumRows}*(kInputs/kGroup)*4),
        metal::BufferStorage::Shared,"flash-int8-head-f32-group-sums");
    conversionDiagnostics=backend.allocateBuffer(kAlignment,metal::BufferStorage::Shared,
        "flash-int8-head-conversion-diagnostics");
    std::memset(conversionDiagnostics.contents(),0,conversionDiagnostics.sizeBytes());
    const std::string fingerprint=std::string(kFlashInt8HeadOperandFormat)+"\n"+
        kFlashInt8HeadSemantics+"\nsource:"+weights.sourceIdentity()+"\nmanifest:"+
        weights.manifestFingerprint()+"\ngeometry:248320,2560,8,64\nstrides:"+
        std::to_string(source.weightRowStrideBytes)+","+std::to_string(source.parameterRowStrideBytes)+
        "\nstorage:"+(originalCodeStorage?"original-q8-byte-view-v1":"copied-q8-layout-v1");
    identity=digest(fingerprint);
    if(!originalCodeStorage) {
      metal::CommandGraph graph;
      graph.add("flash_int8_head_expand",{source.weights->buffer,codes,conversionDiagnostics},params(source,1),
          {(uint64_t{kOutputs}*kInputs+255)/256,1,1},{256,1,1});
      initialization=backend.submitCommand(graph.dispatches());
      uint32_t status=0;std::memcpy(&status,conversionDiagnostics.contents(),sizeof(status));
      if(status) throw std::runtime_error("Flash INT8 head conversion diagnostics failed: "+std::to_string(status));
    }
    const uint64_t after=backend.memoryStats().allocatedBytes;
    if(after<before||after-before>planned) throw std::logic_error("Flash INT8 head allocation exceeds plan");
    allocatedBytes=after-before;
  }
};

FlashInt8Head::FlashInt8Head(metal::MetalBackend &backend,const FlashWeights &weights,
    metal::MetalBuffer sharedPadding):impl_(std::make_unique<Impl>(backend,weights,sharedPadding)) {}
FlashInt8Head::~FlashInt8Head()=default;
FlashInt8Head::FlashInt8Head(FlashInt8Head &&) noexcept=default;
FlashInt8Head &FlashInt8Head::operator=(FlashInt8Head &&) noexcept=default;
uint64_t FlashInt8Head::plannedBytes(const FlashWeights &weights,bool reusePadding) {
  const auto &source=checkedProjection(weights);
  uint64_t total=2*kAlignment;
  if(source.weightRowStrideBytes!=kInputs) total=plus(total,rounded(uint64_t{kOutputs}*kInputs));
  if(!reusePadding) total=plus(total,rounded(uint64_t{kMaximumRows}*kInputs*2));
  return total;
}
uint64_t FlashInt8Head::allocatedBytes() const noexcept {return impl_?impl_->allocatedBytes:0;}
bool FlashInt8Head::usesOriginalCodeStorage() const noexcept {return impl_&&impl_->originalCodeStorage;}
const char *FlashInt8Head::codeStorageSemantics() const noexcept {
  return usesOriginalCodeStorage()?"head-original-q8-byte-view-zero-copy-v1":"head-original-q8-layout-copy-v1";
}
const std::string &FlashInt8Head::identitySha256() const {
  if(!impl_) throw std::logic_error("Flash INT8 head is not initialized");
  return impl_->identity;
}
metal::CommandTiming FlashInt8Head::initializationTiming() const noexcept {
  return impl_?impl_->initialization:metal::CommandTiming{};
}
std::vector<metal::MetalBuffer> FlashInt8Head::immutableWeightBuffers() const {
  return impl_?std::vector<metal::MetalBuffer>{impl_->codes}:std::vector<metal::MetalBuffer>{};
}
std::vector<metal::MetalBuffer> FlashInt8Head::scratchBuffers() const {
  return impl_?std::vector<metal::MetalBuffer>{impl_->padded,impl_->sums,impl_->conversionDiagnostics}:
      std::vector<metal::MetalBuffer>{};
}
void FlashInt8Head::addProjection(metal::CommandGraph &graph,metal::MetalBuffer input,
    metal::MetalBuffer output,metal::MetalBuffer diagnostics,uint32_t rows) const {
  if(!impl_||rows<2||rows>kMaximumRows) throw std::invalid_argument("Flash INT8 head requires real rows2..16");
  const auto p=params(impl_->source,rows);
  requireBuffer(input,uint64_t{rows}*kInputs*2,2);
  requireBuffer(output,uint64_t{rows}*kOutputs*2,2);
  requireBuffer(diagnostics,4,4);
  requireBuffer(impl_->padded,uint64_t{p.padded_rows}*kInputs*2,2);
  requireBuffer(impl_->sums,uint64_t{p.padded_rows}*(kInputs/kGroup)*4,4);
  const std::array immutable{impl_->source.weights->buffer,impl_->source.scales->buffer,
      impl_->source.biases->buffer,impl_->codes,input,impl_->conversionDiagnostics};
  const std::array writable{impl_->padded,impl_->sums,output,diagnostics};
  for(const auto &a:immutable) for(const auto &b:writable)
    if(overlaps(a,b)) throw std::invalid_argument("Flash INT8 head writable view aliases immutable source/input");
  for(size_t a=0;a<writable.size();++a) for(size_t b=a+1;b<writable.size();++b)
    if(overlaps(writable[a],writable[b])) throw std::invalid_argument("Flash INT8 head writable views overlap");
  graph.add("flash_int8_head_pad",{input,impl_->padded,diagnostics},p,
      {(uint64_t{p.padded_rows}*kInputs+255)/256,1,1},{256,1,1});
  graph.add("flash_int8_head_group_sums",{impl_->padded,impl_->sums,diagnostics},p,
      {p.padded_rows,kInputs/kGroup,1},{32,1,1});
  graph.add(p.tile_rows==8?"flash_int8_head_m8_n64":"flash_int8_head_m16_n64",
      {impl_->padded,impl_->codes,impl_->source.scales->buffer,impl_->source.biases->buffer,
       impl_->sums,output,diagnostics},p,{kOutputs/64,p.padded_rows/p.tile_rows,1},{128,1,1});
}

} // namespace splash::flash
