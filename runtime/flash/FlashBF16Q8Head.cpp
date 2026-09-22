#include "FlashBF16Q8Head.hpp"
#include "metal/abi/FlashInt8Head.h"
#include "metal/abi/FlashDenseSmallRows.h"
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash {
bool flashBF16Q8HeadEnabled() {
  static const bool enabled=[] {
    const char *value=std::getenv("SPLASH_FLASH_MTP_Q8_BF16_REGISTER");
    if(!value||std::string_view(value)=="0")return false;
    if(std::string_view(value)=="1")return true;
    throw std::invalid_argument("SPLASH_FLASH_MTP_Q8_BF16_REGISTER must be 0 or 1");
  }();return enabled;
}
bool flashBF16Q8HeadGeometry(const FlashAffineProjection &p,uint32_t rows) noexcept {
  return rows>=2&&rows<=4&&p.experts==1&&p.outputSize==248320&&p.inputSize==2560&&
      p.bits==8&&p.groupSize==64&&p.weightRowStrideBytes>=2560&&
      p.weightRowStrideBytes<=(UINT64_MAX-2560)/248319&&
      p.parameterRowStrideBytes>=80&&p.parameterRowStrideBytes%2==0&&
      p.parameterRowStrideBytes<=(UINT64_MAX-80)/248319;
}
namespace {
void requireBuffer(const metal::MetalBuffer &buffer,uint64_t bytes,uint64_t alignment,const char *name) {
  if(!buffer||buffer.sizeBytes()<bytes||buffer.storage()!=metal::BufferStorage::Shared||
      !buffer.contents()||reinterpret_cast<uintptr_t>(buffer.contents())%alignment)
    throw std::invalid_argument(std::string("Flash BF16 Q8 head missing/short/misaligned Shared ")+name);
}
bool overlaps(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  const uintptr_t first=reinterpret_cast<uintptr_t>(a.contents()),second=reinterpret_cast<uintptr_t>(b.contents());
  if(!first||!second||first>UINTPTR_MAX-a.sizeBytes()||second>UINTPTR_MAX-b.sizeBytes())
    throw std::invalid_argument("Flash BF16 Q8 head invalid buffer address extent");
  return first<second+b.sizeBytes()&&second<first+a.sizeBytes();
}
}
FlashBF16Q8Head::FlashBF16Q8Head(metal::MetalBackend &backend,const FlashWeights &weights,metal::MetalBuffer padding)
    :source_(weights.projection("language_model.lm_head")),padded_(padding) {
#if !defined(SPLASH_INT8_EXPERIMENT)||!SPLASH_INT8_EXPERIMENT
  throw std::invalid_argument("Flash BF16 Q8 register head requires hybrid Metal 4.1 build");
#endif
  if(weights.sourceIdentity()!="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"||
      !flashBF16Q8HeadGeometry(source_,4)||!source_.weights||!source_.scales||!source_.biases||
      source_.weights->dtype!=FlashDType::U32||source_.scales->dtype!=FlashDType::BF16||
      source_.biases->dtype!=FlashDType::BF16||source_.weights->shape!=std::vector<uint64_t>{248320,640}||
      source_.scales->shape!=std::vector<uint64_t>{248320,40}||source_.biases->shape!=source_.scales->shape)
    throw std::invalid_argument("Flash BF16 Q8 register head requires qualified original Qwen4 Q8/G64 source");
  const uint64_t weightBytes=uint64_t{248319}*source_.weightRowStrideBytes+2560;
  const uint64_t parameterBytes=uint64_t{248319}*source_.parameterRowStrideBytes+80;
  if(source_.weights->logicalBytes<weightBytes||source_.scales->logicalBytes<parameterBytes||source_.biases->logicalBytes<parameterBytes)
    throw std::invalid_argument("Flash BF16 Q8 register head source logical extent is short");
  requireBuffer(source_.weights->buffer,weightBytes,1,"original codes");
  requireBuffer(source_.scales->buffer,parameterBytes,2,"original scales");
  requireBuffer(source_.biases->buffer,parameterBytes,2,"original biases");
  requireBuffer(padded_,8*2560*2,2,"existing padding");
  padded_=backend.view(padded_,0,8*2560*2);
}
void FlashBF16Q8Head::addProjection(metal::CommandGraph &graph,metal::MetalBuffer input,
    metal::MetalBuffer output,metal::MetalBuffer diagnostics,uint32_t rows) const {
  if(!flashBF16Q8HeadGeometry(source_,rows))
    throw std::invalid_argument("Flash BF16 Q8 register head requires 2..4 real Last rows");
  requireBuffer(input,uint64_t{rows}*2560*2,2,"input");
  requireBuffer(output,uint64_t{rows}*248320*2,2,"output");requireBuffer(diagnostics,4,4,"diagnostics");
  const std::array immutable{source_.weights->buffer,source_.scales->buffer,source_.biases->buffer,input};
  const std::array writable{padded_,output,diagnostics};
  for(const auto &a:immutable)for(const auto &b:writable)
    if(overlaps(a,b))throw std::invalid_argument("Flash BF16 Q8 register head writable aliases source/input");
  for(size_t a=0;a<writable.size();++a)for(size_t b=a+1;b<writable.size();++b)
    if(overlaps(writable[a],writable[b]))throw std::invalid_argument("Flash BF16 Q8 register head writable buffers overlap");
  const FlashDenseSmallRowsParams pad{rows,8,2560,248320,0,248320,8,64};
  graph.add("flash_dense_small_rows_pad",{input,padded_,diagnostics},pad,{80,1,1},{256,1,1});
  const FlashInt8HeadParams p{rows,8,2560,248320,8,64,8,32,
      source_.weightRowStrideBytes,source_.parameterRowStrideBytes};
  graph.add("flash_bf16_q8_head_register_m8_n32_k64_s1",
      {padded_,source_.weights->buffer,source_.scales->buffer,source_.biases->buffer,padded_,output,diagnostics},
      p,{7760,1,1},{32,1,1});
}
} // namespace splash::flash
