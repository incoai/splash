#include "FlashQSABulk.hpp"
#include "metal/abi/FlashQSAFast.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace splash::flash {
namespace {
using metal::MetalBuffer;using metal::CommandGraph;using metal::ComputeDispatch;
void require(bool value,const char *message) {if (!value) throw std::invalid_argument(message);}
bool enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
bool isOne(const char *name) {
  const char *value = std::getenv(name);
  return value && std::string_view(value) == "1";
}
std::vector<MetalBuffer> bindings(const ComputeDispatch &d) {
  std::vector<MetalBuffer> result;
  for (const auto &b:d.buffers) {require(b.index==result.size(),"Bulk QSA binding order drift");result.push_back(b.buffer);}
  return result;
}
void append(CommandGraph &graph,const ComputeDispatch &d) {
  require(d.bytes.size()==1,"Bulk QSA parameter binding drift");
  if (d.bytes[0].sizeBytes==sizeof(FlashQSAParams)) {
    FlashQSAParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));
    graph.add(d.pipelineName,bindings(d),p,d.threadgroups,d.threadsPerThreadgroup);
  } else {
    require(d.bytes[0].sizeBytes==sizeof(FlashQSAFastParams),"Bulk QSA parameter ABI drift");
    FlashQSAFastParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));
    graph.add(d.pipelineName,bindings(d),p,d.threadgroups,d.threadsPerThreadgroup);
  }
}
void prefix(metal::MetalBackend &backend,CommandGraph &result,const FlashQSAFastInputs &input,
    FlashQSAState &state,FlashQSAWorkspace &ordinary,FlashQSAFastWorkspace &fast,
    FlashQSABulkPreparedWorkspace &prepared,uint32_t begin,uint32_t rows) {
  require(begin==0 && rows==2048 && state.capacity>=2048,"Bulk QSA gate requires fresh dense2K append");
  CommandGraph qualified;
  addQSABulkPrepared(backend,qualified,input,state,ordinary,fast,prepared,begin,rows);
  const auto dispatches=qualified.dispatches();
  require(dispatches.size()==35 && dispatches[0].pipelineName=="flash_qsa_fast_prepare" &&
          dispatches[1].pipelineName.starts_with("flash_qsa_pool_rope_") &&
          dispatches[2].pipelineName=="flash_qsa_select_blocks","Bulk prefix authoritative plan drift");
  for (uint32_t i=0;i<3;++i) append(result,dispatches[i]);
}
FlashQSAFastParams parameters(const FlashQSAState &state,uint32_t maximumPartitions) {
  return {{2048,0,state.capacity,(state.capacity+3)/4,0,512,0,0,1e-6f,1e7f,0,0},0,0,1,maximumPartitions};
}
} // namespace
bool qsaBulkPrefillEnabled() {
  if (!enabled("SPLASH_FLASH_QSA_BULK_PREFILL")) return false;
  if (!isOne("SPLASH_FLASH_QSA_F32") || !isOne("SPLASH_FLASH_QSA_MPP") ||
      !isOne("SPLASH_FLASH_QSA_ROW_TILES") || !qsaOnlineMPPRowTilesEnabled())
    throw std::invalid_argument("SPLASH_FLASH_QSA_BULK_PREFILL requires QSA_F32=1, QSA_MPP=1 and QSA_ROW_TILES=1");
  return true;
}
bool qsaBulkPrefillSG8Enabled(bool bulkEnabled) {
  if (!enabled("SPLASH_FLASH_QSA_BULK_PREFILL_SG8")) return false;
  if (!bulkEnabled)
    throw std::invalid_argument("SPLASH_FLASH_QSA_BULK_PREFILL_SG8 requires SPLASH_FLASH_QSA_BULK_PREFILL=1 and its dependencies");
  return true;
}
bool qsaBulkPrefillGeometry(uint32_t begin, uint32_t rows, uint32_t capacity,
                            bool verification) noexcept {
  return !verification && begin == 0 && rows == 2048 && capacity >= 2048;
}
uint64_t qsaBulkWorkspacePlannedBytes() {
  return qsaBulkPreparedWorkspacePlannedBytes(2048)+uint64_t(2048)*24*4*258*sizeof(float);
}
FlashQSABulkWorkspace allocateQSABulkWorkspace(metal::MetalBackend &backend) {
  FlashQSABulkWorkspace result;
  result.prepared=allocateQSABulkPreparedWorkspace(backend,2048);
  result.partials={2048,4,
      backend.allocateBuffer(uint64_t(2048)*24*4*2*4,metal::BufferStorage::Shared,"flash-qsa-bulk-partition-statistics"),
      backend.allocateBuffer(uint64_t(2048)*24*4*256*4,metal::BufferStorage::Shared,"flash-qsa-bulk-partition-numerators")};
  return result;
}
void addQSABulkPrefill(metal::MetalBackend &backend,CommandGraph &graph,const FlashQSAFastInputs &input,
    FlashQSAState &state,FlashQSAWorkspace &ordinary,FlashQSAFastWorkspace &ordinaryFast,
    FlashQSABulkWorkspace &bulk,uint32_t begin,uint32_t rows,bool sg8) {
  require(bulk.partials.maximumRows==2048 && bulk.partials.maximumPartitions==4 &&
          bulk.partials.partitionStatistics.sizeBytes()>=uint64_t(2048)*24*4*2*4 &&
          bulk.partials.partitionValues.sizeBytes()>=uint64_t(2048)*24*4*256*4,"Bulk partial geometry invalid");
  CommandGraph result;prefix(backend,result,input,state,ordinary,ordinaryFast,bulk.prepared,begin,rows);
  auto p=parameters(state,4);p.partitions=4;
  const std::vector<MetalBuffer> attentionBindings{bulk.prepared.queries,state.keys,state.values,
      bulk.prepared.selectedBlocks,bulk.partials.partitionStatistics,bulk.partials.partitionValues,input.diagnostics};
  if (sg8) {
    result.add("flash_qsa_mpp_prefill_bulk_early_2048",attentionBindings,p,{3328,1,1},{128,1,1});
    result.add("flash_qsa_mpp_prefill_bulk_temporal_sg8_2048",attentionBindings,p,{4608,1,1},{256,1,1});
  } else result.add("flash_qsa_mpp_prefill_bulk_2048",attentionBindings,p,{7936,1,1},{128,1,1});
  result.add("flash_qsa_fast_prefill_bulk_reduce_2048",{bulk.partials.partitionStatistics,bulk.partials.partitionValues,
      input.qProjection,input.output,input.diagnostics},p,{2048,24,1},{256,1,1});
  for (const auto &d:result.dispatches()) append(graph,d);
}
} // namespace splash::flash
