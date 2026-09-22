#include "FlashQSABulkPrepare.hpp"
#include "metal/abi/FlashQSAFast.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace splash::flash {
namespace {
using metal::CommandGraph;
using metal::MetalBuffer;
using metal::ComputeDispatch;
[[noreturn]] void fail(const char *message) { throw std::invalid_argument(message); }
void require(bool valid, const char *message) { if (!valid) fail(message); }
std::vector<MetalBuffer> bindings(const ComputeDispatch &dispatch) {
  std::vector<MetalBuffer> result;
  result.reserve(dispatch.buffers.size());
  for (const auto &binding : dispatch.buffers) {
    require(binding.index == result.size(), "Bulk QSA binding order changed");
    result.push_back(binding.buffer);
  }
  return result;
}
template<class Params> Params parameters(const ComputeDispatch &dispatch) {
  require(dispatch.bytes.size() == 1 && dispatch.bytes[0].sizeBytes == sizeof(Params),
          "Bulk QSA parameter ABI changed");
  Params result{};
  std::memcpy(&result, dispatch.bytes[0].data, sizeof(result));
  return result;
}
void append(CommandGraph &graph, const ComputeDispatch &dispatch) {
  if (dispatch.bytes.size() != 1) fail("Bulk QSA parameter binding count changed");
  if (dispatch.bytes[0].sizeBytes == sizeof(FlashQSAParams))
    graph.add(dispatch.pipelineName, bindings(dispatch), parameters<FlashQSAParams>(dispatch),
              dispatch.threadgroups, dispatch.threadsPerThreadgroup);
  else if (dispatch.bytes[0].sizeBytes == sizeof(FlashQSAFastParams))
    graph.add(dispatch.pipelineName, bindings(dispatch), parameters<FlashQSAFastParams>(dispatch),
              dispatch.threadgroups, dispatch.threadsPerThreadgroup);
  else fail("Bulk QSA unrecognized authoritative parameter size");
}
void fullBuffer(const MetalBuffer &buffer, uint64_t bytes) {
  require(buffer && buffer.contents() && buffer.sizeBytes() >= bytes,
          "Bulk QSA insufficient full input/bulk plane");
}
void requireDisjoint(const MetalBuffer &a, const MetalBuffer &b) {
  const auto aa = reinterpret_cast<uintptr_t>(a.contents());
  const auto bb = reinterpret_cast<uintptr_t>(b.contents());
  require(aa && bb, "Bulk QSA requires addressable planes");
  require(aa <= bb ? bb-aa >= a.sizeBytes() : aa-bb >= b.sizeBytes(),
          "Bulk QSA full planes overlap");
}
} // namespace

uint64_t qsaBulkPreparedWorkspacePlannedBytes(uint32_t rows) {
  require(rows >= 256 && rows <= 2048, "Bulk QSA workspace rows must be256..2048");
  return uint64_t(rows) * (6144 * 2 + 512 * 2 + 512 * 4);
}
FlashQSABulkPreparedWorkspace allocateQSABulkPreparedWorkspace(metal::MetalBackend &backend,
                                                        uint32_t rows) {
  (void)qsaBulkPreparedWorkspacePlannedBytes(rows);
  return {rows,
      backend.allocateBuffer(uint64_t(rows)*6144*2, metal::BufferStorage::Shared,
                             "flash-qsa-bulk-queries"),
      backend.allocateBuffer(uint64_t(rows)*512*2, metal::BufferStorage::Shared,
                             "flash-qsa-bulk-index-queries"),
      backend.allocateBuffer(uint64_t(rows)*512*4, metal::BufferStorage::Shared,
                             "flash-qsa-bulk-dense-selection")};
}
bool qsaBulkPreparedGeometry(uint32_t begin, uint32_t rows, uint32_t capacity) noexcept {
  return capacity && capacity <= 262144 && rows >= 256 && rows <= 2048 &&
         begin <= 2048 && rows <= 2048-begin && begin <= capacity && rows <= capacity-begin;
}
FlashQSAFastInputs sliceQSABulkInputs(metal::MetalBackend &backend,
    const FlashQSAFastInputs &input, uint32_t offset, uint32_t rows) {
  const auto slice = [&](const MetalBuffer &buffer, uint32_t width, uint32_t bytes=2) {
    const uint64_t rowBytes = uint64_t(width)*bytes;
    return backend.view(buffer, uint64_t(offset)*rowBytes, uint64_t(rows)*rowBytes);
  };
  auto result = input;
  result.qProjection = slice(input.qProjection,12288);
  result.kProjection = slice(input.kProjection,512);
  result.vProjection = slice(input.vProjection,512);
  result.indexProjection = slice(input.indexProjection,640);
  result.output = slice(input.output,6144);
  if (input.positions) result.positions = slice(input.positions,1,8);
  return result;
}
void addQSAChronologicalChunk(CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &workspace, FlashQSAFastWorkspace &fast,
    uint32_t begin, uint32_t rows) {
  const auto partitions = qsaOnlineMPPRoutePartitions(begin,rows);
  if (partitions)
    addQSAOnlineMPP(graph,input,state,workspace,fast,begin,rows,partitions,true);
  else
    addQSAFast(graph,input,state,workspace,fast,begin,rows,
               FlashQSAFastMode::PartitionedF32Probabilities,4,true);
}
void addQSAChronologicalChunks(metal::MetalBackend &backend, CommandGraph &graph,
    const FlashQSAFastInputs &input, FlashQSAState &state, FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast, uint32_t begin, uint32_t rows) {
  require(ordinary.maximumRows == 128 && rows && begin <= state.capacity &&
          rows <= state.capacity-begin, "Ordinary QSA chunk geometry invalid");
  for (uint32_t offset=0; offset<rows; offset+=128) {
    const auto count=std::min(128u,rows-offset);
    const auto part=sliceQSABulkInputs(backend,input,offset,count);
    addQSAChronologicalChunk(graph,part,state,ordinary,fast,begin+offset,count);
  }
}
void addQSABulkPrepared(metal::MetalBackend &backend, CommandGraph &graph,
    const FlashQSAFastInputs &input, FlashQSAState &state, FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast, FlashQSABulkPreparedWorkspace &bulk,
    uint32_t begin, uint32_t rows) {
  require(qsaBulkPreparedGeometry(begin,rows,state.capacity) &&
          ordinary.maximumRows == 128 && rows <= bulk.maximumRows,
          "Bulk QSA only supports dense appends ending<=2048 with128-row attention");
  for (const auto &[buffer,width,bytes] :
       {std::tuple{input.qProjection,12288u,2u},std::tuple{input.kProjection,512u,2u},
        std::tuple{input.vProjection,512u,2u},std::tuple{input.indexProjection,640u,2u},
        std::tuple{input.output,6144u,2u},std::tuple{bulk.queries,6144u,2u},
        std::tuple{bulk.indexQueries,512u,2u},std::tuple{bulk.selectedBlocks,512u,4u}})
    fullBuffer(buffer,uint64_t(rows)*width*bytes);
  if (input.positions) fullBuffer(input.positions,uint64_t(rows)*8);
  const std::vector planes{input.qProjection,input.kProjection,input.vProjection,
      input.indexProjection,input.output,bulk.queries,bulk.indexQueries,bulk.selectedBlocks};
  for (size_t i=0;i<planes.size();++i)
    for (size_t j=i+1;j<planes.size();++j) requireDisjoint(planes[i],planes[j]);

  std::vector<MetalBuffer> existing{state.keys,state.values,state.rawIndexKeys,
      state.pooledKeys,state.indexPositions,fast.partitionStatistics,
      fast.partitionValues,input.diagnostics};
  if (input.positions) existing.push_back(input.positions);
  for (const auto *tensor:{input.qNorm,input.kNorm,input.indexQNorm,input.indexKNorm}) {
    require(tensor,"Bulk QSA missing immutable norm tensor");
    existing.push_back(tensor->buffer);
  }
  for (const auto &plane:{bulk.queries,bulk.indexQueries,bulk.selectedBlocks})
    for (const auto &buffer:existing) requireDisjoint(plane,buffer);

  // Complete validation before changing the caller's graph. The authoritative
  // graph for each original128-row window owns all route/parameter decisions.
  std::vector<CommandGraph> chunks;
  chunks.reserve((rows+127)/128);
  for (uint32_t offset=0; offset<rows; offset+=128) {
    const auto count=std::min(128u,rows-offset);
    auto workspace=ordinary;
    workspace.queries=backend.view(bulk.queries,uint64_t(offset)*6144*2,uint64_t(count)*6144*2);
    workspace.indexQueries=backend.view(bulk.indexQueries,uint64_t(offset)*512*2,uint64_t(count)*512*2);
    workspace.selectedBlocks=backend.view(bulk.selectedBlocks,uint64_t(offset)*512*4,uint64_t(count)*512*4);
    auto &qualified=chunks.emplace_back();
    addQSAChronologicalChunk(qualified,sliceQSABulkInputs(backend,input,offset,count),
                        state,workspace,fast,begin+offset,count);
    require(qualified.dispatches().size() >= 4 && qualified.dispatches().size() <= 5,
            "Coalesced dense QSA authoritative graph gained a new phase");
    require(qualified.dispatches()[0].pipelineName == "flash_qsa_fast_prepare",
            "Bulk QSA authoritative preparation changed");
    for (const auto &dispatch:qualified.dispatches())
      require(dispatch.pipelineName != "flash_qsa_index_scores",
              "Bulk QSA must never skip a sparse indexer");
    require(qualified.dispatches().back().pipelineName == "flash_qsa_fast_f32_reduce",
            "Bulk QSA authoritative attention reducer changed");
  }
  const auto source=chunks.front().dispatches();
  const auto &prepare=source.front();
  auto prepareParams=parameters<FlashQSAFastParams>(prepare);
  auto &common=prepareParams.common;
  common.rows=rows;
  common.begin=begin;
  common.first_block=begin/4;
  common.new_blocks=(begin+rows)/4-begin/4;
  auto prepareBindings=bindings(prepare);
  require(prepareBindings.size() == 15, "Bulk QSA preparation binding ABI changed");
  prepareBindings[0]=input.qProjection; prepareBindings[1]=input.kProjection;
  prepareBindings[2]=input.vProjection; prepareBindings[3]=input.indexProjection;
  prepareBindings[7]=input.positions ? input.positions : state.indexPositions;
  prepareBindings[8]=bulk.queries; prepareBindings[10]=bulk.indexQueries;

  const ComputeDispatch *pool=nullptr,*selection=nullptr;
  for (const auto &dispatch:source) {
    if (dispatch.pipelineName.starts_with("flash_qsa_pool_rope_")) pool=&dispatch;
    if (dispatch.pipelineName == "flash_qsa_select_blocks") selection=&dispatch;
  }
  require(pool && selection, "Bulk QSA authoritative pool/selection disappeared");
  auto poolParams=parameters<FlashQSAParams>(*pool);
  poolParams.rows=rows; poolParams.begin=begin;
  poolParams.first_block=common.first_block; poolParams.new_blocks=common.new_blocks;
  auto selectionParams=parameters<FlashQSAParams>(*selection);
  selectionParams.rows=rows; selectionParams.begin=begin;
  selectionParams.first_block=common.first_block; selectionParams.new_blocks=common.new_blocks;
  auto selectionBindings=bindings(*selection);
  require(selectionBindings.size()==3,"Bulk QSA selection binding ABI changed");
  selectionBindings[1]=bulk.selectedBlocks;

  CommandGraph result;
  result.add(prepare.pipelineName,std::move(prepareBindings),prepareParams,{rows,30,1},
             prepare.threadsPerThreadgroup);
  result.add(pool->pipelineName,bindings(*pool),poolParams,{common.new_blocks,1,1},
             pool->threadsPerThreadgroup);
  // Every query has<=512 completed blocks. The shader's dense early return
  // fills chronological IDs without reading the bounded ordinary score sheet.
  result.add(selection->pipelineName,std::move(selectionBindings),selectionParams,{rows,1,1},
             selection->threadsPerThreadgroup);
  for (const auto &chunk:chunks) {
    const auto dispatches=chunk.dispatches();
    append(result,dispatches[dispatches.size()-2]);
    append(result,dispatches.back());
  }
  for (const auto &dispatch:result.dispatches()) append(graph,dispatch);
}
} // namespace splash::flash
