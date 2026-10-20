#include "video/h3/ops/DitIo.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/DitIoABI.h"

namespace h3::ops {
namespace {

DitIoParams params(const model::SequenceLayout &lay) {
  return {lay.text, lay.audio, H3_DIT_VIDEO_CHANNELS, H3_DIT_AUDIO_CHANNELS, H3_DIT_HIDDEN, device::kNormEps};
}

} // namespace

std::vector<std::string> ioKernels() { return {"dit_io_embed", "dit_io_final"}; }

void embed(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const model::SequenceLayout &lay,
           const model::DitIoWeights &w, const IoBuffers &x) {
  const DitIoParams p = params(lay);
  [e setComputePipelineState:b.pipeline("dit_io_embed")];
  metal::bind(e, x.residual, 0);
  metal::bind(e, x.text, 1);
  metal::bind(e, x.audioIn, 2);
  metal::bind(e, x.videoIn, 3);
  metal::bind(e, w.audioProj, 4);
  metal::bind(e, w.audioProjBias, 5);
  metal::bind(e, w.videoProj, 6);
  metal::bind(e, w.videoProjBias, 7);
  metal::bindBytes(e, p, 8);
  metal::dispatch(e, MTLSizeMake(H3_DIT_HIDDEN / 256, lay.rows(), 1), MTLSizeMake(256, 1, 1));
}

void finalLayer(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const model::SequenceLayout &lay,
                const model::DitIoWeights &w, const IoBuffers &x) {
  const DitIoParams p = params(lay);
  [e setComputePipelineState:b.pipeline("dit_io_final")];
  metal::bind(e, x.residual, 0);
  metal::bind(e, x.tags, 1);
  metal::bind(e, w.finalNorm, 2);
  metal::bind(e, x.adaFinal, 3);
  metal::bind(e, w.videoOut, 4);
  metal::bind(e, w.videoOutBias, 5);
  metal::bind(e, w.audioOut, 6);
  metal::bind(e, w.audioOutBias, 7);
  metal::bind(e, x.videoOut, 8);
  metal::bind(e, x.audioOut, 9);
  metal::bindBytes(e, p, 10);
  metal::dispatch(e, MTLSizeMake(lay.audio + lay.frames * lay.tokensPerFrame, 1, 1), MTLSizeMake(256, 1, 1));
}

} // namespace h3::ops
