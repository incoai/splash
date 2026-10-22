#include "video/h3/ops/DitIo.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/DitIoABI.h"

namespace h3::ops {
namespace {

DitIoParams params(const model::SequenceLayout &lay) {
  return {lay.text, lay.audio, H3_DIT_VIDEO_CHANNELS, H3_DIT_AUDIO_CHANNELS, H3_DIT_HIDDEN, device::kNormEps, lay.cond};
}

} // namespace

void embed(Encoder &e, const model::SequenceLayout &lay, const model::DitIoWeights &w, const IoBuffers &x) {
  const DitIoParams p = params(lay);
  e.pipeline("dit_io_embed");
  e.bind(x.residual, 0);
  e.bind(x.text, 1);
  e.bind(x.audioIn, 2);
  e.bind(x.videoIn, 3);
  e.bind(w.audioProj, 4);
  e.bind(w.audioProjBias, 5);
  e.bind(w.videoProj, 6);
  e.bind(w.videoProjBias, 7);
  e.bytes(p, 8);
  e.dispatch(size3(H3_DIT_HIDDEN / 256, lay.rows(), 1), size3(256, 1, 1));
}

void finalLayer(Encoder &e, const model::SequenceLayout &lay, const model::DitIoWeights &w, const IoBuffers &x) {
  const DitIoParams p = params(lay);
  e.pipeline("dit_io_final");
  e.bind(x.residual, 0);
  e.bind(x.tags, 1);
  e.bind(w.finalNorm, 2);
  e.bind(x.adaFinal, 3);
  e.bind(w.videoOut, 4);
  e.bind(w.videoOutBias, 5);
  e.bind(w.audioOut, 6);
  e.bind(w.audioOutBias, 7);
  e.bind(x.videoOut, 8);
  e.bind(x.audioOut, 9);
  e.bytes(p, 10);
  e.dispatch(size3(lay.audio + lay.frames * lay.tokensPerFrame, 1, 1), size3(256, 1, 1));
}

} // namespace h3::ops
