#include "video/h3/ops/PreviewDecoder.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/PreviewABI.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace h3::ops {
namespace {

std::string tileName(device::Tile t) {
  return std::to_string(t.tm) + "x" + std::to_string(t.tn) + "x" + std::to_string(t.nsg);
}

// decoded frame g -> output frame, or -1 for the 3 dropped frames of every 20
int32_t outputIndex(int32_t g) { return (g % 20) < 3 ? -1 : g - 3 * (g / 20) - 3; }

device::Tile parseTile(const std::string &name) {
  device::Tile t{};
  sscanf(name.c_str(), "%dx%dx%d", &t.tm, &t.tn, &t.nsg);
  return t;
}

} // namespace

Status PreviewDecoder::create(metal::MetalBackend &backend, std::unique_ptr<model::PreviewWeights> weights,
                              int32_t latentHeight, int32_t latentWidth, std::unique_ptr<PreviewDecoder> &out) {
  const std::string kTile256 = tileName(device::kPreviewTile256), kTile128 = tileName(device::kPreviewTile128),
                    kTile64 = tileName(device::kPreviewTile64), kTileOut = tileName(device::kPreviewTileOut);
  std::vector<std::string> kernels{"preview_input", "preview_copy", "preview_rgb_to_yuv420",
                                   "preview_conv_out_" + kTileOut};
  for (const std::string &t : {kTile256, kTile128, kTile64})
    for (const char *k : {"preview_conv_relu_", "preview_conv_memory_", "preview_conv_residual_"})
      kernels.push_back(k + t);
  for (const std::string &t : {kTile256, kTile128}) kernels.push_back("preview_tgrow_" + t);
  for (const std::string &t : {kTile128, kTile64}) kernels.push_back("preview_upsample_" + t);
  H3_TRY(backend.prepare(kernels));
  auto d = std::unique_ptr<PreviewDecoder>(new PreviewDecoder(backend));
  d->w_ = std::move(weights);
  d->LH_ = latentHeight;
  d->LW_ = latentWidth;
  d->chunk_ = device::kPreviewChunk;
  d->tile256_ = kTile256;
  d->tile128_ = kTile128;
  d->tile64_ = kTile64;
  d->tileOut_ = kTileOut;
  const int32_t NC = d->chunk_, C[3] = {256, 128, 64};
  H3_TRY(d->activation(latentHeight, latentWidth, H3_PREVIEW_CHANNELS_IN, NC, d->input_));
  for (int L = 0; L < 3; ++L) {
    const int32_t H = latentHeight << L, W = latentWidth << L, frames = L == 2 ? 2 * NC : NC;
    for (int j = 0; j < 4; ++j) H3_TRY(d->activation(H, W, C[L], frames, d->blocks_[L][j]));
    H3_TRY(d->activation(H, W, C[L], frames, d->y1_[L]));
    H3_TRY(d->activation(H, W, C[L], frames, d->y2_[L]));
    H3_TRY(d->activation(H, W, C[L], L == 0 ? frames : 2 * frames, d->grown_[L]));
  }
  H3_TRY(d->activation(latentHeight * 8, latentWidth * 8, 64, 4 * NC, d->upsampled_));
  const size_t frameBytes = static_cast<size_t>(latentHeight) * 16 * latentWidth * 16 * 3;
  for (int s = 0; s < 2; ++s) {
    H3_TRY(backend.allocate(static_cast<size_t>(NC) * latentHeight * latentWidth * 24 * 4, d->latents_[s],
                            "preview latents"));
    H3_TRY(backend.allocate(frameBytes * 4 * NC, d->frames_[s], "preview frames"));
    H3_TRY(backend.allocate(4 * NC * 4, d->frameMap_[s], "preview frame map"));
  }
  H3_TRY(backend.allocate(16, d->bad_, "preview non-finite counter"));
  H3_TRY(backend.allocate(16, d->null_, "preview null"));
  std::memset(d->bad_.contents, 0, 16);
  out = std::move(d);
  return {};
}

// Zero-initialised: the padding rows / columns of every frame are read as zeros by the convolutions.
Status PreviewDecoder::activation(int32_t H, int32_t W, int32_t C, int32_t frames, Activation &out) {
  out.H = H;
  out.W = W;
  out.Wp = W + 2;
  out.C = C;
  out.FS = ((H + 2) * out.Wp + 15) & ~15;
  const size_t rows = static_cast<size_t>(out.FS) * (frames + 1) + 2 * out.Wp + 256;  // slack for garbage-row reads
  H3_TRY(b_.allocate(rows * C * 2, out.buffer, "preview activation"));
  std::memset(out.buffer.contents, 0, out.buffer.length);
  return {};
}

void PreviewDecoder::conv(id<MTLComputeCommandEncoder> e, const Conv &c) {
  const device::Tile t = parseTile(c.tile);
  const Activation &src = *c.src;
  const int32_t mt = (src.H * src.Wp + t.tm - 1) / t.tm;
  PreviewConvParams p{};
  p.H = src.H;
  p.W = src.W;
  p.Wp = src.Wp;
  p.Mrows = mt * t.tm;
  p.Cin = src.C;
  p.Cout = c.cout;
  p.FSin = src.FS;
  p.FSout = c.dst ? c.dst->FS : 0;
  p.FSres = c.residual ? c.residual->FS : 0;
  p.WpOut = c.dst ? c.dst->Wp : 0;
  p.S = c.split;
  p.relu = c.relu;
  p.Hout = src.H * 2;
  p.Wout = src.W * 2;
  [e setComputePipelineState:b_.pipeline(std::string(c.kind) + "_" + c.tile)];
  [e setBuffer:src.buffer offset:src.base() atIndex:0];
  metal::bind(e, c.weight, 1);
  metal::bind(e, c.bias ? c.bias : metal::BufferView{c.weight.buffer, 0}, 2);
  if (c.dst) [e setBuffer:c.dst->buffer offset:c.dst->base() atIndex:3];
  else metal::bind(e, null_, 3);
  if (c.residual) [e setBuffer:c.residual->buffer offset:c.residual->base() atIndex:4];
  else metal::bind(e, null_, 4);
  metal::bind(e, c.slot >= 0 ? frames_[c.slot] : null_, 5);
  metal::bind(e, c.slot >= 0 ? frameMap_[c.slot] : null_, 6);
  metal::bindBytes(e, p, 7);
  metal::bind(e, bad_, 8);
  metal::dispatch(e, MTLSizeMake(c.cout / t.tn, mt, c.frames * (c.upsample ? 4 : 1)), MTLSizeMake(32 * t.nsg, 1, 1));
}

void PreviewDecoder::encodeChunk(id<MTLComputeCommandEncoder> e, int s, int32_t n, int32_t kept, bool yuv) {
  const std::string &kTile256 = tile256_, &kTile128 = tile128_, &kTile64 = tile64_, &kTileOut = tileOut_;
  {
    PreviewConvParams c{};
    c.H = LH_;
    c.W = LW_;
    c.Wp = input_.Wp;
    c.FSout = input_.FS;
    [e setComputePipelineState:b_.pipeline("preview_input")];
    metal::bind(e, latents_[s], 0);
    [e setBuffer:input_.buffer offset:input_.base() atIndex:1];
    metal::bindBytes(e, c, 2);
    [e dispatchThreads:MTLSizeMake(LW_, LH_, n) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  }
  const model::PreviewWeights &w = *w_;
  conv(e, {"preview_conv_relu", kTile256, &input_, n, w.inputConv, w.inputBias, &blocks_[0][0], nullptr, 256, 1, 0,
           false, -1});
  const int32_t C[3] = {256, 128, 64};
  const std::string *tiles[3] = {&kTile256, &kTile128, &kTile64};
  int32_t frames = n;
  for (int L = 0; L < 3; ++L) {
    for (int j = 0; j < 3; ++j) {  // MemBlock: relu(conv(cat(x, past))) -> relu(conv) -> relu(conv + x)
      const model::PreviewMemBlock &m = w.memBlocks[L * 3 + j];
      Activation &in = blocks_[L][j], &out = blocks_[L][j + 1];
      conv(e, {"preview_conv_memory", *tiles[L], &in, frames, m.conv0, m.bias0, &y1_[L], nullptr, C[L], 1, 0, false,
               -1});
      conv(e, {"preview_conv_relu", *tiles[L], &y1_[L], frames, m.conv2, m.bias2, &y2_[L], nullptr, C[L], 1, 0, false,
               -1});
      conv(e, {"preview_conv_residual", *tiles[L], &y2_[L], frames, m.conv4, m.bias4, &out, &in, C[L], 1, 0, false,
               -1});
    }
    const int32_t split = L == 0 ? 1 : 2, grownChannels = C[L] * split;
    const std::string &tg = grownChannels >= 128 ? (grownChannels == 256 ? kTile256 : kTile128) : kTile64;
    conv(e, {"preview_tgrow", tg, &blocks_[L][3], frames, w.tgrow[L], {}, &grown_[L], nullptr, grownChannels, split, 0,
             false, -1});
    frames *= split;
    const int32_t cout = L < 2 ? C[L + 1] : 64;
    Activation *dst = L < 2 ? &blocks_[L + 1][0] : &upsampled_;
    conv(e, {"preview_upsample", cout == 128 ? kTile128 : kTile64, &grown_[L], frames, w.upsample[L], {}, dst, nullptr,
             cout, 1, L == 2 ? 1 : 0, true, -1});
  }
  conv(e, {"preview_conv_out", kTileOut, &upsampled_, frames, w.outputConv, w.outputBias, nullptr, nullptr,
           H3_PREVIEW_OUT_CHANNELS, 1, 0, false, s});
  if (yuv) {  // kept frames of the chunk -> yuv420p staging
    const PreviewYuvParams d{LH_ * 16, LW_ * 16, kept, 0};
    [e setComputePipelineState:b_.pipeline("preview_rgb_to_yuv420")];
    metal::bind(e, frames_[s], 0);
    metal::bind(e, yuv_[s], 1);
    metal::bindBytes(e, d, 2);
    [e dispatchThreads:MTLSizeMake(d.W / 2, d.H / 2, kept) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  }
  // MemBlock state: the last input frame of every MemBlock -> its memory slot
  for (int L = 0; L < 3; ++L)
    for (int j = 0; j < 3; ++j) {
      const Activation &a = blocks_[L][j];
      const int32_t last = (L == 2 ? 2 * n : n) - 1;
      const uint32_t count = static_cast<uint32_t>(static_cast<size_t>(a.FS) * a.C * 2 / 16);
      [e setComputePipelineState:b_.pipeline("preview_copy")];
      [e setBuffer:a.buffer offset:a.base() + static_cast<size_t>(last) * a.FS * a.C * 2 atIndex:0];
      [e setBuffer:a.buffer offset:0 atIndex:1];
      metal::bindBytes(e, count, 2);
      [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

Status PreviewDecoder::decode(const float *latents, int32_t L, uint8_t *frames, uint8_t *yuv, PreviewStats &stats) {
  const auto t0 = std::chrono::steady_clock::now();
  stats = PreviewStats{};
  if (L % 5 != 2) return error(Code::Argument, "preview: latent frames must be 5 k + 2 (got %d)", L);
  const size_t frameBytes = static_cast<size_t>(LH_) * 16 * LW_ * 16 * 3, yuvBytes = frameBytes / 2;
  if (yuv && !yuv_[0])
    for (int s = 0; s < 2; ++s) H3_TRY(b_.allocate(yuvBytes * 4 * chunk_, yuv_[s], "preview yuv"));
  *static_cast<uint32_t *>(bad_.contents) = 0;
  for (int lv = 0; lv < 3; ++lv)  // first chunk: zero MemBlock state
    for (int j = 0; j < 3; ++j) std::memset(blocks_[lv][j].buffer.contents, 0, blocks_[lv][j].base());
  const size_t latentFrameBytes = static_cast<size_t>(LH_) * LW_ * 24;
  const int32_t chunks = (L + chunk_ - 1) / chunk_;
  id<MTLCommandBuffer> pending[2] = {nil, nil};
  int32_t first[2] = {0, 0}, kept[2] = {0, 0};
  double gpuS = 0, copyS = 0;
  Status status;
  auto drain = [&](int s) {
    if (!pending[s]) return;
    [pending[s] waitUntilCompleted];
    if (pending[s].status == MTLCommandBufferStatusError && status.ok())
      status = error(Code::Gpu, "preview command buffer: %s", pending[s].error.localizedDescription.UTF8String);
    gpuS += pending[s].GPUEndTime - pending[s].GPUStartTime;
    const auto c0 = std::chrono::steady_clock::now();
    std::memcpy(frames + static_cast<size_t>(first[s]) * frameBytes, frames_[s].contents, kept[s] * frameBytes);
    if (yuv) std::memcpy(yuv + static_cast<size_t>(first[s]) * yuvBytes, yuv_[s].contents, kept[s] * yuvBytes);
    copyS += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    pending[s] = nil;
  };
  for (int32_t k = 0; k < chunks; ++k) {
    const int s = k & 1;
    drain(s);
    const int32_t start = k * chunk_, n = std::min(chunk_, L - start);
    std::memcpy(latents_[s].contents, latents + static_cast<size_t>(start) * latentFrameBytes,
                static_cast<size_t>(n) * latentFrameBytes * 4);
    int32_t *map = static_cast<int32_t *>(frameMap_[s].contents);
    int32_t firstOut = -1, keptOut = 0;
    for (int32_t i = 0; i < 4 * n; ++i) {
      const int32_t og = outputIndex(4 * start + i);
      if (og >= 0 && firstOut < 0) firstOut = og;
      map[i] = og >= 0 ? og - firstOut : -1;
      keptOut += og >= 0;
    }
    first[s] = firstOut;
    kept[s] = keptOut;
    id<MTLCommandBuffer> cb = [b_.queue() commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    encodeChunk(e, s, n, keptOut, yuv != nullptr);
    [e endEncoding];
    [cb commit];
    pending[s] = cb;
  }
  drain(0);
  drain(1);
  H3_TRY(status);
  stats.wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  stats.gpuS = gpuS;
  stats.hostCopyS = copyS;
  stats.nonfinite = *static_cast<const uint32_t *>(bad_.contents);
  stats.chunks = chunks;
  stats.frames = 4 * L - 3 * ((4 * L + 19) / 20);
  stats.deviceMb = b_.allocatedMb();
  return {};
}

} // namespace h3::ops
