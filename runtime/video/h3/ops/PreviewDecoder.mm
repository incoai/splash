#include "video/h3/ops/PreviewDecoder.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/PreviewABI.h"

#include "AwakeClock.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

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

PreviewDecoder::PreviewDecoder(metal::MetalBackend &backend, std::unique_ptr<model::PreviewWeights> weights,
                               int32_t latentHeight, int32_t latentWidth)
    : b_(backend), w_(std::move(weights)), LH_(latentHeight), LW_(latentWidth), chunk_(device::kPreviewChunk),
      tile256_(tileName(device::kPreviewTile256)), tile128_(tileName(device::kPreviewTile128)),
      tile64_(tileName(device::kPreviewTile64)), tileOut_(tileName(device::kPreviewTileOut)) {
  const int32_t NC = chunk_, C[3] = {256, 128, 64};
  input_ = activation(latentHeight, latentWidth, H3_PREVIEW_CHANNELS_IN, NC);
  for (int L = 0; L < 3; ++L) {
    const int32_t H = latentHeight << L, W = latentWidth << L, frames = L == 2 ? 2 * NC : NC;
    for (int j = 0; j < 4; ++j) blocks_[L][j] = activation(H, W, C[L], frames);
    y1_[L] = activation(H, W, C[L], frames);
    y2_[L] = activation(H, W, C[L], frames);
    grown_[L] = activation(H, W, C[L], L == 0 ? frames : 2 * frames);
  }
  upsampled_ = activation(latentHeight * 8, latentWidth * 8, 64, 4 * NC);
  const size_t frameBytes = static_cast<size_t>(latentHeight) * 16 * latentWidth * 16 * 3;
  for (int s = 0; s < 2; ++s) {
    latents_[s] = allocate(b_, static_cast<size_t>(NC) * latentHeight * latentWidth * 24 * 4, "preview latents");
    frames_[s] = allocate(b_, frameBytes * 4 * NC, "preview frames");
    frameMap_[s] = allocate(b_, 4 * NC * 4, "preview frame map");
  }
  bad_ = allocate(b_, 16, "preview non-finite counter");
  null_ = allocate(b_, 16, "preview null");
  std::memset(bad_.contents(), 0, 16);
}

// Zero-initialised: the padding rows / columns of every frame are read as zeros by the convolutions.
PreviewDecoder::Activation PreviewDecoder::activation(int32_t H, int32_t W, int32_t C, int32_t frames) {
  Activation out;
  out.H = H;
  out.W = W;
  out.Wp = W + 2;
  out.C = C;
  out.FS = ((H + 2) * out.Wp + 15) & ~15;
  const size_t rows = static_cast<size_t>(out.FS) * (frames + 1) + 2 * out.Wp + 256;  // slack for garbage-row reads
  out.buffer = allocate(b_, rows * C * 2, "preview activation");
  std::memset(out.buffer.contents(), 0, out.buffer.sizeBytes());
  return out;
}

void PreviewDecoder::conv(Encoder &e, const Conv &c) {
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
  e.pipeline(std::string(c.kind) + "_" + c.tile);
  e.bind({src.buffer, src.base()}, 0);
  e.bind(c.weight, 1);
  e.bind(c.bias ? c.bias : BufferView{c.weight.buffer, 0}, 2);
  if (c.dst) e.bind({c.dst->buffer, c.dst->base()}, 3);
  else e.bind(null_, 3);
  if (c.residual) e.bind({c.residual->buffer, c.residual->base()}, 4);
  else e.bind(null_, 4);
  e.bind(c.slot >= 0 ? frames_[c.slot] : null_, 5);
  e.bind(c.slot >= 0 ? frameMap_[c.slot] : null_, 6);
  e.bytes(p, 7);
  e.bind(bad_, 8);
  e.dispatch(size3(c.cout / t.tn, mt, c.frames * (c.upsample ? 4 : 1)), size3(32 * t.nsg));
}

void PreviewDecoder::encodeChunk(Encoder &e, int s, int32_t n, int32_t kept, bool yuv) {
  const std::string &kTile256 = tile256_, &kTile128 = tile128_, &kTile64 = tile64_, &kTileOut = tileOut_;
  {
    PreviewConvParams c{};
    c.H = LH_;
    c.W = LW_;
    c.Wp = input_.Wp;
    c.FSout = input_.FS;
    e.pipeline("preview_input");
    e.bind(latents_[s], 0);
    e.bind({input_.buffer, input_.base()}, 1);
    e.bytes(c, 2);
    e.dispatchThreads(size3(LW_, LH_, n), size3(32, 4));
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
    e.pipeline("preview_rgb_to_yuv420");
    e.bind(frames_[s], 0);
    e.bind(yuv_[s], 1);
    e.bytes(d, 2);
    e.dispatchThreads(size3(d.W / 2, d.H / 2, kept), size3(32, 8));
  }
  // MemBlock state: the last input frame of every MemBlock -> its memory slot
  for (int L = 0; L < 3; ++L)
    for (int j = 0; j < 3; ++j) {
      const Activation &a = blocks_[L][j];
      const int32_t last = (L == 2 ? 2 * n : n) - 1;
      const uint32_t count = static_cast<uint32_t>(static_cast<size_t>(a.FS) * a.C * 2 / 16);
      e.pipeline("preview_copy");
      e.bind({a.buffer, a.base() + static_cast<size_t>(last) * a.FS * a.C * 2}, 0);
      e.bind({a.buffer, 0}, 1);
      e.bytes(count, 2);
      e.dispatchThreads(size3(count), size3(256));
    }
}

void PreviewDecoder::decode(const float *latents, int32_t L, uint8_t *frames, uint8_t *yuv, PreviewStats &stats) {
  const auto t0 = splash::AwakeClock::now();
  stats = PreviewStats{};
  if (L % 5 != 2) throw std::invalid_argument("preview: latent frames must be 5 k + 2 (got " + std::to_string(L) + ")");
  const size_t frameBytes = static_cast<size_t>(LH_) * 16 * LW_ * 16 * 3, yuvBytes = frameBytes / 2;
  if (yuv && !yuv_[0])
    for (int s = 0; s < 2; ++s) yuv_[s] = allocate(b_, yuvBytes * 4 * chunk_, "preview yuv");
  *static_cast<uint32_t *>(bad_.contents()) = 0;
  for (int lv = 0; lv < 3; ++lv)  // first chunk: zero MemBlock state
    for (int j = 0; j < 3; ++j) std::memset(blocks_[lv][j].buffer.contents(), 0, blocks_[lv][j].base());
  const size_t latentFrameBytes = static_cast<size_t>(LH_) * LW_ * 24;
  const int32_t chunks = (L + chunk_ - 1) / chunk_;
  int32_t first[2] = {0, 0}, kept[2] = {0, 0};
  double copyS = 0;
  auto copyOut = [&](int s) {
    const auto c0 = splash::AwakeClock::now();
    std::memcpy(frames + static_cast<size_t>(first[s]) * frameBytes, frames_[s].contents(), kept[s] * frameBytes);
    if (yuv) std::memcpy(yuv + static_cast<size_t>(first[s]) * yuvBytes, yuv_[s].contents(), kept[s] * yuvBytes);
    copyS += splash::millisecondsSince(c0) / 1e3;
  };
  metal::CommandTicket previous;
  for (int32_t k = 0; k < chunks; ++k) {
    const int s = k & 1;
    const int32_t start = k * chunk_, n = std::min(chunk_, L - start);
    int32_t firstOut = -1, keptOut = 0;
    std::vector<int32_t> map(4 * n);
    for (int32_t i = 0; i < 4 * n; ++i) {
      const int32_t og = outputIndex(4 * start + i);
      if (og >= 0 && firstOut < 0) firstOut = og;
      map[i] = og >= 0 ? og - firstOut : -1;
      keptOut += og >= 0;
    }
    Encoder e(b_);
    encodeChunk(e, s, n, keptOut, yuv != nullptr);
    // one command in flight: chunk k starts once chunk k - 1 completed, whose frames are copied while k runs
    if (k > 0) stats.gpuS += wait(previous);
    std::memcpy(latents_[s].contents(), latents + static_cast<size_t>(start) * latentFrameBytes,
                static_cast<size_t>(n) * latentFrameBytes * 4);
    std::memcpy(frameMap_[s].contents(), map.data(), map.size() * sizeof(int32_t));
    first[s] = firstOut;
    kept[s] = keptOut;
    previous = submit(b_, e);
    if (k > 0) copyOut(s ^ 1);
  }
  stats.gpuS += wait(previous);
  copyOut((chunks - 1) & 1);
  stats.wallS = splash::millisecondsSince(t0) / 1e3;
  stats.hostCopyS = copyS;
  stats.nonfinite = *static_cast<const uint32_t *>(bad_.contents());
  stats.chunks = chunks;
  stats.frames = 4 * L - 3 * ((4 * L + 19) / 20);
  stats.deviceMb = deviceMb(b_);
}

} // namespace h3::ops
