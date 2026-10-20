// C ABI of libh3.dylib (include/h3/h3.h) over h3::Engine.
#include "video/h3/h3.h"

#include "video/h3/engine/Engine.hpp"
#include "video/h3/media/Mp4Writer.hpp"

#include <cstring>
#include <memory>
#include <string>

struct h3_engine {
  std::unique_ptr<h3::Engine> engine;
  std::string error;
  std::unique_ptr<h3::model::Prepared> prepared;  // h3_prep outputs
  uint64_t seed = 0;
};

namespace {

h3::Request toRequest(const h3_request &r) {
  h3::Request q;
  q.variant = r.variant == H3_VARIANT_DEFAULT ? h3::Variant::Default : h3::Variant::Turbo;
  q.frames = r.frames;
  q.width = r.width;
  q.height = r.height;
  q.windowChunk = r.window_chunk;
  q.windowRadius = r.window_radius;
  q.qkfix = r.qkfix != 0;
  q.qkfixBlocks.assign(r.qkfix_blocks, r.qkfix_blocks + std::max(0, std::min(r.n_qkfix_blocks, H3_MAX_QKFIX_BLOCKS)));
  q.stream = r.stream != 0;
  q.streamSlots = r.stream_slots;
  q.streamImage = r.stream_image ? r.stream_image : "";
  q.vaePrecision = r.vae_precision == H3_VAE_F16 ? h3::VaePrecision::F16 : h3::VaePrecision::W8;
  return q;
}

h3_status report(h3_engine *e, const h3::Status &s) {
  if (s.ok()) return H3_OK;
  if (e) e->error = s.message();
  fprintf(stderr, "h3: %s\n", s.message().c_str());
  return static_cast<h3_status>(s.code());
}

void copyStats(const h3::ForwardStats &s, h3_forward_stats *out) {
  if (!out) return;
  *out = h3_forward_stats{s.wallMs,        s.embedMs,         s.blocksSpanMs, s.blocksGpuMs,    s.finalMs,
                          s.linearMs,      s.attentionMs,     s.otherMs,      s.branchMs,       s.gpuBusyMs,
                          s.gpuIdleMs,     s.gpuGapMaxMs,     s.gpuGaps,      s.commandBuffers, s.streamWaitMs,
                          s.streamWaitMaxMs, s.streamReadGb, s.streamReadS, s.streamReadMaxMs, s.streamWaits,
                          s.streamBlocks,  s.deviceMb};
}

} // namespace

extern "C" {

h3_request h3_default_request(void) {
  h3_request r{};
  r.variant = H3_VARIANT_TURBO;
  r.frames = 73;
  r.width = 1344;
  r.height = 768;
  r.window_chunk = 5;
  r.window_radius = 1;
  r.qkfix = 1;
  r.qkfix_blocks[0] = 45;
  r.qkfix_blocks[1] = 48;
  r.qkfix_blocks[2] = 49;
  r.n_qkfix_blocks = 3;
  r.stream = 0;
  r.stream_slots = 8;
  r.stream_image = nullptr;
  r.vae_precision = H3_VAE_W8;
  return r;
}

h3_status h3_geometry_of(const h3_request *request, h3_geometry *out) {
  if (!request || !out) return H3_ERROR_ARGUMENT;
  h3::model::VideoGeometry g;
  h3::model::VaePlan plan;
  h3::Status s = h3::model::VideoGeometry::of(request->frames, request->width, request->height, g);
  if (s.ok()) s = h3::model::planVaeDecode(g, plan);
  if (!s.ok()) return report(nullptr, s);
  *out = h3_geometry{g.frames,       g.width,       g.height,           g.latentFrames,
                     g.latentHeight, g.latentWidth, g.videoRows(),      g.audioRows,
                     plan.outputFrames(), h3::model::previewFrames(g.latentFrames),
                     static_cast<int32_t>(h3::ops::AudioDecoder::samples(g.audioRows / 2))};
  return H3_OK;
}

int64_t h3_window_table(const h3_request *request, int32_t text_rows, int32_t *words, int64_t capacity) {
  if (!request) return -1;
  h3::WindowTable table;
  if (report(nullptr, h3::Engine::windowTable(toRequest(*request), text_rows, table)) != H3_OK) return -1;
  const int64_t n = static_cast<int64_t>(table.words().size());
  if (words && capacity >= n) std::memcpy(words, table.words().data(), static_cast<size_t>(n) * 4);
  return n;
}

h3_status h3_create(const h3_config *config, const h3_request *request, h3_engine **out) {
  if (!config || !request || !out || !config->metallib || !config->weights) return H3_ERROR_ARGUMENT;
  *out = nullptr;
  auto e = std::make_unique<h3_engine>();
  const h3_status s = report(e.get(), h3::Engine::create({config->metallib, config->weights}, toRequest(*request),
                                                         e->engine));
  if (s != H3_OK) return s;
  *out = e.release();
  return H3_OK;
}

void h3_destroy(h3_engine *engine) { delete engine; }

const char *h3_error(const h3_engine *engine) { return engine ? engine->error.c_str() : ""; }

h3_status h3_prep(h3_engine *e, const h3_prep_inputs *in, h3_prepared *out) {
  if (!e || !in || !out) return H3_ERROR_ARGUMENT;
  auto p = std::make_unique<h3::model::Prepared>();
  const h3_status r = report(e, e->engine->prep({in->hidden, in->text_rows, in->steps, in->seed}, *p));
  if (r != H3_OK) return r;
  e->prepared = std::move(p);
  e->seed = in->seed;
  const h3::model::Prepared &q = *e->prepared;
  *out = h3_prepared{q.textRows,        q.rows,           q.steps,          static_cast<int32_t>(q.table.size()),
                     q.tags.data(),     q.rope.data(),    q.pos.data(),     q.text.data(),
                     q.adaBlocks.data(), q.adaFinal.data(), q.vT.data(),      q.aT.data(),
                     q.vSigmas.data(),  q.aSigmas.data(), q.table.data(),   q.tv.data(),
                     q.ta.data(),       q.videoNoise.data(), q.audioNoise.data(), q.totalS};
  return H3_OK;
}

h3_status h3_prep_write(h3_engine *e, const char *dir) {
  if (!e || !dir) return H3_ERROR_ARGUMENT;
  if (!e->prepared) return report(e, h3::error(h3::Code::State, "h3_prep_write before h3_prep"));
  return report(e, h3::model::Prep::write(dir, *e->prepared, e->engine->request().variant, e->seed, "h3_prep"));
}

void h3_prep_release(h3_engine *e) {
  if (e) e->prepared.reset();
}

h3_status h3_dit_load(h3_engine *e, const h3_dit_inputs *in) {
  if (!e || !in) return H3_ERROR_ARGUMENT;
  return report(e, e->engine->loadDit({in->text_rows, in->tags, in->rope, in->text}));
}

h3_status h3_dit_info_of(const h3_engine *e, h3_dit_info *out) {
  if (!e || !out) return H3_ERROR_ARGUMENT;
  h3::DitInfo info;
  const h3::Status s = e->engine->ditInfo(info);
  if (!s.ok()) return static_cast<h3_status>(s.code());
  const h3::model::SequenceLayout &lay = *e->engine->layout();
  *out = h3_dit_info{info.rows,          info.paddedRows,  lay.text,          lay.audio,
                     lay.frames * lay.tokensPerFrame,     info.windowItems, info.chunkedGroups, info.partialSlots,
                     info.density,       info.streamSlots, info.weightsMb,   info.activationsMb, info.deviceMb};
  return H3_OK;
}

h3_status h3_dit_forward(h3_engine *e, const uint16_t *ada_blocks, const uint16_t *ada_final, const float *video_in,
                         const float *audio_in, float *video_out, float *audio_out, h3_forward_stats *stats) {
  if (!e || !ada_blocks || !ada_final || !video_in || !video_out) return H3_ERROR_ARGUMENT;
  h3::ForwardStats s;
  const h3_status r = report(e, e->engine->forward(ada_blocks, ada_final, video_in, audio_in, video_out, audio_out, s));
  copyStats(s, stats);
  return r;
}

h3_status h3_denoise(h3_engine *e, const h3_schedule *schedule, const float *video_noise, const float *audio_noise,
                     float *video_out, float *audio_out, h3_step_callback callback, void *user) {
  if (!e || !schedule || !video_noise || !video_out) return H3_ERROR_ARGUMENT;
  h3::Schedule s{schedule->steps,        schedule->ada_blocks,   schedule->ada_final,   schedule->video_t,
                 schedule->audio_t,      schedule->video_sigmas, schedule->audio_sigmas};
  h3::StepCallback onStep;
  if (callback)
    onStep = [&](int32_t step, const h3::ForwardStats &stats, const float *video, const float *audio) {
      h3_forward_stats c;
      copyStats(stats, &c);
      callback(user, step, &c, video, audio);
    };
  return report(e, e->engine->denoise(s, video_noise, audio_noise, video_out, audio_out, onStep));
}

void h3_dit_release(h3_engine *e) {
  if (e) e->engine->releaseDit();
}

h3_status h3_decode(h3_engine *e, const float *video_latents, uint8_t *frames, h3_decode_stats *stats) {
  if (!e || !video_latents || !frames) return H3_ERROR_ARGUMENT;
  h3::DecodeStats s;
  const h3_status r = report(e, e->engine->decode(video_latents, frames, s));
  if (stats) *stats = h3_decode_stats{s.wallS, s.gpuS, s.hostCopyS, s.uploadS, s.gpuSpanS, s.nonfinite, s.deviceMb};
  return r;
}

h3_status h3_preview(h3_engine *e, const float *video_latents, uint8_t *frames, uint8_t *yuv420,
                     h3_preview_stats *stats) {
  if (!e || !video_latents || !frames) return H3_ERROR_ARGUMENT;
  h3::PreviewStats s;
  const h3_status r = report(e, e->engine->preview(video_latents, frames, yuv420, s));
  if (stats) *stats = h3_preview_stats{s.wallS, s.gpuS, s.hostCopyS, s.nonfinite, s.chunks, s.frames, s.deviceMb};
  return r;
}

h3_status h3_decode_audio(h3_engine *e, const float *audio_latents, float *wav, h3_audio_stats *stats) {
  if (!e || !audio_latents || !wav) return H3_ERROR_ARGUMENT;
  h3::AudioStats s;
  const h3_status r = report(e, e->engine->decodeAudio(audio_latents, wav, s));
  if (stats) *stats = h3_audio_stats{s.wallS, s.gpuS, s.samples, s.nonfinite, s.deviceMb};
  return r;
}

h3_status h3_write_mp4(const char *path, const uint8_t *frames, int32_t frame_count, int32_t width, int32_t height,
                       const float *wav, int64_t samples, h3_mp4_stats *stats) {
  if (!path || !frames) return H3_ERROR_ARGUMENT;
  h3::media::Mp4Stats s;
  const h3_status r = report(nullptr, h3::media::writeMp4(path, frames, frame_count, width, height, wav, samples, s));
  if (stats)
    *stats = h3_mp4_stats{s.wallS, s.frames, s.audioSamples, s.videoSeconds, s.audioSeconds, s.bytes, s.videoBitrate};
  return r;
}

} // extern "C"
