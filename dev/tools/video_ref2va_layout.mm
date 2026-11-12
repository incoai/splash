// Ref2VA row layout against a Pika H3 Flex dump directory: the engine's row classes (runtime/video/ops/RowLayout.hpp)
// and Prep::referenceRows on the dump's conditioning, compared with its layout and first-forward inputs. CPU only: no
// weights, no device.
//
//   video-ref2va-layout DUMP_DIRECTORY [--forward MODEL_DIRECTORY --metallib LIB [--out DIRECTORY]]
//   video-ref2va-layout DUMP_DIRECTORY --generate MODEL_DIRECTORY --metallib LIB --out DIRECTORY
//
// Reads run.json (presentation.geometry, step0.unique_timesteps), conditioning.npz (text_token_tags and the reference
// latents refN_<kind>_{video,audio}_latents: video [1][F][24][h][w], audio [1][T][64]), layout.npz, initial_noise.npz
// and step0.npz. Prints the row classes, the attention metadata, every layout array exact or not (position_ids
// float64 bit for bit, token_tags, timestep_indices at step 0, the three index arrays, the condition-row counts), the
// step-0 input rows (target rows from initial_noise.npz bit for bit, audio condition rows bit for bit; the video
// condition rows' augmentation noise comes from Pika's CUDA generator, so they are compared clean, with the implied
// noise's statistics) and the activation plan the sequence needs. Exit 0 when every layout array is exact.
//
// --forward (GPU): one step-0 forward of the dump through the engine: model::Prep::run of a pika-6 assembly on the
// dump's prompt_embeds, text tags and reference latents, the dumped video / audio condition rows (step0.npz, so the
// augmentation noise is Pika's) and target rows (initial_noise.npz) in place of the engine's draws, then one DiT
// forward; its target velocities against step0_velocity.npz (relRMS = ||ours - ref|| / ||ref||, h3quality.py's), the
// times and the process's peak footprint; --out writes step0_velocity.npz (video, audio).
// --generate (GPU): the whole clip of the dump on the step-0 forward's inputs (the dumped condition rows, which Pika
// augments once per request and keeps for every step, and the dumped initial noise): the variant's sampler
// (DenoisingStage; pika-6: renoise, its fresh noise the engine's own draws of seed 42 + 1, not Pika's CUDA stream), then
// DecodingStage, AudioDecodingStage and MuxStage. --out gets out.mp4, final.npz (video [rows][96], audio [rows][32]),
// step0_velocity.npz and run.json (phases, per-forward wall, peak footprint, relRMS vs the dump's step0_velocity.npz
// and final.npz).
#include "Numpy.hpp"
#include "metal/abi/video/Layout.h"
#include "metal/abi/video/VsaABI.h"
#include "AwakeClock.hpp"
#include "video/model/Adapter.hpp"
#include "video/model/DitWeights.hpp"
#include "video/model/FastH3.hpp"
#include "video/model/MemoryPlan.hpp"
#include "video/model/Prep.hpp"
#include "video/ops/VsaTable.hpp"
#include "video/stages/AudioDecodingStage.hpp"
#include "video/stages/DecodingStage.hpp"
#include "video/stages/DenoisingStage.hpp"
#include "video/stages/MuxStage.hpp"

#import <Foundation/Foundation.h>
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <format>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace {

using namespace splash::video;
using splash::dev::NpyArray;

const NpyArray &need(const std::map<std::string, NpyArray> &npz, const std::string &key, const char *dtype) {
  const auto it = npz.find(key);
  if (it == npz.end()) throw std::runtime_error("missing array " + key);
  if (it->second.dtype != dtype) throw std::runtime_error(key + ": dtype " + it->second.dtype + ", want " + dtype);
  return it->second;
}

NSDictionary *readJson(const std::string &path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  if (!data) throw std::runtime_error("cannot read " + path);
  NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  if (![json isKindOfClass:[NSDictionary class]]) throw std::runtime_error(path + ": not a JSON object");
  return json;
}

int32_t jsonInt(NSDictionary *d, const char *key) {
  id v = d[[NSString stringWithUTF8String:key]];
  if (![v isKindOfClass:[NSNumber class]]) throw std::runtime_error(std::string("run.json: no ") + key);
  return [v intValue];
}

double relRms(const float *a, const float *b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = double(a[i]) - double(b[i]);
    num += d * d;
    den += double(b[i]) * double(b[i]);
  }
  return std::sqrt(num / double(n)) / std::max(std::sqrt(den / double(n)), 1e-30);
}

bool failed = false;
void report(const char *what, const char *status, const std::string &detail) {
  std::printf("  %-28s %-10s %s\n", what, status, detail.c_str());
}
// an array that must match bit for bit
void report(const char *what, bool exact, const std::string &detail) {
  report(what, exact ? "exact" : "NOT EXACT", detail);
  failed |= !exact;
}
// a layout array: dtype, shape and bytes
template <class T>
bool sameInt64(const NpyArray &want, const std::vector<T> &ours, const char *what) {
  bool exact = want.count() == ours.size();
  size_t first = ours.size();
  for (size_t i = 0; exact && i < ours.size(); ++i)
    if (want.as<int64_t>()[i] != int64_t(ours[i])) {
      exact = false;
      first = i;
    }
  report(what, exact,
         exact ? (ours.size() == 1 ? "= " + std::to_string(int64_t(ours[0])) : std::to_string(ours.size()) + " values")
               : "ours " + std::to_string(ours.size()) + " vs " + std::to_string(want.count()) +
                     (first < ours.size() ? ", first difference at " + std::to_string(first) : ""));
  return exact;
}

struct Options {
  std::string model, metallib, out;
  bool generate = false;
};

double peakFootprintGb() {
  rusage_info_v4 ru{};
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&ru)) != 0) return -1;
  return double(ru.ri_lifetime_max_phys_footprint) / 1e9;
}

// The stages' events of one clip: the denoising's per-forward completion times.
struct ClipEvents final : StageEvents {
  splash::AwakeClock::time_point start = splash::AwakeClock::now();
  std::vector<double> done;  // seconds since start at the end of each forward
  void progress(uint32_t) override { done.push_back(splash::millisecondsSince(start) / 1e3); }
  void output(splash::protocol::VideoOutputKind, uint32_t, std::vector<uint8_t>) override {}
  [[nodiscard]] bool cancelled() const override { return false; }
};

NSNumber *num(double v) { return [NSNumber numberWithDouble:std::round(v * 1e4) / 1e4]; }

void writeRunJson(const std::string &path, NSDictionary *d) {
  NSData *data = [NSJSONSerialization dataWithJSONObject:d
                                                 options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                   error:nil];
  if (!data || ![data writeToFile:[NSString stringWithUTF8String:path.c_str()] atomically:YES])
    throw std::runtime_error("cannot write " + path);
}

// --generate: the whole clip from the prepared request (its condition rows and initial noise Pika's), through the
// runtime's stages.
void generateClip(splash::metal::MetalBackend &backend, const Options &opt, const ops::VideoGeometry &g,
                  model::Prepared &&fp, const std::string &dir, double prepS) {
  const auto tStart = splash::AwakeClock::now();
  std::filesystem::create_directories(opt.out);
  VideoBatch batch;
  batch.requestId = 1;
  batch.options.dit = "pika-6";
  batch.options.frames = g.frames;
  batch.options.width = g.width;
  batch.options.height = g.height;
  batch.geometry = g;
  batch.seed = 42;
  batch.audio = true;
  const int32_t steps = fp.steps;
  NSMutableArray *vs = [NSMutableArray array], *as = [NSMutableArray array];
  for (float s : fp.vSigmas) [vs addObject:num(s)];
  for (float s : fp.aSigmas) [as addObject:num(s)];
  std::printf("generate: %d steps (sampler %s), video sigmas", steps, fp.renoise ? "renoise" : "euler");
  for (float s : fp.vSigmas) std::printf(" %.6g", s);
  std::printf("; audio sigmas");
  for (float s : fp.aSigmas) std::printf(" %.6g", s);
  std::printf("\n");
  std::fflush(stdout);
  batch.prepared = std::move(fp);
  ClipEvents events;
  NSMutableDictionary *run = [NSMutableDictionary dictionary];
  run[@"dump"] = [NSString stringWithUTF8String:dir.c_str()];
  run[@"asm"] = [NSString stringWithUTF8String:opt.model.c_str()];
  run[@"variant"] = @"pika-6";
  run[@"recipe"] = @"pika-ref2va-6";
  run[@"task"] = @"ref2va";
  run[@"size"] = [NSString stringWithFormat:@"%dx%d", g.width, g.height];
  run[@"seconds"] = @5;
  run[@"seed"] = @42;
  run[@"video_sigmas"] = vs;
  run[@"audio_sigmas"] = as;
  run[@"inputs_used"] = @{
    @"condition_video_rows" : @"dump step0.npz video_in (Pika's augmentation, fixed for every step as Pika caches it)",
    @"condition_audio_rows" : @"engine Prep::referenceRows (bit-equal to step0.npz audio_in)",
    @"target_initial_noise" : @"dump initial_noise.npz",
    @"renoise_noise" : @"engine MLX draws, seed 42 + 1 (Pika: CUDA torch.Generator(seed + 1); not in the dump)",
    @"prompt_embeds" : @"dump conditioning.npz",
    @"reference_latents" : @"dump conditioning.npz"
  };
  const auto td = splash::AwakeClock::now();
  DenoisingStage(backend, opt.model).forward(batch, events);
  const double denoiseS = splash::millisecondsSince(td) / 1e3;
  NSMutableArray *fwd = [NSMutableArray array], *evts = [NSMutableArray arrayWithObject:@[ @0, @0.0 ]];
  double prev = (std::chrono::duration<double>(td - events.start)).count();
  for (size_t i = 0; i < events.done.size(); ++i) {
    [fwd addObject:num(events.done[i] - prev)];
    [evts addObject:@[ @(i + 1), num(events.done[i] - (std::chrono::duration<double>(td - events.start)).count()) ]];
    std::printf("  forward %zu: %.1f s\n", i + 1, events.done[i] - prev);
    prev = events.done[i];
  }
  run[@"forward_wall_s"] = fwd;
  run[@"denoise_events"] = evts;
  // step 0 against the dump (the same inputs: a check of the sampler's first forward); none without a dump
  NSMutableDictionary *rel = [NSMutableDictionary dictionary];
  if (!batch.step0Video.empty()) {
    const NpyArray v = NpyArray::of("<f4", {int64_t(batch.step0Video.size() / 96), 96}, batch.step0Video.data(),
                                    batch.step0Video.size() * 4);
    const NpyArray a = NpyArray::of("<f4", {int64_t(batch.step0Audio.size() / 32), 32}, batch.step0Audio.data(),
                                    batch.step0Audio.size() * 4);
    splash::dev::writeNpz(opt.out + "/step0_velocity.npz", {{"video", &v}, {"audio", &a}});
  }
  if (!dir.empty()) {
    const auto vz = splash::dev::readNpz(dir + "/step0_velocity.npz");
    const NpyArray &rv = need(vz, "video", "<f4"), &ra = need(vz, "audio", "<f4");
    if (batch.step0Video.size() == rv.count() && batch.step0Audio.size() == ra.count()) {
      const double v0 = relRms(batch.step0Video.data(), rv.as<float>(), rv.count());
      const double a0 = relRms(batch.step0Audio.data(), ra.as<float>(), ra.count());
      rel[@"step0_velocity_video"] = num(v0);
      rel[@"step0_velocity_audio"] = num(a0);
      std::printf("  step-0 velocity relRMS vs dump: video %.4f, audio %.4f\n", v0, a0);
    }
  }
  NSMutableDictionary *phases = [NSMutableDictionary dictionary];
  phases[@"prepare"] = num(prepS);
  phases[@"denoise"] = num(denoiseS);
  run[@"phases_s"] = phases;
  run[@"relrms"] = rel;
  run[@"peak_footprint_gb_after_denoise"] = num(peakFootprintGb());
  {
    const NpyArray v = NpyArray::of("<f4", {int64_t(batch.videoLatents.size() / 96), 96}, batch.videoLatents.data(),
                                    batch.videoLatents.size() * 4);
    const NpyArray a = NpyArray::of("<f4", {int64_t(batch.audioLatents.size() / 32), 32}, batch.audioLatents.data(),
                                    batch.audioLatents.size() * 4);
    splash::dev::writeNpz(opt.out + "/final.npz", {{"video", &v}, {"audio", &a}});
    const auto fz = dir.empty() ? std::map<std::string, NpyArray>{} : splash::dev::readNpz(dir + "/final.npz");
    const NpyArray *fvp = fz.count("video") ? &need(fz, "video", "<f4") : nullptr;
    const NpyArray *fap = fz.count("audio") ? &need(fz, "audio", "<f4") : nullptr;
    if (fvp && fap && fvp->count() == batch.videoLatents.size() && fap->count() == batch.audioLatents.size()) {
      const NpyArray &fv = *fvp, &fa = *fap;
      const double fvr = relRms(batch.videoLatents.data(), fv.as<float>(), fv.count());
      const double far = relRms(batch.audioLatents.data(), fa.as<float>(), fa.count());
      rel[@"final_video"] = num(fvr);
      rel[@"final_audio"] = num(far);
      std::printf("  final latents relRMS vs dump final.npz: video %.4f, audio %.4f (a different take after step 0)\n",
                  fvr, far);
    }
  }
  auto timed = [&](PipelineStage &&stage, NSString *name) {
    const auto t = splash::AwakeClock::now();
    stage.forward(batch, events);
    phases[name] = num(splash::millisecondsSince(t) / 1e3);
  };
  timed(DecodingStage(backend, opt.model), @"decode");
  timed(AudioDecodingStage(backend, opt.model), @"audio");
  timed(MuxStage(), @"mux");
  {
    const std::string mp4 = opt.out + "/out.mp4";
    FILE *f = std::fopen(mp4.c_str(), "wb");
    if (!f || std::fwrite(batch.mp4.data(), 1, batch.mp4.size(), f) != batch.mp4.size())
      throw std::runtime_error("cannot write " + mp4);
    std::fclose(f);
  }
  const int64_t frames = int64_t(batch.frames.size() / (size_t(g.width) * g.height * 3));
  run[@"frames"] = @(frames);
  run[@"status"] = @"ok";
  run[@"wall_s_request"] = num(prepS + splash::millisecondsSince(tStart) / 1e3);
  run[@"peak_footprint_gb"] = num(peakFootprintGb());
  writeRunJson(opt.out + "/run.json", run);
  std::printf("generate: %lld frames, out.mp4 %zu bytes; phases prepare %.1f s, denoise %.1f s, decode %s s, audio %s "
              "s; peak footprint %.2f GB\n",
              (long long)frames, batch.mp4.size(), prepS, denoiseS, [[phases[@"decode"] stringValue] UTF8String],
              [[phases[@"audio"] stringValue] UTF8String], peakFootprintGb());
}

// The row classes of a layout and its attention metadata.
void printRows(const ops::RowLayout &l) {
  std::printf("row classes (%d rows, target clock %.6g)\n", l.rows(), l.targetClock);
  std::printf("  %-15s %8s %8s %4s %4s %4s  %-9s %-9s %-11s %s\n", "class", "start", "rows", "ref", "tag", "slot",
              "time", "update", "attention", "t range");
  static const char *times[] = {"video", "condition", "clean", "audio"};
  static const char *updates[] = {"fixed", "augmented", "noised"};
  static const char *roles[] = {"prefix", "video VSA"};
  for (const ops::RowSegment &s : l.segments) {
    const ops::RowClassInfo &c = ops::rowClass(s.cls);
    std::printf("  %-15s %8d %8d %4d %4d %4d  %-9s %-9s %-11s %.6g .. %.6g\n", c.name, s.start, s.rows, s.ref, c.tag,
                c.slot, times[int(c.time)], updates[int(c.update)], roles[int(c.attention)], l.pos[size_t(s.start) * 3],
                l.pos[size_t(s.end() - 1) * 3]);
  }
  auto ranges = [](const std::vector<ops::RowRange> &rs) {
    std::string out;
    for (const ops::RowRange &r : rs) out += std::format(" [{}, {})", r.start, r.end());
    return out;
  };
  std::printf("attention metadata\n  dense prefix rows%s\n  video VSA region%s\n  fixed rows%s\n",
              ranges(l.ranges(ops::RowAttention::Prefix)).c_str(),
              ranges(l.ranges(ops::RowAttention::VideoRegion)).c_str(), ranges(l.fixedRanges()).c_str());
}

// Every layout array against layout.npz (unique: the dump's step-0 timesteps).
void compareLayout(const ops::RowLayout &l, const std::map<std::string, NpyArray> &lz, NSArray *unique) {
  std::printf("layout vs layout.npz\n");
  {
    const NpyArray &pos = need(lz, "position_ids", "<f8");
    const bool shape = pos.shape == std::vector<int64_t>{l.rows(), 3};
    double maxDiff = shape ? 0 : INFINITY;
    for (size_t i = 0; shape && i < l.pos.size(); ++i)
      maxDiff = std::max(maxDiff, std::fabs(l.pos[i] - pos.as<double>()[i]));
    const bool exact = shape && std::memcmp(pos.data.data(), l.pos.data(), l.pos.size() * 8) == 0;
    report("position_ids (float64)", exact, std::format("[{}][3], max |diff| {:.3g}", l.rows(), maxDiff));
  }
  sameInt64(need(lz, "token_tags", "<i8"), l.tags, "token_tags");
  std::vector<float> table;
  std::vector<int64_t> index;
  l.timesteps(0.0f, 0.0f, table, index);  // step 0: sigma 1, clean-time t 0
  bool tableExact = [unique isKindOfClass:[NSArray class]] && unique.count == table.size();
  std::string tableText;
  for (size_t i = 0; i < table.size(); ++i) {
    tableText += std::format("{}{:.9g}", i ? ", " : "", table[i]);
    if (tableExact) tableExact = static_cast<float>([unique[i] doubleValue]) == table[i];
  }
  report("step-0 timestep table", tableExact, "{" + tableText + "} vs run.json step0.unique_timesteps");
  sameInt64(need(lz, "timestep_indices", "<i8"), index, "timestep_indices (step 0)");
  sameInt64(need(lz, "video_indices", "<i8"), l.videoIndices, "video_indices");
  sameInt64(need(lz, "audio_indices", "<i8"), l.audioIndices, "audio_indices");
  sameInt64(need(lz, "text_indices", "<i8"), l.textIndices, "text_indices");
  sameInt64(need(lz, "num_condition_video_rows", "<i8"), std::vector<int64_t>{l.condVideoRows},
            "num_condition_video_rows");
  sameInt64(need(lz, "num_condition_audio_rows", "<i8"), std::vector<int64_t>{l.condAudioRows},
            "num_condition_audio_rows");
}

void run(const std::string &dir, const Options &opt) {
  NSDictionary *runJson = readJson(dir + "/run.json");
  NSDictionary *geo = runJson[@"presentation"][@"geometry"];
  if (![geo isKindOfClass:[NSDictionary class]]) throw std::runtime_error("run.json: no presentation.geometry");
  const ops::VideoGeometry g =
      ops::VideoGeometry::of(jsonInt(geo, "num_frames"), jsonInt(geo, "width"), jsonInt(geo, "height"));
  if (g.latentFrames != jsonInt(geo, "num_latent_frames") || g.latentHeight != jsonInt(geo, "latent_height") ||
      g.latentWidth != jsonInt(geo, "latent_width") || g.audioRows != 2 * jsonInt(geo, "num_audio_latents"))
    throw std::runtime_error("run.json geometry differs from the engine's VideoGeometry");

  const auto cz = splash::dev::readNpz(dir + "/conditioning.npz");
  const auto lz = splash::dev::readNpz(dir + "/layout.npz");
  const auto nz = splash::dev::readNpz(dir + "/initial_noise.npz");
  const auto sz = splash::dev::readNpz(dir + "/step0.npz");

  const NpyArray &textTags = need(cz, "text_token_tags", "<i8");
  std::vector<uint8_t> tags(textTags.count());
  for (size_t i = 0; i < tags.size(); ++i) tags[i] = static_cast<uint8_t>(textTags.as<int64_t>()[i]);

  // references: refN_<kind>_video_latents [1][F][24][h][w] -> [24][F][h][w]; refN_<kind>_audio_latents [1][T][64]
  struct Ref {
    std::string kind;
    const NpyArray *video = nullptr, *audio = nullptr;
    std::vector<float> channelFirst;
  };
  std::map<int, Ref> refs;
  const std::regex key(R"(ref(\d+)_(video|image|audio)_(video|audio)_latents)");
  for (const auto &[name, array] : cz) {
    std::smatch m;
    if (!std::regex_match(name, m, key)) continue;
    Ref &r = refs[std::stoi(m[1])];
    r.kind = m[2];
    (m[3] == "video" ? r.video : r.audio) = &array;
  }
  model::PrepInputs in;
  in.textRows = static_cast<int32_t>(tags.size());
  in.textTags = tags.data();
  std::printf("%s\n  canvas %d x %d, %d frames: latents %d x %d x %d, %d audio rows; %d text rows\n  references:",
              dir.c_str(), g.width, g.height, g.frames, g.latentFrames, g.latentHeight, g.latentWidth, g.audioRows,
              in.textRows);
  for (auto &[index, r] : refs) {
    model::ReferenceInput ri;
    using Kind = ops::ReferenceShape::Kind;
    ri.shape.kind = r.kind == "video" ? Kind::Video : r.kind == "image" ? Kind::Image : Kind::Audio;
    if (r.video) {
      const std::vector<int64_t> &s = r.video->shape;
      if (r.video->dtype != "<f4" || s.size() != 5 || s[0] != 1 || s[2] != H3_VAE_LATENT)
        throw std::runtime_error("reference " + std::to_string(index) + ": video latents [1][F][24][h][w] f32");
      const int64_t F = s[1], h = s[3], w = s[4];
      r.channelFirst.resize(size_t(F) * 24 * h * w);
      for (int64_t f = 0; f < F; ++f)
        for (int64_t c = 0; c < 24; ++c)
          std::memcpy(&r.channelFirst[((c * F + f) * h) * w], &r.video->as<float>()[((f * 24 + c) * h) * w],
                      size_t(h * w) * 4);
      ri.shape.frames = int32_t(F);
      ri.shape.latentHeight = int32_t(h);
      ri.shape.latentWidth = int32_t(w);
      ri.video = r.channelFirst.data();
    }
    if (r.audio) {
      if (r.audio->dtype != "<f4" || r.audio->shape.size() != 3 || r.audio->shape[2] != 64)
        throw std::runtime_error("reference " + std::to_string(index) + ": audio latents [1][T][64] f32");
      ri.shape.audioLatents = int32_t(r.audio->shape[1]);
      ri.audio = r.audio->as<float>();
    }
    std::printf(" %d %s (%d x %d x %d latents, %d audio latents)", index, r.kind.c_str(), ri.shape.frames,
                ri.shape.latentHeight, ri.shape.latentWidth, ri.shape.audioLatents);
    in.references.push_back(ri);
  }
  std::printf("\n");
  model::Prepared p;
  model::Prep::referenceRows(g, in, p);
  const ops::RowLayout &l = p.sequence;

  printRows(l);
  NSArray *unique = runJson[@"step0"][@"unique_timesteps"];
  compareLayout(l, lz, unique);

  std::printf("step-0 inputs vs step0.npz (target rows from initial_noise.npz)\n");
  const NpyArray &vin = need(sz, "video_in", "<f4"), &ain = need(sz, "audio_in", "<f4");
  const NpyArray &nv = need(nz, "video", "<f4"), &na = need(nz, "audio", "<f4");
  const size_t W = H3_DIT_VIDEO_CHANNELS, A = H3_DIT_AUDIO_CHANNELS;
  const size_t cv = size_t(l.condVideoRows), ca = size_t(l.condAudioRows);
  if (vin.count() != l.videoIndices.size() * W || ain.count() != l.audioIndices.size() * A ||
      nv.count() != size_t(l.targetVideoRows) * W || na.count() != size_t(l.targetAudioRows) * A)
    throw std::runtime_error("step0 / initial_noise shapes differ from the layout's rows");
  std::vector<float> videoIn(p.cond), audioIn(p.condAudio);
  videoIn.insert(videoIn.end(), nv.as<float>(), nv.as<float>() + nv.count());
  audioIn.insert(audioIn.end(), na.as<float>(), na.as<float>() + na.count());
  auto bitExact = [](const float *a, const float *b, size_t n) { return std::memcmp(a, b, n * 4) == 0; };
  report("target video rows", bitExact(videoIn.data() + cv * W, vin.as<float>() + cv * W, nv.count()),
         std::format("[{}][96] = initial_noise video", l.targetVideoRows));
  report("target audio rows", bitExact(audioIn.data() + ca * A, ain.as<float>() + ca * A, na.count()),
         std::format("[{}][32] = initial_noise audio", l.targetAudioRows));
  report("audio condition rows", bitExact(audioIn.data(), ain.as<float>(), ca * A),
         std::format("[{}][32] clean, channel-major", ca));
  const double cleanRel = relRms(p.cond.data(), vin.as<float>(), cv * W);
  report("video condition rows", "n/a",
         std::format("[{}][96] clean vs augmented: relRMS {:.3e} (the 1e-3 noise term; eps from CUDA)", cv, cleanRel));
  // the noise the dump implies, eps = (x' - 0.999 x) / (1 - 0.999), and x' rebuilt from it by the engine's arithmetic
  std::vector<float> eps(cv * W);
  double mean = 0, sq = 0;
  const float t32 = ops::kConditionTime, omt = 1.0f - t32;
  for (size_t i = 0; i < eps.size(); ++i) {
    eps[i] = (vin.as<float>()[i] - t32 * p.cond[i]) / omt;
    mean += eps[i];
    sq += double(eps[i]) * eps[i];
  }
  mean /= double(eps.size());
  const double sd = std::sqrt(sq / double(eps.size()) - mean * mean);
  model::PrepInputs again = in;
  again.referenceNoise = eps.data();
  model::Prepared q;
  model::Prep::referenceRows(g, again, q);
  size_t same = 0;
  for (size_t i = 0; i < q.cond.size(); ++i) same += std::memcmp(&q.cond[i], &vin.as<float>()[i], 4) == 0;
  report("video condition noise", "n/a",
         std::format("implied eps mean {:.4f} std {:.4f}; x' rebuilt from it: {:.4f}% bit-equal, relRMS {:.3e}", mean,
                     sd, 100.0 * double(same) / double(q.cond.size()),
                     relRms(q.cond.data(), vin.as<float>(), q.cond.size())));
  report("whole step-0 input", "n/a",
         std::format("relRMS video_in {:.3e}, audio_in {:.3e} (video condition rows clean)",
                     relRms(videoIn.data(), vin.as<float>(), vin.count()),
                     relRms(audioIn.data(), ain.as<float>(), ain.count())));

  // the activation plan of the sequence as the dense (pika-6) DiT plans it, and the prep's token refiner scratch (per
  // text row: the fp32 hidden row, its bf16 copy, condition_proj, one block's n1 qkv q k att o x1 n2 fc1 act m x2,
  // the final norm: 278,016 bytes)
  const ops::SequenceLayout seq = l.sequence(g, H3_VSA_TILE);
  ops::VsaSpec dense;
  dense.dense = true;
  const ops::VsaTable vsa(seq, dense);
  ops::SequenceLayout t2v = seq;
  t2v.cond = 0;
  t2v.condAudio = 0;
  t2v.segments.clear();
  t2v.text = 389;
  const ops::VsaTable vsaT2v(t2v, dense);
  std::printf("activation plan (dense DiT, model::MemoryPlan::bytesFor)\n  %d rows (C %d, Ca %d, %d adaLN slots): "
              "%.2f GB; text-to-video of the canvas (389 text rows, %d rows): %.2f GB\n  token refiner scratch, %d text "
              "rows: %.2f GB\n",
              seq.rows(), seq.cond, seq.condAudio, seq.adaModalities(),
              double(model::MemoryPlan::bytesFor(seq, 0, &vsa)) / 1e9, t2v.rows(),
              double(model::MemoryPlan::bytesFor(t2v, 0, &vsaT2v)) / 1e9, l.textRows, l.textRows * 278016.0 / 1e9);
  if (opt.model.empty()) return;

  // ---- one step-0 forward through the engine (GPU)
  std::printf("step-0 forward through the engine (%s)\n", opt.model.c_str());
  const auto t0 = splash::AwakeClock::now();
  splash::metal::MetalBackend backend(opt.metallib);
  const NpyArray &embeds = need(cz, "prompt_embeds", "<f4");
  if (embeds.shape != std::vector<int64_t>{in.textRows, 5120})
    throw std::runtime_error("prompt_embeds is not [text rows][5120]");
  const std::string ditName = "pika-6";
  const std::string ditDir = opt.model + "/" + model::ditDirectory(ditName);
  model::PrepInputs fin = in;
  fin.hidden = embeds.as<float>();
  fin.seed = 42;
  fin.adapters = model::adapterSpecs(ditDir, std::nullopt);
  model::Prepared fp;
  model::Prep::run(backend, opt.model, ditName, g, fin, fp);
  const double prepS = splash::millisecondsSince(t0) / 1e3;
  // Pika's inputs in place of the engine's draws: the dumped condition rows (augmented with Pika's CUDA noise) and
  // the initial noise of the target rows
  fp.cond.assign(vin.as<float>(), vin.as<float>() + cv * W);
  if (!bitExact(fp.condAudio.data(), ain.as<float>(), ca * A)) throw std::runtime_error("audio condition rows differ");
  fp.videoNoise.assign(nv.as<float>(), nv.as<float>() + nv.count());
  fp.audioNoise.assign(na.as<float>(), na.as<float>() + na.count());
  if (opt.generate) {
    std::printf("  prep %.1f s (adaLN %.1f s)\n", prepS, fp.adalnS);
    generateClip(backend, opt, g, std::move(fp), dir, prepS);
    return;
  }
  const ops::VsaSpec spec = model::fastH3Contract(ditDir).vsa;
  const std::vector<model::AdapterSpec> specs = model::adapterSpecs(ditDir, std::nullopt);
  if (specs.size() > 1) throw std::runtime_error("one runtime adapter at a time");
  DitInputs inputs{fp.textRows, fp.tags.data(), fp.rope.data(), fp.text.data(), fp.sequence.condVideoRows,
                   fp.cond.data()};
  inputs.condAudioRows = fp.sequence.condAudioRows;
  inputs.condAudio = fp.condAudio.data();
  inputs.segments = &fp.sequence.engineSegments;
  std::vector<float> video(nv.count()), audio(na.count());
  ForwardStats stats;
  double buildS = 0;
  {
    const auto tb = splash::AwakeClock::now();
    std::unique_ptr<model::DitWeights> weights = model::DitWeights::load(backend, ditDir);
    std::unique_ptr<model::Adapter> adapter =
        specs.empty() ? nullptr : model::Adapter::load(backend, specs[0], weights->qkfixBlocks());
    Dit dit(backend, g, inputs, std::move(weights), spec, std::move(adapter));
    buildS = splash::millisecondsSince(tb) / 1e3;
    dit.forward(fp.adaBlocks.data(), fp.adaFinal.data(), fp.videoNoise.data(), fp.audioNoise.data(), video.data(),
                audio.data(), stats);
  }
  const auto vz = splash::dev::readNpz(dir + "/step0_velocity.npz");
  const NpyArray &rv = need(vz, "video", "<f4"), &ra = need(vz, "audio", "<f4");
  if (rv.count() != video.size() || ra.count() != audio.size())
    throw std::runtime_error("step0_velocity.npz shapes differ from the target rows");
  std::printf("  step-0 velocity relRMS vs step0_velocity.npz (target rows): video %.4f, audio %.4f\n",
              relRms(video.data(), rv.as<float>(), video.size()), relRms(audio.data(), ra.as<float>(), audio.size()));
  std::printf("  times: prep %.1f s (adaLN %.1f s), DiT build %.1f s, forward %.1f s (blocks %.1f s, GPU %.1f s); "
              "device %.0f MB after the forward; process peak footprint %.2f GB\n",
              prepS, fp.adalnS, buildS, stats.wallMs / 1e3, stats.blocksSpanMs / 1e3, stats.blocksGpuMs / 1e3,
              stats.deviceMb, peakFootprintGb());
  if (!opt.out.empty()) {
    const int64_t nvRows = int64_t(nv.count() / W), naRows = int64_t(na.count() / A);
    const NpyArray v = NpyArray::of("<f4", {nvRows, int64_t(W)}, video.data(), video.size() * 4);
    const NpyArray a = NpyArray::of("<f4", {naRows, int64_t(A)}, audio.data(), audio.size() * 4);
    std::filesystem::create_directories(opt.out);
    splash::dev::writeNpz(opt.out + "/step0_velocity.npz", {{"video", &v}, {"audio", &a}});
  }
}

} // namespace

int main(int argc, char **argv) {
  const char *usage = "usage: video-ref2va-layout DUMP_DIRECTORY [--forward MODEL_DIRECTORY --metallib LIB [--out DIR]]\n"
                      "       video-ref2va-layout DUMP_DIRECTORY --generate MODEL_DIRECTORY --metallib LIB --out DIR\n";
  std::vector<std::string> args(argv + 1, argv + argc);
  Options opt;
  std::string dump;
  for (size_t i = 0; i < args.size(); ++i) {
    const bool more = i + 1 < args.size();
    if (args[i] == "--forward" && more) opt.model = args[++i];
    else if (args[i] == "--generate" && more) {
      opt.model = args[++i];
      opt.generate = true;
    } else if (args[i] == "--metallib" && more) opt.metallib = args[++i];
    else if (args[i] == "--out" && more) opt.out = args[++i];
    else if (dump.empty() && args[i].rfind("--", 0) != 0) dump = args[i];
    else {
      std::fprintf(stderr, "%s", usage);
      return 64;
    }
  }
  if (dump.empty() || (!opt.model.empty() && opt.metallib.empty()) || (opt.generate && opt.out.empty())) {
    std::fprintf(stderr, "%s", usage);
    return 64;
  }
  @autoreleasepool {
    try {
      run(dump, opt);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "video-ref2va-layout: %s\n", e.what());
      return 1;
    }
  }
  std::printf(failed ? "layout: NOT EXACT\n" : "layout: exact\n");
  return failed ? 1 : 0;
}
