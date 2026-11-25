#include "video/model/FastH3.hpp"

#include "video/model/WeightFiles.hpp"

#include <algorithm>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace splash::video::model {
namespace {

std::vector<double> numbers(id value) {
  std::vector<double> out;
  if (![value isKindOfClass:[NSArray class]]) return out;
  for (id x in (NSArray *)value) {
    if (![x isKindOfClass:[NSNumber class]]) return {};
    out.push_back([x doubleValue]);
  }
  return out;
}

} // namespace

bool isFastH3(const std::string &ditDir) {
  const std::filesystem::path path = std::filesystem::path(ditDir) / "manifest.json";
  if (!std::filesystem::exists(path)) return false;
  return [readManifest(path)[@"format"] isEqual:[NSString stringWithUTF8String:kFastH3Format]];
}

std::string ditDirectory(std::string_view dit) { return "dit-" + std::string(dit); }

FastH3Contract fastH3Contract(const std::string &ditDir) {
  const std::string path = ditDir + "/manifest.json";
  NSDictionary *manifest = readManifest(path);
  NSDictionary *schedule = manifest[@"schedule"], *vsa = manifest[@"vsa"];
  if (![manifest[@"format"] isEqual:[NSString stringWithUTF8String:kFastH3Format]] || ![schedule isKindOfClass:[NSDictionary class]] ||
      ![vsa isKindOfClass:[NSDictionary class]])
    throw std::invalid_argument(path + ": not a FastH3 DiT of format " + kFastH3Format);
  FastH3Contract c;
  for (double r : numbers(schedule[@"dmd_rungs"])) c.rungs.push_back(static_cast<int32_t>(r));
  const std::vector<double> tile = numbers(vsa[@"tile"]), timesteps = numbers(manifest[@"adaln_timesteps"]);
  NSNumber *video = schedule[@"video_shift"], *audio = schedule[@"audio_shift"], *sparsity = vsa[@"sparsity"];
  if (c.rungs.empty() || tile.size() != 3 || timesteps.empty() || ![video isKindOfClass:[NSNumber class]] ||
      ![audio isKindOfClass:[NSNumber class]] || ![sparsity isKindOfClass:[NSNumber class]])
    throw std::invalid_argument(path + ": no schedule {dmd_rungs, video_shift, audio_shift}, vsa {tile, sparsity} or "
                                       "adaln_timesteps");
  c.videoShift = video.doubleValue;
  c.audioShift = audio.doubleValue;
  c.vsa = {static_cast<int32_t>(tile[0]), static_cast<int32_t>(tile[1]), static_cast<int32_t>(tile[2]),
           sparsity.doubleValue};
  for (double t : timesteps) c.adalnTimesteps.push_back(static_cast<float>(t));
  NSString *attention = vsa[@"attention"];
  if (attention && ![attention isEqual:@"sparse"] && ![attention isEqual:@"dense"])
    throw std::invalid_argument(path + ": vsa.attention must be sparse or dense");
  c.vsa.dense = [attention isEqual:@"dense"];
  if (c.vsa.dense && c.vsa.sparsity != 0.0)
    throw std::invalid_argument(path + ": vsa.attention dense takes sparsity 0 (a DiT without the compression gate)");
  NSString *sampler = schedule[@"sampler"], *precision = schedule[@"sigma_precision"];
  if (sampler && ![sampler isEqual:@"euler"] && ![sampler isEqual:@"renoise"])
    throw std::invalid_argument(path + ": schedule.sampler must be euler or renoise");
  if (precision && ![precision isEqual:@"float32"] && ![precision isEqual:@"float64"])
    throw std::invalid_argument(path + ": schedule.sigma_precision must be float32 or float64");
  c.sampler = [sampler isEqual:@"renoise"] ? Sampler::Renoise : Sampler::Euler;
  c.sigma64 = [precision isEqual:@"float64"];
  if (std::optional<int64_t> offset = integer(schedule[@"renoise_seed_offset"])) c.renoiseSeedOffset = uint64_t(*offset);
  return c;
}

void Prep::scheduleRungs(const FastH3Contract &c, bool conditioned, Prepared &p) {
  const std::vector<int32_t> &rungs = c.rungs;
  for (size_t i = 0; i < rungs.size(); ++i)
    if (rungs[i] <= 0 || rungs[i] > 1000 || (i && rungs[i] >= rungs[i - 1]))
      throw std::invalid_argument("prep: DMD rungs must be strictly decreasing integers in (0, 1000]");
  // torch: base = tensor([step / 1000.0 ...] + [0.0], float32); sigmas = shift * base / (1 + (shift - 1) * base) in
  // fp32 element ops; scheduler timesteps 1 - sigma
  std::vector<float> base;
  for (int32_t r : rungs) base.push_back(static_cast<float>(static_cast<double>(r) / 1000.0));
  base.push_back(0.0f);
  auto sigmas = [&](double shift) {
    const float shift32 = static_cast<float>(shift), shiftM1 = static_cast<float>(shift - 1.0);
    std::vector<float> v;
    if (c.sigma64) {  // Pika's H3SigmaSchedule: q = knot / 1000 in float64, s q / (1 + (s - 1) q) in float64, fp32
      for (int32_t r : rungs) {
        const double q = static_cast<double>(r) / 1000.0;
        v.push_back(static_cast<float>(shift * q / (1.0 + (shift - 1.0) * q)));
      }
      v.push_back(0.0f);
      return v;
    }
    for (float b : base) {
      const float num = shift32 * b;
      const float den0 = shiftM1 * b;
      const float den = 1.0f + den0;
      v.push_back(num / den);
    }
    return v;
  };
  p.vSigmas = sigmas(c.videoShift);
  p.aSigmas = sigmas(c.audioShift);
  p.renoise = c.sampler == Sampler::Renoise;
  p.renoiseSeedOffset = c.renoiseSeedOffset;
  p.steps = static_cast<int32_t>(rungs.size());
  auto timesteps = [](const std::vector<float> &s) {
    std::vector<float> t(s.size() - 1);
    for (size_t i = 0; i + 1 < s.size(); ++i) t[i] = static_cast<float>(1.0 - static_cast<double>(s[i]));
    return t;
  };
  p.vT = timesteps(p.vSigmas);
  p.aT = timesteps(p.aSigmas);
  std::set<float> distinct(p.vT.begin(), p.vT.end());
  distinct.insert(p.aT.begin(), p.aT.end());
  p.cT.clear();
  if (conditioned) {  // as schedule(): conditioning rows at max(t, KEYFRAME_NOISE_AUG) in Python floats, as fp32
    for (float t : p.vT) p.cT.push_back(static_cast<float>(std::max(static_cast<double>(t), 0.999)));
    distinct.insert(p.cT.begin(), p.cT.end());
  }
  p.table.assign(distinct.begin(), distinct.end());
  auto index = [&](const std::vector<float> &t) {
    std::vector<int64_t> idx(t.size());
    for (size_t i = 0; i < t.size(); ++i)
      idx[i] = std::lower_bound(p.table.begin(), p.table.end(), t[i]) - p.table.begin();
    return idx;
  };
  p.tv = index(p.vT);
  p.ta = index(p.aT);
  p.tc = index(p.cT);
}

} // namespace splash::video::model
