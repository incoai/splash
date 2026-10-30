#include "engine/VideoFrames.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <QuartzCore/QuartzCore.h>
#import <VideoToolbox/VideoToolbox.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sysexits.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace splash::video {
namespace {

// Pillow's fixed-point resampling (libImaging/Resample.c): weights carry 22
// fraction bits, leaving 8 for a pixel and 2 for the sum's overshoot.
constexpr int kPrecisionBits = 32 - 8 - 2;
// The widest frame dimension accepted on either side, as images.
constexpr int kMaxDimension = 16384;
// A time a decoded frame may differ from the probe's and be the same frame:
// far below the spacing of frames at any rate a camera records.
constexpr double kTimeTolerance = 0.0005;
// How long the hardware decoder may go without returning a frame before
// decode gives up on it: wired memory near the kernel's limit stalls Apple's
// decoder for good (with a model's weights wired by Metal).
constexpr double kStallSeconds = 5.0;
// The exit status that tells the server so (sysexits EX_TEMPFAIL): it decodes
// again with --software.
constexpr int kStalled = 75;
// The exit status of a --software decode of a codec that has no software
// decoder (sysexits EX_UNAVAILABLE): only the hardware decoder can read it.
constexpr int kNoSoftwareDecoder = 69;

class NoSoftwareDecoder final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

double bicubic(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0)
    return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

// Pillow's precompute_coeffs and normalize_coeffs_8bpc for the box [0, in):
// each output position's first source index and count, and its weights.
struct Coefficients {
  int ksize = 0;
  std::vector<int> bounds;
  std::vector<int32_t> weights;
};

Coefficients coefficients(int inSize, int outSize) {
  const float in0 = 0.0f, in1 = static_cast<float>(inSize);
  const double scale = static_cast<double>(in1 - in0) / outSize;
  const double filterScale = std::max(scale, 1.0);
  const double support = 2.0 * filterScale;
  Coefficients result;
  result.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  result.bounds.resize(static_cast<size_t>(outSize) * 2);
  result.weights.assign(static_cast<size_t>(outSize) * result.ksize, 0);
  std::vector<double> k(static_cast<size_t>(result.ksize));
  for (int xx = 0; xx < outSize; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    const double ss = 1.0 / filterScale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0)
      xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > inSize)
      xmax = inSize;
    xmax -= xmin;
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      k[x] = bicubic((x + xmin - center + 0.5) * ss);
      ww += k[x];
    }
    for (int x = 0; x < xmax; ++x) {
      const double w = ww != 0.0 ? k[x] / ww : k[x];
      result.weights[static_cast<size_t>(xx) * result.ksize + x] =
          static_cast<int32_t>(w < 0 ? -0.5 + w * (1 << kPrecisionBits)
                                     : 0.5 + w * (1 << kPrecisionBits));
    }
    result.bounds[xx * 2] = xmin;
    result.bounds[xx * 2 + 1] = xmax;
  }
  return result;
}

uint8_t clip8(int32_t sum) {
  if (sum >= (1 << kPrecisionBits << 8))
    return 255;
  if (sum <= 0)
    return 0;
  return static_cast<uint8_t>(sum >> kPrecisionBits);
}

// An upright frame's pixel (X, Y) reads the coded frame's pixel through the
// inverse of the track's display transform, a quarter turn, optionally
// mirrored.
struct Orientation {
  // display = coded * [[a, b], [c, d]], each -1, 0 or 1, then translated
  // so that the frame starts at the origin.
  int a = 1, b = 0, c = 0, d = 1;

  [[nodiscard]] bool swapsAxes() const noexcept { return a == 0; }
};

Orientation orientation(CGAffineTransform transform) {
  const auto unit = [](CGFloat value) -> int {
    if (std::fabs(value) < 1e-6)
      return 0;
    if (std::fabs(std::fabs(value) - 1.0) < 1e-6)
      return value > 0 ? 1 : -1;
    return 2;
  };
  const Orientation result{unit(transform.a), unit(transform.b),
                           unit(transform.c), unit(transform.d)};
  const bool straight = result.b == 0 && result.c == 0 &&
                        std::abs(result.a) == 1 && std::abs(result.d) == 1;
  const bool turned = result.a == 0 && result.d == 0 &&
                      std::abs(result.b) == 1 && std::abs(result.c) == 1;
  if (!straight && !turned)
    throw VideoError("the video's display transform is not a quarter turn");
  return result;
}

int64_t ticks(CMTime time, int32_t timescale) {
  return CMTimeConvertScale(time, timescale, kCMTimeRoundingMethod_Default).value;
}

void writeAll(const void *data, size_t size) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  while (size > 0) {
    const ssize_t written = ::write(STDOUT_FILENO, bytes, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      throw std::system_error(errno, std::generic_category(), "cannot write frames");
    bytes += written;
    size -= static_cast<size_t>(written);
  }
}

// Holds a Core Foundation reference and releases it.
template <typename Ref> class Owned final {
public:
  Owned() = default;
  explicit Owned(Ref ref) noexcept : ref_(ref) {}
  ~Owned() { reset(); }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  Owned(Owned &&other) noexcept : ref_(std::exchange(other.ref_, nullptr)) {}
  Owned &operator=(Owned &&other) noexcept {
    if (this != &other) {
      reset();
      ref_ = std::exchange(other.ref_, nullptr);
    }
    return *this;
  }
  void reset(Ref ref = nullptr) noexcept {
    if (ref_)
      CFRelease(ref_);
    ref_ = ref;
  }
  [[nodiscard]] Ref get() const noexcept { return ref_; }
  explicit operator bool() const noexcept { return ref_ != nullptr; }

private:
  Ref ref_ = nullptr;
};

// The first video track of a file AVFoundation reads.
struct Video {
  AVURLAsset *asset = nil;
  AVAssetTrack *track = nil;
  // The frame's displayed size before it is turned upright: the clean
  // aperture at the pixel aspect ratio.
  int width = 0, height = 0;
  Orientation orientation;
  int32_t timescale = 1;
  // The codec's FourCC, as errors name it.
  std::string codec;

  [[nodiscard]] int displayWidth() const noexcept {
    return orientation.swapsAxes() ? height : width;
  }
  [[nodiscard]] int displayHeight() const noexcept {
    return orientation.swapsAxes() ? width : height;
  }
};

// mime names the container where its file's name would not: AVFoundation
// chooses a reader by the extension, and the server's file has none of its
// own.
Video openAVFoundation(const char *path, NSString *mime) {
  Video video;
  NSMutableDictionary *options =
      [@{AVURLAssetPreferPreciseDurationAndTimingKey : @YES} mutableCopy];
  if (mime)
    options[AVURLAssetOverrideMIMETypeKey] = mime;
  video.asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(path)] options:options];
  dispatch_semaphore_t loaded = dispatch_semaphore_create(0);
  __block NSArray<AVAssetTrack *> *tracks = nil;
  [video.asset loadTracksWithMediaType:AVMediaTypeVideo
                     completionHandler:^(NSArray<AVAssetTrack *> *found, NSError *) {
                       tracks = found;
                       dispatch_semaphore_signal(loaded);
                     }];
  dispatch_semaphore_wait(loaded, DISPATCH_TIME_FOREVER);
  if (tracks.count == 0)
    throw VideoError("the file has no video track this decoder reads (MP4, QuickTime, "
                     "MPEG program and transport streams)");
  AVAssetTrack *track = video.track = tracks[0];
  NSArray<NSString *> *keys = @[
    @"naturalSize", @"preferredTransform", @"segments", @"naturalTimeScale",
    @"formatDescriptions"
  ];
  [track loadValuesAsynchronouslyForKeys:keys
                       completionHandler:^{
                         dispatch_semaphore_signal(loaded);
                       }];
  dispatch_semaphore_wait(loaded, DISPATCH_TIME_FOREVER);
  for (NSString *key in keys) {
    NSError *error = nil;
    if ([track statusOfValueForKey:key error:&error] != AVKeyValueStatusLoaded)
      throw VideoError("the video track cannot be read");
  }
  CGSize size = track.naturalSize;
  if (track.formatDescriptions.count) {
    const auto format = (__bridge CMVideoFormatDescriptionRef)track.formatDescriptions[0];
    // The clean aperture at the pixel aspect ratio, as it is displayed.
    size = CMVideoFormatDescriptionGetPresentationDimensions(format, true, true);
    const FourCharCode code = CMFormatDescriptionGetMediaSubType(format);
    for (int shift = 24; shift >= 0; shift -= 8) {
      const char c = static_cast<char>(code >> shift & 0xFF);
      video.codec += c >= 32 && c < 127 ? c : '?';
    }
  }
  if (!(size.width >= 1 && size.height >= 1 && size.width <= kMaxDimension &&
        size.height <= kMaxDimension))
    throw VideoError("video dimensions must be 1-16384 pixels");
  video.width = static_cast<int>(std::lround(size.width));
  video.height = static_cast<int>(std::lround(size.height));
  video.orientation = orientation(track.preferredTransform);
  video.timescale = track.naturalTimeScale > 0 ? track.naturalTimeScale : 600;
  return video;
}

// The container, from the file's first bytes, named by MIME type where its
// file would need another extension than QuickTime's (MP4 and 3GP read as
// one).
Video openVideo(const char *path) {
  uint8_t head[400] = {};
  const int fd = ::open(path, O_RDONLY);
  if (fd < 0)
    throw VideoError("the video cannot be read");
  const ssize_t got = ::read(fd, head, sizeof head);
  ::close(fd);
  const size_t size = got > 0 ? static_cast<size_t>(got) : 0;
  // MPEG transport packets of 188 bytes, or 192 with a timecode first
  // (M2TS, as AVCHD cameras write): a sync byte starts each.
  for (const size_t packet : {size_t{188}, size_t{192}}) {
    const size_t first = packet - 188;
    if (size > first + 2 * packet && head[first] == 0x47 && head[first + packet] == 0x47 &&
        head[first + 2 * packet] == 0x47)
      return openAVFoundation(path, @"video/mp2t");
  }
  if (size >= 4 && head[0] == 0 && head[1] == 0 && head[2] == 1 && head[3] == 0xBA)
    return openAVFoundation(path, @"video/mpeg");
  if (size >= 12 && std::memcmp(head, "RIFF", 4) == 0 && std::memcmp(head + 8, "AVI ", 4) == 0)
    return openAVFoundation(path, @"video/avi");
  return openAVFoundation(path, nil);
}

// A sample's media time in the track's timeline, through the edit list; none
// for a sample no edit shows.
std::optional<CMTime> shownAt(NSArray<AVAssetTrackSegment *> *segments, CMTime time) {
  for (AVAssetTrackSegment *segment in segments) {
    if (segment.empty)
      continue;
    const CMTimeMapping mapping = segment.timeMapping;
    if (!CMTimeRangeContainsTime(mapping.source, time))
      continue;
    // An edit that keeps the media's rate shifts it exactly.
    return CMTimeCompare(mapping.source.duration, mapping.target.duration) == 0
               ? CMTimeAdd(CMTimeSubtract(time, mapping.source.start), mapping.target.start)
               : CMTimeMapTimeFromRangeToRange(time, mapping.source, mapping.target);
  }
  return std::nullopt;
}

struct Index {
  std::vector<int64_t> frames, keyframes;
};

// The displayed frames' presentation times and those of the keyframes among
// them, in the track's timeline, read from the samples, none decoded.
Index frameIndex(const Video &video) {
  NSError *error = nil;
  AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:video.asset error:&error];
  if (!reader)
    throw VideoError("the video cannot be read");
  AVAssetReaderTrackOutput *output =
      [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:video.track
                                                 outputSettings:nil];
  output.alwaysCopiesSampleData = NO;
  [reader addOutput:output];
  if (![reader startReading])
    throw VideoError("the video cannot be read");
  struct Sample {
    CMTime time;
    bool keyframe;
  };
  std::vector<Sample> samples;
  while (CMSampleBufferRef buffer = [output copyNextSampleBuffer]) {
    const CMItemCount count = CMSampleBufferGetNumSamples(buffer);
    std::vector<CMSampleTimingInfo> timing(static_cast<size_t>(std::max<CMItemCount>(count, 0)));
    CMItemCount timed = 0;
    // A buffer without samples marks a point in the stream; it holds no frame.
    if (count > 0 && CMSampleBufferGetSampleTimingInfoArray(
                         buffer, count, timing.data(), &timed) == noErr) {
      CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(buffer, false);
      for (CMItemCount i = 0; i < timed; ++i) {
        bool shown = true, keyframe = true;
        if (attachments && i < CFArrayGetCount(attachments)) {
          const auto attached =
              static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(attachments, i));
          shown = CFDictionaryGetValue(attached, kCMSampleAttachmentKey_DoNotDisplay) !=
                  kCFBooleanTrue;
          keyframe = CFDictionaryGetValue(attached, kCMSampleAttachmentKey_NotSync) !=
                     kCFBooleanTrue;
        }
        if (shown && CMTIME_IS_NUMERIC(timing[i].presentationTimeStamp))
          samples.push_back({timing[i].presentationTimeStamp, keyframe});
      }
    }
    CFRelease(buffer);
  }
  // A file that ends early still lists the frames before the cut.
  if (reader.status == AVAssetReaderStatusFailed && samples.empty())
    throw VideoError("the video cannot be read");
  Index index;
  NSArray<AVAssetTrackSegment *> *segments = video.track.segments;
  for (const Sample &sample : samples) {
    if (const std::optional<CMTime> shown = shownAt(segments, sample.time)) {
      index.frames.push_back(ticks(*shown, video.timescale));
      if (sample.keyframe)
        index.keyframes.push_back(index.frames.back());
    }
  }
  for (std::vector<int64_t> *times : {&index.frames, &index.keyframes}) {
    std::sort(times->begin(), times->end());
    times->erase(std::unique(times->begin(), times->end()), times->end());
  }
  if (index.frames.empty())
    throw VideoError("the video has no frames");
  return index;
}

int probe(const char *path) {
  const Video video = openVideo(path);
  const Index found = frameIndex(video);
  std::string json = "{\"width\":" + std::to_string(video.displayWidth()) +
                     ",\"height\":" + std::to_string(video.displayHeight()) +
                     ",\"timescale\":" + std::to_string(video.timescale) + ",\"frames\":[";
  for (size_t i = 0; i < found.frames.size(); ++i)
    json += (i ? "," : "") + std::to_string(found.frames[i]);
  json += "],\"keyframes\":[";
  for (size_t i = 0; i < found.keyframes.size(); ++i)
    json += (i ? "," : "") + std::to_string(found.keyframes[i]);
  json += "]}\n";
  writeAll(json.data(), json.size());
  return 0;
}

// What decode reads from stdin: the frames to return, in increasing order of
// their probe times, the ranges they are read in, each the indices of its
// first and last frame, and the probe's keyframes. A range is decoded from
// the keyframe before its first frame, so the server starts one where a
// keyframe lies between two frames (server/videos.py read_ranges).
struct Request {
  int32_t timescale = 0;
  std::vector<int64_t> frames;
  std::vector<std::pair<size_t, size_t>> ranges;
  std::vector<int64_t> keyframes;

  [[nodiscard]] double seconds(size_t index) const {
    return static_cast<double>(frames[index]) / timescale;
  }
};

int64_t integer(id value, const char *what) {
  if (![value isKindOfClass:NSNumber.class] ||
      CFNumberIsFloatType((__bridge CFNumberRef)value))
    throw VideoError(std::string("invalid decode request: ") + what);
  return [value longLongValue];
}

Request readRequest() {
  NSError *error = nil;
  NSData *input =
      [NSFileHandle.fileHandleWithStandardInput readDataToEndOfFileAndReturnError:&error];
  if (!input)
    throw VideoError("invalid decode request");
  id parsed = [NSJSONSerialization JSONObjectWithData:input options:0 error:&error];
  if (![parsed isKindOfClass:NSDictionary.class])
    throw VideoError("invalid decode request");
  NSDictionary *object = parsed;
  Request request;
  const int64_t timescale = integer(object[@"timescale"], "timescale");
  if (timescale < 1 || timescale > std::numeric_limits<int32_t>::max())
    throw VideoError("invalid decode request: timescale");
  request.timescale = static_cast<int32_t>(timescale);
  NSArray *frames = object[@"frames"], *ranges = object[@"ranges"];
  if (![frames isKindOfClass:NSArray.class] || ![ranges isKindOfClass:NSArray.class] ||
      frames.count == 0 || ranges.count == 0)
    throw VideoError("invalid decode request: frames and ranges");
  for (id frame in frames) {
    request.frames.push_back(integer(frame, "frames"));
    if (request.frames.size() > 1 &&
        request.frames.back() <= request.frames[request.frames.size() - 2])
      throw VideoError("invalid decode request: frames must increase");
  }
  size_t next = 0;
  for (id range in ranges) {
    if (![range isKindOfClass:NSArray.class] || [range count] != 2)
      throw VideoError("invalid decode request: ranges");
    const int64_t first = integer(range[0], "ranges"), last = integer(range[1], "ranges");
    if (first != static_cast<int64_t>(next) || last < first ||
        last >= static_cast<int64_t>(request.frames.size()))
      throw VideoError("invalid decode request: ranges must cover the frames in order");
    request.ranges.emplace_back(static_cast<size_t>(first), static_cast<size_t>(last));
    next = static_cast<size_t>(last) + 1;
  }
  if (next != request.frames.size())
    throw VideoError("invalid decode request: ranges must cover the frames in order");
  NSArray *keyframes = object[@"keyframes"];
  if (keyframes && ![keyframes isKindOfClass:NSArray.class])
    throw VideoError("invalid decode request: keyframes");
  for (id keyframe in keyframes)
    request.keyframes.push_back(integer(keyframe, "keyframes"));
  std::sort(request.keyframes.begin(), request.keyframes.end());
  return request;
}

// One decoded frame as packed RGB24, upright and resized to the display size
// width x height: the clean aperture (or the track's crop) of the BGRA
// buffer is resized in its coded orientation, then turned.
std::vector<uint8_t> upright(CVPixelBufferRef image, const Video &video, int width,
                             int height) {
  if (CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess)
    throw VideoError("a decoded frame cannot be read");
  struct Unlock {
    CVPixelBufferRef image;
    ~Unlock() { CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly); }
  } unlock{image};
  if (CVPixelBufferGetPixelFormatType(image) != kCVPixelFormatType_32BGRA)
    throw VideoError("a decoded frame is not BGRA");
  const int bufferWidth = static_cast<int>(CVPixelBufferGetWidth(image));
  const int bufferHeight = static_cast<int>(CVPixelBufferGetHeight(image));
  const size_t stride = CVPixelBufferGetBytesPerRow(image);
  const auto *base = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(image));
  if (!base || bufferWidth < 1 || bufferHeight < 1)
    throw VideoError("a decoded frame cannot be read");
  const CGRect clean = CVImageBufferGetCleanRect(image);
  int x0 = static_cast<int>(std::lround(clean.origin.x));
  int y0 = static_cast<int>(std::lround(clean.origin.y));
  int cleanWidth = static_cast<int>(std::lround(clean.size.width));
  int cleanHeight = static_cast<int>(std::lround(clean.size.height));
  if (cleanWidth < 1 || cleanHeight < 1 || x0 < 0 || y0 < 0 ||
      x0 + cleanWidth > bufferWidth || y0 + cleanHeight > bufferHeight) {
    x0 = y0 = 0;
    cleanWidth = bufferWidth;
    cleanHeight = bufferHeight;
  }
  const Orientation &o = video.orientation;
  const int codedWidth = o.swapsAxes() ? height : width;
  const int codedHeight = o.swapsAxes() ? width : height;
  // Pillow resizes the upright frame along its x first; a quarter turn makes
  // that the coded frame's y.
  const std::vector<uint8_t> resized =
      resizeBicubic(base + static_cast<size_t>(y0) * stride + static_cast<size_t>(x0) * 4,
                    cleanWidth, cleanHeight, stride, 4, codedWidth, codedHeight,
                    o.swapsAxes());
  // display = coded * M, shifted to start at 0: a pixel's centre maps to
  // a pixel's centre, so the inverse, M's transpose, reads whole pixels.
  const int shiftX = std::min(0, o.a * codedWidth) + std::min(0, o.c * codedHeight);
  const int shiftY = std::min(0, o.b * codedWidth) + std::min(0, o.d * codedHeight);
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Twice the centre's coordinates keep the arithmetic in integers.
      const int dx = 2 * (x + shiftX) + 1, dy = 2 * (y + shiftY) + 1;
      const int cx = (o.a * dx + o.b * dy - 1) / 2, cy = (o.c * dx + o.d * dy - 1) / 2;
      const uint8_t *bgra = &resized[(static_cast<size_t>(cy) * codedWidth + cx) * 4];
      uint8_t *out = &rgb[(static_cast<size_t>(y) * width + x) * 3];
      out[0] = bgra[2];
      out[1] = bgra[1];
      out[2] = bgra[0];
    }
  }
  return rgb;
}

// Ends the process once the hardware decoder has returned no frame for
// kStallSeconds: cancelReading does not return a reader stalled inside it.
class Watchdog final {
public:
  Watchdog() : last_(std::make_shared<std::atomic<double>>(CACurrentMediaTime())) {
    timer_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                    dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    std::shared_ptr<std::atomic<double>> last = last_;
    dispatch_source_set_timer(timer_, dispatch_time(DISPATCH_TIME_NOW, 250 * NSEC_PER_MSEC),
                              250 * NSEC_PER_MSEC, 50 * NSEC_PER_MSEC);
    dispatch_source_set_event_handler(timer_, ^{
      if (CACurrentMediaTime() - last->load() <= kStallSeconds)
        return;
      static constexpr char message[] = "the hardware video decoder stopped returning frames\n";
      (void)::write(STDERR_FILENO, message, sizeof message - 1);
      _exit(kStalled);
    });
    dispatch_activate(timer_);
  }
  ~Watchdog() { dispatch_source_cancel(timer_); }
  Watchdog(const Watchdog &) = delete;
  Watchdog &operator=(const Watchdog &) = delete;

  void progress() noexcept { last_->store(CACurrentMediaTime()); }

private:
  std::shared_ptr<std::atomic<double>> last_;
  dispatch_source_t timer_;
};

std::string missing(const Request &request, size_t index) {
  return "a frame at " + std::to_string(request.seconds(index)) + " s could not be decoded";
}

// AVFoundation decodes, on Apple's hardware decoder for the codecs it has
// one for. VideoToolbox converts with the stream's own matrix and range, and
// no colour management, as FFmpeg's YUV to RGB conversion does.
int decodeHardware(const Request &request, const Video &video, int width, int height) {
  NSError *error = nil;
  AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:video.asset error:&error];
  if (!reader)
    throw VideoError("the video cannot be read");
  NSDictionary *settings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
  AVAssetReaderTrackOutput *output =
      [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:video.track
                                                 outputSettings:settings];
  output.alwaysCopiesSampleData = NO;
  output.supportsRandomAccess = YES;
  [reader addOutput:output];
  const auto range = [&](size_t index) {
    const auto [first, last] = request.ranges[index];
    return CMTimeRangeFromTimeToTime(
        CMTimeMake(request.frames[first], request.timescale),
        CMTimeMake(request.frames[last] + 1, request.timescale));
  };
  reader.timeRange = range(0);
  Watchdog watchdog;
  if (![reader startReading])
    throw VideoError(reader.error.code == AVErrorDecoderNotFound ||
                             reader.error.code == AVErrorDecoderTemporarilyUnavailable ||
                             reader.error.code == AVErrorFormatUnsupported
                         ? "the video's codec " + video.codec + " has no decoder on this Mac"
                         : "the video cannot be read");
  size_t next = 0, current = 0;
  while (next < request.frames.size()) {
    CMSampleBufferRef buffer = [output copyNextSampleBuffer];
    if (!buffer) {
      // A reader that fails has reached a part of the file it cannot
      // decode, as at the cut of one that ends early: the frames before it
      // are returned.
      if (reader.status != AVAssetReaderStatusReading)
        break;
      // Frames the range should have held and did not.
      if (next <= request.ranges[current].second)
        throw VideoError(missing(request, next));
      // One range at a time: a reset to several ranges at once stops
      // returning frames after the first (macOS 26).
      ++current;
      [output resetForReadingTimeRanges:@[ [NSValue valueWithCMTimeRange:range(current)] ]];
      continue;
    }
    const Owned<CMSampleBufferRef> owned(buffer);
    CVImageBufferRef image = CMSampleBufferGetImageBuffer(buffer);
    if (!image)
      continue;
    watchdog.progress();
    const double time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer));
    if (time < request.seconds(next) - kTimeTolerance)
      continue;
    if (time > request.seconds(next) + kTimeTolerance)
      throw VideoError(missing(request, next));
    const std::vector<uint8_t> frame = upright(image, video, width, height);
    writeAll(frame.data(), frame.size());
    ++next;
  }
  [reader cancelReading];
  return 0;
}

// A track's compressed samples between two times, in decoding order, from
// the keyframe at or before the first; nullptr at the end. ended() tells
// whether the file ended early or is damaged where reading stopped.
class Samples final {
public:
  Samples(const Video &video, CMTimeRange range) {
    NSError *error = nil;
    reader_ = [AVAssetReader assetReaderWithAsset:video.asset error:&error];
    if (!reader_)
      throw VideoError("the video cannot be read");
    output_ = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:video.track
                                                         outputSettings:nil];
    output_.alwaysCopiesSampleData = NO;
    [reader_ addOutput:output_];
    reader_.timeRange = range;
    if (![reader_ startReading])
      throw VideoError("the video cannot be read");
  }
  ~Samples() { [reader_ cancelReading]; }

  Owned<CMSampleBufferRef> next() {
    while (CMSampleBufferRef buffer = [output_ copyNextSampleBuffer]) {
      if (CMSampleBufferGetNumSamples(buffer) > 0)
        return Owned<CMSampleBufferRef>(buffer);
      CFRelease(buffer);
    }
    return {};
  }
  [[nodiscard]] bool ended() const { return reader_.status == AVAssetReaderStatusFailed; }

private:
  AVAssetReader *reader_ = nil;
  AVAssetReaderTrackOutput *output_ = nil;
};

// VideoToolbox's software decoder on the compressed samples, which needs no
// wired memory. Ranges run from the keyframe at or before their first frame
// to the first keyframe after their last, which the frames between may
// reference (B-frames, and an open GOP's leading pictures); ranges that
// would overlap are read as one, and each is decoded on a core of its own,
// stopping once its frames are out. Frames come in decoding order and are
// written in the order of their times.
int decodeSoftware(const Request &request, const Video &video, int width, int height) {
  const std::vector<int64_t> &keys = request.keyframes;
  struct Span {
    int64_t start;
    std::optional<int64_t> end; // the keyframe after, read too; none to the end
    size_t first, last;
  };
  std::vector<Span> spans;
  for (const auto &[first, last] : request.ranges) {
    const auto before = std::upper_bound(keys.begin(), keys.end(), request.frames[first]);
    const int64_t start = before == keys.begin() ? request.frames[first] : *std::prev(before);
    const auto after = std::upper_bound(keys.begin(), keys.end(), request.frames[last]);
    const std::optional<int64_t> end =
        after == keys.end() ? std::nullopt : std::optional<int64_t>(*after);
    if (!spans.empty() && (!spans.back().end || *spans.back().end >= start)) {
      spans.back().end = end;
      spans.back().last = last;
    } else {
      spans.push_back({start, end, first, last});
    }
  }
  struct Result {
    std::map<size_t, std::vector<uint8_t>> frames;
    std::optional<std::string> failure;
    bool noSoftwareDecoder = false;
    bool ended = false;
  };
  std::vector<Result> results(spans.size());
  NSDictionary *decoder = @{
    (id)kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder : @NO
  };
  // Without an IOSurface, the decoder's buffer is ordinary memory.
  NSDictionary *destination =
      @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
  Result *out = results.data();
  const Span *spanAt = spans.data();
  dispatch_apply(spans.size(), DISPATCH_APPLY_AUTO, ^(size_t s) {
    @autoreleasepool {
      const Span &span = spanAt[s];
      Result &result = out[s];
      try {
        const CMTime start = CMTimeMake(span.start, request.timescale);
        Samples samples(video, span.end ? CMTimeRangeFromTimeToTime(
                                              start, CMTimeMake(*span.end + 1, request.timescale))
                                        : CMTimeRangeMake(start, kCMTimePositiveInfinity));
        VTDecompressionSessionRef session = nullptr;
        Owned<CMFormatDescriptionRef> format;
        struct Close {
          VTDecompressionSessionRef &session;
          ~Close() {
            if (session) {
              VTDecompressionSessionInvalidate(session);
              CFRelease(session);
            }
          }
        } close{session};
        const auto lockHolder = std::make_shared<std::mutex>();
        std::mutex *lock = lockHolder.get();
        std::map<size_t, std::vector<uint8_t>> *frames = &result.frames;
        std::optional<std::string> *failure = &result.failure;
        const size_t wanted = span.last - span.first + 1;
        // The span's frame at a decoded frame's time, which VideoToolbox
        // gives in the track's timeline (the sample's output presentation
        // time, the edit list applied).
        const auto target = [&](CMTime time) -> std::optional<size_t> {
          if (!CMTIME_IS_NUMERIC(time))
            return std::nullopt;
          const double seconds = CMTimeGetSeconds(time);
          const auto begin = request.frames.begin() + static_cast<ptrdiff_t>(span.first);
          const auto end = request.frames.begin() + static_cast<ptrdiff_t>(span.last) + 1;
          const auto at = std::lower_bound(
              begin, end, seconds - kTimeTolerance, [&](int64_t frame, double value) {
                return static_cast<double>(frame) / request.timescale < value;
              });
          if (at == end ||
              static_cast<double>(*at) / request.timescale > seconds + kTimeTolerance)
            return std::nullopt;
          return static_cast<size_t>(at - request.frames.begin());
        };
        while (true) {
          {
            std::lock_guard guard(*lock);
            if (frames->size() == wanted || *failure)
              break;
          }
          const Owned<CMSampleBufferRef> sample = samples.next();
          if (!sample)
            break;
          CMFormatDescriptionRef sampleFormat = CMSampleBufferGetFormatDescription(sample.get());
          if (!session || !CMFormatDescriptionEqual(format.get(), sampleFormat)) {
            if (session) {
              VTDecompressionSessionWaitForAsynchronousFrames(session);
              VTDecompressionSessionInvalidate(session);
              CFRelease(session);
              session = nullptr;
            }
            format.reset(static_cast<CMFormatDescriptionRef>(CFRetain(sampleFormat)));
            if (VTDecompressionSessionCreate(nullptr, format.get(),
                                             (__bridge CFDictionaryRef)decoder,
                                             (__bridge CFDictionaryRef)destination, nullptr,
                                             &session) != noErr)
              throw NoSoftwareDecoder("the video's codec " + video.codec +
                                      " has no software decoder");
          }
          // A sample the decoder rejects, as a damaged one, gives no frame;
          // the frames that need it are reported missing. Decoding runs
          // ahead of the frames coming out, as a pipeline.
          VTDecompressionSessionDecodeFrameWithOutputHandler(
              session, sample.get(), kVTDecodeFrame_EnableAsynchronousDecompression, nullptr,
              ^(OSStatus status, VTDecodeInfoFlags, CVImageBufferRef image, CMTime time,
                CMTime) {
                if (status != noErr || !image)
                  return;
                const std::optional<size_t> index = target(time);
                if (!index)
                  return;
                {
                  std::lock_guard guard(*lock);
                  if (frames->count(*index))
                    return;
                }
                try {
                  std::vector<uint8_t> frame = upright(image, video, width, height);
                  std::lock_guard guard(*lock);
                  (*frames)[*index] = std::move(frame);
                } catch (const std::exception &error) {
                  std::lock_guard guard(*lock);
                  *failure = error.what();
                }
              });
        }
        if (session) {
          VTDecompressionSessionFinishDelayedFrames(session);
          VTDecompressionSessionWaitForAsynchronousFrames(session);
        }
        result.ended = samples.ended();
      } catch (const NoSoftwareDecoder &error) {
        result.failure = error.what();
        result.noSoftwareDecoder = true;
      } catch (const std::exception &error) {
        result.failure = error.what();
      }
    }
  });
  for (size_t s = 0; s < spans.size(); ++s) {
    const Result &result = results[s];
    if (result.noSoftwareDecoder)
      throw NoSoftwareDecoder(*result.failure);
    if (result.failure)
      throw VideoError(*result.failure);
    for (size_t i = spans[s].first; i <= spans[s].last; ++i) {
      const auto frame = result.frames.find(i);
      if (frame == result.frames.end()) {
        // A file that ends early yields the frames before its cut.
        if (result.ended)
          return 0;
        throw VideoError(missing(request, i));
      }
      writeAll(frame->second.data(), frame->second.size());
    }
  }
  return 0;
}

int decode(const char *path, int width, int height, bool software) {
  const Request request = readRequest();
  const Video video = openVideo(path);
  return software ? decodeSoftware(request, video, width, height)
                  : decodeHardware(request, video, width, height);
}

int dimension(const char *text) {
  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno || !end || *end || value < 1 || value > kMaxDimension)
    throw VideoError("frame dimensions must be 1-16384 pixels");
  return static_cast<int>(value);
}

} // namespace

std::vector<uint8_t> resizeBicubic(const uint8_t *source, int width, int height,
                                   size_t stride, int channels, int outWidth,
                                   int outHeight, bool verticalFirst) {
  const Coefficients horizontal = coefficients(width, outWidth);
  const Coefficients vertical = coefficients(height, outHeight);
  // Integer sums are exact in any order, so the passes split rows across
  // cores and add a pixel's channels, and a row's columns, side by side.
  // Pillow sums in 32 bits too, which no weighted 8-bit sum overflows.
  constexpr int32_t half = 1 << (kPrecisionBits - 1);
  // One pass along x: rows [rowFirst, rowLast) of in into packed rows of
  // outWidth pixels.
  const auto alongX = [&](const uint8_t *in, size_t inStride, int rowFirst, int rowLast) {
    std::vector<uint8_t> out(static_cast<size_t>(rowLast - rowFirst) * outWidth * channels);
    uint8_t *output = out.data();
    const Coefficients *h = &horizontal;
    dispatch_apply(static_cast<size_t>(rowLast - rowFirst), DISPATCH_APPLY_AUTO, ^(size_t r) {
      const uint8_t *row = in + (rowFirst + r) * inStride;
      uint8_t *result = output + r * outWidth * channels;
      int32_t sums[4];
      for (int xx = 0; xx < outWidth; ++xx) {
        const int xmin = h->bounds[xx * 2], xmax = h->bounds[xx * 2 + 1];
        const int32_t *k = &h->weights[static_cast<size_t>(xx) * h->ksize];
        const uint8_t *pixel = row + static_cast<size_t>(xmin) * channels;
        for (int c = 0; c < channels; ++c)
          sums[c] = half;
        for (int x = 0; x < xmax; ++x, pixel += channels)
          for (int c = 0; c < channels; ++c)
            sums[c] += pixel[c] * k[x];
        for (int c = 0; c < channels; ++c)
          result[xx * channels + c] = clip8(sums[c]);
      }
    });
    return out;
  };
  // One pass along y: in holds rows from rowOffset on, of columns pixels;
  // the result is outHeight packed rows.
  const auto alongY = [&](const uint8_t *in, size_t inStride, int rowOffset, int columns) {
    std::vector<uint8_t> out(static_cast<size_t>(outHeight) * columns * channels);
    uint8_t *output = out.data();
    const Coefficients *v = &vertical;
    const size_t values = static_cast<size_t>(columns) * channels;
    dispatch_apply(static_cast<size_t>(outHeight), DISPATCH_APPLY_AUTO, ^(size_t yy) {
      const int ymin = v->bounds[yy * 2] - rowOffset, ymax = v->bounds[yy * 2 + 1];
      const int32_t *k = &v->weights[yy * v->ksize];
      std::vector<int32_t> sums(values, half);
      for (int y = 0; y < ymax; ++y) {
        const uint8_t *row = in + static_cast<size_t>(y + ymin) * inStride;
        const int32_t weight = k[y];
        for (size_t i = 0; i < values; ++i)
          sums[i] += row[i] * weight;
      }
      uint8_t *result = output + yy * values;
      for (size_t i = 0; i < values; ++i)
        result[i] = clip8(sums[i]);
    });
    return out;
  };
  const auto packed = [&](const uint8_t *in, size_t inStride, int columns, int rows) {
    std::vector<uint8_t> out(static_cast<size_t>(columns) * rows * channels);
    for (int y = 0; y < rows; ++y)
      std::copy_n(in + static_cast<size_t>(y) * inStride, static_cast<size_t>(columns) * channels,
                  &out[static_cast<size_t>(y) * columns * channels]);
    return out;
  };
  const bool resizeWidth = outWidth != width, resizeHeight = outHeight != height;
  if (!resizeWidth && !resizeHeight)
    return packed(source, stride, width, height);
  if (!resizeWidth)
    return alongY(source, stride, 0, width);
  if (!resizeHeight)
    return alongX(source, stride, 0, height);
  if (verticalFirst) {
    const std::vector<uint8_t> rows = alongY(source, stride, 0, width);
    return alongX(rows.data(), static_cast<size_t>(width) * channels, 0, outHeight);
  }
  // Only the source rows the vertical pass reads.
  const int first = vertical.bounds[0];
  const int last = vertical.bounds[outHeight * 2 - 2] + vertical.bounds[outHeight * 2 - 1];
  const std::vector<uint8_t> rows = alongX(source, stride, first, last);
  return alongY(rows.data(), static_cast<size_t>(outWidth) * channels, first, outWidth);
}

int runVideoFrames(int argc, char **argv) {
  // argv: splash video-frames probe FILE | decode FILE WIDTH HEIGHT
  // [--software]. The process runs with no core files; its supervisor
  // bounds its time, memory and output.
  const rlimit noCore{0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
  // VP9's decoder is one macOS loads only when asked, and AV1's where the
  // Mac has one.
  VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_VP9);
  VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_AV1);
  const std::string_view mode = argc > 2 ? argv[2] : "";
  try {
    @autoreleasepool {
      if (mode == "probe" && argc == 4)
        return probe(argv[3]);
      const bool software = argc == 7 && std::string_view(argv[6]) == "--software";
      if (mode == "decode" && (argc == 6 || software))
        return decode(argv[3], dimension(argv[4]), dimension(argv[5]), software);
    }
  } catch (const NoSoftwareDecoder &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return kNoSoftwareDecoder;
  } catch (const VideoError &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EX_DATAERR;
  }
  std::fprintf(stderr,
               "usage: %s video-frames probe FILE | decode FILE WIDTH HEIGHT [--software]\n",
               argc > 0 ? argv[0] : "splash");
  return EX_USAGE;
}

} // namespace splash::video
