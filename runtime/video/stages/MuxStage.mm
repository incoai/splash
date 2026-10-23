#include "video/stages/MuxStage.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace splash::video {
namespace {

constexpr int32_t kFps = 24, kSampleRate = 32000, kAudioBitrate = 192000;
constexpr int64_t kAudioChunk = 8192;  // frames per PCM sample buffer
// Bits per pixel and frame of the H.264 track (19.8 Mb/s at 1344 x 768): at least the quality of the former libx264
// CRF 18 mux on the golden clips, decoded back (mean / min PSNR against the uint8 frames):
//   3 s:  libx264 CRF 18 42.55 / 41.96 dB; VideoToolbox 13 / 16 Mb/s 42.59 / 41.65, 42.78 / 41.93 dB
//   15 s: libx264 CRF 18 41.40 / 40.40 dB; VideoToolbox 13 / 16 / 20 / 24 Mb/s 40.88 / 39.84, 41.27 / 40.27,
//         41.52 / 40.54, 41.73 / 40.88 dB
// (4:2:0 chroma bounds it near 43 dB.)
constexpr double kBitsPerPixel = 0.8;

void fillBgra(CVPixelBufferRef pb, const uint8_t *rgb, int32_t width, int32_t height) {
  CVPixelBufferLockBaseAddress(pb, 0);
  uint8_t *base = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(pb));
  const size_t stride = CVPixelBufferGetBytesPerRow(pb);
  dispatch_apply(static_cast<size_t>(height), DISPATCH_APPLY_AUTO, ^(size_t y) {
    const uint8_t *s = rgb + y * size_t(width) * 3;
    uint8_t *d = base + y * stride;
    for (int32_t x = 0; x < width; ++x, s += 3, d += 4) {
      d[0] = s[2];
      d[1] = s[1];
      d[2] = s[0];
      d[3] = 255;
    }
  });
  CVPixelBufferUnlockBaseAddress(pb, 0);
}

// rgb: [frames][height][width][3]; wav (nullable): [2][samples] fp32 in [-1, 1] at 32 kHz.
void writeMp4(const std::string &path, const uint8_t *rgb, int32_t frames, int32_t width, int32_t height,
              const float *wav, int64_t samples) {
  if (!rgb || frames < 1 || width < 16 || height < 16 || (width & 1) || (height & 1))
    throw std::invalid_argument("mp4: bad frames (" + std::to_string(frames) + " x " + std::to_string(width) + " x " +
                                std::to_string(height) + ")");
  const auto bitrate = static_cast<int32_t>(kBitsPerPixel * width * height * kFps);
  @autoreleasepool {
    NSURL *url = [NSURL fileURLWithPath:@(path.c_str())];
    [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
    NSError *err = nil;
    AVAssetWriter *writer = [AVAssetWriter assetWriterWithURL:url fileType:AVFileTypeMPEG4 error:&err];
    if (!writer) throw std::runtime_error("mp4 " + path + ": " + err.localizedDescription.UTF8String);
    NSDictionary *compression = @{
      AVVideoAverageBitRateKey : @(bitrate),
      AVVideoProfileLevelKey : AVVideoProfileLevelH264HighAutoLevel,
      AVVideoH264EntropyModeKey : AVVideoH264EntropyModeCABAC,
      AVVideoExpectedSourceFrameRateKey : @(kFps),
      AVVideoMaxKeyFrameIntervalKey : @(2 * kFps),
      AVVideoAllowFrameReorderingKey : @YES,
    };
    NSDictionary *videoSettings = @{
      AVVideoCodecKey : AVVideoCodecTypeH264,
      AVVideoWidthKey : @(width),
      AVVideoHeightKey : @(height),
      AVVideoCompressionPropertiesKey : compression,
      AVVideoColorPropertiesKey : @{
        AVVideoColorPrimariesKey : AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey : AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey : AVVideoYCbCrMatrix_ITU_R_709_2,
      },
    };
    AVAssetWriterInput *vin = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                                  outputSettings:videoSettings];
    vin.expectsMediaDataInRealTime = NO;
    NSDictionary *pixels = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
      (id)kCVPixelBufferWidthKey : @(width),
      (id)kCVPixelBufferHeightKey : @(height),
    };
    AVAssetWriterInputPixelBufferAdaptor *adaptor =
        [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:vin
                                                                         sourcePixelBufferAttributes:pixels];
    if (![writer canAddInput:vin]) throw std::runtime_error("mp4: H.264 input rejected");
    [writer addInput:vin];
    AVAssetWriterInput *ain = nil;
    CMAudioFormatDescriptionRef pcm = nullptr;
    AudioChannelLayout layout{};
    layout.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo;
    if (wav && samples > 0) {
      NSDictionary *audioSettings = @{
        AVFormatIDKey : @(kAudioFormatMPEG4AAC),
        AVSampleRateKey : @(kSampleRate),
        AVNumberOfChannelsKey : @2,
        AVEncoderBitRateKey : @(kAudioBitrate),
        AVChannelLayoutKey : [NSData dataWithBytes:&layout length:sizeof layout],
      };
      ain = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeAudio outputSettings:audioSettings];
      ain.expectsMediaDataInRealTime = NO;
      if (![writer canAddInput:ain]) throw std::runtime_error("mp4: AAC input rejected");
      [writer addInput:ain];
      AudioStreamBasicDescription asbd{};
      asbd.mSampleRate = kSampleRate;
      asbd.mFormatID = kAudioFormatLinearPCM;
      asbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
      asbd.mBytesPerPacket = 8;
      asbd.mFramesPerPacket = 1;
      asbd.mBytesPerFrame = 8;
      asbd.mChannelsPerFrame = 2;
      asbd.mBitsPerChannel = 32;
      if (CMAudioFormatDescriptionCreate(kCFAllocatorDefault, &asbd, sizeof layout, &layout, 0, nullptr, nullptr,
                                         &pcm) != noErr)
        throw std::runtime_error("mp4: PCM format description failed");
    }
    if (![writer startWriting])
      throw std::runtime_error("mp4 " + path + ": " + writer.error.localizedDescription.UTF8String);
    [writer startSessionAtSourceTime:kCMTimeZero];
    dispatch_group_t group = dispatch_group_create();
    auto failed = std::make_shared<std::atomic<bool>>(false);
    __block int32_t nextFrame = 0;
    dispatch_group_enter(group);
    dispatch_queue_t vq = dispatch_queue_create("h3.mp4.video", DISPATCH_QUEUE_SERIAL);
    [vin requestMediaDataWhenReadyOnQueue:vq
                               usingBlock:^{
                                 while (vin.readyForMoreMediaData && nextFrame < frames) {
                                   CVPixelBufferRef pb = nullptr;
                                   if (CVPixelBufferPoolCreatePixelBuffer(nullptr, adaptor.pixelBufferPool, &pb) !=
                                           kCVReturnSuccess ||
                                       !pb) {
                                     *failed = true;
                                     nextFrame = frames;
                                     break;
                                   }
                                   fillBgra(pb, rgb + size_t(nextFrame) * width * height * 3, width, height);
                                   if (![adaptor appendPixelBuffer:pb withPresentationTime:CMTimeMake(nextFrame, kFps)])
                                     *failed = true;
                                   CVPixelBufferRelease(pb);
                                   ++nextFrame;
                                 }
                                 if (nextFrame >= frames) {
                                   nextFrame = frames + 1;  // finish once
                                   [vin markAsFinished];
                                   dispatch_group_leave(group);
                                 }
                               }];
    __block int64_t nextSample = 0;
    if (ain) {
      dispatch_group_enter(group);
      dispatch_queue_t aq = dispatch_queue_create("h3.mp4.audio", DISPATCH_QUEUE_SERIAL);
      __block std::vector<float> interleaved(size_t(kAudioChunk) * 2);
      [ain requestMediaDataWhenReadyOnQueue:aq
                                 usingBlock:^{
                                   while (ain.readyForMoreMediaData && nextSample < samples) {
                                     const int64_t n = std::min<int64_t>(kAudioChunk, samples - nextSample);
                                     for (int64_t i = 0; i < n; ++i) {
                                       interleaved[2 * i] = wav[nextSample + i];
                                       interleaved[2 * i + 1] = wav[samples + nextSample + i];
                                     }
                                     CMBlockBufferRef block = nullptr;
                                     CMSampleBufferRef sb = nullptr;
                                     const size_t bytes = size_t(n) * 8;
                                     OSStatus st = CMBlockBufferCreateWithMemoryBlock(
                                         kCFAllocatorDefault, nullptr, bytes, kCFAllocatorDefault, nullptr, 0, bytes,
                                         kCMBlockBufferAssureMemoryNowFlag, &block);
                                     if (st == noErr) st = CMBlockBufferReplaceDataBytes(interleaved.data(), block, 0, bytes);
                                     if (st == noErr)
                                       st = CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                                           kCFAllocatorDefault, block, pcm, n, CMTimeMake(nextSample, kSampleRate),
                                           nullptr, &sb);
                                     if (st != noErr || ![ain appendSampleBuffer:sb]) *failed = true;
                                     if (sb) CFRelease(sb);
                                     if (block) CFRelease(block);
                                     nextSample += n;
                                     if (*failed) nextSample = samples;
                                   }
                                   if (nextSample >= samples) {
                                     nextSample = samples + 1;
                                     [ain markAsFinished];
                                     dispatch_group_leave(group);
                                   }
                                 }];
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [writer finishWritingWithCompletionHandler:^{
      dispatch_semaphore_signal(done);
    }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (pcm) CFRelease(pcm);
    if (*failed || writer.status != AVAssetWriterStatusCompleted)
      throw std::runtime_error("mp4 " + path + ": writing failed: " +
                               (writer.error ? writer.error.localizedDescription.UTF8String : "append rejected"));
  }
}

} // namespace

std::vector<uint8_t> muxMp4(std::span<const uint8_t> frames, int32_t width, int32_t height, std::span<const float> wav,
                            uint64_t requestId) {
  // AVAssetWriter writes a file: the request's own, read back and removed.
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("splash-video-" + std::to_string(getpid()) + "-" + std::to_string(requestId) +
                                      ".mp4");
  struct Removal {
    const std::filesystem::path &path;
    ~Removal() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } removal{path};
  const size_t frameBytes = size_t(width) * height * 3;
  writeMp4(path.string(), frames.data(), frameBytes ? static_cast<int32_t>(frames.size() / frameBytes) : 0, width,
           height, wav.empty() ? nullptr : wav.data(), static_cast<int64_t>(wav.size() / 2));
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.empty()) throw std::runtime_error("cannot write the mp4");
  return bytes;
}

void MuxStage::forward(VideoBatch &batch, StageEvents &) {
  batch.mp4 = muxMp4(batch.frames, batch.geometry.width, batch.geometry.height, batch.wav, batch.requestId);
}

} // namespace splash::video
