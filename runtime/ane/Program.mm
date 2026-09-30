#import "ane/Program.hpp"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <sys/qos.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

// The private AppleNeuralEngine interface this file uses.
@protocol SplashAneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
- (NSDictionary *)modelAttributes;
@end
@protocol SplashAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos
                    error:(NSError **)error;
@end
@protocol SplashAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol SplashAneRequest
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs
          outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber *)procedure sharedEvents:(id)events transactionHandle:(NSNumber *)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end
@protocol SplashAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
@end

namespace splash::ane {
namespace {

constexpr unsigned kQos = QOS_CLASS_DEFAULT;

Class requireClass(const char *name) {
  static const bool loaded =
      dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
  Class result = loaded ? NSClassFromString(@(name)) : nil;
  if (!result) throw std::runtime_error(std::string("AppleNeuralEngine does not provide ") + name);
  return result;
}

[[noreturn]] void fail(const char *what, NSError *error) {
  throw std::runtime_error(std::string("ANE ") + what + " failed" +
                           (error ? std::string(": ") + error.description.UTF8String : std::string()));
}

id<SplashAneClient> client() { return [(Class<SplashAneClient>)requireClass("_ANEClient") sharedConnection]; }

uint32_t elementBytes(Surface::Element element) noexcept { return element == Surface::Element::Int8 ? 1 : 2; }
uint64_t rowBytes(uint32_t width, Surface::Element element) noexcept {
  return (uint64_t{width} * elementBytes(element) + 63) / 64 * 64;
}

} // namespace

uint64_t Surface::bytes(uint32_t rows, uint32_t width, Element element) noexcept {
  return (rowBytes(width, element) * rows + 16383) / 16384 * 16384;
}

Surface Surface::create(metal::MetalBackend &backend, uint32_t rows, uint32_t width, Element element) {
  const uint64_t stride = rowBytes(width, element), size = bytes(rows, width, element);
  NSDictionary *properties = @{
    (id)kIOSurfaceWidth : @(width),
    (id)kIOSurfaceHeight : @(rows),
    (id)kIOSurfaceBytesPerElement : @(elementBytes(element)),
    (id)kIOSurfaceBytesPerRow : @(stride),
    (id)kIOSurfaceAllocSize : @(size),
    (id)kIOSurfacePixelFormat : @(element == Element::Int8 ? 0x4c303038 : 0x4c303068), // 'L008', 'L00h'
  };
  IOSurfaceRef created = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
  if (!created) throw std::runtime_error("IOSurface creation failed");
  std::shared_ptr<void> owner(created, [](void *surface) { CFRelease(surface); });
  Surface result;
  result.buffer = backend.wrapSharedMemory(IOSurfaceGetBaseAddress(created), IOSurfaceGetAllocSize(created), owner,
                                           "ane surface");
  result.surface = std::move(owner);
  result.strideBytes = static_cast<uint32_t>(IOSurfaceGetBytesPerRow(created));
  return result;
}

struct Program::Impl {
  std::filesystem::path directory;
  __strong id model = nil;
  std::vector<std::string> inputs;
  bool loaded = false;

  ~Impl() {
    if (loaded) [client() unloadModel:model options:@{} qos:kQos error:nil];
    std::error_code ignored;
    if (!directory.empty()) std::filesystem::remove_all(directory, ignored);
  }
};

Program::Program(std::string_view mil, std::span<const uint8_t> weights) : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    std::string pattern = (std::filesystem::temp_directory_path() / "splash-ane-XXXXXX").string();
    if (!mkdtemp(pattern.data())) throw std::runtime_error("unable to create the ANE program directory");
    impl_->directory = pattern;
    const auto write = [&](const char *name, const void *data, size_t bytes) {
      std::ofstream file(impl_->directory / name, std::ios::binary);
      file.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
      if (!file) throw std::runtime_error(std::string("unable to write the ANE program's ") + name);
    };
    write("model.mil", mil.data(), mil.size());
    write("weights.bin", weights.data(), weights.size());

    NSString *path = @(impl_->directory.c_str());
    impl_->model = [(Class<SplashAneModel>)requireClass("_ANEModel") modelAtURL:[NSURL fileURLWithPath:path
                                                                                           isDirectory:YES]
                                                                            key:path];
    if (!impl_->model) fail("model creation", nil);
    NSError *error = nil;
    if (![client() compileModel:impl_->model
                        options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"}
                            qos:kQos
                          error:&error])
      fail("compilation", error);
    if (![client() loadModel:impl_->model options:@{} qos:kQos error:&error]) fail("load", error);
    impl_->loaded = true;
    NSArray *symbols =
        [impl_->model modelAttributes][@"ANEFModelDescription"][@"kANEFModelInputSymbolsArrayKey"];
    for (NSString *symbol in symbols) impl_->inputs.emplace_back(symbol.UTF8String);
  }
}

Program::~Program() = default;

const std::vector<std::string> &Program::inputs() const noexcept { return impl_->inputs; }

void Program::enqueue(std::span<const Surface> inputs, const Surface &output, const metal::SharedEvent &event,
                      uint64_t wait, uint64_t signal, std::function<void(bool)> done) {
  if (inputs.size() != impl_->inputs.size()) throw std::invalid_argument("ANE program input count mismatch");
  @autoreleasepool {
    Class<SplashAneSurface> surfaces = (Class<SplashAneSurface>)requireClass("_ANEIOSurfaceObject");
    NSMutableArray *objects = [NSMutableArray arrayWithCapacity:inputs.size()];
    NSMutableArray *indices = [NSMutableArray arrayWithCapacity:inputs.size()];
    for (size_t index = 0; index < inputs.size(); ++index) {
      [objects addObject:[surfaces objectWithIOSurface:(IOSurfaceRef)inputs[index].surface.get()]];
      [indices addObject:@(index)];
    }
    id native = (__bridge id)event.nativeHandle();
    id events = [(Class<SplashAneEvents>)requireClass("_ANESharedEvents")
        sharedEventsWithSignalEvents:@[ [(Class<SplashAneEvents>)requireClass("_ANESharedSignalEvent")
                                         signalEventWithValue:signal
                                                  symbolIndex:0
                                                    eventType:0
                                                  sharedEvent:native] ]
                          waitEvents:@[ [(Class<SplashAneEvents>)requireClass("_ANESharedWaitEvent")
                                         waitEventWithValue:wait
                                                sharedEvent:native
                                                  eventType:0] ]];
    id request = [(Class<SplashAneRequest>)requireClass("_ANERequest")
        requestWithInputs:objects
             inputIndices:indices
                  outputs:@[ [surfaces objectWithIOSurface:(IOSurfaceRef)output.surface.get()] ]
            outputIndices:@[ @0 ]
            weightsBuffer:nil
                perfStats:nil
           procedureIndex:@0
             sharedEvents:events
        transactionHandle:nil];
    if (!request) fail("request creation", nil);
    // An evaluation with shared events runs asynchronously and requires a
    // completion handler.
    metal::SharedEvent retained = event;
    [request setCompletionHandler:^(BOOL success, NSError *) {
      if (!success) release(retained, signal);
      done(success);
    }];
    NSError *error = nil;
    if (![client() evaluateWithModel:impl_->model options:@{} request:request qos:kQos error:&error])
      fail("evaluation", error);
  }
}

void Program::release(const metal::SharedEvent &event, uint64_t value) noexcept {
  id<MTLSharedEvent> native = (__bridge id<MTLSharedEvent>)event.nativeHandle();
  if (native.signaledValue < value) native.signaledValue = value;
}

} // namespace splash::ane
