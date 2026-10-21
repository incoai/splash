#import "ane/Program.hpp"

#include "Checked.hpp"
#ifdef SPLASH_ANE_INSTRUMENTATION
#include "ane/ProgramInstrumentation.hpp"
#endif

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <sys/qos.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <variant>

// The private AppleNeuralEngine interface this file uses.
@protocol SplashAneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
- (NSDictionary *)modelAttributes;
@end
@protocol SplashAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
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

// Each method declared above as the framework must define it: its class, '+'
// for a class method or '-' for an instance method, its selector, and the
// type encoding its declaration implies on arm64, where BOOL is B, without
// the frame offsets. A method whose encoding differs takes or returns other
// types than declared, which a message sent as declared would get wrong.
struct Signature final {
  const char *owner;
  char kind;
  const char *selector;
  const char *types;
};
constexpr Signature kSignatures[] = {
    {"_ANEModel", '+', "modelAtURL:key:", "@@:@@"},
    {"_ANEModel", '-', "modelAttributes", "@@:"},
    {"_ANEClient", '+', "sharedConnection", "@@:"},
    {"_ANEClient", '-', "compileModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "compiledModelExistsFor:", "B@:@"},
    {"_ANEClient", '-', "purgeCompiledModel:", "v@:@"},
    {"_ANEClient", '-', "loadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "unloadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "evaluateWithModel:options:request:qos:error:", "B@:@@@I^@"},
    {"_ANEIOSurfaceObject", '+', "objectWithIOSurface:", "@@:^{__IOSurface=}"},
    {"_ANERequest", '+',
     "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"
     "transactionHandle:",
     "@@:@@@@@@@@@"},
    {"_ANERequest", '-', "setCompletionHandler:", "v@:@?"},
    {"_ANESharedWaitEvent", '+', "waitEventWithValue:sharedEvent:eventType:", "@@:Q@Q"},
    {"_ANESharedSignalEvent", '+', "signalEventWithValue:symbolIndex:eventType:sharedEvent:", "@@:QIq@"},
    {"_ANESharedEvents", '+', "sharedEventsWithSignalEvents:waitEvents:", "@@:@@"},
};

constexpr unsigned kQos = QOS_CLASS_DEFAULT;
// How often a caller waiting for the service asks whether to stop.
constexpr auto kWaitSlice = std::chrono::milliseconds(50);

[[noreturn]] void fail(const char *what, NSError *error) {
  throw std::runtime_error(std::string("ANE ") + what + " failed" +
                           (error ? std::string(": ") + error.description.UTF8String : std::string()));
}

// Runs `body`, which messages the private interface, and throws an
// Objective-C exception it raises as a std::runtime_error naming `step`:
// std::exception handlers do not catch an NSException.
template <class F> decltype(auto) guarded(const char *step, F &&body) {
  @try {
    return std::forward<F>(body)();
  } @catch (NSException *exception) {
    throw std::runtime_error(
        [NSString stringWithFormat:@"ANE %s raised %@: %@", step, exception.name, exception.reason].UTF8String);
  }
}

// The private interface's classes, the service's client and the queue the
// service's work for every Program runs on, one block at a time: a block the
// service never answers holds those queued after it.
struct Api final {
  __strong id<SplashAneClient> client = nil;
  Class<SplashAneModel> model = nil;
  Class<SplashAneSurface> surface = nil;
  Class<SplashAneEvents> events = nil, signalEvent = nil, waitEvent = nil;
  Class<SplashAneRequest> request = nil;
  __strong dispatch_queue_t queue = nil;
};

// +[class selector] or -[class selector], as a method is named.
std::string methodName(const Signature &signature) {
  return std::string(1, signature.kind) + "[" + signature.owner + " " + signature.selector + "]";
}

// Checks that the framework has each method of `signatures` with its type
// encoding. Throws "AppleNeuralEngine lacks" a class or method, or "changed"
// a method, at the first that fails.
void check(std::span<const Signature> signatures) {
  for (const Signature &signature : signatures) {
    Class owner = NSClassFromString(@(signature.owner));
    if (!owner) throw std::runtime_error(std::string("AppleNeuralEngine lacks ") + signature.owner);
    const SEL selector = sel_registerName(signature.selector);
    Method method =
        signature.kind == '+' ? class_getClassMethod(owner, selector) : class_getInstanceMethod(owner, selector);
    if (!method) throw std::runtime_error("AppleNeuralEngine lacks " + methodName(signature));
    const char *encoding = method_getTypeEncoding(method);
    std::string types = encoding ? encoding : "";
    std::erase_if(types, [](char c) { return c >= '0' && c <= '9'; });
    if (types != signature.types) throw std::runtime_error("AppleNeuralEngine changed " + methodName(signature));
  }
}

// Loads the framework, checks its methods and finds its classes and the
// shared client. Throws at the first failure.
Api resolve() {
  if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
    const char *reason = dlerror();
    throw std::runtime_error(std::string("AppleNeuralEngine does not load: ") + (reason ? reason : "no reason"));
  }
  check(kSignatures);
  const auto named = [](const char *name) { return NSClassFromString(@(name)); };
  Api api;
  api.model = named("_ANEModel");
  api.surface = named("_ANEIOSurfaceObject");
  api.events = named("_ANESharedEvents");
  api.signalEvent = named("_ANESharedSignalEvent");
  api.waitEvent = named("_ANESharedWaitEvent");
  api.request = named("_ANERequest");
  Class client = named("_ANEClient");
  id shared = [(Class<SplashAneClient>)client sharedConnection];
  if (![shared isKindOfClass:client]) throw std::runtime_error("AppleNeuralEngine lacks a shared connection");
  api.client = shared;
  api.queue = dispatch_queue_create("splash.ane", DISPATCH_QUEUE_SERIAL);
  return api;
}

// The private interface, resolved on first use. A resolution that fails is
// not retried: each use throws its message. Never destroyed: work on the
// service's queue may use it while the process exits.
const Api &api() {
  static const auto *const resolved = new std::variant<Api, std::string>([]() -> std::variant<Api, std::string> {
    @autoreleasepool {
      try {
        return guarded("interface lookup", resolve);
      } catch (const std::exception &error) {
        return std::string(error.what());
      }
    }
  }());
  if (const std::string *failure = std::get_if<std::string>(resolved)) throw std::runtime_error(*failure);
  return std::get<Api>(*resolved);
}

// What work on the service's queue tells the caller waiting for it.
struct Outcome final {
  std::mutex mutex;
  std::condition_variable changed;
  bool finished = false;
  std::exception_ptr failure;
};

std::string seconds(AwakeClock::duration duration) {
  char text[32];
  std::snprintf(text, sizeof text, "%g", std::chrono::duration<double>(duration).count());
  return text;
}

// Runs `body` on the service's queue and waits for it under `limits`, asking
// limits.interrupted() every kWaitSlice; rethrows what `body` throws. `body`
// holds what it uses: it runs to its end after the caller stops waiting.
void runOnQueue(const Api &api, const Program::Limits &limits, std::function<void()> body) {
  const auto outcome = std::make_shared<Outcome>();
  dispatch_async(api.queue, ^{
    std::exception_ptr failure;
    @autoreleasepool {
      try {
        body();
      } catch (...) {
        failure = std::current_exception();
      }
    }
    std::lock_guard lock(outcome->mutex);
    outcome->finished = true;
    outcome->failure = failure;
    outcome->changed.notify_all();
  });
  const AwakeClock::time_point deadline = AwakeClock::now() + limits.limit;
  std::unique_lock lock(outcome->mutex);
  while (!outcome->finished) {
    lock.unlock();
    const bool interrupted = limits.interrupted && limits.interrupted();
    lock.lock();
    if (outcome->finished) break;
    if (interrupted) throw Interrupted("interrupted while waiting for the Neural Engine");
    const AwakeClock::time_point now = AwakeClock::now();
    if (now >= deadline)
      throw std::runtime_error("the Neural Engine did not answer within " + seconds(limits.limit) + " s");
    outcome->changed.wait_until(lock, std::min(deadline, now + kWaitSlice), [&] { return outcome->finished; });
  }
  if (outcome->failure) std::rethrow_exception(outcome->failure);
}

// Hands an evaluation's report to its `done` once enqueue() has queued it:
// the service may report on its thread before evaluateWithModel returns, or
// report an evaluation enqueue() then throws for, which `done` must not see.
struct Completion final {
  std::mutex mutex;
  std::function<void(bool)> done;
  std::optional<bool> result;
  bool queued = false;

  // Calls `done` once the evaluation is queued and reported, with the first
  // report, and never again.
  void settle(std::unique_lock<std::mutex> lock) {
    if (!queued || !result) return;
    const std::function<void(bool)> call = std::exchange(done, nullptr);
    const bool success = *result;
    lock.unlock();
    if (call) call(success);
  }
};

uint32_t elementBytes(Surface::Element element) noexcept { return element == Surface::Element::Int8 ? 1 : 2; }

// FNV-1a of `bytes`, continuing `hash`, in hexadecimal: the names of the
// cache's files.
constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
uint64_t fnv1a(std::span<const uint8_t> bytes, uint64_t hash = kFnvBasis) {
  for (const uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ULL;
  return hash;
}
std::string hex(uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
  return text;
}
std::span<const uint8_t> bytesOf(std::string_view text) {
  return {reinterpret_cast<const uint8_t *>(text.data()), text.size()};
}

// Whether `file` holds `bytes` and nothing else.
bool holds(const std::filesystem::path &file, std::span<const uint8_t> bytes) {
  std::ifstream in(file, std::ios::binary | std::ios::ate);
  if (!in || static_cast<uint64_t>(in.tellg()) != bytes.size()) return false;
  std::vector<uint8_t> contents(bytes.size());
  in.seekg(0);
  return in.read(reinterpret_cast<char *>(contents.data()), static_cast<std::streamsize>(contents.size())) &&
         std::ranges::equal(contents, bytes);
}

// Writes `file` whole: another process may read it at the same time. A write
// that fails leaves no partial file.
void writeWhole(const std::filesystem::path &file, std::span<const uint8_t> bytes) {
  const std::filesystem::path partial = file.string() + "." + std::to_string(getpid()) + ".partial";
  // Removes the partial file on the way out: once renamed into place, there
  // is none.
  struct Removal final {
    const std::filesystem::path &path;
    ~Removal() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } removal{partial};
  std::ofstream out(partial, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  out.close();
  if (!out) throw std::runtime_error("unable to write " + file.string());
  std::filesystem::rename(partial, file);
}

#ifdef SPLASH_ANE_INSTRUMENTATION
// The faults ProgramInstrumentation::arm() set for the next Program.
struct Armed final {
  std::mutex mutex;
  std::optional<ProgramInstrumentation::Faults> faults;
};
Armed &armed() {
  static Armed value;
  return value;
}

// The armed faults, which the program constructed now takes, or none.
ProgramInstrumentation::Faults takeArmed() {
  std::lock_guard lock(armed().mutex);
  return std::exchange(armed().faults, std::nullopt).value_or(ProgramInstrumentation::Faults{});
}

// kSignatures with the encoding of `method` made one no method has.
std::vector<Signature> changed(std::string_view method) {
  std::vector<Signature> signatures(std::begin(kSignatures), std::end(kSignatures));
  const auto found = std::ranges::find(signatures, method, methodName);
  if (found == signatures.end())
    throw std::invalid_argument("AppleNeuralEngine has no method " + std::string(method) + " to change");
  found->types = "";
  return signatures;
}

// Runs `evaluate`, the sequence-th evaluation, or the fault `faults` puts in
// its place: a failure reported from another thread without running, neither
// a run nor a report, the run faults.delay later, reporting its failure, or
// `poison`.
void evaluateWithFaults(const ProgramInstrumentation::Faults &faults, uint64_t sequence,
                        void (^report)(BOOL, NSError *), const std::function<void()> &evaluate,
                        const std::function<void()> &poison) {
  dispatch_queue_t elsewhere = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
  if (sequence == faults.poisonedEvaluation) {
    poison();
  } else if (sequence == faults.failingEvaluation) {
    dispatch_async(elsewhere, ^{
      report(NO, nil);
    });
  } else if (sequence == faults.delayedEvaluation) {
    const std::function<void()> later = evaluate;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, std::chrono::nanoseconds(faults.delay).count()), elsewhere, ^{
      try {
        later();
      } catch (const std::exception &) {
        report(NO, nil);
      }
    });
  } else if (sequence != faults.stalledEvaluation) {
    evaluate();
  }
}
#endif

// The file recall() reads under `key`, which remember() writes.
std::filesystem::path rememberedFile(std::string_view key) {
  return cacheDirectory() / ("remembered-" + hex(fnv1a(bytesOf(key))));
}

} // namespace

std::filesystem::path cacheDirectory() {
  char path[PATH_MAX];
  const size_t size = confstr(_CS_DARWIN_USER_CACHE_DIR, path, sizeof path);
  if (!size || size > sizeof path) throw std::runtime_error("macOS names no per-user cache directory");
  return std::filesystem::path(path) / "splash-ane";
}

std::optional<std::string> unavailable() {
  try {
    static_cast<void>(api());
  } catch (const std::exception &error) {
    return error.what();
  }
  return std::nullopt;
}

uint32_t Surface::rowBytes(uint32_t width, Element element) noexcept {
  return (width * elementBytes(element) + 63) / 64 * 64;
}

uint64_t Surface::bytes(uint32_t rows, uint32_t width, Element element) noexcept {
  return alignUp(uint64_t{rowBytes(width, element)} * rows);
}

const char *Surface::milType(Element element) noexcept { return element == Element::Int8 ? "int8" : "fp16"; }

std::string Surface::strides(uint32_t rows) const {
  const uint64_t stride = strideBytes / elementBytes(element), plane = uint64_t{rows} * stride;
  return "[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(stride) + ", 1]";
}

std::string Surface::bufferType(uint32_t rows, uint32_t width) const {
  return std::string("tensor_buffer<") + milType(element) + ", shape=[1, 1, " + std::to_string(rows) + ", " +
         std::to_string(width) + "], strides=" + strides(rows) + ", interleave_factors=[1, 1, 1, 1]>";
}

Surface Surface::create(metal::MetalBackend &backend, uint32_t rows, uint32_t width, Element element) {
  const uint32_t stride = rowBytes(width, element);
  const uint64_t size = bytes(rows, width, element);
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
  // Programs and kernels address the rows at the stride asked for.
  if (IOSurfaceGetBytesPerRow(created) != stride || IOSurfaceGetAllocSize(created) != size)
    throw std::runtime_error("IOSurface took another row stride or size");
  Surface result;
  result.buffer = backend.wrapSharedMemory(IOSurfaceGetBaseAddress(created), size, owner, "ane surface");
  result.surface = std::move(owner);
  result.element = element;
  result.strideBytes = stride;
  return result;
}

std::vector<uint8_t> constantBlob(std::span<const _Float16> values) {
  constexpr uint64_t kData = 128;
  std::vector<uint8_t> blob(kData + values.size_bytes());
  const auto put = [&](uint64_t offset, auto value) { std::memcpy(blob.data() + offset, &value, sizeof value); };
  put(0, uint32_t{1}); // blobs
  put(4, uint32_t{2}); // storage version
  put(kConstantOffset, uint32_t{0xdeadbeef}); // the record's sentinel
  put(kConstantOffset + 4, uint32_t{1});      // fp16
  put(kConstantOffset + 8, uint64_t{values.size_bytes()});
  put(kConstantOffset + 16, kData);
  std::memcpy(blob.data() + kData, values.data(), values.size_bytes());
  return blob;
}

// A procedure's function, and its inputs and output as the model's symbols:
// their names, and their indices among all procedures', which a request
// takes.
struct Procedure final {
  std::string function;
  std::vector<std::string> inputs;
  std::vector<uint32_t> inputSymbols;
  uint32_t outputSymbol = 0;

  bool operator==(const Procedure &) const = default;
};

// A program's state, which the work it queues on the service's queue shares:
// that work may outlive the Program, whose last owner deletes this there.
struct Program::Impl {
  explicit Impl(const Api &api) : api(api) {}
  // Runs on the service's queue, as the work that loaded the model did.
  ~Impl() { release(); }

  // guarded() for a step of this program's.
  template <class F> decltype(auto) guarded(const char *step, F &&body);
  // Whether the service holds a compilation of the model.
  [[nodiscard]] bool compiled();
  void compile();
  // Loads the compiled model. When `recompile`, a load that fails purges the
  // compilation, compiles the model again and loads it once more: the
  // service may keep a compilation it cannot load, from another compiler,
  // say.
  void load(bool recompile);
  // The procedures the loaded model's attributes describe.
  [[nodiscard]] std::vector<Procedure> describe();
  void unload();
  // unload(), ignoring a failure, for a program given up.
  void release() noexcept;

  const Api &api;
  // Set by the constructor's work on the service's queue.
  __strong id model = nil;
  // By procedure, set by the constructor's work on the service's queue.
  std::vector<Procedure> procedures;
  // Changed only on the service's queue.
  std::atomic<bool> loaded = false;
#ifdef SPLASH_ANE_INSTRUMENTATION
  ProgramInstrumentation::Faults faults;
  std::atomic<uint64_t> enqueues = 0;
#endif
};

struct Program::Binding::State final {
  // The program's state it was bound against.
  std::weak_ptr<const Program::Impl> program;
  // A request's arguments but its events.
  __strong NSNumber *procedure = nil;
  __strong NSArray *inputs = nil, *inputIndices = nil, *outputs = nil, *outputIndices = nil;
  // The IOSurfaces the service's objects wrap.
  std::vector<std::shared_ptr<void>> surfaces;
};

template <class F> decltype(auto) Program::Impl::guarded(const char *step, F &&body) {
#ifdef SPLASH_ANE_INSTRUMENTATION
  return ane::guarded(step, [&]() -> decltype(auto) {
    if (faults.raiseAt == step) [NSException raise:@"SplashInjectedFault" format:@"an injected fault"];
    return std::forward<F>(body)();
  });
#else
  return ane::guarded(step, std::forward<F>(body));
#endif
}

bool Program::Impl::compiled() {
  return guarded("compiled model lookup", [&] { return [api.client compiledModelExistsFor:model]; });
}

void Program::Impl::compile() {
  guarded("compilation", [&] {
    NSDictionary *options = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
    NSError *error = nil;
    if (![api.client compileModel:model options:options qos:kQos error:&error]) fail("compilation", error);
  });
}

void Program::Impl::load(bool recompile) {
  guarded("load", [&] {
    NSError *error = nil;
    if ([api.client loadModel:model options:@{} qos:kQos error:&error]) return;
    if (!recompile) fail("load", error);
    guarded("purge", [&] { [api.client purgeCompiledModel:model]; });
    compile();
    error = nil;
    if (![api.client loadModel:model options:@{} qos:kQos error:&error]) fail("load", error);
  });
  loaded = true;
}

std::vector<Procedure> Program::Impl::describe() {
  // Each value of the attributes is checked for its kind before it is
  // messaged.
  return guarded("model attributes", [&] {
    const auto undescribed = [] { return std::runtime_error("ANE program does not describe its procedures"); };
    NSDictionary *attributes = [model modelAttributes];
    if (![attributes isKindOfClass:NSDictionary.class]) throw undescribed();
    NSDictionary *description = attributes[@"ANEFModelDescription"];
    if (![description isKindOfClass:NSDictionary.class]) throw undescribed();
    NSArray *symbols = description[@"kANEFModelInputSymbolsArrayKey"];
    NSDictionary *functions = description[@"kANEFModelProcedureNameToIDMapKey"];
    NSArray *entries = description[@"ANEFModelProcedures"];
    if (![symbols isKindOfClass:NSArray.class] || ![functions isKindOfClass:NSDictionary.class] ||
        ![entries isKindOfClass:NSArray.class] || entries.count != functions.count)
      throw undescribed();
    std::vector<Procedure> result(entries.count);
    for (NSString *function in functions) {
      NSNumber *index = functions[function];
      if (![function isKindOfClass:NSString.class] || ![index isKindOfClass:NSNumber.class]) throw undescribed();
      if (index.unsignedIntegerValue >= entries.count)
        throw std::runtime_error("ANE program names an unknown procedure");
      result[index.unsignedIntegerValue].function = function.UTF8String;
    }
    for (NSDictionary *entry in entries) {
      if (![entry isKindOfClass:NSDictionary.class]) throw undescribed();
      NSNumber *index = entry[@"ANEFModelProcedureID"];
      NSArray *outputs = entry[@"ANEFModelOutputSymbolIndexArray"];
      NSArray *inputs = entry[@"ANEFModelInputSymbolIndexArray"];
      if (![index isKindOfClass:NSNumber.class] || ![outputs isKindOfClass:NSArray.class] ||
          ![inputs isKindOfClass:NSArray.class])
        throw undescribed();
      if (index.unsignedIntegerValue >= entries.count || outputs.count != 1)
        throw std::runtime_error("ANE program has a procedure of other than one output");
      NSNumber *output = outputs[0];
      if (![output isKindOfClass:NSNumber.class]) throw undescribed();
      Procedure &procedure = result[index.unsignedIntegerValue];
      procedure.outputSymbol = output.unsignedIntValue;
      for (NSNumber *symbol in inputs) {
        if (![symbol isKindOfClass:NSNumber.class]) throw undescribed();
        if (symbol.unsignedIntegerValue >= symbols.count)
          throw std::runtime_error("ANE program names an unknown input");
        NSString *input = symbols[symbol.unsignedIntegerValue];
        if (![input isKindOfClass:NSString.class]) throw undescribed();
        procedure.inputSymbols.push_back(symbol.unsignedIntValue);
        procedure.inputs.emplace_back(input.UTF8String);
      }
    }
    return result;
  });
}

void Program::Impl::unload() {
  guarded("unload", [&] {
    NSError *error = nil;
    if (![api.client unloadModel:model options:@{} qos:kQos error:&error]) fail("unload", error);
  });
  loaded = false;
}

void Program::Impl::release() noexcept {
  if (!loaded) return;
  try {
    unload();
  } catch (...) {
  }
}

Program::Program(std::string_view mil, std::span<const uint8_t> weights, Limits limits,
                 const std::filesystem::path &cache)
    // The last owner, this program or work it queued, deletes the state on
    // the service's queue: a model still loaded is unloaded there in turn,
    // with no caller waiting for it.
    : impl_(new Impl(api()),
            [](Impl *impl) {
              dispatch_async(impl->api.queue, ^{
                @autoreleasepool {
                  delete impl;
                }
              });
            }),
      limits_(std::move(limits)) {
#ifdef SPLASH_ANE_INSTRUMENTATION
  impl_->faults = takeArmed();
  if (!impl_->faults.changedMethod.empty()) check(changed(impl_->faults.changedMethod));
#endif
  // The source's hash names its directory and the service's key; a file
  // there that does not hold the source's bytes is written again.
  const std::string key = hex(fnv1a(weights, fnv1a(bytesOf(mil))));
  const std::filesystem::path directory = cache / key;
  std::filesystem::create_directories(directory);
  for (const auto &[name, bytes] : {std::pair{"model.mil", bytesOf(mil)}, std::pair{"weights.bin", weights}})
    if (!holds(directory / name, bytes)) writeWhole(directory / name, bytes);

  // Compiling, loading and reading the model's description are one piece of
  // the service's work.
  runOnQueue(impl_->api, limits_, [impl = impl_, directory, key] {
#ifdef SPLASH_ANE_INSTRUMENTATION
    if (impl->faults.construction) throw std::runtime_error("ANE construction failed: an injected fault");
#endif
    impl->model = impl->guarded("model creation", [&] {
      return [impl->api.model modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES]
                                     key:@(key.c_str())];
    });
    if (!impl->model) fail("model creation", nil);
    const bool compiled = impl->compiled();
    if (!compiled) impl->compile();
    impl->load(compiled);
    impl->procedures = impl->describe();
  });
}

Program::~Program() {
  try {
    unload(limits_);
  } catch (...) {
  }
}

uint32_t Program::procedure(std::string_view function) const {
  const auto found = std::ranges::find(impl_->procedures, function, &Procedure::function);
  if (found == impl_->procedures.end())
    throw std::invalid_argument("ANE program has no function " + std::string(function));
  return static_cast<uint32_t>(found - impl_->procedures.begin());
}

const std::vector<std::string> &Program::inputs(uint32_t procedure) const {
  return impl_->procedures.at(procedure).inputs;
}

Program::Binding Program::bind(uint32_t procedure, std::span<const Surface> inputs, const Surface &output) const {
#ifdef SPLASH_ANE_INSTRUMENTATION
  const auto &[bound, instead] = impl_->faults.swappedBinding;
  if (!bound.empty() && impl_->procedures.at(procedure).function == bound) procedure = Program::procedure(instead);
#endif
  const Procedure &called = impl_->procedures.at(procedure);
  if (inputs.size() != called.inputs.size()) throw std::invalid_argument("ANE program input count mismatch");
  auto state = std::make_shared<Binding::State>();
  state->program = impl_;
  @autoreleasepool {
    const auto object = [&](const Surface &surface) {
      if (!surface.surface) throw std::invalid_argument("ANE program bound to a surface of no IOSurface");
      state->surfaces.push_back(surface.surface);
      return impl_->guarded("surface object creation", [&] {
        id result = [impl_->api.surface objectWithIOSurface:(IOSurfaceRef)surface.surface.get()];
        if (!result) fail("surface object creation", nil);
        return result;
      });
    };
    NSMutableArray *objects = [NSMutableArray arrayWithCapacity:inputs.size()];
    NSMutableArray *indices = [NSMutableArray arrayWithCapacity:inputs.size()];
    for (size_t index = 0; index < inputs.size(); ++index) {
      [objects addObject:object(inputs[index])];
      [indices addObject:@(called.inputSymbols[index])];
    }
    state->inputs = [objects copy];
    state->inputIndices = [indices copy];
    state->outputs = @[ object(output) ];
    state->outputIndices = @[ @(called.outputSymbol) ];
    state->procedure = @(procedure);
  }
  return Binding(std::move(state));
}

void Program::enqueue(const Binding &binding, const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
                      std::function<void(bool)> done) {
#ifdef SPLASH_ANE_INSTRUMENTATION
  const uint64_t sequence = ++impl_->enqueues;
  if (sequence == impl_->faults.throwingEnqueue) throw std::runtime_error("ANE enqueue failed: an injected fault");
#endif
  if (!binding.state_ || binding.state_->program.lock() != impl_)
    throw std::invalid_argument("ANE program given another program's binding");
  if (!impl_->loaded) throw std::runtime_error("ANE program is not loaded");
  const Binding::State &bound = *binding.state_;
  const Api &api = impl_->api;
  @autoreleasepool {
    id native = (__bridge id)event.nativeHandle();
    // The service shares the event by its Mach port, which an event Metal's
    // validation layer wraps lacks: it would raise an Objective-C exception.
    if (![native respondsToSelector:NSSelectorFromString(@"eventPort")])
      throw std::runtime_error("the Neural Engine cannot share a Metal event the validation layer wraps");
    id events = impl_->guarded("event creation", [&] {
      id signalEvent = [api.signalEvent signalEventWithValue:signal symbolIndex:0 eventType:0 sharedEvent:native];
      id waitEvent = [api.waitEvent waitEventWithValue:wait sharedEvent:native eventType:0];
      if (!signalEvent || !waitEvent) fail("event creation", nil);
      id result = [api.events sharedEventsWithSignalEvents:@[ signalEvent ] waitEvents:@[ waitEvent ]];
      if (!result) fail("event creation", nil);
      return result;
    });
    id request = impl_->guarded("request creation", [&] {
      id result = [api.request requestWithInputs:bound.inputs
                                    inputIndices:bound.inputIndices
                                         outputs:bound.outputs
                                   outputIndices:bound.outputIndices
                                   weightsBuffer:nil
                                       perfStats:nil
                                  procedureIndex:bound.procedure
                                    sharedEvents:events
                               transactionHandle:nil];
      if (!result) fail("request creation", nil);
      return result;
    });
    // An evaluation with shared events runs asynchronously and requires a
    // completion handler, which only reports the evaluation's result.
    const auto completion = std::make_shared<Completion>();
    completion->done = std::move(done);
    void (^report)(BOOL, NSError *) = ^(BOOL success, NSError *) {
      std::unique_lock lock(completion->mutex);
      if (!completion->result) completion->result = success;
      completion->settle(std::move(lock));
    };
    const std::function<void()> evaluate = [impl = impl_, request] {
      impl->guarded("evaluation", [&] {
        NSError *error = nil;
        if (![impl->api.client evaluateWithModel:impl->model options:@{} request:request qos:kQos error:&error])
          fail("evaluation", error);
      });
    };
    try {
      impl_->guarded("completion handler", [&] { [request setCompletionHandler:report]; });
#ifdef SPLASH_ANE_INSTRUMENTATION
      const auto poison = [event, wait, signal, output = bound.surfaces.back(), report] {
        event.notify(wait, [event, signal, output, report] {
          *static_cast<uint16_t *>(IOSurfaceGetBaseAddress(static_cast<IOSurfaceRef>(output.get()))) = 0x7c00;
          event.signal(signal);
          report(YES, nil);
        });
      };
      const std::function<void()> lagged = [event, wait, evaluate, report, lag = impl_->faults.lag] {
        event.notify(wait, [evaluate, report, lag] {
          dispatch_after(dispatch_time(DISPATCH_TIME_NOW, std::chrono::nanoseconds(lag).count()),
                         dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
                           try {
                             evaluate();
                           } catch (const std::exception &) {
                             report(NO, nil);
                           }
                         });
        });
      };
      evaluateWithFaults(impl_->faults, sequence, report, impl_->faults.lag.count() ? lagged : evaluate, poison);
#else
      evaluate();
#endif
    } catch (...) {
      std::lock_guard lock(completion->mutex);
      completion->done = nullptr;
      throw;
    }
    std::unique_lock lock(completion->mutex);
    completion->queued = true;
    completion->settle(std::move(lock));
  }
}

void Program::unload(Limits limits) {
  limits_ = std::move(limits);
  runOnQueue(impl_->api, limits_, [impl = impl_] {
    if (impl->loaded) impl->unload();
  });
}

void Program::load(Limits limits) {
  limits_ = std::move(limits);
  runOnQueue(impl_->api, limits_, [impl = impl_] {
    if (impl->loaded) return;
#ifdef SPLASH_ANE_INSTRUMENTATION
    if (impl->faults.failingLoad) throw std::runtime_error("ANE load failed: an injected fault");
#endif
    if (!impl->compiled()) throw std::runtime_error("ANE program is not compiled");
    impl->load(false);
    try {
      if (impl->describe() != impl->procedures)
        throw std::runtime_error("ANE program describes other procedures once loaded again");
    } catch (...) {
      impl->release();
      throw;
    }
  });
}

#ifdef SPLASH_ANE_INSTRUMENTATION
void ProgramInstrumentation::arm(Faults faults) {
  std::lock_guard lock(armed().mutex);
  armed().faults = std::move(faults);
}
#endif

std::optional<std::string> recall(std::string_view key) noexcept {
  try {
    std::ifstream file(rememberedFile(key));
    std::string line;
    if (std::getline(file, line)) return line;
  } catch (const std::exception &) {
    // Nothing remembered can be read.
  }
  return std::nullopt;
}

void remember(std::string_view key, std::string_view line) noexcept {
  try {
    std::filesystem::create_directories(cacheDirectory());
    writeWhole(rememberedFile(key), bytesOf(line));
  } catch (const std::exception &) {
    // The next start calibrates again.
  }
}

void forget(std::string_view key) noexcept {
  std::error_code error;
  try {
    std::filesystem::remove(rememberedFile(key), error);
  } catch (const std::exception &) {
    // The next start takes what is remembered.
  }
}

} // namespace splash::ane
