#pragma once

#include "AwakeClock.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::ane {

// Memory the GPU and the Neural Engine share without a copy: an IOSurface of
// `rows` rows of `width` fp16 or int8 elements, each row padded to 64 bytes,
// and the Metal buffer over it.
struct Surface final {
  enum class Element : uint8_t { Float16, Int8 };

  [[nodiscard]] static Surface create(metal::MetalBackend &backend, uint32_t rows, uint32_t width,
                                      Element element);
  // The bytes of a row, and the whole pages such a surface takes.
  [[nodiscard]] static uint32_t rowBytes(uint32_t width, Element element) noexcept;
  [[nodiscard]] static uint64_t bytes(uint32_t rows, uint32_t width, Element element) noexcept;

  // An element's MIL type.
  [[nodiscard]] static const char *milType(Element element) noexcept;
  // The MIL strides of this surface's leading `rows` rows, and the
  // tensor_buffer type of their leading `width` elements, as a program reads
  // or writes them.
  [[nodiscard]] std::string strides(uint32_t rows) const;
  [[nodiscard]] std::string bufferType(uint32_t rows, uint32_t width) const;

  std::shared_ptr<void> surface;
  metal::MetalBuffer buffer;
  Element element = Element::Float16;
  uint32_t strideBytes = 0;
};

// The weight blob of one fp16 constant tensor, which a program's MIL names as
// BLOBFILE(path = string("@model_path/weights.bin"), offset = kConstantOffset).
inline constexpr uint64_t kConstantOffset = 64;
[[nodiscard]] std::vector<uint8_t> constantBlob(std::span<const _Float16> values);

// splash-ane in the per-user cache directory, where programs keep their
// sources and recall() keeps what remember() wrote. Throws if macOS names no
// per-user cache directory.
[[nodiscard]] std::filesystem::path cacheDirectory();

// Why no Program can run in this process, as constructing one would throw
// it: the private interface it uses does not load, or lacks or changed a
// class or method; none when one can. Resolves the interface once, compiling
// nothing.
[[nodiscard]] std::optional<std::string> unavailable();

// What a Program throws when its caller's Limits::interrupted() ended the
// wait for the service: the caller stopped, the service did not fail.
class Interrupted final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// A MIL program compiled and loaded on the Neural Engine through the private
// AppleNeuralEngine client, the service behind Core ML. Each of its functions
// is a procedure of the loaded model. The service holds memory for a loaded
// model's intermediate values beside the surfaces, which its procedures share:
// one program of several functions holds what its largest function needs,
// where a program per function would hold their sum. Evaluations run
// asynchronously, ordered against Metal work by a shared event. A program
// throws a std::runtime_error for a class or selector the client lacks, a
// method whose type encoding differs from the one this client declares, an
// object it does not return and an Objective-C exception it raises.
class Program final {
public:
  // How long a caller waits for the service to compile, load or unload a
  // program, and what it asks between the slices of that wait: true ends the
  // wait with Interrupted, the limit with "the Neural Engine did not answer
  // within N s". The service's work for every Program in the process runs in
  // turn on one queue, so a wait also covers the work queued before it. Work
  // whose caller stopped waiting runs to its end, and a program whose
  // constructor gave up on it is unloaded then.
  struct Limits final {
    AwakeClock::duration limit;
    std::function<bool()> interrupted;
  };

  // `weights` is the blob file the program's constants name. The source and
  // the blob stay in a directory of `cache` named by their hash, the service's
  // key for the compiled program, so that later processes load it without
  // compiling it again. Compiles the program unless the service holds it, and
  // loads it, under `limits`.
  Program(std::string_view mil, std::span<const uint8_t> weights, Limits limits,
          const std::filesystem::path &cache = cacheDirectory());
  // Unloads the program as unload() does, without throwing: an unload whose
  // wait the limits end stays queued, and the service runs it in turn.
  ~Program();
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  // The procedure of the function named `function`.
  [[nodiscard]] uint32_t procedure(std::string_view function) const;
  // A procedure's input names in the order bind() takes their surfaces.
  [[nodiscard]] const std::vector<std::string> &inputs(uint32_t procedure) const;

  // A procedure over fixed surfaces as the service takes an evaluation of it:
  // the service's objects for the surfaces and the indices of their symbols.
  // It keeps the surfaces alive; copies share it.
  class Binding final {
  private:
    struct State;
    explicit Binding(std::shared_ptr<const State> state) : state_(std::move(state)) {}
    std::shared_ptr<const State> state_;
    friend class Program;
  };
  // `procedure` over `inputs`, in the order inputs() names them, and `output`.
  // Throws if their count is not the procedure's.
  [[nodiscard]] Binding bind(uint32_t procedure, std::span<const Surface> inputs, const Surface &output) const;

  // Queues one evaluation of `binding` that starts once `event` reaches
  // `wait` and raises it to `signal` when done. Throws if this program did
  // not bind `binding`, is not loaded or cannot queue the evaluation; `done`
  // is then never called. Otherwise `done` is called at most once, with the
  // result the service reports: true once the evaluation ran, false if it
  // failed, for which the program itself does not raise the event. It runs
  // on a thread of the service's, or on the caller's before enqueue() returns
  // if the service reported first. A service that never reports never calls
  // it.
  void enqueue(const Binding &binding, const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
               std::function<void(bool)> done);

  // Releases the loaded program, whose compilation the service keeps; nothing
  // if it is not loaded. Waits under `limits`, which the destructor's unload
  // keeps, as it keeps load()'s. No evaluation may be in flight.
  void unload(Limits limits);
  // Loads the program from the compilation the service kept, under `limits`;
  // nothing if it is loaded. Never compiles: throws "not compiled" if the
  // service no longer holds the compilation, and throws, leaving the program
  // unloaded, if the loaded model describes other procedures than the
  // constructor found, so that bindings stay valid.
  void load(Limits limits);

private:
  struct Impl;
  // Shared with the work this program queues on the service's queue.
  std::shared_ptr<Impl> impl_;
  Limits limits_;
};

// A line kept beside the compiled programs under `key`, such as a model's
// calibration on this Mac; none until remembered, once forgotten or macOS
// clears the cache directory, or when it cannot be read.
[[nodiscard]] std::optional<std::string> recall(std::string_view key) noexcept;
void remember(std::string_view key, std::string_view line) noexcept;
void forget(std::string_view key) noexcept;

} // namespace splash::ane
