// The serve-native command lines the server writes, as the engine reads them
// (native_command_golden.txt), and the ones the engine refuses.

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "engine/NativeArguments.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace splash;
using engine::NativeArguments;
using engine::parseRetainedCheckpointTokens;
using splash::test::rejects;
using splash::test::require;

// The directory of the model configurations (dev/tests/fixtures/model-configs).
std::filesystem::path modelConfigs;

// A model directory the engine accepts: the target and draft directories and
// the assembly's record and configurations of an MLX Qwen3.8-27B.
class ModelDirectory final {
public:
  ModelDirectory() {
    for (const char *directory : {"target", "draft"})
      std::filesystem::create_directory(path() / directory);
    test::writeFile(path() / "model.json",
                    R"({"version":1,"model":"mlx-community/Qwen3.8-27B-4bit",)"
                    R"("target_format":"mlx-affine","vision_format":"none"})");
    const std::filesystem::path configs = modelConfigs / "qwen3.8-27b";
    std::filesystem::copy_file(configs / "config.json", path() / "config.json");
    std::filesystem::copy_file(configs / "draft" / "config.json", path() / "draft" / "config.json");
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return directory_.path();
  }

private:
  test::TemporaryDirectory directory_{"splash-native-arguments"};
};

// The whole command line, after the executable's name.
NativeArguments parseCommand(const std::vector<std::string> &arguments) {
  std::vector<const char *> argv{"splash"};
  for (const std::string &argument : arguments)
    argv.push_back(argument.c_str());
  return engine::parseNativeArguments(static_cast<int>(argv.size()),
                                      argv.data());
}

// serve-native with the model directory and the arguments after it.
NativeArguments parse(const ModelDirectory &model,
                      const std::vector<std::string> &arguments) {
  std::vector<std::string> command{"serve-native", model.path().string()};
  command.insert(command.end(), arguments.begin(), arguments.end());
  return parseCommand(command);
}

// Each golden line's arguments after MODEL_DIRECTORY, by its name; the serve
// options between are the server's to read.
std::map<std::string, std::vector<std::string>> readGolden(const char *path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error(std::string("cannot read ") + path);
  std::map<std::string, std::vector<std::string>> golden;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#')
      continue;
    const size_t options = line.find('|');
    const size_t arguments = line.find('|', options + 1);
    if (arguments == std::string::npos)
      throw std::runtime_error("golden line without its arguments: " + line);
    std::string name;
    std::istringstream(line.substr(0, options)) >> name;
    std::istringstream words(line.substr(arguments + 1));
    std::vector<std::string> values;
    for (std::string word; words >> word;)
      values.push_back(word);
    golden.emplace(std::move(name), std::move(values));
  }
  return golden;
}

// Every line the server writes reads as the server means it: what the
// engine reads with no options, with what the line changes and nothing else.
void testServerCommandLines(const char *path) {
  const ModelDirectory model;
  constexpr uint64_t kGiB = uint64_t{1} << 30;
  NativeArguments defaults;
  defaults.modelRoot = std::filesystem::canonical(model.path());
  using Change = std::function<void(NativeArguments &)>;
  const std::map<std::string, Change> changes{
      {"defaults", [](NativeArguments &) {}},
      {"limits",
       [](NativeArguments &read) {
         read.maxContext = 102400;
         read.maxMemoryBytes = 32 * kGiB;
       }},
      // No quota is the automatic one.
      {"disk_auto", [](NativeArguments &) {}},
      {"disk_off", [](NativeArguments &read) { read.maxCacheDisk = {}; }},
      {"disk", [](NativeArguments &read) { read.maxCacheDisk = {.bytes = 5 * kGiB}; }},
      {"disk_writes", [](NativeArguments &read) { read.maxCacheDiskWritesBytes = 200 * kGiB; }},
      {"persistent_auto", [](NativeArguments &read) { read.persistentCache = true; }},
      {"persistent",
       [](NativeArguments &read) {
         read.maxCacheDisk = {.bytes = 5 * kGiB};
         read.persistentCache = true;
       }},
      {"persistent_dir",
       [](NativeArguments &read) {
         read.maxCacheDisk = {.bytes = 5 * kGiB};
         read.persistentCache = true;
         read.cacheRoot = "/srv/cache";
       }},
      {"bf16", [](NativeArguments &read) { read.kvFormat = kv::Format::BFloat16; }},
      {"disk_bf16",
       [](NativeArguments &read) {
         read.maxCacheDisk = {.bytes = 5 * kGiB};
         read.kvFormat = kv::Format::BFloat16;
       }},
      {"state_bf16",
       [](NativeArguments &read) { read.stateFormat = ops::GdnStateFormat::BFloat16; }},
      {"idle_release", [](NativeArguments &read) { read.idleReleaseSeconds = 1800.0; }},
      {"idle_release_off",
       [](NativeArguments &read) {
         read.idleReleaseSeconds = std::numeric_limits<double>::infinity();
       }},
      {"decode_share",
       [](NativeArguments &read) {
         read.maxCacheDisk = {.bytes = 5 * kGiB};
         read.decodeShare = 0.0;
       }},
      {"image_patches", [](NativeArguments &read) { read.maxImagePatches = 4096; }},
      {"image_patches_rounded",
       [](NativeArguments &read) { read.maxImagePatches = 3904; }},
      // The server's default pixel limit is the engine's default limit of
      // 4194304 / (16 x 16) patches, so the server leaves it out.
      {"image_patches_default",
       [](NativeArguments &read) { read.maxImagePatches = 16384; }},
      {"idle_sleep", [](NativeArguments &read) { read.preventIdleSleep = false; }},
      {"disable_ane", [](NativeArguments &read) { read.neuralEngine = false; }},
      // A relaunch keeps the first engine's context; with --max-context,
      // which every start holds, its command is the first one's.
      {"relaunch", [](NativeArguments &read) { read.relaunchContext = 61433; }},
      {"relaunch_limits",
       [](NativeArguments &read) {
         read.maxContext = 102400;
         read.maxMemoryBytes = 32 * kGiB;
       }},
      {"ple_ssd_cache", [](NativeArguments &read) { read.pleCacheBytes = 2 * kGiB; }},
      {"expert_memory", [](NativeArguments &read) { read.expertMemoryBytes = 16 * kGiB; }},
  };
  // Every field but the model inspected from the directory.
  const auto same = [](const NativeArguments &read, const NativeArguments &meant) {
    return read.modelRoot == meant.modelRoot && read.maxContext == meant.maxContext &&
           read.relaunchContext == meant.relaunchContext &&
           read.maxMemoryBytes == meant.maxMemoryBytes &&
           read.maxCacheDisk.automatic == meant.maxCacheDisk.automatic &&
           read.maxCacheDisk.bytes == meant.maxCacheDisk.bytes &&
           read.maxCacheDiskWritesBytes == meant.maxCacheDiskWritesBytes &&
           read.persistentCache == meant.persistentCache && read.cacheRoot == meant.cacheRoot &&
           read.kvFormat == meant.kvFormat && read.stateFormat == meant.stateFormat &&
           read.decodeShare == meant.decodeShare &&
           read.maxImagePatches == meant.maxImagePatches &&
           read.idleReleaseSeconds == meant.idleReleaseSeconds &&
           read.preventIdleSleep == meant.preventIdleSleep &&
           read.neuralEngine == meant.neuralEngine && read.pleCacheBytes == meant.pleCacheBytes &&
           read.expertMemoryBytes == meant.expertMemoryBytes;
  };
  const auto golden = readGolden(path);
  require(golden.size() == changes.size(),
          "a golden command line has no meaning to check, or a meaning no line");
  for (const auto &[name, arguments] : golden) {
    const auto change = changes.find(name);
    require(change != changes.end(), "the golden command line " + name + " has no meaning");
    NativeArguments meant = defaults;
    change->second(meant);
    require(same(parse(model, arguments), meant),
            "the engine did not read the server's " + name + " command line as "
            "the server means it");
  }
}

// The Neural Engine split is on unless --ane off turns it off; --ane on, which
// the server never writes, keeps it on.
void testNeuralEngine() {
  const ModelDirectory model;
  require(parse(model, {"auto", "auto"}).neuralEngine &&
              parse(model, {"auto", "auto", "--ane", "on"}).neuralEngine &&
              !parse(model, {"auto", "auto", "--ane", "off"}).neuralEngine,
          "--ane did not reach the arguments as given");
}

// A model directory holds draft/ or, for a model drafting with its MTP head
// (Qwen3.8-Flash-Next from a GGUF), mtp/ beside target/; one with neither
// is refused, as is mtp/ for a family without the head.
void testModelDirectoryRoles() {
  const ModelDirectory model;
  std::filesystem::remove_all(model.path() / "draft");
  rejects([&] { static_cast<void>(parse(model, {"auto", "auto"})); }, "target/ and draft/ directories",
          "a model directory without draft/ or mtp/ was accepted");
  std::filesystem::create_directory(model.path() / "mtp");
  rejects([&] { static_cast<void>(parse(model, {"auto", "auto"})); }, "has no MTP head",
          "a family without an MTP head accepted mtp/ in place of its draft");
  const test::TemporaryDirectory flash{"splash-native-arguments-flash"};
  for (const char *directory : {"target", "mtp"})
    std::filesystem::create_directory(flash.path() / directory);
  test::writeFile(flash.path() / "model.json",
                  R"({"version":1,"model":"unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL",)"
                  R"("target_format":"gguf","vision_format":"none"})");
  std::filesystem::copy_file(modelConfigs / "qwen3.8-flash-next" / "config.json", flash.path() / "config.json");
  const NativeArguments parsed = parseCommand({"serve-native", flash.path().string(), "auto", "auto"});
  require(parsed.modelRoot == std::filesystem::canonical(flash.path()) && !parsed.model.hasDraft(),
          "a Flash-Next directory with mtp/ in place of draft/ was refused");
}

// A command line the server never writes is refused with what is wrong.
void testRefusals() {
  require(parseRetainedCheckpointTokens("0") == 0 && parseRetainedCheckpointTokens("8192") == 8192 &&
              parseRetainedCheckpointTokens("16384") == 16384,
          "the retained checkpoint grid was not parsed");
  for (const char *grid : {"", "-1", "2048", "8193", "4294967296", "8K", "8192x"})
    rejects([&] { static_cast<void>(parseRetainedCheckpointTokens(grid)); },
            "SPLASH_GDN_CHECKPOINT_TOKENS requires", "an invalid retained checkpoint grid was accepted");
  const ModelDirectory model;
  const auto refuses = [&](const std::vector<std::string> &arguments,
                           std::string_view expected) {
    rejects([&] { static_cast<void>(parse(model, arguments)); }, expected,
            "the engine read a command line it must refuse");
  };
  refuses({"auto", "auto", "--kv-format", "fp16"},
          "--kv-format requires int8 or bf16");
  // An option's missing value fails its check.
  refuses({"auto", "auto", "--kv-format"}, "--kv-format requires int8 or bf16");
  refuses({"auto", "auto", "--state-format", "fp16"},
          "--state-format requires fp32 or bf16");
  refuses({"auto", "auto", "--state-format"}, "--state-format requires fp32 or bf16");
  for (const char *share : {"-1", "nan", "inf", "half"})
    refuses({"auto", "auto", "--decode-share", share},
            "--decode-share requires a nonnegative number");
  for (const char *patches : {"0", "6", "16388"})
    refuses({"auto", "auto", "--max-image-patches", patches},
            "--max-image-patches requires a positive multiple of 4");
  // The server passes seconds: a duration's unit is its to resolve.
  for (const char *release : {"0", "-1", "30m", "inf"})
    refuses({"auto", "auto", "--idle-release", release},
            "--idle-release requires off or a positive number of seconds");
  refuses({"auto", "auto", "--idle-sleep", "never"},
          "--idle-sleep requires prevent or allow");
  refuses({"auto", "auto", "--ane", "gpu"}, "--ane requires on or off");
  // The server passes bytes.
  for (const char *size : {"0", "2G", ""})
    refuses({"auto", "auto", "--ple-ssd-cache", size}, "--ple-ssd-cache requires a positive byte count");
  for (const char *size : {"0", "16G", ""})
    refuses({"auto", "auto", "--expert-memory", size}, "--expert-memory requires a positive byte count");
  refuses({"auto", "auto", "--persistent-cache", "yes"}, "--persistent-cache requires on or off");
  refuses({"auto", "auto", "0", "--persistent-cache", "on"},
          "--persistent-cache requires an SSD cache, which MAX_CACHE_DISK_BYTES 0 turns off");
  refuses({"auto", "auto", "5368709120", "--cache-dir", "/srv/cache"},
          "--cache-dir requires --persistent-cache on");
  refuses({"auto", "auto", "5368709120", "--persistent-cache", "on", "--cache-dir", ""},
          "--cache-dir requires a directory");
  refuses({"auto", "auto", "5G"},
          "MAX_CACHE_DISK_BYTES must be a nonnegative integer");
  for (const char *writes : {"200G", "-1", "1.5", ""})
    refuses({"auto", "auto", "5368709120", "--max-cache-disk-writes", writes},
            "--max-cache-disk-writes requires a nonnegative integer byte count");
  refuses({"auto", "auto", "0", "--max-cache-disk-writes", "214748364800"},
          "--max-cache-disk-writes requires an SSD cache, which MAX_CACHE_DISK_BYTES 0 turns off");
  require(parse(model, {"auto", "auto", "5368709120", "--max-cache-disk-writes", "0"})
                  .maxCacheDiskWritesBytes == 0,
          "--max-cache-disk-writes 0 set a write budget");
  refuses({"auto", "auto", "--max-memory", "32G"},
          "unexpected argument --max-memory");
  for (const char *memory : {"0", "32G", "-1"})
    refuses({"auto", memory}, "MAX_MEMORY_BYTES must be auto or a positive integer");
  // The model's own context bounds MAX_CONTEXT.
  const std::string maximum = std::to_string(
      model::inspectModelRoot(model.path()).capabilities.maximumContextTokens);
  for (const std::string &context : {std::string("0"), std::string("100K"),
                                     std::to_string(std::stoull(maximum) + 1)})
    refuses({context, "auto"},
            "MAX_CONTEXT must be auto or an integer in [1, " + maximum + "]");
  require(parse(model, {maximum, "auto"}).maxContext == std::stoul(maximum),
          "the model's whole context was refused");
  // A relaunch's context is one the model holds, and only without
  // MAX_CONTEXT, which every start holds.
  for (const std::string &context : {std::string("0"), std::string("auto"), std::string("100K"),
                                     std::to_string(std::stoull(maximum) + 1), std::string()})
    refuses({"auto", "auto", "--relaunch-context", context},
            "--relaunch-context requires an integer in [1, " + maximum + "]");
  refuses({"auto", "auto", "--relaunch-context"},
          "--relaunch-context requires an integer in [1, " + maximum + "]");
  refuses({"102400", "auto", "--relaunch-context", "102400"},
          "--relaunch-context requires MAX_CONTEXT auto");
  require(parse(model, {"auto", "auto", "--relaunch-context", maximum}).relaunchContext ==
              std::stoul(maximum),
          "the model's whole context was refused a relaunch");

  const auto commandRefused = [](const std::vector<std::string> &command,
                                 std::string_view expected) {
    rejects([&] { static_cast<void>(parseCommand(command)); }, expected,
            "the engine ran a command it must refuse");
  };
  commandRefused({"serve", model.path().string(), "auto", "auto"},
                 "expected the serve-native command");
  commandRefused({"serve-native", model.path().string(), "auto"},
                 "expected the serve-native command");
  commandRefused({"serve-native", (model.path() / "missing").string(), "auto", "auto"},
                 "MODEL_DIRECTORY must name an existing directory");
  const test::TemporaryDirectory empty("splash-native-arguments-empty");
  commandRefused({"serve-native", empty.path().string(), "auto", "auto"},
                 "MODEL_DIRECTORY must hold the model's target/ and draft/ directories");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " native_command_golden.txt MODEL_CONFIGS\n";
    return 2;
  }
  modelConfigs = argv[2];
  try {
    testServerCommandLines(argv[1]);
    testNeuralEngine();
    testRefusals();
    testModelDirectoryRoles();
    std::cout << "native arguments tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native arguments tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
