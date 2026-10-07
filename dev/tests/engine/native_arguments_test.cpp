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
using splash::test::rejects;
using splash::test::require;

// A model directory the engine accepts: the target and draft directories,
// the tokenizer's configuration and the package manifest.
class ModelDirectory final {
public:
  ModelDirectory() {
    for (const char *directory : {"target", "draft", "tokenizer"})
      std::filesystem::create_directory(path() / directory);
    test::writeFile(path() / "tokenizer/config.json",
                    R"({"text_config":{"model_type":"qwen3_5_text",)"
                    R"("max_position_embeddings":262144,"hidden_size":5120,)"
                    R"("vocab_size":248320}})");
    test::writeFile(path() / "manifest.json",
                    R"({"schema_version":3,"model":"Qwen3.8-27B-DFlash2",)"
                    R"("format":{"name":"splash-packed-q4","q4_bits":4,)"
                    R"("q4_group_size":64,"q4_storage_n":256,)"
                    R"("section_alignment_bytes":16384,)"
                    R"("target_layer_magic":"MDFL0006",)"
                    R"("draft_layer_magic":"MDFD0004","vision_magic":"MDFV0001"},)"
                    R"("execution_geometry":{"draft_proposal_tokens":7,)"
                    R"("draft_query_rows":8,"draft_sliding_window":2048,)"
                    R"("maximum_batch_width":4,"prefill_token_budget":2048,)"
                    R"("target_kv_block_tokens":32,"target_verify_rows":8}})");
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
      {"disk", [](NativeArguments &read) { read.maxCacheDiskBytes = 5 * kGiB; }},
      {"persistent",
       [](NativeArguments &read) {
         read.maxCacheDiskBytes = 5 * kGiB;
         read.persistentCacheRoot = "/srv/cache";
       }},
      {"bf16", [](NativeArguments &read) { read.kvFormat = kv::Format::BFloat16; }},
      {"zbf16", [](NativeArguments &read) { read.kvFormat = kv::Format::ZipBFloat16; }},
      {"disk_bf16",
       [](NativeArguments &read) {
         read.maxCacheDiskBytes = 5 * kGiB;
         read.kvFormat = kv::Format::BFloat16;
       }},
      {"idle_release", [](NativeArguments &read) { read.idleReleaseSeconds = 1800.0; }},
      {"idle_release_off",
       [](NativeArguments &read) {
         read.idleReleaseSeconds = std::numeric_limits<double>::infinity();
       }},
      {"decode_share",
       [](NativeArguments &read) {
         read.maxCacheDiskBytes = 5 * kGiB;
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
  };
  // Every field but the model inspected from the directory.
  const auto same = [](const NativeArguments &read, const NativeArguments &meant) {
    return read.modelRoot == meant.modelRoot && read.maxContext == meant.maxContext &&
           read.maxMemoryBytes == meant.maxMemoryBytes &&
           read.maxCacheDiskBytes == meant.maxCacheDiskBytes &&
           read.persistentCacheRoot == meant.persistentCacheRoot &&
           read.kvFormat == meant.kvFormat && read.decodeShare == meant.decodeShare &&
           read.maxImagePatches == meant.maxImagePatches &&
           read.idleReleaseSeconds == meant.idleReleaseSeconds &&
           read.preventIdleSleep == meant.preventIdleSleep &&
           read.neuralEngine == meant.neuralEngine;
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

// A command line the server never writes is refused with what is wrong.
void testRefusals() {
  const ModelDirectory model;
  const auto refuses = [&](const std::vector<std::string> &arguments,
                           std::string_view expected) {
    rejects([&] { static_cast<void>(parse(model, arguments)); }, expected,
            "the engine read a command line it must refuse");
  };
  refuses({"auto", "auto", "--kv-format", "fp16"},
          "--kv-format requires int8, bf16 or zbf16");
  // An option's missing value fails its check.
  refuses({"auto", "auto", "--kv-format"}, "--kv-format requires int8, bf16 or zbf16");
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
  refuses({"auto", "auto", "--cache-dir", "/srv/cache"},
          "--cache-dir requires a MAX_CACHE_DISK_BYTES quota");
  refuses({"auto", "auto", "5368709120", "--cache-dir", ""},
          "--cache-dir requires a directory");
  refuses({"auto", "auto", "5G"},
          "MAX_CACHE_DISK_BYTES must be a nonnegative integer");
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
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " native_command_golden.txt\n";
    return 2;
  }
  try {
    testServerCommandLines(argv[1]);
    testNeuralEngine();
    testRefusals();
    std::cout << "native arguments tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native arguments tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
