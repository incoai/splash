// What the engine's memory and caches take from each family Splash serves
// (model::kFamilies), against family_values_golden.txt: the bytes of a KV
// page of each form in each KV format and the extents a pool of each size
// takes, a lane's and a cached state's bytes in each state format, and the
// persistent cache's namespace of each pair of formats. A change to shared
// code that moves one of them for a family fails here, and names the values
// it now has; a new family adds its lines.

#include "TestChecks.hpp"
#include "TestDrafts.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/ModelDescriptor.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace splash;

namespace {

using splash::test::require;

// The draft each family's values are taken with: the one
// dev/tests/fixtures/model-configs describes for it (TestDrafts.hpp).
model::DFlashDraftLayout draftOf(std::string_view family) {
  if (family == model::Qwen3_8Layout{}.family)
    return test::qwen3_8Draft();
  if (family == model::Qwen3_6MoeLayout{}.family)
    return test::qwen3_6MoeDraft();
  if (family == model::kQwen3_5_122BLayout.family)
    return test::qwen3_5_122BDraft();
  if (family == model::Qwen4ExpLayout{}.family)
    return test::flashNextDraft();
  if (family == model::FlintLayout{}.family)
    return test::flintDraft();
  throw std::logic_error(std::string(family) + " has no draft to take its values with");
}

// Pools of a few pages to a million, around the extent sizes.
constexpr std::array<uint64_t, 22> kPoolPages{
    1, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 511, 512, 1000, 4095, 4096, 10'000, 65'535, 65'536,
    100'000, 262'144, 1'000'000};

std::string familyValues() {
  std::ostringstream out;
  model::forEachFamily([&](const auto &family) {
    const model::DFlashDraftLayout draft = draftOf(family.family);
    // A family whose drafts choose their mask token holds its draft's, the
    // fixture's first reserved token.
    auto target = family;
    if (target.draftChoosesMaskToken()) target.maskToken = target.firstDraftMaskToken;
    const model::ModelDescriptor model =
        model::makeModelDescriptor(std::string(family.family), target, draft,
                                   {.outputHiddenSize = draft.hiddenSize}, model::TargetSource::Gguf,
                                   model::VisionSource::None);
    require(model.valid(), std::string(family.family) + "'s descriptor is not valid");
    const kv::Layout &kv = model.targetKvLayout;
    out << "family " << family.family << '\n'
        << "kv layers " << kv.attentionLayers << " heads " << kv.kvHeads << " dimension " << kv.headDimension
        << " index " << kv.indexDimension << (kv.valuesTokenMajor ? " values-token-major" : "");
    if (kv.windowedLayers)
      out << " window " << kv.windowTokens << " layers 0x" << std::hex << kv.windowedLayers << std::dec;
    out << '\n';
    for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
      kv::Layout layout = kv;
      layout.format = format;
      out << "kv " << kv::formatName(format) << " page " << layout.bytesPerModelPage(kv::PageForm::Full)
          << " index-page " << layout.bytesPerModelPage(kv::PageForm::Index)
          << (layout.slimmable() ? " slim-page " + std::to_string(layout.bytesPerModelPage(kv::PageForm::Slim)) : "")
          << " alignment " << layout.extentAlignmentPages() << " extents " << layout.minimumExtentPages() << ".."
          << layout.maximumExtentPages() << '\n'
          << "kv " << kv::formatName(format) << " extent-pages";
      for (const uint64_t pages : kPoolPages)
        out << ' ' << pages << ':' << layout.extentPagesFor(pages);
      out << '\n';
    }
    for (const ops::GdnStateFormat format : {ops::GdnStateFormat::Float32, ops::GdnStateFormat::BFloat16}) {
      model::CompositeStateLayout states = model.stateLayout;
      states.target.format = format;
      out << "state " << (format == ops::GdnStateFormat::Float32 ? "fp32" : "bf16") << " cell "
          << states.target.cellBytes() << " lane " << states.laneBytes() << " cached " << states.cachedBytes()
          << '\n';
      for (const kv::Format kvFormat : {kv::Format::Int8, kv::Format::BFloat16}) {
        kv::Layout layout = kv;
        layout.format = kvFormat;
        const engine::RuntimeCacheIdentity identity = engine::makeRuntimeCacheIdentity(
            std::string(64, 'a'), std::string(64, 'b'), "build", layout, model.ropeScaling);
        out << "namespace " << kv::formatName(kvFormat) << ' '
            << (format == ops::GdnStateFormat::Float32 ? "fp32" : "bf16") << ' '
            << engine::persistentCacheNamespace(identity, states) << '\n';
      }
    }
  });
  return out.str();
}

// The golden's lines, without its comments.
std::string golden(const char *path) {
  std::ifstream file(path);
  require(static_cast<bool>(file), std::string("cannot read ") + path);
  std::string values, line;
  while (std::getline(file, line))
    if (!line.empty() && line.front() != '#')
      values += line + '\n';
  return values;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " family_values_golden.txt\n";
    return 2;
  }
  try {
    const std::string values = familyValues();
    require(values == golden(argv[1]),
            std::string(argv[1]) + " does not hold the families' values, which are now:\n" + values);
    std::cout << "family values match " << argv[1] << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "family values test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
