#pragma once

#include <functional>
#include <variant>

namespace splash::model {

// Names the files a target is read from without the loaders' headers, so a
// family's header declares its loader alone; QwenTargetLoader.hpp reads the
// files.
class GgufTargetLoader;
class MlxTargetLoader;

// The files a target is read from: the block images a loader writes from an
// MLX source, in its MLX formats, or from a GGUF.
using QwenTargetFiles =
    std::variant<std::reference_wrapper<GgufTargetLoader>, std::reference_wrapper<MlxTargetLoader>>;

} // namespace splash::model
