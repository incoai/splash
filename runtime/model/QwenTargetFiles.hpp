#pragma once

#include <functional>
#include <variant>

namespace splash::model {

// Names the files a target is read from without the loaders' headers, so a
// family's header declares its loader alone; QwenTargetLoader.hpp defines
// PackedTargetFiles and reads the files.
template <class Layout> struct PackedTargetFiles;
class AffineTargetLoader;
class GgufTargetLoader;

// The files a target is read from: packed files (splash-packed-q4 formats),
// or the cached files a loader prepares from an MLX or GGUF source.
template <class Layout>
using QwenTargetFiles = std::variant<PackedTargetFiles<Layout>, std::reference_wrapper<AffineTargetLoader>,
                                     std::reference_wrapper<GgufTargetLoader>>;

} // namespace splash::model
