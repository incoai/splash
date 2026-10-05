#pragma once

#include <functional>
#include <variant>

namespace splash::model {

// Names the files a target is read from without the loaders' headers, so a
// family's header declares its loader alone; QwenTargetLoader.hpp defines
// PackageTargetFiles and reads the files.
template <class Layout> struct PackageTargetFiles;
class AffineTargetLoader;
class GgufTargetLoader;

// The files a target is read from: a package's files (splash-packed-q4
// formats), or the images a loader writes from an MLX or GGUF source.
template <class Layout>
using QwenTargetFiles = std::variant<PackageTargetFiles<Layout>, std::reference_wrapper<AffineTargetLoader>,
                                     std::reference_wrapper<GgufTargetLoader>>;

} // namespace splash::model
