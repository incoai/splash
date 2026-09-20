#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace splash {

// Device features and memory limits used by runtime planning.
struct DeviceCapabilities {
    std::string deviceName = "unknown";
    // Placement-sparse support is queryable from macOS 26.4.
    static constexpr uint32_t kMinimumMacosMajor = 26;
    static constexpr uint32_t kMinimumMacosMinor = 4;
    uint32_t macosMajor = 0;
    uint32_t macosMinor = 0;
    uint32_t macosPatch = 0;
    // Highest supported MTLGPUFamilyAppleN. M2 is Apple8; actual sparse
    // support is probed separately rather than inferred from the GPU family.
    static constexpr uint32_t kMinimumAppleGpuFamily = 8;
    uint32_t appleGpuFamily = 0;
    // IORegistry gpu-core-count; zero means unavailable. Kernel policy then
    // assumes a large GPU when choosing parallelism.
    uint32_t gpuCoreCount = 0;
    uint64_t physicalMemoryBytes = 0;
    uint64_t recommendedMaxWorkingSetBytes = 0;
    uint64_t maxBufferLengthBytes = 0;
    uint64_t maxThreadgroupMemoryBytes = 0;
    // Splash runtime only emits one-dimensional threadgroups. Metal exposes the
    // per-dimension limit as MTLSize; the pipeline state separately validates
    // the total thread count for every dispatch.
    uint64_t maxThreadgroupWidth = 0;
    bool hasUnifiedMemory = false;
    // Exposes the full logical KV address space while committing physical
    // memory only for pages in use.
    bool supportsPlacementSparse = false;

    [[nodiscard]] bool meetsMinimumMacos() const noexcept {
        return macosMajor > kMinimumMacosMajor ||
               (macosMajor == kMinimumMacosMajor &&
                macosMinor >= kMinimumMacosMinor);
    }
    // "major.minor.patch" of the running macOS for status and messages.
    [[nodiscard]] std::string macosVersion() const;

    // Returns a stable machine-readable reason, or nullopt when valid.
    [[nodiscard]] std::optional<std::string> validationError() const;
};

} // namespace splash
