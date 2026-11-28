#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace splash {

// The GPU classes kernel policy tells apart, by how a core runs matrix
// operations: Apple9 (family 9: M3, M4) on the FP32 pipe; Apple10 (family 10:
// M5) on a per-core neural accelerator; Apple11 (family 11 and later: M6)
// likewise, its accelerator no faster on 16 rows than on 8 from 4.5 bits per
// weight. Startup refuses families below 9 (validationError).
enum class GpuClass : uint8_t { Apple9, Apple10, Apple11 };

// Device features and memory limits used by runtime planning.
struct DeviceCapabilities {
    std::string deviceName = "unknown";
    // The tested floor (MACOS_MIN_VERSION in the Makefile); the MPP kernels
    // need macOS 26.2 or newer.
    static constexpr uint32_t kMinimumMacosMajor = 26;
    static constexpr uint32_t kMinimumMacosMinor = 4;
    uint32_t macosMajor = 0;
    uint32_t macosMinor = 0;
    uint32_t macosPatch = 0;
    // Highest supported MTLGPUFamilyAppleN.
    static constexpr uint32_t kMinimumAppleGpuFamily = 9;
    uint32_t appleGpuFamily = 0;
    // IORegistry gpu-core-count; zero means unavailable. plannedGpuCores
    // substitutes kAssumedGpuCores for kernel policy. Keep the missing value
    // here (status reports it).
    uint32_t gpuCoreCount = 0;
    // Missing core metadata uses one intermediate estimate for all families.
    // This is a fallback, not a calibrated optimum. Reported counts always win.
    static constexpr uint32_t kAssumedGpuCores = 32;
    uint64_t physicalMemoryBytes = 0;
    uint64_t recommendedMaxWorkingSetBytes = 0;
    uint64_t maxBufferLengthBytes = 0;
    uint64_t maxThreadgroupMemoryBytes = 0;
    // Splash runtime only emits one-dimensional threadgroups. Metal exposes the
    // per-dimension limit as MTLSize; the pipeline state separately validates
    // the total thread count for every dispatch.
    uint64_t maxThreadgroupWidth = 0;
    bool hasUnifiedMemory = false;
    // The memory rate the laws read (ops::KernelCalibration::memoryBytesPerSecond), in bytes a second: a plain
    // read of a 256 MiB private buffer on a GPU left alone, as a constant of the device's class, its GPU class
    // and core count (classMemoryBytesPerSecond), which the reading of a device sets. It is not measured at a
    // start: a probe took another process's share of the GPU for the device's rate (a 16-core M5 Pro read 147
    // to 251 GB/s beside a second process's steps against 291 alone, and a 32-core M5 Max left alone 429.5 or
    // 450.9), the plans followed it, and one build's greedy output then differed from start to start. 0 for a
    // class the table does not hold: the matrix tile then takes no split (ops matrixSplits) and a decode
    // attention's page model no share of DRAM (ops::decodePassNanoseconds).
    uint64_t memoryBytesPerSecond = 0;

    [[nodiscard]] constexpr GpuClass gpuClass() const noexcept {
        return appleGpuFamily >= 11   ? GpuClass::Apple11
               : appleGpuFamily == 10 ? GpuClass::Apple10
                                      : GpuClass::Apple9;
    }
    // A class's memory rate: what five starts of its Mac read on a GPU held alone, at the nearest 1.05^n (the
    // higher of the M5 Max's two): a 10-core M4 109.6 GB/s, a 12-core M6 161.9, a 16- and a 20-core M5 Pro 290.7,
    // a 32-core M5 Max 450.9. A class not listed, or a device that reports no core count, has none (0).
    [[nodiscard]] static constexpr uint64_t classMemoryBytesPerSecond(GpuClass gpu, uint32_t gpuCores) noexcept {
        struct Class {
            GpuClass gpu;
            uint32_t cores;
            uint64_t bytesPerSecond;
        };
        constexpr Class kClasses[] = {{GpuClass::Apple9, 10, 109'553'136'383},  {GpuClass::Apple10, 16, 290'677'085'356},
                                      {GpuClass::Apple10, 20, 290'677'085'356}, {GpuClass::Apple10, 32, 450'935'564'252},
                                      {GpuClass::Apple11, 12, 161'859'877'733}};
        for (const Class &entry : kClasses)
            if (entry.gpu == gpu && entry.cores == gpuCores) return entry.bytesPerSecond;
        return 0;
    }
    // The GPU core count kernel policy plans for: the reported one, or
    // kAssumedGpuCores when the device does not report it.
    [[nodiscard]] constexpr uint32_t plannedGpuCores() const noexcept {
        return gpuCoreCount ? gpuCoreCount : kAssumedGpuCores;
    }

    [[nodiscard]] bool meetsMinimumMacos() const noexcept {
        return macosMajor > kMinimumMacosMajor ||
               (macosMajor == kMinimumMacosMajor &&
                macosMinor >= kMinimumMacosMinor);
    }
    // "major.minor.patch" of the running macOS for status and messages.
    [[nodiscard]] std::string macosVersion() const;

    // Returns a stable machine-readable reason, or nullopt when valid.
    [[nodiscard]] std::optional<std::string> validationError() const;
    // The same verdict as one line for a person: what Splash needs against
    // what this Mac has, ending with the reason above.
    [[nodiscard]] std::optional<std::string> validationMessage() const;
};

} // namespace splash
