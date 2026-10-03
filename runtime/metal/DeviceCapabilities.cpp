#include "metal/DeviceCapabilities.hpp"

#include <string>

namespace splash {

std::string DeviceCapabilities::macosVersion() const {
    return std::to_string(macosMajor) + '.' + std::to_string(macosMinor) + '.' +
           std::to_string(macosPatch);
}

std::optional<std::string> DeviceCapabilities::validationError() const {
    // As at startup, the operating system is checked before the device.
    if (!meetsMinimumMacos()) return "macos_26_4_required";
    if (!physicalMemoryBytes) return "physical_memory_unavailable";
    if (!recommendedMaxWorkingSetBytes) {
        return "recommended_working_set_unavailable";
    }
    if (recommendedMaxWorkingSetBytes > physicalMemoryBytes) {
        return "recommended_working_set_exceeds_physical_memory";
    }
    if (!maxBufferLengthBytes) return "max_buffer_length_unavailable";
    if (appleGpuFamily < kMinimumAppleGpuFamily) return "apple_gpu_family_9_required";
    if (maxThreadgroupMemoryBytes < 32 * 1024) {
        return "threadgroup_memory_below_32_kib";
    }
    if (maxThreadgroupWidth < 256) {
        return "threadgroup_width_below_256";
    }
    if (!hasUnifiedMemory) return "unified_memory_required";
    return std::nullopt;
}

std::optional<std::string> DeviceCapabilities::validationMessage() const {
    const std::optional<std::string> error = validationError();
    if (!error) return std::nullopt;
    // People know their chip, not its GPU family.
    static_assert(kMinimumAppleGpuFamily == 9, "name the family's first chip");
    const std::string family =
        appleGpuFamily ? "Apple GPU family " + std::to_string(appleGpuFamily)
                       : "no known Apple GPU family";
    return "Splash needs Apple GPU family " +
           std::to_string(kMinimumAppleGpuFamily) +
           " or newer (M3 or later) on macOS " +
           std::to_string(kMinimumMacosMajor) + '.' +
           std::to_string(kMinimumMacosMinor) + " or newer; this Mac has " +
           deviceName + " (" + family + ") on macOS " + macosVersion() + " (" +
           *error + ')';
}

} // namespace splash
