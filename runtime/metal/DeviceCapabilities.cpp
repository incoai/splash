#include "metal/DeviceCapabilities.hpp"

#include <string>

namespace splash {

std::string DeviceCapabilities::macosVersion() const {
    return std::to_string(macosMajor) + '.' + std::to_string(macosMinor) + '.' +
           std::to_string(macosPatch);
}

std::optional<std::string> DeviceCapabilities::validationError() const {
    // The operating system explains a missing placement-sparse query, so it
    // is reported ahead of every device feature.
    if (!meetsMinimumMacos()) return "macos_26_4_required";
    if (!physicalMemoryBytes) return "physical_memory_unavailable";
    if (!recommendedMaxWorkingSetBytes) {
        return "recommended_working_set_unavailable";
    }
    if (recommendedMaxWorkingSetBytes > physicalMemoryBytes) {
        return "recommended_working_set_exceeds_physical_memory";
    }
    if (!maxBufferLengthBytes) return "max_buffer_length_unavailable";
    if (appleGpuFamily < kMinimumAppleGpuFamily) return "apple_gpu_family_8_required";
    if (maxThreadgroupMemoryBytes < 32 * 1024) {
        return "threadgroup_memory_below_32_kib";
    }
    if (maxThreadgroupWidth < 256) {
        return "threadgroup_width_below_256";
    }
    if (!hasUnifiedMemory) return "unified_memory_required";
    if (!supportsPlacementSparse) return "placement_sparse_required";
    return std::nullopt;
}

} // namespace splash
