#include "video/metal/DeviceProfile.hpp"

#include <format>
#include <stdexcept>

namespace splash::video::device {

const Profile &selectProfile(const DeviceCapabilities &caps) {
  const Profile *chosen = nullptr;
  for (const Profile *p : kProfiles)
    if (p->appleGpuFamily <= caps.appleGpuFamily) chosen = p;  // ascending: the highest at or below the device's
  if (!chosen)
    throw std::invalid_argument(std::format("no video profile for Apple GPU family {} (the lowest is {}, family {})",
                                            caps.appleGpuFamily, kProfiles[0]->name, kProfiles[0]->appleGpuFamily));
  return *chosen;
}

bool untunedFamily(const DeviceCapabilities &caps) {
  for (const Profile *p : kProfiles)
    if (p->appleGpuFamily >= caps.appleGpuFamily) return false;
  return true;
}

std::string profileLine(const Profile &profile, const DeviceCapabilities &caps) {
  return std::format("H3 video profile {} device {} family {} cores {}", profile.name, caps.deviceName,
                     caps.appleGpuFamily, caps.gpuCoreCount);
}

std::string untunedFamilyLine(const Profile &profile, const DeviceCapabilities &caps) {
  return std::format("untuned GPU family {}: video profile {}", caps.appleGpuFamily, profile.name);
}

} // namespace splash::video::device
