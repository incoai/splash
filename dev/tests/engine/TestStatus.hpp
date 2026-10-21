#pragma once

#include "engine/Protocol.hpp"

#include <string>

namespace splash::test {

// What a ready runtime's status provider returns, at the current schema.
inline std::string readyStatusJson() {
  return "{\"schema_version\":" +
         std::to_string(protocol::kStatusSchemaVersion) + ",\"ready\":true}";
}

} // namespace splash::test
