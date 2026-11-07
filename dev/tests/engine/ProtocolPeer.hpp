#pragma once

#include "engine/Protocol.hpp"

#include <cstdint>
#include <span>
#include <vector>

// The server's side of the native wire, for tests that drive the engine;
// production is server/protocol.py.
namespace splash::protocol::peer {

// A Ready an engine under test announces where what it says is not what the
// test checks: four lanes, the test model's 1024 tokens and 64-token
// vocabulary, one stop token and the mask rows of a draft of eight.
inline const ReadyEvent kTestReady{4, 1024, false, 64, {1}, 9};

// Header and payload exactly as server/protocol.py writes them, without
// validation, so tests can send the engine invalid frames.
[[nodiscard]] std::vector<uint8_t> serialize(const ClientMessage &message);

// Splits bytes into complete frames and decodes each event, checking only
// what it must to read the payload. Throws std::runtime_error on anything it
// cannot read.
[[nodiscard]] std::vector<EngineEvent>
decodeEvents(std::span<const uint8_t> bytes);

// Splits an event stream whose reads may end inside a frame.
class EventReader final {
public:
  // The events the bytes complete, decoded as decodeEvents() does; a partial
  // frame waits for the bytes that follow it.
  [[nodiscard]] std::vector<EngineEvent> feed(std::span<const uint8_t> bytes);

private:
  std::vector<uint8_t> pending_;
};

} // namespace splash::protocol::peer
