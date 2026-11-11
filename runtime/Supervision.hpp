#pragma once

#include <chrono>

// Every time limit one Splash process holds another to, and those the engine
// holds itself to, in one place. server/protocol.py SUPERVISION mirrors the
// ones between the server and the engine, the app mirrors the stop budget (its
// Supervision.stop), dev/tests/engine/protocol_golden.txt ("supervision")
// pins them on every side, and server/supervision.py derives each threshold
// by which the server fails, stops or relaunches an engine from them.
//
// Stopping, from the outside in:
// 1. The engine, on SIGTERM or the end of its input, stops taking work,
//    gives the newest restore points of a persistent cache up to kFlush to
//    reach the disk, and exits.
// 2. The server stops its engine with SIGTERM, sends SIGKILL to one still
//    running after kStop, then waits up to kStop more for it to exit.
// 3. The server, on SIGTERM or a first SIGINT, stops as in 2; later SIGTERMs
//    are ignored, and a second SIGINT while it stops (the user's double
//    Ctrl+C) kills the engine at once.
// 4. The app stops a server with one signal, SIGINT for its own and SIGTERM
//    for a leftover one (which may already be stopping), and kills the
//    process group only after 2 * kStop, the longest 2 takes. A second
//    signal before kStop could cut a flush 1 allows; one after it adds
//    nothing the server does not do itself.
namespace splash::supervision {

using std::chrono::milliseconds;

// Metal fails an event wait still unmet this long after its queue reached it
// ("Caused GPU Timeout Error"), and the command with it: the work between two
// waits of a command, with the Neural Engine's part, must end within it
// (metal::EventStep). No wait spans a host step, which may take longer (an
// expert load from a slow disk). Metal's limit, not ours.
inline constexpr milliseconds kGpuWaitLimit{5'000};
// A command that makes no progress this long fails the engine
// (metal::CommandWatchdog): a GPU hang, a lost completion, or a host step
// that does not return (a disk that stopped answering), since Metal fails a
// slow Neural Engine wait first.
inline constexpr milliseconds kCommandTimeout{120'000};
// The longest one iteration of the native loop takes (FdTransport::run): its
// input, a control pass and a tick. A frame waits at most this long before
// the loop takes it, and a status request is answered within it.
inline constexpr milliseconds kLoopStall{30'000};
// How long the engine waits for the server's answer to a token-mask request
// before it fails the request (Engine::tick).
inline constexpr milliseconds kMaskWait{5'000};
// What a clean stop gives the newest restore points of a persistent cache to
// reach the disk (main.mm).
inline constexpr milliseconds kFlush{6'000};
// From a stop signal to the engine's exit, its flush included; the server
// kills an engine still running after it.
inline constexpr milliseconds kStop{15'000};
// A start serves on probation this long: a persistent cache it took back is
// dropped if the engine fails within it (RuntimeResources::beginServing), and
// the server counts a failure within it as short-lived, which doubles its
// relaunch backoff, up to this long.
inline constexpr milliseconds kProbation{60'000};
// How long a start waits for memory the host or Metal refused before it
// fails (engine::StartupRetryWindow); a relaunch past its usual start may
// need this much more.
inline constexpr milliseconds kStartRetryWindow{30'000};

static_assert(kGpuWaitLimit < kCommandTimeout);
static_assert(kFlush < kStop);
static_assert(kLoopStall < kProbation);
static_assert(kStartRetryWindow < kProbation);

} // namespace splash::supervision
