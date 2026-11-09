import Foundation
import Testing

@testable import Splash

/// What the Server screen, the menu bar's panel and the composer offer as models start, fail,
/// serve and are deleted.
@MainActor
@Suite(.serialized)
struct ServerScreenTests {
    /// A start that failed leaves nothing running, nothing to stop, and the failure what would let
    /// it start; the bar's Try Again starts the model again. Dismissing the failure clears it with
    /// what it offered.
    @Test func aFailedStartLeavesNothingRunningUntilTriedAgain() async throws {
        let fake = try FakeRuntime("splash-failed-start", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        try fake.install("a/b")
        // A launcher whose start fails for want of memory, as its serve record says.
        try """
        import json, os, sys
        from pathlib import Path
        port = int(sys.argv[sys.argv.index("--port") + 1])
        record = Path(__file__).resolve().parents[1] / "build/runtime" / f"serve-{port}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"pid": os.getpid(), "phase": "failed", "failure": {
            "kind": "memory_budget", "bound": "max_memory", "message": "x", "required_bytes": 1 << 30, "budget_bytes": 1}}))
        sys.exit(1)
        """.write(to: fake.root.appending(path: "install/launcher.py"), atomically: true, encoding: .utf8)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        state.options["--port"] = String(fake.port)
        state.start("a/b")
        try await waitFor(state.serverStatus.look == .failed, "the failure")
        // The least memory limit that holds the model.
        let remedy = Server.Remedy.memoryLimit(ServeSettings.memoryLimits(memory: Catalog.memory).first.map { "\($0)G" })
        #expect(!state.server.running && state.stopItem == nil && state.server.remedy == remedy)
        // Try Again, the bar's one button, starts it again.
        #expect(state.selection == "a/b")
        let failed = state.server.since
        state.start("a/b")
        try await waitFor(state.serverStatus.look == .failed && state.server.since > failed, "the second failure")
        state.server.dismissFailure()
        #expect(state.server.phase == .stopped && state.server.remedy == nil && !state.server.running)
    }
}

extension ServerScreenTests {
    /// A model being deleted doesn't start, from the menu bar's panel, the composer or the Server's
    /// bar, while the others do (`canStart`): choosing it leaves the model that runs as it is, and
    /// the model picked being deleted holds none of the others back.
    @Test func aModelBeingDeletedDoesNotStartWhileOthersDo() async throws {
        let fake = try FakeRuntime("splash-deleting", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let first = "a/First-GGUF:Q4", second = "a/Second-GGUF:Q4"
        try fake.install(first)
        try fake.install(second)
        let hold = fake.root.appending(path: "holding")
        // An installer whose removal runs while the test holds it.
        try """
        import pathlib, time
        while pathlib.Path("\(hold.path)").exists():
            time.sleep(0.02)
        """.write(to: fake.root.appending(path: "install/models.py"), atomically: true, encoding: .utf8)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        func delete(_ model: String) {
            FileManager.default.createFile(atPath: hold.path, contents: nil)
            state.delete(model)
        }

        // The second picked and serving, the first being deleted.
        state.start(second)
        try await waitFor(state.server.phase == .ready, "\(second) ready")
        delete(first)
        #expect(state.removing(first))
        #expect(!state.canStart(first) && state.canStart(second))
        state.choose(first)
        #expect(state.server.phase == .ready && state.server.model == second && state.server.next == nil)
        #expect(state.selection == second)
        try FileManager.default.removeItem(at: hold)
        try await waitFor(!state.installer.running, "the deletion")

        // The first picked and being deleted, nothing serving: the second starts.
        state.stop()
        try await waitFor(!state.server.running, "the stop")
        state.selection = first
        delete(first)
        #expect(!state.canStart(first) && state.canStart(second))
        state.choose(second)
        #expect(state.server.running && state.server.model == second && state.selection == second)
        try FileManager.default.removeItem(at: hold)
        try await waitFor(!state.installer.running, "the deletion")
    }
}
