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

extension ServerScreenTests {
    /// A request's row follows the engine as each status says: back to waiting for memory as the
    /// engine suspends it, then reading its prompt again. A status the server answers from a
    /// snapshot older than what a row shows moves it not; one from a newer one, as through a long
    /// prefill, does. One that no longer has a request seen before it ends its row, whose line
    /// never came, and a quit no longer counts it. Each status here has a request of its own, first
    /// seen in it, to tell when it is taken.
    @Test func aRequestsRowFollowsTheEngineBackAndEndsWithItsStatus() async throws {
        let fake = try FakeRuntime("splash-request-rows", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        func signal(_ name: String, _ text: String = "") throws {
            try text.write(to: fake.root.appending(path: name), atomically: true, encoding: .utf8)
        }
        func request(_ id: Int, _ phase: String, read: Int = 0, written: Int = 0) -> String {
            #"{"id": \#(id), "phase": "\#(phase)", "prompt_tokens": 900, "prompt_processed": \#(read), "generated_tokens": \#(written), "age_ms": 100}"#
        }
        /// Has the status say `requests`, and `marker` first seen, and waits until a status has.
        func status(_ requests: [String], marker: Int) async throws {
            try signal("running.json", "[\((requests + [request(marker, "queued")]).joined(separator: ", "))]")
            try await waitFor(state.server.requests.contains { $0.id == marker }, "the status with request \(marker)")
        }
        func stage(_ id: Int) -> LiveRequest.Stage? { state.server.requests.first { $0.id == id }?.stage }
        func running() -> [Int] { state.server.requests.filter(\.running).map(\.id) }
        state.server.start("example/Model", settings: ServeSettings(values: ["--port": String(fake.port)]))
        try signal("loaded")
        try signal("warm")
        try await waitFor(state.server.phase == .ready, "ready")
        try await status([request(1, "decode", read: 900, written: 52)], marker: 11)
        #expect(stage(1) == .answering(52))
        // Suspended for memory, then its prompt read again.
        try await status([request(1, "waiting_resources", read: 900, written: 52)], marker: 12)
        #expect(stage(1) == .waiting("Waiting for memory"))
        try await status([request(1, "prefill", read: 300, written: 52)], marker: 13)
        #expect(stage(1) == .reading(300))
        // From a snapshot ten minutes old, which has it answering: each status since ended the
        // marker before it.
        try signal("stale")
        try await status([request(1, "decode", read: 900, written: 60)], marker: 14)
        #expect(stage(1) == .reading(300))
        #expect(running() == [1, 13, 14] && state.quitCost.requests == 3)
        // From a snapshot a moment old, newer than the rows: it moves request 1 on, and the
        // markers it no longer has ended.
        try signal("stale", "100")
        try await status([request(1, "prefill", read: 600, written: 52)], marker: 15)
        #expect(stage(1) == .reading(600) && running() == [1, 15])
        // Taken then, without any of them but its marker: they ended, their lines never come.
        try FileManager.default.removeItem(at: fake.root.appending(path: "stale"))
        try await status([], marker: 16)
        #expect(running() == [16])
        #expect(state.server.requests.first { $0.id == 1 }?.end == nil && state.quitCost.requests == 1)
    }
}

extension ServerScreenTests {
    /// The Requests row counts the requests the model answered, as the status counts them and the
    /// menu bar's panel says: the rows stop at the latest 1,000 and hold refused requests too. Its
    /// rows say how many run.
    @Test func theRequestsRowCountsTheRequestsAnswered() throws {
        let refused = try #require(RequestEntry("12:00:01 Error · not_found · HEAD /api/hello"))
        var rows = (1...ServedRequest.kept).map { ServedRequest(id: $0, time: "12:00:00", stage: .answering(1)) }
        rows[0].end = refused
        let activity = Server.Activity(served: 1500, generated: 2_000_000, read: 9_000_000)
        #expect(RequestLog.totals(activity, requests: rows)
            == "\(1500.formatted()) · \(Format.count(2_000_000)) tokens out · \(Format.count(9_000_000)) in · \(999.formatted()) running")
        #expect(RequestLog.totals(Server.Activity(), requests: []) == nil)
        #expect(RequestLog.totals(Server.Activity(), requests: [rows[0]]) == "0 · 0 tokens out · 0 in")
    }
}
