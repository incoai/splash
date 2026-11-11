import AppKit
import Foundation
import SwiftUI
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
    /// The server's line for its own failure of a request the engine took names the request, as
    /// the backend's lines do: it ends that request's row, which stays the one row.
    @Test func aFailureLineThatNamesARequestEndsItsRow() throws {
        let started = try #require(RequestEntry("12:00:00 Started · request 7 · input 1,200 · cached 0"))
        let failed = try #require(RequestEntry("12:00:03 Error · request 7 · request_timeout · POST /v1/chat/completions"))
        #expect(failed.request == 7 && failed.message == "request_timeout · POST /v1/chat/completions")
        var rows = ServedRequest.taking(started, at: .now, into: [])
        rows = ServedRequest.taking(failed, at: .now, into: rows)
        #expect(rows.map(\.id) == [7] && rows.first?.end == failed)
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

extension ServerScreenTests {
    /// Model files on this Mac whose start with a projector and a draft chosen for them fails as
    /// files Splash can't serve start again without them, which they no longer start with; failing
    /// without, they are offered to be taken off the list.
    @Test func modelFilesThatFailStartAgainWithoutTheirProjectorAndDraft() async throws {
        let fake = try FakeRuntime("splash-files-failed", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        // A launcher whose start fails on files Splash can't serve, as its serve record says.
        try """
        import json, os, sys
        from pathlib import Path
        port = int(sys.argv[sys.argv.index("--port") + 1])
        record = Path(__file__).resolve().parents[1] / "build/runtime" / f"serve-{port}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"pid": os.getpid(), "phase": "failed", "failure": {"kind": "model", "message": "x"}}))
        sys.exit(1)
        """.write(to: fake.root.appending(path: "install/launcher.py"), atomically: true, encoding: .utf8)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        state.options["--port"] = String(fake.port)
        let files = "/m/q.gguf"
        state.modelFiles[files] = AppState.ModelFiles(projector: "/m/mmproj.gguf", draft: "/m/draft")
        state.start(files)
        try await waitFor(state.serverStatus.look == .failed, "the failure")
        let remedy = try #require(state.server.remedy)
        #expect(remedy == .without(projector: true, draft: true))
        let failed = state.server.since
        state.retry(files, with: remedy)
        #expect(state.modelFiles[files] == AppState.ModelFiles())
        try await waitFor(state.serverStatus.look == .failed && state.server.since > failed, "the start without them")
        #expect(state.server.settings.projector == nil && state.server.settings.draft == nil)
        #expect(state.server.remedy == .removeFromList)
    }
}

extension ServerScreenTests {
    /// The toolbar has no band of its own (`MainView`): the Server screen, scrolled so that its first
    /// rows would be where the toolbar is, shows nothing of them there, in either appearance; the
    /// toolbar's band draws as it does at the screen's top.
    @Test func theServerScrolledShowsNothingUnderTheToolbar() async throws {
        let fake = try FakeRuntime("splash-scrolled", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake)
        defer { state.quit() }
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: appearance)
            defer { window.orderOut(nil) }
            let scroll = try #require(window.contentView?.pageScroll)
            let top = try #require(await Snapshot.picture(of: window))
            let start = scroll.contentView.bounds.minY
            scroll.scroll(by: 90)
            #expect(scroll.contentView.bounds.minY == start + 90, "the screen is taller than its window")
            let scrolled = try #require(await Snapshot.picture(of: window))
            #expect(try Self.toolbarBand(of: scrolled, in: window) == Self.toolbarBand(of: top, in: window), "\(appearance.rawValue)")
        }
    }

    /// The rows of `picture`, of `window` drawn whole, that its title bar and toolbar take: those
    /// above its content's layout rect.
    static func toolbarBand(of picture: NSBitmapImageRep, in window: NSWindow) throws -> Data {
        let points = window.frame.height - window.contentLayoutRect.maxY
        let rows = Int((points * CGFloat(picture.pixelsHigh) / window.frame.height).rounded(.down))
        return Data(bytes: try #require(picture.bitmapData), count: rows * picture.bytesPerRow)
    }

    /// An app launched on `fake` with the models of a Mac that has a few, named and sized as Splash
    /// suggests them, of the families the runtime serves, so that the Server screen is taller than
    /// the window, and two speech models, of families the runtime doesn't serve, which a newer or
    /// development build installs.
    static func launchedWithModels(_ fake: FakeRuntime) async throws -> AppState {
        try FileManager.default.copyItem(at: Showcase.runtime.suggestedModels, to: Runtime(root: fake.root).suggestedModels)
        try #"print('{"Qwen3.8-27B": {"unusable": []}, "Qwen3.6-35B-A3B": {"unusable": []}, "Flint": {"unusable": []}}')"#
            .write(to: fake.root.appending(path: "install/families.py"), atomically: true, encoding: .utf8)
        let models = [("mlx-community/Qwen3.8-27B-4bit", "Qwen3.8-27B", 20), ("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M", "Qwen3.6-35B-A3B", 24),
                      ("unsloth/flint-GGUF:UD-Q4_K_XL", "Flint", 26),
                      ("mlx-community/Qwen3-ASR-1.7B-4bit", "Qwen3-ASR", 1), ("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit", "Qwen3-TTS", 2)]
        for (model, family, gigabytes) in models {
            try fake.install(model, family: family, bytes: Format.gigabytes(gigabytes))
        }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        await state.launch()
        return state
    }
}

/// The Server screen drawn to `server-*.png` in `SPLASH_APP_SNAPSHOTS`, light and dark, for people
/// to review its look with the models of a Mac that has a few: at its top, and scrolled so that its
/// first rows would be where the toolbar is.
@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct ServerPictures {
    /// About the window's least size, at which the column the screen's cards are in reaches under
    /// the toolbar's controls at its start.
    static let size = CGSize(width: 820, height: 620)

    @Test func atItsTopAndScrolled() async throws {
        let fake = try FakeRuntime("splash-server-pictures", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await ServerScreenTests.launchedWithModels(fake)
        defer { state.quit() }
        await Snapshot.save("server-top", size: Self.size, MainView().environment(state))
        await Snapshot.save("server-scrolled", size: Self.size, MainView().environment(state)) { window in
            window.contentView?.pageScroll?.scroll(by: 90)
            try? await Task.sleep(for: .seconds(0.3))
        }
    }
}
