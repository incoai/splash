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
    /// development build installs. `families` is what the runtime reports of their families, as JSON.
    static func launchedWithModels(
        _ fake: FakeRuntime,
        families: String = #"{"Qwen3.8-27B": {"unusable": []}, "Qwen3.6-35B-A3B": {"unusable": []}, "Flint": {"unusable": []}}"#
    ) async throws -> AppState {
        try FileManager.default.copyItem(at: Showcase.runtime.suggestedModels, to: Runtime(root: fake.root).suggestedModels)
        try "print('\(families)')".write(to: fake.root.appending(path: "install/families.py"), atomically: true, encoding: .utf8)
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

extension ServerScreenTests {
    /// A switch of models changes the control bar at once: each frame drawn through a change shows
    /// the bar of the state before it or of the state after it, never both over each other, as a
    /// cross-fade drew them. The check leaves the row picked before at once, never fading there,
    /// so that one shows at a time.
    @Test func aSwitchChangesTheControlBarAtOnce() async throws {
        let fake = try FakeRuntime("splash-switch", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake)
        defer { state.quit() }
        let flint = "unsloth/flint-GGUF:UD-Q4_K_XL", qwen = "mlx-community/Qwen3.8-27B-4bit"
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start(flint)
        try await waitFor(state.server.phase == .ready, "\(flint) ready")
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        /// The window drawn now, a row of pixels at a time, from the top.
        func rows() async throws -> [Data] {
            let picture = try #require(await Snapshot.picture(of: window))
            let data = try #require(picture.bitmapData)
            return (0..<picture.pixelsHigh).map { Data(bytes: data + $0 * picture.bytesPerRow, count: picture.bytesPerRow) }
        }
        let scale = CGFloat(try #require(await Snapshot.picture(of: window)).pixelsHigh) / window.frame.height
        // The bar's words and buttons: its bottom 48 points, under its divider, from 150 points
        // in, past the mark, which takes its pose within `glide`.
        let barRows = Int(48 * scale), skip = Int(150 * scale) * 4
        func bar(_ rows: [Data]) -> [Data] { rows.suffix(barRows).map { $0.dropFirst(skip) } }
        /// The frames drawn through `change`, as rows, with those before it and once it has changed
        /// and stopped changing, which a slow Mac, as a virtual one is, reaches long after the
        /// animations' length.
        func frames(of change: () -> Void) async throws -> (before: [Data], drawn: [[Data]], after: [Data]) {
            let before = try await rows()
            change()
            var drawn: [[Data]] = []
            for _ in 0..<8 {
                pump(0.03)
                drawn.append(try await rows())
            }
            var after = try await rows()
            for _ in 0..<40 {
                pump(0.25)
                let next = try await rows()
                if next == after && next != before { break }
                after = next
            }
            return (before, drawn, after)
        }

        let pick = try await frames { state.selection = qwen }
        #expect(bar(pick.before) != bar(pick.after))
        #expect(pick.drawn.allSatisfy { bar($0) == bar(pick.before) || bar($0) == bar(pick.after) }, "picking \(qwen)")
        // The rows of the screen that the pick changes, above the bar: those of the check that
        // goes, on Flint's row, under those of the one that comes, on Qwen3.8-27B's.
        let changed = (0..<(pick.after.count - barRows)).filter { pick.before[$0] != pick.after[$0] }
        let gap = try #require(zip(changed, changed.dropFirst()).max { $0.1 - $0.0 < $1.1 - $1.0 })
        let gone = changed.filter { $0 >= gap.1 }
        // A frame that the pick is yet to reach has the check there whole.
        #expect(!gone.isEmpty && pick.drawn.allSatisfy { frame in
            gone.allSatisfy { frame[$0] == pick.after[$0] } || gone.allSatisfy { frame[$0] == pick.before[$0] }
        }, "the check leaves Flint's row at once")

        let switching = try await frames { state.start(qwen) }
        #expect(bar(switching.before) != bar(switching.after))
        #expect(switching.drawn.allSatisfy { bar($0) == bar(switching.before) || bar($0) == bar(switching.after) },
                "switching to \(qwen)")
    }

    /// Picking a model whose family uses other launch options changes the page's height below the
    /// models, and moves nothing above: the page's document took the new height an update late, and
    /// its hosting view centered the content meanwhile, which moved the whole page by half the
    /// change and slid it back, the rows under the pointer passing it as they went.
    @Test func pickingAnotherFamilyMovesNothingAboveItsOptions() async throws {
        let fake = try FakeRuntime("splash-family-pick", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        // Flint's family as the runtime reports it (`ServeSettingsTests.flint`): without Text only, which
        // the Qwen families' models use, among the options of the page's first card.
        let unusable = ServeSettingsTests.flint.unusable.sorted().map { "\"\($0)\"" }.joined(separator: ", ")
        let state = try await Self.launchedWithModels(
            fake, families: #"{"Qwen3.8-27B": {"unusable": []}, "Qwen3.6-35B-A3B": {"unusable": []}, "Flint": {"unusable": [\#(unusable)]}}"#)
        defer { state.quit() }
        let flint = "unsloth/flint-GGUF:UD-Q4_K_XL", qwen = "mlx-community/Qwen3.8-27B-4bit"
        #expect(LaunchOptions.options(.main, for: state.family(of: flint)).map(\.flag)
                != LaunchOptions.options(.main, for: state.family(of: qwen)).map(\.flag))
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start(flint)
        try await waitFor(state.server.phase == .ready, "\(flint) ready")
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        /// The page's first 140 points under the toolbar, Activity's card, as drawn now.
        func top() async throws -> Data {
            let picture = try #require(await Snapshot.picture(of: window))
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height
            let first = Int(((window.frame.height - window.contentLayoutRect.maxY) * scale).rounded(.up))
            return Data(bytes: try #require(picture.bitmapData) + first * picture.bytesPerRow, count: Int(140 * scale) * picture.bytesPerRow)
        }
        for model in [qwen, flint] {
            let before = try await top()
            state.selection = model
            var moved = 0
            for _ in 0..<12 {
                pump(0.02)
                if try await top() != before { moved += 1 }
            }
            #expect(moved == 0, "picking \(model) moved the page's top in \(moved) of 12 frames")
        }
    }
}

extension ServerScreenTests {
    /// The page of `window` drawn through `change`: as drawn before it, every 30 ms or so for 0.3 s
    /// after it, each with how long after, and once it has stopped changing, which a slow Mac
    /// reaches long after the animations' length.
    static func pictures(of window: NSWindow, through change: () -> Void) async throws
        -> (before: NSBitmapImageRep, drawn: [(at: Duration, picture: NSBitmapImageRep)], after: NSBitmapImageRep) {
        let before = try #require(await Snapshot.picture(of: window))
        let start = ContinuousClock.now
        change()
        var drawn: [(at: Duration, picture: NSBitmapImageRep)] = []
        for _ in 0..<10 {
            pump(0.03)
            drawn.append((ContinuousClock.now - start, try #require(await Snapshot.picture(of: window))))
        }
        var after = try #require(await Snapshot.picture(of: window))
        for _ in 0..<40 {
            pump(0.25)
            let next = try #require(await Snapshot.picture(of: window))
            if next.bitmapData.map({ Data(bytes: $0, count: next.bytesPerRow * next.pixelsHigh) })
                == after.bitmapData.map({ Data(bytes: $0, count: after.bytesPerRow * after.pixelsHigh) }) { break }
            after = next
        }
        return (before, drawn, after)
    }

    /// The pixels of `picture`, of `window`, in the page's part between the toolbar and the control
    /// bar, from `left` to `right` points in from the window's left edge, a row at a time.
    static func page(_ picture: NSBitmapImageRep, of window: NSWindow, left: CGFloat = 0, right: CGFloat? = nil) -> [Data] {
        let scale = CGFloat(picture.pixelsHigh) / window.frame.height
        let top = Int(((window.frame.height - window.contentLayoutRect.maxY) * scale).rounded(.up))
        let bottom = Int((window.frame.height - 80) * scale)
        let bytes = picture.bitsPerPixel / 8
        let from = Int(left * scale) * bytes, to = Int((right ?? window.frame.width) * scale) * bytes
        guard let data = picture.bitmapData else { return [] }
        return (top..<bottom).map { Data(bytes: data + $0 * picture.bytesPerRow + from, count: to - from) }
    }

    /// Whether `picture` shows a warning sign's yellow anywhere in the page's part.
    static func showsWarning(_ picture: NSBitmapImageRep, of window: NSWindow) -> Bool {
        let bytes = picture.bitsPerPixel / 8
        return page(picture, of: window).contains { row in
            stride(from: 0, to: row.count - 2, by: bytes).contains { i in
                let (r, g, b) = (Int(row[row.startIndex + i]), Int(row[row.startIndex + i + 1]), Int(row[row.startIndex + i + 2]))
                return r > 180 && g > 130 && b < r - 100
            }
        }
    }

    /// The window of the Server screen of `state`, the model `picked` picked, drawn.
    static func serverWindow(_ state: AppState, picked: String, appearance: NSAppearance.Name = .aqua) async -> NSWindow {
        state.selection = picked
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: appearance)
        pump(0.5)
        return window
    }

    /// Qwen's and Flint's families as the runtime reports them: Flint without Text only, the
    /// thinking options and the others its models can't use, and serving at most its window; the
    /// Qwen families stretch theirs with YaRN.
    static let realisticFamilies = {
        let flint = ServeSettingsTests.flint.unusable.sorted().map { "\"\($0)\"" }.joined(separator: ", ")
        let moe = ServeSettingsTests.moe.unusable.sorted().map { "\"\($0)\"" }.joined(separator: ", ")
        let qwen = #""context_tokens": 262144, "max_context_tokens": 1010000"#
        return #"{"Qwen3.8-27B": {"unusable": ["--disable-ane", "--expert-memory", "--ple-memory", "--sparse-kv-memory"], \#(qwen)}, "#
            + #""Qwen3.6-35B-A3B": {"unusable": [\#(moe)], \#(qwen)}, "#
            + #""Flint": {"unusable": [\#(flint)], "context_tokens": 262144, "max_context_tokens": 262144}}"#
    }()

    /// A launch option's warning comes and goes as a row of the page does: a context past the
    /// model's window shows YaRN's caution only once the rows under it have moved down, and set
    /// back to Auto, the caution goes at once, the rows under it moving up into its place. It
    /// faded in over the rows still moving and out under those moving into its place.
    @Test func anOptionsWarningComesAndGoesAsARowDoes() async throws {
        let fake = try FakeRuntime("splash-option-warning", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake, families: Self.realisticFamilies)
        defer { state.quit() }
        let window = await Self.serverWindow(state, picked: "mlx-community/Qwen3.8-27B-4bit")
        defer { window.orderOut(nil) }
        // Past the window of 256K, short of the most, 1M.
        let coming = try await Self.pictures(of: window) { state.options["--max-context"] = "524288" }
        #expect(Self.showsWarning(coming.after, of: window), "the caution shows once the rows have moved")
        // Drawn within the rows' move, `settle` (0.24 s).
        let moving = coming.drawn.filter { $0.at < .milliseconds(200) }
        let early = moving.filter { Self.showsWarning($0.picture, of: window) }.count
        #expect(!moving.isEmpty && early == 0, "the caution showed in \(early) of \(moving.count) frames as the rows moved down")
        let going = try await Self.pictures(of: window) { state.options["--max-context"] = nil }
        let shown = going.drawn.filter { Self.showsWarning($0.picture, of: window) }.count
        #expect(shown == 0, "the caution showed in \(shown) of \(going.drawn.count) frames after it went")
    }

    /// Restore Defaults coming and going moves nothing on the page: the launch options' header
    /// keeps its height. The button is taller than the header's title, and the card under it moved
    /// as it came and went.
    @Test func restoreDefaultsComingAndGoingMovesNothing() async throws {
        let fake = try FakeRuntime("splash-restore-defaults", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake, families: Self.realisticFamilies)
        defer { state.quit() }
        let window = await Self.serverWindow(state, picked: "mlx-community/Qwen3.8-27B-4bit")
        defer { window.orderOut(nil) }
        let limit = try #require(ServeSettings.memoryLimits(memory: Catalog.memory).first)
        for value in ["\(limit)G", nil] {
            // The page's left part, the options' names: none of it changes.
            let change = try await Self.pictures(of: window) { state.options["--max-memory"] = value }
            let before = Self.page(change.before, of: window, right: 400)
            let moved = (change.drawn.map(\.picture) + [change.after]).filter { Self.page($0, of: window, right: 400) != before }.count
            #expect(moved == 0, "setting the memory limit to \(value ?? "Auto") moved the page in \(moved) frames")
        }
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

/// A switch of models on the Server screen drawn to `switch-*.png` in `SPLASH_APP_SNAPSHOTS`,
/// light and dark, for people to review its motion: each change of it, the pick, the stop, the
/// start, the warm-up and ready, at 0, 25, 50, 75 and 100% of `settle`, and at twice that, once a
/// section that comes has faded in.
@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct ServerSwitchPictures {
    @Test func aSwitchFrameByFrame() async throws {
        let directory = try #require(Snapshot.directory)
        for (suffix, appearance) in [("light", NSAppearance.Name.aqua), ("dark", .darkAqua)] {
            let fake = try FakeRuntime("splash-switch-pictures", launcher: "stepped_server")
            defer { try? FileManager.default.removeItem(at: fake.root) }
            let state = try await ServerScreenTests.launchedWithModels(fake)
            defer { state.quit() }
            let flint = "unsloth/flint-GGUF:UD-Q4_K_XL", qwen = "mlx-community/Qwen3.8-27B-4bit"
            state.options["--port"] = String(fake.port)
            func signal(_ step: String) { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
            signal("loaded")
            signal("warm")
            state.start(flint)
            try await waitFor(state.server.phase == .ready, "\(flint) ready")
            let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: appearance)
            defer { window.orderOut(nil) }
            pump(0.5)
            func frames(_ name: String) async {
                let start = ContinuousClock.now
                for percent in [0, 25, 50, 75, 100, 200] {
                    while ContinuousClock.now < start + .milliseconds(percent * 24 / 10) { pump(0.005) }
                    try? await Snapshot.picture(of: window)?.representation(using: .png, properties: [:])?
                        .write(to: directory.appending(path: "switch-\(name)-\(percent)-\(suffix).png"))
                }
            }
            /// Waits for `condition`, letting the app's own work on the main actor run meanwhile.
            func until(_ condition: () -> Bool) async {
                while !condition() {
                    pump(0.002)
                    await Task.yield()
                }
            }
            state.selection = qwen
            await frames("1-pick")
            for step in ["loaded", "warm"] { try FileManager.default.removeItem(at: fake.root.appending(path: step)) }
            state.start(qwen)
            await frames("2-stop")
            await until { state.server.phase == .starting && state.server.model == qwen }
            await frames("3-start")
            signal("loaded")
            await until { state.server.step == .warming }
            await frames("4-warm")
            signal("warm")
            await until { state.server.phase == .ready }
            await frames("5-ready")
        }
    }
}
