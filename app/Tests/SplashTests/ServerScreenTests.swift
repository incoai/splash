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

    /// A model's phrase changes in place as the server starts it: every frame of the page shows the
    /// row as it was or as it is once the phrase has changed. Animated with the page, the phrase
    /// slid in over the check.
    @Test func aModelsPhraseChangesInPlace() async throws {
        let fake = try FakeRuntime("splash-phrase", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake, families: Self.realisticFamilies)
        defer { state.quit() }
        let qwen = "mlx-community/Qwen3.8-27B-4bit"
        state.options["--port"] = String(fake.port)
        let window = await Self.serverWindow(state, picked: qwen)
        defer { window.orderOut(nil) }
        let start = try await Self.pictures(of: window) { state.start(qwen) }
        let before = Self.page(start.before, of: window), after = Self.page(start.after, of: window)
        #expect(state.server.phase == .starting && before != after)
        let between = start.drawn.filter { Self.page($0.picture, of: window) != before && Self.page($0.picture, of: window) != after }.count
        #expect(between == 0, "\(between) of \(start.drawn.count) frames drew the phrase between none and Starting…")
    }

    /// A launch option's value changes at once, as a control's does: every frame of the page shows
    /// the value before or the value chosen, never one faded over the other.
    @Test func anOptionsValueChangesAtOnce() async throws {
        let fake = try FakeRuntime("splash-option-value", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake, families: Self.realisticFamilies)
        defer { state.quit() }
        // Restore Defaults showing before, so that only the value changes.
        state.options["--port"] = String(fake.port)
        let window = await Self.serverWindow(state, picked: "mlx-community/Qwen3.8-27B-4bit")
        defer { window.orderOut(nil) }
        let limit = try #require(ServeSettings.memoryLimits(memory: Catalog.memory).first)
        for value in ["\(limit)G", nil] {
            let change = try await Self.pictures(of: window) { state.options["--max-memory"] = value }
            let before = Self.page(change.before, of: window), after = Self.page(change.after, of: window)
            #expect(before != after)
            let between = change.drawn.filter { Self.page($0.picture, of: window) != before && Self.page($0.picture, of: window) != after }.count
            #expect(between == 0, "\(between) of \(change.drawn.count) frames drew the memory limit between the value before and \(value ?? "Auto")")
        }
    }

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

    /// How far the page of `rows` is drawn below where `reference` has it, in points: the shift that
    /// lines up the most of their rows that aren't one color throughout.
    static func shift(of rows: [Data], from reference: [Data], scale: CGFloat) -> CGFloat {
        let drawn = rows.indices.filter { Set(rows[$0]).count > 2 }
        let best = (-Int(160 * scale)...Int(160 * scale)).max { a, b in
            func lined(_ d: Int) -> Int { drawn.count { $0 - d >= 0 && $0 - d < reference.count && rows[$0] == reference[$0 - d] } }
            return lined(a) < lined(b)
        } ?? 0
        return CGFloat(best) / scale
    }

    /// A notice going above while the page shows its end moves nothing in view: the page scrolls to
    /// its new end with its content. The scroll view clamped its scroll to the shorter content at
    /// once while the content moved up within `settle`: the page jumped down by the notice's height
    /// and slid back up.
    @Test func aNoticeGoingAboveMovesNothingInView() async throws {
        let fake = try FakeRuntime("splash-notice-above", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.launchedWithModels(fake, families: Self.realisticFamilies)
        defer { state.quit() }
        let qwen = "mlx-community/Qwen3.8-27B-4bit"
        state.options["--port"] = String(fake.port)
        // A start the engine refuses, whose failure the page shows above the models.
        for name in ["loaded", "refused"] { FileManager.default.createFile(atPath: fake.root.appending(path: name).path, contents: nil) }
        state.start(qwen)
        try await waitFor(state.serverStatus.look == .failed, "the failure")
        let window = await Self.serverWindow(state, picked: qwen)
        defer { window.orderOut(nil) }
        // At the page's end.
        let scroll = try #require(window.contentView?.pageScroll)
        let end = try #require(scroll.documentView).frame.height - scroll.contentView.bounds.height + scroll.contentInsets.bottom
        scroll.scroll(by: end - scroll.contentView.bounds.minY)
        pump(0.3)
        let scale = CGFloat(try #require(await Snapshot.picture(of: window)).pixelsHigh) / window.frame.height
        let dismiss = try await Self.pictures(of: window) { state.server.dismissFailure() }
        let before = Self.page(dismiss.before, of: window)
        let shifts = (dismiss.drawn.map(\.picture) + [dismiss.after]).map { Self.shift(of: Self.page($0, of: window), from: before, scale: scale) }
        // The scroll moves on the display's frames, and a picture is taken between them: less than
        // a frame of `settle`'s motion at its fastest, some 16 points for a notice's 100.
        #expect(shifts.allSatisfy { abs($0) <= 16 }, "the page in view moved by \(shifts) points")
    }
}

/// Sections of a grouped page whose rows come and go as a test says: a section whose card comes
/// with its first row, under its header, between two that stay.
@MainActor @Observable final class GroupedSample {
    var card = false
}

private struct GroupedSampleSections: View {
    let sample: GroupedSample

    var body: some View {
        Group {
            GroupedSection { Text("Above") }
            GroupedSection {
                if sample.card { Color.red.frame(height: 60) }
            } header: {
                Text("Comes")
            }
            GroupedSection { Text("Below") }
        }
        .animation(Motion.settle, value: sample.card)
    }
}

extension ServerScreenTests {
    /// Whether `picture` shows a tint of the color channel `channel` (0 red, 1 green, 2 blue) over
    /// the others in the page's part of `window`.
    static func shows(channel: Int, in picture: NSBitmapImageRep, of window: NSWindow) -> Bool {
        let bytes = picture.bitsPerPixel / 8
        return page(picture, of: window).contains { row in
            stride(from: 0, to: row.count - 2, by: bytes).contains { i in
                let c = (0..<3).map { Int(row[row.startIndex + i + $0]) }
                return (0..<3).allSatisfy { $0 == channel || c[channel] - c[$0] > 40 }
            }
        }
    }

    /// A grouped page with the sample's sections, drawn.
    static func groupedSample(_ sample: GroupedSample) async -> (NSWindow, URL) {
        let (state, folder) = AppState.forTests()
        let window = await Snapshot.window(GroupedPage(column: 640) { GroupedSampleSections(sample: sample) }.environment(state),
                                           size: ServerPictures.size, appearance: .aqua)
        pump(0.3)
        return (window, folder)
    }

    /// A section's card that comes with its first row comes as a row does, once what makes room
    /// for it has moved: it faded in at once over the rows still moving down, as the download's
    /// under the first run's welcome.
    @Test func aSectionsCardComesOnceThereIsRoom() async throws {
        let sample = GroupedSample()
        let (window, folder) = await Self.groupedSample(sample)
        defer {
            window.orderOut(nil)
            try? FileManager.default.removeItem(at: folder)
        }
        let change = try await Self.pictures(of: window) { sample.card = true }
        #expect(Self.shows(channel: 0, in: change.after, of: window))
        let moving = change.drawn.filter { $0.at < .milliseconds(200) }
        let early = moving.filter { Self.shows(channel: 0, in: $0.picture, of: window) }.count
        #expect(!moving.isEmpty && early == 0, "the card showed in \(early) of \(moving.count) frames as the rows moved")
    }
}

extension ServerScreenTests {
    /// On the first run, the suggestion downloaded goes from the page at once: once the page has
    /// moved to make room for the download's row, nothing of it shows where it was. It stayed drawn
    /// there, over the download's row fading in, until the change's animations ended.
    @Test func theFirstDownloadsSuggestionGoesAtOnce() async throws {
        let fake = try FakeRuntime("splash-first-download", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        try FileManager.default.copyItem(at: Showcase.runtime.suggestedModels, to: Runtime(root: fake.root).suggestedModels)
        try "print('\(Self.realisticFamilies)')".write(to: fake.root.appending(path: "install/families.py"), atomically: true, encoding: .utf8)
        // An installer that downloads until the test ends.
        try "import time\nwhile True:\n    time.sleep(0.05)\n".write(to: fake.root.appending(path: "install/models.py"), atomically: true, encoding: .utf8)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        try await waitFor(!state.findingCachedModels, "the look through the cache")
        let suggestion = try #require(state.catalog.suggestions(memory: Catalog.memory).first)
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        let start = ContinuousClock.now
        state.download(suggestion.id)
        var drawn: [NSBitmapImageRep] = []
        while ContinuousClock.now < start + .milliseconds(700) {
            pump(0.03)
            // Once the page has moved, `settle`.
            if ContinuousClock.now > start + .milliseconds(350) { drawn.append(try #require(await Snapshot.picture(of: window))) }
        }
        pump(1)
        let after = try #require(await Snapshot.picture(of: window))
        /// Whether `picture` has dark text where the settled page has none, in the download's row
        /// under its bar, where the suggestion's name was.
        func stale(_ picture: NSBitmapImageRep) -> Bool {
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height, bytes = picture.bitsPerPixel / 8
            func light(_ p: NSBitmapImageRep, _ x: Int, _ y: Int) -> Int {
                let pixel = p.bitmapData! + y * p.bytesPerRow + x * bytes
                return (Int(pixel[0]) + Int(pixel[1]) + Int(pixel[2])) / 3
            }
            return (Int(212 * scale)..<Int(260 * scale)).contains { y in
                (Int(110 * scale)..<Int(400 * scale)).contains { x in light(picture, x, y) < 100 && light(after, x, y) > 220 }
            }
        }
        #expect(!drawn.isEmpty)
        let shown = drawn.filter(stale).count
        #expect(shown == 0, "the suggestion downloaded showed in \(shown) of \(drawn.count) frames after the page moved")
    }
}

/// A grouped page of the launch options between sections, the one under them with a red row.
private struct LaunchOptionsSample: View {
    var body: some View {
        Group {
            GroupedSection { ForEach(0..<8, id: \.self) { Text("Row \($0)") } }
            LaunchOptions()
            GroupedSection { Color.red.frame(height: 40) } header: { Text("Below") }
            GroupedSection { ForEach(0..<4, id: \.self) { Text("Last \($0)") } }
        }
    }
}

extension ServerScreenTests {
    /// The height, in points from the window's top, of the first row of `picture` that shows red
    /// in the page's part.
    static func redTop(_ picture: NSBitmapImageRep, of window: NSWindow) -> CGFloat? {
        let scale = CGFloat(picture.pixelsHigh) / window.frame.height, bytes = picture.bitsPerPixel / 8
        let top = Int(((window.frame.height - window.contentLayoutRect.maxY) * scale).rounded(.up))
        let rows = page(picture, of: window)
        return rows.firstIndex { row in
            stride(from: 0, to: row.count - 2, by: bytes).contains { i in
                let c = (0..<3).map { Int(row[row.startIndex + i + $0]) }
                return c[0] - c[1] > 60 && c[0] - c[2] > 60
            }
        }.map { CGFloat($0 + top) / scale }
    }

    /// Opening Advanced with the page at its end moves nothing under the launch options up: the
    /// options it opens push the section under them down. The frame after the click drew the
    /// card without Advanced's row, the section under it moved up by the row's height, as the
    /// switches among the options came in.
    @Test func openingAdvancedAtThePagesEndMovesNothingUp() async throws {
        let (state, folder) = AppState.forTests()
        defer { try? FileManager.default.removeItem(at: folder) }
        let window = await Snapshot.window(GroupedPage(column: 640) { LaunchOptionsSample() }.environment(state),
                                           size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.3)
        let scroll = try #require(window.contentView?.pageScroll)
        let end = try #require(scroll.documentView).frame.height - scroll.contentView.bounds.height + scroll.contentInsets.bottom
        scroll.scroll(by: end - scroll.contentView.bounds.minY)
        pump(0.3)
        // The Advanced row, by the plain view in its background (`ViewAnchor`, which opening it
        // scrolls to the top), so that the click doesn't depend on how a Mac draws its chevron.
        func plainViews(in view: NSView) -> [NSView] {
            view.subviews.flatMap { (type(of: $0) == NSView.self ? [$0] : []) + plainViews(in: $0) }
        }
        let row = try #require(plainViews(in: scroll.documentView!).last { $0.frame.width >= 300 })
        let frame = row.convert(row.bounds, to: nil)
        let point = NSPoint(x: frame.minX + 40, y: frame.midY)
        let closed = scroll.documentView!.frame.height
        let change = try await Self.pictures(of: window) {
            for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                window.sendEvent(NSEvent.mouseEvent(with: type, location: point, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                                                    windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!)
            }
        }
        let before = try #require(Self.redTop(change.before, of: window))
        let tops = change.drawn.map { Self.redTop($0.picture, of: window) }
        #expect(scroll.documentView!.frame.height > closed + 300, "Advanced opened")
        #expect(tops.allSatisfy { ($0 ?? .infinity) >= before - 1 }, "the section under the disclosure, at \(before), was drawn at \(tops)")
    }
}

/// A grouped page whose first section grows as a test says, and under it a row with a button that
/// is enabled as it grows, as the suggestions' Download buttons are as a first download ends.
@MainActor @Observable final class ButtonSample {
    var grown = false
}

private struct ButtonSampleSections: View {
    let sample: ButtonSample

    var body: some View {
        Group {
            GroupedSection { Color.green.frame(height: sample.grown ? 320 : 20) }
            GroupedSection {
                LabeledContent {
                    Button("Download") {}.disabled(!sample.grown)
                } label: {
                    Text("Model")
                }
            }
        }
        .animation(Motion.settle, value: sample.grown)
    }
}

extension ServerScreenTests {
    /// A labeled control moves with its row as one: as the row moves down, the button's title
    /// stays on its bezel and in line with the row's label. The suggestions' Download buttons drew
    /// their titles some 20 points under their bezels as the first download's end moved them.
    @Test func aLabeledButtonMovesWithItsRowAsOne() async throws {
        let sample = ButtonSample()
        let (state, folder) = AppState.forTests()
        defer { try? FileManager.default.removeItem(at: folder) }
        let window = await Snapshot.window(GroupedPage(column: 640) { ButtonSampleSections(sample: sample) }.environment(state),
                                           size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.3)
        /// The middle, in points from the window's top, of the dark pixels from `left` to `right`
        /// points in, in the page's part.
        func middle(_ picture: NSBitmapImageRep, _ left: CGFloat, _ right: CGFloat) -> CGFloat? {
            let rows = Self.page(picture, of: window, left: left, right: right), bytes = picture.bitsPerPixel / 8
            let dark = rows.indices.filter { y in
                stride(from: 0, to: rows[y].count - 2, by: bytes).contains { i in
                    (Int(rows[y][rows[y].startIndex + i]) + Int(rows[y][rows[y].startIndex + i + 1]) + Int(rows[y][rows[y].startIndex + i + 2])) / 3 < 110
                }
            }
            guard let first = dark.first, let last = dark.last else { return nil }
            return CGFloat(first + last) / 2 / (CGFloat(picture.pixelsHigh) / window.frame.height)
        }
        let change = try await Self.pictures(of: window) { sample.grown = true }
        // The row's label, at the column's start, and the button's title, at its end.
        func apart(_ picture: NSBitmapImageRep) -> CGFloat? {
            guard let label = middle(picture, 110, 300), let title = middle(picture, 560, 730) else { return nil }
            return title - label
        }
        let settled = try #require(apart(change.after))
        // A frame that still draws the button's title dimmed, as it was disabled, has no dark title.
        let drawn = change.drawn.compactMap { apart($0.picture) }
        #expect(drawn.count >= change.drawn.count / 2 && drawn.allSatisfy { abs($0 - settled) <= 2 },
                "the button's title was \(drawn) points under the row's label, \(settled) once still")
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
