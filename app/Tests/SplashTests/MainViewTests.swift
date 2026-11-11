import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// The window's toolbar and the Chat's sidebar.
@MainActor
@Suite(.serialized)
struct MainViewTests {
    /// The sidebar shows or hides at launch as it did when the app last quit, as its own defaults
    /// keep it: another app's are not changed.
    @Test func theSidebarIsKeptFromOneLaunchToTheNext() {
        let (state, folder) = AppState.forTests()
        let (other, otherFolder) = AppState.forTests()
        defer { [folder, otherFolder].forEach { try? FileManager.default.removeItem(at: $0) } }
        state.showsSidebar = false
        #expect(!AppState.forTests(in: folder).showsSidebar)
        #expect(other.showsSidebar && AppState.forTests(in: otherFolder).showsSidebar)
        state.showsSidebar = true
        #expect(AppState.forTests(in: folder).showsSidebar)
    }

    /// The sidebar hidden takes the keyboard from its search field, where what is typed would
    /// filter a list no one sees.
    @Test func hidingTheSidebarTakesTheKeyboardFromIt() async throws {
        let (state, folder) = AppState.forTests()
        defer { try? FileManager.default.removeItem(at: folder) }
        state.showsSidebar = true
        _ = NSApplication.shared
        let window = NSWindow(contentViewController: NSHostingController(rootView: WithSidebar {
            TextField("Search", text: .constant(""))
        } content: {
            Color.clear
        }.environment(state)))
        window.setContentSize(CGSize(width: 800, height: 600))
        window.setFrameOrigin(NSPoint(x: -30000, y: -30000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        func field(in view: NSView) -> NSTextField? {
            (view as? NSTextField).flatMap { $0.isEditable ? $0 : nil } ?? view.subviews.lazy.compactMap(field).first
        }
        try await waitFor(window.contentView.flatMap(field) != nil)
        window.makeFirstResponder(try #require(window.contentView.flatMap(field)))
        try await waitFor((window.firstResponder as? NSTextView)?.isFieldEditor == true)
        // SwiftUI sees where the keyboard went, as it does before the user's next key.
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
        state.toggleSidebar()
        try await waitFor(window.firstResponder === window)
    }

    /// The Server button names the model while it starts, runs and answers, and says Server
    /// while none runs: its title holds no step or speed, so its width, and the place of the
    /// toolbar's buttons beside it, stay while the server starts and answers. The status says
    /// how far a warm-up is, as the record counts its steps, and why it can take long. A server
    /// that stops restarting its engine exits, failed in the words of its record, and the app's
    /// status says it ran into a problem.
    @Test func theServerButtonSaysWhatTheServerDoes() async throws {
        let server = try FakeServer()
        defer { server.remove() }
        let state = server.state
        let model = "example/Model"
        let name = state.catalog.name(of: model)
        #expect(state.serverTitle == "Server")
        state.server.start(model, settings: ServeSettings(values: ["--port": String(server.port)]))
        #expect(state.server.step == .starting)
        #expect(state.serverTitle == name)
        try await waitFor(state.server.step == .loading)
        #expect(state.serverTitle == name)
        #expect(state.status == "\(state.displayName(of: model)) · Loading weights…")
        server.signal("loaded")
        try await waitFor(state.server.step == .warming)
        #expect(state.serverTitle == name)
        #expect(state.server.warmup == ServeRecord.Warmup(steps_done: 3, steps: 10))
        #expect(state.status == "\(state.displayName(of: model)) · Warming up · 3 of 10")
        #expect(state.serverStatus.note == Server.Step.warming.note)
        server.signal("warm")
        try await waitFor(state.server.phase == .ready)
        #expect(state.serverTitle == name)
        server.signal("answering")
        try await waitFor(state.server.activity.running > 0 && state.server.activity.generation != nil)
        #expect(state.serverTitle == name)
        server.signal("broken")
        try await waitFor(state.server.phase == .failed("x"))
        #expect(!state.server.running)
        #expect(state.serverStatus.look == .failed)
        #expect(state.status == "\(state.displayName(of: model)) · Ran into a problem")
        #expect(state.serverTitle == "Server")
        state.server.dismissFailure()
        #expect(state.server.phase == .stopped)
        #expect(state.serverTitle == "Server")
    }
}

extension MainViewTests {
    /// A request is one row from its start to its end: its Started line adds it, the status
    /// moves it on, its Done line ends it in its place, and a status that still has it, as one
    /// answered from an earlier snapshot, leaves it ended.
    @Test func aRequestIsOneRowFromItsStartToItsEnd() async throws {
        let server = try FakeServer()
        defer { server.remove() }
        let state = server.state
        state.server.start("example/Model", settings: ServeSettings(values: ["--port": String(server.port)]))
        server.signal("loaded")
        server.signal("warm")
        try await waitFor(state.server.phase == .ready)
        server.signal("answering")
        try await waitFor(state.server.requests.first?.stage == .answering(1))
        #expect(state.server.requests.map(\.id) == [1] && state.server.requests[0].input == 13 && state.server.requests[0].end == nil)
        server.signal("finished")
        try await waitFor(state.server.requests.first?.end != nil)
        // Past the next status, which still reports it running.
        try await Task.sleep(for: .seconds(2.5))
        #expect(state.server.requests.map(\.id) == [1] && state.server.requests[0].end?.output == 1)
    }
}

/// A runtime whose launcher starts as `splash serve` does, one step at a time as the test signals
/// it, then answers as a ready server; its status reports a request running, and once signalled
/// so it stops restarting its engine as a server does: its record says why, and it exits.
@MainActor
private final class FakeServer {
    let root: URL
    let state: AppState
    let port: Int

    init() throws {
        let runtime = try FakeRuntime("splash-fake-server", launcher: "stepped_server")
        root = runtime.root
        port = runtime.port
        state = runtime.state(keychain: Keychain(service: "Splash Tests"))
    }

    /// Lets the launcher go on past the step that waits for `name`.
    func signal(_ name: String) {
        FileManager.default.createFile(atPath: root.appending(path: name).path, contents: nil)
    }

    /// Stops what still runs and removes the runtime.
    func remove() {
        state.server.stop()
        try? FileManager.default.removeItem(at: root)
    }
}

extension MainViewTests {
    /// The menu bar panel's Model rows come and go at once, as a menu's do: each frame drawn as they
    /// open, close and open again shows the panel as it was or as it is to be, its window's height
    /// with it; never rows fading where the rows under them slide, the check under a chevron.
    @Test func thePanelsModelRowsComeAndGoAtOnce() async throws {
        let fake = try FakeRuntime("splash-panel-rows", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await ServerScreenTests.launchedWithModels(fake)
        defer { state.quit() }
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start("unsloth/flint-GGUF:UD-Q4_K_XL")
        try await waitFor(state.server.phase == .ready, "ready")
        let window = await Snapshot.window(MenuBarPanel().environment(state), size: CGSize(width: 300, height: 400), appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        /// The window drawn now, at once, so that the frames come some 20 ms apart.
        func picture() throws -> Data {
            let frame = try #require(window.contentView?.superview)
            let bitmap = try #require(frame.bitmapImageRepForCachingDisplay(in: frame.bounds))
            frame.cacheDisplay(in: frame.bounds, to: bitmap)
            return Data(bytes: try #require(bitmap.bitmapData), count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        }
        let closed = window.frame.height
        /// Clicks the Model row, above its rows while they show, then Stop, Coding Agents and Quit,
        /// and gives the frames drawn after, and the one once the panel has stopped changing.
        func toggle() throws -> (before: Data, drawn: [Data], after: Data) {
            let before = try picture()
            let row = NSPoint(x: 150, y: 106 + window.frame.height - closed)
            for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                window.sendEvent(try #require(NSEvent.mouseEvent(
                    with: type, location: row, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                    windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)))
            }
            var drawn: [Data] = []
            for _ in 0..<15 {
                pump(0.02)
                drawn.append(try picture())
            }
            pump(1)
            return (before, drawn, try picture())
        }
        for change in ["opening", "closing", "opening again"] {
            let height = window.frame.height
            let frames = try toggle()
            #expect(window.frame.height != height && frames.after != frames.before, "\(change)")
            #expect(frames.drawn.allSatisfy { $0 == frames.before || $0 == frames.after }, "\(change)")
        }
    }
}
