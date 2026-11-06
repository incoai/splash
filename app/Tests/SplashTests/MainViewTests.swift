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
    /// toolbar's buttons beside it, stay while the server starts and answers. An engine stopped
    /// under a server that runs on marks it failed, and the app's status says it no longer
    /// responds.
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
        server.signal("warm")
        try await waitFor(state.server.phase == .ready)
        #expect(state.serverTitle == name)
        server.signal("answering")
        try await waitFor(state.server.activity.running > 0 && state.server.activity.generation != nil)
        #expect(state.serverTitle == name)
        server.signal("broken")
        try await waitFor(state.server.phase == .failed("x"))
        #expect(state.server.running)
        #expect(state.serverStatus.look == .failed)
        #expect(state.status == "\(state.displayName(of: model)) · Not responding")
        #expect(state.serverTitle == "Server")
        state.server.stop()
        #expect(state.serverTitle == name)
        try await waitFor(state.server.phase == .stopped)
        #expect(state.serverTitle == "Server")
    }
}

/// A runtime whose launcher starts as `splash serve` does, one step at a time as the test signals
/// it, then answers as a ready server; its status reports a request running, then its engine
/// stopped, once signalled so.
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
