import AppKit
import Foundation
import Testing

@testable import Splash

/// What a quit would cut off, which a quit of the user's asks about first, and whose quit that is.
@MainActor
struct QuitTests {
    /// A quit asks only where it would cut off requests being answered, a download or a terminal's
    /// program, and says each; a model that only runs is no loss.
    @Test func aQuitCostsWhatItCutsOff() {
        let download = "Flash is still downloading. If you quit, the download starts over next time."
        #expect(AppState.QuitCost(model: "27B").message == nil)
        #expect(AppState.QuitCost(model: "27B", requests: 1).message == "27B is answering a request.")
        #expect(AppState.QuitCost(model: "27B", requests: 2).message == "27B is answering 2 requests.")
        #expect(AppState.QuitCost(download: "Flash").message == download)
        #expect(AppState.QuitCost(programs: 1).message == "A terminal is running a program.")
        #expect(AppState.QuitCost(programs: 3).message == "3 terminals are running programs.")
        #expect(AppState.QuitCost(model: "27B", requests: 2, download: "Flash", programs: 1).message
            == "27B is answering 2 requests.\n\n\(download)\n\nA terminal is running a program.")
    }

    /// A quit of the app's own menus and ⌘Q, which comes as no quit event, is the user's, and so is
    /// the Dock's; a quit event whose sender cannot be told, this app's or another process's is not.
    @Test func onlyTheUsersQuitAsks() {
        #expect(AppDelegate.isUserQuit(nil))
        // One made here names no sender: the system names it on the way.
        let event = NSAppleEventDescriptor.appleEvent(
            withEventClass: AEEventClass(kCoreEventClass), eventID: AEEventID(kAEQuitApplication), targetDescriptor: nil,
            returnID: AEReturnID(kAutoGenerateReturnID), transactionID: AETransactionID(kAnyTransactionID))
        #expect(!AppDelegate.isUserQuit(event))
        #expect(!AppDelegate.isDock(getpid()))
        // The Dock runs where a user is logged in.
        if let dock = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.dock").first {
            #expect(AppDelegate.isDock(dock.processIdentifier))
        }
    }

    /// The requests the model answers, as its status says, are what a quit would cut off; the model
    /// idle is not.
    @Test func aQuitCutsOffTheRequestsTheStatusSays() async throws {
        let fake = try FakeRuntime("splash-quit", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let model = "a/First-GGUF:Q4"
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start(model)
        try await waitFor(state.server.phase == .ready, "ready")
        #expect(state.quitCost.message == nil)
        FileManager.default.createFile(atPath: fake.root.appending(path: "answering").path, contents: nil)
        try await waitFor(state.quitCost.requests == 1, "the request the status says runs")
        #expect(state.quitCost.message == "\(state.displayName(of: model)) is answering a request.")
    }
}
