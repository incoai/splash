import AppKit
import SwiftUI

@main
struct SplashApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        let state = delegate.state
        let updates = delegate.updates
        Window("Splash", id: "main") {
            MainView()
                .environment(state)
        }
        .defaultSize(width: 1120, height: 780)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") { updates.check() }
                    .disabled(!updates.canCheck)
            }
            CommandGroup(after: .pasteboard) {
                // In what has the keyboard: a log of the drawer's, a terminal.
                Button("Find…") { NSApp.keyWindow?.firstResponder?.showFindBar() }
                    .keyboardShortcut("f")
            }
            CommandGroup(replacing: .newItem) {
                Button("New Chat") {
                    state.section = .chat
                    state.chats.newChat()
                    state.draft.focus()
                }
                .keyboardShortcut("n")
                .disabled(!state.available)
            }
            // The toolbar holds the switch to the Chat and the Server's button; it is not to be
            // hidden. The sidebar is.
            CommandGroup(replacing: .toolbar) {
                Button(state.showsSidebar ? "Hide Sidebar" : "Show Sidebar") { state.toggleSidebar() }
                    .keyboardShortcut("b")
                    .disabled(!state.available || state.section == .server)
                Button(state.showsDrawer ? "Hide Terminal and Logs" : "Show Terminal and Logs") { state.toggleDrawer() }
                    .keyboardShortcut("j")
                    .disabled(!state.available)
                Button("New Terminal") {
                    state.newTerminal()
                    state.showDrawer(.terminal)
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
                .disabled(!state.available)
                Divider()
                Button("Server") { state.section = .server }
                    .keyboardShortcut("1")
                    .disabled(!state.available)
                Button("Chat") { state.section = .chat }
                    .keyboardShortcut("2")
                    .disabled(!state.available)
            }
            CommandMenu("Server") {
                ServerMenuItems()
                    .environment(state)
            }
            CommandGroup(replacing: .help) {
                Button("Show Server Log") { NSWorkspace.shared.open(state.server.log) }
                Button("Show Download Log") { NSWorkspace.shared.open(state.installer.log) }
            }
        }

        MenuBarExtra {
            MenuBarPanel()
                .environment(state)
        } label: {
            Image(nsImage: SplashMark.menuBarImage(serving: state.server.answers))
                .accessibilityLabel("Splash")
        }
        .menuBarExtraStyle(.window)
    }
}

/// Owns the app's state and its updates, starts them on launch, and stops the model and the
/// downloads before the app quits, for the Quit command, the system, `kill` and an update alike.
/// The drawer's shells end as the app does, which hangs them up (`TerminalSession`).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let state: AppState
    let updates = Updates()

    override convenience init() {
        guard let runtime = Runtime.locate() else {
            let alert = NSAlert()
            alert.messageText = "Splash is damaged"
            alert.informativeText = "Its runtime is missing. Download Splash again."
            alert.runModal()
            exit(1)
        }
        self.init(state: AppState(runtime: runtime))
    }

    init(state: AppState) {
        self.state = state
        super.init()
    }

    /// Quits as the Quit command does when the system or `kill` asks the app to end, without
    /// asking first.
    private var termination: DispatchSourceSignal?
    private var signalled = false

    /// SIGTERM caught by a handler that does nothing, for `termination` alone to act on. Not
    /// ignored: a program the app starts keeps a signal ignored, but not one caught, so that what
    /// the drawer's terminals run ends on SIGTERM as anywhere.
    static func catchTerminate() {
        signal(SIGTERM) { _ in }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // One Splash at a time: another would take over the first one's model. Of two opened at
        // once, the one launched first stays, and the other shows its window; but a copy that runs
        // nothing, from the disk image, gives way to this one.
        if let id = Bundle.main.bundleIdentifier, let first = NSRunningApplication.runningApplications(withBundleIdentifier: id)
            .filter({ !$0.isTerminated })
            .min(by: { ($0.launchDate ?? .distantFuture, $0.processIdentifier) < ($1.launchDate ?? .distantFuture, $1.processIdentifier) }),
            first != .current, let bundle = first.bundleURL {
            if AppState.mustMoveToApplications(bundle) {
                first.terminate()
            } else {
                NSWorkspace.shared.openApplication(at: bundle, configuration: NSWorkspace.OpenConfiguration())
                NSApp.terminate(nil)
                return
            }
        }
        Self.catchTerminate()
        termination = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        termination?.setEventHandler { [weak self] in
            self?.signalled = true
            // From the run loop, not this main-queue block: terminating waits for the model to stop,
            // which is reported on the main queue.
            NSApp.perform(#selector(NSApplication.terminate(_:)), with: nil, afterDelay: 0)
        }
        termination?.resume()
        updates.start()
        Task { await state.launch() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    /// The chats' files written before the app ends, a reply stopped as it quit among them.
    func applicationWillTerminate(_ notification: Notification) {
        state.chats.flush()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let state = state
        guard state.busy else { return .terminateNow }
        if let model = state.installer.model, !signalled {
            let alert = NSAlert()
            alert.messageText = "Quit Splash?"
            alert.informativeText = "\(state.displayName(of: model)) is still downloading. If you quit, the download starts over next time."
            alert.addButton(withTitle: "Quit")
            alert.addButton(withTitle: "Cancel")
            guard alert.runModal() == .alertFirstButtonReturn else { return .terminateCancel }
        }
        state.quit()
        Task {
            while state.busy {
                try? await Task.sleep(for: .milliseconds(200))
            }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

extension NSResponder {
    /// Shows the find bar of this responder, or of the first after it that has one, as a text
    /// app's Find… does.
    func showFindBar() {
        let find = NSMenuItem()
        find.tag = Int(NSFindPanelAction.showFindPanel.rawValue)
        _ = tryToPerform(#selector(NSTextView.performFindPanelAction(_:)), with: find)
    }
}
