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
        .commands { SplashCommands(state: state, updates: updates) }

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

/// The menus' commands. Those of the window bring it forward, and open it where it is closed, as
/// the app runs on without it: the sidebar and the drawer then show.
struct SplashCommands: Commands {
    let state: AppState
    let updates: Updates
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        let shows = state.windowShows
        CommandGroup(after: .appInfo) {
            Button("Check for Updates…") { updates.check() }
                .disabled(!updates.canCheck)
        }
        CommandGroup(after: .pasteboard) {
            // In what the drawer shows: a terminal, a log.
            Button("Find…") { state.drawer.find() }
                .keyboardShortcut("f")
                .disabled(!state.available || !shows || !state.showsDrawer)
        }
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                inWindow {
                    state.section = .chat
                    state.chats.newChat()
                    state.draft.focus()
                }
            }
            .keyboardShortcut("n")
            .disabled(!state.available)
        }
        // The toolbar holds the switch to the Chat and the Server's button; it is not to be
        // hidden. The sidebar is.
        CommandGroup(replacing: .toolbar) {
            Button(shows && state.showsSidebar ? "Hide Sidebar" : "Show Sidebar") {
                if shows { state.toggleSidebar() } else { inWindow { state.showsSidebar = true } }
            }
            .keyboardShortcut("b")
            .disabled(!state.available || state.section == .server)
            Button(shows && state.showsDrawer ? "Hide Terminal and Logs" : "Show Terminal and Logs") {
                if shows { state.toggleDrawer() } else { inWindow { state.showDrawer(state.drawer.part) } }
            }
            .keyboardShortcut("j")
            .disabled(!state.available)
            Button("New Terminal") {
                inWindow {
                    state.newTerminal()
                    state.showDrawer(.terminal)
                }
            }
            .keyboardShortcut("t", modifiers: [.command, .shift])
            .disabled(!state.available)
            Divider()
            Button("Server") { inWindow { state.section = .server } }
                .keyboardShortcut("1")
                .disabled(!state.available)
            Button("Chat") { inWindow { state.section = .chat } }
                .keyboardShortcut("2")
                .disabled(!state.available)
        }
        CommandGroup(replacing: .help) {
            Button("Show Server Log") { Self.open(state.server.log) }
            Button("Show Download Log") { Self.open(state.installer.log) }
        }
    }

    /// Does `change`, then brings the window forward, opening it where it is closed.
    private func inWindow(_ change: () -> Void) {
        change()
        openWindow(id: "main")
    }

    /// Opens log `url` in the app that shows logs; one not written yet is made, empty, as the
    /// drawer's Logs make it, for that app to follow.
    private static func open(_ url: URL) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        NSWorkspace.shared.open(url)
    }
}

/// Owns the app's state and its updates, starts them on launch, and stops the model and the
/// downloads before the app quits, for the Quit command, the system, `kill` and an update alike;
/// a quit of the user's asks first where it would cut off work (`AppState.QuitCost`). The
/// drawer's shells end as the app does, which hangs them up (`TerminalSession`).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let state: AppState
    let updates = Updates()
    /// The question a quit of the user's asks.
    private let question = QuitQuestion()

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
    /// Whether the quit under way goes ahead without asking: SIGTERM asked for it, or the user
    /// answered the question with Quit.
    private var quitsAtOnce = false

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
            self?.quitsAtOnce = true
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

    /// The chats' files written before the app ends, a reply stopped as it quit among them, and
    /// an API key typed into the Keychain.
    func applicationWillTerminate(_ notification: Notification) {
        state.chats.flush()
        state.keepKey()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let state = state
        if asksFirst, let message = state.quitCost.message {
            question.ask(message) { [weak self] in
                self?.quitsAtOnce = true
                NSApp.terminate(nil)
            }
            return .terminateCancel
        }
        // A quit that goes ahead takes away a question still asking.
        question.close()
        guard state.busy else { return .terminateNow }
        state.quit()
        Task {
            while state.busy {
                try? await Task.sleep(for: .milliseconds(200))
            }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    /// Whether this quit asks first: one of the user's (`isUserQuit`) that is not to go ahead at
    /// once (`quitsAtOnce`).
    private var asksFirst: Bool {
        !quitsAtOnce && Self.isUserQuit(NSAppleEventManager.shared().currentAppleEvent)
    }

    /// Whether a quit that came as `event` is the user's: one of the app's own menus and ⌘Q, which
    /// come as no quit event, or the Dock's; not the quit event of a script, of the system as it
    /// logs out, of an update's installer, or of a sender that cannot be told.
    static func isUserQuit(_ event: NSAppleEventDescriptor?) -> Bool {
        guard let event else { return true }
        return isDock(event.attributeDescriptor(forKeyword: AEKeyword(keySenderPIDAttr))?.int32Value)
    }

    /// Whether process `pid` is the Dock.
    static func isDock(_ pid: pid_t?) -> Bool {
        pid.flatMap(NSRunningApplication.init(processIdentifier:))?.bundleIdentifier == "com.apple.dock"
    }
}

/// Asks whether to quit, and quits on Quit: as a sheet of the main window where it shows, else as
/// an alert of its own. Neither holds the app modal, as `runModal` would, which refuses the quit
/// event of a script, and of the system as it logs out. A second quit while it asks brings it
/// forward.
@MainActor
private final class QuitQuestion: NSObject {
    private var alert: NSAlert?
    private var quit: (() -> Void)?

    func ask(_ message: String, quit: @escaping () -> Void) {
        // The system decides whether the app comes forward: in the background, as an app the Dock
        // quits can be, it may stay behind, and its icon bounces until it does.
        NSApp.activate()
        NSApp.requestUserAttention(.criticalRequest)
        if let alert {
            (alert.window.sheetParent ?? alert.window).makeKeyAndOrderFront(nil)
            return
        }
        let alert = NSAlert()
        alert.messageText = "Quit Splash?"
        alert.informativeText = message
        // Return chooses the first, Escape Cancel, as AppKit gives an alert's buttons.
        for (title, action) in [("Quit", #selector(quitChosen)), ("Cancel", #selector(cancelChosen))] {
            let button = alert.addButton(withTitle: title)
            button.target = self
            button.action = action
        }
        alert.layout()
        self.alert = alert
        self.quit = quit
        if let window = NSApp.mainWindow, window.isVisible, window.attachedSheet == nil {
            window.beginSheet(alert.window)
        } else {
            alert.window.center()
            alert.window.makeKeyAndOrderFront(nil)
        }
    }

    @objc private func quitChosen() {
        let quit = quit
        close()
        quit?()
    }

    @objc private func cancelChosen() {
        close()
    }

    func close() {
        guard let window = alert?.window else { return }
        if let parent = window.sheetParent { parent.endSheet(window) } else { window.orderOut(nil) }
        alert = nil
        quit = nil
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
