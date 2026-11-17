import AppKit
import Darwin
import Foundation
import Testing

@testable import Splash

/// Waits until `condition` holds, polling it, and fails the test once `seconds` have passed.
@MainActor
func waitFor(_ condition: @autoclosure () async -> Bool, _ what: @autoclosure () -> String = "the condition",
             seconds: Double = 10, sourceLocation: SourceLocation = #_sourceLocation) async throws {
    let deadline = ContinuousClock.now + .seconds(seconds)
    while !(await condition()) {
        // Let queued main-actor callbacks run before the final deadline check. Snapshot work
        // can hold the actor past the deadline with a completion waiting behind the last poll.
        try await Task.sleep(for: .milliseconds(20))
        if ContinuousClock.now >= deadline {
            try #require(await condition(), "timed out waiting for \(what())", sourceLocation: sourceLocation)
            return
        }
    }
}

/// What `read` gives of the app's state, from now on: its value now, then at each change of it, as
/// the turn of the main actor that changed it left it, which is what a window could draw. A test
/// then says every state a flow went through, not only those it stopped to look at.
@MainActor
final class Seen<Value: Equatable> {
    private var seen: [Value] = []
    private let read: () -> Value

    init(_ read: @escaping () -> Value) {
        self.read = read
        look()
    }

    /// The values so far, each unlike the one before it, the one the state has now last: a change
    /// just made is there, and stays, though the turn that looks at it has yet to run and may
    /// find the next one.
    var values: [Value] {
        note(read())
        return seen
    }

    /// What `part` of the values was, each unlike the one before it: the order a part went
    /// through its states in, whatever else changed between them.
    func each<Part: Equatable>(_ part: (Value) -> Part) -> [Part] {
        values.map(part).reduce(into: []) { parts, next in
            if parts.last != next { parts.append(next) }
        }
    }

    private func look() {
        // Told before the change is made: the next turn reads what it left.
        note(withObservationTracking(read) { [weak self] in
            Task { @MainActor in self?.look() }
        })
    }

    private func note(_ value: Value) {
        if seen.last != value { seen.append(value) }
    }
}

/// A loopback port nothing listens on now, as the system picks one.
func freePort() -> Int {
    let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
    defer { close(socket) }
    var address = sockaddr_in(sin_len: UInt8(MemoryLayout<sockaddr_in>.size), sin_family: sa_family_t(AF_INET),
                              sin_port: 0, sin_addr: in_addr(s_addr: inet_addr("127.0.0.1")), sin_zero: (0, 0, 0, 0, 0, 0, 0, 0))
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            _ = bind(socket, $0, length)
            _ = getsockname(socket, $0, &length)
        }
    }
    return Int(UInt16(bigEndian: address.sin_port))
}

extension NSView {
    /// The first text view in this view that is no field editor: the composer's, in the Chat.
    var firstTextView: NSTextView? {
        func textViews(in view: NSView) -> [NSTextView] {
            view.subviews.flatMap { ($0 as? NSTextView).map { [$0] } ?? textViews(in: $0) }
        }
        return textViews(in: self).first { !$0.isFieldEditor }
    }

    /// The scroll view of the page this view shows (`AppKitScroll`): of those in it, the one whose
    /// content is the tallest.
    var pageScroll: NSScrollView? {
        func scrollViews(in view: NSView) -> [NSScrollView] {
            view.subviews.flatMap { ($0 as? NSScrollView).map { [$0] } ?? scrollViews(in: $0) }
        }
        return scrollViews(in: self).max { ($0.documentView?.frame.height ?? 0) < ($1.documentView?.frame.height ?? 0) }
    }
}

/// Runs the main run loop for `seconds`, as an app's event loop runs it: its observers commit
/// SwiftUI's updates and its display link moves animations on, which a test's main actor alone
/// never does.
@MainActor func pump(_ seconds: Double) {
    _ = CFRunLoopRunInMode(.defaultMode, seconds, false)
}

extension NSScrollView {
    /// Scrolls its content up by `distance` from where it is.
    func scroll(by distance: CGFloat) {
        contentView.scroll(to: NSPoint(x: contentView.bounds.minX, y: contentView.bounds.minY + distance))
        reflectScrolledClipView(contentView)
    }
}

extension UserDefaults {
    /// Defaults of a test's own, kept in `folder` for the run, apart from the user's and every other
    /// test's and run's: a run's start empty, and the calls of a run with one folder share them, as
    /// the launches of one app share its defaults. A domain named by a path keeps its file there,
    /// which goes with the folder. A write after the folder is gone makes it again, so a test stops
    /// what writes them before it removes the folder.
    static func kept(in folder: URL) -> UserDefaults {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return UserDefaults(suiteName: folder.appending(path: "defaults-\(run)").path(percentEncoded: false))!
    }

    /// The run's part of the name of each domain it opens, so that no run opens a domain another
    /// opened: on macOS 27 a process that opens a domain whose file holds a key goes on reading the
    /// value it found after it writes the key, though the write reaches the file, and what a run
    /// left can outlive the folder.
    private static let run = UUID().uuidString
}

extension Shell {
    /// zsh without the user's startup files, which can take seconds, or wait for input, before
    /// the shell reads what a test types.
    static let bare = Shell(path: "/bin/zsh", arguments: ["-f"], name: "zsh")
}

extension AppState.Folders {
    /// The chats in `folder`, the logs in its Logs: a test's own, apart from the user's.
    init(in folder: URL) {
        self.init(logs: folder.appending(path: "Logs"), chats: folder)
    }
}

extension AppState {
    /// An app whose chats are in `folder`, its logs in its Logs and its defaults `defaults`, else
    /// there too, on `runtime`, else on an empty one in the folder; its API key is not the user's
    /// either, and its terminals run `Shell.bare`. Apps made with one folder's defaults are
    /// launches of one app.
    static func forTests(in folder: URL, runtime: Runtime? = nil, defaults: UserDefaults? = nil) -> AppState {
        AppState(runtime: runtime ?? Runtime(root: folder.appending(path: "Runtime")), folders: Folders(in: folder),
                 keychain: Keychain(service: "Splash Tests"), defaults: defaults ?? .kept(in: folder), shell: { .bare })
    }

    /// The same in a new temporary folder, with that folder, for the test to remove.
    static func forTests() -> (AppState, URL) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-tests-\(UUID().uuidString)")
        return (forTests(in: folder), folder)
    }
}
