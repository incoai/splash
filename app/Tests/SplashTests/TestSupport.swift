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
        try #require(ContinuousClock.now < deadline, "timed out waiting for \(what())", sourceLocation: sourceLocation)
        try await Task.sleep(for: .milliseconds(20))
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

extension UserDefaults {
    /// Defaults of a test's own, kept in `folder`, apart from the user's and every other test's: a
    /// domain named by a path keeps its file there, which goes with the folder. A write after the
    /// folder is gone makes it again, so a test stops what writes them before it removes the folder.
    static func kept(in folder: URL) -> UserDefaults {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return UserDefaults(suiteName: folder.appending(path: "defaults").path(percentEncoded: false))!
    }
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
