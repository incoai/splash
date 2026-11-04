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

extension AppState.Folders {
    /// The chats in `folder`, the logs in its Logs: a test's own, apart from the user's.
    init(in folder: URL) {
        self.init(logs: folder.appending(path: "Logs"), chats: folder)
    }
}

extension AppState {
    /// An app on `runtime` whose chats are in `folder` and its logs in its Logs, and whose API key
    /// is not the user's either.
    static func forTests(in folder: URL, runtime: Runtime = Showcase.runtime) -> AppState {
        AppState(runtime: runtime, folders: Folders(in: folder), keychain: Keychain(service: "Splash Tests"))
    }

    /// The same in a new temporary folder, with that folder, for the test to remove.
    static func forTests() -> (AppState, URL) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-tests-\(UUID().uuidString)")
        return (forTests(in: folder), folder)
    }
}
