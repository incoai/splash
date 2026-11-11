import Foundation
import Testing

@testable import Splash

/// A runtime of a test's own, in a temporary folder, whose launcher is a Python script of Fakes/
/// that the system's Python runs, and a loopback port nothing serves, for the script to serve on.
struct FakeRuntime {
    let root: URL
    let port: Int

    init(_ name: String, launcher fake: String) throws {
        let script = try #require(Bundle.module.url(forResource: fake, withExtension: "py", subdirectory: "Fakes"))
        let root = FileManager.default.temporaryDirectory.appending(path: "\(name)-\(UUID().uuidString)")
        let manager = FileManager.default
        for folder in ["install", ".venv/bin", "chats"] {
            try manager.createDirectory(at: root.appending(path: folder), withIntermediateDirectories: true)
        }
        // /usr/bin/python3 can be Apple's launcher; resolve the real interpreter before linking
        // it under the name `python`, which the launcher otherwise tries to find on PATH.
        let resolver = Process()
        resolver.executableURL = URL(filePath: "/usr/bin/python3")
        resolver.arguments = ["-c", "import sys; print(sys.executable)"]
        let output = Pipe()
        resolver.standardOutput = output
        try resolver.run()
        let executable = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        resolver.waitUntilExit()
        try #require(resolver.terminationStatus == 0 && !executable.isEmpty)
        try manager.createSymbolicLink(at: root.appending(path: ".venv/bin/python"), withDestinationURL: URL(filePath: executable))
        try manager.copyItem(at: script, to: root.appending(path: "install/launcher.py"))
        // What the launchers share, which each runs from beside itself.
        let shared = try #require(Bundle.module.url(forResource: "app_bound", withExtension: "py", subdirectory: "Fakes"))
        try manager.copyItem(at: shared, to: root.appending(path: "install/app_bound.py"))
        self.root = root
        port = freePort()
    }

    /// Lists `model` as installed, of `family` where one is given: an assembly record of no files,
    /// or, given `bytes`, of one that takes them, the launcher's.
    func install(_ model: String, family: String? = nil, bytes: Int64? = nil) throws {
        let link = Runtime(root: root).models.appending(path: model)
        try FileManager.default.createDirectory(at: link, withIntermediateDirectories: true)
        let family = family.map { #", "family": "\#($0)""# } ?? ""
        let launcher = root.appending(path: "install/launcher.py").path(percentEncoded: false)
        let files = bytes.map { #"{"target": {"path": "\#(launcher)", "bytes": \#($0)}}"# } ?? "{}"
        try #"{"model": "\#(model)"\#(family), "files": \#(files)}"#.write(to: link.appending(path: "model.json"), atomically: true, encoding: .utf8)
    }

    /// The folder of the chats of `state`.
    var chats: URL { root.appending(path: "chats") }

    /// An app's state on this runtime, its logs and defaults in the runtime's folder.
    @MainActor func state(keychain: Keychain) -> AppState {
        AppState(runtime: Runtime(root: root), folders: .init(logs: root, chats: chats), keychain: keychain,
                 defaults: .kept(in: root), shell: { .bare })
    }
}
