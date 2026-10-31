import Foundation

/// The Splash runtime the app runs: the packaged release tree inside the app bundle, or, for
/// development, the tree `SPLASH_RUNTIME` names (a packaged tree or a source checkout). Its layout
/// and data folders are the ones `install/paths.py` resolves.
struct Runtime: Sendable {
    let root: URL

    static func locate() -> Runtime? {
        let candidates = [
            ProcessInfo.processInfo.environment["SPLASH_RUNTIME"].map { URL(filePath: $0) },
            Bundle.main.resourceURL?.appending(path: "splash"),
        ]
        return candidates.compactMap { $0 }
            .first { FileManager.default.fileExists(atPath: $0.appending(path: "install/launcher.py").path) }
            .map(Runtime.init)
    }

    /// Whether this is a packaged tree (`release.json`) rather than a source checkout.
    var packaged: Bool { FileManager.default.fileExists(atPath: root.appending(path: "release.json").path) }

    var python: URL { root.appending(path: packaged ? "python/bin/python3" : ".venv/bin/python") }
    var engine: URL { root.appending(path: packaged ? "engine/splash" : "build/splash") }
    var suggestedModels: URL { root.appending(path: "install/suggested-models.tsv") }
    /// The web chat's Markdown and code highlighting, which the Chat's transcript uses too.
    var markdownScripts: [URL] {
        ["markdown-it-15.0.2.min.js", "highlight-11.12.0.min.js"].map { root.appending(path: "server/chat-assets/\($0)") }
    }

    private var data: URL { packaged ? URL.applicationSupportDirectory.appending(path: "Splash") : root }
    /// Where installed models' selection links live.
    var models: URL { data.appending(path: packaged ? "models" : "install/models") }
    /// Where `splash serve` keeps its locks.
    var locks: URL { packaged ? data.appending(path: "runtime") : root.appending(path: "build/runtime") }

    /// The command line of `splash ARGUMENTS`. Isolated from the user's Python settings and
    /// site-packages, and never writing bytecode into the signed bundle.
    func command(_ arguments: [String]) -> [String] {
        [python.path, "-I", "-B", "-u", root.appending(path: "install/launcher.py").path] + arguments
    }

    /// The command line that downloads and installs `model`, the installer `splash serve` runs
    /// before it serves, reporting download progress as JSON lines.
    func installCommand(_ model: String) -> [String] {
        [python.path, "-I", "-B", "-u", root.appending(path: "install/models.py").path,
         "--model", model, "prepare", "--json-progress"]
    }

    /// The installer of the models the Hugging Face cache holds whole that Splash serves, without
    /// the Hub (`install/cached.py`), to run.
    func cachedInstaller() -> Process {
        let process = Process()
        process.executableURL = python
        process.arguments = ["-I", "-B", "-u", root.appending(path: "install/cached.py").path, "--models", models.path]
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        return process
    }

    /// The environment of the programs the app runs: `userEnvironment`, kept from the user's
    /// site-packages in the programs Python starts. Run with -B, the launcher keeps them from
    /// writing bytecode.
    var environment: [String: String] {
        var environment = userEnvironment
        environment["PYTHONNOUSERSITE"] = "1"
        return environment
    }

    /// The app's own environment (an app does not get the shell's), without an API key the server
    /// would require of the app: the drawer's terminals', whose commands are the user's, finding
    /// the user's site-packages as in their terminal.
    var userEnvironment: [String: String] {
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "SPLASH_API_KEY")
        environment.removeValue(forKey: "SPLASH_RUNTIME")
        return environment
    }

    /// The models installed here: each selection link's assembly record names its model.
    func installedModels() -> [InstalledModel] {
        struct Record: Decodable {
            struct File: Decodable { let path: String; let bytes: Int64 }
            let model: String
            let files: [String: File]
        }
        let manager = FileManager.default
        let owners = (try? manager.contentsOfDirectory(at: models, includingPropertiesForKeys: nil)) ?? []
        return owners.filter { !$0.lastPathComponent.hasPrefix(".") }
            .flatMap { (try? manager.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil)) ?? [] }
            .compactMap { link in
                guard let data = try? Data(contentsOf: link.appending(path: "model.json")),
                      let record = try? JSONDecoder().decode(Record.self, from: data)
                else { return nil }
                // An assembly can link one file under two names (a checkpoint's vision tower).
                let files = Dictionary(record.files.values.map { ($0.path, $0.bytes) }) { first, _ in first }
                return InstalledModel(id: record.model, bytes: files.values.reduce(0, +))
            }
            .sorted { $0.id < $1.id }
    }

    /// The process serving `port`, as `splash serve` records it in the port's lock: it holds the
    /// lock from before it checks the port until it exits, so a server answering the port is it.
    func serveLockOwner(port: Int) -> pid_t? {
        struct Owner: Decodable { let pid: Int32 }
        guard let data = try? Data(contentsOf: locks.appending(path: "serve-\(port).lock")) else { return nil }
        return (try? JSONDecoder().decode(Owner.self, from: data))?.pid
    }

    /// Why this Mac cannot run Splash, or nil when it can: the engine's own check, which the
    /// launcher repeats before every start; nil too where the check does not answer in 30 s.
    func unsupportedReason() async -> String? {
        guard let ran = await Program.run([engine.path, "device-check"], timeout: .seconds(30)), ran.status != 0
        else { return nil }
        let reason = ran.error.trimmingCharacters(in: .whitespacesAndNewlines).replacing(#/^error: /#, with: "")
        return reason.isEmpty ? nil : reason
    }
}

struct InstalledModel: Identifiable, Hashable, Sendable {
    /// Its Hugging Face ID, or the path of model files on this Mac.
    let id: String
    /// What its files take on disk, shared with any other installation of the same files; nil for
    /// model files on this Mac not measured yet.
    var bytes: Int64?
    /// Whether model files on this Mac are no longer where they were added from.
    var missing = false

    /// Whether it is model files on this Mac rather than a model Splash installed.
    var local: Bool { id.hasPrefix("/") }

    /// The model files at `path`, a GGUF file or a folder, measured.
    static func local(_ path: String) -> InstalledModel {
        let url = URL(filePath: path).resolvingSymlinksInPath()
        guard let kind = try? url.resourceValues(forKeys: [.isDirectoryKey]) else { return InstalledModel(id: path, missing: true) }
        let files = kind.isDirectory == true
            ? (FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)?.allObjects as? [URL]) ?? []
            : [url]
        let bytes = files.reduce(Int64(0)) { total, file in
            total + Int64((try? file.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return InstalledModel(id: path, bytes: bytes)
    }
}
