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
    /// The web chat's Markdown, code highlighting and link rules, which the Chat's transcript uses too.
    var markdownScripts: [URL] {
        ["markdown-it-15.0.2.min.js", "highlight-11.12.0.min.js", "markdown-links.js"]
            .map { root.appending(path: "server/chat-assets/\($0)") }
    }

    private var data: URL { packaged ? URL.applicationSupportDirectory.appending(path: "Splash") : root }
    /// Where installed models' selection links live.
    var models: URL { data.appending(path: packaged ? "models" : "install/models") }
    /// Where `splash serve` keeps its locks.
    var locks: URL { packaged ? data.appending(path: "runtime") : root.appending(path: "build/runtime") }

    /// The command line of `splash ARGUMENTS`.
    func command(_ arguments: [String]) -> [String] {
        scriptCommand("install/launcher.py", arguments)
    }

    /// The command line that downloads and installs `model`, the installer `splash serve` runs
    /// before it serves, reporting download progress as JSON lines.
    func installCommand(_ model: String) -> [String] {
        scriptCommand("install/models.py", ["--model", model, "prepare", "--json-progress"])
    }

    /// Whether `pid` runs this runtime's installer as an app starts it (`installCommand`), for any
    /// model, as the kernel keeps its arguments; `splash serve` runs it with others. Its program is
    /// left out: an interpreter can start another in its place, under the same ID and arguments,
    /// as a framework build of Python does.
    func runsInstaller(_ pid: pid_t) -> Bool {
        guard let arguments = processArguments(pid), let flag = arguments.firstIndex(of: "--model"),
              flag + 1 < arguments.endIndex
        else { return false }
        return arguments.dropFirst() == installCommand(arguments[flag + 1]).dropFirst()
    }

    /// The command line of the installer of the models the Hugging Face cache holds whole that
    /// Splash serves, without the Hub (`install/cached.py`).
    var cachedInstallCommand: [String] {
        scriptCommand("install/cached.py", ["--models", models.path])
    }

    /// The command line of the Python `script` in the runtime: isolated from the user's Python
    /// settings and site-packages, unbuffered, and never writing bytecode into the signed bundle.
    private func scriptCommand(_ script: String, _ arguments: [String]) -> [String] {
        [python.path, "-I", "-B", "-u", root.appending(path: script).path] + arguments
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

    /// The models installed here: each selection link's assembly record names its model and its
    /// family; the model is on this Mac while every file the record names is, its links followed
    /// (as `present` in `install/cached.py` has it): a model whose files were deleted, as from the
    /// Hugging Face cache, is not listed, and installs again once the cache holds it again or it is
    /// downloaded again.
    func installedModels() -> [InstalledModel] {
        struct Record: Decodable {
            struct File: Decodable { let path: String; let bytes: Int64 }
            let model: String
            let family: String?
            let files: [String: File]
        }
        let manager = FileManager.default
        let owners = (try? manager.contentsOfDirectory(at: models, includingPropertiesForKeys: nil)) ?? []
        return owners.filter { !$0.lastPathComponent.hasPrefix(".") }
            .flatMap { (try? manager.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil)) ?? [] }
            .compactMap { link in
                guard let data = try? Data(contentsOf: link.appending(path: "model.json")),
                      let record = try? JSONDecoder().decode(Record.self, from: data),
                      record.files.values.allSatisfy({ manager.fileExists(atPath: $0.path) })
                else { return nil }
                // An assembly can link one file under two names (a checkpoint's vision tower).
                let files = Dictionary(record.files.values.map { ($0.path, $0.bytes) }) { first, _ in first }
                return InstalledModel(id: record.model, bytes: files.values.reduce(0, +), family: record.family)
            }
            .sorted { $0.id < $1.id }
    }

    /// The family of each model prepared here under a selection link of its options' hash
    /// (`.selections`), by the model its assembly record names: model files on this Mac, by their
    /// path (`install/local.py`), and a model installed with other options than the app's.
    func preparedFamilies() -> [String: String] {
        struct Record: Decodable {
            let model: String
            let family: String
        }
        let links = (try? FileManager.default.contentsOfDirectory(at: models.appending(path: ".selections"),
                                                                   includingPropertiesForKeys: nil)) ?? []
        let records = links.compactMap { link in
            (try? Data(contentsOf: link.appending(path: "model.json"))).flatMap { try? JSONDecoder().decode(Record.self, from: $0) }
        }
        return Dictionary(records.map { ($0.model, $0.family) }) { first, _ in first }
    }

    /// What each model family makes of the launch options, by its name, as `install/families.py`
    /// reports it on its last line; empty where it fails, or runs past 30 s.
    @MainActor
    func families() async -> [String: ModelFamily] {
        let output = FileManager.default.temporaryDirectory.appending(path: "splash-families-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: output) }
        guard await ChildProcess.run(scriptCommand("install/families.py", []), environment: environment, log: output,
                                     timeout: .seconds(30)) == 0,
              let said = try? String(contentsOf: output, encoding: .utf8),
              let report = said.split(whereSeparator: \.isNewline).last
        else { return [:] }
        return (try? JSONDecoder().decode([String: ModelFamily].self, from: Data(report.utf8))) ?? [:]
    }

    /// The process holding `port`'s lock, as it records itself there; nil while none holds it. The
    /// probe takes the lock shared without waiting, which only a holder refuses.
    func serveLockHolder(port: Int) -> pid_t? {
        let lock = locks.appending(path: "serve-\(port).lock")
        let descriptor = open(lock.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else { return nil }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_SH | LOCK_NB) != 0 else {
            flock(descriptor, LOCK_UN)
            return nil
        }
        struct Owner: Decodable { let pid: Int32 }
        guard errno == EWOULDBLOCK, let data = try? Data(contentsOf: lock) else { return nil }
        return (try? JSONDecoder().decode(Owner.self, from: data))?.pid
    }

    /// What the launcher serving `port`, or the server it runs as, last wrote of its start
    /// (`server/serve_record.py`), beside the port's lock; nil before one has.
    func serveRecord(port: Int) -> ServeRecord? {
        guard let data = try? Data(contentsOf: locks.appending(path: "serve-\(port).json")) else { return nil }
        return try? JSONDecoder().decode(ServeRecord.self, from: data)
    }

    /// Why this Mac cannot run Splash, as the engine's own check says, which the launcher repeats
    /// before every start; nil where the check passes or gives no reason: it does not start, runs
    /// past 30 s, or fails saying nothing.
    @MainActor
    func unsupportedReason() async -> String? {
        let output = FileManager.default.temporaryDirectory.appending(path: "splash-device-check-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: output) }
        guard let status = await ChildProcess.run([engine.path, "device-check"], environment: environment, log: output,
                                                   timeout: .seconds(30)),
              status != 0, let said = try? String(contentsOf: output, encoding: .utf8)
        else { return nil }
        let reason = said.trimmingCharacters(in: .whitespacesAndNewlines).replacing(#/^error: /#, with: "")
        return reason.isEmpty ? nil : reason
    }
}

/// What a launcher writes of its start, as itself or as the server it runs as: the process, the
/// phase of its start, and, for a start that failed, why: the class of the failure, with what it
/// needed against what it had for a shortfall, and its message.
struct ServeRecord: Decodable, Equatable {
    /// Starting; loading the model's weights; running the model once (warming); answering; or
    /// failed.
    enum Phase: String, Decodable {
        case starting, loading, warming, ready, failed
    }

    struct Failure: Decodable, Equatable {
        /// The class of the failure: a `StartFailureKind` name, lowercased, which the engine, the
        /// launcher or the server writes, or the frontend's own "port", "upgrade" or "install"
        /// (`server/serve_record.py`).
        let kind: String
        let message: String
        /// What bounds a memory or context shortfall: "device_memory", "max_memory" or "disk".
        let bound: String?
        let required_bytes: Int64?
        let budget_bytes: Int64?
        let plan_context_tokens: Int?
    }

    let pid: Int32
    let phase: Phase
    let failure: Failure?
}

struct InstalledModel: Identifiable, Hashable, Sendable {
    /// Its Hugging Face ID, or the path of model files on this Mac.
    let id: String
    /// What its files take on disk, shared with any other installation of the same files; nil for
    /// model files on this Mac that are missing.
    var bytes: Int64?
    /// Whether model files on this Mac are no longer where they were added from.
    var missing = false
    /// Its family, as its assembly record names it; nil for model files on this Mac not prepared yet.
    var family: String?

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
