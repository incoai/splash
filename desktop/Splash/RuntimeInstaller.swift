import CryptoKit
import Foundation

enum RuntimeOrigin: String {
    case managed
    case external
}

struct RuntimeLocation {
    let root: URL
    let version: String
    let origin: RuntimeOrigin

    var command: SplashCommand {
        SplashCommand(executable: root.appendingPathComponent("python/bin/python3"),
                      prefixArguments: ["-u", root.appendingPathComponent("install/launcher.py").path],
                      environment: ["PYTHONPATH": root.path, "PYTHONDONTWRITEBYTECODE": "1"])
    }
}

struct RuntimeManifest: Decodable {
    let version: String
    let archiveURL: URL
    let sha256: String
    let minimumCompatibleVersion: String
}

enum RuntimeError: LocalizedError {
    case unavailable
    case invalidManifest
    case checksumMismatch
    case invalidArchive
    case verificationFailed

    var errorDescription: String? {
        switch self {
        case .unavailable: return "No compatible Splash runtime was found. Reinstall Splash or install a compatible runtime."
        case .invalidManifest: return "The bundled runtime manifest is invalid."
        case .checksumMismatch: return "The runtime download failed its SHA-256 check."
        case .invalidArchive: return "The runtime archive has an unexpected layout."
        case .verificationFailed: return "The Splash runtime could not be verified."
        }
    }
}

@MainActor
final class RuntimeInstaller: ObservableObject {
    @Published private(set) var location: RuntimeLocation?
    @Published private(set) var progress = ""
    @Published private(set) var busy = false

    private let root: URL
    private let manifest: RuntimeManifest?
    private let externalRoots: [URL]
    private var resolutionTask: Task<RuntimeLocation, Error>?

    init(root: URL = FileManager.default.homeDirectoryForCurrentUser
             .appendingPathComponent("Library/Application Support/Splash/desktop/runtimes"),
         manifest: RuntimeManifest? = RuntimeInstaller.bundledManifest(),
         externalRoots: [URL] = [URL(fileURLWithPath: "/opt/homebrew/opt/splash/libexec")]) {
        self.root = root
        self.manifest = manifest
        self.externalRoots = externalRoots
    }

    func resolve() async throws -> RuntimeLocation {
        if let location { return location }
        if let resolutionTask { return try await resolutionTask.value }
        let task = Task { try await resolveUncached() }
        resolutionTask = task
        defer { resolutionTask = nil }
        return try await task.value
    }

    private func resolveUncached() async throws -> RuntimeLocation {
        let minimum = manifest?.minimumCompatibleVersion ?? "1.4.0"
        if let manifest,
           let version = Self.verifiedVersion(at: root.appendingPathComponent(manifest.version)),
           version == manifest.version {
            let found = RuntimeLocation(root: root.appendingPathComponent(manifest.version),
                                        version: version, origin: .managed)
            location = found
            return found
        }
        let external = externalRoots.compactMap { candidate -> RuntimeLocation? in
            guard let version = Self.verifiedVersion(at: candidate),
                  Self.compareVersions(version, minimum) >= 0 else { return nil }
            return RuntimeLocation(root: candidate, version: version, origin: .external)
        }.first
        if let external, newestManaged(minimum: minimum) == nil {
            location = external
            return external
        }
        guard let manifest else {
            if let found = newestManaged(minimum: minimum) ?? external {
                location = found
                return found
            }
            throw RuntimeError.unavailable
        }
        busy = true
        defer { busy = false }
        do {
            let installed = try await install(manifest)
            location = installed
            return installed
        } catch {
            if let fallback = newestManaged(minimum: minimum) ?? external {
                location = fallback
                return fallback
            }
            throw error
        }
    }

    private func newestManaged(minimum: String) -> RuntimeLocation? {
        let versions = (try? FileManager.default.contentsOfDirectory(at: root,
            includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
        return versions.compactMap { directory -> RuntimeLocation? in
            guard let version = Self.verifiedVersion(at: directory),
                  Self.compareVersions(version, minimum) >= 0 else { return nil }
            return RuntimeLocation(root: directory, version: version, origin: .managed)
        }.sorted { Self.compareVersions($0.version, $1.version) > 0 }.first
    }

    nonisolated private static func verifiedVersion(at directory: URL) -> String? {
        let release = directory.appendingPathComponent("release.json")
        guard let data = try? Data(contentsOf: release),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = json["version"] as? String,
              FileManager.default.isExecutableFile(atPath: directory.appendingPathComponent("python/bin/python3").path),
              FileManager.default.fileExists(atPath: directory.appendingPathComponent("install/launcher.py").path)
        else { return nil }
        return version
    }

    private func install(_ manifest: RuntimeManifest) async throws -> RuntimeLocation {
        guard manifest.sha256.count == 64,
              manifest.sha256.unicodeScalars.allSatisfy({ CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0) }),
              !manifest.version.isEmpty else { throw RuntimeError.invalidManifest }
        progress = "Downloading Splash \(manifest.version)..."
        let manager = FileManager.default
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        let stage = root.appendingPathComponent(".install-\(UUID().uuidString)")
        try manager.createDirectory(at: stage, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: stage) }
        let archive = stage.appendingPathComponent("runtime.tar.gz")
        if manifest.archiveURL.isFileURL {
            try manager.copyItem(at: manifest.archiveURL, to: archive)
        } else {
            let transfer = try await URLSession.shared.download(from: manifest.archiveURL)
            try manager.moveItem(at: transfer.0, to: archive)
        }
        let digest = try await Task.detached(priority: .userInitiated) {
            try Self.sha256(archive)
        }.value
        guard digest.caseInsensitiveCompare(manifest.sha256) == .orderedSame else {
            throw RuntimeError.checksumMismatch
        }
        progress = "Installing Splash \(manifest.version)..."
        let root = self.root
        let destination = try await Task.detached(priority: .userInitiated) {
            try Self.extractAndVerify(archive: archive, manifest: manifest, root: root, stage: stage)
        }.value
        progress = "Splash \(manifest.version) is ready."
        return RuntimeLocation(root: destination, version: manifest.version, origin: .managed)
    }

    nonisolated private static func extractAndVerify(archive: URL, manifest: RuntimeManifest,
                                                      root: URL, stage: URL) throws -> URL {
        let manager = FileManager.default
        let folder = "splash-\(manifest.version)-arm64-macos26"
        let entries = try archiveEntries(archive)
        guard !entries.isEmpty, entries.allSatisfy({ entry in
            entry == folder || (entry.hasPrefix(folder + "/") &&
                !entry.split(separator: "/").contains(".."))
        }) else { throw RuntimeError.invalidArchive }
        try run("/usr/bin/tar", ["-xzf", archive.path, "-C", stage.path])
        let candidate = stage.appendingPathComponent(folder)
        guard verifiedVersion(at: candidate) == manifest.version else { throw RuntimeError.invalidArchive }
        do {
            try run(candidate.appendingPathComponent("python/bin/python3").path,
                    ["-u", candidate.appendingPathComponent("install/launcher.py").path, "--help"])
        } catch { throw RuntimeError.verificationFailed }
        let destination = root.appendingPathComponent(manifest.version)
        if !manager.fileExists(atPath: destination.path) {
            try manager.moveItem(at: candidate, to: destination)
        }
        guard verifiedVersion(at: destination) == manifest.version else { throw RuntimeError.verificationFailed }
        return destination
    }

    nonisolated private static func archiveEntries(_ archive: URL) throws -> [String] {
        let output = try run("/usr/bin/tar", ["-tzf", archive.path])
        return output.split(separator: "\n").map(String.init).map { $0.hasSuffix("/") ? String($0.dropLast()) : $0 }
    }

    @discardableResult
    nonisolated private static func run(_ executable: String, _ arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw RuntimeError.verificationFailed }
        return String(decoding: output, as: UTF8.self)
    }

    static func compareVersions(_ left: String, _ right: String) -> Int {
        let a = left.split(separator: ".").compactMap { Int($0) }
        let b = right.split(separator: ".").compactMap { Int($0) }
        for index in 0..<max(a.count, b.count) {
            let difference = (index < a.count ? a[index] : 0) - (index < b.count ? b[index] : 0)
            if difference != 0 { return difference < 0 ? -1 : 1 }
        }
        return 0
    }

    nonisolated static func sha256(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while let chunk = try file.read(upToCount: 1024 * 1024), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    nonisolated static func bundledManifest() -> RuntimeManifest? {
        guard let url = Bundle.main.url(forResource: "runtime-manifest", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(RuntimeManifest.self, from: data)
    }
}
