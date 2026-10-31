import XCTest
@testable import Splash

@MainActor
final class RuntimeInstallerTests: XCTestCase {
    func testFindsCompatibleExternalRuntime() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let external = base.appendingPathComponent("external")
        try FileManager.default.createDirectory(at: external.appendingPathComponent("python/bin"),
                                                withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: external.appendingPathComponent("install"),
                                                withIntermediateDirectories: true)
        try Data(#"{"version":"1.4.0"}"#.utf8).write(to: external.appendingPathComponent("release.json"))
        let python = external.appendingPathComponent("python/bin/python3")
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: python)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)
        try Data().write(to: external.appendingPathComponent("install/launcher.py"))
        let installer = RuntimeInstaller(root: base.appendingPathComponent("managed"),
                                         manifest: nil, externalRoots: [external])
        let location = try await installer.resolve()
        XCTAssertEqual(location.origin, .external)
        XCTAssertEqual(location.version, "1.4.0")
        XCTAssertEqual(location.command.arguments(modelID: "owner/repo", port: 8000,
                                                  idleRelease: .tenMinutes).suffix(2),
                       ["--idle-release", "10m"])
    }

    func testStreamingDigest() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("abc".utf8).write(to: file)
        XCTAssertEqual(try RuntimeInstaller.sha256(file),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testInstallsVerifiedArchiveAndKeepsPriorOnChecksumFailure() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let old = base.appendingPathComponent("managed/1.4.0")
        try makeRuntime(at: old, version: "1.4.0")
        let package = base.appendingPathComponent("package")
        let newRuntime = package.appendingPathComponent("splash-1.5.0-arm64-macos26")
        try makeRuntime(at: newRuntime, version: "1.5.0")
        let archive = base.appendingPathComponent("runtime.tar.gz")
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-czf", archive.path, "-C", package.path,
                         "splash-1.5.0-arm64-macos26"]
        try tar.run()
        tar.waitUntilExit()
        XCTAssertEqual(tar.terminationStatus, 0)

        let bad = RuntimeManifest(version: "1.5.0", archiveURL: archive,
                                  sha256: String(repeating: "0", count: 64),
                                  minimumCompatibleVersion: "1.4.0")
        let fallback = RuntimeInstaller(root: base.appendingPathComponent("managed"),
                                        manifest: bad, externalRoots: [])
        let fallbackLocation = try await fallback.resolve()
        XCTAssertEqual(fallbackLocation.version, "1.4.0")
        XCTAssertFalse(FileManager.default.fileExists(atPath: base.appendingPathComponent("managed/1.5.0").path))

        let good = RuntimeManifest(version: "1.5.0", archiveURL: archive,
                                   sha256: try RuntimeInstaller.sha256(archive),
                                   minimumCompatibleVersion: "1.4.0")
        let installer = RuntimeInstaller(root: base.appendingPathComponent("managed"),
                                         manifest: good, externalRoots: [])
        let location = try await installer.resolve()
        XCTAssertEqual(location.version, "1.5.0")
        XCTAssertEqual(location.origin, .managed)
        XCTAssertTrue(FileManager.default.fileExists(atPath: old.appendingPathComponent("release.json").path))
    }

    func testConcurrentResolveInstallsTargetOnlyOnce() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let package = base.appendingPathComponent("package")
        let runtime = package.appendingPathComponent("splash-1.5.0-arm64-macos26")
        try makeRuntime(at: runtime, version: "1.5.0")
        let verificationLog = base.appendingPathComponent("verifications.log")
        let release = base.appendingPathComponent("release-verification")
        let python = runtime.appendingPathComponent("python/bin/python3")
        let script = """
        #!/bin/sh
        printf 'verified\\n' >> "\(verificationLog.path)"
        attempts=0
        while [ ! -e "\(release.path)" ]; do
            attempts=$((attempts + 1))
            [ "$attempts" -lt 200 ] || exit 1
            /bin/sleep 0.05
        done
        exit 0
        """
        try Data(script.utf8).write(to: python)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)
        let archive = base.appendingPathComponent("runtime.tar.gz")
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-czf", archive.path, "-C", package.path, runtime.lastPathComponent]
        try tar.run()
        tar.waitUntilExit()
        XCTAssertEqual(tar.terminationStatus, 0)
        let manifest = RuntimeManifest(version: "1.5.0", archiveURL: archive,
                                       sha256: try RuntimeInstaller.sha256(archive),
                                       minimumCompatibleVersion: "1.4.0")
        let installerRoot = base.appendingPathComponent("managed")
        let installer = RuntimeInstaller(root: installerRoot,
                                         manifest: manifest, externalRoots: [])

        let first = Task { try await installer.resolve() }
        defer { try? Data().write(to: release) }
        for _ in 0..<100 where !FileManager.default.fileExists(atPath: verificationLog.path) {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: verificationLog.path))
        XCTAssertEqual(installStageCount(in: installerRoot), 1)
        var secondCallEntered = false
        let second = Task {
            secondCallEntered = true
            return try await installer.resolve()
        }
        for _ in 0..<100 where !secondCallEntered { await Task.yield() }
        XCTAssertTrue(secondCallEntered)
        var peakStageCount = installStageCount(in: installerRoot)
        for _ in 0..<50 where peakStageCount < 2 {
            try await Task.sleep(nanoseconds: 20_000_000)
            peakStageCount = max(peakStageCount, installStageCount(in: installerRoot))
        }
        try Data().write(to: release)
        let firstLocation = try await first.value
        let secondLocation = try await second.value
        XCTAssertEqual(firstLocation.root, secondLocation.root)
        XCTAssertEqual(peakStageCount, 1)
        XCTAssertEqual(verificationCount(at: verificationLog), 1)
    }

    func testInvalidArchiveWithoutFallbackResetsBusyAndRemovesStage() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let package = base.appendingPathComponent("package")
        let unexpected = package.appendingPathComponent("not-a-splash-runtime")
        try FileManager.default.createDirectory(at: unexpected, withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: unexpected.appendingPathComponent("file"))
        let archive = base.appendingPathComponent("runtime.tar.gz")
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-czf", archive.path, "-C", package.path, unexpected.lastPathComponent]
        try tar.run()
        tar.waitUntilExit()
        XCTAssertEqual(tar.terminationStatus, 0)
        let manifest = RuntimeManifest(version: "1.5.0", archiveURL: archive,
                                       sha256: try RuntimeInstaller.sha256(archive),
                                       minimumCompatibleVersion: "1.4.0")
        let managed = base.appendingPathComponent("managed")
        let installer = RuntimeInstaller(root: managed, manifest: manifest, externalRoots: [])

        do {
            _ = try await installer.resolve()
            XCTFail("Invalid archive should not resolve without a fallback")
        } catch RuntimeError.invalidArchive {
        } catch {
            XCTFail("Expected invalid archive, got \(error)")
        }
        XCTAssertNil(installer.location)
        XCTAssertFalse(installer.busy)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: managed.path)
        XCTAssertFalse(remaining.contains { $0.hasPrefix(".install-") })
    }

    private func verificationCount(at url: URL) -> Int {
        guard let log = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return log.split(separator: "\n").count
    }

    private func installStageCount(in root: URL) -> Int {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        return names.filter { $0.hasPrefix(".install-") }.count
    }

    private func makeRuntime(at root: URL, version: String) throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("python/bin"),
                                                withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("install"),
                                                withIntermediateDirectories: true)
        try Data("{\"version\":\"\(version)\"}".utf8).write(to: root.appendingPathComponent("release.json"))
        let python = root.appendingPathComponent("python/bin/python3")
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: python)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)
        try Data().write(to: root.appendingPathComponent("install/launcher.py"))
    }
}
