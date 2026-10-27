import Darwin
import Carbon
import XCTest
@testable import Splash

@MainActor
final class DesktopStateTests: XCTestCase {
    func testModelsIsDefaultManagerSection() {
        XCTAssertEqual(makeState().managerSection, .models)
    }

    func testLoginItemEventSuppressesManagerWindow() {
        XCTAssertTrue(ManagerLaunchPolicy.shouldOpenWindow(for: nil))
        let loginEvent = NSAppleEventDescriptor(
            eventClass: AEEventClass(kCoreEventClass), eventID: AEEventID(kAEOpenApplication),
            targetDescriptor: nil, returnID: AEReturnID(kAutoGenerateReturnID),
            transactionID: AETransactionID(kAnyTransactionID))
        XCTAssertTrue(ManagerLaunchPolicy.shouldOpenWindow(for: loginEvent))
        loginEvent.setParam(NSAppleEventDescriptor(boolean: true),
                            forKeyword: AEKeyword(keyAELaunchedAsLogInItem))
        XCTAssertFalse(ManagerLaunchPolicy.shouldOpenWindow(for: loginEvent))
    }

    func testActionsRequireSelectionAndReadyService() {
        let state = makeState()
        XCTAssertFalse(state.canStart)
        XCTAssertFalse(state.canOpenChat)
        XCTAssertFalse(state.canLaunchChat(for: "owner/repo"))
        state.selectModel("owner/repo")
        XCTAssertNil(state.selectedModelID)
        XCTAssertFalse(state.canStart)
        XCTAssertFalse(state.canOpenChat)
    }

    func testInstalledSelectionCanRun() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let assembly = root.appendingPathComponent(".resolved/one")
        let owner = root.appendingPathComponent("unsloth")
        try FileManager.default.createDirectory(at: assembly, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: owner, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: assembly.appendingPathComponent("model.json"))
        try FileManager.default.createSymbolicLink(at: owner.appendingPathComponent("Qwen3.8-27B-GGUF:UD-Q4_K_M"),
                                                   withDestinationURL: assembly)
        let catalog = ModelCatalog(modelsRoot: root)
        catalog.refresh()
        let state = DesktopState(settings: AppSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!),
                                 service: ServiceController(), catalog: catalog,
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: []))
        let officialID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        state.selectModel(officialID)
        XCTAssertTrue(state.canStart)
        XCTAssertTrue(state.canRun(officialID))
        XCTAssertTrue(state.canLaunchChat(for: officialID))
        XCTAssertFalse(state.canRun("owner/repo"))
    }

    func testSecondChatRequestCannotReplacePendingModelStart() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let assembly = root.appendingPathComponent(".resolved/one")
        let owner = root.appendingPathComponent("unsloth")
        try FileManager.default.createDirectory(at: assembly, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: owner, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: assembly.appendingPathComponent("model.json"))
        let firstID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let secondID = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
        for id in [firstID, secondID] {
            let name = String(id.split(separator: "/", maxSplits: 1)[1])
            try FileManager.default.createSymbolicLink(at: owner.appendingPathComponent(name),
                                                       withDestinationURL: assembly)
        }
        let catalog = ModelCatalog(modelsRoot: root)
        catalog.refresh()
        let defaultsName = UUID().uuidString
        let defaults = UserDefaults(suiteName: defaultsName)!
        defer { defaults.removePersistentDomain(forName: defaultsName) }
        let state = DesktopState(settings: AppSettings(defaults: defaults), service: ServiceController(),
                                 catalog: catalog, runtime: RuntimeInstaller(manifest: nil, externalRoots: []))

        XCTAssertTrue(state.beginChat(for: firstID))
        XCTAssertTrue(state.beginChat(for: firstID))
        XCTAssertFalse(state.beginChat(for: secondID))
        XCTAssertEqual(state.chatRequestedModelID, firstID)
        for _ in 0..<50 where state.chatStartPending {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(state.chatStartPending)
    }

    func testDownloadingAnotherModelDoesNotBlockInstalledModel() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelsRoot = root.appendingPathComponent("models")
        let assembly = modelsRoot.appendingPathComponent(".resolved/one")
        let owner = modelsRoot.appendingPathComponent("unsloth")
        try FileManager.default.createDirectory(at: assembly, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: owner, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: assembly.appendingPathComponent("model.json"))
        let installedID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        try FileManager.default.createSymbolicLink(
            at: owner.appendingPathComponent("Qwen3.8-27B-GGUF:UD-Q4_K_M"),
            withDestinationURL: assembly)
        let python = root.appendingPathComponent("python/bin")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: python, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: python.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        try Data("{\"version\":\"1.4.0\"}".utf8).write(to: root.appendingPathComponent("release.json"))
        let helper = install.appendingPathComponent("models.py")
        let source = """
        import json, sys, time
        if sys.argv[-1] == 'plan':
            print(json.dumps({'model_revision': 'target-abc',
                              'draft_repository': 'incoai/Qwen3.8-27B-DFlash2',
                              'draft_revision': 'abc', 'draft_needed': False,
                              'draft_size_bytes': 0}))
        else:
            print('downloading', flush=True)
            time.sleep(10)
        """
        try Data(source.utf8).write(to: helper)
        try Data("import time\ntime.sleep(10)\n".utf8)
            .write(to: install.appendingPathComponent("launcher.py"))
        let catalog = ModelCatalog(modelsRoot: modelsRoot, downloadToolsURL: helper)
        catalog.refresh()
        let settings = AppSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        settings.port = try XCTUnwrap((22001...23000).first(where: PortAvailability.isFree))
        let state = DesktopState(settings: settings,
                                 service: ServiceController(), catalog: catalog,
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: [root]))

        let downloadID = "mlx-community/Qwen3.8-27B-4bit"
        state.beginModelDownload(downloadID)
        for _ in 0..<100 where !(catalog.operationLogs[downloadID] ?? "").contains("downloading") {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue((catalog.operationLogs[downloadID] ?? "").contains("downloading"))
        XCTAssertTrue(catalog.activeOperationIDs.contains(downloadID))
        XCTAssertTrue(catalog.busy)
        XCTAssertTrue(state.canRun(installedID))
        await state.startModel(installedID)
        XCTAssertEqual(state.service.runningModelID, installedID)
        XCTAssertNotNil(state.service.runningPort)
        state.cancelAllModelDownloads()
        await state.stop(force: true)
        for _ in 0..<100 where !state.downloadingModelIDs.isEmpty {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(state.downloadingModelIDs.isEmpty)
        XCTAssertFalse(catalog.busy)
    }

    func testMultipleDownloadsTrackAndCancelSeparately() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        try Data("{\"version\":\"1.4.0\"}".utf8).write(to: runtime.appendingPathComponent("release.json"))
        try Data("".utf8).write(to: install.appendingPathComponent("launcher.py"))
        let helper = install.appendingPathComponent("models.py")
        let source = """
        import json, sys, time
        if sys.argv[-1] == 'plan':
            print(json.dumps({'model_revision': 'target-abc',
                              'draft_repository': 'incoai/Qwen3.8-27B-DFlash2',
                              'draft_revision': 'abc', 'draft_needed': False,
                              'draft_size_bytes': 0}))
        else:
            model = sys.argv[sys.argv.index('--model') + 1]
            print(model, flush=True)
            time.sleep(3)
        """
        try Data(source.utf8).write(to: helper)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"),
                                   downloadToolsURL: helper)
        let state = DesktopState(settings: AppSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!),
                                 service: ServiceController(), catalog: catalog,
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: [runtime]))
        let firstID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let secondID = "mlx-community/Qwen3.8-27B-4bit"
        state.beginModelDownload(firstID)
        state.beginModelDownload(secondID)
        for _ in 0..<40 where catalog.activeOperationIDs.count < 2 {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertEqual(state.downloadingModelIDs, Set([firstID, secondID]))
        XCTAssertEqual(catalog.activeOperationIDs.count, 2)
        state.cancelModelDownload(firstID)
        for _ in 0..<40 where state.downloadingModelIDs.contains(firstID) {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertEqual(state.downloadingModelIDs, Set([secondID]))
        XCTAssertTrue(catalog.activeOperationIDs.contains(secondID))
        state.cancelAllModelDownloads()
        for _ in 0..<40 where !state.downloadingModelIDs.isEmpty {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertTrue(state.downloadingModelIDs.isEmpty)
        XCTAssertFalse(catalog.busy)
    }

    func testSharedDraftHasIndependentCancellationAndCanBeRetried() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        try Data("{\"version\":\"1.4.0\"}".utf8).write(to: runtime.appendingPathComponent("release.json"))
        try Data("".utf8).write(to: install.appendingPathComponent("launcher.py"))
        try Data("".utf8).write(to: install.appendingPathComponent("models.py"))
        let helper = runtime.appendingPathComponent("download_tools.py")
        let source = """
        import json, pathlib, sys, time
        command = sys.argv[-1]
        if command == 'plan':
            print(json.dumps({'model_revision': 'target-abc',
                              'draft_repository': 'incoai/Qwen3.8-27B-DFlash2',
                              'draft_revision': 'abc', 'draft_needed': True,
                              'draft_size_bytes': 2048}))
        elif command == 'assemble':
            print('assembled', flush=True)
        else:
            print(command, flush=True)
            gate = pathlib.Path(__file__).parent / ('release-draft' if command == 'fetch-draft' else 'release-model')
            while not gate.exists():
                time.sleep(0.05)
        """
        try Data(source.utf8).write(to: helper)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"),
                                   downloadToolsURL: helper)
        let state = DesktopState(settings: AppSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!),
                                 service: ServiceController(), catalog: catalog,
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: [runtime]))
        let firstID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let secondID = "mlx-community/Qwen3.8-27B-4bit"
        state.beginModelDownload(firstID)
        state.beginModelDownload(secondID)
        for _ in 0..<100 where catalog.activeOperationIDs.count < 3 {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        let draftID = try XCTUnwrap(state.downloadingDrafts.keys.first)
        XCTAssertEqual(state.downloadingDrafts.count, 1)
        XCTAssertEqual(catalog.activeOperationIDs.count, 3)

        state.cancelModelDownload(firstID)
        for _ in 0..<100 where state.downloadingModelIDs.contains(firstID) {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(state.downloadingDrafts.keys.contains(draftID))
        XCTAssertTrue(state.downloadingModelIDs.contains(secondID))

        state.cancelDraftDownload(draftID)
        for _ in 0..<100 where !state.downloadingDrafts.isEmpty {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(state.downloadingModelIDs.contains(secondID))
        try Data().write(to: runtime.appendingPathComponent("release-model"))
        for _ in 0..<100 where !state.waitingForDraftModelIDs.contains(secondID) {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(state.waitingForDraftModelIDs.contains(secondID))

        state.retryDraft(for: secondID)
        for _ in 0..<100 where state.downloadingDrafts.isEmpty {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(state.downloadingDrafts.count, 1)
        try Data().write(to: runtime.appendingPathComponent("release-draft"))
        for _ in 0..<100 where state.downloadingModelIDs.contains(secondID) {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(state.downloadingModelIDs.isEmpty)
        XCTAssertTrue(state.downloadingDrafts.isEmpty)
        XCTAssertTrue((catalog.operationLogs[secondID] ?? "").contains("assembled"))
    }

    func testStartingServiceIdentifiesItsModel() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let script = root.appendingPathComponent("fake_service.sh")
        try Data("#!/bin/sh\nsleep 5\n".utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let service = ServiceController()
        let port = try XCTUnwrap((19001...20000).first(where: PortAvailability.isFree))
        let id = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        try service.start(command: SplashCommand(executable: script, prefixArguments: [], environment: [:]),
                          modelID: id, port: port, idleRelease: .tenMinutes)
        let state = DesktopState(settings: AppSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!),
                                 service: service, catalog: ModelCatalog(modelsRoot: root),
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: []))
        XCTAssertTrue(state.isStartingModel(id))
        XCTAssertFalse(state.isStartingModel("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"))
        await service.stop(force: true)
        XCTAssertFalse(state.isStartingModel(id))
    }

    func testSavedUnsupportedSelectionIsIgnored() {
        let name = UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let settings = AppSettings(defaults: defaults)
        settings.selectedModelID = "owner/custom"
        let state = DesktopState(settings: settings, service: ServiceController(),
                                 catalog: ModelCatalog(modelsRoot: FileManager.default.temporaryDirectory),
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: []))
        XCTAssertNil(state.selectedModelID)
        XCTAssertFalse(state.canStart)
    }

    func testChangingModelCacheDirectoryUpdatesCatalogAndSavedSetting() throws {
        let name = UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let settings = AppSettings(defaults: defaults)
        let catalog = ModelCatalog(modelsRoot: FileManager.default.temporaryDirectory)
        let state = DesktopState(settings: settings, service: ServiceController(), catalog: catalog,
                                 runtime: RuntimeInstaller(manifest: nil, externalRoots: []))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(state.changeModelCacheDirectory(to: directory))
        XCTAssertEqual(catalog.hubCacheRoot, directory.standardizedFileURL)
        XCTAssertEqual(AppSettings(defaults: defaults).modelCacheURL, directory.standardizedFileURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        let reloadedCatalog = ModelCatalog(modelsRoot: FileManager.default.temporaryDirectory)
        _ = DesktopState(settings: AppSettings(defaults: defaults), service: ServiceController(),
                         catalog: reloadedCatalog,
                         runtime: RuntimeInstaller(manifest: nil, externalRoots: []))
        XCTAssertEqual(reloadedCatalog.hubCacheRoot, directory.standardizedFileURL)
        XCTAssertTrue(state.changeModelCacheDirectory(to: nil))
        XCTAssertEqual(catalog.hubCacheRoot, ModelCatalog.hubCacheURL())
        XCTAssertNil(settings.modelCacheURL)
    }

    func testOccupiedPortDoesNotReplaceSavedPort() throws {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: INADDR_LOOPBACK.bigEndian)
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bound, 0)
        XCTAssertEqual(Darwin.listen(descriptor, 1), 0)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                _ = getsockname(descriptor, $0, &length)
            }
        }
        let state = makeState()
        state.portText = String(UInt16(bigEndian: address.sin_port))
        XCTAssertFalse(state.savePort())
        XCTAssertEqual(state.settings.port, 8000)
    }

    private func makeState() -> DesktopState {
        let name = UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return DesktopState(settings: AppSettings(defaults: defaults), service: ServiceController(),
                            catalog: ModelCatalog(modelsRoot: FileManager.default.temporaryDirectory),
                            runtime: RuntimeInstaller(manifest: nil, externalRoots: []))
    }
}
