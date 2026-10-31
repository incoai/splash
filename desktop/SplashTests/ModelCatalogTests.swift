import Darwin
import XCTest
@testable import Splash

@MainActor
final class ModelCatalogTests: XCTestCase {
    func testHubCacheLocationFollowsHuggingFaceEnvironment() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        XCTAssertEqual(ModelCatalog.hubCacheURL(environment: [:]).path,
                       "\(home)/.cache/huggingface/hub")
        XCTAssertEqual(ModelCatalog.hubCacheURL(environment: ["XDG_CACHE_HOME": "/tmp/cache"]).path,
                       "/tmp/cache/huggingface/hub")
        XCTAssertEqual(ModelCatalog.hubCacheURL(environment: ["HF_HOME": "/tmp/hf"]).path,
                       "/tmp/hf/hub")
        XCTAssertEqual(ModelCatalog.hubCacheURL(environment: [
            "HF_HOME": "/tmp/hf", "HUGGINGFACE_HUB_CACHE": "/tmp/legacy", "HF_HUB_CACHE": "/tmp/models"
        ]).path, "/tmp/models")
    }

    func testCachedBytesSurviveIncompleteBlobRename() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: cache) }
        let blobs = cache.appendingPathComponent("models--prism-ml--Ternary-Bonsai-2-27B-gguf/blobs")
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let incomplete = blobs.appendingPathComponent("weight.incomplete")
        try Data(repeating: 0, count: 1024).write(to: incomplete)
        XCTAssertEqual(ModelCatalog.cachedBytes(repository: "prism-ml/Ternary-Bonsai-2-27B-gguf",
                                                in: cache), 1024)
        try FileManager.default.moveItem(at: incomplete, to: blobs.appendingPathComponent("weight"))
        XCTAssertEqual(ModelCatalog.cachedBytes(repository: "prism-ml/Ternary-Bonsai-2-27B-gguf",
                                                in: cache), 1024)
        XCTAssertEqual(ModelCatalog.cachedBytes(repository: "other/repo", in: cache), 0)
    }

    func testCancellingSizeEstimateStopsHelperPromptly() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let pidFile = runtime.appendingPathComponent("pid")
        let executable = bin.appendingPathComponent("python3")
        let script = """
        #!/usr/bin/python3
        import os
        import pathlib
        import time
        pathlib.Path(\(String(reflecting: pidFile.path))).write_text(str(os.getpid()))
        time.sleep(4)
        """
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)

        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"))
        let source = ModelCatalog.families[0].sources[0]
        let estimate = Task { await catalog.estimateSizes(for: source, runtimeRoot: runtime) }
        for _ in 0..<100 where !FileManager.default.fileExists(atPath: pidFile.path) {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8)))
        let cancelledAt = Date()
        estimate.cancel()
        await estimate.value

        XCTAssertLessThan(Date().timeIntervalSince(cancelledAt), 2)
        XCTAssertFalse(catalog.estimatingSourceIDs.contains(source.id))
        XCTAssertNotEqual(kill(pid, 0), 0)
    }

    func testListsOnlyValidInstalledSelections() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let assembly = root.appendingPathComponent(".resolved/one")
        try FileManager.default.createDirectory(at: assembly, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: assembly.appendingPathComponent("model.json"))
        let owner = root.appendingPathComponent("owner")
        try FileManager.default.createDirectory(at: owner, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: owner.appendingPathComponent("repo"), withDestinationURL: assembly)
        try FileManager.default.createSymbolicLink(at: owner.appendingPathComponent("broken"),
                                                   withDestinationURL: root.appendingPathComponent("missing"))
        try FileManager.default.createDirectory(at: owner.appendingPathComponent("directory"),
                                                withIntermediateDirectories: true)
        let officialID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let officialOwner = root.appendingPathComponent("unsloth")
        try FileManager.default.createDirectory(at: officialOwner, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: officialOwner.appendingPathComponent("Qwen3.8-27B-GGUF:UD-Q4_K_M"),
            withDestinationURL: assembly)
        let smallerID = "unsloth/Qwen3.8-27B-GGUF:UD-Q2_K_XL"
        try FileManager.default.createSymbolicLink(
            at: officialOwner.appendingPathComponent("Qwen3.8-27B-GGUF:UD-Q2_K_XL"),
            withDestinationURL: assembly)
        let catalog = ModelCatalog(modelsRoot: root)
        XCTAssertEqual(try catalog.installed().map(\.id), ["owner/repo", smallerID, officialID])
        catalog.refresh()
        let entries = catalog.entries()
        XCTAssertGreaterThan(entries.count, 2)
        XCTAssertFalse(entries.contains { $0.id == "owner/repo" })
        XCTAssertEqual(entries.first(where: { $0.id == officialID })?.isInstalled, true)
        XCTAssertEqual(entries.first(where: { $0.id == "mlx-community/Qwen3.8-27B-4bit" })?.isInstalled, false)
        XCTAssertEqual(entries.first(where: { $0.id == officialID })?
            .huggingFaceURL?.absoluteString, "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF")
    }

    func testInstalledModelSizeCountsUniqueManifestFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let assembly = root.appendingPathComponent(".resolved/one")
        let owner = root.appendingPathComponent("unsloth")
        try FileManager.default.createDirectory(at: assembly, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: owner, withIntermediateDirectories: true)
        let manifest = """
        {"files":{"target/model.gguf":{"path":"/cache/model.gguf","bytes":1000},
                  "draft/model.gguf":{"path":"/cache/draft.gguf","bytes":200},
                  "shared/model.gguf":{"path":"/cache/model.gguf","bytes":1000}}}
        """
        try Data(manifest.utf8).write(to: assembly.appendingPathComponent("model.json"))
        let id = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        try FileManager.default.createSymbolicLink(
            at: owner.appendingPathComponent("Qwen3.8-27B-GGUF:UD-Q4_K_M"),
            withDestinationURL: assembly)
        let catalog = ModelCatalog(modelsRoot: root)
        catalog.refresh()
        XCTAssertEqual(catalog.entries().first(where: { $0.id == id })?.sizeBytes, 1000)
    }

    func testPrepareStreamsOutputBeforeCommandFinishes() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        let script = install.appendingPathComponent("models.py")
        try Data("import time\nprint('first', flush=True)\ntime.sleep(1)\nprint('last', flush=True)\n".utf8)
            .write(to: script)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"))
        let id = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let preparation = Task {
            try await catalog.prepare(id: id, runtimeRoot: runtime)
        }
        for _ in 0..<20 where !(catalog.operationLogs[id] ?? "").contains("first") {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertTrue(catalog.busy)
        XCTAssertTrue(catalog.activeOperationIDs.contains(id))
        XCTAssertTrue((catalog.operationLogs[id] ?? "").contains("first"))
        try await preparation.value
        XCTAssertFalse(catalog.busy)
        XCTAssertTrue((catalog.operationLogs[id] ?? "").contains("last"))
        XCTAssertFalse(catalog.activeOperationIDs.contains(id))
    }

    func testPrepareTracksDraftAndModelDownloadStages() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        let script = install.appendingPathComponent("models.py")
        let source = """
        import sys, time
        sys.stdout.write('Fetch')
        sys.stdout.flush()
        time.sleep(0.1)
        print('ing 1 file(s), 3.85 GB, from incoai/Qwen3.8-27B-DFlash2@abc; cached files are reused.', flush=True)
        time.sleep(0.5)
        print('Installing prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0 as Qwen3.8-27B (gguf); draft incoai/Qwen3.8-27B-DFlash2; vision enabled.', flush=True)
        time.sleep(0.5)
        print('Fetching 1 file(s), 7.21 GB, from prism-ml/Ternary-Bonsai-2-27B-gguf@abc; cached files are reused.', flush=True)
        """
        try Data(source.utf8).write(to: script)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"))
        let id = "prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0"
        let preparation = Task { try await catalog.prepare(id: id, runtimeRoot: runtime) }
        for _ in 0..<40 where catalog.downloadPhaseByID[id] != .draft {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertEqual(catalog.downloadPhaseByID[id], .draft)
        for _ in 0..<40 where catalog.downloadPhaseByID[id] != .model {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertEqual(catalog.downloadPhaseByID[id], .model)
        try await preparation.value
    }

    func testCancellingPreparationStopsDownloadAndKeepsPartialCache() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        let script = install.appendingPathComponent("models.py")
        let source = """
        import os, pathlib, time
        blobs = pathlib.Path(os.environ['HF_HUB_CACHE']) / 'models--unsloth--Qwen3.8-27B-GGUF' / 'blobs'
        blobs.mkdir(parents=True)
        (blobs / 'weight.incomplete').write_bytes(b'partial')
        print('downloading', flush=True)
        time.sleep(2)
        """
        try Data(source.utf8).write(to: script)
        let cache = runtime.appendingPathComponent("cache")
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"),
                                   hubCacheRoot: cache)
        let id = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let preparation = Task {
            try await catalog.prepare(id: id, runtimeRoot: runtime)
        }
        for _ in 0..<20 where !(catalog.operationLogs[id] ?? "").contains("downloading") {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertTrue((catalog.operationLogs[id] ?? "").contains("downloading"))
        let cancelledAt = Date()
        preparation.cancel()
        do {
            try await preparation.value
            XCTFail("Cancelled download completed normally")
        } catch is CancellationError {
            XCTAssertLessThan(Date().timeIntervalSince(cancelledAt), 1)
        }
        XCTAssertFalse(catalog.busy)
        XCTAssertTrue(FileManager.default.fileExists(atPath: cache.appendingPathComponent(
            "models--unsloth--Qwen3.8-27B-GGUF/blobs/weight.incomplete").path))
    }

    func testPrepareReportsCacheGrowthWithoutLogOutput() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        let script = install.appendingPathComponent("models.py")
        let source = """
        import os, pathlib, time
        blobs = pathlib.Path(os.environ['HF_HUB_CACHE']) / 'models--unsloth--Qwen3.8-27B-GGUF' / 'blobs'
        blobs.mkdir(parents=True)
        (blobs / 'weight.incomplete').write_bytes(b'x' * 2048)
        time.sleep(2)
        """
        try Data(source.utf8).write(to: script)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"),
                                   hubCacheRoot: runtime.appendingPathComponent("cache"))
        let id = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let preparation = Task {
            try await catalog.prepare(id: id, runtimeRoot: runtime)
        }
        for _ in 0..<60 where (catalog.downloadedBytesByID[id] ?? 0) == 0 {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertTrue(catalog.busy)
        XCTAssertEqual(catalog.operationLogs[id], "")
        XCTAssertEqual(catalog.downloadedBytesByID[id], 2048)
        try await preparation.value
    }

    func testPrepareReportsMLXVariantCacheGrowth() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        let script = install.appendingPathComponent("models.py")
        let source = """
        import os, pathlib, time
        blobs = pathlib.Path(os.environ['HF_HUB_CACHE']) / 'models--mlx-community--Qwen3.8-27B-4bit' / 'blobs'
        blobs.mkdir(parents=True)
        (blobs / 'weight.incomplete').write_bytes(b'x' * 2048)
        time.sleep(2)
        """
        try Data(source.utf8).write(to: script)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"),
                                   hubCacheRoot: runtime.appendingPathComponent("cache"))
        let id = "mlx-community/Qwen3.8-27B-4bit"
        let preparation = Task {
            try await catalog.prepare(id: id, runtimeRoot: runtime)
        }
        for _ in 0..<60 where (catalog.downloadedBytesByID[id] ?? 0) == 0 {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertTrue(catalog.busy)
        XCTAssertEqual(catalog.downloadedBytesByID[id], 2048)
        try await preparation.value
    }

    func testConcurrentDownloadsCanBeCancelledIndependently() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        let script = install.appendingPathComponent("models.py")
        let source = """
        import sys, time
        model = sys.argv[sys.argv.index('--model') + 1]
        print(model, flush=True)
        time.sleep(2)
        """
        try Data(source.utf8).write(to: script)
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"))
        let firstID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        let secondID = "mlx-community/Qwen3.8-27B-4bit"
        let first = Task { try await catalog.prepare(id: firstID, runtimeRoot: runtime) }
        let second = Task { try await catalog.prepare(id: secondID, runtimeRoot: runtime) }
        for _ in 0..<40 where catalog.activeOperationIDs.count < 2 {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertEqual(catalog.activeOperationIDs, Set([firstID, secondID]))
        first.cancel()
        do {
            try await first.value
            XCTFail("Cancelled download completed normally")
        } catch is CancellationError {
        }
        XCTAssertTrue(catalog.activeOperationIDs.contains(secondID))
        XCTAssertTrue(catalog.busy)
        try await second.value
        XCTAssertFalse(catalog.busy)
        XCTAssertTrue((catalog.operationLogs[secondID] ?? "").contains(secondID))
    }

    func testOnlyOfficialIDsAreSupported() {
        let catalog = ModelCatalog()
        XCTAssertTrue(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"))
        XCTAssertTrue(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:UD-Q2_K_XL"))
        XCTAssertTrue(catalog.isSupportedModelID("mlx-community/Qwen3.8-27B-8bit"))
        XCTAssertFalse(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:BF16"))
        XCTAssertFalse(catalog.isSupportedModelID("mlx-community/Qwen3.8-27B-bf16"))
        XCTAssertFalse(catalog.isSupportedModelID("mlx-community/Qwen3.8-27B-4bit-DWQ"))
        XCTAssertTrue(catalog.isSupportedModelID("prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0"))
        XCTAssertFalse(catalog.isSupportedModelID("prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0"))
        XCTAssertTrue(catalog.isSupportedModelID("ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M"))
        XCTAssertTrue(catalog.isSupportedModelID("ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit"))
        XCTAssertFalse(catalog.isSupportedModelID("ornith-ai/Ornith-1.5-35B-A3B-GGUF:BF16"))
        XCTAssertFalse(catalog.isSupportedModelID("owner/repo"))
        XCTAssertFalse(catalog.isSupportedModelID("owner/repo:UD-Q4_K_M"))
        XCTAssertFalse(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:UD-Q7_K_M"))
    }

    func testTextOnlyOrnithSelectionAppearsInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let assembly = root.appendingPathComponent(".resolved/one")
        let selections = root.appendingPathComponent(".selections")
        try FileManager.default.createDirectory(at: assembly, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: selections, withIntermediateDirectories: true)
        try Data(#"{"files":{}}"#.utf8).write(to: assembly.appendingPathComponent("model.json"))
        let id = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
        try FileManager.default.createSymbolicLink(
            at: selections.appendingPathComponent(
                "a715abbab148724d2ca6c27338f598b81d766227129c9ce7485b22e619fdf1ab"),
            withDestinationURL: assembly)
        let catalog = ModelCatalog(modelsRoot: root)
        for bit in ["4bit", "6bit", "8bit"] {
            XCTAssertTrue(catalog.isLanguageOnly("ornith-ai/Ornith-1.5-35B-A3B-MLX-\(bit)"))
        }
        XCTAssertFalse(catalog.isLanguageOnly("ornith-ai/Ornith-1.5-122B-MLX-4bit"))
        catalog.refresh()
        XCTAssertEqual(catalog.entries().first(where: { $0.id == id })?.isInstalled, true)
    }

    func testTextOnlyOrnithPreparePassesLanguageOnly() async throws {
        let runtime = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: runtime) }
        let bin = runtime.appendingPathComponent("python/bin")
        let install = runtime.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python3"),
                                                   withDestinationURL: URL(fileURLWithPath: "/usr/bin/python3"))
        try Data("import sys\nprint(' '.join(sys.argv))\n".utf8)
            .write(to: install.appendingPathComponent("models.py"))
        let catalog = ModelCatalog(modelsRoot: runtime.appendingPathComponent("models"))
        let id = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
        try await catalog.prepare(id: id, runtimeRoot: runtime)
        XCTAssertTrue((catalog.operationLogs[id] ?? "").contains("--language-only prepare"))
    }

    func testConfiguredVariantsAreAvailableWithoutHubDiscovery() {
        let catalog = ModelCatalog()
        XCTAssertTrue(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:UD-IQ1_M"))
        XCTAssertTrue(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL"))
        XCTAssertFalse(catalog.isSupportedModelID("unsloth/Qwen3.8-27B-GGUF:UD-Q7_K_M"))
        let ids = catalog.entries().map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count)
        for family in ModelCatalog.families {
            for source in family.sources {
                XCTAssertTrue(source.quantizations.contains(source.recommendedQuantization), source.id)
            }
        }
    }

    func testFlintListsReleasedGGUFVariantsOnly() {
        let catalog = ModelCatalog()
        let ids = catalog.entries().map(\.id).filter { $0.hasPrefix("unsloth/flint-GGUF:") }
        XCTAssertEqual(Set(ids), Set([
            "unsloth/flint-GGUF:UD-IQ1_M", "unsloth/flint-GGUF:UD-IQ2_XXS",
            "unsloth/flint-GGUF:UD-IQ3_XXS", "unsloth/flint-GGUF:UD-Q2_K_XL",
            "unsloth/flint-GGUF:UD-Q3_K_XL", "unsloth/flint-GGUF:UD-Q4_K_XL",
            "unsloth/flint-GGUF:UD-Q5_K_XL", "unsloth/flint-GGUF:UD-Q6_K_XL",
            "unsloth/flint-GGUF:Q8_0"
        ]))
        XCTAssertTrue(catalog.isSupportedModelID("unsloth/flint-GGUF:UD-Q4_K_XL"))
        XCTAssertFalse(catalog.isSupportedModelID("unsloth/flint-GGUF:BF16"))
        XCTAssertFalse(catalog.isSupportedModelID("unsloth/flint-MLX-4bit"))
    }

    func testFlintUsesItsModelLogo() {
        let family = ModelCatalog.families.first { $0.id == "flint" }
        XCTAssertEqual(family?.iconAssetName, "ModelFlint")
        XCTAssertEqual(ModelCatalog().entries().first { $0.id == "unsloth/flint-GGUF:UD-Q4_K_XL" }?
            .iconAssetName, "ModelFlint")
    }

}
