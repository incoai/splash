import CryptoKit
import Foundation

struct InstalledModel: Identifiable, Hashable {
    let id: String
    let assembly: URL
    let sizeBytes: Int64?
}

struct ModelEntry: Identifiable {
    let id: String
    let title: String
    let iconAssetName: String
    let format: String
    let quantization: String
    let languageOnly: Bool
    let assembly: URL?
    let sizeBytes: Int64?

    var isInstalled: Bool { assembly != nil }
    var huggingFaceURL: URL? {
        guard let repo = id.split(separator: ":", maxSplits: 1).first, repo.contains("/") else {
            return nil
        }
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repo)"
        return components.url
    }
}

enum ModelFormat: String, Hashable {
    case gguf = "GGUF"
    case mlx = "MLX"
}

enum ModelDownloadPhase: Equatable {
    case preparing
    case draft
    case model
    case installing
}

struct ModelDownloadPlan: Decodable {
    let modelRevision: String
    let draftRepository: String
    let draftRevision: String
    let draftNeeded: Bool
    let draftSizeBytes: Int64?

    var draftID: String { "draft:\(draftRepository)@\(draftRevision)" }
}

struct ModelSource: Identifiable, Hashable {
    let format: ModelFormat
    let repository: String
    let quantizations: [String]
    let recommendedQuantization: String
    let languageOnly: Bool

    init(format: ModelFormat, repository: String, quantizations: [String],
         recommendedQuantization: String, languageOnly: Bool = false) {
        self.format = format
        self.repository = repository
        self.quantizations = quantizations
        self.recommendedQuantization = recommendedQuantization
        self.languageOnly = languageOnly
    }

    var id: String { "\(format.rawValue):\(repository)" }
    func modelID(for variant: String) -> String {
        switch format {
        case .gguf: return "\(repository):\(variant)"
        case .mlx: return "\(repository)-\(variant)"
        }
    }

    func variant(in modelID: String) -> String? {
        quantizations.first { self.modelID(for: $0) == modelID }
    }
}

struct ModelFamily: Identifiable {
    let id: String
    let title: String
    let iconAssetName: String
    let sources: [ModelSource]
}

private struct ModelManifest: Decodable {
    struct FileRecord: Decodable {
        let path: String
        let bytes: Int64
    }

    let files: [String: FileRecord]

    var sizeBytes: Int64 {
        var paths = Set<String>()
        return files.reduce(into: 0) { total, entry in
            guard !entry.key.hasPrefix("draft/") else { return }
            let file = entry.value
            if paths.insert(file.path).inserted { total += file.bytes }
        }
    }
}

@MainActor
final class ModelCatalog: ObservableObject {
    @Published private(set) var models: [InstalledModel] = []
    @Published private(set) var activeOperationIDs: Set<String> = []
    @Published private(set) var operationLogs: [String: String] = [:]
    @Published private(set) var downloadedBytesByID: [String: Int64] = [:]
    @Published private(set) var downloadPhaseByID: [String: ModelDownloadPhase] = [:]
    @Published private(set) var estimatedSizes: [String: Int64] = [:]
    @Published private(set) var estimatingSourceIDs: Set<String> = []
    private var activeOperations: [String: (id: UUID, process: Process)] = [:]
    private var pendingLogLines: [String: String] = [:]
    private let downloadToolsURL: URL?

    var busy: Bool { !activeOperationIDs.isEmpty }

    static let families: [ModelFamily] = [
        ModelFamily(id: "qwen3.8-27b", title: "Qwen3.8-27B",
                    iconAssetName: "ModelQwen", sources: [
            ModelSource(format: .gguf, repository: "unsloth/Qwen3.8-27B-GGUF",
                        quantizations: ["UD-IQ1_M", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q4_K_M",
                                        "UD-Q5_K_M", "UD-Q6_K", "UD-Q8_K_XL", "Q8_0"],
                        recommendedQuantization: "UD-Q4_K_M"),
            ModelSource(format: .mlx, repository: "mlx-community/Qwen3.8-27B",
                        quantizations: ["4bit", "8bit", "mxfp4", "nvfp4"],
                        recommendedQuantization: "4bit")
        ]),
        ModelFamily(id: "qwen3.6-35b-a3b", title: "Qwen3.6-35B-A3B",
                    iconAssetName: "ModelQwen", sources: [
            ModelSource(format: .gguf, repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
                        quantizations: ["UD-IQ1_M", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q4_K_M",
                                        "UD-Q5_K_M", "UD-Q6_K", "UD-Q8_K_XL", "Q8_0"],
                        recommendedQuantization: "UD-Q4_K_M"),
            ModelSource(format: .mlx, repository: "mlx-community/Qwen3.6-35B-A3B",
                        quantizations: ["4bit", "5bit", "6bit", "8bit", "mxfp4", "nvfp4"],
                        recommendedQuantization: "4bit")
        ]),
        ModelFamily(id: "ornith-1.5-35b-a3b", title: "Ornith-1.5-35B-A3B",
                    iconAssetName: "ModelOrnithAI", sources: [
            ModelSource(format: .gguf, repository: "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
                        quantizations: ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                        recommendedQuantization: "Q4_K_M"),
            ModelSource(format: .mlx, repository: "ornith-ai/Ornith-1.5-35B-A3B-MLX",
                        quantizations: ["4bit", "6bit", "8bit"],
                        recommendedQuantization: "4bit", languageOnly: true)
        ]),
        ModelFamily(id: "ornith-1.5-122b", title: "Ornith-1.5-122B",
                    iconAssetName: "ModelOrnithAI", sources: [
            ModelSource(format: .gguf, repository: "unsloth/Ornith-1.5-122B-GGUF",
                        quantizations: ["UD-IQ1_M", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q2_K_XL",
                                        "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"],
                        recommendedQuantization: "UD-Q4_K_XL"),
            ModelSource(format: .mlx, repository: "ornith-ai/Ornith-1.5-122B-MLX",
                        quantizations: ["4bit", "8bit"], recommendedQuantization: "4bit")
        ]),
        ModelFamily(id: "qwen3.8-flash-next", title: "Qwen3.8-Flash-Next",
                    iconAssetName: "ModelQwen", sources: [
            ModelSource(format: .gguf, repository: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                        quantizations: ["IQ2_XS", "IQ3_S", "IQ3_XXS", "Q2_0"],
                        recommendedQuantization: "Q2_0")
        ]),
        ModelFamily(id: "flint", title: "Flint",
                    iconAssetName: "ModelFlint", sources: [
            ModelSource(format: .gguf, repository: "unsloth/flint-GGUF",
                        quantizations: ["UD-IQ1_M", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q2_K_XL",
                                        "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"],
                        recommendedQuantization: "UD-Q4_K_XL")
        ]),
        ModelFamily(id: "ternary-bonsai-2-27b", title: "Ternary Bonsai 2 27B",
                    iconAssetName: "ModelPrismML", sources: [
            ModelSource(format: .gguf, repository: "prism-ml/Ternary-Bonsai-2-27B-gguf",
                        quantizations: ["PQ2_0"], recommendedQuantization: "PQ2_0")
        ])
    ]

    let modelsRoot: URL
    @Published private(set) var hubCacheRoot: URL

    init(modelsRoot: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Splash/models"),
         hubCacheRoot: URL? = nil, downloadToolsURL: URL? = nil) {
        self.modelsRoot = modelsRoot
        self.hubCacheRoot = hubCacheRoot ?? Self.hubCacheURL()
        self.downloadToolsURL = downloadToolsURL
    }

    private var toolsURL: URL? {
        downloadToolsURL ?? Bundle.main.url(forResource: "model_tools", withExtension: "py")
    }

    static func hubCacheURL(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        let defaultHome = "\(environment["XDG_CACHE_HOME"] ?? "~/.cache")/huggingface"
        let home = environment["HF_HOME"] ?? defaultHome
        let path = environment["HF_HUB_CACHE"] ?? environment["HUGGINGFACE_HUB_CACHE"] ?? "\(home)/hub"
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
    }

    func setHubCacheRoot(_ url: URL) {
        hubCacheRoot = url.standardizedFileURL
        downloadedBytesByID = [:]
        estimatedSizes = [:]
    }

    nonisolated static func cachedBytes(repository: String, in cache: URL) -> Int64 {
        let blobs = cache.appendingPathComponent("models--\(repository.replacingOccurrences(of: "/", with: "--"))/blobs")
        let files = (try? FileManager.default.contentsOfDirectory(
            at: blobs, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
        )) ?? []
        return files.reduce(0) { sum, file in
            guard let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true else { return sum }
            return sum + Int64(values.fileSize ?? 0)
        }
    }

    nonisolated private static func cachedBytesInBackground(repository: String, in cache: URL) async -> Int64 {
        await Task.detached(priority: .utility) {
            cachedBytes(repository: repository, in: cache)
        }.value
    }

    func refresh() {
        models = (try? installed()) ?? []
    }

    func entries() -> [ModelEntry] {
        let installedByID = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0) })
        return Self.families.flatMap { family in
            family.sources.flatMap { source in
                source.quantizations.map { quantization in
                    let id = source.modelID(for: quantization)
                    return ModelEntry(id: id, title: family.title,
                                      iconAssetName: family.iconAssetName,
                                      format: "\(source.format.rawValue) · \(quantization)",
                                      quantization: quantization,
                                      languageOnly: source.languageOnly,
                                      assembly: installedByID[id]?.assembly,
                                      sizeBytes: installedByID[id]?.sizeBytes)
                }
            }
        }
    }

    func isSupportedModelID(_ id: String) -> Bool {
        source(for: id) != nil
    }

    func isLanguageOnly(_ id: String) -> Bool {
        source(for: id)?.languageOnly == true
    }

    func source(for id: String) -> ModelSource? {
        Self.families.lazy.flatMap(\.sources).first { $0.variant(in: id) != nil }
    }

    func installed() throws -> [InstalledModel] {
        guard FileManager.default.fileExists(atPath: modelsRoot.path) else { return [] }
        let owners = try FileManager.default.contentsOfDirectory(at: modelsRoot,
            includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        var found: [InstalledModel] = []
        for owner in owners where owner.lastPathComponent != ".selections" {
            guard let repos = try? FileManager.default.contentsOfDirectory(at: owner,
                includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
            for link in repos {
                let id = "\(owner.lastPathComponent)/\(link.lastPathComponent)"
                guard validModelID(id), !isLanguageOnly(id),
                      let model = installedModel(at: link, id: id) else { continue }
                found.append(model)
            }
        }
        for source in Self.families.lazy.flatMap(\.sources) where source.languageOnly {
            for variant in source.quantizations {
                let id = source.modelID(for: variant)
                guard let link = languageOnlySelectionLink(for: id),
                      let model = installedModel(at: link, id: id) else { continue }
                found.append(model)
            }
        }
        return found.sorted { $0.id < $1.id }
    }

    private func installedModel(at link: URL, id: String) -> InstalledModel? {
        guard (try? FileManager.default.destinationOfSymbolicLink(atPath: link.path)) != nil else {
            return nil
        }
        let assembly = link.resolvingSymlinksInPath()
        let manifestURL = assembly.appendingPathComponent("model.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        let size = (try? Data(contentsOf: manifestURL))
            .flatMap { try? JSONDecoder().decode(ModelManifest.self, from: $0) }?.sizeBytes
        return InstalledModel(id: id, assembly: assembly, sizeBytes: size)
    }

    private func languageOnlySelectionLink(for id: String) -> URL? {
        guard let json = try? JSONSerialization.data(
            withJSONObject: [id, NSNull(), true, NSNull()], options: [.withoutEscapingSlashes]
        ) else { return nil }
        let digest = SHA256.hash(data: json).map { String(format: "%02x", $0) }.joined()
        return modelsRoot.appendingPathComponent(".selections").appendingPathComponent(digest)
    }

    func prepare(id: String, runtimeRoot: URL) async throws {
        guard let source = source(for: id) else {
            throw ModelCatalogError.unsupportedModel
        }
        let options = source.languageOnly ? ["--language-only"] : []
        try await run(modelID: id, runtimeRoot: runtimeRoot,
                      script: runtimeRoot.appendingPathComponent("install/models.py"),
                      arguments: ["--models", modelsRoot.path, "--model", id] + options + ["prepare"],
                      downloadRepository: source.format == .mlx ? id : source.repository)
        refresh()
    }

    func planDownload(id: String, runtimeRoot: URL) async throws -> ModelDownloadPlan {
        guard let source = source(for: id), let helper = toolsURL else {
            throw ModelCatalogError.unsupportedModel
        }
        let options = source.languageOnly ? ["--language-only"] : []
        let output = try await helperOutput(runtimeRoot: runtimeRoot, script: helper,
            arguments: ["--models", modelsRoot.path, "--model", id] + options + ["plan"])
        guard let line = output.split(whereSeparator: \.isNewline).last,
              let data = String(line).data(using: .utf8) else {
            throw ModelCatalogError.commandFailed(output)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ModelDownloadPlan.self, from: data)
    }

    func fetchModel(id: String, plan: ModelDownloadPlan, runtimeRoot: URL) async throws {
        guard let source = source(for: id), let helper = toolsURL else {
            throw ModelCatalogError.unsupportedModel
        }
        let options = source.languageOnly ? ["--language-only"] : []
        try await run(modelID: id, runtimeRoot: runtimeRoot, script: helper,
                      arguments: ["--models", modelsRoot.path, "--model", id,
                                  "--revision", plan.modelRevision] + options + ["fetch-model"],
                      downloadRepository: source.format == .mlx ? id : source.repository,
                      initialPhase: .model,
                      tracksPhaseFromLog: false)
    }

    func fetchDraft(_ plan: ModelDownloadPlan, runtimeRoot: URL) async throws {
        guard let helper = toolsURL else { throw ModelCatalogError.unsupportedModel }
        if let size = plan.draftSizeBytes { estimatedSizes[plan.draftID] = size }
        try await run(modelID: plan.draftID, runtimeRoot: runtimeRoot, script: helper,
                      arguments: ["--repository", plan.draftRepository,
                                  "--revision", plan.draftRevision, "fetch-draft"],
                      downloadRepository: plan.draftRepository, initialPhase: .draft,
                      tracksPhaseFromLog: false)
    }

    func assemble(id: String, plan: ModelDownloadPlan, runtimeRoot: URL) async throws {
        guard let source = source(for: id), let helper = toolsURL else {
            throw ModelCatalogError.unsupportedModel
        }
        let options = source.languageOnly ? ["--language-only"] : []
        try await run(modelID: id, runtimeRoot: runtimeRoot, script: helper,
                      arguments: ["--models", modelsRoot.path, "--model", id,
                                  "--revision", plan.modelRevision,
                                  "--repository", plan.draftRepository,
                                  "--draft-revision", plan.draftRevision] + options + ["assemble"],
                      initialPhase: .installing, tracksPhaseFromLog: false)
        refresh()
    }

    private func helperOutput(runtimeRoot: URL, script: URL, arguments: [String]) async throws -> String {
        let process = Process()
        process.executableURL = runtimeRoot.appendingPathComponent("python/bin/python3")
        process.arguments = ["-u", script.path] + arguments
        process.environment = ProcessInfo.processInfo.environment.merging(
            ["PYTHONPATH": runtimeRoot.path, "PYTHONDONTWRITEBYTECODE": "1",
             "HF_HUB_CACHE": hubCacheRoot.path]
        ) { _, new in new }
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try process.run()
            let data = await withCheckedContinuation { continuation in
                DispatchQueue.global().async {
                    let output = pipe.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    continuation.resume(returning: output)
                }
            }
            try Task.checkCancellation()
            let output = String(decoding: data, as: UTF8.self)
            guard process.terminationStatus == 0 else {
                throw ModelCatalogError.commandFailed(output)
            }
            return output
        } onCancel: {
            Task { @MainActor in
                if process.isRunning { process.terminate() }
            }
        }
    }

    func estimateSizes(for source: ModelSource, runtimeRoot: URL) async {
        guard source.quantizations.contains(where: { estimatedSizes[source.modelID(for: $0)] == nil }),
              !estimatingSourceIDs.contains(source.id),
              let helper = toolsURL else { return }
        estimatingSourceIDs.insert(source.id)
        defer { estimatingSourceIDs.remove(source.id) }

        let process = Process()
        process.executableURL = runtimeRoot.appendingPathComponent("python/bin/python3")
        process.arguments = ["-u", helper.path, "--repository", source.repository,
                             "--format", source.format.rawValue,
                             "--variants", source.quantizations.joined(separator: ","), "sizes"]
        process.environment = ProcessInfo.processInfo.environment.merging(
            ["PYTHONPATH": runtimeRoot.path, "PYTHONDONTWRITEBYTECODE": "1",
             "HF_HUB_CACHE": hubCacheRoot.path]
        ) { _, new in new }
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try process.run()
                let data = await withCheckedContinuation { continuation in
                    DispatchQueue.global().async {
                        let output = pipe.fileHandleForReading.readDataToEndOfFile()
                        process.waitUntilExit()
                        continuation.resume(returning: output)
                    }
                }
                try Task.checkCancellation()
                guard process.terminationStatus == 0 else { return }
                let sizes = try JSONDecoder().decode([String: Int64].self, from: data)
                for (variant, size) in sizes where source.quantizations.contains(variant) && size > 0 {
                    estimatedSizes[source.modelID(for: variant)] = size
                }
            } onCancel: {
                Task { @MainActor in
                    if process.isRunning { process.terminate() }
                }
            }
        } catch { return }
    }

    private func run(modelID: String, runtimeRoot: URL, script: URL, arguments: [String],
                     downloadRepository: String? = nil,
                     initialPhase: ModelDownloadPhase = .preparing,
                     tracksPhaseFromLog: Bool = true) async throws {
        guard !activeOperationIDs.contains(modelID) else {
            throw ModelCatalogError.commandFailed("A model operation is already running.")
        }
        activeOperationIDs.insert(modelID)
        operationLogs[modelID] = ""
        downloadedBytesByID[modelID] = 0
        downloadPhaseByID[modelID] = initialPhase
        pendingLogLines[modelID] = ""
        let operationID = UUID()
        var progressTask: Task<Void, Never>?
        defer {
            progressTask?.cancel()
            if activeOperations[modelID]?.id == operationID { activeOperations[modelID] = nil }
            activeOperationIDs.remove(modelID)
            downloadPhaseByID[modelID] = nil
            pendingLogLines[modelID] = nil
        }
        let cache = hubCacheRoot
        let startingBytes: Int64
        if let downloadRepository {
            startingBytes = await Self.cachedBytesInBackground(repository: downloadRepository, in: cache)
        } else {
            startingBytes = 0
        }
        let process = Process()
        process.executableURL = runtimeRoot.appendingPathComponent("python/bin/python3")
        process.arguments = ["-u", script.path] + arguments
        process.environment = ProcessInfo.processInfo.environment.merging(
            ["PYTHONPATH": runtimeRoot.path, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_CACHE": cache.path]
        ) { _, new in new }
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        progressTask = downloadRepository.map { repository in
            Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    let current = await Self.cachedBytesInBackground(repository: repository, in: cache)
                    guard let self, !Task.isCancelled else { return }
                    let progress = max(
                        self.downloadedBytesByID[modelID] ?? 0, current - startingBytes)
                    if self.downloadedBytesByID[modelID] != progress {
                        self.downloadedBytesByID[modelID] = progress
                    }
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
        }
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try process.run()
            activeOperations[modelID] = (operationID, process)
            let output = AsyncStream<String> { continuation in
                DispatchQueue.global().async {
                    while true {
                        let data = pipe.fileHandleForReading.availableData
                        guard !data.isEmpty else { break }
                        continuation.yield(String(decoding: data, as: UTF8.self))
                    }
                    continuation.finish()
                }
            }
            for await chunk in output {
                appendLog(chunk, for: modelID, targetRepository: downloadRepository,
                          tracksPhase: tracksPhaseFromLog)
            }
            await withCheckedContinuation { continuation in
                DispatchQueue.global().async {
                    process.waitUntilExit()
                    continuation.resume()
                }
            }
            if let downloadRepository {
                let current = await Self.cachedBytesInBackground(repository: downloadRepository, in: cache)
                let progress = max(
                    downloadedBytesByID[modelID] ?? 0,
                    current - startingBytes)
                if downloadedBytesByID[modelID] != progress {
                    downloadedBytesByID[modelID] = progress
                }
            }
            try Task.checkCancellation()
            guard process.terminationStatus == 0 else {
                throw ModelCatalogError.commandFailed(operationLogs[modelID] ?? "")
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                self?.terminateOperation(modelID: modelID, id: operationID)
            }
        }
    }

    private func terminateOperation(modelID: String, id: UUID) {
        guard let operation = activeOperations[modelID], operation.id == id,
              operation.process.isRunning else { return }
        operation.process.terminate()
    }

    func terminateAllOperations() {
        for operation in activeOperations.values where operation.process.isRunning {
            operation.process.terminate()
        }
    }

    private func appendLog(_ text: String, for modelID: String, targetRepository: String?,
                           tracksPhase: Bool) {
        operationLogs[modelID] = String(((operationLogs[modelID] ?? "") + text).suffix(24_000))
        guard tracksPhase else { return }
        let lines = ((pendingLogLines[modelID] ?? "") + text)
            .split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        pendingLogLines[modelID] = String(lines.last ?? "")
        for line in lines.dropLast() {
            let message = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.hasPrefix("Fetching "), let targetRepository {
                downloadPhaseByID[modelID] = message.contains(" from \(targetRepository)@") ? .model : .draft
            } else if message.hasPrefix("Installing ") {
                downloadPhaseByID[modelID] = .model
            }
        }
    }

    private func validModelID(_ id: String) -> Bool {
        let parts = id.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count <= 2 else { return false }
        let repo = String(parts[0])
        guard repo.range(of: #"^[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?$"#,
                         options: .regularExpression) != nil,
              !repo.contains("--"), !repo.contains(".."), !repo.hasSuffix(".git") else { return false }
        guard parts.count == 2 else { return true }
        let variant = String(parts[1])
        return variant.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"#,
                             options: .regularExpression) != nil && !variant.contains("..")
    }
}

enum ModelCatalogError: LocalizedError {
    case unsupportedModel
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedModel: return "This model is not in the supported list."
        case .commandFailed(let output): return output.isEmpty ? "Model operation failed." : output
        }
    }
}
