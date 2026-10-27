import AppKit
import Foundation

struct DraftDownload: Identifiable {
    let id: String
    let repository: String
}

@MainActor
final class DesktopState: ObservableObject {
    static let shared = DesktopState()

    let settings: AppSettings
    let service: ServiceController
    let catalog: ModelCatalog
    let runtime: RuntimeInstaller

    @Published var managerSection = ManagerSection.models
    @Published private(set) var chatRequestedModelID: String?
    @Published private(set) var chatStartPending = false
    @Published var modelID = ""
    @Published var portText = "8000"
    @Published var errorMessage: String?
    @Published var operationInProgress = false
    @Published private(set) var startingModelID: String?
    @Published private(set) var downloadingModelIDs: Set<String> = []
    @Published private(set) var cancellingModelIDs: Set<String> = []
    @Published private(set) var downloadingDrafts: [String: DraftDownload] = [:]
    @Published private(set) var cancellingDraftIDs: Set<String> = []
    @Published private(set) var waitingForDraftModelIDs: Set<String> = []

    private var initialized = false
    private var modelDownloadTasks: [String: Task<Void, Never>] = [:]
    private var draftDownloadTasks: [String: Task<Void, Never>] = [:]
    private var modelRequests: [String: (plan: ModelDownloadPlan, runtimeRoot: URL)] = [:]
    private var completedDraftIDs: Set<String> = []

    init(settings: AppSettings? = nil, service: ServiceController? = nil,
         catalog: ModelCatalog? = nil, runtime: RuntimeInstaller? = nil) {
        self.settings = settings ?? AppSettings()
        self.service = service ?? ServiceController()
        self.catalog = catalog ?? ModelCatalog(hubCacheRoot: self.settings.modelCacheURL)
        self.runtime = runtime ?? RuntimeInstaller()
        if let modelCacheURL = self.settings.modelCacheURL {
            self.catalog.setHubCacheRoot(modelCacheURL)
        }
        let savedID = self.settings.selectedModelID ?? ""
        modelID = self.catalog.isSupportedModelID(savedID) ? savedID : ""
        portText = String(self.settings.port)
    }

    var serviceIsBusy: Bool {
        operationInProgress || service.status == .starting || service.status == .stopping
    }

    var serviceIsReady: Bool {
        if case .ready = service.status { return true }
        return false
    }

    var selectedModelID: String? {
        let value = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    var canStart: Bool { selectedModelID.map(canRun) ?? false }
    var canOpenChat: Bool { serviceIsReady && service.runningPort != nil }
    var chatURL: URL? {
        guard canOpenChat, let port = service.runningPort else { return nil }
        return URL(string: "http://127.0.0.1:\(port)/")
    }

    func isStartingModel(_ id: String) -> Bool {
        startingModelID == id || (service.status == .starting && service.runningModelID == id)
    }

    func canRun(_ id: String) -> Bool {
        catalog.isSupportedModelID(id) && catalog.models.contains { $0.id == id } && !serviceIsBusy &&
            !downloadingModelIDs.contains(id) && !catalog.activeOperationIDs.contains(id) &&
            service.runningModelID != id
    }

    func canLaunchChat(for id: String) -> Bool {
        guard catalog.isSupportedModelID(id), catalog.models.contains(where: { $0.id == id }) else {
            return false
        }
        if chatStartPending { return chatRequestedModelID == id }
        return isStartingModel(id) || (service.runningModelID == id && canOpenChat) || canRun(id)
    }

    @discardableResult
    func beginChat(for id: String) -> Bool {
        guard canLaunchChat(for: id) else { return false }
        if chatStartPending { return true }
        chatRequestedModelID = id
        errorMessage = nil
        if !isStartingModel(id) && !(service.runningModelID == id && canOpenChat) {
            chatStartPending = true
            Task {
                await startModel(id)
                chatStartPending = false
            }
        }
        return true
    }

    func initialize() async {
        guard !initialized else { return }
        initialized = true
        catalog.refresh()
        do { _ = try await runtime.resolve() }
        catch { errorMessage = error.localizedDescription }
    }

    func selectModel(_ id: String) {
        guard id.isEmpty || catalog.isSupportedModelID(id) else { return }
        modelID = id
        settings.selectedModelID = id
    }

    func startSelectedModel() async {
        guard let selectedModelID, canRun(selectedModelID) else { return }
        errorMessage = nil
        operationInProgress = true
        startingModelID = selectedModelID
        defer {
            startingModelID = nil
            operationInProgress = false
        }
        do {
            let location = try await runtime.resolve()
            if service.runningPort != nil, !(await service.stop()) {
                errorMessage = "Could not verify active requests. Use Force Stop before switching models."
                return
            }
            settings.selectedModelID = selectedModelID
            try service.start(command: location.command, modelID: selectedModelID,
                                  port: settings.port, idleRelease: settings.idleRelease,
                                  modelCacheURL: catalog.hubCacheRoot,
                                  languageOnly: catalog.isLanguageOnly(selectedModelID))
        } catch { errorMessage = error.localizedDescription }
    }

    func startModel(_ id: String) async {
        guard canRun(id) else { return }
        selectModel(id)
        await startSelectedModel()
    }

    func stop(force: Bool = false) async {
        if force {
            await service.stop(force: true)
            return
        }
        guard !operationInProgress else { return }
        operationInProgress = true
        defer { operationInProgress = false }
        if !(await service.stop(force: force)) {
            errorMessage = "Could not verify active requests. Use Force Stop to interrupt the service."
        }
    }

    func beginModelDownload(_ id: String) {
        guard catalog.isSupportedModelID(id), !downloadingModelIDs.contains(id),
              !catalog.models.contains(where: { $0.id == id }) else { return }
        downloadingModelIDs.insert(id)
        modelDownloadTasks[id] = Task {
            errorMessage = nil
            do {
                let location = try await runtime.resolve()
                try Task.checkCancellation()
                let plan = try await catalog.planDownload(id: id, runtimeRoot: location.root)
                try Task.checkCancellation()
                modelRequests[id] = (plan, location.root)
                if plan.draftNeeded { beginDraftDownload(plan, runtimeRoot: location.root) }
                try await catalog.fetchModel(id: id, plan: plan, runtimeRoot: location.root)
                try Task.checkCancellation()
                modelDownloadTasks[id] = nil
                if draftIsReady(plan) {
                    beginAssembly(id)
                } else {
                    waitingForDraftModelIDs.insert(id)
                }
            } catch is CancellationError {
                finishModelDownload(id)
            } catch {
                if !Task.isCancelled { errorMessage = error.localizedDescription }
                finishModelDownload(id)
            }
        }
    }

    func cancelModelDownload(_ id: String) {
        guard downloadingModelIDs.contains(id), !cancellingModelIDs.contains(id) else { return }
        if waitingForDraftModelIDs.contains(id) {
            finishModelDownload(id)
            return
        }
        cancellingModelIDs.insert(id)
        modelDownloadTasks[id]?.cancel()
    }

    func cancelDraftDownload(_ id: String) {
        guard downloadingDrafts[id] != nil, !cancellingDraftIDs.contains(id) else { return }
        cancellingDraftIDs.insert(id)
        draftDownloadTasks[id]?.cancel()
    }

    func retryDraft(for modelID: String) {
        guard waitingForDraftModelIDs.contains(modelID), let request = modelRequests[modelID] else { return }
        beginDraftDownload(request.plan, runtimeRoot: request.runtimeRoot)
    }

    func isDraftDownloading(for modelID: String) -> Bool {
        guard let plan = modelRequests[modelID]?.plan else { return false }
        return downloadingDrafts[plan.draftID] != nil
    }

    func cancelAllModelDownloads() {
        for id in Array(downloadingModelIDs) { cancelModelDownload(id) }
        for id in Array(downloadingDrafts.keys) { cancelDraftDownload(id) }
        catalog.terminateAllOperations()
    }

    private func beginDraftDownload(_ plan: ModelDownloadPlan, runtimeRoot: URL) {
        let id = plan.draftID
        guard !completedDraftIDs.contains(id), draftDownloadTasks[id] == nil else { return }
        downloadingDrafts[id] = DraftDownload(id: id, repository: plan.draftRepository)
        draftDownloadTasks[id] = Task {
            do {
                try await catalog.fetchDraft(plan, runtimeRoot: runtimeRoot)
                try Task.checkCancellation()
                completedDraftIDs.insert(id)
                for modelID in Array(waitingForDraftModelIDs)
                    where modelRequests[modelID]?.plan.draftID == id {
                    beginAssembly(modelID)
                }
            } catch is CancellationError {
            } catch {
                if !Task.isCancelled { errorMessage = error.localizedDescription }
            }
            draftDownloadTasks[id] = nil
            downloadingDrafts[id] = nil
            cancellingDraftIDs.remove(id)
        }
    }

    private func draftIsReady(_ plan: ModelDownloadPlan) -> Bool {
        !plan.draftNeeded || completedDraftIDs.contains(plan.draftID)
    }

    private func beginAssembly(_ id: String) {
        guard downloadingModelIDs.contains(id), let request = modelRequests[id] else { return }
        waitingForDraftModelIDs.remove(id)
        modelDownloadTasks[id] = Task {
            do {
                try await catalog.assemble(id: id, plan: request.plan,
                                           runtimeRoot: request.runtimeRoot)
            } catch is CancellationError {
            } catch {
                if !Task.isCancelled { errorMessage = error.localizedDescription }
            }
            finishModelDownload(id)
        }
    }

    private func finishModelDownload(_ id: String) {
        modelDownloadTasks[id] = nil
        modelRequests[id] = nil
        downloadingModelIDs.remove(id)
        cancellingModelIDs.remove(id)
        waitingForDraftModelIDs.remove(id)
    }

    @discardableResult
    func savePort() -> Bool {
        guard let candidate = Int(portText), (1...65535).contains(candidate) else {
            errorMessage = "Enter a port from 1 to 65535."
            return false
        }
        if candidate != settings.port && candidate != service.runningPort &&
            !PortAvailability.isFree(candidate) {
            errorMessage = "Port \(candidate) is in use."
            return false
        }
        if candidate != settings.port {
            settings.port = candidate
            errorMessage = service.runningPort == nil ? nil : "Restart the service to use port \(candidate). Browser history remains on the old port."
        } else { errorMessage = nil }
        return true
    }

    func copyAPIAddress() {
        guard let port = service.runningPort else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://127.0.0.1:\(port)/v1", forType: .string)
    }

    @discardableResult
    func changeModelCacheDirectory(to directory: URL?) -> Bool {
        guard !catalog.busy && downloadingModelIDs.isEmpty && downloadingDrafts.isEmpty else {
            errorMessage = "Wait for the current model task to finish before changing the cache directory."
            return false
        }
        let target = (directory ?? ModelCatalog.hubCacheURL()).standardizedFileURL
        guard target.isFileURL else {
            errorMessage = "Choose a local model cache directory."
            return false
        }
        do {
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            guard FileManager.default.isWritableFile(atPath: target.path) else {
                errorMessage = "The selected model cache directory is not writable."
                return false
            }
            catalog.setHubCacheRoot(target)
            completedDraftIDs = []
            settings.modelCacheURL = directory
            errorMessage = nil
            return true
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }
}
