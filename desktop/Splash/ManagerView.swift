import AppKit
import ServiceManagement
import SwiftUI
import WebKit

enum ManagerSection: String, CaseIterable {
    case models = "Models"
    case service = "Service"
    case settings = "Settings"

    var symbol: String {
        switch self {
        case .service: return "server.rack"
        case .models: return "square.stack.3d.up"
        case .settings: return "gearshape"
        }
    }
}

private enum ModelsTab: String, CaseIterable {
    case all = "All"
    case installed = "Installed"
    case downloads = "Downloads"
}

struct ManagerView: View {
    @ObservedObject var state: DesktopState
    @ObservedObject private var service: ServiceController
    @ObservedObject private var catalog: ModelCatalog
    @ObservedObject private var runtime: RuntimeInstaller
    @ObservedObject private var settings: AppSettings
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var modelsTab = ModelsTab.installed
    @State private var didChooseInitialModelsTab = false
    @State private var hoveredModelID: String?
    @State private var hoveredDownloadID: String?
    @State private var downloadSourceID: String?
    @State private var showingPortEditor = false
    @State private var portError: String?
    @State private var loginItemStatus: SMAppService.Status = .notRegistered
    @State private var launchAtLoginError: String?
    @FocusState private var portFieldFocused: Bool

    init(state: DesktopState) {
        self.state = state
        service = state.service
        catalog = state.catalog
        runtime = state.runtime
        settings = state.settings
    }

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            VStack(spacing: 0) {
                List(selection: $state.managerSection) {
                    ForEach(ManagerSection.allCases, id: \.self) { item in
                        Label(item.rawValue, systemImage: item.symbol).tag(item)
                    }
                }
                .listStyle(.sidebar)
                Link(destination: URL(string: "https://github.com/incoai/splash")!) {
                    HStack(spacing: 6) {
                        Image("SplashMark")
                            .renderingMode(.template)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .foregroundStyle(Color.splashBrand)
                            .frame(width: 16, height: 16)
                        Text("incoai/splash").lineLimit(1)
                        Image(systemName: "arrow.up.right")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 14)
                    .frame(maxWidth: .infinity, minHeight: 38, alignment: .leading)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(.leading, 8)
                .padding(.bottom, 8)
                .help("Open incoai/splash on GitHub")
            }
            .frame(minWidth: 170)
        } detail: {
            Group {
                switch state.managerSection {
                case .service: serviceView
                case .models: modelsView
                case .settings: settingsView
                }
            }
            .frame(minWidth: 620, minHeight: 520)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if let message = state.errorMessage {
                    Divider()
                    HStack(spacing: 12) {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button {
                            state.errorMessage = nil
                        } label: {
                            Image(systemName: "xmark")
                                .frame(width: 26, height: 26)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help("Dismiss error")
                        .accessibilityLabel("Dismiss error")
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                }
            }
            .navigationTitle(state.managerSection.rawValue)
        }
        .navigationSplitViewStyle(.balanced)
        .toolbar {
            if state.managerSection == .models {
                ToolbarItem(placement: .primaryAction) {
                    Picker("Models", selection: $modelsTab) {
                        Text("All").tag(ModelsTab.all)
                        Text("Installed").tag(ModelsTab.installed)
                        if downloadCount > 0 {
                            Text("Downloads (\(downloadCount))").tag(ModelsTab.downloads)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize(horizontal: true, vertical: false)
                    .id(downloadCount)
                }
            }
            if state.managerSection == .service {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button("Chat", systemImage: "bubble.left") {
                        if let modelID = serviceChatModelID { openChat(for: modelID) }
                    }
                    .labelStyle(.iconOnly)
                    .help("Chat")
                    .disabled(serviceChatModelID.map { !state.canLaunchChat(for: $0) } ?? true)
                    if service.runningPort != nil {
                        if service.status == .stopping {
                            Button("Force Stop", systemImage: "xmark.octagon") {
                                Task { await state.stop(force: true) }
                            }
                            .labelStyle(.iconOnly)
                            .help("Force stop; active requests may be interrupted")
                        } else {
                            Button("Stop", systemImage: "stop") { Task { await state.stop() } }
                                .labelStyle(.iconOnly)
                                .help("Stop")
                                .disabled(state.operationInProgress)
                        }
                    }
                }
            }
        }
        .task { await state.initialize() }
        .sheet(isPresented: $showingPortEditor) { portEditor }
        .onChange(of: downloadCount) { _, count in
            if count == 0 && modelsTab == .downloads {
                modelsTab = installedEntries.isEmpty ? .all : .installed
            }
        }
    }

    private var installedEntries: [ModelEntry] {
        catalog.entries().filter(\.isInstalled)
    }

    private var downloadCount: Int {
        state.downloadingModelIDs.count + state.downloadingDrafts.count
    }

    private var downloadingDrafts: [DraftDownload] {
        state.downloadingDrafts.values.sorted { $0.repository < $1.repository }
    }

    private var downloadingEntries: [ModelEntry] {
        catalog.entries().filter { state.downloadingModelIDs.contains($0.id) }
    }

    private func estimatedDownloadProgress(for id: String) -> Double? {
        guard let total = catalog.estimatedSizes[id], total > 0 else { return nil }
        return min(max(Double(catalog.downloadedBytesByID[id] ?? 0) / Double(total), 0), 0.99)
    }

    private func downloadPercent(for id: String) -> String {
        guard let progress = estimatedDownloadProgress(for: id) else { return "--%" }
        return "\(Int(progress * 100))%"
    }

    private func missingEntries(for source: ModelSource) -> [ModelEntry] {
        entries(for: source).filter { !$0.isInstalled }
    }

    private func entries(for source: ModelSource) -> [ModelEntry] {
        let entries = Dictionary(uniqueKeysWithValues: catalog.entries().map { ($0.id, $0) })
        return source.quantizations.compactMap { entries[source.modelID(for: $0)] }
    }

    private var modelsView: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if modelsTab == .installed {
                        if installedEntries.isEmpty {
                            ContentUnavailableView("No supported models installed", systemImage: "square.stack.3d.up")
                                .frame(maxWidth: .infinity, minHeight: 260)
                        } else {
                            LazyVStack(spacing: 0) {
                                ForEach(Array(installedEntries.enumerated()), id: \.element.id) { item in
                                    if item.offset > 0 {
                                        modelDivider(above: installedEntries[item.offset - 1].id,
                                                     below: item.element.id)
                                    }
                                    let entry = item.element
                                    modelRow(entry)
                                }
                            }
                        }
                    } else if modelsTab == .all {
                        let sources = ModelCatalog.families.flatMap { family in
                            family.sources.map { (source: $0, iconAssetName: family.iconAssetName) }
                        }
                        LazyVStack(spacing: 0) {
                            ForEach(Array(sources.enumerated()), id: \.element.source.id) { item in
                                if item.offset > 0 {
                                    modelDivider(above: sources[item.offset - 1].source.id,
                                                 below: item.element.source.id)
                                }
                                catalogSourceRow(item.element.source,
                                                 iconAssetName: item.element.iconAssetName)
                            }
                        }
                    } else if downloadCount > 0 {
                        LazyVStack(spacing: 0) {
                            if !downloadingDrafts.isEmpty {
                                Text("Drafts")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                ForEach(downloadingDrafts) { draft in
                                    draftDownloadingRow(draft)
                                }
                            }
                            if !downloadingEntries.isEmpty {
                                Text("Models")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                ForEach(Array(downloadingEntries.enumerated()), id: \.element.id) { item in
                                    if item.offset > 0 {
                                        modelDivider(above: downloadingEntries[item.offset - 1].id,
                                                     below: item.element.id)
                                    }
                                    downloadingRow(item.element)
                                }
                            }
                        }
                    } else {
                        ContentUnavailableView("No downloads in progress", systemImage: "arrow.down")
                            .frame(maxWidth: .infinity, minHeight: 260)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
            }
            .id(modelsTab)
        }
        .onAppear {
            catalog.refresh()
            guard !didChooseInitialModelsTab else { return }
            modelsTab = installedEntries.isEmpty ? .all : .installed
            didChooseInitialModelsTab = true
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            catalog.refresh()
        }
    }

    private func modelDivider(above: String, below: String) -> some View {
        Divider()
            .opacity(hoveredModelID == above || hoveredModelID == below ? 0 : 1)
    }

    private func downloadingRow(_ entry: ModelEntry) -> some View {
        let phase = catalog.downloadPhaseByID[entry.id] ?? .preparing
        let waitingForDraft = state.waitingForDraftModelIDs.contains(entry.id)
        return HStack(spacing: 12) {
            ModelIcon(assetName: entry.iconAssetName)
            VStack(alignment: .leading, spacing: 6) {
                Text(entry.id)
                    .font(.subheadline.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                downloadDetails(id: entry.id,
                                status: waitingForDraft ? "Waiting for draft" :
                                    downloadStatus(for: entry.id, phase: phase),
                                showProgress: phase == .model || waitingForDraft,
                                showBytes: phase == .model,
                                complete: waitingForDraft,
                                retryModelID: waitingForDraft && !state.isDraftDownloading(for: entry.id) ? entry.id : nil)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button { state.cancelModelDownload(entry.id) } label: {
                ModelActionLabel(systemName: "xmark")
            }
            .buttonStyle(.plain)
            .help(state.cancellingModelIDs.contains(entry.id) ? "Cancelling download" : "Cancel download")
            .accessibilityLabel("Cancel download")
            .disabled(state.cancellingModelIDs.contains(entry.id))
        }
        .padding(.leading, 12)
        .padding(.vertical, 8)
        .background(hoveredModelID == entry.id ? Color.primary.opacity(0.06) : .clear,
                    in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onHover { hovering in
            if hovering {
                hoveredModelID = entry.id
            } else if hoveredModelID == entry.id {
                hoveredModelID = nil
            }
        }
        .task(id: entry.id) {
            if let source = catalog.source(for: entry.id) {
                await estimateSizes(for: source)
            }
        }
    }

    private func draftDownloadingRow(_ draft: DraftDownload) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 18))
                .foregroundStyle(.secondary)
                .frame(width: 34, height: 34)
            VStack(alignment: .leading, spacing: 6) {
                Text(draft.repository)
                    .font(.subheadline.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                downloadDetails(id: draft.id,
                                status: state.cancellingDraftIDs.contains(draft.id) ? "Cancelling" : "Downloading draft",
                                showProgress: true, showBytes: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button { state.cancelDraftDownload(draft.id) } label: {
                ModelActionLabel(systemName: "xmark")
            }
            .buttonStyle(.plain)
            .help("Cancel draft download")
            .accessibilityLabel("Cancel draft download")
            .disabled(state.cancellingDraftIDs.contains(draft.id))
        }
        .padding(.leading, 12)
        .padding(.vertical, 8)
        .background(hoveredModelID == draft.id ? Color.primary.opacity(0.06) : .clear,
                    in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onHover { hoveredModelID = $0 ? draft.id : nil }
    }

    private func downloadDetails(id: String, status: String, showProgress: Bool,
                                 showBytes: Bool, complete: Bool = false,
                                 retryModelID: String? = nil) -> some View {
        let progress = complete ? 1.0 : (showProgress ? estimatedDownloadProgress(for: id) : nil)
        return VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                if let progress {
                    ProgressView(value: progress)
                } else {
                    ProgressView().controlSize(.small)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                Text(complete ? "100%" : progress == nil ? "" : downloadPercent(for: id))
                    .monospacedDigit()
                    .frame(width: 42, alignment: .trailing)
            }
            .font(.caption)
            .help("Approximate transfer progress; cached files may be reused.")
            HStack(spacing: 4) {
                Text(status)
                if showBytes, let received = catalog.downloadedBytesByID[id], received > 0 {
                    Text("·")
                    Text(ByteCountFormatter.string(
                        fromByteCount: received,
                        countStyle: .file))
                    if let total = catalog.estimatedSizes[id] {
                        Text("/ \(ByteCountFormatter.string(fromByteCount: total, countStyle: .file))")
                    }
                }
                if let retryModelID {
                    Button("Retry draft", systemImage: "arrow.clockwise") {
                        state.retryDraft(for: retryModelID)
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)
                    .help("Retry draft download")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private func downloadStatus(for id: String, phase: ModelDownloadPhase) -> String {
        if state.cancellingModelIDs.contains(id) { return "Cancelling" }
        switch phase {
        case .preparing: return "Preparing download"
        case .draft: return "Downloading draft"
        case .model: return "Downloading model"
        case .installing: return "Installing model"
        }
    }

    private func modelRow(_ entry: ModelEntry) -> some View {
        HStack(spacing: 12) {
            ModelIcon(assetName: entry.iconAssetName)
            VStack(alignment: .leading, spacing: 5) {
                Text(entry.id)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: 6) {
                    if entry.languageOnly { modelBadge("Text only") }
                    if let size = entry.sizeBytes {
                        Text(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
                    }
                    if service.runningModelID == entry.id || state.isStartingModel(entry.id) {
                        Text("·")
                        Text(state.isStartingModel(entry.id) ? "Starting" : runningModelStatus)
                            .foregroundStyle(runningModelColor)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .help(entry.id)
            HStack(spacing: 0) {
                if isOperating(on: entry.id) ||
                    (service.runningModelID == entry.id && service.status == .stopping) {
                    ProgressView().controlSize(.small).frame(width: 36, height: 56)
                } else if service.runningModelID == entry.id {
                    Button { Task { await state.stop() } } label: {
                        ModelActionLabel(systemName: "stop")
                    }
                    .help("Stop")
                    .accessibilityLabel("Stop")
                    .disabled(state.operationInProgress)
                } else {
                    Button { Task { await state.startModel(entry.id) } } label: {
                        ModelActionLabel(systemName: "play")
                    }
                    .help("Run")
                    .accessibilityLabel("Run")
                    .disabled(!state.canRun(entry.id))
                }
                Button { openChat(for: entry.id) } label: {
                    ModelActionLabel(systemName: "bubble.left")
                }
                .help("Chat")
                .accessibilityLabel("Chat")
                .disabled(!state.canLaunchChat(for: entry.id))
                if let url = entry.huggingFaceURL {
                    Link(destination: url) {
                        ModelActionLabel(systemName: "arrow.up.right")
                    }
                    .help("Open in Hugging Face")
                    .accessibilityLabel("Open in Hugging Face")
                }
            }
            .buttonStyle(.plain)
        }
        .padding(.leading, 12)
        .frame(minHeight: 56)
        .background(hoveredModelID == entry.id ? Color.primary.opacity(0.06) : .clear,
                    in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onHover { hovering in
            if hovering {
                hoveredModelID = entry.id
            } else if hoveredModelID == entry.id {
                hoveredModelID = nil
            }
        }
    }

    @ViewBuilder private func catalogSourceRow(_ source: ModelSource,
                                               iconAssetName: String) -> some View {
        let missing = missingEntries(for: source)
        let linkedEntry = missing.first { $0.quantization == source.recommendedQuantization }
            ?? missing.first
            ?? catalog.entries().first { $0.id == source.modelID(for: source.recommendedQuantization) }
        let showQuantizations = Binding(
            get: { downloadSourceID == source.id },
            set: { if !$0 && downloadSourceID == source.id { downloadSourceID = nil } }
        )
        HStack(spacing: 0) {
            HStack(spacing: 12) {
                ModelIcon(assetName: iconAssetName)
                VStack(alignment: .leading, spacing: 4) {
                    Text(source.repository)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    HStack(spacing: 6) {
                        if source.languageOnly { modelBadge("Text only") }
                        Text("\(source.quantizations.count - missing.count)/\(source.quantizations.count) installed")
                        if missing.count == 1 {
                            Text("·")
                            Text(missing[0].quantization)
                        }
                        if let entry = missing.first, missing.count == 1,
                           let size = catalog.estimatedSizes[entry.id] {
                            Text("·")
                            Text("Approx. \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.leading, 12)
            .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
            Button { downloadSourceID = source.id } label: {
                ModelActionLabel(systemName: "arrow.down")
            }
            .buttonStyle(.plain)
            .help("Choose quantization")
            .accessibilityLabel("Choose quantization for \(source.repository)")
            .popover(isPresented: showQuantizations) {
                quantizationPicker(source, iconAssetName: iconAssetName)
                    .task { await estimateSizes(for: source) }
            }
            if let url = linkedEntry?.huggingFaceURL {
                Link(destination: url) {
                    ModelActionLabel(systemName: "arrow.up.right")
                }
                .buttonStyle(.plain)
                .help("Open in Hugging Face")
                .accessibilityLabel("Open in Hugging Face")
            }
        }
        .frame(minHeight: 56)
        .background(hoveredModelID == source.id ? Color.primary.opacity(0.06) : .clear,
                    in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onHover { hoveredModelID = $0 ? source.id : nil }
        .task {
            if missing.count == 1 { await estimateSizes(for: source) }
        }
    }

    private func quantizationPicker(_ source: ModelSource,
                                    iconAssetName: String) -> some View {
        let variants = entries(for: source)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ModelIcon(assetName: iconAssetName, size: 24)
                Text(source.repository)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 8)
                if source.languageOnly { modelBadge("Text only") }
            }
            .padding(.horizontal, 4)
            Divider()
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(variants) { entry in
                        HStack(spacing: 10) {
                            Text(entry.quantization)
                                .font(.subheadline.weight(.medium))
                            if entry.quantization == source.recommendedQuantization {
                                modelBadge("Recommended")
                            }
                            Spacer(minLength: 8)
                            if entry.isInstalled {
                                if let size = entry.sizeBytes {
                                    Text(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
                                }
                                Text("Installed")
                                    .foregroundStyle(.secondary)
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                                    .font(.system(size: 14, weight: .medium))
                                    .frame(width: 36, height: 44)
                                    .accessibilityLabel("Installed")
                            } else {
                                if let size = catalog.estimatedSizes[entry.id] {
                                    Text("Approx. \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")
                                } else if catalog.estimatingSourceIDs.contains(source.id) {
                                    Text("Checking size...")
                                } else {
                                    Text("Size unknown")
                                }
                                if state.downloadingModelIDs.contains(entry.id) {
                                    Button {
                                        state.cancelModelDownload(entry.id)
                                    } label: {
                                        Group {
                                            if hoveredDownloadID == entry.id &&
                                                !state.cancellingModelIDs.contains(entry.id) {
                                                Image(systemName: "xmark")
                                                    .font(.system(size: 14, weight: .medium))
                                            } else {
                                                Text(downloadPercent(for: entry.id))
                                                    .font(.caption.monospacedDigit())
                                            }
                                        }
                                        .foregroundStyle(hoveredDownloadID == entry.id
                                                         ? Color.splashBrand : Color.primary)
                                        .frame(width: 36, height: 44)
                                        .background {
                                            RoundedRectangle(cornerRadius: 6)
                                                .fill(hoveredDownloadID == entry.id
                                                      ? Color.splashBrand.opacity(0.16) : .clear)
                                                .frame(width: 28, height: 28)
                                        }
                                        .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .onHover { hoveredDownloadID = $0 ? entry.id : nil }
                                    .help("Cancel download")
                                    .accessibilityLabel("Cancel download, \(downloadPercent(for: entry.id)) complete")
                                    .disabled(state.cancellingModelIDs.contains(entry.id))
                                } else {
                                    Button {
                                        state.beginModelDownload(entry.id)
                                    } label: {
                                        ModelActionLabel(systemName: "arrow.down", height: 44)
                                    }
                                    .buttonStyle(.plain)
                                    .help("Download")
                                    .accessibilityLabel("Download")
                                    .disabled(runtime.busy)
                                }
                            }
                        }
                        .frame(minHeight: 44)
                        .padding(.horizontal, 4)
                        Divider()
                    }
                }
            }
            .frame(height: min(CGFloat(variants.count) * 45, 325))
        }
        .buttonStyle(.borderless)
        .padding(10)
        .frame(width: 480)
    }

    private func estimateSizes(for source: ModelSource) async {
        guard let location = try? await runtime.resolve() else { return }
        await catalog.estimateSizes(for: source, runtimeRoot: location.root)
    }

    private func openChat(for modelID: String) {
        guard state.beginChat(for: modelID) else { return }
        ChatWindowController.shared.show(state: state)
    }

    private func modelBadge(_ title: String) -> some View {
        Text(title)
            .font(.caption2.weight(.medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(Color.primary.opacity(0.07), in: RoundedRectangle(cornerRadius: 5))
            .fixedSize(horizontal: true, vertical: false)
    }

    private func isOperating(on id: String) -> Bool {
        state.downloadingModelIDs.contains(id) || catalog.activeOperationIDs.contains(id) ||
            state.isStartingModel(id)
    }

    private var runningModelStatus: String {
        switch service.status {
        case .starting: return "Starting"
        case .ready(_, let released): return released ? "Weights released" : "Running"
        case .stopping: return "Stopping"
        case .failed: return "Failed"
        case .externalPortOwner: return "Port in use"
        case .stopped: return "Stopped"
        }
    }

    private var runningModelColor: Color {
        if case .ready(_, let released) = service.status, !released { return .green }
        return .secondary
    }

    private var serviceView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                serviceSummary
                Divider().padding(.vertical, 12)
                serviceModelControls
                Divider().padding(.vertical, 12)
                serviceConnection
                if !serviceActivityLog.isEmpty {
                    Divider().padding(.vertical, 12)
                    Text(serviceActivityTitle)
                        .font(.subheadline.weight(.semibold))
                        .padding(.bottom, 8)
                    logView(serviceActivityLog)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
        }
    }

    private var serviceSummary: some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(serviceStatusColor)
                .frame(width: 9, height: 9)
                .padding(.top, 5)
            VStack(alignment: .leading, spacing: 4) {
                Text(serviceStatusTitle)
                    .font(.headline)
                    .foregroundStyle(serviceStatusColor)
                if let detail = serviceStatusDetail {
                    Text(verbatim: detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(detail)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if let location = runtime.location {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Splash \(location.version)")
                    Text(location.origin.rawValue.capitalized)
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
                .fixedSize(horizontal: true, vertical: false)
            } else if runtime.busy {
                ProgressView().controlSize(.small)
            } else {
                Text("Runtime unavailable")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var serviceModelControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model").font(.subheadline.weight(.medium))
            if supportedInstalledModels.isEmpty {
                HStack {
                    Text("No supported models installed").foregroundStyle(.secondary)
                    Spacer()
                    Button("Manage Models", systemImage: "square.stack.3d.up") {
                        state.managerSection = .models
                    }
                }
            } else {
                HStack(spacing: 8) {
                    Menu {
                        ForEach(supportedInstalledModels) { model in
                            Button { state.selectModel(model.id) } label: {
                                if selectedInstalledModelID == model.id {
                                    Label(model.id, systemImage: "checkmark")
                                } else {
                                    Text(model.id)
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 8) {
                            Text(selectedInstalledModelID ?? "Choose a model")
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 8)
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .frame(maxWidth: .infinity, minHeight: 28)
                        .background(Color(nsColor: .controlBackgroundColor),
                                    in: RoundedRectangle(cornerRadius: 6))
                    }
                    .menuStyle(.borderlessButton)
                    .frame(maxWidth: 360, alignment: .leading)
                    .disabled(state.serviceIsBusy)
                    .help(selectedInstalledModelID ?? "Choose a model")
                    Spacer(minLength: 8)
                    Button {
                        Task { await state.startSelectedModel() }
                    } label: {
                        HStack(spacing: 6) {
                            Group {
                                if serviceActionIsBusy {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Image(systemName: serviceActionSymbol)
                                }
                            }
                            .frame(width: 16, height: 16)
                            Text(serviceActionTitle)
                        }
                    }
                    .disabled(!state.canStart)
                    .controlSize(.small)
                }
                .frame(maxWidth: .infinity)
            }
        }
    }

    private var supportedInstalledModels: [InstalledModel] {
        catalog.models.filter { catalog.isSupportedModelID($0.id) }
    }

    private var selectedInstalledModelID: String? {
        guard let id = state.selectedModelID,
              supportedInstalledModels.contains(where: { $0.id == id }) else { return nil }
        return id
    }

    private var serviceChatModelID: String? {
        service.runningModelID ?? selectedInstalledModelID
    }

    private var serviceConnection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Connection").font(.subheadline.weight(.medium))
            HStack(spacing: 12) {
                if state.serviceIsReady, let port = service.runningPort {
                    Text(verbatim: "http://127.0.0.1:\(port)/v1")
                        .font(.system(.body, design: .monospaced))
                        .lineLimit(1)
                        .textSelection(.enabled)
                    Spacer()
                    Button("Copy API Address", systemImage: "doc.on.doc") {
                        state.copyAPIAddress()
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)
                    .modifier(ModelActionHover())
                    .help("Copy API address")
                } else {
                    Text(service.runningPort == nil ? "Configured port" : "Service port")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(verbatim: String(service.runningPort ?? settings.port))
                        .monospacedDigit()
                    if case .externalPortOwner = service.status {
                        Button("Change Port", systemImage: "gearshape") {
                            state.managerSection = .settings
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.plain)
                        .modifier(ModelActionHover())
                        .help("Change port")
                    }
                }
            }
        }
    }

    private var serviceActionTitle: String {
        if let startingModelID = state.startingModelID {
            return service.runningPort != nil && service.runningModelID != startingModelID
                ? "Switching" : "Starting"
        }
        switch service.status {
        case .starting: return "Starting"
        case .stopping: return "Stopping"
        case .ready:
            return service.runningModelID == state.selectedModelID ? "Running" : "Switch"
        case .failed, .externalPortOwner: return "Retry"
        case .stopped: return "Run"
        }
    }

    private var serviceActionSymbol: String {
        state.serviceIsReady && service.runningModelID == state.selectedModelID ? "checkmark" : "play"
    }

    private var serviceActionIsBusy: Bool {
        state.startingModelID != nil || service.status == .starting || service.status == .stopping
    }

    private var serviceStatusTitle: String {
        switch service.status {
        case .stopped: return state.operationInProgress ? "Preparing model" : "Service stopped"
        case .starting: return "Starting model"
        case .ready(_, let released): return released ? "Ready · weights released" : "Running"
        case .stopping: return "Stopping"
        case .failed: return "Failed"
        case .externalPortOwner: return "Port in use"
        }
    }

    private var serviceStatusDetail: String? {
        switch service.status {
        case .stopped: return state.operationInProgress ? state.selectedModelID : nil
        case .starting, .stopping: return service.runningModelID
        case .ready(let modelID, _): return modelID
        case .failed(let reason): return reason
        case .externalPortOwner:
            return "Port \(service.runningPort ?? settings.port) is used by another process."
        }
    }

    private var serviceStatusColor: Color {
        switch service.status {
        case .stopped: return state.operationInProgress ? .orange : .secondary
        case .starting, .stopping: return .orange
        case .ready: return .green
        case .failed, .externalPortOwner: return .red
        }
    }

    private var serviceActivityLog: String {
        runtime.busy ? runtime.progress : service.log
    }

    private var serviceActivityTitle: String {
        if runtime.busy { return "Runtime setup" }
        return service.runningPort == nil ? "Last session" : "Activity"
    }

    private func logView(_ log: String) -> some View {
        Text(log)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .textSelection(.enabled)
            .padding(10)
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(Rectangle().stroke(Color(nsColor: .separatorColor)))
    }

    private var settingsView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(spacing: 8) {
                    settingsRow("Launch at login", subtitle: "Open Splash in the menu bar after signing in.") {
                        Toggle("Launch at login", isOn: Binding(
                            get: { loginItemStatus == .enabled || loginItemStatus == .requiresApproval },
                            set: { updateLaunchAtLogin($0) }
                        ))
                        .labelsHidden()
                        .toggleStyle(.switch)
                    }
                    settingsRow("Server port", subtitle: "The port the API listens on. Default 8000.") {
                        Button(String(settings.port)) { openPortEditor() }
                            .buttonStyle(.bordered)
                            .monospacedDigit()
                    }
                    settingsRow("Release idle weights", subtitle: "Release model weights after inactivity.") {
                        Picker("Release idle weights", selection: Binding(
                            get: { settings.idleRelease },
                            set: { settings.idleRelease = $0 }
                        )) {
                            ForEach(IdleRelease.allCases) { item in Text(item.title).tag(item) }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                    }
                    settingsRow("Model directory", subtitle: "Where downloaded models are stored.") {
                        HStack(spacing: 6) {
                            Button { chooseModelCache() } label: {
                                HStack(spacing: 6) {
                                    Text(modelCacheDisplayPath)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    Image(systemName: "folder")
                                }
                                .frame(maxWidth: 200)
                                .fixedSize(horizontal: true, vertical: false)
                            }
                            .help("Choose model cache directory")
                            .accessibilityLabel("Choose model cache directory")
                            .accessibilityValue(modelCacheDisplayPath)
                            .disabled(catalog.busy || !state.downloadingModelIDs.isEmpty)
                            if settings.modelCacheURL != nil {
                                Button("Use Default") {
                                    state.changeModelCacheDirectory(to: nil)
                                }
                                .help("Restore default model cache directory")
                                .disabled(catalog.busy || !state.downloadingModelIDs.isEmpty)
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                if loginItemStatus == .requiresApproval {
                    HStack(spacing: 8) {
                        Text("Allow Splash under Login Items in System Settings.")
                        Button("Open Login Items") { SMAppService.openSystemSettingsLoginItems() }
                    }
                    .font(.caption)
                }
                if let launchAtLoginError {
                    Text(launchAtLoginError).font(.caption).foregroundStyle(.red)
                }
                Text("Port and idle weight-release changes take effect on the next service start.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("New downloads use the selected cache. Existing files stay in place; the service uses it on the next start.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let port = service.runningPort, port != settings.port {
                    Text("Running on port \(port). Restart the service to use port \(settings.port).")
                        .font(.caption)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .controlSize(.small)
        }
        .onAppear { refreshLaunchAtLogin() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refreshLaunchAtLogin()
        }
    }

    private func settingsRow<Control: View>(
        _ title: String, subtitle: String, @ViewBuilder control: () -> Control
    ) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13))
                Text(subtitle)
                    .font(.system(size: 10.5))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 12)
            control()
        }
        .frame(maxWidth: .infinity, minHeight: 38)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private var portEditor: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Server port").font(.headline)
            TextField("Port", text: $state.portText)
                .textFieldStyle(.roundedBorder)
                .focused($portFieldFocused)
                .onSubmit { savePortFromSheet() }
            if let portError {
                Text(portError).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button("Cancel") {
                    state.portText = String(settings.port)
                    showingPortEditor = false
                }
                .keyboardShortcut(.cancelAction)
                Button("Save") { savePortFromSheet() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 340)
        .controlSize(.small)
        .onAppear { portFieldFocused = true }
    }

    private func openPortEditor() {
        state.portText = String(settings.port)
        portError = nil
        showingPortEditor = true
    }

    private func savePortFromSheet() {
        if state.savePort() {
            showingPortEditor = false
        } else {
            portError = state.errorMessage
            state.errorMessage = nil
        }
    }

    private var modelCacheURL: URL { catalog.hubCacheRoot }

    private var modelCacheDisplayPath: String {
        let path = modelCacheURL.path
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return path.hasPrefix(home + "/") ? "~" + path.dropFirst(home.count) : path
    }

    private func chooseModelCache() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = modelCacheURL
        panel.prompt = "Use Folder"
        panel.message = "Future downloads will use this folder. Existing cached files will not be moved."
        panel.begin { response in
            if response == .OK, let directory = panel.url {
                state.changeModelCacheDirectory(to: directory)
            }
        }
    }

    private func refreshLaunchAtLogin() {
        loginItemStatus = SMAppService.mainApp.status
    }

    private func updateLaunchAtLogin(_ enabled: Bool) {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            launchAtLoginError = nil
        } catch {
            launchAtLoginError = error.localizedDescription
        }
        refreshLaunchAtLogin()
    }

}

struct ChatWindowView: View {
    @ObservedObject var state: DesktopState
    @ObservedObject private var service: ServiceController
    @State private var loadError: String?

    init(state: DesktopState) {
        self.state = state
        service = state.service
    }

    var body: some View {
        Group {
            if let modelID = state.chatRequestedModelID,
               service.runningModelID == modelID, let url = state.chatURL {
                if let loadError {
                    VStack(spacing: 12) {
                        Image(systemName: "wifi.exclamationmark")
                            .font(.system(size: 30))
                            .foregroundStyle(.secondary)
                        Text("Could not load chat").font(.headline)
                        Text(loadError).font(.caption).foregroundStyle(.secondary)
                        Button("Reload chat") { self.loadError = nil }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ChatWebView(url: url) { self.loadError = $0 }
                        .id(modelID)
                }
            } else {
                VStack(spacing: 12) {
                    if isLoading {
                        ProgressView().controlSize(.large)
                        Text("Starting model").font(.headline)
                    } else {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(.system(size: 30))
                            .foregroundStyle(.secondary)
                        Text("Chat unavailable")
                            .font(.headline)
                        Text(statusMessage).foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(minWidth: 720, minHeight: 560)
        .onChange(of: service.runningModelID) { _, _ in loadError = nil }
        .onChange(of: service.runningPort) { _, _ in loadError = nil }
    }

    private var isLoading: Bool {
        guard let id = state.chatRequestedModelID else { return false }
        return state.chatStartPending || state.isStartingModel(id)
    }

    private var statusMessage: String {
        if let error = state.errorMessage { return error }
        if case .failed(let reason) = service.status { return reason }
        if service.status == .externalPortOwner { return "The configured port is in use." }
        if service.status == .stopping { return "The service is stopping." }
        return "Run a model to chat."
    }
}

private struct ChatWebView: NSViewRepresentable {
    let url: URL
    let onError: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onError: onError) }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.uiDelegate = context.coordinator
        context.coordinator.loadedURL = url
        view.load(URLRequest(url: url))
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        context.coordinator.onError = onError
        guard context.coordinator.loadedURL != url else { return }
        context.coordinator.loadedURL = url
        view.load(URLRequest(url: url))
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        var loadedURL: URL?
        var onError: (String) -> Void

        init(onError: @escaping (String) -> Void) { self.onError = onError }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            report(error)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
                     withError error: Error) {
            report(error)
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let target = navigationAction.request.url else {
                decisionHandler(.allow)
                return
            }
            guard target.scheme == "http" || target.scheme == "https" else {
                decisionHandler(.allow)
                return
            }
            if target.host == loadedURL?.host, target.port == loadedURL?.port,
               navigationAction.targetFrame != nil {
                decisionHandler(.allow)
                return
            }
            NSWorkspace.shared.open(target)
            decisionHandler(.cancel)
        }

        func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                     initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
            guard let window = webView.window else {
                completionHandler(false)
                return
            }
            let alert = NSAlert()
            alert.messageText = "Confirm deletion"
            alert.informativeText = message
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Cancel")
            alert.addButton(withTitle: "Delete")
            alert.beginSheetModal(for: window) { response in
                completionHandler(response == .alertSecondButtonReturn)
            }
        }

        private func report(_ error: Error) {
            guard (error as NSError).code != NSURLErrorCancelled else { return }
            onError(error.localizedDescription)
        }
    }
}

struct ModelIcon: View {
    let assetName: String
    var size: CGFloat = 34

    var body: some View {
        Image(assetName)
            .resizable()
            .scaledToFill()
            .frame(width: size, height: size)
            .background(.white)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.08)))
            .accessibilityHidden(true)
    }

}

struct ModelActionIconLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.icon
            .font(.system(size: 14, weight: .medium))
            .frame(width: 28, height: 28)
            .contentShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct ModelActionLabel: View {
    let systemName: String
    var height: CGFloat = 56

    var body: some View {
        Image(systemName: systemName)
            .modifier(ModelActionHover(hitWidth: 36, hitHeight: height))
    }
}

struct ModelActionHover: ViewModifier {
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovered = false
    var hitWidth: CGFloat = 28
    var hitHeight: CGFloat = 28

    func body(content: Content) -> some View {
        let highlighted = isHovered && isEnabled
        content
            .font(.system(size: 14, weight: .medium))
            .frame(width: hitWidth, height: hitHeight)
            .tint(highlighted ? Color.splashBrand : Color.primary)
            .foregroundStyle(highlighted ? Color.splashBrand : Color.primary)
            .background {
                RoundedRectangle(cornerRadius: 6)
                    .fill(highlighted ? Color.splashBrand.opacity(0.16) : .clear)
                    .frame(width: 28, height: 28)
            }
            .contentShape(Rectangle())
            .onHover { isHovered = $0 }
    }
}
