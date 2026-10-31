import AppKit
import Carbon
import SwiftUI

extension Color {
    static let splashBrand = Color(red: 227.0 / 255.0, green: 131.0 / 255.0, blue: 101.0 / 255.0)
}

@main
struct SplashApp: App {
    @NSApplicationDelegateAdaptor(SplashAppDelegate.self) private var delegate
    @StateObject private var state = DesktopState.shared

    var body: some Scene {
        MenuBarExtra {
            MenuPanel(state: state)
        } label: {
            MenuBarLaunchLabel()
        }
        .menuBarExtraStyle(.window)
        Window(state.managerSection.rawValue, id: "manager") {
            ManagerView(state: state)
                .onAppear { WindowActivation.setVisible("manager", true) }
                .onDisappear { WindowActivation.setVisible("manager", false) }
        }
        .defaultLaunchBehavior(.suppressed)
    }
}

enum ManagerLaunchPolicy {
    static func shouldOpenWindow(for event: NSAppleEventDescriptor?) -> Bool {
        event?.paramDescriptor(forKeyword: AEKeyword(keyAELaunchedAsLogInItem)) == nil
    }
}

private extension Notification.Name {
    static let openSplashManager = Notification.Name("OpenSplashManager")
}

private struct MenuBarLaunchLabel: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Image("SplashMenuMark")
            .accessibilityLabel("Splash")
            .onReceive(NotificationCenter.default.publisher(for: .openSplashManager)) { _ in
                NSApplication.shared.setActivationPolicy(.regular)
                openWindow(id: "manager")
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
    }
}

@MainActor
private enum WindowActivation {
    static var visible: Set<String> = []

    static func setVisible(_ id: String, _ isVisible: Bool) {
        if isVisible { visible.insert(id) } else { visible.remove(id) }
        NSApplication.shared.setActivationPolicy(visible.isEmpty ? .accessory : .regular)
    }
}

// SwiftUI Window scenes are auxiliary; this window needs primary full-screen behavior.
@MainActor
final class ChatWindowController: NSObject, NSWindowDelegate {
    static let shared = ChatWindowController()
    private static let frameAutosaveName = "SplashChatWindow"
    private var window: NSWindow?

    func show(state: DesktopState) {
        let window = window ?? makeWindow(state: state)
        window.title = state.chatRequestedModelID.map { "Splash Chat · \($0)" } ?? "Splash Chat"
        WindowActivation.setVisible("chat", true)
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    private func makeWindow(state: DesktopState) -> NSWindow {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1000, height: 800),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.collectionBehavior = [.primary, .fullScreenPrimary]
        window.contentMinSize = NSSize(width: 720, height: 560)
        window.contentViewController = NSHostingController(rootView: ChatWindowView(state: state))
        window.isReleasedWhenClosed = false
        window.delegate = self
        if !window.setFrameUsingName(Self.frameAutosaveName) { window.center() }
        window.setFrameAutosaveName(Self.frameAutosaveName)
        self.window = window
        return window
    }

    func windowWillClose(_ notification: Notification) {
        WindowActivation.setVisible("chat", false)
        window = nil
    }
}

private struct MenuPanel: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var state: DesktopState
    @ObservedObject private var service: ServiceController
    @ObservedObject private var catalog: ModelCatalog
    @ObservedObject private var runtime: RuntimeInstaller
    @State private var hoveringOpenModels = false
    @State private var hoveringCopyAddress = false

    init(state: DesktopState) {
        self.state = state
        service = state.service
        catalog = state.catalog
        runtime = state.runtime
    }

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            Divider()
            if installedEntries.count > 4 {
                ScrollView { installedModels.padding(6) }
                    .frame(height: 252)
            } else {
                installedModels.padding(6)
            }
            Divider()
            openModelsRow.padding(6)
            if let message = state.errorMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 6)
            }
            Divider()
            footer
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
        }
        .frame(width: 320)
        .background(Color(nsColor: .windowBackgroundColor))
        .task { await state.initialize() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 7) {
                Text("Splash").font(.headline)
                Circle().fill(statusColor).frame(width: 7, height: 7)
                    .help(statusText)
                Spacer()
            }
            HStack(spacing: 7) {
                Text(service.runningPort.map { "127.0.0.1:\($0)" } ?? statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .help(statusText)
                if service.runningPort != nil {
                    Button("Copy API Address", systemImage: "doc.on.doc") { state.copyAPIAddress() }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.plain)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 26, height: 26)
                        .background(hoveringCopyAddress ? Color.primary.opacity(0.08) : .clear,
                                    in: RoundedRectangle(cornerRadius: 5))
                        .contentShape(Rectangle())
                        .onHover { hoveringCopyAddress = $0 }
                        .help("Copy API address")
                        .accessibilityLabel("Copy API address")
                }
                Spacer()
                if service.status == .stopping {
                    Button("Force Stop", systemImage: "xmark.octagon") {
                        Task { await state.stop(force: true) }
                    }
                    .labelStyle(ModelActionIconLabelStyle())
                    .buttonStyle(.borderless)
                    .modifier(ModelActionHover())
                    .help("Force stop; active requests may be interrupted")
                } else if service.runningPort != nil {
                    Button("Stop Service", systemImage: "stop") {
                        Task { await state.stop() }
                    }
                    .labelStyle(ModelActionIconLabelStyle())
                    .buttonStyle(.borderless)
                    .modifier(ModelActionHover())
                    .help("Stop Service")
                    .disabled(state.operationInProgress)
                }
            }
        }
    }

    private var installedEntries: [ModelEntry] { catalog.entries().filter(\.isInstalled) }

    @ViewBuilder private var installedModels: some View {
        if installedEntries.isEmpty {
            Text("No supported models installed")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                .padding(.horizontal, 8)
        } else {
            modelSection("Installed models", entries: installedEntries)
        }
    }

    private func modelSection(_ title: String, entries: [ModelEntry]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .padding(.horizontal, 8)
                .padding(.bottom, 2)
            ForEach(entries) { entry in
                MenuModelRow(entry: entry,
                             status: modelStatus(entry),
                             symbol: actionSymbol(entry),
                             actionTitle: actionTitle(entry),
                             isBusy: rowIsBusy(entry),
                             isEnabled: rowIsEnabled(entry),
                             chatEnabled: state.canLaunchChat(for: entry.id),
                             action: { performAction(for: entry) },
                             chatAction: { openChat(for: entry.id) })
            }
        }
    }

    private func modelStatus(_ entry: ModelEntry) -> (text: String, color: Color)? {
        if state.isStartingModel(entry.id) { return ("Starting", .secondary) }
        guard service.runningModelID == entry.id,
              case .ready(_, let released) = service.status else { return nil }
        return released ? ("Weights released", .secondary) : ("Running", .green)
    }

    private func actionSymbol(_ entry: ModelEntry) -> String {
        if service.runningModelID == entry.id { return "stop" }
        return "play"
    }

    private func actionTitle(_ entry: ModelEntry) -> String {
        if state.isStartingModel(entry.id) { return "Starting" }
        if service.runningModelID == entry.id { return "Stop" }
        return "Run"
    }

    private func rowIsBusy(_ entry: ModelEntry) -> Bool {
        state.downloadingModelIDs.contains(entry.id) ||
            catalog.activeOperationIDs.contains(entry.id) ||
            state.isStartingModel(entry.id) ||
            (service.runningModelID == entry.id && service.status == .stopping)
    }

    private func rowIsEnabled(_ entry: ModelEntry) -> Bool {
        if rowIsBusy(entry) { return false }
        if service.runningModelID == entry.id { return !state.operationInProgress }
        return state.canRun(entry.id)
    }

    private func performAction(for entry: ModelEntry) {
        if service.runningModelID == entry.id {
            Task { await state.stop() }
        } else {
            Task { await state.startModel(entry.id) }
        }
    }

    private func openChat(for modelID: String) {
        guard state.beginChat(for: modelID) else { return }
        NSApplication.shared.keyWindow?.orderOut(nil)
        ChatWindowController.shared.show(state: state)
    }

    private var openModelsRow: some View {
        Button { openManager(.models) } label: {
            HStack(spacing: 10) {
                Image(systemName: "square.grid.2x2")
                    .frame(width: 26, height: 26)
                    .background(Color.primary.opacity(0.06), in: Circle())
                VStack(alignment: .leading, spacing: 2) {
                    Text("Browse models").font(.subheadline)
                    Text("Supported models")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
            .frame(height: 44)
            .background(hoveringOpenModels ? Color.primary.opacity(0.06) : .clear,
                        in: RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hoveringOpenModels = $0 }
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Text(appVersion)
                .foregroundStyle(.secondary)
            if let version = runtime.location?.version {
                Text("· splash-\(version)").foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            Button("Settings") { openManager(.settings) }
            Button("Quit") { NSApplication.shared.terminate(nil) }
        }
        .font(.caption)
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    private var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "SplashReleaseVersion") as? String ?? "dev"
    }

    private func openManager(_ section: ManagerSection) {
        NSApplication.shared.keyWindow?.orderOut(nil)
        state.managerSection = section
        NSApplication.shared.setActivationPolicy(.regular)
        openWindow(id: "manager")
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    private var statusColor: Color {
        switch service.status {
        case .ready: return .green
        case .starting, .stopping: return .orange
        case .failed, .externalPortOwner: return .red
        case .stopped: return .secondary
        }
    }

    private var statusText: String {
        switch service.status {
        case .stopped: return "Service stopped"
        case .starting: return "Starting model"
        case .ready(_, let released): return released ? "Ready · weights released" : "Running"
        case .stopping: return "Stopping"
        case .failed(let reason): return reason
        case .externalPortOwner: return "Port in use"
        }
    }
}

private struct MenuModelRow: View {
    let entry: ModelEntry
    let status: (text: String, color: Color)?
    let symbol: String
    let actionTitle: String
    let isBusy: Bool
    let isEnabled: Bool
    let chatEnabled: Bool
    let action: () -> Void
    let chatAction: () -> Void
    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 10) {
            ModelIcon(assetName: entry.iconAssetName, size: 26)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.title).font(.subheadline).lineLimit(1)
                HStack(spacing: 0) {
                    Text(entry.format).foregroundStyle(.secondary)
                    if let status {
                        Text(" · ").foregroundStyle(.secondary)
                        Text(status.text).foregroundStyle(status.color)
                    }
                }
                .font(.caption)
                .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            HStack(spacing: 0) {
                if isBusy {
                    ProgressView().controlSize(.small).frame(width: 36, height: 44)
                } else {
                    Button(action: action) {
                        ModelActionLabel(systemName: symbol, height: 44)
                    }
                    .disabled(!isEnabled)
                    .help(actionTitle)
                    .accessibilityLabel(actionTitle)
                }
                Button(action: chatAction) {
                    ModelActionLabel(systemName: "bubble.left", height: 44)
                }
                .disabled(!chatEnabled)
                .help("Chat")
                .accessibilityLabel("Chat")
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 8)
        .frame(height: 44)
        .background(isHovered && (isEnabled || chatEnabled) ? Color.primary.opacity(0.06) : .clear,
                    in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
    }
}

private final class SplashAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        NotificationCenter.default.post(name: .openSplashManager, object: nil)
        return true
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let shouldOpenManager = ManagerLaunchPolicy.shouldOpenWindow(
            for: NSAppleEventManager.shared().currentAppleEvent)
        Task { @MainActor in
            if shouldOpenManager {
                NotificationCenter.default.post(name: .openSplashManager, object: nil)
            }
            await DesktopState.shared.initialize()
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let state = DesktopState.shared
        state.cancelAllModelDownloads()
        let service = state.service
        guard service.runningPort != nil else { return .terminateNow }
        Task { @MainActor in
            let stopped = await service.stop()
            if !stopped {
                state.errorMessage = "Could not verify active requests. Use Force Stop before quitting."
            }
            sender.reply(toApplicationShouldTerminate: stopped)
        }
        return .terminateLater
    }
}
