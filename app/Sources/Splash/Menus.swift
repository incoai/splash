import AppKit
import Observation
import SwiftUI

/// Starting the model picked, or stopping the one that runs, and the coding agents on it.
struct ServerMenuItems: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if let stop = state.stopItem {
            Button("Stop \(state.displayName(of: stop.model))") { state.stop() }
                .disabled(!stop.enabled)
        } else if let model = state.modelToStart {
            Button("Start \(state.displayName(of: model))") { state.start(model) }
                .disabled(!state.canStart)
        }
        Menu("Coding Agents") { AgentMenuItems() }
    }
}

/// The coding agents, and the address other apps reach the model at.
struct AgentMenuItems: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Section("Open in Terminal") {
            ForEach(Agent.allCases) { agent in
                Button("\(agent.title)…") { state.open(agent) }
                    .disabled(state.server.phase != .ready)
            }
        }
        Divider()
        Button("Copy API Address") { state.copyAPIAddress() }
    }
}

/// The menu bar item and its menu, a menu of AppKit's: a press on the item can drag to a row and
/// release on it, as in any menu. The rows are made each time the menu opens.
@MainActor
final class StatusMenu: NSObject, NSMenuDelegate {
    private let state: AppState
    private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

    init(state: AppState) {
        self.state = state
        super.init()
        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.delegate = self
        item.menu = menu
        item.button?.setAccessibilityLabel("Splash")
        followServer()
    }

    /// Keeps the mark as the server is.
    private func followServer() {
        let serving = withObservationTracking {
            state.server.answers
        } onChange: { [weak self] in
            Task { @MainActor in self?.followServer() }
        }
        item.button?.image = SplashMark.menuBarImage(serving: serving)
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        let state = state
        let server = state.server
        menu.removeAllItems()
        menu.addItem(header())
        menu.addItem(.separator())
        menu.addItem(ActionItem("Open Server") { Self.show(.server, in: state) })
        menu.addItem(ActionItem("Open Chat") { Self.show(.chat, in: state) })
        menu.addItem(.separator())
        if let stop = state.stopItem {
            let item = ActionItem("Stop \(state.displayName(of: stop.model))") { state.stop() }
            item.isEnabled = stop.enabled
            menu.addItem(item)
        } else if let start = state.modelToStart {
            let item = ActionItem("Start \(state.displayName(of: start))") { state.start(start) }
            item.isEnabled = state.canStart
            menu.addItem(item)
        }
        if server.phase == .ready {
            let agents = NSMenuItem(title: "Coding Agents", action: nil, keyEquivalent: "")
            agents.submenu = agentsMenu()
            menu.addItem(agents)
        }
        if menu.items.last?.isSeparatorItem == false { menu.addItem(.separator()) }
        menu.addItem(ActionItem("Quit Splash", key: "q") { NSApp.terminate(nil) })
    }

    private func header() -> NSMenuItem {
        let view = NSHostingView(rootView: MenuBarHeader().environment(state))
        view.frame.size = view.fittingSize
        let item = NSMenuItem()
        item.view = view
        return item
    }

    private func agentsMenu() -> NSMenu {
        let state = state
        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.addItem(.sectionHeader(title: "Open in Terminal"))
        for agent in Agent.allCases {
            menu.addItem(ActionItem("\(agent.title)…") { state.open(agent) })
        }
        menu.addItem(.separator())
        menu.addItem(ActionItem("Copy API Address") { state.copyAPIAddress() })
        return menu
    }

    /// Shows the window, which may have closed: the app opened again brings it back, as a second
    /// launch does, and the window in turn brings back the Dock and the menu bar.
    private static func show(_ section: AppState.Section, in state: AppState) {
        state.section = section
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: NSWorkspace.OpenConfiguration())
        NSApp.activate()
    }
}

/// A row of the menu that runs `run`.
private final class ActionItem: NSMenuItem {
    private let run: @MainActor () -> Void

    init(_ title: String, key: String = "", run: @escaping @MainActor () -> Void) {
        self.run = run
        super.init(title: title, action: #selector(choose), keyEquivalent: key)
        target = self
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("not used")
    }

    @MainActor @objc private func choose() {
        run()
    }
}

/// The top of the menu bar's menu, in Splash's own style: what runs, with its totals while it
/// serves, and any download. It stays current while the menu is open.
struct MenuBarHeader: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let server = state.server
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 10) {
                StatusMark(look: state.serverStatus.look, since: server.since)
                VStack(alignment: .leading, spacing: 1) {
                    Text(server.model.map(state.displayName(of:)) ?? "No model running")
                        .font(.headline)
                        .truncationMode(.middle)
                    if server.model != nil { Text(state.serverStatus.phrase).font(.callout).foregroundStyle(.secondary) }
                }
                .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            if server.phase == .ready {
                totals(server.activity)
            }
            if let download = state.downloadStatus {
                Text(download).font(.callout).foregroundStyle(.secondary).padding(.horizontal, 10).padding(.bottom, 6)
            }
        }
        .padding(.horizontal, 6)
        .frame(width: 300, alignment: .leading)
    }

    /// How the model does since it started: answering speed, requests and memory; the tokens it
    /// generated, read and took from its cache.
    private func totals(_ activity: Server.Activity) -> some View {
        let memory = activity.memory.map { Format.parts(Format.memory($0)) }
        return VStack(spacing: 10) {
            HStack(spacing: 0) {
                Figure(value: activity.generation.map(Format.whole), unit: "tok/s", caption: "Answering", compact: true)
                Figure(value: activity.served.formatted(), caption: "Requests", compact: true)
                Figure(value: memory?.0, unit: memory?.1, caption: "Memory", compact: true)
            }
            HStack(spacing: 0) {
                Figure(value: Format.count(activity.generated), caption: "Generated", compact: true)
                Figure(value: Format.count(activity.read), caption: "Prompt read", compact: true)
                Figure(value: Format.count(activity.cached), caption: "From cache", compact: true)
            }
        }
        .padding(.horizontal, 10)
        .padding(.bottom, 8)
        .animation(Motion.settle, value: activity)
    }
}
