import SwiftUI

/// Stopping the model that runs, and the coding agents on it; models start on the Server screen.
struct ServerMenuItems: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if let stop = state.stopItem {
            Button("Stop \(state.displayName(of: stop.model))") { state.stop() }
                .disabled(!stop.enabled)
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

/// The menu bar item's panel, in Splash's own style: what runs, with its totals while it serves,
/// any download, then the window's two parts, stopping the model, its coding agents, and quitting.
struct MenuBarPanel: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow
    @State private var agents = false

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
            Divider().padding(.horizontal, 8).padding(.vertical, 4)
            PanelRow(title: "Open Server") { show(.server) }
            PanelRow(title: "Open Chat") { show(.chat) }
            Divider().padding(.horizontal, 8).padding(.vertical, 4)
            if let stop = state.stopItem {
                PanelRow(title: "Stop \(state.displayName(of: stop.model))") { state.stop() }
                    .disabled(!stop.enabled)
            }
            if server.phase == .ready {
                PanelRow(title: "Coding Agents", open: agents) { withAnimation(Motion.settle) { agents.toggle() } }
                if agents {
                    ForEach(Agent.allCases) { agent in
                        PanelRow(title: "\(agent.title)…", indent: true) { state.open(agent) }
                    }
                    CopyButton(copy: state.copyAPIAddress) { PanelLabel(title: "Copy API Address", indent: true, check: $0) }
                        .buttonStyle(.hoverFlush)
                }
            }
            if server.running {
                Divider().padding(.horizontal, 8).padding(.vertical, 4)
            }
            PanelRow(title: "Quit Splash", shortcut: "⌘Q") { NSApp.terminate(nil) }
        }
        .padding(6)
        .frame(width: 300)
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

    private func show(_ section: AppState.Section) {
        state.section = section
        openWindow(id: "main")
        NSApp.activate()
    }
}

/// A row of the menu bar's panel, highlighted under the pointer as the app's popups are; one that
/// opens rows under it shows a chevron.
private struct PanelRow: View {
    let title: String
    var shortcut: String?
    var open: Bool?
    var indent = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            PanelLabel(title: title, shortcut: shortcut, open: open, indent: indent)
        }
        .buttonStyle(.hoverFlush)
        .accessibilityValue(open.map { $0 ? "Expanded" : "Collapsed" } ?? "")
    }
}

/// What a row of the menu bar's panel shows: its title, and a shortcut, a chevron or a check.
private struct PanelLabel: View {
    let title: String
    var shortcut: String?
    var open: Bool?
    var indent = false
    var check = false

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            if let shortcut { Text(shortcut).foregroundStyle(.secondary) }
            if let open {
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .rotationEffect(.degrees(open ? 90 : 0))
                    .accessibilityHidden(true)
            }
            if check {
                Image(systemName: "checkmark").font(.system(size: 12, weight: .semibold)).foregroundStyle(.secondary)
            }
        }
        .padding(.leading, indent ? 22 : 10)
        .padding(.trailing, 10)
        .padding(.vertical, 5)
    }
}
