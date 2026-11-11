import SwiftUI

/// The menu bar item's panel, in Splash's own style: what runs, with its totals while it serves,
/// any download, then the window's two parts, the models on this Mac to run, stopping the model,
/// its coding agents, and quitting.
struct MenuBarPanel: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow
    @State private var models = false
    /// The height of the Model rows, all of them, which the panel shows up to its limit.
    @State private var modelRows: CGFloat = 0
    @State private var agents = false

    var body: some View {
        let server = state.server
        // The model stopped is the window's to name, as the one to start again.
        let model = server.phase == .stopped ? nil : server.model
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 10) {
                StatusMark(look: state.serverStatus.look, since: server.since)
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.map(state.displayName(of:)) ?? "No model running")
                        .font(.headline)
                        .truncationMode(.middle)
                    if model != nil { Text(state.serverStatus.phrase).font(.callout).foregroundStyle(.secondary) }
                }
                .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            // While warming, why the start can take long.
            if let note = state.serverStatus.note {
                Text(note).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 10).padding(.bottom, 6)
            }
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
            if !state.installed.isEmpty {
                // Its rows, as Coding Agents' do, come and go at once, as a menu's items do: the
                // panel's window takes its new height at once. Faded and slid in, they drew over
                // the rows moving past them, a check under a chevron, the panel's content laid out
                // off its window's middle until the animation ended.
                PanelRow(title: "Model", open: models) { models.toggle() }
                if models {
                    let checked = state.checkedModel
                    // As high as the rows up to some ten of them, then scrolling, so that a long
                    // list of models keeps the panel on the screen. A scroll view asks for no
                    // height of its own, so the rows' is measured.
                    ScrollView {
                        VStack(spacing: 1) {
                            ForEach(state.installed) { model in
                                PanelRow(title: state.displayName(of: model.id), indent: true, check: model.id == checked) {
                                    state.choose(model.id)
                                }
                                .disabled(!state.canStart(model.id))
                            }
                        }
                        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { modelRows = $0 }
                    }
                    .frame(height: min(modelRows, 280))
                    .scrollBounceBehavior(.basedOnSize)
                }
            }
            if let stop = state.stopItem {
                PanelRow(title: "Stop \(state.displayName(of: stop.model))") { state.stop() }
                    .disabled(!stop.enabled)
            }
            if server.phase == .ready {
                PanelRow(title: "Coding Agents", open: agents) { agents.toggle() }
                if agents {
                    // In the window's drawer, which comes forward; a folder is asked for first.
                    ForEach(Agent.allCases) { agent in
                        PanelRow(title: state.agentFolder == nil ? "\(agent.title)…" : agent.title, indent: true) {
                            openWindow(id: "main")
                            state.open(agent)
                        }
                    }
                    PanelRow(title: "Choose Folder…", shortcut: state.agentFolder?.lastPathComponent, indent: true) {
                        state.chooseAgentFolder()
                    }
                    CopyButton(copy: state.copyAPIAddress) { PanelLabel(title: "Copy API Address", indent: true, check: $0) }
                        .buttonStyle(.hoverFlush)
                }
            }
            if server.running || !state.installed.isEmpty {
                Divider().padding(.horizontal, 8).padding(.vertical, 4)
            }
            PanelRow(title: "Quit Splash", shortcut: "⌘Q") { NSApp.terminate(nil) }
        }
        .padding(6)
        .frame(width: 300)
        .onAppear { state.panelShows = true }
        .onDisappear { state.panelShows = false }
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
/// opens rows under it shows a chevron, the one chosen of the rows under it a check.
private struct PanelRow: View {
    let title: String
    var shortcut: String?
    var open: Bool?
    var indent = false
    var check = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            PanelLabel(title: title, shortcut: shortcut, open: open, indent: indent, check: check)
        }
        .buttonStyle(.hoverFlush)
        .accessibilityValue(open.map { $0 ? "Expanded" : "Collapsed" } ?? "")
        .accessibilityAddTraits(check ? .isSelected : [])
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
