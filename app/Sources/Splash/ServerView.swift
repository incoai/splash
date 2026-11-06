import SwiftUI
import UniformTypeIdentifiers

/// The width of the Server screen's column.
private let column: CGFloat = 640

/// The Server screen, where the window opens: the models on this Mac to pick from, how they
/// start, the suggested ones this Mac can run to download, and, under them all, what runs with
/// Start, Stop and Switch. Models start and change only here.
struct ServerView: View {
    @Environment(AppState.self) private var state
    @State private var customModel = false
    @State private var width: CGFloat = 0
    /// For pictures of the screen: the launch options' Advanced group and the request log open.
    var advanced = false

    var body: some View {
        let suggestions = state.catalog.suggestions(memory: Catalog.memory)
        let installedNames = Set(state.installed.map { state.catalog.name(of: $0.id) })
        let available = suggestions.filter { !installedNames.contains($0.name) && $0.id != state.installer.model }
        let first = state.installed.isEmpty
        let failed = if case .failed = state.server.phase { true } else { false }
        Form {
            if state.server.phase == .ready {
                ActivitySection(advanced: advanced)
            }
            if first || failed || state.installer.failure != nil || state.installer.running || state.keyProblem != nil {
                Section {
                    if let problem = state.keyProblem {
                        ProblemRow(title: "Splash can’t use its API key",
                                   message: "The Keychain says: \(problem) Allow Splash to use it, then try again.",
                                   action: ("Try Again", state.readKey))
                    }
                    if case .failed(let message) = state.server.phase, let model = state.server.model {
                        ProblemRow(title: "\(state.displayName(of: model)) ran into a problem", message: message,
                                   log: .server, dismiss: state.server.dismissFailure,
                                   // Files on this Mac that can't serve won't serve on a second try.
                                   action: model.hasPrefix("/") && !state.server.running
                                       ? ("Remove from List", { state.server.dismissFailure(); state.removeLocal(model) }) : nil)
                    }
                    if let failure = state.installer.failure {
                        ProblemRow(title: "\(state.displayName(of: failure.model)) couldn’t download",
                                   message: failure.message, log: .downloads, dismiss: state.installer.dismissFailure,
                                   action: ("Download Again", { state.download(failure.model) }))
                    }
                    if state.installer.running { DownloadRow() }
                } header: {
                    if first { welcome }
                }
            }
            if !first {
                Section {
                    ForEach(state.installed) { ModelRow(model: $0) }
                } header: {
                    SectionTitle(text: "Models on This Mac")
                } footer: {
                    otherModels
                }
                LaunchOptions(advanced: advanced)
            }
            Section {
                ForEach(available, id: \.id) { variant in
                    SuggestedRow(variant: variant, recommended: variant == suggestions.first,
                                 lead: first && !state.installer.running && !state.findingCachedModels && variant == suggestions.first)
                }
            } header: {
                if !available.isEmpty { SectionTitle(text: "Available to Download") }
            } footer: {
                if suggestions.isEmpty, let minimum = state.catalog.minimumMemory {
                    Text("The suggested models need a Mac with at least \(minimum) GB of memory; this one has \(Catalog.memory) GB.")
                }
                if first { otherModels }
            }
        }
        .formStyle(.grouped)
        // The screen changes its shape as one: rows and sections fade in and out as the models,
        // downloads and options change, the rows after them moving along; as the server's stage
        // changes too, which the window animates (`MainView`).
        .animation(Motion.settle, value: Layout(installed: state.installed.map(\.id), downloading: state.installer.running,
                                                failure: state.installer.failure?.message, keyProblem: state.keyProblem,
                                                finding: state.findingCachedModels, options: state.options,
                                                family: state.family(of: state.selection)))
        .contentMargins(.horizontal, max(0, (width - column) / 2), for: .scrollContent)
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width = $0 }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if !first || state.server.running {
                VStack(spacing: 0) {
                    Divider()
                    ControlBar()
                }
                .background(.bar)
            }
        }
        .onAppear { state.refresh() }
        .sheet(isPresented: $customModel) { CustomModelSheet() }
    }

    /// What the screen holds besides the server's stage: the models, a download or a problem, and
    /// the launch options, those the family of the model picked uses; the model picked is not in
    /// it, so that picking one of the same family is immediate.
    private struct Layout: Equatable {
        var installed: [String]
        var downloading: Bool
        var failure: String?
        var keyProblem: String?
        var finding: Bool
        var options: ServeSettings
        var family: ModelFamily
    }

    /// Ways to models the list doesn't offer: one from Hugging Face by its ID, or files on this Mac.
    private var otherModels: some View {
        HStack(spacing: 4) {
            Button("Download Another Model…") { customModel = true }
                .disabled(state.installer.running || !state.ready)
            Button("Add Model from This Mac…", action: addLocal)
                .disabled(!state.ready)
        }
        .buttonStyle(.hover)
        .foregroundStyle(.tint)
        .padding(.leading, -6)
    }

    /// Asks for model files on this Mac: a GGUF file, or the folder of an MLX or GGUF model.
    private func addLocal() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [.folder] + (UTType(filenameExtension: "gguf").map { [$0] } ?? [])
        panel.prompt = "Add"
        panel.message = "Choose a GGUF file, or the folder of an MLX or GGUF model."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        state.addLocal(url)
    }

    /// The screen's heading before a model is on this Mac, while the ones other tools left are
    /// looked for and after.
    private var welcome: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Welcome to Splash")
                .font(.title2.weight(.semibold))
                .foregroundStyle(.primary)
            if state.findingCachedModels {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Looking for models already on this Mac…")
                }
                .font(.body)
                .foregroundStyle(.secondary)
            } else {
                Text("Splash runs AI models privately on this Mac. Download one to get started.")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
        }
        .textCase(nil)
        .padding(.top, 12)
        .padding(.bottom, 16)
    }
}

/// Under the screen, in two lines: the model running, or the one picked, and what it does, with
/// what to do next: start it or try again, cancel, stop, switch to the one picked, restart with
/// changed options, open a coding agent, or chat.
private struct ControlBar: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let server = state.server
        let model = server.running ? server.model : state.selection
        HStack(spacing: 12) {
            StatusMark(look: look, since: server.since)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.map(state.displayName(of:)) ?? "")
                    .font(.headline)
                    .truncationMode(.middle)
                    .contentTransition(.opacity)
                detail
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            actions.fixedSize()
        }
        .lineLimit(1)
        .controlSize(.large)
        .padding(.horizontal, 10)
        .frame(maxWidth: column - 40)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        // The model picked and changed options change it with one cross-fade, as the server's
        // stage does (`MainView`).
        .animation(Motion.settle, value: state.selection)
        .animation(Motion.settle, value: state.optionsChanged)
    }

    /// Whether the model picked is the one that failed, so that Start tries it again.
    private var failedPicked: Bool {
        if case .failed = state.server.phase { state.server.model == state.selection } else { false }
    }

    /// Whether the server's failure is of another model than the one the bar names, which then
    /// shows as not running.
    private var othersFailure: Bool {
        if case .failed = state.server.phase { !state.server.running && !failedPicked } else { false }
    }

    private var look: MarkLook { othersFailure ? .off : state.serverStatus.look }

    @ViewBuilder private var detail: some View {
        let server = state.server
        switch server.phase {
        case .starting:
            // The step, with its seconds once it takes long enough to wonder about.
            let step = server.step.rawValue
            TimelineView(.periodic(from: server.stepSince, by: 1)) { context in
                let seconds = Int(context.date.timeIntervalSince(server.stepSince))
                Text(seconds < 3 ? "\(step)…" : "\(step) · \(seconds)s").monospacedDigit()
            }
            .help(server.step == .warming ? "Running the model once so that its first answer comes quickly. The first start after an update also prepares its GPU programs, which can take a minute." : "")
        case .ready where !server.recovering:
            HStack(spacing: 4) {
                Text(verbatim: "\(server.serving.rawValue) · Port \(server.settings.port)")
                    .help(server.asleep ? "It freed its memory while idle and loads again with the next message." : "")
                CopyButton(copy: state.copyAPIAddress)
                    .buttonStyle(.hover)
                    .help("Copy API Address: \(server.settings.url.appending(path: "v1").absoluteString)")
                    .accessibilityLabel("Copy API Address")
            }
        case .failed where othersFailure:
            Text("Not running")
        default:
            Text(state.serverStatus.phrase)
        }
    }

    @ViewBuilder private var actions: some View {
        let server = state.server
        switch server.phase {
        case .stopped, .failed:
            // A server that runs but serves nothing gives way to the model picked.
            let switching = server.running && state.selection != server.model
            Button(failedPicked ? "Try Again" : switching ? "Switch to \(state.selection.map(state.displayName(of:)) ?? "")" : "Start") {
                if let model = state.selection { state.start(model) }
            }
            .buttonStyle(.borderedProminent)
            .disabled(state.selection == nil || !state.ready || state.keyProblem != nil)
        case .starting:
            Button("Cancel") { state.stop() }
        case .stopping:
            if server.next != nil { Button("Cancel") { state.stop() } }
        case .ready:
            if let selection = state.selection, selection != server.model {
                Button("Stop") { state.stop() }
                Button("Switch to \(state.displayName(of: selection))") { state.start(selection) }
                    .buttonStyle(.borderedProminent)
                    .disabled(state.keyProblem != nil)
            } else {
                AgentsPopup()
                Button("Stop") { state.stop() }
                if state.optionsChanged, let model = server.model {
                    Button("Restart to Apply") { state.start(model) }
                        .buttonStyle(.borderedProminent)
                        .disabled(state.keyProblem != nil)
                } else {
                    Button("Open Chat") { state.section = .chat }.buttonStyle(.borderedProminent)
                }
            }
        }
    }
}

/// What the model that runs has done: its figures and its requests. It reads them itself, so that a
/// poll or a request finishing redraws this section alone, not the whole screen.
private struct ActivitySection: View {
    @Environment(AppState.self) private var state
    let advanced: Bool

    var body: some View {
        Section {
            ActivityFigures(activity: state.server.activity, context: state.server.context)
            RequestLog(entries: state.server.requestLog, totals: RequestLog.totals(state.server.activity), open: advanced)
        } header: {
            SectionTitle(text: "Activity")
        }
    }
}

/// How the model that runs does since it started: how fast it answers and reads prompts on average,
/// how soon an answer starts; the context limit it serves, and the memory it holds now.
private struct ActivityFigures: View {
    let activity: Server.Activity
    let context: Int?

    var body: some View {
        let latency = activity.firstToken.map(Format.latency)
        let memory = activity.memory.map { Format.parts(Format.memory($0)) }
        HStack(spacing: 0) {
            Figure(value: activity.generation.map(Format.whole), unit: "tok/s", caption: "Answering",
                   help: "Average since the model started")
            Figure(value: activity.reading.map(Format.whole), unit: "tok/s", caption: "Reading prompts",
                   help: "Average since the model started")
            Figure(value: latency?.0, unit: latency?.1, caption: "To first token", help: "Median since the model started")
            Figure(value: context.map(ContextLength.figure), caption: "Context limit", help: "The tokens a conversation can hold")
            Figure(value: memory?.0, unit: memory?.1, caption: "Memory", help: "What the engine holds in memory now")
        }
        .padding(.vertical, 6)
        .animation(Motion.settle, value: activity)
    }
}

/// The requests since the model started, their totals on the row, and when opened, one line for
/// each in a box of its own that scrolls, from the top down, following the latest once full.
private struct RequestLog: View {
    let entries: [RequestEntry]
    let totals: String?
    @State var open = false

    var body: some View {
        DisclosureRow(title: "Requests", open: $open, detail: totals)
        if open {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if entries.isEmpty {
                        Text("No requests yet.").font(.callout).foregroundStyle(.tertiary).frame(height: 22)
                    }
                    ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                        RequestRow(entry: entry)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
            }
            .frame(height: 180)
            .defaultScrollAnchor(.top, for: .alignment)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
            .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 8))
        }
    }

    /// How many requests, the tokens they wrote and read, those taken from the cache, and any
    /// running; none before the first.
    static func totals(_ activity: Server.Activity) -> String? {
        guard activity.served > 0 || activity.running > 0 else { return nil }
        var parts = [activity.served.formatted(), "\(Format.count(activity.generated)) tokens out", "\(Format.count(activity.read)) in"]
        if activity.cached > 0 { parts.append("\(Format.count(activity.cached)) cached") }
        if activity.running > 0 { parts.append("\(activity.running) running") }
        return parts.joined(separator: " · ")
    }
}

/// A request as a line: when, the tokens it wrote, and how fast, how soon and from how much prompt;
/// or that it was cancelled, or why it failed. The server's own line is a hover away.
private struct RequestRow: View {
    let entry: RequestEntry

    var body: some View {
        HStack(spacing: 8) {
            Text(entry.time).font(.subheadline).foregroundStyle(.tertiary).frame(width: 56, alignment: .leading)
            switch entry.outcome {
            case .done:
                Text("\((entry.output ?? 0).formatted()) tokens")
                Text(facts).foregroundStyle(.secondary)
            case .cancelled:
                Text("Cancelled").foregroundStyle(.secondary)
            case .error:
                Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor).font(.system(size: 11))
                Text(entry.message ?? "").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .monospacedDigit()
        .lineLimit(1)
        .frame(height: 22)
        .help(entry.raw)
        .contextMenu {
            Button("Copy Line") { NSPasteboard.put(entry.raw) }
        }
    }

    private var facts: String {
        [entry.rate.map { "\($0.formatted(.number.precision(.fractionLength(1)))) tok/s" },
         entry.firstToken.map { "first \(Format.seconds($0))" },
         entry.input.map { "\($0.formatted()) in" },
         entry.cached.flatMap { $0 > 0 ? "\($0.formatted()) cached" : nil }]
            .compactMap { $0 }.joined(separator: " · ")
    }
}

/// The coding agents to open in Terminal on the model that runs, and its API address to copy.
private struct AgentsPopup: View {
    @Environment(AppState.self) private var state
    private static let copy = "copy"

    var body: some View {
        PopupMenu(header: "Open in Terminal",
                  items: Agent.allCases.map { PopupItem(id: $0.rawValue, title: "\($0.title)…") }
                      + [PopupItem(id: Self.copy, title: "Copy API Address", separated: true)],
                  chosen: nil, above: true) { choice in
            if let agent = Agent(rawValue: choice) {
                state.open(agent)
            } else {
                state.copyAPIAddress()
            }
        } label: {
            HStack(spacing: 4) {
                Text("Coding Agents")
                MenuChevron()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
        }
        .fixedSize()
        .help("Open a coding agent in Terminal, connected to this model")
    }
}

/// A section's title, the same in every state of the screen.
struct SectionTitle: View {
    let text: String

    var body: some View {
        Text(text).font(.headline).foregroundStyle(.primary).textCase(nil)
    }
}

/// Something that failed: why, in a line or three, with any log and a way out.
private struct ProblemRow: View {
    @Environment(AppState.self) private var state
    let title: String
    let message: String
    /// The log that says more, which Show Log opens in the window's drawer.
    var log: LogSource?
    var dismiss: (() -> Void)?
    /// The way out, with its button's title.
    var action: (title: String, run: () -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(3)
                    .help(message)
            }
            Spacer(minLength: 12)
            if let log {
                Button("Show Log") { state.showLog(log) }
            }
            if let action { Button(action.title, action: action.run) }
            if let dismiss {
                Button(action: dismiss) { Image(systemName: "xmark") }
                    .buttonStyle(.hover)
                    .foregroundStyle(.secondary)
                    .help("Dismiss")
                    .accessibilityLabel("Dismiss")
            }
        }
        .padding(.vertical, 4)
    }
}

private struct DownloadRow: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let installer = state.installer
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Downloading \(installer.model.map(state.displayName(of:)) ?? "")").font(.headline)
                Spacer()
                Button("Cancel") { installer.cancel() }
            }
            if let progress = installer.progress {
                ProgressView(value: progress.fraction)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                Text(caption).font(.callout).foregroundStyle(.secondary).monospacedDigit()
            }
        }
        .padding(.vertical, 4)
    }

    private var caption: String {
        let installer = state.installer
        guard let progress = installer.progress else { return "Preparing download…" }
        let amount = "\(Format.size(progress.done)) of \(Format.size(progress.total))"
        guard let remaining = installer.remaining else { return amount }
        if remaining < .seconds(60) { return "\(amount) — less than a minute left" }
        let time = remaining.formatted(.units(allowed: [.hours, .minutes], width: .wide, maximumUnitCount: 1))
        return "\(amount) — about \(time) left"
    }
}

/// A model on this Mac, to pick for Start: its name, and under it its Hugging Face ID, which tells its
/// variants apart, and its size; what the server does with it and the check of the one picked at
/// the right.
private struct ModelRow: View {
    @Environment(AppState.self) private var state
    let model: InstalledModel

    var body: some View {
        let selected = state.selection == model.id
        let detail = ([model.id, model.missing ? "Not found" : model.bytes.map(Format.size)] as [String?])
            .compactMap { $0 }.joined(separator: " · ")
        Button { state.selection = model.id } label: {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(state.catalog.name(of: model.id))
                    Text(detail)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                if let phase { Text(phase).foregroundStyle(.secondary) }
                Image(systemName: "checkmark")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.tint)
                    .frame(width: 16)
                    .opacity(selected ? 1 : 0)
                    .animation(Motion.quick, value: selected)
            }
        }
        .buttonStyle(.hoverRow)
        .hoverRowAligned()
        .accessibilityLabel("\(state.displayName(of: model.id)), \(detail)")
        .accessibilityValue(phase ?? "")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .help(state.catalog.about(of: model.id) ?? model.id)
        .contextMenu {
            if model.local {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(filePath: model.id)]) }
                Button("Remove from List") { state.removeLocal(model.id) }
            } else {
                CopyModelID(id: model.id)
            }
        }
    }

    /// What the server does with this model, if anything.
    private var phase: String? {
        let server = state.server
        if server.next == model.id && server.model != model.id { return "Up next" }
        guard server.running, server.model == model.id, server.phase != .stopped else { return nil }
        // Switched away from, it says Stopping…: the next model's own row says Up next.
        if server.phase == .stopping, server.next != model.id { return "Stopping…" }
        return state.serverStatus.phrase
    }
}

private struct SuggestedRow: View {
    @Environment(AppState.self) private var state
    let variant: Catalog.Variant
    let recommended: Bool
    let lead: Bool

    var body: some View {
        LabeledContent {
            Button("Download") { state.download(variant.id) }
                .disabled(state.installer.running || !state.ready)
                .help(state.installer.running ? "One download at a time" : "")
                .prominent(lead)
        } label: {
            // The anatomy of a model row.
            VStack(alignment: .leading, spacing: 2) {
                Text(variant.name)
                Text(([recommended ? "Recommended" : nil, variant.about, Format.size(Format.gigabytes(variant.size))]
                    as [String?]).compactMap { $0 }.joined(separator: " · "))
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .help(variant.id)
        .contextMenu { CopyModelID(id: variant.id) }
    }
}

private struct CopyModelID: View {
    let id: String

    var body: some View {
        Button("Copy Model ID") { NSPasteboard.put(id) }
    }
}

/// Downloads any model `splash serve --model` takes.
private struct CustomModelSheet: View {
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss
    @State private var model = ""
    @FocusState private var focused: Bool

    var body: some View {
        Form {
            Section {
                TextField("Model", text: $model, prompt: Text("owner/model or owner/model:variant"))
                    .monospaced()
                    .focused($focused)
                    .onSubmit(download)
            } header: {
                Text("Download Another Model").font(.headline)
            } footer: {
                Text("Any Hugging Face model Splash runs, such as unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M.")
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .onAppear { focused = true }
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            ToolbarItem(placement: .confirmationAction) {
                Button("Download", action: download).disabled(id == nil)
            }
        }
    }

    /// The model typed or pasted, its Hugging Face address too, if `splash serve --model` takes it
    /// (`serve_options.parse_model_id`'s rules).
    private var id: String? {
        let text = model.trimmingCharacters(in: .whitespaces).trimmingPrefix("https://huggingface.co/")
        guard let match = text.wholeMatch(of: /([A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?\/[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?)(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?/),
              !text.contains(".."), !match.1.contains("--"), !match.1.hasSuffix(".git")
        else { return nil }
        return String(text)
    }

    private func download() {
        guard let id else { return }
        state.download(id)
        dismiss()
    }
}

extension View {
    /// The screen's one prominent button, which Return presses.
    @ViewBuilder func prominent(_ on: Bool) -> some View {
        if on { buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction) } else { self }
    }
}
