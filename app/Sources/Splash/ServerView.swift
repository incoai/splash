import SwiftUI
import UniformTypeIdentifiers

/// The width of the Server screen's column.
private let column: CGFloat = 640

/// The Server screen, where the window opens: the models on this Mac to pick from, how they
/// start, the suggested ones this Mac can run to download, and, under them all, what runs with
/// Start, Stop and Switch. Models start and change here, from the menu bar's panel
/// (`MenuBarPanel`) and from the composer (`EngineSlot`), with the same launch options and by the
/// same rule (`AppState.canStart`).
struct ServerView: View {
    @Environment(AppState.self) private var state
    @State private var customModel = false
    /// For pictures of the screen: the launch options' Advanced group and the request log open.
    var advanced = false

    var body: some View {
        GroupedPage(column: column) {
            ServerSections(advanced: advanced, customModel: $customModel)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if !state.installed.isEmpty || state.server.running {
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
}

/// The Server screen's sections, in the hierarchy of its page (`GroupedPage`), where they change
/// shape as one: rows and sections fade in and out as the models, downloads and options change, the
/// rows after them moving along; as the server's stage changes too, as the rest of the window does
/// (`MainView`).
private struct ServerSections: View {
    @Environment(AppState.self) private var state
    let advanced: Bool
    /// Whether the sheet for another model from Hugging Face shows.
    @Binding var customModel: Bool

    var body: some View {
        let suggestions = state.catalog.suggestions(memory: Catalog.memory)
        let installedNames = Set(state.installed.map { state.catalog.name(of: $0.id) })
        let available = suggestions.filter {
            !installedNames.contains($0.name) && $0.id != state.installer.model && $0.id != state.arriving
        }
        let first = state.installed.isEmpty
        let failed = if case .failed = state.server.phase { true } else { false }
        let warming = state.server.phase == .starting && state.server.step == .warming
        Group {
            if state.server.phase == .ready {
                ActivitySection(advanced: advanced)
            }
            if first || failed || warming || state.installer.failure != nil || state.installer.finished != nil
                || state.installer.running || state.keyProblem != nil {
                GroupedSection {
                    if let problem = state.keyProblem {
                        NoticeRow(title: "Splash can’t use its API key",
                                  message: "The Keychain says: \(problem) Allow Splash to use it, then try again.",
                                  action: ("Try Again", state.retryKey))
                    }
                    if case .failed(let message) = state.server.phase, let model = state.server.model {
                        // What its failure says would help: a start that would fit, or, for files on
                        // this Mac Splash can't serve, the start without the projector or draft
                        // chosen for them, else taking them off the list.
                        NoticeRow(title: "\(state.displayName(of: model)) ran into a problem", message: message,
                                  log: .server, dismiss: state.server.dismissFailure,
                                  action: state.server.remedy.map { remedy in (remedy.title, { state.retry(model, with: remedy) }) })
                    }
                    if warming { WarmupRow() }
                    if let failure = state.installer.failure {
                        let name = state.displayName(of: failure.model)
                        switch state.installer.work {
                        case .download:
                            NoticeRow(title: "\(name) couldn’t download", message: failure.message, log: .downloads,
                                      dismiss: state.installer.dismissFailure,
                                      action: ("Download Again", { state.download(failure.model) }))
                        case .check:
                            NoticeRow(title: "Splash couldn’t check \(name) for updates", message: failure.message,
                                      log: .downloads, dismiss: state.installer.dismissFailure,
                                      action: ("Check Again", { state.checkForUpdates(failure.model) }))
                        case .removal:
                            NoticeRow(title: "Splash couldn’t delete \(name)", message: failure.message, log: .downloads,
                                      dismiss: state.installer.dismissFailure)
                        }
                    }
                    if let finished = state.installer.finished {
                        FinishedRow(model: finished.model, result: finished.result)
                    }
                    if state.installer.running { DownloadRow() }
                } header: {
                    if first { welcome }
                }
            }
            if !first {
                GroupedSection {
                    ForEach(state.installed) { ModelRow(model: $0) }
                } header: {
                    SectionTitle(text: "Models on This Mac")
                } footer: {
                    otherModels
                }
                LaunchOptions(advanced: advanced)
            }
            GroupedSection {
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
        .transition(Motion.comingAndGoing)
        .animation(Motion.settle, value: Layout(installed: state.installed.map(\.id), downloading: state.installer.running,
                                                warming: warming, failure: state.installer.failure?.message,
                                                finished: state.installer.finished != nil,
                                                keyProblem: state.keyProblem, finding: state.findingCachedModels,
                                                options: state.options, family: state.family(of: state.selection)))
        .animation(Motion.settle, value: state.server.stage)
    }

    /// What the screen holds besides the server's stage: the models, a download, a start's warm-up,
    /// a problem or what a check or a deletion did, and the launch options, those the family of the
    /// model picked uses; the model picked is not in it, so that picking one of the same family is
    /// immediate.
    private struct Layout: Equatable {
        var installed: [String]
        var downloading: Bool
        var warming: Bool
        var failure: String?
        var finished: Bool
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
        // Its words and buttons change at once, as a control's title does: cross-faded as the
        // server's stage changed the window (`MainView`), the old ones drew over the new.
        .transaction { $0.animation = nil }
    }

    /// Whether the model picked is the one that failed, so that Start tries it again.
    private var failedPicked: Bool {
        if case .failed = state.server.phase { state.server.model == state.selection } else { false }
    }

    /// Whether the server's failure is of another model than the one the bar names, which then
    /// shows as not running.
    private var othersFailure: Bool {
        if case .failed = state.server.phase { !failedPicked } else { false }
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
            .help(server.step.note ?? "")
        case .ready where !server.recovering:
            HStack(spacing: 4) {
                Text(verbatim: "\(server.serving.rawValue) · Port \(server.settings.port)")
                    .help(server.asleep ? "It freed its memory while idle and loads again with the next message." : "")
                CopyButton(copy: state.copyAPIAddress)
                    .buttonStyle(.hover)
                    .help("Copy API Address: \(state.apiAddress)")
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
            Button { if let model = state.selection { state.start(model) } } label: {
                Text(failedPicked ? "Try Again" : "Start").frame(maxWidth: 240)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!(state.selection.map(state.canStart) ?? false))
        case .starting:
            LateButton(title: "Cancel") { state.stop() }
        case .stopping:
            if server.next != nil { LateButton(title: "Cancel") { state.stop() } }
        case .ready:
            if let selection = state.selection, selection != server.model {
                Button("Stop") { state.stop() }
                Button { state.start(selection) } label: {
                    Text("Switch to \(state.displayName(of: selection))").frame(maxWidth: 240)
                }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canStart(selection))
            } else {
                AgentsPopup()
                Button("Stop") { state.stop() }
                if state.optionsChanged, let model = server.model {
                    Button("Restart to Apply") { state.start(model) }
                        .buttonStyle(.borderedProminent)
                        .disabled(!state.canStart(model))
                } else {
                    Button("Open Chat") { state.section = .chat }.buttonStyle(.borderedProminent)
                }
            }
        }
    }
}

/// A button that takes clicks half a second after it shows: one that takes the place of the button
/// just clicked would take the second click of a double click.
private struct LateButton: View {
    let title: String
    let action: () -> Void
    @State private var armed = false

    var body: some View {
        Button(title) { if armed { action() } }
            .task {
                try? await Task.sleep(for: .milliseconds(500))
                armed = true
            }
    }
}

/// What the model that runs has done: its figures and its requests. It reads them itself, so that a
/// poll or a request finishing redraws this section alone, not the whole screen.
private struct ActivitySection: View {
    @Environment(AppState.self) private var state
    let advanced: Bool

    var body: some View {
        GroupedSection {
            ActivityFigures(activity: state.server.activity, context: state.server.context)
            RequestLog(requests: state.server.requests, totals: RequestLog.totals(state.server.activity, requests: state.server.requests),
                       open: advanced)
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

/// The requests since the model started, their totals on the row, and when opened, a line for
/// each in a box of its own that scrolls, from the top down, following the latest once full.
struct RequestLog: View {
    let requests: [ServedRequest]
    let totals: String?
    @State var open = false

    var body: some View {
        DisclosureRow(title: "Requests", open: $open, detail: totals)
        if open {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if requests.isEmpty {
                        Text("No requests yet.").font(.callout).foregroundStyle(.tertiary).frame(height: 22)
                    }
                    ForEach(requests) { request in
                        RequestRow(request: request)
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

    /// How many requests the model answered, as the status counts them and the menu bar's panel
    /// says, the tokens they wrote and read, those taken from the cache, and the rows of `requests`
    /// running; none before the first. The rows are no count of the requests answered: they keep
    /// the latest 1,000 (`ServedRequest.kept`), and those refused too.
    static func totals(_ activity: Server.Activity, requests: [ServedRequest]) -> String? {
        guard activity.served > 0 || !requests.isEmpty else { return nil }
        let running = requests.count(where: \.running)
        var parts = [activity.served.formatted(), "\(Format.count(activity.generated)) tokens out", "\(Format.count(activity.read)) in"]
        if activity.cached > 0 { parts.append("\(Format.count(activity.cached)) cached") }
        if running > 0 { parts.append("\(running) running") }
        return parts.joined(separator: " · ")
    }
}

/// A request as a line: when it came; while it runs, a spinner and how far it has got, reading
/// its prompt or answering, or why it waits; once it ended, the tokens it wrote, how fast, how soon
/// and from how much prompt, or that it was cancelled, or why it failed, the server's line a hover
/// away, or that it ended, where that line never came.
private struct RequestRow: View {
    let request: ServedRequest

    var body: some View {
        HStack(spacing: 8) {
            Text(request.time).font(.subheadline).foregroundStyle(.tertiary).frame(width: 56, alignment: .leading)
            if let end = request.end {
                switch end.outcome {
                case .cancelled:
                    Text("Cancelled").foregroundStyle(.secondary)
                case .error:
                    Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor).font(.system(size: 11))
                    Text(end.message ?? "").foregroundStyle(.secondary)
                case .done, .started:
                    Text(tokens(end.output ?? 0))
                    Text(Self.facts(end)).foregroundStyle(.secondary)
                }
            } else if request.gone {
                Text("Ended").foregroundStyle(.secondary)
            } else {
                let input = request.input ?? 0
                ProgressView().controlSize(.mini)
                switch request.stage {
                case .waiting(let why):
                    Text(why)
                    Text("\(input.formatted()) in").foregroundStyle(.secondary)
                case .reading(let read):
                    Text("Reading \(Int(Double(read) / Double(max(input, 1)) * 100))%")
                    Text("\(read.formatted()) of \(input.formatted()) in").foregroundStyle(.secondary)
                case .answering(let written):
                    Text(tokens(written))
                    Text("Answering · \(input.formatted()) in").foregroundStyle(.secondary)
                }
            }
        }
        .font(.callout)
        .monospacedDigit()
        .lineLimit(1)
        .frame(height: 22)
        .help(request.end?.raw ?? "")
        .contextMenu {
            if let line = request.end?.raw {
                Button("Copy Line") { NSPasteboard.put(line) }
            }
        }
    }

    private static func facts(_ end: RequestEntry) -> String {
        [end.rate.map { "\($0.formatted(.number.precision(.fractionLength(1)))) tok/s" },
         end.firstToken.map { "first \(Format.seconds($0))" },
         end.input.map { "\($0.formatted()) in" },
         end.cached.flatMap { $0 > 0 ? "\($0.formatted()) cached" : nil }]
            .compactMap { $0 }.joined(separator: " · ")
    }
}

/// So many tokens, as a request's line says them.
private func tokens(_ count: Int) -> String {
    "\(count.formatted()) \(count == 1 ? "token" : "tokens")"
}

/// The coding agents to open on the model that runs, in the drawer's terminals, in the folder they
/// work in, which it names; that folder to choose; and the model's API address to copy.
private struct AgentsPopup: View {
    @Environment(AppState.self) private var state
    private static let folder = "folder", copy = "copy"

    var body: some View {
        let folder = state.agentFolder?.lastPathComponent
        PopupMenu(header: folder.map { "Open in \($0)" } ?? "Open in a Terminal",
                  items: Agent.allCases.map { PopupItem(id: $0.rawValue, title: folder == nil ? "\($0.title)…" : $0.title) }
                      + [PopupItem(id: Self.folder, title: "Choose Folder…", separated: true),
                         PopupItem(id: Self.copy, title: "Copy API Address", separated: true)],
                  chosen: nil, above: true) { choice in
            if let agent = Agent(rawValue: choice) {
                state.open(agent)
            } else if choice == Self.folder {
                state.chooseAgentFolder()
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
        .help("Open a coding agent in a terminal of the drawer, connected to this model")
    }
}

/// A section's title, the same in every state of the screen.
struct SectionTitle: View {
    let text: String

    var body: some View {
        Text(text).font(.headline).foregroundStyle(.primary).textCase(nil)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Something that ended: why it failed, or how it went, in a line or three, with any log and a way
/// out.
private struct NoticeRow: View {
    @Environment(AppState.self) private var state
    /// In place of the warning sign, the quieter symbol of something that only says how it went.
    var quietSymbol: String?
    let title: String
    let message: String
    /// The log that says more, which Show Log opens in the window's drawer.
    var log: LogSource?
    var dismiss: (() -> Void)?
    /// The way out, with its button's title.
    var action: (title: String, run: () -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            if let quietSymbol {
                Image(systemName: quietSymbol).foregroundStyle(.secondary)
            } else {
                Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor)
            }
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
                let name = installer.model.map(state.displayName(of:)) ?? ""
                // A check says so until it finds something to download.
                Text(installer.work.phrase(name, downloading: installer.progress != nil))
                    .font(.headline)
                Spacer()
                // A deletion is quick, and cut short it would leave files no one uses.
                if installer.work != .removal { Button("Cancel") { installer.cancel() } }
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
        guard let progress = installer.progress else {
            switch installer.work {
            case .download: return "Preparing download…"
            case .check: return "Asking Hugging Face…"
            case .removal: return "Removing its files…"
            }
        }
        let amount = "\(Format.size(progress.done)) of \(Format.size(progress.total))"
        guard let remaining = installer.remaining else { return amount }
        if remaining < .seconds(60) { return "\(amount) — less than a minute left" }
        let time = remaining.formatted(.units(allowed: [.hours, .minutes], width: .wide, maximumUnitCount: 1))
        return "\(amount) — about \(time) left"
    }
}

/// A start's warm-up: how far it is, as the engine counts its steps (`Server.warmup`), and why it
/// can take long. A step is one of the commands it runs, so the bar moves by them, not by time.
private struct WarmupRow: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let server = state.server
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Warming up \(server.model.map(state.displayName(of:)) ?? "")").font(.headline)
                Spacer()
                if let warmup = server.warmup {
                    Text("\(warmup.steps_done) of \(warmup.steps) steps").foregroundStyle(.secondary).monospacedDigit()
                }
            }
            if let warmup = server.warmup {
                ProgressView(value: Double(warmup.steps_done), total: Double(warmup.steps))
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            Text(server.step.note ?? "").font(.callout).foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

/// A model on this Mac, to pick for Start: its name, and under it its Hugging Face ID, which tells its
/// variants apart, and its size; what the server does with it and the check of the one picked at
/// the right.
private struct ModelRow: View {
    @Environment(AppState.self) private var state
    let model: InstalledModel
    @State private var hovering = false
    /// Whether the question to delete the model asks.
    @State private var deleting = false

    var body: some View {
        let selected = state.selection == model.id
        let files = state.modelFiles[model.id]
        // A projector, which Text only leaves out of the start, says so.
        let projector = files?.projector.map { _ in state.settings(for: model.id).textOnly ? "projector off (Text only)" : "with projector" }
        let detail = ([model.id, model.missing ? "Not found" : model.bytes.map(Format.size), projector,
                       files?.draft.map { _ in "with draft" }] as [String?])
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
                Text(phase ?? "").foregroundStyle(.secondary)
                // Room for the menu's button, which shows under the pointer, so that nothing moves.
                Color.clear.frame(width: 24, height: 24)
                // One check at a time: it shows within `quick` and leaves at once.
                Image(systemName: "checkmark")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.tint)
                    .frame(width: 16)
                    .opacity(selected ? 1 : 0)
                    .animation(selected ? Motion.quick : nil, value: selected)
            }
            // Its words change in place, and the room they take from the name and the ID with
            // them: animated with the page, a phrase that came or grew (Starting…, Warming up ·
            // 3 of 10) slid in over the check, the ID's line, cut anew, morphed off its row over
            // the next one, and a phrase that faded drew over the rows moving past it. The row
            // moves with the page as one (`geometryGroup`), its content laid out in it at once;
            // without the group, its content would leave the row for where it ends up.
            .transaction { $0.animation = nil }
            .geometryGroup()
        }
        // The highlight kept while the pointer is on the menu's button.
        .buttonStyle(.hoverRow(hovered: hovering))
        .overlay(alignment: .trailing) {
            // Before the check.
            if hovering { RowMenuButton(trailing: 8 + 16 + 8) { actions } }
        }
        .onHover { hovering = $0 }
        .hoverRowAligned()
        .accessibilityLabel("\(state.displayName(of: model.id)), \(detail)")
        .accessibilityValue(phase ?? "")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .help(state.catalog.about(of: model.id) ?? model.id)
        .contextMenu { actions }
        // The question goes where its answer no longer could, as a start begun from the menu bar.
        .onChange(of: state.canDelete(model.id)) { _, can in if !can { deleting = false } }
        .confirmationDialog("Delete \(state.displayName(of: model.id))?", isPresented: $deleting) {
            Button("Delete", role: .destructive) { state.delete(model.id) }
        } message: {
            Text("Its files\(model.bytes.map { ", \(Format.size($0))," } ?? "") go from the Hugging Face cache, where the splash "
                + "command and other apps may use them too, unless another model on this Mac uses them.")
        }
    }

    /// What can be done with the model, as its menu's button and a right click on its row give it.
    /// Model files on this Mac: choose a vision projector, a GGUF's alone, and a draft in place of
    /// their family's, show them in the Finder or take them off the list. A model Splash installed:
    /// check it for updates, which its starts don't ask the Hub for, copy its ID or delete it.
    @ViewBuilder private var actions: some View {
        if model.local {
            let files = state.modelFiles[model.id] ?? AppState.ModelFiles()
            if model.gguf {
                Button(files.projector == nil ? "Choose Vision Projector…" : "Change Vision Projector…") { chooseProjector() }
                if files.projector != nil {
                    Button("Remove Vision Projector") { state.modelFiles[model.id, default: files].projector = nil }
                }
            }
            Button(files.draft == nil ? "Choose Draft Model…" : "Change Draft Model…") { chooseDraft() }
            if files.draft != nil {
                Button("Use the Family’s Draft") { state.modelFiles[model.id, default: files].draft = nil }
            }
            Divider()
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(filePath: model.id)]) }
            Button("Remove from List") { state.removeLocal(model.id) }
        } else {
            // One download at a time; the model that runs is checked or deleted once stopped.
            let busy = !state.ready || state.installer.running || state.serving(model.id)
            Button("Check for Updates") { state.checkForUpdates(model.id) }
                .disabled(busy)
            CopyModelID(id: model.id)
            Divider()
            Button("Delete…", role: .destructive) { deleting = true }
                .disabled(!state.canDelete(model.id))
        }
    }

    /// Asks for the GGUF vision projector the model files start with, from their folder.
    private func chooseProjector() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = UTType(filenameExtension: "gguf").map { [$0] } ?? []
        panel.directoryURL = model.id.hasSuffix(".gguf") ? URL(filePath: model.id).deletingLastPathComponent() : URL(filePath: model.id)
        panel.prompt = "Choose"
        panel.message = "Choose the GGUF vision projector of this model, such as mmproj-BF16.gguf."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        state.modelFiles[model.id, default: AppState.ModelFiles()].projector = url.path
    }

    /// Asks for the folder of the DFlash2 draft the model files start with.
    private func chooseDraft() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.prompt = "Choose"
        panel.message = "Choose the folder of a DFlash2 draft for this model."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        state.modelFiles[model.id, default: AppState.ModelFiles()].draft = url.path
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
                .help(state.downloadStatus.map { "One at a time: \($0)" } ?? "")
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

/// What a check for updates or a deletion did, as the installer reported it: the model downloaded
/// what had moved on, which its next start uses, had it all, couldn't be checked or updated, where
/// the log says why, or was deleted, freeing the space its files took unless other models or tools
/// use them.
private struct FinishedRow: View {
    @Environment(AppState.self) private var state
    let model: String
    let result: Installer.Result

    var body: some View {
        let words = Installer.words(result, name: state.displayName(of: model))
        NoticeRow(quietSymbol: words.done ? "checkmark.circle.fill" : "exclamationmark.circle.fill", title: words.title,
                  message: words.detail, log: words.done ? nil : .downloads, dismiss: state.installer.dismissFinished)
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
    /// The screen's one prominent button.
    @ViewBuilder func prominent(_ on: Bool) -> some View {
        // No Return for it: a key pressed in a drawer's terminal or a field must not start a download.
        if on { buttonStyle(.borderedProminent) } else { self }
    }
}
