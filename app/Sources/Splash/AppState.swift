import AppKit
import Foundation
import Observation

/// What the app's window and menus show: the runtime, the server, downloads, the models and the
/// conversations.
@MainActor @Observable
final class AppState {
    /// The window's two parts: the Server, where models download, start and change, which the
    /// button at the toolbar's end opens, and the Chat with the model that runs. The window opens
    /// on the Server.
    enum Section: Hashable { case server, chat }

    /// Where the app keeps what it writes: the logs of its server and downloads, and the chats.
    struct Folders {
        let logs: URL
        let chats: URL

        static let user = Folders(logs: URL.libraryDirectory.appending(path: "Logs/Splash"),
                                  chats: URL.applicationSupportDirectory.appending(path: "Splash/Chats"))
    }

    /// Why the app cannot run models here.
    struct Blocker: Equatable {
        let title: String
        let symbol: String
        let message: String
    }

    let runtime: Runtime
    /// Where the API key is kept.
    let keychain: Keychain
    /// Where the preferences and the records of what the app runs (`ChildRecord`) are kept.
    let defaults: UserDefaults
    let server: Server
    let installer: Installer
    let catalog: Catalog
    let chats: ChatStore
    /// The count of the next prompt, which the composer's ring shows.
    let chatContext = ChatContext()
    /// The message being written, kept while the window shows something else.
    let draft: Draft
    /// The open conversation's page, kept while the window shows something else, so that the Chat
    /// shows again at once.
    let transcript: Transcript
    /// Whether the Chat shows its sidebar, as the toolbar's button and ⌘B have it, from one launch
    /// to the next.
    var showsSidebar: Bool {
        didSet { defaults.set(showsSidebar, forKey: Preference.sidebar) }
    }
    /// The terminals and logs under the window's content, whether they show, as ⌘J has it, and
    /// the height the user dragged them to.
    let drawer: Drawer
    var showsDrawer = false
    var drawerHeight: CGFloat = 280
    /// The models on this Mac to pick from, as last listed (`refresh`): those of a family the runtime
    /// serves (`families`). A model of another, which a newer or development build of Splash installs
    /// into the models folder every Splash on the Mac shares, can only fail to start: it is left
    /// out, as a model whose files are gone is. Model files on this Mac not prepared yet have no
    /// family until a start prepares them, and stay; so does every model until the runtime says
    /// which families it serves.
    var installed: [InstalledModel] {
        guard !families.isEmpty else { return listedModels }
        return listedModels.filter { model in model.family.map { families[$0] != nil } ?? true }
    }
    /// The models on this Mac as last listed, of every family.
    private var listedModels: [InstalledModel] = []
    /// What each model family the runtime serves makes of the launch options, by its name, as the
    /// runtime reports it at launch; none before, or where the report failed: every option then
    /// shows and passes, and every model is offered.
    private(set) var families: [String: ModelFamily] = [:]
    private(set) var blocker: Blocker?
    /// Whether the checks of `launch` are done; nothing starts before.
    private(set) var launched = false
    /// What `launch` waits for while it stops what an earlier run of the app left running, which
    /// can take some seconds; the window says it under its spinner.
    private(set) var launchWait: String?
    /// Whether the window is open, and the menu bar's panel; while neither is, the server's status
    /// is asked less often (`Server.unwatched`). The menus' commands of the window open it where
    /// it is closed (`SplashCommands`).
    var windowShows = true { didSet { server.unwatched = !windowShows && !panelShows } }
    @ObservationIgnored var panelShows = false { didSet { server.unwatched = !windowShows && !panelShows } }
    var section = Section.server
    /// The launch options the next start uses; their API key reaches the Keychain once typed
    /// (`keepKey`).
    var options: ServeSettings {
        didSet { options.save(defaults) }
    }
    /// The API key the Keychain holds, as last read or written.
    @ObservationIgnored private var keptKey: String?
    /// Why the Keychain didn't give or keep the API key; nothing starts until it does, so that a
    /// model never serves without the key it is meant to ask for.
    private(set) var keyProblem: String?
    /// A model downloaded that the listing hasn't found yet, whose suggestion stays away meanwhile.
    private(set) var arriving: String?
    /// Set once the app quits: nothing starts any more.
    private(set) var quitting = false
    /// The installer of the models the Hugging Face cache holds, while it runs (`launch`).
    private var cachedInstaller: ChildProcess?
    /// The listing of the models under way or last done (`refresh`), and the count of listings
    /// started, so that only the latest one counts.
    @ObservationIgnored private var listing: Task<Void, Never>?
    @ObservationIgnored private var listings = 0
    /// The model to pick once the latest listing has it (`refresh`).
    @ObservationIgnored private var picking: String?

    init(runtime: Runtime, folders: Folders = .user, keychain: Keychain = .app, defaults: UserDefaults = .standard,
         shell: @escaping () -> Shell = { .login }) {
        self.runtime = runtime
        self.keychain = keychain
        self.defaults = defaults
        showsSidebar = defaults.object(forKey: Preference.sidebar) as? Bool ?? true
        options = ServeSettings.current(defaults)
        modelFiles = Self.keptModelFiles(defaults)
        drawer = Drawer(shell: shell)
        chats = ChatStore(folder: folders.chats)
        draft = Draft(files: chats.attachments)
        server = Server(runtime: runtime, log: folders.logs.appending(path: "server.log"), defaults: defaults)
        installer = Installer(runtime: runtime, log: folders.logs.appending(path: "install.log"), defaults: defaults)
        catalog = Catalog(contentsOf: runtime.suggestedModels)
        transcript = Transcript(runtime: runtime, attachments: chats.attachments)
        installer.installed = { [weak self] in self?.didInstall($0) }
        installer.changed = { [weak self] in self?.refresh() }
        // The last terminal closed, the drawer showing terminals closes too.
        drawer.emptied = { [weak self] in
            guard let self, drawer.part == .terminal else { return }
            hideDrawer()
        }
        // A start prepares model files on this Mac, which then have a family, and the model that
        // runs stays listed only while it runs.
        server.phaseChanged = { [weak self] in self?.refresh() }
        // Asked from the page, which then has the keyboard: the composer takes it back.
        transcript.askAgain = { [weak self] in
            self?.retry()
            self?.draft.focus()
        }
        readKey()
        refresh()
    }

    /// Whether the app runs on this Mac and has done its checks: the window then shows the Server
    /// and the Chat.
    var available: Bool { launched && blocker == nil }

    /// Whether models can download, and start where the Keychain gives the API key (`canStart`).
    var ready: Bool { available && !quitting }

    /// Whether `model` can start now, or the one that runs give way to it: the app is ready, the
    /// Keychain gives the API key (`keyProblem`), and `model` is not being deleted. `start` asks
    /// it, and so do the Server's buttons, the menu bar's panel and the composer's model menu, of
    /// each model they start.
    func canStart(_ model: String) -> Bool { ready && keyProblem == nil && !removing(model) }

    /// Whether `model` is being deleted.
    func removing(_ model: String) -> Bool {
        installer.running && installer.work == .removal && installer.model == model
    }

    /// Whether `model` is the one the server runs or a switch starts, which is checked or deleted
    /// once stopped.
    func serving(_ model: String) -> Bool {
        server.running && (server.model == model || server.next == model)
    }

    /// Whether the app runs something it must stop or wait for before it quits.
    var busy: Bool { server.running || installer.running || cachedInstaller != nil }

    /// Whether the models the Hugging Face cache holds are still being looked for.
    var findingCachedModels: Bool { cachedInstaller != nil }

    /// What the server does: the look of its mark, the phrase most views say of the model that
    /// runs, and why a start's step can take long, which views say under it where there is room.
    struct ServerStatus {
        let look: MarkLook
        let phrase: String
        var note: String?
    }

    var serverStatus: ServerStatus {
        switch server.phase {
        case .stopped:
            return ServerStatus(look: .off, phrase: "Not running")
        case .starting:
            return ServerStatus(look: .loading, phrase: server.stepPhrase, note: server.step.note)
        case .stopping:
            guard let next = server.next else { return ServerStatus(look: .off, phrase: "Stopping…") }
            return ServerStatus(look: .off, phrase: next == server.model ? "Restarting…" : "Switching to \(displayName(of: next))…")
        case .ready:
            let look: MarkLook = switch server.serving {
            case .running: .on
            case .sleeping: .asleep
            case .recovering: .loading
            }
            return ServerStatus(look: look, phrase: server.serving.rawValue)
        case .failed:
            return ServerStatus(look: .failed, phrase: "Ran into a problem")
        }
    }

    /// What runs, in a sentence: VoiceOver announces it, and the Chat's composer says it while a
    /// model starts, stops or fails.
    var status: String {
        guard server.phase != .stopped, let model = server.model else { return "No model running" }
        return "\(displayName(of: model)) · \(serverStatus.phrase)"
    }

    /// What the installer does, a download, a check or a deletion, in the words the menus use.
    var downloadStatus: String? {
        guard let model = installer.model else { return nil }
        let name = displayName(of: model)
        guard let progress = installer.progress else { return "\(installer.work.phrase(name, downloading: false))…" }
        return "\(installer.work.phrase(name, downloading: true)) · \(progress.fraction.formatted(.percent.precision(.fractionLength(0))))"
    }

    /// The model picked on the Server screen: the last one picked while it is listed, else the
    /// first one listed.
    var selection: String? {
        get {
            access(keyPath: \.selection)
            let picked = defaults.string(forKey: Preference.model)
            return picked.flatMap { isInstalled($0) ? $0 : nil } ?? installed.first?.id
        }
        set { withMutation(keyPath: \.selection) { defaults.set(newValue, forKey: Preference.model) } }
    }

    /// Whether the model that runs would start otherwise with the options set now.
    var optionsChanged: Bool {
        guard server.phase == .ready, let model = server.model else { return false }
        return server.settings.differs(from: settings(for: model), model: model, family: family(of: model))
    }

    /// What the family of `model`, a listed one, makes of the launch options; unknown for model
    /// files on this Mac not prepared yet, or before the runtime has reported its families.
    func family(of model: String?) -> ModelFamily {
        installed.first { $0.id == model }?.family.flatMap { families[$0] } ?? .unknown
    }

    /// The name to show for model `id`; its variant tells installed variants of a model apart.
    func displayName(of id: String) -> String {
        let name = catalog.name(of: id)
        guard installed.filter({ catalog.name(of: $0.id) == name }).count > 1 else { return name }
        return "\(name) (\(Catalog.variant(of: id)))"
    }

    /// Checks this Mac and the app's location, waits for the list of the models on it and for the
    /// families the runtime serves, which pick those it offers, stops what the app left running last
    /// time, and then, in the background, installs the models the Hugging Face cache holds.
    func launch() async {
        defer { launched = true }
        // The shell's Hugging Face settings, read beside the checks, for all the app runs after them.
        let shellSettings = Task { await Runtime.readShellSettings(drawer.shell()) }
        if let reason = await runtime.unsupportedReason() {
            blocker = Blocker(title: "This Mac Can’t Run Splash", symbol: "exclamationmark.triangle", message: reason)
            return
        }
        if Self.mustMoveToApplications(Bundle.main.bundleURL) {
            blocker = Blocker(title: "Move Splash to Applications", symbol: "externaldrive",
                              message: "Splash can’t run from the disk image. Drag it to your Applications folder, then open it from there.")
            return
        }
        chats.adoptFiles()
        runtime.shellSettings.set(await shellSettings.value)
        // Which families the runtime serves, read beside the listing and the stops, and known before
        // the window shows the models, so that it lists from the first only those it offers.
        let served = Task { await runtime.families() }
        await listed()
        await installer.stopLeftover { self.launchWait = "Stopping the download Splash left running…" }
        await server.stopLeftover { self.launchWait = "Stopping the model Splash left running…" }
        families = await served.value
        launchWait = nil
        installCached()
    }

    /// Installs, without downloading, the models other tools or a source checkout of Splash left
    /// in the Hugging Face cache that Splash serves (`install/cached.py`), then lists them.
    private func installCached() {
        cachedInstaller = try? ChildProcess(command: runtime.cachedInstallCommand, environment: runtime.environment,
                                            log: URL(filePath: "/dev/null")) { [weak self] _ in
            self?.cachedInstaller = nil
            self?.refresh()
        }
    }

    /// Reads the API key kept; a Keychain that doesn't give it is a problem to solve first.
    func readKey() {
        do {
            keptKey = try keychain.read()
            keyProblem = nil
            options.apiKey = keptKey
        } catch {
            keyProblem = error.localizedDescription
        }
    }

    /// Keeps the options' API key in the Keychain where it differs from the one kept: as its field
    /// is left, as a start uses it and as the app quits, rather than at each key typed, which could
    /// have the Keychain ask each time.
    func keepKey() {
        if options.apiKey != keptKey { keep(options.apiKey) }
    }

    /// Tries again what the Keychain refused: keeping the key typed, where it differs from the one
    /// kept, or else reading the key kept.
    func retryKey() {
        if options.apiKey != keptKey { keep(options.apiKey) } else { readKey() }
    }

    private func keep(_ key: String?) {
        do {
            try keychain.write(key)
            // The Keychain holds the key the options have.
            keptKey = key
            keyProblem = nil
        } catch {
            keyProblem = error.localizedDescription
        }
    }

    /// Starts `model` with the launch options, stopping the one that runs first, where it can start
    /// now (`canStart`).
    func start(_ model: String) {
        keepKey()
        guard canStart(model) else { return }
        selection = model
        chats.stop()
        server.start(model, settings: settings(for: model))
    }

    func stop() {
        chats.stop()
        server.stop()
    }

    /// Starts `model` again with the launch options its failure asks for, or without the projector
    /// or draft chosen for it, or takes it off the list (`Server.remedy`).
    func retry(_ model: String, with remedy: Server.Remedy) {
        switch remedy {
        case .memoryLimit(let limit): options["--max-memory"] = limit
        case .autoContext: options["--max-context"] = nil
        case .without(let projector, let draft):
            if projector { modelFiles[model]?.projector = nil }
            if draft { modelFiles[model]?.draft = nil }
        case .removeFromList:
            server.dismissFailure()
            return removeLocal(model)
        }
        start(model)
    }

    /// The model Stop names, while a server runs: the one a switch goes to, else the one that runs;
    /// and whether Stop does anything now, which it doesn't once a stop with no switch is under way.
    var stopItem: (model: String, enabled: Bool)? {
        guard server.running, let model = server.next ?? server.model else { return nil }
        return (model, !(server.phase == .stopping && server.next == nil))
    }

    /// The model the menu bar's panel and the composer's model menu check: the one a switch goes
    /// to, else the one that starts or serves; none while it stops or once it failed, so that
    /// choosing it starts it again.
    var checkedModel: String? {
        if let next = server.next { return next }
        return server.phase == .starting || server.phase == .ready ? server.model : nil
    }

    /// What choosing `model` in the menu bar's panel or the composer does: starts it, or switches to
    /// it from the one that runs; the one checked goes on as it is.
    func choose(_ model: String) {
        guard model != checkedModel else { return }
        start(model)
    }

    func download(_ model: String) {
        guard ready else { return }
        installer.install(model)
    }

    /// What quitting now would cut off: the requests the model answers, as its last status said
    /// (`Server.Activity.running`), or the Chat's reply as it streams; a download, which starts
    /// over next time; and the programs the drawer's terminals run. A model that idles, sleeps,
    /// starts or switches is no loss: quitting stops it, as it always does.
    struct QuitCost {
        var model: String?
        var requests = 0
        var download: String?
        var programs = 0

        /// A line for each loss, for the question a quit asks; nil where there is none, and the
        /// app quits at once.
        var message: String? {
            var lines: [String] = []
            if requests > 0, let model {
                lines.append("\(model) is answering \(requests == 1 ? "a request" : "\(requests) requests").")
            }
            if let download {
                lines.append("\(download) is still downloading. If you quit, the download starts over next time.")
            }
            if programs > 0 {
                lines.append(programs == 1 ? "A terminal is running a program." : "\(programs) terminals are running programs.")
            }
            return lines.isEmpty ? nil : lines.joined(separator: "\n\n")
        }
    }

    var quitCost: QuitCost {
        // The status is asked as seldom as every 10 s; the requests the server's log has started
        // and not ended count at once.
        let live = server.requests.count(where: \.running)
        let requests = server.phase == .ready ? max(server.activity.running, live, chats.streaming == nil ? 0 : 1) : 0
        // A download, or a check's update, starts over; a check that has found nothing loses nothing,
        // and a deletion, quick, ends first.
        let downloading = installer.running && (installer.work == .download || installer.progress != nil)
        return QuitCost(model: server.model.map(displayName(of:)), requests: requests,
                        download: downloading ? installer.model.map(displayName(of:)) : nil, programs: drawer.runningPrograms)
    }

    /// Stops the download, the model and the installer of cached models, which the app waits for
    /// as it quits, and the drawer's terminals, and starts nothing after. The installer of cached
    /// models gets SIGINT, as Ctrl+C sends, and SIGKILL 5 s later.
    func quit() {
        quitting = true
        chats.stop()
        drawer.endAll()
        // A deletion cut short would leave files no one uses: the quit waits for it.
        if installer.work != .removal { installer.cancel() }
        server.stop()
        cachedInstaller?.stop(SIGINT, killAfter: .seconds(5))
    }

    /// The next prompt, the open conversation and the draft, for the composer to count; nil while
    /// a reply streams, there is nothing to count, or the draft can't go: a count would read and
    /// upload what the server refuses, at every pause in the typing.
    var contextRequest: ChatContext.Request? {
        guard let target, let limit = target.context, limit > 0, chats.streaming == nil, attachmentProblem == nil else { return nil }
        let history = chats.current?.messages ?? []
        let text = draft.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !history.isEmpty || !text.isEmpty || !draft.attachments.isEmpty else { return nil }
        return ChatContext.Request(history: history, text: text, attachments: draft.attachments, target: target,
                                   started: server.since, files: chats.attachments, conversation: chats.selection)
    }

    /// Whether the composer shows its ring: a model with a context runs, and the open chat has a
    /// prompt to count. Whether it is counted yet never decides it, so that the ring stays as
    /// chats open and their prompts are counted, and only its figure changes.
    var showsContext: Bool {
        guard let limit = target?.context, limit > 0 else { return false }
        return !(chats.current?.messages ?? []).isEmpty || !draft.isEmpty
    }

    /// The count the composer's ring shows: the next prompt's, or the last one of the open
    /// conversation while the next is counted or a reply streams; nil, an empty ring, for a chat
    /// not counted yet.
    var contextUsage: ContextUsage? {
        guard showsContext, let target else { return nil }
        return chatContext.shown(in: chats.selection, history: chats.current?.messages ?? [], target: target,
                                 started: server.since)
    }

    /// Whether the draft can go: its images are all attached, a model answers, no reply streams,
    /// and the model's context can hold the prompt, as far as it has been counted. A count not
    /// done, or failed, holds nothing back: the server checks the prompt too, and says so if it is
    /// too long.
    var canSendDraft: Bool {
        guard !draft.isEmpty, draft.attaching == 0, server.answers, chats.streaming == nil,
              attachmentProblem == nil else { return false }
        return overflow == nil
    }

    /// Why the draft's attachments can't go to the model that runs (`attachmentProblem(_:vision:limits:)`).
    var attachmentProblem: String? {
        Self.attachmentProblem(draft.attachments, vision: server.acceptsImages, limits: server.limits)
    }

    /// Why `attachments` can't go to a model with `vision` and `limits`: one it can't take, a PDF
    /// over its bytes or pages, or more than the request takes of them together; nil when they can
    /// go, or the model doesn't say its limits.
    nonisolated static func attachmentProblem(_ attachments: [Attachment], vision: Bool, limits: Server.Limits?) -> String? {
        guard !attachments.isEmpty else { return nil }
        let pdfs = attachments.filter { $0.kind == .pdf }
        guard vision else { return pdfs.isEmpty ? "This model can’t see images" : "This model can’t read PDFs" }
        guard let limits else { return nil }
        if let pdf = pdfs.first(where: { $0.bytes > limits.pdf_bytes }) {
            return "\(pdf.name ?? "This PDF") is larger than the model’s \(Format.size(Int64(limits.pdf_bytes))) limit"
        }
        if let pdf = pdfs.first(where: { ($0.pages ?? 0) > limits.pdf_pages }) {
            return "\(pdf.name ?? "This PDF") has more than the model’s \(limits.pdf_pages) pages"
        }
        if pdfs.reduce(0, { $0 + $1.bytes }) > limits.pdf_bytes {
            return "These PDFs are larger together than the model’s \(Format.size(Int64(limits.pdf_bytes))) limit"
        }
        if attachments.contains(where: { $0.kind == .image && $0.bytes > limits.image_bytes }) {
            return "An image is larger than the model’s \(Format.size(Int64(limits.image_bytes))) limit"
        }
        if attachments.reduce(0, { $0 + ($1.kind == .pdf ? $1.pages ?? 1 : 1) }) > limits.images {
            return "The images and PDF pages are more than the model’s \(limits.images)"
        }
        // Sent as base64, a third larger.
        if (attachments.reduce(0, { $0 + $1.bytes }) + 2) / 3 * 4 > limits.request_bytes {
            return "The attachments are larger than one message to the model takes"
        }
        return nil
    }

    /// The count of the next prompt when the model's context cannot hold it; nil while it fits or
    /// is not yet counted.
    var overflow: ContextUsage? {
        contextRequest.flatMap(chatContext.usage(for:)).flatMap { $0.fits ? nil : $0 }
    }

    /// Sends the draft to the running model, leaving the composer empty.
    func sendDraft() {
        guard canSendDraft, let target else { return }
        let message = draft.take()
        chats.send(message.text, attachments: message.attachments, to: target) { [weak self] refused in
            self?.draft.restore(refused)
        }
    }

    /// Asks the running model again for the open conversation's last answer.
    func retry() {
        guard let target else { return }
        chats.retry(with: target) { [weak self] refused in self?.draft.restore(refused) }
    }

    private var target: ChatTarget? {
        guard server.phase == .ready, let model = server.model else { return nil }
        return ChatTarget(url: server.settings.url, model: model, acceptsImages: server.acceptsImages, effort: effort,
                          context: server.context, apiKey: server.settings.apiKey)
    }

    /// The reasoning effort messages to the model that runs ask for, as chosen for it; empty for
    /// its default.
    var effort: String {
        get {
            access(keyPath: \.effort)
            let efforts = defaults.dictionary(forKey: Preference.efforts) as? [String: String]
            return server.model.flatMap { efforts?[$0] } ?? ""
        }
        set {
            guard let model = server.model else { return }
            withMutation(keyPath: \.effort) {
                var efforts = defaults.dictionary(forKey: Preference.efforts) as? [String: String] ?? [:]
                efforts[model] = newValue.isEmpty ? nil : newValue
                defaults.set(efforts, forKey: Preference.efforts)
            }
        }
    }

    /// The folder the coding agents work in, the last one chosen; none before the first.
    var agentFolder: URL? {
        get {
            access(keyPath: \.agentFolder)
            return defaults.url(forKey: Preference.agentFolder)
        }
        set { withMutation(keyPath: \.agentFolder) { defaults.set(newValue, forKey: Preference.agentFolder) } }
    }

    /// Asks for the folder the coding agents work in, from the last one; whether one was chosen.
    /// `missing` says the last one is gone.
    @discardableResult
    func chooseAgentFolder(missing: Bool = false) -> Bool {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Choose"
        let gone = missing ? agentFolder.map { "“\($0.lastPathComponent)” can’t be found. " } ?? "" : ""
        panel.message = gone + "Choose the project folder the coding agents work in."
        panel.directoryURL = agentFolder
        NSApp.activate()
        guard panel.runModal() == .OK, let folder = panel.url else { return false }
        agentFolder = folder
        return true
    }

    /// Opens `agent` on the model that serves, in a new terminal of the window's drawer, in the
    /// agents' folder, which it asks for the first time; that terminal holds the server's API key.
    func open(_ agent: Agent) {
        // A folder renamed, deleted or on a disk no longer there is asked for again: the shell
        // would start in `/`.
        let exists = agentFolder.map { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true } ?? false
        guard exists || chooseAgentFolder(missing: agentFolder != nil), let folder = agentFolder else { return }
        var environment = runtime.userEnvironment
        environment["SPLASH_API_KEY"] = server.settings.apiKey
        drawer.add(TerminalSession(shell: drawer.shell(), environment: environment, folder: folder,
                                   typing: agent.command(runtime: runtime, port: server.settings.port)))
        showDrawer(.terminal)
        NSApp.activate()
    }

    func copyAPIAddress() {
        NSPasteboard.put(apiAddress)
    }

    /// The model's API address, as Copy API Address gives it.
    var apiAddress: String {
        let settings = server.running ? server.settings : options
        return settings.address().appending(path: "v1").absoluteString
    }

    /// Lists the models installed here, the suggested ones first in the list's order, then the model
    /// files on this Mac the user added, measured, with the family of their preparation once
    /// prepared; the model that runs stays listed as it was while it runs, its files gone or taken
    /// off the list. It reads them off the main thread, as the Hugging Face cache may be on another
    /// volume and a folder picked large, and the list stays as it was until it is done; only the
    /// latest listing counts. `picking` is picked once that listing is done, as the pick of a model
    /// not listed falls to the first one listed (`selection`).
    @discardableResult
    func refresh(picking model: String? = nil) -> Task<Void, Never> {
        let runtime = runtime
        let paths = localModels
        if let model { picking = model }
        listings += 1
        let count = listings
        let task = Task {
            let (models, prepared, local) = await Task.detached {
                (runtime.installedModels(), runtime.preparedFamilies(), paths.map(InstalledModel.local))
            }.value
            guard count == listings else { return }
            let found = models + local.map { model in
                var model = model
                model.family = prepared[model.id]
                return model
            }
            let all = found + listedModels.filter { model in
                server.running && server.model == model.id && !found.contains { $0.id == model.id }
            }
            let rank = { (id: String) in self.catalog.variants.firstIndex { $0.id == id } ?? Int.max }
            let listed = all.filter { !$0.local }.sorted { (rank($0.id), $0.id) < (rank($1.id), $1.id) }
                + all.filter(\.local)
            if listed != listedModels { listedModels = listed }
            if let arriving, listed.contains(where: { $0.id == arriving }) { self.arriving = nil }
            if let picked = picking {
                picking = nil
                selection = picked
            }
        }
        listing = task
        return task
    }

    /// Waits for the latest listing of the models (`refresh`) to be done.
    func listed() async {
        while let task = listing {
            await task.value
            if task == listing { return }
        }
    }

    private var localModels: [String] {
        get { defaults.stringArray(forKey: Preference.localModels) ?? [] }
        set { defaults.set(newValue, forKey: Preference.localModels) }
    }

    /// Adds model files on this Mac, a GGUF file or a model's folder, and picks them: Start has
    /// Splash prepare them, and says why if it cannot serve them.
    func addLocal(_ url: URL) {
        if !localModels.contains(url.path) { localModels.append(url.path) }
        refresh(picking: url.path)
    }

    /// Takes model files off the list, and what they started with; the files stay where they are.
    func removeLocal(_ path: String) {
        localModels.removeAll { $0 == path }
        modelFiles[path] = nil
        refresh()
    }

    /// What model files on this Mac start with besides: a GGUF's vision projector, without which
    /// it serves text alone, and a draft in place of its family's released one.
    struct ModelFiles: Equatable {
        var projector: String?
        var draft: String?
    }

    /// What each of the model files on this Mac starts with besides, by its path, kept as they
    /// are set; read once, as the rows that show them render often.
    var modelFiles: [String: ModelFiles] = [:] {
        didSet {
            let kept = modelFiles.filter { $0.value != ModelFiles() }
                .mapValues { ["projector": $0.projector, "draft": $0.draft].compactMapValues { $0 } }
            defaults.set(kept, forKey: Preference.localModelFiles)
        }
    }

    /// The model files' projector and draft `defaults` keep.
    private static func keptModelFiles(_ defaults: UserDefaults) -> [String: ModelFiles] {
        let kept = defaults.dictionary(forKey: Preference.localModelFiles) as? [String: [String: String]] ?? [:]
        return kept.mapValues { ModelFiles(projector: $0["projector"], draft: $0["draft"]) }
    }

    /// The settings `model` starts with: the launch options it takes, and, for model files on this
    /// Mac, their own projector and draft.
    func settings(for model: String) -> ServeSettings {
        var settings = options.usable(by: family(of: model))
        let files = modelFiles[model]
        settings.projector = files?.projector
        settings.draft = files?.draft
        return settings
    }

    /// Whether `model` can be deleted now: not the model that runs, which is deleted once it is
    /// stopped, nor during a download, while a start prepares its model's files, or while the
    /// Hugging Face cache is looked through for models.
    func canDelete(_ model: String) -> Bool {
        ready && !installer.running && !serving(model) && server.phase != .starting && server.next == nil
            && !findingCachedModels
    }

    /// Deletes `model`, a model Splash installed, with the files of it nothing else uses.
    func delete(_ model: String) {
        guard canDelete(model) else { return }
        installer.remove(model)
    }

    /// Asks the Hub whether `model` or its draft has moved on, and installs what has, as its
    /// download did: the starts skip that question (`--offline`).
    func checkForUpdates(_ model: String) {
        guard ready, !serving(model) else { return }
        installer.install(model, .check)
    }

    func isInstalled(_ model: String) -> Bool {
        installed.contains { $0.id == model }
    }

    /// Picks the model downloaded, for the user to start, its suggestion away until it is listed.
    private func didInstall(_ model: String) {
        arriving = model
        refresh(picking: model)
        if NSApp?.isActive == false { NSApp.requestUserAttention(.informationalRequest) }
    }

    /// Whether the app at `bundle` runs from the disk image or a translocated copy: its path then
    /// changes on every launch and its volume may go away while a model runs.
    static func mustMoveToApplications(_ bundle: URL) -> Bool {
        guard bundle.pathExtension == "app" else { return false }
        return bundle.path.contains("/AppTranslocation/")
            || (try? bundle.resourceValues(forKeys: [.volumeIsReadOnlyKey]).volumeIsReadOnly) == true
    }
}
