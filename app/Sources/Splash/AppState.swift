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
    private(set) var installed: [InstalledModel] = []
    /// What each model family makes of the launch options, by its name, as the runtime reports it
    /// at launch; none before, or where the report failed: every option then shows and passes.
    private(set) var families: [String: ModelFamily] = [:]
    private(set) var blocker: Blocker?
    /// Whether the checks of `launch` are done; nothing starts before.
    private(set) var launched = false
    /// What `launch` waits for while it stops what an earlier run of the app left running, which
    /// can take some seconds; the window says it under its spinner.
    private(set) var launchWait: String?
    /// Whether the window is open, and the menu bar's panel; while neither is, the server's status
    /// is asked less often (`Server.unwatched`).
    @ObservationIgnored var windowShows = true { didSet { server.unwatched = !windowShows && !panelShows } }
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
        drawer = Drawer(shell: shell)
        chats = ChatStore(folder: folders.chats)
        draft = Draft(files: chats.attachments)
        server = Server(runtime: runtime, log: folders.logs.appending(path: "server.log"), defaults: defaults)
        installer = Installer(runtime: runtime, log: folders.logs.appending(path: "install.log"), defaults: defaults)
        catalog = Catalog(contentsOf: runtime.suggestedModels)
        transcript = Transcript(runtime: runtime, attachments: chats.attachments)
        installer.installed = { [weak self] in self?.didInstall($0) }
        installer.removed = { [weak self] _ in self?.refresh() }
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

    /// Whether a model can start, or the one that runs give way to another, now: the app is ready
    /// and the Keychain gives the API key (`keyProblem`). The Server's buttons, the menu bar's
    /// panel and the composer's model menu ask it.
    var canStart: Bool { ready && keyProblem == nil }

    /// Whether the app runs something it must stop or wait for before it quits.
    var busy: Bool { server.running || installer.running || cachedInstaller != nil }

    /// Whether the models the Hugging Face cache holds are still being looked for.
    var findingCachedModels: Bool { cachedInstaller != nil }

    /// What the server does: the look of its mark, and the phrase most views say of the model
    /// that runs.
    struct ServerStatus {
        let look: MarkLook
        let phrase: String
    }

    var serverStatus: ServerStatus {
        switch server.phase {
        case .stopped:
            return ServerStatus(look: .off, phrase: "Not running")
        case .starting:
            return ServerStatus(look: .loading, phrase: "\(server.step.rawValue)…")
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
            return ServerStatus(look: .failed, phrase: server.running ? "Not responding" : "Ran into a problem")
        }
    }

    /// What runs, in a sentence: VoiceOver announces it, and the Chat's composer says it while a
    /// model starts, stops or fails.
    var status: String {
        guard server.phase != .stopped, let model = server.model else { return "No model running" }
        return "\(displayName(of: model)) · \(serverStatus.phrase)"
    }

    /// The download under way, in the words the menus use.
    var downloadStatus: String? {
        guard let model = installer.model else { return nil }
        let name = displayName(of: model)
        guard let progress = installer.progress else { return "Downloading \(name)…" }
        return "Downloading \(name) · \(progress.fraction.formatted(.percent.precision(.fractionLength(0))))"
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

    /// Checks this Mac and the app's location, waits for the list of the models on it, stops what
    /// the app left running last time, and then, in the background, installs the models the Hugging
    /// Face cache holds; meanwhile it reads what each model family makes of the launch options.
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
        runtime.shellSettings.set(await shellSettings.value)
        await listed()
        Task { families = await runtime.families() }
        await installer.stopLeftover { self.launchWait = "Stopping the download Splash left running…" }
        await server.stopLeftover { self.launchWait = "Stopping the model Splash left running…" }
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

    /// Starts `model` with the launch options, stopping the one that runs first.
    func start(_ model: String) {
        keepKey()
        guard canStart else { return }
        selection = model
        chats.stop()
        server.start(model, settings: settings(for: model))
    }

    func stop() {
        chats.stop()
        server.stop()
    }

    /// Starts `model` again with the launch options its failure asks for (`Server.remedy`).
    func retry(_ model: String, with remedy: Server.Remedy) {
        switch remedy {
        case .memoryLimit(let limit): options["--max-memory"] = limit
        case .autoContext: options["--max-context"] = nil
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
        let requests = server.phase == .ready ? max(server.activity.running, chats.streaming == nil ? 0 : 1) : 0
        return QuitCost(model: server.model.map(displayName(of:)), requests: requests,
                        download: installer.model.map(displayName(of:)), programs: drawer.runningPrograms)
    }

    /// Stops the download, the model and the installer of cached models, which the app waits for
    /// as it quits, and the drawer's terminals, and starts nothing after. The installer of cached
    /// models gets SIGINT, as Ctrl+C sends, and SIGKILL 5 s later.
    func quit() {
        quitting = true
        chats.stop()
        drawer.endAll()
        installer.cancel()
        server.stop()
        cachedInstaller?.stop(SIGINT, killAfter: .seconds(5))
    }

    /// The next prompt, the open conversation and the draft, for the composer to count; nil while
    /// a reply streams or there is nothing to count.
    var contextRequest: ChatContext.Request? {
        guard let target, let limit = target.context, limit > 0, chats.streaming == nil else { return nil }
        let history = chats.current?.messages ?? []
        let text = draft.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !history.isEmpty || !text.isEmpty || !draft.attachments.isEmpty else { return nil }
        return ChatContext.Request(history: history, text: text, attachments: draft.attachments, target: target,
                                   started: server.since, files: chats.attachments)
    }

    /// The count the composer's ring shows: the next prompt's, or the last one of the open
    /// conversation while the next is counted or a reply streams; none in an empty chat.
    var contextUsage: ContextUsage? {
        guard let target else { return nil }
        let history = chats.current?.messages ?? []
        guard !history.isEmpty || !draft.isEmpty else { return nil }
        return chatContext.shown(history: history, target: target, started: server.since)
    }

    /// Whether the draft can go: its images are all attached, a model answers, no reply streams,
    /// and the model's context can hold the prompt, as far as it has been counted. A count not
    /// done, or failed, holds nothing back: the server checks the prompt too, and says so if it is
    /// too long.
    var canSendDraft: Bool {
        guard !draft.isEmpty, draft.attaching == 0, server.answers, chats.streaming == nil,
              draft.attachments.isEmpty || server.acceptsImages else { return false }
        return overflow == nil
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
        chats.send(message.text, attachments: message.attachments, to: target)
    }

    /// Asks the running model again for the open conversation's last answer.
    func retry() {
        target.map(chats.retry(with:))
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
    @discardableResult
    func chooseAgentFolder() -> Bool {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Choose"
        panel.message = "Choose the project folder the coding agents work in."
        panel.directoryURL = agentFolder
        NSApp.activate()
        guard panel.runModal() == .OK, let folder = panel.url else { return false }
        agentFolder = folder
        return true
    }

    /// Opens `agent` on the model that serves, in a new terminal of the window's drawer, in the
    /// agents' folder, which it asks for the first time; that terminal holds the server's API key.
    func open(_ agent: Agent) {
        guard agentFolder != nil || chooseAgentFolder(), let folder = agentFolder else { return }
        var environment = runtime.userEnvironment
        environment["SPLASH_API_KEY"] = server.settings.apiKey
        drawer.add(TerminalSession(shell: drawer.shell(), environment: environment, folder: folder,
                                   typing: agent.command(runtime: runtime, port: server.settings.port)))
        showDrawer(.terminal)
        NSApp.activate()
    }

    func copyAPIAddress() {
        let settings = server.running ? server.settings : options
        NSPasteboard.put(settings.address().appending(path: "v1").absoluteString)
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
            let all = found + installed.filter { model in
                server.running && server.model == model.id && !found.contains { $0.id == model.id }
            }
            let rank = { (id: String) in self.catalog.variants.firstIndex { $0.id == id } ?? Int.max }
            let listed = all.filter { !$0.local }.sorted { (rank($0.id), $0.id) < (rank($1.id), $1.id) }
                + all.filter(\.local)
            if listed != installed { installed = listed }
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
    /// are set.
    var modelFiles: [String: ModelFiles] {
        get {
            access(keyPath: \.modelFiles)
            let kept = defaults.dictionary(forKey: Preference.localModelFiles) as? [String: [String: String]] ?? [:]
            return kept.mapValues { ModelFiles(projector: $0["projector"], draft: $0["draft"]) }
        }
        set {
            withMutation(keyPath: \.modelFiles) {
                let kept = newValue.filter { $0.value != ModelFiles() }
                    .mapValues { ["projector": $0.projector, "draft": $0.draft].compactMapValues { $0 } }
                defaults.set(kept, forKey: Preference.localModelFiles)
            }
        }
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

    /// Deletes `model`, a model Splash installed, with the files of it nothing else uses; the
    /// model that runs is deleted once it is stopped.
    func delete(_ model: String) {
        guard ready, !(server.running && server.model == model) else { return }
        installer.remove(model)
    }

    /// Asks the Hub whether `model` or its draft has moved on, and installs what has, as its
    /// download did: the starts skip that question (`--offline`).
    func checkForUpdates(_ model: String) {
        guard ready else { return }
        installer.install(model, .check)
    }

    func isInstalled(_ model: String) -> Bool {
        installed.contains { $0.id == model }
    }

    /// Picks the model downloaded, for the user to start.
    private func didInstall(_ model: String) {
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
