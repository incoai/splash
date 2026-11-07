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
    let draft = Draft()
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
    var section = Section.server
    /// The launch options the next start uses.
    var options: ServeSettings {
        didSet {
            options.save(defaults)
            if options.apiKey != keptKey { keep(options.apiKey) }
        }
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
        server = Server(runtime: runtime, log: folders.logs.appending(path: "server.log"), defaults: defaults)
        installer = Installer(runtime: runtime, log: folders.logs.appending(path: "install.log"), defaults: defaults)
        catalog = Catalog(contentsOf: runtime.suggestedModels)
        transcript = Transcript(runtime: runtime)
        installer.installed = { [weak self] in self?.didInstall($0) }
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

    /// Whether models can start and download.
    var ready: Bool { available && !quitting }

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
        return server.settings.differs(from: options, model: model, family: family(of: model))
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
        if let reason = await runtime.unsupportedReason() {
            blocker = Blocker(title: "This Mac Can’t Run Splash", symbol: "exclamationmark.triangle", message: reason)
            return
        }
        if Self.mustMoveToApplications(Bundle.main.bundleURL) {
            blocker = Blocker(title: "Move Splash to Applications", symbol: "externaldrive",
                              message: "Splash can’t run from the disk image. Drag it to your Applications folder, then open it from there.")
            return
        }
        await listed()
        Task { families = await runtime.families() }
        await installer.stopLeftover()
        await server.stopLeftover()
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
        guard ready, keyProblem == nil else { return }
        selection = model
        chats.stop()
        server.start(model, settings: options.usable(by: family(of: model)))
    }

    func stop() {
        chats.stop()
        server.stop()
    }

    /// The model Stop names, while a server runs: the one a switch goes to, else the one that runs;
    /// and whether Stop does anything now, which it doesn't once a stop with no switch is under way.
    var stopItem: (model: String, enabled: Bool)? {
        guard server.running, let model = server.next ?? server.model else { return nil }
        return (model, !(server.phase == .stopping && server.next == nil))
    }

    func download(_ model: String) {
        guard ready else { return }
        installer.install(model)
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
        guard !history.isEmpty || !text.isEmpty || !draft.images.isEmpty else { return nil }
        return ChatContext.Request(history: history, text: text, images: draft.images, target: target, started: server.since)
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
              draft.images.isEmpty || server.acceptsImages else { return false }
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
        chats.send(message.text, images: message.images, to: target)
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

    /// Opens `agent` in Terminal, on the model that serves.
    func open(_ agent: Agent) {
        NSApp.activate()
        agent.open(runtime: runtime, settings: server.settings, defaults: defaults)
    }

    func copyAPIAddress() {
        let settings = server.running ? server.settings : options
        NSPasteboard.put(settings.url.appending(path: "v1").absoluteString)
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

    /// Takes model files off the list; the files stay where they are.
    func removeLocal(_ path: String) {
        localModels.removeAll { $0 == path }
        refresh()
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
