import Darwin
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// What the app does, end to end, with a real runtime and real models: the app's own state and
/// views, driven as its buttons drive them. Runs when `SPLASH_RUNTIME` names a packaged runtime and
/// `SPLASH_APP_MODELS` installed models (two to switch between); `SPLASH_APP_PORT` (a free one) is
/// the port it serves on, `SPLASH_APP_DOWNLOAD` a model to download, cancel and download again, and
/// `SPLASH_APP_SNAPSHOTS` a folder for the windows it renders on the way.
enum AppFlow {
    static let environment = ProcessInfo.processInfo.environment
    static let models = environment["SPLASH_APP_MODELS"]
        .map { $0.split(separator: " ").map(String.init) }.flatMap { $0.isEmpty ? nil : $0 }
    static let port = Int(environment["SPLASH_APP_PORT"] ?? "") ?? freePort()
    /// The defaults of every test of a run, kept from one to the next as an app's are, apart from
    /// the user's: each test's launch is a relaunch of one app, which stops the server an earlier
    /// one left running before it starts its own on the same port.
    @MainActor static let defaults = UserDefaults.kept(in: FileManager.default.temporaryDirectory.appending(path: "splash-app-flow"))
}

/// A class, so that each test's instance, which Swift Testing lets go as the test ends, however it
/// ends, quits what the test started: the server, a download. A test that failed threw before its
/// own stop, and left the model server running, its model resident beside the next run's.
@MainActor
@Suite(.serialized, .enabled(if: AppFlow.models != nil))
final class AppFlowTests {
    static let environment = AppFlow.environment
    static let models = AppFlow.models
    static let port = AppFlow.port

    /// The launch options of every run: the server's defaults, on the test's port.
    static let options = ServeSettings(values: ["--port": String(port)])
    /// The seconds a server is given to reach each step of a start, a switch or a failure.
    static let stepTimeout = 600.0

    let state: AppState
    let model: String
    /// The conversations and logs of this test, apart from the user's.
    let folder = FileManager.default.temporaryDirectory.appending(path: "splash-app-flow-\(UUID().uuidString)")

    init() async throws {
        Self.options.save(AppFlow.defaults)
        let runtime = try #require(Runtime.locate(), "SPLASH_RUNTIME names no runtime")
        #expect(runtime.packaged)
        state = AppState.forTests(in: folder, runtime: runtime, defaults: AppFlow.defaults)
        model = Self.models![0]
        await state.launch()
        // The window opens on the Server, and nothing starts until the user starts it.
        #expect(state.section == .server)
        #expect(!state.server.running)
    }

    isolated deinit {
        state.quit()
        try? FileManager.default.removeItem(at: folder)
    }

    /// The next launch of the app, as after a quit or a crash of this one.
    func relaunch() -> AppState {
        AppState.forTests(in: folder, runtime: state.runtime, defaults: AppFlow.defaults)
    }

    @Test func thisMacIsSupportedAndGetsSuggestions() async throws {
        #expect(await state.runtime.unsupportedReason() == nil)
        let suggestions = state.catalog.suggestions(memory: Catalog.memory)
        print("Suggested for \(Catalog.memory) GB: \(suggestions.map(\.id))")
        #expect(!suggestions.isEmpty)
        await Snapshot.save("server", MainView().environment(state))
        await Snapshot.save("server-advanced", size: CGSize(width: 1120, height: 1100), ServerView(advanced: true).environment(state))
        state.section = .chat
        await Snapshot.save("chat-stopped", MainView().environment(state))
        // A Mac with no model yet: a runtime whose models folder is empty.
        let fresh = folder.appending(path: "first-run")
        try FileManager.default.createDirectory(at: fresh.appending(path: "install"), withIntermediateDirectories: true)
        try? FileManager.default.copyItem(at: state.runtime.suggestedModels, to: Runtime(root: fresh).suggestedModels)
        let first = AppState.forTests(in: fresh, runtime: Runtime(root: fresh))
        await first.listed()
        #expect(first.installed.isEmpty && first.selection == nil)
        await first.launch()
        await Snapshot.save("server-first-run", MainView().environment(first))
    }

    @Test func aModelInstallsStartsServesAndStops() async throws {
        try await install(model)
        state.start(model)
        #expect(state.selection == model)
        try await waitFor(state.server.phase == .starting, "starting (\(state.server.phase))", seconds: Self.stepTimeout)
        await Snapshot.save("server-starting", MainView().environment(state))
        // It says what it does, as its serve record tells it: it reads the weights, then warms up.
        try await waitFor(state.server.step == .warming, "warming up (\(state.server.phase))", seconds: Self.stepTimeout)
        await Snapshot.save("server-warming", MainView().environment(state))
        try await ready()
        #expect((state.server.context ?? 0) > 0)
        await Snapshot.save("server-ready", MainView().environment(state))
        let reply = try await chat("Reply with one word: hello")
        #expect(!reply.isEmpty)
        state.section = .chat
        await Snapshot.save("chat-new", MainView().environment(state))
        // The app's own chat: the reply streams into a conversation it keeps.
        try await converse("Write a haiku about the sea, then a Swift function that adds two numbers.")
        let conversation = try #require(state.chats.current)
        #expect(conversation.messages.map(\.role) == [.user, .assistant])
        #expect(conversation.messages[1].model == model)
        state.chats.flush()
        #expect(ChatStore(folder: folder).conversations.first?.messages == conversation.messages, "saved")
        await Snapshot.save("chat-ready", MainView().environment(state))
        #expect(state.server.requests.contains { $0.id > 0 && $0.end?.outcome == .done && ($0.end?.output ?? 0) > 0 })
        #expect(state.server.activity.generated > 0)
        await Snapshot.save("server-activity", size: CGSize(width: 1120, height: 1100), ServerView(advanced: true).environment(state))
        await Snapshot.save("menu-bar", size: CGSize(width: 300, height: 400), MenuBarPanel().environment(state))
        await Snapshot.save("popup-thinking", size: CGSize(width: 300, height: 260), PopupList(
            header: "Thinking",
            items: [PopupItem(id: "xhigh", title: "Extra High", badge: "Default"),
                    PopupItem(id: "medium", title: "Medium"), PopupItem(id: "low", title: "Low"),
                    PopupItem(id: "none", title: "No Thinking")],
            chosen: "xhigh", highlighted: .constant("medium")) { _ in })
        await Snapshot.save("popup-context", size: CGSize(width: 260, height: 360), PopupList(
            header: nil,
            items: [PopupItem(id: "", title: "Auto", badge: "Default")]
                + ContextLength.lengths.map { PopupItem(id: String($0), title: ContextLength.title($0)) }
                + [PopupItem(id: "custom", title: "Custom…", separated: true)],
            chosen: "", highlighted: .constant(nil)) { _ in })
        // Changed options ask for a restart, which starts the model with them.
        state.options["--max-context"] = "32768"
        #expect(state.optionsChanged)
        state.section = .server
        await Snapshot.save("server-options-changed", MainView().environment(state))
        state.start(model)
        #expect(state.status.hasSuffix(" · Restarting…"))
        try await waitFor(state.server.phase == .starting, "restarting (\(state.server.phase))", seconds: Self.stepTimeout)
        #expect(state.server.settings["--max-context"] == "32768" && !state.optionsChanged)
        try await ready()
        let pid = try #require(state.runtime.serveRecord(port: Self.port)?.pid)
        state.stop()
        #expect(state.server.phase == .stopping)
        try await waitFor(state.server.phase == .stopped, "stopped (\(state.server.phase))", seconds: 60)
        #expect(kill(pid, 0) != 0)
    }

    @Test func switchingStopsOneModelAndStartsTheNext() async throws {
        guard let models = Self.models, models.count > 1 else { return }
        try await install(models[1])
        state.start(model)
        try await ready()
        try await converse("Say hi.")
        let first = try #require(state.runtime.serveRecord(port: Self.port)?.pid)
        // Picking another model on the Server offers the switch; it switches only when asked to.
        state.selection = models[1]
        #expect(state.server.model == model && state.server.phase == .ready)
        await Snapshot.save("server-switch", MainView().environment(state))
        state.start(models[1])
        #expect(state.server.phase == .stopping)
        #expect(state.server.next == models[1])
        #expect(state.status.contains(" · Switching to "))
        // The conversation stays while the next model starts.
        state.section = .chat
        await Snapshot.save("chat-switching", MainView().environment(state))
        try await waitFor(state.server.model == models[1] && state.server.phase != .stopping,
                          "next starting (\(state.server.phase))", seconds: Self.stepTimeout)
        #expect(state.server.phase == .starting)
        #expect(kill(first, 0) != 0)
        try await ready()
        // The next model answers the same conversation.
        try await converse("Now say bye.")
        #expect(state.chats.current?.messages.count == 4)
        #expect(state.chats.current?.messages.last?.model == models[1])
        try await stop()
    }

    @Test func modelFilesOnThisMacStartAndAnswer() async throws {
        // The GGUF a download of the model left in the Hugging Face cache, added as files on this Mac;
        // they serve with the family's draft at the commit the download recorded for its branch.
        try await install(model)
        let file = try #require(Self.gguf(of: model), "no GGUF of \(model) in the Hugging Face cache")
        state.addLocal(file)
        // The list outlives the test, so that a failed one leaves the next a Mac without the files.
        defer { state.removeLocal(file.path) }
        await state.listed()
        #expect(state.selection == file.path && state.isInstalled(file.path))
        state.start(file.path)
        try await ready()
        await Snapshot.save("server-local", MainView().environment(state))
        // The API names it by the files' name, and the app's chat answers with it.
        let (data, _) = try await URLSession.shared.data(from: state.server.settings.url.appending(path: "v1/models"))
        struct Models: Decodable { struct Model: Decodable { let id: String }; let data: [Model] }
        #expect(try JSONDecoder().decode(Models.self, from: data).data.first?.id == ServeSettings.localID(file.path))
        try await converse("Say hi.")
        try await stop()
        state.removeLocal(file.path)
        await state.listed()
        #expect(!state.isInstalled(file.path))
    }

    @Test func aStartThatFailsSaysWhy() async throws {
        state.options["--max-memory"] = "1G"
        state.start(model)
        try await waitFor({ if case .failed = state.server.phase { true } else { false } }(),
                          "failed (\(state.server.phase))", seconds: Self.stepTimeout)
        guard case .failed(let message) = state.server.phase else { return }
        #expect(message.hasPrefix("It needs ")
                && message.hasSuffix("but the memory limit is 1 GB. Raise the memory limit in Launch Options."),
                "\(message)")
        await Snapshot.save("server-failed", MainView().environment(state))
        state.section = .chat
        await Snapshot.save("chat-failed", MainView().environment(state))
        state.section = .server
        // A download that fails says why too.
        state.installer.install("splash-app-test/no-such-model")
        try await waitFor(!state.installer.running, "the download failed", seconds: 120)
        #expect(state.installer.failure?.model == "splash-app-test/no-such-model")
        #expect(state.installer.failure?.message.contains("Hugging Face has no public model") == true)
        await Snapshot.save("server-download-failed", MainView().environment(state))
        state.installer.dismissFailure()
        state.server.dismissFailure()
        #expect(state.installer.failure == nil && state.server.phase == .stopped)
    }

    @Test func aBusyPortFailsAtOnce() async throws {
        let listener = Process()
        listener.executableURL = URL(filePath: "/usr/bin/python3")
        listener.arguments = ["-m", "http.server", "--bind", "127.0.0.1", String(Self.port)]
        listener.standardOutput = FileHandle.nullDevice
        listener.standardError = FileHandle.nullDevice
        try listener.run()
        defer { listener.terminate() }
        try await waitFor(portInUse(Self.port), "the listener (\(state.server.phase))", seconds: Self.stepTimeout)
        state.start(model)
        guard case .failed(let message) = state.server.phase else {
            Issue.record("\(state.server.phase)")
            return
        }
        #expect(message.hasPrefix("Port \(Self.port) is in use"))
        #expect(!state.server.running)
    }

    @Test func quittingStopsEverythingAndStartsNothing() async throws {
        state.start(model)
        try await ready()
        // Quit in the middle of a switch: the switch is dropped.
        state.start(model)
        state.quit()
        try await waitFor(!state.server.running, "stopped (\(state.server.phase))", seconds: 60)
        try await Task.sleep(for: .seconds(2))
        #expect(!state.server.running)
        state.start(model)
        #expect(!state.server.running)
        // The model stays picked for the next launch, which starts nothing by itself.
        let next = relaunch()
        await next.listed()
        #expect(next.selection == model)
    }

    @Test func aRelaunchStopsTheServerACrashLeftBehind() async throws {
        state.start(model)
        try await ready()
        let pid = try #require(state.runtime.serveRecord(port: Self.port)?.pid)
        // The next launch of the app, as after a crash of this one.
        await relaunch().server.stopLeftover()
        #expect(kill(pid, 0) != 0)
        try await waitFor({ if case .failed = state.server.phase { true } else { false } }(),
                          "noticed (\(state.server.phase))", seconds: Self.stepTimeout)
    }

    @Test func aDownloadReportsProgressCancelsAndResumes() async throws {
        guard let download = Self.environment["SPLASH_APP_DOWNLOAD"] else { return }
        let installer = state.installer
        installer.install(download)
        try await waitFor((installer.progress?.done ?? 0) > 0, "progress", seconds: 120)
        #expect(installer.progress?.repo == String(download.split(separator: ":")[0]))
        #expect(state.downloadStatus?.hasPrefix("Downloading ") == true)
        #expect(state.serverTitle == "Downloading…")
        await Snapshot.save("server-downloading", MainView().environment(state))
        let clock = ContinuousClock.now
        installer.cancel()
        try await waitFor(!installer.running, "cancelled", seconds: 30)
        #expect(ContinuousClock.now - clock < .seconds(3), "the installer stops on SIGINT")
        #expect(installer.failure == nil)
        #expect(!state.isInstalled(download))
        // A crash of the app leaves the download running; the next launch stops it.
        installer.install(download)
        try await waitFor((installer.progress?.done ?? 0) > 0, "progress again", seconds: 120)
        let pid = pid_t(try #require((state.defaults.array(forKey: Preference.installerPID) as? [Int])?.first))
        await relaunch().installer.stopLeftover()
        #expect(kill(pid, 0) != 0)
        try await waitFor(!installer.running, "noticed", seconds: 30)
        try await install(download)
    }

    // MARK: -

    private func install(_ model: String) async throws {
        await state.refresh().value
        guard !state.isInstalled(model) else { return }
        state.installer.install(model)
        try await waitFor(!state.installer.running, "installed \(model)", seconds: 3600)
        #expect(state.installer.failure == nil, "\(String(describing: state.installer.failure))")
        #expect(state.isInstalled(model))
        // A model downloaded is picked, not started.
        #expect(state.selection == model && !state.server.running)
    }

    private func ready() async throws {
        try await waitFor(state.server.phase == .ready || { if case .failed = state.server.phase { true } else { false } }(),
                          "ready (\(state.server.phase))", seconds: 900)
        #expect(state.server.phase == .ready, "\(state.server.phase); see \(state.server.log.path)")
    }

    private func stop() async throws {
        state.server.stop()
        try await waitFor(!state.server.running, "stopped (\(state.server.phase))", seconds: 60)
    }

    /// The file of `model`, a GGUF's `OWNER/REPO:VARIANT`, that a download left in the Hugging Face
    /// cache.
    private static func gguf(of model: String) -> URL? {
        let parts = model.split(separator: ":")
        guard parts.count == 2 else { return nil }
        let cache = environment["HF_HUB_CACHE"].map { URL(filePath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
        let snapshots = cache.appending(path: "models--\(parts[0].replacing("/", with: "--"))/snapshots")
        let files = FileManager.default.enumerator(at: snapshots, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []
        return files.first { $0.lastPathComponent.hasSuffix("\(parts[1]).gguf") }
    }

    /// Sends `prompt` as the app's composer does and waits for the reply to end.
    private func converse(_ prompt: String) async throws {
        state.draft.text = prompt
        state.sendDraft()
        #expect(state.chats.streaming != nil)
        try await waitFor(state.chats.streaming == nil, "the reply", seconds: 300)
        let reply = try #require(state.chats.current?.messages.last)
        #expect(reply.error == nil, "\(reply.error ?? "")")
        #expect(!reply.text.isEmpty)
    }

    private func chat(_ prompt: String) async throws -> String {
        var request = URLRequest(url: state.server.settings.url.appending(path: "v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": prompt]], "max_tokens": 32, "reasoning_effort": "none",
        ])
        request.timeoutInterval = 300
        let (data, _) = try await URLSession.shared.data(for: request)
        struct Reply: Decodable {
            struct Choice: Decodable { struct Message: Decodable { let content: String? }; let message: Message }
            let choices: [Choice]
        }
        return try JSONDecoder().decode(Reply.self, from: data).choices.first?.message.content ?? ""
    }
}
