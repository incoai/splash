import Foundation
import Observation
import SplashChat

@MainActor @Observable
final class ChatStore {
    var messages: [ChatMessage] = []
    var draft = ""
    var address: String
    var apiKey = ""
    var thinking: Bool { didSet { defaults.set(thinking, forKey: "thinking") } }
    var model = ""
    var connected = false
    var connecting = false
    var generating = false
    var connectionError: String?
    var replyError: String?
    var storageError: String?
    var tokensPerSecond: Double?

    private let defaults: UserDefaults
    private let conversationURL: URL
    private var connectionTask: Task<Void, Never>?
    private var replyTask: Task<Void, Never>?
    private var retryMessages: [ChatMessage]?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        address = defaults.string(forKey: "serverAddress") ?? "http://127.0.0.1:8000"
        thinking = defaults.object(forKey: "thinking") as? Bool ?? true
        conversationURL = URL.applicationSupportDirectory
            .appendingPathComponent("Splash Desktop", isDirectory: true)
            .appendingPathComponent("conversation.json")
        if FileManager.default.fileExists(atPath: conversationURL.path) {
            do {
                messages = try JSONDecoder().decode([ChatMessage].self, from: Data(contentsOf: conversationURL))
            } catch {
                storageError = "Your last chat couldn’t be restored. \(error.localizedDescription)"
            }
        }
    }

    var canSend: Bool {
        connected && !generating && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func connect() {
        guard !generating else { return }
        connectionTask?.cancel()
        connecting = true
        connectionTask = Task {
            do {
                let client = try SplashClient(address: address, apiKey: apiKey)
                let models = try await client.models()
                try Task.checkCancellation()
                model = models[0]
                connected = true
                connectionError = nil
            } catch {
                guard !Task.isCancelled else { return }
                connected = false
                connectionError = Self.connectionMessage(error)
            }
            connecting = false
        }
    }

    func configure(address: String, apiKey: String) throws {
        let client = try SplashClient(address: address, apiKey: apiKey)
        self.address = client.baseURL.absoluteString
        self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        defaults.set(self.address, forKey: "serverAddress")
        connected = false
        connect()
    }

    func send() {
        guard canSend else { return }
        messages.append(ChatMessage(role: "user", content: draft.trimmingCharacters(in: .whitespacesAndNewlines)))
        draft = ""
        generate()
    }

    func retry() {
        guard !generating, connected, let previous = retryMessages else { return }
        messages = previous
        generate()
    }

    private func generate() {
        connectionTask?.cancel()
        connecting = false
        let history = messages
        retryMessages = history
        messages.append(ChatMessage(role: "assistant"))
        let replyID = messages.last!.id
        generating = true
        replyError = nil
        tokensPerSecond = nil
        save()
        replyTask = Task {
            do {
                let client = try SplashClient(address: address, apiKey: apiKey)
                try await client.complete(messages: history, model: model, thinking: thinking) { [weak self] delta in
                    await self?.append(delta, to: replyID)
                }
                retryMessages = nil
            } catch {
                if !Task.isCancelled {
                    replyError = Self.connectionMessage(error)
                }
            }
            // Keep useful partial text when stopped; never send an empty assistant turn.
            if let last = messages.last, last.id == replyID, last.content.isEmpty && last.reasoning.isEmpty {
                messages.removeLast()
            }
            generating = false
            replyTask = nil
            save()
        }
    }

    private func append(_ delta: ChatDelta, to id: UUID) {
        guard let index = messages.indices.last, messages[index].id == id else { return }
        messages[index].content += delta.content
        messages[index].reasoning += delta.reasoning
        if let speed = delta.tokensPerSecond, speed.isFinite && speed > 0 { tokensPerSecond = speed }
    }

    func stop() { replyTask?.cancel() }

    func prepareToQuit() {
        stop()
        save()
    }

    func clear() {
        guard !generating else { return }
        messages = []
        retryMessages = nil
        replyError = nil
        tokensPerSecond = nil
        save()
    }

    private func save() {
        do {
            let directory = conversationURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            let data = try JSONEncoder().encode(messages.filter { !$0.content.isEmpty || !$0.reasoning.isEmpty })
            try data.write(to: conversationURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: conversationURL.path)
            storageError = nil
        } catch {
            storageError = "This chat couldn’t be saved. \(error.localizedDescription)"
        }
    }

    private static func connectionMessage(_ error: Error) -> String {
        if let urlError = error as? URLError,
           [.cannotConnectToHost, .cannotFindHost, .networkConnectionLost, .timedOut].contains(urlError.code) {
            return "Can’t reach Splash. Start the server, then reconnect."
        }
        return error.localizedDescription
    }
}
