import Foundation
import Observation

/// A message of a conversation, as the app keeps it.
struct ChatMessage: Codable, Identifiable, Equatable, Sendable {
    enum Role: String, Codable { case user, assistant }

    var id = UUID()
    var role: Role
    var text: String
    /// What the model thought before it answered.
    var reasoning = ""
    /// Images the user attached, as PNG or JPEG data.
    var images: [Data] = []
    /// The model that answered.
    var model: String?
    var tokensPerSecond: Double?
    /// Seconds the model thought before it answered, and the tokens its thinking took.
    var thinkingTime: Double?
    var reasoningTokens: Int?
    /// Why an answer stopped short.
    var error: String?
    /// Whether the user, or a switch of models, stopped the answer.
    var stopped: Bool?

    /// Whether the answer holds nothing to send back to the model.
    var isEmpty: Bool { text.isEmpty && reasoning.isEmpty }
}

struct Conversation: Codable, Identifiable, Equatable {
    var id = UUID()
    var title = ""
    var updated = Date.now
    var messages: [ChatMessage] = []
}

/// The model a message goes to: the server's address, the model it serves and whether it sees
/// images, and the reasoning effort to ask for.
struct ChatTarget: Equatable, Sendable {
    let url: URL
    let model: String
    let acceptsImages: Bool
    let effort: String
    /// The tokens the model's context holds, if the server said.
    var context: Int?
    /// The key the server asks for, if any.
    var apiKey: String?
}

/// The conversations, one JSON file each in the app's folder, and the one reply that streams.
@MainActor @Observable
final class ChatStore {
    /// The reply that streams: when it was asked for, when the model began to think, and how much
    /// of the prompt it has read, while it reads.
    struct Live: Equatable {
        let asked: Date
        var thinking: Date?
        var read: Double?
    }

    /// The reply that streams as it stands, apart from the conversations, which hold it empty until
    /// it ends and goes into its conversation, once: an object of its own, which the transcript
    /// alone reads, so that each flush of it draws the transcript and nothing else, not the list of
    /// conversations and its search, nor the composer, which read the conversations.
    @MainActor @Observable
    final class Reply {
        let conversation: UUID
        fileprivate(set) var message: ChatMessage
        fileprivate(set) var live: Live

        fileprivate init(_ message: ChatMessage, in conversation: UUID) {
            self.conversation = conversation
            self.message = message
            live = Live(asked: .now)
        }
    }

    private(set) var conversations: [Conversation] = []
    /// The open conversation; nil is a new one, saved once it has a message.
    var selection: UUID?
    private(set) var reply: Reply?

    /// The conversation whose reply streams now, the open one or another.
    var streaming: UUID? { reply?.conversation }

    private let folder: URL
    private var replying: Task<Void, Never>?
    /// Writes and removes the conversations' files in the order they change, off the main thread:
    /// a conversation with images takes milliseconds to encode and write, which a send and a
    /// reply's end would otherwise wait for.
    private let files = DispatchQueue(label: "ChatStore.files", qos: .utility)

    init(folder: URL) {
        self.folder = folder
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        conversations = files.filter { $0.pathExtension == "json" }
            .compactMap { try? JSONDecoder().decode(Conversation.self, from: Data(contentsOf: $0)) }
            .map(Self.interrupted)
            .sorted { $0.updated > $1.updated }
    }

    var current: Conversation? {
        conversations.first { $0.id == selection }
    }

    func newChat() {
        selection = nil
    }

    func delete(_ id: UUID) {
        if streaming == id { stop() }
        conversations.removeAll { $0.id == id }
        let file = file(id)
        files.async { try? FileManager.default.removeItem(at: file) }
        if selection == id { selection = nil }
    }

    func rename(_ id: UUID, to title: String) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].title = title
        write(conversations[index])
    }

    /// Sends `text` and `images` in the open conversation, or a new one, to the model `server`
    /// serves, and streams its reply into the conversation.
    func send(_ text: String, images: [Data], to server: ChatTarget) {
        guard streaming == nil else { return }
        var conversation = current ?? Conversation()
        if conversation.title.isEmpty {
            let line = text.split(separator: "\n").first.map(String.init) ?? "Image"
            conversation.title = String(line.prefix(80))
        }
        conversation.messages.append(ChatMessage(role: .user, text: text, images: images))
        stream(conversation, to: server)
    }

    /// Answers the open conversation's last message again, in place of the last answer.
    func retry(with server: ChatTarget) {
        guard streaming == nil, var conversation = current, conversation.messages.last?.role == .assistant else { return }
        conversation.messages.removeLast()
        stream(conversation, to: server)
    }

    /// Stops the reply; what it said so far stays.
    func stop() {
        replying?.cancel()
    }

    private func stream(_ conversation: Conversation, to server: ChatTarget) {
        var conversation = conversation
        let history = conversation.messages
        var answer = ChatMessage(role: .assistant, text: "", model: server.model)
        conversation.messages.append(answer)
        conversation.updated = .now
        put(conversation)
        selection = conversation.id
        let reply = Reply(answer, in: conversation.id)
        self.reply = reply
        replying = Task { [weak self] in
            var shown = ContinuousClock.now
            var thinking: Date?
            var read: Double?
            do {
                for try await delta in ChatClient.stream(history, with: server) {
                    if !delta.reasoning.isEmpty, thinking == nil {
                        thinking = .now
                        reply.live.thinking = thinking
                    }
                    if !delta.text.isEmpty, answer.text.isEmpty, let thinking {
                        answer.thinkingTime = Date.now.timeIntervalSince(thinking)
                    }
                    answer.reasoning += delta.reasoning
                    answer.text += delta.text
                    answer.tokensPerSecond = delta.tokensPerSecond ?? answer.tokensPerSecond
                    answer.reasoningTokens = delta.reasoningTokens ?? answer.reasoningTokens
                    read = delta.read ?? read
                    // Asked for no limit, a reply runs to the end of the context.
                    if delta.cut { answer.error = "The reply reached the end of the model’s context." }
                    // A redraw per token would be wasted work: the page lets out what comes at an
                    // even pace (page.js).
                    if ContinuousClock.now - shown > .milliseconds(50) {
                        shown = .now
                        reply.live.read = read
                        reply.message = answer
                    }
                }
                // Cancelled, the stream just ends; what it said stays.
                if Task.isCancelled { answer.stopped = true }
            } catch {
                answer.error = error.localizedDescription
            }
            // A reply that ended while the model still thought.
            if let thinking, answer.thinkingTime == nil { answer.thinkingTime = Date.now.timeIntervalSince(thinking) }
            self?.finish(answer, in: conversation.id)
            self?.reply = nil
        }
    }

    /// Puts the reply that ended, `message`, in its conversation, which it brings to the top, and
    /// writes it.
    private func finish(_ message: ChatMessage, in conversation: UUID) {
        guard var updated = conversations.first(where: { $0.id == conversation }),
              let index = updated.messages.firstIndex(where: { $0.id == message.id })
        else { return }
        updated.messages[index] = message
        updated.updated = .now
        put(updated)
    }

    /// Keeps `conversation` first, the most recent, and writes it.
    private func put(_ conversation: Conversation) {
        conversations.removeAll { $0.id == conversation.id }
        conversations.insert(conversation, at: 0)
        write(conversation)
    }

    private func write(_ conversation: Conversation) {
        let (folder, file) = (folder, file(conversation.id))
        files.async {
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try? JSONEncoder().encode(conversation).write(to: file, options: .atomic)
        }
    }

    /// Returns once every change made so far is on disk, as the app quits.
    func flush() {
        files.sync {}
    }

    private func file(_ id: UUID) -> URL {
        folder.appending(path: "\(id.uuidString).json")
    }

    /// `conversation` as the app left it: an answer saved before it streamed anything, when the
    /// app quit or crashed meanwhile, says it was interrupted.
    private static func interrupted(_ conversation: Conversation) -> Conversation {
        var conversation = conversation
        for index in conversation.messages.indices where conversation.messages[index].role == .assistant
            && conversation.messages[index].isEmpty && conversation.messages[index].error == nil
            && conversation.messages[index].stopped != true {
            conversation.messages[index].error = "The reply was interrupted."
        }
        return conversation
    }
}

/// The server's Chat Completions API, streamed.
enum ChatClient {
    struct Delta: Equatable {
        var text = ""
        var reasoning = ""
        var tokensPerSecond: Double?
        /// Whether the reply ended at its length limit.
        var cut = false
        /// How much of the prompt the model has read, from 0 to 1, while it reads.
        var read: Double?
        /// The tokens the model's thinking took, once the reply ends.
        var reasoningTokens: Int?
    }

    enum Event: Equatable {
        case delta(Delta)
        case done
    }

    struct Failure: LocalizedError {
        let errorDescription: String?
    }

    /// The tokens `messages` take as the model reads them, its template, images and earlier
    /// thinking included, counted without inference and without waking a sleeping engine.
    static func countTokens(_ messages: [ChatMessage], with server: ChatTarget, session: URLSession = .shared) async throws -> Int {
        let request = try countRequest(messages, with: server)
        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            throw Failure(errorDescription: message(of: data, for: messages, server: server)
                ?? HTTPURLResponse.localizedString(forStatusCode: status))
        }
        struct Count: Decodable { let input_tokens: Int }
        guard let count = try? JSONDecoder().decode(Count.self, from: data), count.input_tokens >= 0 else {
            throw Failure(errorDescription: "The model returned an invalid token count.")
        }
        return count.input_tokens
    }

    /// The request counting `messages`: the prompt `stream` sends, as it sends it.
    static func countRequest(_ messages: [ChatMessage], with server: ChatTarget) throws -> URLRequest {
        let json = try JSONSerialization.data(withJSONObject: prompt(messages, effort: server.effort, images: server.acceptsImages))
        return request("v1/chat/completions/count_tokens", json, to: server, timeout: 30)
    }

    /// A POST of the JSON `body` to `path` on `server`, with the key it asks for.
    private static func request(_ path: String, _ body: Data, to server: ChatTarget, timeout: TimeInterval) -> URLRequest {
        var request = URLRequest(url: server.url.appending(path: path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = server.apiKey { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        request.httpBody = body
        request.timeoutInterval = timeout
        return request
    }

    static func stream(_ messages: [ChatMessage], with server: ChatTarget) -> AsyncThrowingStream<Delta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let json = try body(messages, effort: server.effort, images: server.acceptsImages)
                    let (bytes, response) = try await URLSession.shared.bytes(for: request("v1/chat/completions", json, to: server, timeout: 3600))
                    if let status = (response as? HTTPURLResponse)?.statusCode, status != 200 {
                        var data = Data()
                        for try await byte in bytes { data.append(byte) }
                        throw Failure(errorDescription: message(of: data, for: messages, server: server)
                            ?? HTTPURLResponse.localizedString(forStatusCode: status))
                    }
                    for try await line in bytes.lines {
                        switch try parse(line) {
                        case .delta(let delta): continuation.yield(delta)
                        case .done: return continuation.finish()
                        case nil: continue
                        }
                    }
                    throw Failure(errorDescription: "The connection closed before the reply ended.")
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The request for `messages`: the prompt, streamed, with its progress and the usage asked for.
    static func body(_ messages: [ChatMessage], effort: String, images: Bool) throws -> Data {
        var body = prompt(messages, effort: effort, images: images)
        body["stream"] = true
        body["stream_options"] = ["include_usage": true]
        body["return_progress"] = true
        return try JSONSerialization.data(withJSONObject: body)
    }

    /// The prompt of `messages`: images as data URLs, or a note in their place for a model that
    /// cannot see them; earlier answers with what the model thought, empty ones left out; and the
    /// reasoning effort unless it is the model's default.
    private static func prompt(_ messages: [ChatMessage], effort: String, images: Bool) -> [String: Any] {
        let wire: [[String: Any]] = messages.compactMap { message in
            switch message.role {
            case .user where !message.images.isEmpty && images:
                let parts: [[String: Any]] = message.images.map { image in
                    ["type": "image_url", "image_url": ["url": "data:\(mediaType(image));base64,\(image.base64EncodedString())"]]
                }
                return ["role": "user", "content": parts + [["type": "text", "text": message.text]]]
            case .user where !message.images.isEmpty:
                let note = message.images.count == 1 ? "[An image this model can’t see]" : "[\(message.images.count) images this model can’t see]"
                return ["role": "user", "content": message.text.isEmpty ? note : "\(note)\n\n\(message.text)"]
            case .user:
                return ["role": "user", "content": message.text]
            case .assistant where message.isEmpty:
                return nil
            case .assistant:
                return ["role": "assistant", "content": message.text, "reasoning_content": message.reasoning]
            }
        }
        var body: [String: Any] = ["messages": wire]
        if !effort.isEmpty { body["reasoning_effort"] = effort }
        return body
    }

    /// The event a line of the event stream carries, nil for a line that carries none (a
    /// keepalive comment); an error event throws.
    static func parse(_ line: String) throws -> Event? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" { return .done }
        struct Chunk: Decodable {
            struct Choice: Decodable {
                struct Content: Decodable { let content: String?; let reasoning_content: String? }
                let delta: Content?
                let finish_reason: String?
            }
            struct Metrics: Decodable {
                struct Latency: Decodable { let stream_tokens_per_second: Double? }
                let request_latency: Latency?
            }
            struct Progress: Decodable { let total: Int; let processed: Int }
            struct Usage: Decodable {
                struct Details: Decodable { let reasoning_tokens: Int? }
                let completion_tokens_details: Details?
            }
            struct Problem: Decodable { let message: String }
            let choices: [Choice]?
            let metrics: Metrics?
            let prompt_progress: Progress?
            let usage: Usage?
            let error: Problem?
        }
        let chunk = try JSONDecoder().decode(Chunk.self, from: Data(payload.utf8))
        if let error = chunk.error { throw Failure(errorDescription: error.message) }
        let choice = chunk.choices?.first
        let progress = chunk.prompt_progress.flatMap { $0.total > 0 ? min(Double($0.processed) / Double($0.total), 1) : nil }
        return .delta(Delta(text: choice?.delta?.content ?? "", reasoning: choice?.delta?.reasoning_content ?? "",
                            tokensPerSecond: chunk.metrics?.request_latency?.stream_tokens_per_second,
                            cut: choice?.finish_reason == "length", read: progress,
                            reasoningTokens: chunk.usage?.completion_tokens_details?.reasoning_tokens))
    }

    /// What the error in a response's body to `messages` says: a prompt too long for the model's
    /// context, in the app's words; any other, in the server's.
    static func message(of data: Data, for messages: [ChatMessage], server: ChatTarget) -> String? {
        struct Body: Decodable { struct Problem: Decodable { let message: String; let code: String? }; let error: Problem }
        guard let problem = try? JSONDecoder().decode(Body.self, from: data).error else { return nil }
        guard problem.code == "context_length_exceeded" else { return problem.message }
        let context = server.context.map { "the model’s \(ContextLength.figure($0))-token context" } ?? "the model’s context"
        // A new chat leaves the earlier messages behind; a first message has none to leave.
        return messages.count == 1 ? "This message is longer than \(context)."
            : "This chat is longer than \(context). Start a new chat to continue."
    }

    /// The type of an image the app keeps, PNG or JPEG.
    static func mediaType(_ image: Data) -> String {
        image.starts(with: [0x89, 0x50, 0x4E, 0x47]) ? "image/png" : "image/jpeg"
    }
}
