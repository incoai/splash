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
    /// What the user attached (`Attachment`); nil for nothing, as in a message an earlier app kept.
    var attachments: [Attachment]?
    /// Images an earlier app kept in the message itself, as PNG or JPEG data, which
    /// `ChatStore.adoptFiles` moves into files of their own; none after. Always written, as the
    /// 1.6.0 app requires it of every message, and skips a chat that lacks it, as after a downgrade.
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

/// A file the user attached to a message: an image, PNG or JPEG, or a PDF. It is kept as a file of
/// its own in the attachments folder (`ChatStore.attachments`), not inside its conversation's file,
/// and read as a request or the transcript needs it.
struct Attachment: Codable, Identifiable, Equatable, Sendable {
    enum Kind: String, Codable, Sendable { case image, pdf }

    var id = UUID()
    var kind: Kind
    /// Its media type: image/png, image/jpeg or application/pdf.
    var type: String
    var bytes: Int
    /// A PDF's name and pages, which its chip shows.
    var name: String?
    var pages: Int?

    /// Its file's name in the attachments folder.
    var file: String {
        let suffix = switch type {
        case "image/png": "png"
        case "application/pdf": "pdf"
        default: "jpg"
        }
        return "\(id.uuidString).\(suffix)"
    }

    /// Writes `data`, its content, as its file in `folder`, made where there is none; false where
    /// it can't be written.
    func write(_ data: Data, in folder: URL) -> Bool {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return (try? data.write(to: folder.appending(path: file), options: .atomic)) != nil
    }

    /// The record of an image, of its PNG or JPEG data.
    static func image(_ data: Data) -> Attachment {
        Attachment(kind: .image, type: ChatClient.mediaType(data), bytes: data.count)
    }

    /// Its pages, where they are known, and its size, as a PDF's chip says them.
    var details: String {
        [pages.map { $0 == 1 ? "1 page" : "\($0) pages" }, Format.size(Int64(bytes))].compactMap { $0 }.joined(separator: " · ")
    }
}

struct Conversation: Codable, Identifiable, Equatable {
    var id = UUID()
    var title = ""
    var updated = Date.now
    var messages: [ChatMessage] = []
    /// When it was pinned to the top of the list; nil where it is not.
    var pinned: Date?

    /// The files its attachments are kept in, by name in the attachments' folder
    /// (`ChatStore.attachments`).
    var attachmentFiles: [String] { messages.flatMap { ($0.attachments ?? []).map(\.file) } }
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

/// The conversations, one JSON file each in the app's folder, and the one request under way: the
/// reply that streams, and until the request is its chat's for good, what it would make of the chat.
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

    /// What the request under way would make of its chat, held apart from the conversations kept,
    /// which the list, the search and the files show, until it is the chat's for good: a message
    /// once the server takes the request, an answer asked again once it says something in the
    /// earlier one's place (`stream`). Until then nothing is kept that a refusal, a failure or a
    /// stop would have to take back: the chat stays as it was, and a new one was never listed,
    /// opened or written.
    private struct Pending {
        /// What a request asks for: a new chat's first message, which has no conversation to show
        /// in, and which the composer holds meanwhile; a later message, which its chat shows, the
        /// composer having given it up; or an answer asked again.
        enum Ask { case first, later, again }

        let conversation: Conversation
        let ask: Ask
        /// Tells that the message shows in its conversation, for the composer to give it up.
        let shown: () -> Void
    }

    private(set) var conversations: [Conversation] = []
    /// The open conversation; nil is a new one, kept once the server takes its first message.
    var selection: UUID?
    private(set) var reply: Reply?
    private var pending: Pending?

    /// The conversation whose request is under way, the open one or another, or a new chat's,
    /// which is none of those kept until the server takes its first message.
    var streaming: UUID? { reply?.conversation }

    /// Whether the composer's Stop stops the request under way: its reply streams in the open
    /// conversation, or the server has yet to take a new chat's first message, which the composer
    /// holds whichever chat is open (`sendingFirstMessage`). Stopped under another chat, the
    /// message comes back to the new chat, which opens (`stream`).
    var answering: Bool {
        guard let reply else { return false }
        return pending?.ask == .first || reply.conversation == selection
    }

    /// Whether the server has yet to take a new chat's first message: the composer holds the
    /// message meanwhile, as it was sent, whichever chat is open, Stop in Send's place.
    var sendingFirstMessage: Bool { pending?.ask == .first }

    private let folder: URL
    /// Where the conversations' attachments are kept, a file each (`Attachment.file`).
    nonisolated let attachments: URL
    private var replying: Task<Void, Never>?
    /// The requests for titles under way, by conversation (`title`).
    @ObservationIgnored private var titling: [UUID: Task<Void, Never>] = [:]
    /// Writes and removes the conversations' files in the order they change, off the main thread:
    /// a conversation with images takes milliseconds to encode and write, which a send and a
    /// reply's end would otherwise wait for.
    private let files = DispatchQueue(label: "ChatStore.files", qos: .utility)
    /// Whether every conversation's file could be read, so that `adoptFiles` knows each file a
    /// conversation names.
    @ObservationIgnored private var readAll = false

    init(folder: URL) {
        self.folder = folder
        attachments = folder.appending(path: "Attachments")
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        let kept = files.filter { $0.pathExtension == "json" }
        let read = kept.compactMap { try? JSONDecoder().decode(Conversation.self, from: Data(contentsOf: $0)) }
        conversations = read.map(Self.interrupted).sorted { $0.updated > $1.updated }
        readAll = read.count == kept.count
    }

    /// Brings the folder to how this app keeps it: the images an earlier app kept inside its
    /// conversations go into files of their own; then the attachments no conversation names go,
    /// as those of a draft the app quit with, unless a conversation that can't be read may name
    /// them. In that order: the removal, off the main thread, would take the moved images for
    /// strays. The app does it as it launches, before any draft (`AppState.launch`), once it is
    /// the one Splash that runs, from where it can: a copy that gives way or one on the disk image
    /// leaves the chats as the app that goes on running reads them.
    func adoptFiles() {
        for index in conversations.indices where conversations[index].messages.contains(where: { !$0.images.isEmpty }) {
            conversations[index] = Self.movingImages(of: conversations[index], to: attachments)
            write(conversations[index])
        }
        guard readAll else { return }
        let named = Set(conversations.flatMap(\.attachmentFiles))
        let attachments = attachments
        files.async {
            let stray = ((try? FileManager.default.contentsOfDirectory(at: attachments, includingPropertiesForKeys: nil)) ?? [])
                .filter { !named.contains($0.lastPathComponent) }
            stray.forEach { try? FileManager.default.removeItem(at: $0) }
        }
    }

    /// `conversation` with the images an earlier app kept in its messages moved into files of
    /// their own in `folder`, in their order; one that can't be written stays where it was.
    nonisolated static func movingImages(of conversation: Conversation, to folder: URL) -> Conversation {
        var conversation = conversation
        for index in conversation.messages.indices {
            let images = conversation.messages[index].images
            guard !images.isEmpty else { continue }
            var attached = conversation.messages[index].attachments ?? []
            var kept: [Data] = []
            for image in images {
                let attachment = Attachment.image(image)
                if attachment.write(image, in: folder) {
                    attached.append(attachment)
                } else {
                    kept.append(image)
                }
            }
            conversation.messages[index].attachments = attached.isEmpty ? nil : attached
            conversation.messages[index].images = kept
        }
        return conversation
    }

    /// The open conversation as it shows: with what the request under way would make of it, as a
    /// message shows in its chat while the server has yet to take it. A new chat shows none.
    var current: Conversation? {
        guard var open = conversations.first(where: { $0.id == selection }) else { return nil }
        if let pending, pending.conversation.id == open.id { open.messages = pending.conversation.messages }
        return open
    }

    /// The conversations whose title or messages hold `query`, as the list's search finds them, in
    /// the list's order; every one for an empty query.
    func matching(_ query: String) -> [Conversation] {
        guard !query.isEmpty else { return conversations }
        return conversations.filter { conversation in
            conversation.title.localizedStandardContains(query)
                || conversation.messages.contains { $0.text.localizedStandardContains(query) }
        }
    }

    func newChat() {
        selection = nil
    }

    /// Deletes the conversations of `ids` it holds, and their files, stopping a reply that streams
    /// in one; the open one deleted, a new chat opens.
    func delete(_ ids: Set<UUID>) {
        if let streaming, ids.contains(streaming) { stop() }
        for id in ids { titling.removeValue(forKey: id)?.cancel() }
        let deleted = conversations.filter { ids.contains($0.id) }
        let removed = deleted.map { file($0.id) }
            + deleted.flatMap(\.attachmentFiles).map { attachments.appending(path: $0) }
        conversations.removeAll { ids.contains($0.id) }
        files.async { removed.forEach { try? FileManager.default.removeItem(at: $0) } }
        if let selection, ids.contains(selection) { self.selection = nil }
    }

    /// Renames conversation `id` `title`, trimmed; a name of nothing leaves it as it was.
    func rename(_ id: UUID, to title: String) {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        titling.removeValue(forKey: id)?.cancel()
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].title = title
        write(conversations[index])
    }

    /// Pins conversation `id` to the top of the list, or takes it off there.
    func pin(_ id: UUID, _ pinned: Bool) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].pinned = pinned ? .now : nil
        write(conversations[index])
    }

    /// What a request that came to nothing gives back to the composer, in its conversation, nil
    /// for a new one: the message the composer gave up, where it did, and why the request came to
    /// nothing, unless it was stopped. A new chat's first message, which the composer still
    /// holds, and a reply asked again give back only why. Its chat is the open one by then
    /// (`stream`).
    struct Refused: Equatable {
        let text: String
        let attachments: [Attachment]
        let reason: String?
        let conversation: UUID?
    }

    /// Sends `text` and `attachments`, kept in the attachments folder already, in the open
    /// conversation, or a new one, to the model `server` serves, and streams its reply into the
    /// conversation. The message is its chat's once the server takes the request (`stream`): an
    /// open conversation shows it at once, and a new chat becomes a conversation, listed, written
    /// and open, only then. `shown` tells that the message shows in its conversation, for the
    /// composer to give it up: at once in an open one, and for a new chat once the server has
    /// taken it. A request the server refuses, that never reaches it or that is stopped before
    /// it takes it leaves the chats as they were, and goes to `refused`, with what the composer
    /// gave up of it and why, its chat open again (`stream`).
    func send(_ text: String, attachments: [Attachment], to server: ChatTarget,
              refused: @escaping (Refused) -> Void = { _ in }, shown: @escaping () -> Void = {}) {
        guard streaming == nil else { return }
        let open = current
        var conversation = open ?? Conversation()
        if conversation.title.isEmpty { conversation.title = Self.firstLine(text, attachments: attachments) }
        conversation.messages.append(ChatMessage(role: .user, text: text, attachments: attachments.isEmpty ? nil : attachments))
        stream(conversation, new: open == nil, to: server, shown: shown) { reason in
            if let open {
                refused(Refused(text: text, attachments: attachments, reason: reason, conversation: open.id))
            } else if let reason {
                refused(Refused(text: "", attachments: [], reason: reason, conversation: nil))
            }
        }
    }

    /// A chat's title until the model sums it up (`title`): the first line of its first message,
    /// or, for attachments alone, the first PDF's name, else Image.
    static func firstLine(_ text: String, attachments: [Attachment] = []) -> String {
        let line = text.split(separator: "\n").first.map(String.init) ?? attachments.lazy.compactMap(\.name).first ?? "Image"
        return String(line.prefix(80))
    }

    /// Titles conversation `id` as the model sums up its first exchange, `question` and `answer`,
    /// while its title is still the first line; renaming or deleting it meanwhile cancels the
    /// request, and where the model gives no title, the first line stays.
    private func title(_ id: UUID, question: ChatMessage, answer: String, with server: ChatTarget) {
        guard conversations.first(where: { $0.id == id })?.title == Self.firstLine(question.text, attachments: question.attachments ?? [])
        else { return }
        titling[id]?.cancel()
        titling[id] = Task { [weak self] in
            let title = try? await ChatClient.title(question: question.text, answer: answer, with: server)
            guard let self, !Task.isCancelled else { return }
            titling[id] = nil
            guard let title, let index = conversations.firstIndex(where: { $0.id == id }) else { return }
            conversations[index].title = title
            write(conversations[index])
        }
    }

    /// Answers the open conversation's last message again, in place of the last answer, which
    /// stays the chat's until the new one says something (`stream`): where the server refuses the
    /// request, or the new answer ends before it says anything, stopped or failed, the earlier
    /// one shows again, and a refusal or a failure goes to `refused` with why, its chat open
    /// again (`stream`). Stopped, it gives nothing back, and the open chat stays: the request's
    /// end alone tells of it (`streaming`).
    func retry(with server: ChatTarget, refused: @escaping (Refused) -> Void = { _ in }) {
        guard streaming == nil, var conversation = current, let previous = conversation.messages.last,
              previous.role == .assistant else { return }
        conversation.messages.removeLast()
        let id = conversation.id
        stream(conversation, new: false, to: server, instead: previous) { reason in
            if let reason { refused(Refused(text: "", attachments: [], reason: reason, conversation: id)) }
        }
    }

    /// Stops the reply; what it said so far stays.
    func stop() {
        replying?.cancel()
    }

    /// Stops the request under way as the app quits. A later message the server has yet to take
    /// is its chat's first, as though the server had taken it: its chat shows it, and the
    /// composer, which gave it up, goes with the app, as would the attachments no chat names
    /// (`adoptFiles`). A new chat's first message, which the composer still holds, goes with the
    /// composer, as any draft does, and an answer asked again leaves the earlier one.
    func quit() {
        if pending?.ask == .later { keep() }
        stop()
    }

    /// Streams the reply to `conversation`'s last message, asked again `instead` of an earlier
    /// answer where one is given. The open conversation shows the request at once, which `shown`
    /// tells; a `new` chat, which has no conversation, shows nothing of it. What the request
    /// makes of its chat is the chat's for good (`keep`) once nothing can take it back: a message
    /// once the server takes the request (`ChatClient.reply`), and an answer asked again once it
    /// says something in the earlier one's place. A request that ends before, refused, failed or
    /// stopped, leaves every chat as it was, and goes to `back` with why, nil for a stop, unless
    /// its chat was deleted meanwhile, which its message went with. What it gives back, a message
    /// or why, comes back to its chat, which opens first; an answer asked again with no why, as
    /// one stopped, gives nothing back, and opens nothing. One that ends after leaves its answer
    /// in the chat, saying why it stopped short.
    private func stream(_ conversation: Conversation, new: Bool, to server: ChatTarget, instead previous: ChatMessage? = nil,
                        shown: @escaping () -> Void = {}, back: @escaping (String?) -> Void) {
        var conversation = conversation
        let history = conversation.messages
        var answer = ChatMessage(role: .assistant, text: "", model: server.model)
        conversation.messages.append(answer)
        pending = Pending(conversation: conversation, ask: new ? .first : previous == nil ? .later : .again, shown: shown)
        if !new { shown() }
        let reply = Reply(answer, in: conversation.id)
        self.reply = reply
        let files = attachments
        replying = Task { [weak self] in
            var drawn = ContinuousClock.now
            var thinking: Date?
            var read: Double?
            do {
                let deltas = try await ChatClient.reply(to: history, with: server, files: files)
                // The server has taken the request: a message is its chat's, unless stopped by then.
                if previous == nil, !Task.isCancelled { self?.keep() }
                for try await delta in deltas {
                    // Stopped, it reads no more: what it said stays, and what was not kept is not.
                    if Task.isCancelled { break }
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
                    // An answer asked again is its chat's once it says something.
                    if !answer.isEmpty { self?.keep() }
                    // A redraw per token would be wasted work: the page lets out what comes at an
                    // even pace (page.js).
                    if ContinuousClock.now - drawn > .milliseconds(50) {
                        drawn = .now
                        reply.live.read = read
                        reply.message = answer
                    }
                }
            } catch {
                // A request stopped ends as it can, which is no failure.
                if !Task.isCancelled { answer.error = error.localizedDescription }
            }
            if Task.isCancelled { answer.stopped = true }
            guard let self else { return }
            self.reply = nil
            if pending != nil {
                pending = nil
                let listed = conversations.contains { $0.id == conversation.id }
                guard new || listed else { return }
                // What comes back comes back to its chat, which opens first, the new chat for a new
                // chat's first message, which nothing lists: the composer says why in the chat it
                // is about alone, and Send sends what it holds to the open chat. Under a chat
                // opened while the server had the message, a message refused came back with its
                // reason unsaid, and one stopped from elsewhere, as a switch of models stops it,
                // with nothing said, each for Send to send there. An answer asked again with no
                // why, as one stopped, gives nothing back, and opens nothing.
                if previous == nil || answer.error != nil { selection = listed ? conversation.id : nil }
                back(answer.error)
                return
            }
            // A reply that ended while the model still thought.
            if let thinking, answer.thinkingTime == nil { answer.thinkingTime = Date.now.timeIntervalSince(thinking) }
            finish(answer, in: conversation.id)
            // A first exchange answered in full gets the title the model sums it up in.
            if history.count == 1, answer.error == nil, answer.stopped == nil, !answer.text.isEmpty {
                title(conversation.id, question: history[0], answer: answer.text, with: server)
            }
        }
    }

    /// Makes what the request under way would make of its chat the chat's, once: its messages
    /// in the conversation as it is now, so that a pin, a name or a title it took meanwhile
    /// stays, and its title where the chat has none; first in the list, and written. A chat
    /// deleted meanwhile stays gone. A new chat becomes a conversation, which opens where the
    /// new chat is still open, and tells that it shows its message; another chat opened
    /// meanwhile stays open.
    private func keep() {
        guard let pending else { return }
        self.pending = nil
        guard var conversation = pending.ask == .first ? pending.conversation
            : conversations.first(where: { $0.id == pending.conversation.id }) else { return }
        conversation.messages = pending.conversation.messages
        if conversation.title.isEmpty { conversation.title = pending.conversation.title }
        conversation.updated = .now
        put(conversation)
        guard pending.ask == .first else { return }
        if selection == nil { selection = conversation.id }
        pending.shown()
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
    static func countTokens(_ messages: [ChatMessage], with server: ChatTarget, files: URL,
                            session: URLSession = .shared) async throws -> Int {
        let request = try countRequest(messages, with: server, files: files)
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

    /// The request counting `messages`, their attachments in `files`: the prompt `reply` sends, as
    /// it sends it.
    static func countRequest(_ messages: [ChatMessage], with server: ChatTarget, files: URL) throws -> URLRequest {
        let json = try JSONSerialization.data(withJSONObject: prompt(messages, effort: server.effort, vision: server.acceptsImages,
                                                                     files: files))
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

    /// Asks `server` for the reply to `messages`, their attachments in `files`, and returns once
    /// the server has taken the request, with the reply as it streams; throws where it hasn't,
    /// with the server's error in the app's words (`message`), or why it could not be reached.
    /// The server takes a request as it starts its response, with status 200: an error it finds
    /// before, it answers with in the response's place, under the error's status, and one it finds
    /// after comes as an event of the stream (`parse`), which ends the reply (`guarded_stream` in
    /// server/server.py). It starts the response as its engine starts the request, or with a
    /// keepalive once the request has waited 2 s for the engine, as while a model that slept
    /// loads again. So the response's start is the first word no refusal can follow, and it
    /// comes within about 2 s of the server reading the request.
    static func reply(to messages: [ChatMessage], with server: ChatTarget, files: URL) async throws -> AsyncThrowingStream<Delta, Error> {
        let json = try body(messages, effort: server.effort, vision: server.acceptsImages, files: files)
        let (bytes, response) = try await URLSession.shared.bytes(for: request("v1/chat/completions", json, to: server, timeout: 3600))
        if let status = (response as? HTTPURLResponse)?.statusCode, status != 200 {
            var data = Data()
            for try await byte in bytes { data.append(byte) }
            throw Failure(errorDescription: message(of: data, for: messages, server: server)
                ?? HTTPURLResponse.localizedString(forStatusCode: status))
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
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

    /// A title of a few words for a chat that begins with `question` and `answer`, as the model
    /// sums them up; nil where its reply holds none, or its few tokens cut it short.
    static func title(question: String, answer: String, with server: ChatTarget, session: URLSession = .shared) async throws -> String? {
        let ask = try titleBody(question: question, answer: answer)
        let (data, response) = try await session.data(for: request("v1/chat/completions", ask, to: server, timeout: 120))
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        struct Completion: Decodable {
            struct Choice: Decodable {
                struct Message: Decodable { let content: String? }
                let message: Message
                let finish_reason: String?
            }
            let choices: [Choice]
        }
        guard let choice = (try? JSONDecoder().decode(Completion.self, from: data))?.choices.first,
              choice.finish_reason != "length" else { return nil }
        return choice.message.content.flatMap(title(in:))
    }

    /// The request for a title: the start of a chat's first exchange and what to answer, in the
    /// background, where a request of the user's comes first, without thinking, in a few tokens and
    /// not streamed.
    static func titleBody(question: String, answer: String) throws -> Data {
        let ask = "Give the chat below a title of at most six words, in the chat's language. Reply with the title alone."
            + "\n\nUser: \(question.prefix(1000))\n\nAssistant: \(answer.prefix(1000))"
        return try JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": ask]], "priority": "background", "reasoning_effort": "none",
            "thinking_token_budget": 0, "max_tokens": 32, "temperature": 0.2,
        ] as [String: Any])
    }

    /// The title in a model's reply: its first line that holds any, without a Title: label, quotes,
    /// Markdown's marks or a closing stop, in at most 80 characters; nil where none is left.
    static func title(in reply: String) -> String? {
        // A heading's marks lead it, emphasis and quotes wrap it; a # within, as C#'s, stays.
        func tidy(_ text: some StringProtocol) -> String {
            String(text.drop { $0 == "#" }).trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’「」*`").union(.whitespaces))
        }
        var line = tidy(reply.split(whereSeparator: \.isNewline).first { !tidy($0).isEmpty } ?? "")
        if line.lowercased().hasPrefix("title:") { line = tidy(line.dropFirst(6)) }
        while let last = line.last, ".。".contains(last) { line = tidy(line.dropLast()) }
        return line.isEmpty ? nil : String(line.prefix(80))
    }

    /// The request for `messages`, their attachments in `files`: the prompt, streamed, with its
    /// progress and the usage asked for.
    static func body(_ messages: [ChatMessage], effort: String, vision: Bool, files: URL) throws -> Data {
        var body = prompt(messages, effort: effort, vision: vision, files: files)
        body["stream"] = true
        body["stream_options"] = ["include_usage": true]
        body["return_progress"] = true
        return try JSONSerialization.data(withJSONObject: body)
    }

    /// The prompt of `messages`: their attachments, read from `files`, as data URLs, or a note in
    /// their place for a model without `vision`; earlier answers with what the model thought, empty
    /// ones left out; and the reasoning effort unless it is the model's default.
    private static func prompt(_ messages: [ChatMessage], effort: String, vision: Bool, files: URL) -> [String: Any] {
        let wire: [[String: Any]] = messages.compactMap { message in
            let attachments = message.attachments ?? []
            switch message.role {
            case .user where !attachments.isEmpty && vision:
                let parts: [[String: Any]] = attachments.compactMap { attachment in
                    guard let data = try? Data(contentsOf: files.appending(path: attachment.file)) else { return nil }
                    let url = "data:\(attachment.type);base64,\(data.base64EncodedString())"
                    return switch attachment.kind {
                    case .image: ["type": "image_url", "image_url": ["url": url]]
                    case .pdf: ["type": "file", "file": ["filename": attachment.name ?? "document.pdf", "file_data": url]]
                    }
                }
                return ["role": "user", "content": parts + [["type": "text", "text": message.text]]]
            case .user where !attachments.isEmpty:
                let note = unseen(attachments)
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

    /// What a model that sees no attachments is told in their place.
    private static func unseen(_ attachments: [Attachment]) -> String {
        let images = attachments.count { $0.kind == .image }, pdfs = attachments.count { $0.kind == .pdf }
        let notes = [images == 1 ? "[An image this model can’t see]" : images > 1 ? "[\(images) images this model can’t see]" : nil,
                     pdfs == 1 ? "[A PDF this model can’t read]" : pdfs > 1 ? "[\(pdfs) PDFs this model can’t read]" : nil]
        return notes.compactMap { $0 }.joined(separator: " ")
    }

    /// The type of an image the app keeps, PNG or JPEG.
    static func mediaType(_ image: Data) -> String {
        image.starts(with: [0x89, 0x50, 0x4E, 0x47]) ? "image/png" : "image/jpeg"
    }
}
