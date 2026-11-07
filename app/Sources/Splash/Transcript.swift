import SwiftUI
import WebKit

/// The open conversation, shown as a page of the app's (`WebPage`, Transcript/transcript.html):
/// only data goes to it, as JSON: the whole conversation when it changes, and while a reply
/// streams, that reply alone. The page outlives the Chat it shows in, so that showing the Chat
/// again, or another conversation, never loads it again. The user's images it loads from the app
/// (`ChatImages`).
@MainActor
final class Transcript {
    /// What Ask Again asks of the app.
    var askAgain: () -> Void = {}

    let page: WebPage
    private let images = ChatImages()
    /// What the page shows, or will once it has loaded: the conversation as kept, and the reply
    /// that streams in it, in place of its last message.
    private var shown: TranscriptPage?
    private var reply: TranscriptMessage?

    init(runtime: Runtime) {
        page = WebPage("transcript", in: "Transcript", runtime: runtime,
                       glyphs: ["again": Glyph("arrow.clockwise", size: 12)], schemes: [ChatImages.scheme: images])
        page.receive = { [weak self] message in
            if message["askAgain"] != nil { self?.askAgain() }
        }
        page.didLoad = { [weak self] in
            guard let self, let shown else { return }
            page.call("transcript.show(JSON.parse(json))", with: shown.streaming(reply))
        }
    }

    /// Shows `conversation` and `reply`, the reply that streams in it where it is its last
    /// message: the reply alone while only it changed, as each flush of it does, which the
    /// conversation, the same, tells at once; the whole conversation otherwise, but its last
    /// message alone where only that changed, as a reply that ended; nothing when the page shows
    /// them already.
    func show(_ conversation: TranscriptPage, reply: TranscriptMessage?) {
        let reply = reply.flatMap { conversation.messages.last?.id == $0.id ? $0 : nil }
        defer { (shown, self.reply) = (conversation, reply) }
        if conversation == shown {
            guard reply != self.reply, let last = reply ?? conversation.messages.last else { return }
            return page.call("transcript.update(JSON.parse(json))", with: last)
        }
        images.images = Dictionary(conversation.messages.compactMap { $0.images.isEmpty ? nil : ($0.id, $0.images) }) { _, last in last }
        switch Self.change(from: shown?.streaming(self.reply), to: conversation.streaming(reply)) {
        case .none: break
        case .message(let message): page.call("transcript.update(JSON.parse(json))", with: message)
        case .page(let conversation): page.call("transcript.show(JSON.parse(json))", with: conversation)
        }
    }

    enum Change: Equatable {
        case none
        /// The last message alone, as a reply that ended.
        case message(TranscriptMessage)
        case page(TranscriptPage)
    }

    /// What the page needs to be sent to go from `shown` to `page`.
    static func change(from shown: TranscriptPage?, to page: TranscriptPage) -> Change {
        guard let shown, shown.id == page.id, shown.askAgain == page.askAgain,
              shown.messages.count == page.messages.count, let last = page.messages.last,
              shown.messages.last?.id == last.id, shown.messages.dropLast() == page.messages.dropLast()
        else { return .page(page) }
        return shown.messages.last == last ? .none : .message(last)
    }
}

/// The transcript in the Chat's layout: a view of its own that takes the transcript's one web view
/// in, so that the page outlives the Chat.
struct TranscriptView: NSViewRepresentable {
    let transcript: Transcript
    let page: TranscriptPage
    /// The reply that streams in it.
    var reply: TranscriptMessage?

    func makeNSView(context: Context) -> NSView {
        transcript.page.container()
    }

    func updateNSView(_ container: NSView, context: Context) {
        transcript.show(page, reply: reply)
    }
}

/// A conversation as the transcript's page shows it.
struct TranscriptPage: Encodable, Equatable {
    /// Ask Again, which the last reply has: whether it can ask now, and what it says.
    struct AskAgain: Encodable, Equatable {
        let enabled: Bool
        let help: String
    }

    let id: UUID
    private(set) var messages: [TranscriptMessage]
    let askAgain: AskAgain

    /// `conversation` as kept, its models named by `name`.
    init(_ conversation: Conversation, askAgain: AskAgain, name: (String) -> String) {
        id = conversation.id
        messages = conversation.messages.map { TranscriptMessage($0, live: nil, asleep: false, name: name) }
        self.askAgain = askAgain
    }

    /// The page with `reply`, the reply that streams, in place of its last message.
    func streaming(_ reply: TranscriptMessage?) -> TranscriptPage {
        guard let reply, messages.last?.id == reply.id else { return self }
        var page = self
        page.messages[page.messages.count - 1] = reply
        return page
    }
}

/// A message as the page shows it.
struct TranscriptMessage: Encodable, Equatable {
    let id: UUID
    let role: ChatMessage.Role
    let text: String
    let reasoning: String
    /// The user's images, sent as the addresses the page loads them from (`ChatImages`).
    let images: [Data]
    let error: String?
    /// The line over a reply, if it has one.
    let line: ReplyLine?
    /// Whether the reply streams.
    let live: Bool
    /// The model that answered, how fast, and whether it was stopped.
    let details: String

    init(_ message: ChatMessage, live: ChatStore.Live?, asleep: Bool, name: (String) -> String) {
        id = message.id
        role = message.role
        text = message.text
        reasoning = message.reasoning
        images = message.images
        error = message.error
        line = message.role == .assistant ? ReplyLine(message, live: live, asleep: asleep) : nil
        self.live = live != nil
        details = [message.model.map(name), message.tokensPerSecond.map { "\(Format.whole($0)) tok/s" },
                   message.stopped == true ? "Stopped" : nil]
            .compactMap { $0 }.joined(separator: " · ")
    }

    private enum CodingKeys: CodingKey {
        case id, role, text, reasoning, images, error, line, live, details
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(role, forKey: .role)
        try container.encode(text, forKey: .text)
        try container.encode(reasoning, forKey: .reasoning)
        try container.encode(images.indices.map { ChatImages.url(message: id, index: $0) }, forKey: .images)
        try container.encode(error, forKey: .error)
        try container.encode(line, forKey: .line)
        try container.encode(live, forKey: .live)
        try container.encode(details, forKey: .details)
    }
}

/// The line over a reply, of what comes before its answer: Loading into memory while a model the
/// idle release put to sleep loads again, Reading while the prompt is read, then Thinking while the
/// model thinks, and once it answers, what the thinking took. While it waits, the page counts the
/// seconds since `since` itself, once a second, from the first.
struct ReplyLine: Encodable, Equatable {
    let verb: String
    /// Its figures, in a fainter tone.
    let facts: String?
    let since: Date?

    // Every key, null for none, as the page reads them.
    private enum CodingKeys: CodingKey {
        case verb, facts, since
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(verb, forKey: .verb)
        try container.encode(facts, forKey: .facts)
        try container.encode(since, forKey: .since)
    }
}

extension ReplyLine {
    /// The line over `message`, streaming while `live` is given; nil for a reply that neither
    /// waits for its answer nor thought.
    init?(_ message: ChatMessage, live: ChatStore.Live?, asleep: Bool) {
        // Once the answer comes, the line says what the thinking took.
        let waiting = message.text.isEmpty ? live : nil
        if let live = waiting, message.reasoning.isEmpty {
            // After the idle release, the model loads again before it reads.
            verb = asleep ? "Loading into memory" : "Reading"
            facts = asleep ? nil : live.read.map { "\(Int($0 * 100))%" }
            since = facts == nil ? live.asked : nil
        } else if let started = waiting?.thinking {
            verb = "Thinking"
            facts = nil
            since = started
        } else if !message.reasoning.isEmpty {
            verb = "Thinking"
            let facts = [message.reasoningTokens.map { "\(Format.count($0)) tokens" }, message.thinkingTime.map(Format.seconds)]
                .compactMap { $0 }.joined(separator: " · ")
            self.facts = facts.isEmpty ? nil : facts
            since = nil
        } else {
            return nil
        }
    }
}

/// The user's images in the Chat, which the page loads as chat-image://MESSAGE/INDEX: those of the
/// conversation it shows, from what the app holds, so that the data sent to the page holds no
/// image, nor its comparisons of a message as last drawn, and an image is sent once, as it shows.
@MainActor
final class ChatImages: NSObject, WKURLSchemeHandler {
    nonisolated static let scheme = "chat-image"

    /// The images of the conversation shown, by message.
    var images: [UUID: [Data]] = [:]

    /// Where the page loads the image at `index` of `message`.
    nonisolated static func url(message: UUID, index: Int) -> String {
        "\(scheme)://\(message.uuidString)/\(index)"
    }

    /// The image `url` names, if it names one of the conversation shown.
    func image(_ url: URL) -> Data? {
        guard url.scheme == Self.scheme, let message = url.host().flatMap(UUID.init(uuidString:)),
              let index = Int(url.lastPathComponent), let images = images[message], images.indices.contains(index)
        else { return nil }
        return images[index]
    }

    func webView(_ webView: WKWebView, start task: any WKURLSchemeTask) {
        guard let url = task.request.url, let image = image(url) else {
            return task.didFailWithError(URLError(.fileDoesNotExist))
        }
        task.didReceive(URLResponse(url: url, mimeType: ChatClient.mediaType(image), expectedContentLength: image.count,
                                    textEncodingName: nil))
        task.didReceive(image)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: any WKURLSchemeTask) {}
}
