import Foundation
import Observation

/// How much of the running model's context the next prompt takes. Thinking and the reply share
/// what it leaves: a prompt that leaves less than the reply allowance still goes, as a reply may be
/// short, but one the context cannot hold does not.
struct ContextUsage: Equatable, Sendable {
    /// How full the context is: room for a reply, less than the allowance, or no room at all.
    enum Level { case normal, tight, full }

    let input: Int
    let limit: Int

    /// The room a reply is meant to have: 4,096 tokens, or an eighth of a smaller context.
    var allowance: Int { min(4096, limit / 8) }
    var fits: Bool { input < limit }
    var level: Level { !fits ? .full : input > limit - allowance ? .tight : .normal }
    /// The share of the context the prompt takes, at most all of it.
    var fraction: Double { min(1, Double(input) / Double(max(1, limit))) }

    /// The exact figures: 12,345 of 262,144 tokens · 5%.
    var summary: String {
        let share = (Double(input) / Double(max(1, limit))).formatted(.percent.precision(.fractionLength(0)))
        return "\(input.formatted()) of \(limit.formatted()) tokens · \(share)"
    }

    /// What a prompt short of room means for the reply, nil while there is room.
    var note: String? {
        switch level {
        case .normal: nil
        case .tight: "\((limit - input).formatted()) tokens are left for thinking and the reply."
        case .full: "Too long for the model’s context. Start a new chat, or shorten the message."
        }
    }

    /// The figures and the note, as the composer's ring says them.
    var explanation: String {
        [summary, note].compactMap(\.self).joined(separator: "\n")
    }
}

/// Counts the next prompt, the open conversation and the draft as the model reads them, after a
/// pause in typing. A count belongs to its whole prompt: history, draft, images, model, reasoning
/// effort and context; one finishing after the prompt changed is dropped.
@MainActor @Observable
final class ChatContext {
    struct Request: Equatable, Sendable {
        let history: [ChatMessage]
        let text: String
        let images: [Data]
        let target: ChatTarget
        /// A restarted model may use another tokenizer even under the same name.
        var started: Date? = nil

        var messages: [ChatMessage] {
            history + (text.isEmpty && images.isEmpty ? [] : [ChatMessage(role: .user, text: text, images: images)])
        }
    }

    typealias Counter = @Sendable (Request) async throws -> Int

    /// The last prompt counted, and its count.
    private(set) var counted: (request: Request, usage: ContextUsage)?
    @ObservationIgnored private let counter: Counter
    @ObservationIgnored private let delay: Duration
    /// The prompt being counted, or waiting for typing to pause.
    @ObservationIgnored private var pending: Request?
    @ObservationIgnored private var task: Task<Void, Never>?

    init(delay: Duration = .milliseconds(300),
         counter: @escaping Counter = { try await ChatClient.countTokens($0.messages, with: $0.target) }) {
        self.delay = delay
        self.counter = counter
    }

    /// The count of exactly `request`, nil until it is counted.
    func usage(for request: Request) -> ContextUsage? {
        counted.flatMap { $0.request == request ? $0.usage : nil }
    }

    /// The count to show for a conversation of `history` with `target`: the last one, while the
    /// next is counted, a reply streams or another reasoning effort is chosen, as long as the
    /// conversation has only grown since by the draft counted and its reply, or had its last answer
    /// asked again.
    func shown(history: [ChatMessage], target: ChatTarget, started: Date?) -> ContextUsage? {
        guard let counted, counted.request.target.sameModel(as: target), counted.request.started == started
        else { return nil }
        let base = counted.request.history
        let grown = history.dropFirst(base.count)
        // A new chat's count shows only on the chat its message started: other chats share its
        // empty history. Message ids tell any other conversation apart.
        let sent = history.starts(with: base) && grown.count <= 2 && (!base.isEmpty || grown.first.map {
            $0.role == .user && $0.text == counted.request.text && $0.images == counted.request.images
        } ?? true)
        let askedAgain = base.last?.role == .assistant && history.count == base.count
            && history.dropLast().elementsEqual(base.dropLast())
        return sent || askedAgain ? counted.usage : nil
    }

    /// Counts `request` once typing pauses, in place of the prompt counted before; nil counts
    /// nothing. A count that fails leaves none: the server checks every prompt it is sent.
    func refresh(_ request: Request?) {
        guard request != pending else { return }
        task?.cancel()
        task = nil
        pending = request
        guard let request, usage(for: request) == nil, let limit = request.target.context, limit > 0 else { return }
        let counter = counter, delay = delay
        task = Task { [weak self] in
            guard (try? await Task.sleep(for: delay)) != nil else { return }
            let count = try? await counter(request)
            guard let self, !Task.isCancelled, pending == request else { return }
            counted = count.map { (request, ContextUsage(input: $0, limit: limit)) }
        }
    }
}

extension ChatTarget {
    /// Whether `other` is the same model, served the same way, whatever reasoning effort it asks for.
    func sameModel(as other: ChatTarget) -> Bool {
        url == other.url && model == other.model && acceptsImages == other.acceptsImages && context == other.context
            && apiKey == other.apiKey
    }
}
