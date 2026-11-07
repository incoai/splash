import Foundation
import Synchronization
import Testing

@testable import Splash

@MainActor
struct ChatContextTests {
    private let target = ChatTarget(url: URL(string: "http://127.0.0.1:9")!, model: "a/b", acceptsImages: true,
                                    effort: "high", context: 32768, apiKey: "secret")

    private func request(_ text: String = "Hello", history: [ChatMessage] = []) -> ChatContext.Request {
        ChatContext.Request(history: history, text: text, images: [], target: target)
    }

    /// The ring's three states: room for a reply, less than the reply allowance, no room at all.
    @Test func theRingFillsAndWarnsAsThePromptGrows() {
        let room = ContextUsage(input: 28672, limit: 32768)
        #expect(room.allowance == 4096 && room.level == .normal && room.fits && room.note == nil)
        #expect(room.summary == "\(28672.formatted()) of \(32768.formatted()) tokens · \(percent(0.875))")
        #expect(room.fraction == 0.875 && room.explanation == room.summary)
        let tight = ContextUsage(input: 28673, limit: 32768)
        #expect(tight.level == .tight && tight.fits)
        #expect(tight.note == "\(4095.formatted()) tokens are left for thinking and the reply.")
        #expect(tight.explanation == "\(tight.summary)\n\(tight.note!)")
        #expect(ContextUsage(input: 32767, limit: 32768).level == .tight)
        let full = ContextUsage(input: 32768, limit: 32768)
        #expect(full.level == .full && !full.fits && full.note?.hasPrefix("Too long") == true)
        let over = ContextUsage(input: 40000, limit: 32768)
        #expect(over.level == .full && over.fraction == 1 && over.summary.hasSuffix(percent(1.22)))
        // A small context keeps an eighth of itself for the reply.
        #expect(ContextUsage(input: 896, limit: 1024).level == .normal)
        #expect(ContextUsage(input: 897, limit: 1024).level == .tight)
        #expect(ContextUsage(input: 0, limit: 262_144).summary == "0 of \(262_144.formatted()) tokens · \(percent(0))")
    }

    @Test func countsTheSameImagesHistoryAndReasoningAsGeneration() throws {
        let history = [ChatMessage(role: .user, text: "你好", images: [Data([0x89, 0x50, 0x4E, 0x47])]),
                       ChatMessage(role: .assistant, text: "A cat", reasoning: "Whiskers"),
                       ChatMessage(role: .assistant, text: "", error: "Interrupted")]
        let input = request("Sure?", history: history)
        let count = try ChatClient.countRequest(input.messages, with: input.target)
        var generation = try #require(JSONSerialization.jsonObject(with: ChatClient.body(input.messages, effort: "high", images: true)) as? [String: Any])
        for key in ["stream", "stream_options", "return_progress"] { generation.removeValue(forKey: key) }
        let body = try #require(count.httpBody)
        let counted = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect((counted as NSDictionary).isEqual(to: generation))
        #expect(count.url?.path == "/v1/chat/completions/count_tokens")
        #expect(count.httpMethod == "POST")
        #expect(count.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
    }

    /// A count belongs to its whole prompt: another draft, model, image or start has none.
    @Test func aCountBelongsToItsPrompt() async throws {
        let probe = CounterProbe()
        let context = ChatContext(delay: .zero) { try await probe.count($0) }
        let input = request()
        context.refresh(input)
        try await waitFor(context.usage(for: input) != nil)
        #expect(context.usage(for: input)?.input == input.text.utf8.count)
        // The same prompt again is not counted again.
        context.refresh(input)
        #expect(await probe.calls == 1)
        #expect(context.usage(for: request("Different")) == nil)
        var smaller = target
        smaller.context = 1024
        #expect(context.usage(for: .init(history: [], text: input.text, images: [], target: smaller)) == nil)
        let other = ChatTarget(url: target.url, model: "a/other", acceptsImages: true, effort: "high", context: 32768)
        #expect(context.usage(for: .init(history: [], text: input.text, images: [], target: other)) == nil)
        #expect(context.usage(for: .init(history: [], text: input.text, images: [Data([1])], target: target)) == nil)
        #expect(context.usage(for: .init(history: [], text: input.text, images: [], target: target, started: .now)) == nil)
    }

    @Test func anOldCountCannotReplaceANewerPromptsCount() async throws {
        let probe = SuspendedCounter()
        let context = ChatContext(delay: .zero) { await probe.count($0.text) }
        let old = request("old"), current = request("current")
        context.refresh(old)
        try await waitFor(await probe.started("old"))
        context.refresh(current)
        try await waitFor(await probe.started("current"))
        await probe.finish("current", with: 20)
        try await waitFor(context.usage(for: current) != nil)
        // This counter ignores cancellation, as an HTTP response may arrive meanwhile: its count
        // must still be dropped.
        await probe.finish("old", with: 99999)
        try await Task.sleep(for: .milliseconds(50))
        #expect(context.usage(for: current)?.input == 20)
        #expect(context.usage(for: old) == nil)
    }

    /// Typing waits for a pause; a count that fails leaves none, not the last prompt's, and the
    /// next edit counts again.
    @Test func typingWaitsAndAFailedCountLeavesNone() async throws {
        let probe = CounterProbe(failing: ["Hey?"])
        let context = ChatContext(delay: .milliseconds(300)) { try await probe.count($0) }
        let history = [ChatMessage(role: .user, text: "Hi"), ChatMessage(role: .assistant, text: "Hello.")]
        context.refresh(request("H", history: history))
        context.refresh(request("He", history: history))
        let input = request("Hey", history: history)
        context.refresh(input)
        try await Task.sleep(for: .milliseconds(100))
        #expect(await probe.calls == 0)
        try await waitFor(context.usage(for: input) != nil)
        #expect(await probe.calls == 1)
        #expect(context.shown(history: history, target: target, started: nil) != nil)
        context.refresh(request("Hey?", history: history))
        try await waitFor(context.counted == nil)
        #expect(await probe.calls == 2)
        #expect(context.shown(history: history, target: target, started: nil) == nil)
        let next = request("Hey!", history: history)
        context.refresh(next)
        try await waitFor(context.usage(for: next) != nil)
        #expect(await probe.calls == 3)
    }

    /// The ring keeps the last count while the next is counted, the reply streams, the last answer
    /// is asked again or another reasoning effort is chosen, and drops it for another conversation
    /// or model.
    @Test func theRingKeepsTheLastCountOfItsConversation() async throws {
        let context = ChatContext(delay: .zero) { $0.text.utf8.count + $0.history.count }
        let history = [ChatMessage(role: .user, text: "Hi"), ChatMessage(role: .assistant, text: "Hello.")]
        context.refresh(request("Next", history: history))
        try await waitFor(context.counted != nil)
        let usage = context.counted?.usage
        #expect(context.shown(history: history, target: target, started: nil) == usage)
        // The question sent and its reply streaming.
        let sent = history + [ChatMessage(role: .user, text: "Next"), ChatMessage(role: .assistant, text: "Wo")]
        #expect(context.shown(history: sent, target: target, started: nil) == usage)
        #expect(context.shown(history: sent + [ChatMessage(role: .user, text: "More")], target: target, started: nil) == nil)
        // The last answer asked again.
        let again = Array(history.dropLast()) + [ChatMessage(role: .assistant, text: "")]
        #expect(context.shown(history: again, target: target, started: nil) == usage)
        #expect(context.shown(history: [ChatMessage(role: .user, text: "Other")], target: target, started: nil) == nil)
        #expect(context.shown(history: history, target: target, started: .now) == nil)
        let lighter = ChatTarget(url: target.url, model: target.model, acceptsImages: true, effort: "low",
                                 context: 32768, apiKey: "secret")
        #expect(context.shown(history: history, target: lighter, started: nil) == usage)
        let model = ChatTarget(url: target.url, model: "a/other", acceptsImages: true, effort: "high",
                               context: 32768, apiKey: "secret")
        #expect(context.shown(history: history, target: model, started: nil) == nil)
        var other = target
        other.context = 1024
        #expect(context.shown(history: history, target: other, started: nil) == nil)

        // A new chat's count shows only on the chat its message started, not on another one of
        // one exchange.
        context.refresh(request("Summarize this"))
        try await waitFor(context.counted?.request.history.isEmpty == true)
        let fresh = context.counted?.usage
        #expect(context.shown(history: history, target: target, started: nil) == nil)
        let started = [ChatMessage(role: .user, text: "Summarize this"), ChatMessage(role: .assistant, text: "Wo")]
        #expect(context.shown(history: started, target: target, started: nil) == fresh)
        #expect(context.shown(history: [], target: target, started: nil) == fresh)
    }

    @Test func validatesCountResponsesAndSurfacesServerErrors() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CountProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        for (status, payload, expected) in [
            (200, #"{"input_tokens":123}"#, Optional(123)),
            (200, #"{"input_tokens":-1}"#, nil),
            (200, "not json", nil),
            (503, #"{"error":{"message":"Please retry"}}"#, nil),
        ] {
            let url = URL(string: "http://\(UUID().uuidString).invalid")!
            let target = ChatTarget(url: url, model: "a/b", acceptsImages: false, effort: "")
            let endpoint = url.appending(path: "v1/chat/completions/count_tokens")
            CountProtocol.responses.withLock { $0[endpoint.absoluteString] = (status, Data(payload.utf8)) }
            defer { _ = CountProtocol.responses.withLock { $0.removeValue(forKey: endpoint.absoluteString) } }
            do {
                let result = try await ChatClient.countTokens([ChatMessage(role: .user, text: "Hi")], with: target, session: session)
                #expect(result == expected)
            } catch {
                #expect(expected == nil)
                #expect(error.localizedDescription == (status == 503 ? "Please retry" : "The model returned an invalid token count."))
            }
        }
    }

    private func percent(_ share: Double) -> String {
        share.formatted(.percent.precision(.fractionLength(0)))
    }
}

private actor CounterProbe {
    private(set) var calls = 0
    /// The drafts whose count fails.
    let failing: Set<String>
    init(failing: Set<String> = []) { self.failing = failing }
    func count(_ request: ChatContext.Request) throws -> Int {
        calls += 1
        if failing.contains(request.text) { throw ChatClient.Failure(errorDescription: "Offline") }
        return request.text.utf8.count
    }
}

private actor SuspendedCounter {
    private var pending: [String: CheckedContinuation<Int, Never>] = [:]
    func count(_ text: String) async -> Int {
        await withCheckedContinuation { pending[text] = $0 }
    }
    func started(_ text: String) -> Bool { pending[text] != nil }
    func finish(_ text: String, with count: Int) { pending.removeValue(forKey: text)?.resume(returning: count) }
}

private final class CountProtocol: URLProtocol, @unchecked Sendable {
    static let responses = Mutex<[String: (Int, Data)]>([:])
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url, let (status, data) = Self.responses.withLock({ $0[url.absoluteString] }),
              let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
