import Foundation
import Testing
@testable import SplashChat

@Suite("Splash streaming protocol")
struct ReplyDecoderTests {
    private func decode(_ text: String) throws -> [ReplyDecoder.Event] {
        var decoder = ReplyDecoder()
        var events: [ReplyDecoder.Event] = []
        for byte in text.utf8 {
            if let event = try decoder.append(byte) { events.append(event) }
        }
        if let event = try decoder.finish() { events.append(event) }
        return events
    }

    @Test func separatesThinkingAndUnicodeContent() throws {
        let events = try decode("""
        : keepalive

        data: {"choices":[{"delta":{"reasoning_content":"Let’s think…"}}]}

        data: {"choices":[{"delta":{"content":"Hello 🌊"}}]}

        data: [DONE]


        """)
        #expect(events == [
            .delta(ChatDelta(content: "", reasoning: "Let’s think…", tokensPerSecond: nil)),
            .delta(ChatDelta(content: "Hello 🌊", reasoning: "", tokensPerSecond: nil)),
            .done,
        ])
    }

    @Test func handlesCRLFMultilineEventsAndFinalUnterminatedLine() throws {
        let events = try decode("data: {\r\ndata: \"choices\": []}\r\n\r\ndata: [DONE]")
        #expect(events == [.delta(ChatDelta(content: "", reasoning: "", tokensPerSecond: nil)), .done])
    }

    @Test func acceptsUsageOnlyChunks() throws {
        let events = try decode("data: {\"choices\":[],\"metrics\":{\"request_latency\":{\"stream_tokens_per_second\":92.4}}}\n\n")
        #expect(events == [.delta(ChatDelta(content: "", reasoning: "", tokensPerSecond: 92.4))])
    }

    @Test func reportsInStreamErrors() {
        #expect(throws: (any Error).self) {
            try decode("data: {\"error\":{\"message\":\"Context limit exceeded\"}}\n\n")
        }
    }

    @Test func rejectsMalformedPayload() {
        #expect(throws: (any Error).self) { try decode("data: not json\n\n") }
    }
}

private final class StubProtocol: URLProtocol, @unchecked Sendable {
    final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var status = 200
        private var body = ""
        private var holdOpen = false
        private var requests: [URLRequest] = []

        func configure(status: Int = 200, body: String, holdOpen: Bool = false) {
            lock.lock()
            defer { lock.unlock() }
            self.status = status
            self.body = body
            self.holdOpen = holdOpen
            requests = []
        }

        func record(_ request: URLRequest) -> (Int, String, Bool) {
            lock.lock()
            defer { lock.unlock() }
            requests.append(request)
            return (status, body, holdOpen)
        }

        var lastRequest: URLRequest? {
            lock.lock()
            defer { lock.unlock() }
            return requests.last
        }
    }

    static let state = State()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        var captured = request
        if captured.httpBody == nil, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var body = Data()
            var buffer = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                body.append(contentsOf: buffer.prefix(count))
            }
            captured.httpBody = body
        }
        let (status, text, holdOpen) = Self.state.record(captured)
        let response = HTTPURLResponse(url: request.url!, statusCode: status,
                                       httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Split inside UTF-8 characters and SSE frame boundaries, as a real socket may.
        let bytes = Array(text.utf8)
        for offset in stride(from: 0, to: bytes.count, by: 7) {
            client?.urlProtocol(self, didLoad: Data(bytes[offset..<min(offset + 7, bytes.count)]))
        }
        if !holdOpen { client?.urlProtocolDidFinishLoading(self) }
    }
    override func stopLoading() {}
}

@Suite("Splash HTTP client", .serialized)
struct SplashClientTests {
    private func client(key: String = "") throws -> SplashClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return try SplashClient(address: "http://127.0.0.1:8000/", apiKey: key,
                                session: URLSession(configuration: configuration))
    }

    @Test func discoversModelsAndSendsCredentialsOnlyAsHeader() async throws {
        StubProtocol.state.configure(body: "{\"data\":[{\"id\":\"local-model\"}]}")
        let models = try await client(key: "test-key").models()
        #expect(models == ["local-model"])
        let request = try #require(StubProtocol.state.lastRequest)
        #expect(request.url?.path == "/v1/models")
        #expect(request.url?.query == nil)
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-key")
    }

    @Test func streamsReplyAndEncodesConversation() async throws {
        StubProtocol.state.configure(body: "data: {\"choices\":[{\"delta\":{\"content\":\"Hi 🌊\"}}]}\n\ndata: [DONE]\n\n")
        actor Collector {
            var text = ""
            func append(_ delta: ChatDelta) { text += delta.content }
        }
        let collector = Collector()
        try await client().complete(messages: [ChatMessage(role: "user", content: "Hello")],
                                    model: "local-model", thinking: false) { await collector.append($0) }
        #expect(await collector.text == "Hi 🌊")
        let request = try #require(StubProtocol.state.lastRequest)
        #expect(request.httpMethod == "POST")
        #expect(request.url?.path == "/v1/chat/completions")
        let body = try #require(request.httpBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(json["model"] as? String == "local-model")
        #expect(json["reasoning_effort"] as? String == "none")
        let messages = try #require(json["messages"] as? [[String: String]])
        #expect(messages == [["role": "user", "content": "Hello"]])
    }

    @Test func refusesTruncatedStream() async throws {
        StubProtocol.state.configure(body: "data: {\"choices\":[{\"delta\":{\"content\":\"Partial\"}}]}\n\n")
        do {
            try await client().complete(messages: [], model: "local-model", thinking: true) { _ in }
            Issue.record("A stream without [DONE] must fail")
        } catch SplashError.incompleteResponse {} catch { throw error }
    }

    @Test func cancellationStopsAnOpenResponse() async throws {
        StubProtocol.state.configure(body: "data: {\"choices\":[{\"delta\":{\"content\":\"Partial\"}}]}\n\n", holdOpen: true)
        let client = try client()
        let task = Task {
            try await client.complete(messages: [], model: "local-model", thinking: true) { _ in }
        }
        for _ in 0..<100 {
            if StubProtocol.state.lastRequest != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(StubProtocol.state.lastRequest != nil)
        task.cancel()
        switch await task.result {
        case .success: Issue.record("A cancelled stream must stop with cancellation")
        case let .failure(error):
            #expect(error is CancellationError || (error as? URLError)?.code == .cancelled)
        }
    }

    @Test func preservesPriorThinkingAndOmitsReasoningOnlyStoppedTurn() async throws {
        StubProtocol.state.configure(body: "data: [DONE]\n\n")
        try await client().complete(messages: [
            ChatMessage(role: "user", content: "Question"),
            ChatMessage(role: "assistant", content: "Answer", reasoning: "Private thought"),
            ChatMessage(role: "assistant", reasoning: "Stopped before an answer"),
        ], model: "local-model", thinking: true) { _ in }
        let body = try #require(StubProtocol.state.lastRequest?.httpBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(json["reasoning_effort"] == nil)
        let messages = try #require(json["messages"] as? [[String: String]])
        #expect(messages.count == 2)
        #expect(messages.last?["content"] == "Answer")
        #expect(messages.last?["reasoning_content"] == "Private thought")
    }

    @Test func reportsAuthenticationAndServerErrors() async throws {
        StubProtocol.state.configure(status: 401, body: "{\"error\":{\"message\":\"invalid or missing API key\"}}")
        do {
            _ = try await client().models()
            Issue.record("An unauthenticated request must fail")
        } catch SplashError.server(let status, _) { #expect(status == 401) }

        StubProtocol.state.configure(status: 400, body: "{\"error\":{\"message\":\"Context limit exceeded\"}}")
        do {
            try await client().complete(messages: [], model: "local-model", thinking: true) { _ in }
            Issue.record("A rejected chat request must fail")
        } catch SplashError.server(let status, let message) {
            #expect(status == 400)
            #expect(message == "Context limit exceeded")
        }
    }

    @Test func rejectsEmptyModelList() async throws {
        StubProtocol.state.configure(body: "{\"data\":[]}")
        do {
            _ = try await client().models()
            Issue.record("An empty model list must fail")
        } catch SplashError.noModels {} catch { throw error }
    }

    @Test func rejectsCredentialsAndPathsInServerAddress() {
        for address in ["file:///etc/passwd", "http://user:secret@localhost:8000",
                        "http://localhost:8000/v1", "http://localhost:8000?key=secret", "localhost:8000"] {
            #expect(throws: (any Error).self) { try SplashClient(address: address) }
        }
    }
}
