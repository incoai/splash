import Foundation

public struct ChatMessage: Codable, Identifiable, Sendable, Equatable {
    public var id: UUID
    public let role: String
    public var content: String
    public var reasoning: String

    public init(role: String, content: String = "", reasoning: String = "") {
        id = UUID()
        self.role = role
        self.content = content
        self.reasoning = reasoning
    }
}

public struct ChatDelta: Sendable, Equatable {
    public let content: String
    public let reasoning: String
    public let tokensPerSecond: Double?
}

public enum SplashError: LocalizedError {
    case invalidAddress, invalidResponse, incompleteResponse, noModels
    case server(Int, String)

    public var errorDescription: String? {
        switch self {
        case .invalidAddress:
            "Enter a server address such as http://127.0.0.1:8000."
        case .invalidResponse:
            "Splash sent an unreadable reply."
        case .incompleteResponse:
            "The connection closed before Splash finished. Try again."
        case .noModels:
            "Splash has no model loaded. Start a model and reconnect."
        case let .server(status, message):
            status == 401 ? "Splash requires an API key. Enter it in Connection settings." : message
        }
    }
}

/// Decodes complete SSE events, including comments, CRLF and multiline data.
/// Requiring [DONE] lets the caller distinguish a complete reply from a broken socket.
public struct ReplyDecoder {
    private var line: [UInt8] = []
    private var data: [String] = []

    public enum Event: Equatable {
        case delta(ChatDelta)
        case done
    }

    public init() {}

    public mutating func append(_ byte: UInt8) throws -> Event? {
        guard byte == 10 else {
            line.append(byte)
            return nil
        }
        return try consumeLine()
    }

    public mutating func finish() throws -> Event? {
        if !line.isEmpty, let event = try consumeLine() { return event }
        return try dispatch()
    }

    private mutating func consumeLine() throws -> Event? {
        if line.last == 13 { line.removeLast() }
        guard let text = String(bytes: line, encoding: .utf8) else {
            throw SplashError.invalidResponse
        }
        line.removeAll(keepingCapacity: true)
        if text.isEmpty { return try dispatch() }
        if text == "data" { data.append("") }
        if text.hasPrefix("data:") {
            var value = String(text.dropFirst(5))
            if value.first == " " { value.removeFirst() }
            data.append(value)
        }
        return nil
    }

    private mutating func dispatch() throws -> Event? {
        guard !data.isEmpty else { return nil }
        let text = data.joined(separator: "\n")
        data.removeAll(keepingCapacity: true)
        if text == "[DONE]" { return .done }
        guard let chunk = try? JSONDecoder().decode(Chunk.self, from: Data(text.utf8)) else {
            throw SplashError.invalidResponse
        }
        if let error = chunk.error { throw SplashError.server(200, error.message) }
        let delta = chunk.choices?.first?.delta
        return .delta(ChatDelta(
            content: delta?.content ?? "",
            reasoning: delta?.reasoning_content ?? "",
            tokensPerSecond: chunk.metrics?.request_latency?.stream_tokens_per_second
        ))
    }

    private struct Chunk: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable {
                let content: String?
                let reasoning_content: String?
            }
            let delta: Delta?
        }
        struct Metrics: Decodable {
            struct Latency: Decodable { let stream_tokens_per_second: Double? }
            let request_latency: Latency?
        }
        let choices: [Choice]?
        let metrics: Metrics?
        let error: ServerErrorBody.Detail?
    }
}

private struct ServerErrorBody: Decodable {
    struct Detail: Decodable { let message: String }
    let error: Detail
}

public struct SplashClient: Sendable {
    public let baseURL: URL
    private let apiKey: String
    private let session: URLSession

    public init(address: String, apiKey: String = "", session: URLSession = .shared) throws {
        guard var parts = URLComponents(string: address.trimmingCharacters(in: .whitespacesAndNewlines)),
              ["http", "https"].contains(parts.scheme?.lowercased() ?? ""),
              let host = parts.host, !host.isEmpty,
              parts.user == nil, parts.password == nil,
              parts.query == nil, parts.fragment == nil,
              parts.path.isEmpty || parts.path == "/"
        else { throw SplashError.invalidAddress }
        parts.path = ""
        guard let url = parts.url else { throw SplashError.invalidAddress }
        baseURL = url
        self.apiKey = apiKey
        self.session = session
    }

    public func models() async throws -> [String] {
        let (data, response) = try await session.data(for: request("v1/models", timeout: 5))
        try validate(response, body: data)
        struct ModelList: Decodable {
            struct Model: Decodable { let id: String }
            let data: [Model]
        }
        let models = try JSONDecoder().decode(ModelList.self, from: data).data.map(\.id)
        guard !models.isEmpty else { throw SplashError.noModels }
        return models
    }

    public func complete(
        messages: [ChatMessage], model: String, thinking: Bool,
        onDelta: @Sendable (ChatDelta) async -> Void
    ) async throws {
        var request = request("v1/chat/completions", timeout: 600)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        // Preserve reasoning separately for the model's chat template and prefix cache.
        // Local UUIDs are never sent to the server.
        var body: [String: Any] = [
            "model": model,
            "messages": messages.filter { $0.role != "assistant" || !$0.content.isEmpty }
                .map { message -> [String: String] in
                    var fields = ["role": message.role, "content": message.content]
                    if message.role == "assistant", !message.reasoning.isEmpty {
                        fields["reasoning_content"] = message.reasoning
                    }
                    return fields
                },
            "stream": true,
            "stream_options": ["include_usage": true],
        ]
        if !thinking { body["reasoning_effort"] = "none" }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw SplashError.invalidResponse }
        if !(200...299).contains(http.statusCode) {
            var body = Data()
            for try await byte in bytes {
                body.append(byte)
                if body.count >= 64 * 1024 { break }
            }
            try validate(response, body: body)
        }
        var decoder = ReplyDecoder()
        for try await byte in bytes {
            try Task.checkCancellation()
            switch try decoder.append(byte) {
            case .done: return
            case let .delta(delta): await onDelta(delta)
            case nil: break
            }
        }
        switch try decoder.finish() {
        case .done: return
        case let .delta(delta): await onDelta(delta)
        case nil: break
        }
        throw SplashError.incompleteResponse
    }

    private func request(_ path: String, timeout: TimeInterval) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path), timeoutInterval: timeout)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        return request
    }

    private func validate(_ response: URLResponse, body: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw SplashError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            let detail = try? JSONDecoder().decode(ServerErrorBody.self, from: body)
            throw SplashError.server(http.statusCode, detail?.error.message ?? "Splash returned HTTP \(http.statusCode).")
        }
    }
}
