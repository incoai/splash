import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// Drives the real state/process/HTTP path against a small server, without model weights: the
/// composer counts the next prompt as it is written, and Send holds back only a prompt the model's
/// 1,024-token context cannot hold.
@MainActor
@Suite(.serialized)
struct ChatContextFlowTests {
    @Test func aPromptTheContextCannotHoldStaysAndANewChatSendsIt() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = """
                Summarize our decisions and next steps.
                Use the attached image to explain the
                interface changes, and recommend what
                we should implement next.
                """
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            let text = state.draft.text, images = state.draft.images
            let usage = try await count(state)
            #expect(usage.level == .full && state.contextUsage == usage)
            #expect(!state.canSendDraft && state.overflow == usage)
            state.sendDraft()
            #expect(state.draft.text == text && state.draft.images == images)
            #expect(state.chats.current?.messages == fixture.history.messages)
            #expect(try fixture.sent().isEmpty)
            await picture("chat-context-full", state: state)

            // A new chat keeps the draft and leaves the long one saved.
            state.chats.newChat()
            #expect(state.overflow == nil)
            #expect(try await count(state).level == .normal)
            #expect(state.canSendDraft)
            await picture("chat-context-room", state: state)
            let counts = try fixture.counts()
            state.sendDraft()
            #expect(state.draft.isEmpty)
            try await waitFor(state.chats.streaming == nil)
            #expect(state.chats.current?.messages.last?.text == "Answer")
            let sent = try fixture.sent()
            #expect(sent.count == 1 && sent[0].count == 1)
            #expect(sent[0][0]["role"] as? String == "user")
            // Sending doesn't count the prompt again.
            #expect(try fixture.counts() == counts)
            #expect(state.chats.conversations.first { $0.id == fixture.history.id } == fixture.history)
        }
    }

    @Test func aPromptInsideTheReplyAllowanceStillGoes() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "Short?"
            let usage = try await count(state)
            #expect(usage.level == .tight && usage.note != nil)
            #expect(state.canSendDraft)
            await picture("chat-context-tight", state: state)
            state.sendDraft()
            #expect(state.draft.isEmpty)
            try await waitFor(state.chats.streaming == nil)
            #expect(state.chats.current?.messages.last?.text == "Answer")
            #expect(try fixture.sent().count == 1)
        }
    }

    /// A count that fails, or hasn't finished, holds the message back no more than none would.
    @Test func aFailedOrSlowCountHoldsNothingBack() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "offline"
            state.chatContext.refresh(state.contextRequest)
            try await waitFor((try? fixture.counts()) == 1)
            try await Task.sleep(for: .milliseconds(200))
            #expect(state.contextRequest.flatMap(state.chatContext.usage(for:)) == nil)
            #expect(state.canSendDraft)
            state.sendDraft()
            #expect(state.draft.isEmpty && state.chats.streaming != nil)
            try await waitFor(state.chats.streaming == nil)
            #expect(state.chats.current?.messages.last?.text == "Answer")

            state.draft.text = "slow question"
            state.chatContext.refresh(state.contextRequest)
            try await waitFor(FileManager.default.fileExists(atPath: fixture.root.appending(path: "counting").path))
            #expect(state.canSendDraft)
            state.sendDraft()
            #expect(state.draft.isEmpty && state.chats.streaming != nil)
            try await waitFor(state.chats.streaming == nil)
            #expect(state.chats.current?.messages.last?.text == "Answer")
            #expect(try fixture.sent().count == 2)
        }
    }

    /// The draft's prompt, counted as the composer counts it.
    private func count(_ state: AppState) async throws -> ContextUsage {
        let request = try #require(state.contextRequest)
        state.chatContext.refresh(request)
        try await waitFor(state.chatContext.usage(for: request) != nil)
        return try #require(state.chatContext.usage(for: request))
    }

    private func withServer(_ test: (ContextFixture) async throws -> Void) async throws {
        let fixture = try ContextFixture()
        fixture.state.server.start("test-model", settings: ServeSettings(values: ["--port": String(fixture.port)]))
        do {
            try await waitFor(fixture.state.server.phase == .ready)
            try await test(fixture)
        } catch {
            await fixture.close()
            throw error
        }
        await fixture.close()
    }

    private func picture(_ name: String, state: AppState, width: CGFloat = 680) async {
        // As in MainView, give the composer a definite width before it measures the text field.
        await Snapshot.save(name, size: CGSize(width: width, height: 360), Offered {
            VStack {
                Spacer(minLength: 0)
                Composer(draft: state.draft).padding(16)
            }
        }.environment(state))
    }
}

@MainActor
private final class ContextFixture {
    let history = Conversation(title: "Saved chat", messages: [ChatMessage(role: .user, text: String(repeating: "h", count: 800)),
                                                              ChatMessage(role: .assistant, text: "Saved answer")])
    let root: URL
    let state: AppState
    let port: Int

    init() throws {
        let runtime = try FakeRuntime("splash-context-flow", launcher: Self.script)
        try JSONEncoder().encode(history).write(to: runtime.chats.appending(path: "\(history.id).json"))
        root = runtime.root
        port = runtime.port
        state = runtime.state(keychain: Keychain(service: "Splash Context Tests"))
        state.chats.selection = history.id
    }

    /// The messages of each generation the server was asked for.
    func sent() throws -> [[[String: Any]]] {
        try lines("sent.jsonl").map {
            try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [[String: Any]])
        }
    }

    /// The prompts the server was asked to count.
    func counts() throws -> Int {
        try lines("counted.jsonl").count
    }

    private func lines(_ name: String) throws -> [Substring] {
        let file = root.appending(path: name)
        guard FileManager.default.fileExists(atPath: file.path) else { return [] }
        return try String(contentsOf: file, encoding: .utf8).split(separator: "\n")
    }

    func close() async {
        state.chatContext.refresh(nil)
        state.server.stop()
        let deadline = ContinuousClock.now + .seconds(5)
        while state.server.running && ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(20))
        }
        try? FileManager.default.removeItem(at: root)
    }

    private static let script = #"""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    port = int(sys.argv[sys.argv.index('--port') + 1])
    def size(content):
        if isinstance(content, str): return len(content.encode())
        return sum(64 if part['type'] == 'image_url' else len(part['text'].encode()) for part in content)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, body, status=200, kind='application/json'):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path == '/v1/models':
                self.reply({'data': [{'context_length': 1024, 'input_modalities': ['text', 'image']}]})
            else: self.reply({})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/count_tokens'):
                with (root / 'counted.jsonl').open('a') as output:
                    output.write(json.dumps(body['messages']) + '\n')
                text = body['messages'][-1]['content']
                if text == 'offline':
                    return self.reply({'error': {'message': 'Try again shortly'}}, 503)
                if text == 'slow question':
                    (root / 'counting').touch()
                    time.sleep(5)
                count = 100 + sum(size(m['content']) + len(m.get('reasoning_content', '').encode()) for m in body['messages'])
                return self.reply({'input_tokens': count})
            with (root / 'sent.jsonl').open('a') as output:
                output.write(json.dumps(body['messages']) + '\n')
            self.reply('data: {"choices":[{"delta":{"content":"Answer"}}]}\n\ndata: [DONE]\n\n', kind='text/event-stream')
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    record = root / 'build/runtime' / f'serve-{port}.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({'pid': os.getpid(), 'phase': 'ready'}))
    server.serve_forever()
    """#
}
