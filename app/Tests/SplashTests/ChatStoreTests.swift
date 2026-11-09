import Foundation
import Testing

@testable import Splash

/// The chats as the app keeps them: what a message the server refuses takes back of its chat and
/// gives back to the composer, and what an earlier app reads of their files.
@MainActor
@Suite(.serialized)
struct ChatStoreTests {
    /// A chat this app writes, the 1.6.0 app reads, which requires every message's images: an
    /// empty list, once they are files of their own.
    @Test func the160AppReadsTheChatsThisOneWrites() throws {
        struct Message160: Decodable {
            let id: UUID
            let role: String
            let text: String
            let reasoning: String
            let images: [Data]
        }
        struct Conversation160: Decodable {
            let id: UUID
            let title: String
            let updated: Date
            let messages: [Message160]
        }
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let id = UUID()
        let written = #"{"id": "\#(id)", "title": "Plan", "updated": 0, "messages": [{"id": "\#(UUID())", "role": "user", "#
            + #""text": "Read this", "reasoning": "", "images": [], "attachments": [{"id": "\#(UUID())", "kind": "pdf", "#
            + #""type": "application/pdf", "bytes": 3, "name": "a.pdf", "pages": 1}]}]}"#
        try Data(written.utf8).write(to: folder.appending(path: "\(id).json"))
        let store = ChatStore(folder: folder)
        let message = try #require(store.conversations.first?.messages.first)
        #expect(message.images.isEmpty && message.attachments?.first?.name == "a.pdf")
        store.rename(id, to: "Renamed")
        store.flush()
        let read = try JSONDecoder().decode(Conversation160.self, from: Data(contentsOf: folder.appending(path: "\(id).json")))
        #expect(read.title == "Renamed" && read.messages.map(\.images) == [[]] && read.messages.map(\.text) == ["Read this"])
    }

    /// A message the server refuses goes back to the composer before what was typed there since,
    /// which stays; a reply asked again gives no text back.
    @Test func aRefusedMessageGoesBackBeforeWhatWasTypedSince() {
        let draft = Draft(files: FileManager.default.temporaryDirectory.appending(path: "splash-draft-\(UUID().uuidString)"))
        let refused = ChatStore.Refused(text: "Summarize this", attachments: [], reason: "Too large", conversation: nil)
        draft.text = "and then this"
        draft.restore(refused)
        #expect(draft.text == "Summarize this\n\nand then this")
        #expect(draft.notice == Draft.Notice(text: "Too large", conversation: nil))
        draft.text = " \n"
        draft.restore(refused)
        #expect(draft.text == "Summarize this")
        draft.restore(ChatStore.Refused(text: "", attachments: [], reason: "Too large", conversation: nil))
        #expect(draft.text == "Summarize this")
    }

    /// A message the server refuses comes out of its chat as the chat is now: a pin and a name
    /// given while the server read it stay, on disk too, and the chat goes back to its place.
    @Test func aRefusalTakesBackItsMessageAndKeepsWhatChangedMeanwhile() async throws {
        let fake = try RefusingServer(["answer", "refuse"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        store.send("Hello", attachments: [], to: fake.target)
        try await waitFor(store.streaming == nil && store.current?.title == "A Title", "the first exchange's title")
        let chat = try #require(store.current)
        var refusals: [ChatStore.Refused] = []
        store.send("refuse this", attachments: [], to: fake.target) { refusals.append($0) }
        #expect(store.current?.messages.count == 4)
        store.pin(chat.id, true)
        store.rename(chat.id, to: "Renamed")
        fake.release()
        try await waitFor(store.streaming == nil && !refusals.isEmpty, "the refusal")
        #expect(refusals == [ChatStore.Refused(text: "refuse this", attachments: [], reason: "Request body too large", conversation: chat.id)])
        let kept = try #require(store.current)
        #expect(kept.messages == chat.messages && kept.title == "Renamed" && kept.pinned != nil && kept.updated == chat.updated)
        store.flush()
        let read = try #require(ChatStore(folder: fake.chats).conversations.first)
        #expect(read.messages == chat.messages && read.title == "Renamed" && read.pinned != nil)
    }

    /// A refusal that comes once its chat is deleted leaves the chat deleted, on disk too, and
    /// gives nothing back to the composer: the message went with the chat.
    @Test func aRefusalAfterADeleteLeavesTheChatDeleted() async throws {
        let fake = try RefusingServer(["answer", "refuse"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        store.send("Hello", attachments: [], to: fake.target)
        try await waitFor(store.streaming == nil && store.current?.title == "A Title", "the first exchange's title")
        let chat = try #require(store.current)
        var refusals = 0
        store.send("refuse this", attachments: [], to: fake.target) { _ in refusals += 1 }
        try await waitFor(fake.holding, "the server to hold the message")
        fake.release()
        // The main thread, where the reply takes the refusal in, held until the chat is deleted:
        // the refusal has come by then, as it can just before a Delete.
        let deadline = ContinuousClock.now + .seconds(10)
        while !fake.refused && ContinuousClock.now < deadline { usleep(10_000) }
        try #require(fake.refused)
        usleep(300_000)
        store.delete([chat.id])
        try await waitFor(store.streaming == nil, "the reply to end")
        #expect(store.conversations.isEmpty && refusals == 0)
        store.flush()
        #expect(ChatStore(folder: fake.chats).conversations.isEmpty)
    }

    /// An answer asked again that the server refuses gives the earlier one back, in the chat as it
    /// is now: a pin given meanwhile stays.
    @Test func aRefusedRetryGivesTheEarlierAnswerBack() async throws {
        let fake = try RefusingServer(["answer", "refuse"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        store.send("Hello", attachments: [], to: fake.target)
        try await waitFor(store.streaming == nil && store.current?.title == "A Title", "the first exchange's title")
        let chat = try #require(store.current)
        var refusals: [ChatStore.Refused] = []
        store.retry(with: fake.target) { refusals.append($0) }
        #expect(store.current?.messages.count == 2 && store.current?.messages.last?.id != chat.messages.last?.id)
        store.pin(chat.id, true)
        fake.release()
        try await waitFor(store.streaming == nil && !refusals.isEmpty, "the refusal")
        #expect(refusals == [ChatStore.Refused(text: "", attachments: [], reason: "Request body too large", conversation: chat.id)])
        #expect(store.current?.messages == chat.messages && store.current?.pinned != nil)
        store.flush()
        let read = try #require(ChatStore(folder: fake.chats).conversations.first)
        #expect(read.messages == chat.messages && read.pinned != nil)
    }
}

/// The fake server that refuses requests as a test lets it (Fakes/refusing_server.py), with a
/// folder of its own, which holds the chats too.
@MainActor
private final class RefusingServer {
    let folder = FileManager.default.temporaryDirectory.appending(path: "splash-refusals-\(UUID().uuidString)")
    let target: ChatTarget
    private let process = Process()

    var chats: URL { folder.appending(path: "chats") }

    /// Whether it holds a request it is to refuse, and whether it has sent the refusal.
    var holding: Bool { FileManager.default.fileExists(atPath: folder.appending(path: "holding").path) }
    var refused: Bool { FileManager.default.fileExists(atPath: folder.appending(path: "refused").path) }

    init(_ actions: [String]) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        process.executableURL = URL(filePath: "/usr/bin/python3")
        process.arguments = [try #require(Bundle.module.path(forResource: "refusing_server", ofType: "py", inDirectory: "Fakes")),
                             folder.path] + actions
        let output = Pipe()
        process.standardOutput = output
        try process.run()
        let port = try #require(String(data: output.fileHandleForReading.availableData, encoding: .utf8)
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        target = ChatTarget(url: URL(string: "http://127.0.0.1:\(port)")!, model: "a/b", acceptsImages: false, effort: "")
    }

    /// Lets it refuse the request it holds.
    func release() {
        FileManager.default.createFile(atPath: folder.appending(path: "release").path, contents: nil)
    }

    func close() {
        process.terminate()
        try? FileManager.default.removeItem(at: folder)
    }
}
