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
        let files = FileManager.default.temporaryDirectory.appending(path: "splash-draft-\(UUID().uuidString)")
        let draft = Draft(files: files, kept: files.appending(path: "Drafts.plist")) { nil }
        let refused = ChatStore.Refused(text: "Summarize this", attachments: [], reason: "Too large", conversation: nil)
        draft.text = "and then this"
        draft.restore(refused)
        #expect(draft.text == "Summarize this\n\nand then this")
        #expect(draft.notice == "Too large")
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

/// A new chat is no chat until the server takes its first message, and a message is its chat's from
/// then on: every state the chats go through as a message is sent, held by the server, and then
/// refused, lost, stopped, taken or failed.
extension ChatStoreTests {
    /// What the chats show: those listed, in the list's order, the open one, and its messages.
    struct Shown: Equatable {
        var listed: [UUID] = []
        var open: UUID?
        var messages: [String] = []
    }

    /// The chats on disk in `folder`, once `store` has written what it changed so far.
    private func written(_ store: ChatStore, in folder: URL) -> [Conversation] {
        store.flush()
        return ChatStore(folder: folder).conversations
    }

    /// A chat with one exchange, last changed `ago` seconds ago, written in `folder`.
    private func saved(_ title: String, ago: Double, in folder: URL) throws -> Conversation {
        let chat = Conversation(title: title, updated: .now - ago, messages: [
            ChatMessage(role: .user, text: title), ChatMessage(role: .assistant, text: "Yes."),
        ])
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try JSONEncoder().encode(chat).write(to: folder.appending(path: "\(chat.id.uuidString).json"))
        return chat
    }

    /// The first message of a new chat the server refuses leaves no chat: none is listed, open or
    /// written while the server holds the message, nor once it has refused it, and the refusal
    /// gives the composer, which still holds the message, only why. Sent again and taken, the
    /// chat is listed once, open and written, from when the server takes it.
    @Test func aNewChatIsNoChatUntilTheServerTakesItsFirstMessage() async throws {
        let fake = try RefusingServer(["refuse", "take"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        var refusals: [ChatStore.Refused] = []
        store.send("Plan a trip to Kyoto", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message")
        #expect(store.streaming != nil && written(store, in: fake.chats).isEmpty)
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the refusal")
        #expect(refusals == [ChatStore.Refused(text: "", attachments: [], reason: "Request body too large", conversation: nil)])
        #expect(shown.values == [Shown()], "no chat, through the refusal")
        #expect(written(store, in: fake.chats).isEmpty)

        store.send("Plan a trip to Kyoto", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message again")
        #expect(shown.values == [Shown()] && written(store, in: fake.chats).isEmpty)
        try await fake.proceed()
        try await waitFor(store.streaming == nil && store.current?.title == "A Title", "the reply and its title")
        let chat = try #require(store.current)
        #expect(refusals.count == 1)
        #expect(shown.each(\.listed) == [[], [chat.id]], "listed once, from when the server takes it")
        #expect(shown.each(\.open) == [nil, chat.id])
        #expect(shown.each(\.messages.first) == [nil, "Plan a trip to Kyoto"] && chat.messages.map(\.text) == ["Plan a trip to Kyoto", "Answer"])
        #expect(written(store, in: fake.chats).map(\.id) == [chat.id])
    }

    /// The server takes a request by starting its response: a request that waits for the engine
    /// gets the response's start and a keepalive, and its new chat is kept from then on, though
    /// the reply has not begun.
    @Test func aNewChatIsKeptOnceTheServerStartsItsResponse() async throws {
        let fake = try RefusingServer(["wait"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        store.send("Hello", attachments: [], to: fake.target)
        try await waitFor(fake.holding, "the server to hold the message")
        #expect(shown.values == [Shown()] && written(store, in: fake.chats).isEmpty)
        try await fake.proceed()
        try await waitFor(store.conversations.count == 1, "the chat, once the response has started")
        let id = try #require(store.conversations.first?.id)
        #expect(store.selection == id && store.streaming == id)
        #expect(written(store, in: fake.chats).map { $0.messages.map(\.text) } == [["Hello", ""]])
        try await waitFor(fake.holding, "the server to hold the reply")
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the reply")
        #expect(store.current?.messages.map(\.text) == ["Hello", "Answer"])
        #expect(shown.each(\.listed) == [[], [id]] && shown.each(\.open) == [nil, id])
    }

    /// A first message that never reaches the server, or whose connection goes without a byte of
    /// a response, leaves no chat either, and says why.
    @Test func aNewChatTheServerNeverAnswersWasNeverKept() async throws {
        let fake = try RefusingServer(["drop"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        var refusals: [ChatStore.Refused] = []
        // A port nothing listens on, as a server that has just gone leaves it.
        let nowhere = ChatTarget(url: URL(string: "http://127.0.0.1:\(freePort())")!, model: "a/b", acceptsImages: false, effort: "")
        store.send("Hello", attachments: [], to: nowhere) { refusals.append($0) }
        try await waitFor(store.streaming == nil, "the connection to fail")
        store.send("Hello", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message")
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the connection to go")
        #expect(refusals.count == 2 && refusals.allSatisfy { $0.text.isEmpty && $0.conversation == nil && $0.reason?.isEmpty == false })
        #expect(shown.values == [Shown()] && written(store, in: fake.chats).isEmpty)
    }

    /// A request that fails once the server has taken it leaves its chat, the new one too, with
    /// the answer saying why, as it does a later message: nothing goes back to the composer.
    @Test func aFailureOnceTheServerTookItStaysInTheChat() async throws {
        let fake = try RefusingServer(["fail"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        var refusals = 0
        store.send("Hello", attachments: [], to: fake.target) { _ in refusals += 1 }
        try await waitFor(fake.holding, "the server to hold the message")
        #expect(shown.values == [Shown()] && written(store, in: fake.chats).isEmpty)
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the failure")
        let chat = try #require(store.current)
        #expect(chat.messages.map(\.text) == ["Hello", ""] && chat.messages.last?.error == "The engine stopped")
        #expect(refusals == 0 && store.conversations.map(\.id) == [chat.id])
        #expect(shown.each(\.listed) == [[], [chat.id]], "listed once, from when the server took it")
        #expect(written(store, in: fake.chats).first?.messages.last?.error == "The engine stopped")
    }

    /// Stopped while the server holds it, as Stop, a switch of models and a quit stop it, a first
    /// message leaves no chat, in the list or on disk, and gives the composer, which still holds
    /// it, nothing: nor why, as there is no why.
    @Test func aFirstMessageStoppedBeforeTheServerTakesItLeavesNoChat() async throws {
        let fake = try RefusingServer(["take"])
        defer { fake.close() }
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        var refusals = 0
        store.send("Hello", attachments: [], to: fake.target) { _ in refusals += 1 }
        try await waitFor(fake.holding, "the server to hold the message")
        store.stop()
        try await waitFor(store.streaming == nil, "the stop")
        #expect(refusals == 0 && shown.values == [Shown()])
        // As the app quits: what it has written is all there is.
        #expect(written(store, in: fake.chats).isEmpty)
    }

    /// Another chat opened while the server holds a new chat's first message stays open as the
    /// server takes it: the new chat joins the list at its top. The new chat open again by then,
    /// as New Chat leaves it, the chat taken opens.
    @Test func anotherChatOpenedMeanwhileStaysOpen() async throws {
        let fake = try RefusingServer(["take", "take"])
        defer { fake.close() }
        let other = try saved("Bread recipe", ago: 3600, in: fake.chats)
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        let there = Shown(listed: [other.id], open: other.id, messages: ["Bread recipe", "Yes."])
        store.send("Hello", attachments: [], to: fake.target)
        try await waitFor(fake.holding, "the server to hold the message")
        store.selection = other.id
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the reply")
        #expect(store.selection == other.id)
        let new = try #require(store.conversations.first { $0.id != other.id })
        #expect(store.conversations.map(\.id) == [new.id, other.id] && new.messages.map(\.text) == ["Hello", "Answer"])
        // Only ever the other chat's two states, a new chat open or it, until the new one is listed.
        #expect(shown.values.filter { $0.listed == [other.id] }.allSatisfy { $0 == there || $0 == Shown(listed: [other.id]) })
        #expect(shown.values.filter { $0.listed != [other.id] }.allSatisfy { $0.listed == [new.id, other.id] && $0.open == other.id })

        // New Chat again while the server holds the message: one new chat, which the chat taken opens.
        store.newChat()
        store.send("Again", attachments: [], to: fake.target)
        try await waitFor(fake.holding, "the server to hold the message")
        store.selection = other.id
        store.newChat()
        #expect(store.selection == nil && store.conversations.count == 2)
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the reply")
        #expect(store.conversations.count == 3 && store.selection == store.conversations[0].id)
        #expect(store.current?.messages.map(\.text) == ["Again", "Answer"])
    }

    /// A new chat's first message that comes to nothing opens the new chat again, where the
    /// composer, which still holds it, says why: refused once another chat is open, its server
    /// not to be reached, the other chat opened before the connection fails, and stopped from
    /// elsewhere, as a switch of models stops it, which gives the composer nothing but the
    /// message it holds, to edit. No chat is kept of it, in the list or on disk. Left under the
    /// other chat, the message was that chat's to send, its reason unsaid there.
    @Test func aFirstMessageThatComesBackOpensTheNewChat() async throws {
        let fake = try RefusingServer(["refuse", "take"])
        defer { fake.close() }
        let other = try saved("Bread recipe", ago: 3600, in: fake.chats)
        let store = ChatStore(folder: fake.chats)
        let shown = Seen { Shown(of: store) }
        var refusals: [ChatStore.Refused] = []
        let there = Shown(listed: [other.id], open: other.id, messages: ["Bread recipe", "Yes."])
        store.send("Hello", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message")
        store.selection = other.id
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the refusal")
        #expect(store.selection == nil && refusals.map(\.reason) == ["Request body too large"])

        // A port nothing listens on, as a server that has just gone leaves it.
        let nowhere = ChatTarget(url: URL(string: "http://127.0.0.1:\(freePort())")!, model: "a/b", acceptsImages: false, effort: "")
        store.send("Hello", attachments: [], to: nowhere) { refusals.append($0) }
        store.selection = other.id
        try await waitFor(store.streaming == nil, "the connection to fail")
        #expect(store.selection == nil && refusals.count == 2)
        #expect(refusals.allSatisfy { $0.text.isEmpty && $0.attachments.isEmpty && $0.conversation == nil && $0.reason?.isEmpty == false })

        store.send("Hello", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message again")
        store.selection = other.id
        store.stop()
        try await waitFor(store.streaming == nil, "the stop")
        #expect(store.selection == nil && refusals.count == 2)
        // Only ever the other chat as it was, open or under the new chat.
        #expect(shown.values.allSatisfy { $0 == there || $0 == Shown(listed: [other.id]) })
        #expect(shown.values.last == Shown(listed: [other.id]) && written(store, in: fake.chats) == [other])
    }

    /// A later message that comes to nothing comes back to its chat, which opens again where
    /// another was opened meanwhile, for the composer to hold it there, and say why: refused, its
    /// server not to be reached, the other chat opened before the connection fails, and stopped
    /// from elsewhere, as a switch of models stops it. Left under the chat opened meanwhile, the
    /// message was that chat's to send, its reason unsaid there. Stopped in its own chat, as Stop
    /// stops it, its chat stays open. Each time its chat is as it was, in the list and on disk.
    @Test func aLaterMessageThatComesBackOpensItsChat() async throws {
        let fake = try RefusingServer(["refuse", "take", "take"])
        defer { fake.close() }
        let older = try saved("Bread recipe", ago: 7200, in: fake.chats), newer = try saved("Kyoto in spring", ago: 3600, in: fake.chats)
        let store = ChatStore(folder: fake.chats)
        store.selection = older.id
        var refusals: [ChatStore.Refused] = []
        store.send("And rye?", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message")
        store.selection = newer.id
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the refusal")
        #expect(store.selection == older.id && store.current == older)

        // A port nothing listens on, as a server that has just gone leaves it.
        let nowhere = ChatTarget(url: URL(string: "http://127.0.0.1:\(freePort())")!, model: "a/b", acceptsImages: false, effort: "")
        store.send("And rye?", attachments: [], to: nowhere) { refusals.append($0) }
        store.selection = newer.id
        try await waitFor(store.streaming == nil, "the connection to fail")
        #expect(store.selection == older.id && store.current == older)

        // Stopped once another chat is open, then in its own chat.
        for elsewhere in [true, false] {
            store.send("And rye?", attachments: [], to: fake.target) { refusals.append($0) }
            try await waitFor(fake.holding, "the server to hold the message")
            if elsewhere { store.selection = newer.id }
            store.stop()
            try await waitFor(store.streaming == nil, "the stop")
            #expect(store.selection == older.id && store.current == older)
            // The request stopped, which the server held, goes on to nothing.
            try await fake.proceed()
        }
        #expect(refusals.map(\.text) == ["And rye?", "And rye?", "And rye?", "And rye?"])
        #expect(refusals.allSatisfy { $0.conversation == older.id && $0.attachments.isEmpty })
        #expect(refusals.first?.reason == "Request body too large" && refusals.map { $0.reason != nil } == [true, true, false, false])
        #expect(store.conversations == [newer, older] && written(store, in: fake.chats) == [newer, older])
    }

    /// An answer asked again gives back only why it came to nothing: refused once another chat is
    /// open, it opens its chat again, where the composer says why and the earlier answer shows.
    /// Stopped from elsewhere, as a switch of models stops it, the chat open by then stays open,
    /// and nothing goes to the composer, which keeps what it holds and says: the request's end
    /// alone tells of the stop.
    @Test func anAnswerAskedAgainOpensItsChatOnlyToSayWhy() async throws {
        let fake = try RefusingServer(["refuse", "take"])
        defer { fake.close() }
        let older = try saved("Bread recipe", ago: 7200, in: fake.chats), newer = try saved("Kyoto in spring", ago: 3600, in: fake.chats)
        let store = ChatStore(folder: fake.chats)
        store.selection = older.id
        var refusals: [ChatStore.Refused] = []
        store.retry(with: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the answer asked again")
        store.selection = newer.id
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the refusal")
        #expect(store.selection == older.id && store.current == older)
        #expect(refusals == [ChatStore.Refused(text: "", attachments: [], reason: "Request body too large", conversation: older.id)])

        store.retry(with: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the answer asked again")
        store.selection = newer.id
        store.stop()
        try await waitFor(store.streaming == nil, "the stop")
        #expect(store.selection == newer.id)
        #expect(refusals.count == 1 && !store.answering)
        #expect(store.conversations == [newer, older] && written(store, in: fake.chats) == [newer, older])
    }

    /// A later message shows in its chat at once, and changes nothing the app keeps until the
    /// server takes it: the list keeps its order and the chat's file its content while the server
    /// holds the message, and for good where the server refuses it, its connection goes without a
    /// byte or it is stopped. Each time the message goes back to the composer, with why but for
    /// the stop.
    @Test func aLaterMessageChangesNothingKeptUntilTheServerTakesIt() async throws {
        let fake = try RefusingServer(["refuse", "drop", "take", "take"])
        defer { fake.close() }
        let older = try saved("Bread recipe", ago: 7200, in: fake.chats), newer = try saved("Kyoto in spring", ago: 3600, in: fake.chats)
        let store = ChatStore(folder: fake.chats)
        store.selection = older.id
        let shown = Seen { Shown(of: store) }
        var refusals: [ChatStore.Refused] = []
        for end in ["refused", "dropped", "stopped"] {
            store.send("And rye?", attachments: [], to: fake.target) { refusals.append($0) }
            #expect(store.current?.messages.map(\.text) == ["Bread recipe", "Yes.", "And rye?", ""])
            try await waitFor(fake.holding, "the server to hold the message")
            #expect(store.conversations == [newer, older] && written(store, in: fake.chats) == [newer, older])
            if end == "stopped" { store.stop() } else { try await fake.proceed() }
            try await waitFor(store.streaming == nil, "the message \(end)")
            #expect(store.current == older && written(store, in: fake.chats) == [newer, older])
        }
        #expect(refusals.map(\.text) == ["And rye?", "And rye?", "And rye?"] && refusals.allSatisfy { $0.conversation == older.id && $0.attachments.isEmpty })
        #expect(refusals.first?.reason == "Request body too large" && refusals.dropFirst().first?.reason?.isEmpty == false
            && refusals.last?.reason == nil)
        #expect(shown.values.allSatisfy { $0.listed == [newer.id, older.id] }, "the list's order, through them")
        // The request stopped, which the server held, goes on to nothing.
        try await fake.proceed()

        store.send("And rye?", attachments: [], to: fake.target) { refusals.append($0) }
        try await waitFor(fake.holding, "the server to hold the message again")
        #expect(store.conversations == [newer, older])
        try await fake.proceed()
        try await waitFor(store.streaming == nil, "the reply")
        #expect(refusals.count == 3 && store.conversations.map(\.id) == [older.id, newer.id])
        #expect(written(store, in: fake.chats).first?.messages.map(\.text) == ["Bread recipe", "Yes.", "And rye?", "Answer"])
    }
}

extension ChatStoreTests.Shown {
    /// What `store` shows now.
    @MainActor init(of store: ChatStore) {
        self.init(listed: store.conversations.map(\.id), open: store.selection, messages: store.current?.messages.map(\.text) ?? [])
    }
}

/// The fake server that holds requests and answers them as a test lets it
/// (Fakes/refusing_server.py), with a folder of its own, which holds the chats too.
@MainActor
private final class RefusingServer {
    let folder = FileManager.default.temporaryDirectory.appending(path: "splash-refusals-\(UUID().uuidString)")
    let target: ChatTarget
    private let process = Process()

    var chats: URL { folder.appending(path: "chats") }

    /// Whether it holds a request, and whether it has sent a refusal.
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

    /// Lets it go on with the request it holds.
    func release() {
        FileManager.default.createFile(atPath: folder.appending(path: "release").path, contents: nil)
    }

    /// Lets it go on with the request it holds, and waits until it has: what it holds from then
    /// on is another request, or the same one again.
    func proceed() async throws {
        release()
        try await waitFor(!FileManager.default.fileExists(atPath: folder.appending(path: "release").path), "the server to go on")
    }

    func close() {
        process.terminate()
        try? FileManager.default.removeItem(at: folder)
    }
}
