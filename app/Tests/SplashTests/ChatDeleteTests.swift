import Foundation
import Testing

@testable import Splash

struct ChatDeleteTests {
    @MainActor @Test func deletesSeveralOrAllChats() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let first = Conversation(title: "First", messages: [ChatMessage(role: .user, text: "1")])
        let second = Conversation(title: "Second", messages: [ChatMessage(role: .user, text: "2")])
        let third = Conversation(title: "Third", messages: [ChatMessage(role: .user, text: "3")])
        for conversation in [first, second, third] {
            try JSONEncoder().encode(conversation)
                .write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        }
        let stray = folder.appending(path: "stray.json")
        try Data("not a conversation".utf8).write(to: stray)
        let store = ChatStore(folder: folder)
        store.selection = first.id
        store.delete([first.id, third.id])
        store.flush()
        #expect(Set(store.conversations.map(\.id)) == Set([second.id]))
        #expect(store.selection == nil)
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "\(first.id.uuidString).json").path))
        #expect(FileManager.default.fileExists(atPath: folder.appending(path: "\(second.id.uuidString).json").path))
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "\(third.id.uuidString).json").path))
        store.selection = second.id
        store.deleteAll()
        store.flush()
        #expect(store.conversations.isEmpty && store.selection == nil)
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "\(second.id.uuidString).json").path))
        #expect(!FileManager.default.fileExists(atPath: stray.path))
    }

    /// A chat deleted while its reply streams stays deleted: the reply that ends afterwards, and
    /// the writes queued before the delete, bring none of it back.
    @MainActor @Test func aChatDeletedWhileItsReplyStreamsStaysDeleted() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let store = ChatStore(folder: folder)
        let nowhere = ChatTarget(url: URL(string: "http://127.0.0.1:9")!, model: "a/b", acceptsImages: false, effort: "")
        for delete in [{ store.deleteAll() }, { store.delete([store.conversations[0].id]) }] {
            store.send("Hello", images: [], to: nowhere)
            #expect(store.streaming != nil)
            delete()
            try await waitFor(store.streaming == nil, "the reply to end")
            store.flush()
            #expect(store.conversations.isEmpty)
            let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            #expect(files.filter { $0.pathExtension == "json" }.isEmpty)
        }
    }
}
