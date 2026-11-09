import Foundation
import Testing

@testable import Splash

/// The chats as the app keeps them: what a message the server refuses takes back of its chat and
/// gives back to the composer, and what an earlier app reads of their files.
@MainActor
@Suite(.serialized)
struct ChatStoreTests {
    /// A chat this app writes, the 1.6.0 app reads, which requires every message's images: an
    /// empty list, once they are files of their own. One whose messages lack them, as an earlier
    /// 1.6.1 wrote it, reads here.
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
            + #""text": "Read this", "reasoning": "", "attachments": [{"id": "\#(UUID())", "kind": "pdf", "#
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
}
