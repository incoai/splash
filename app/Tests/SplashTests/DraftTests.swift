import AppKit
import Foundation
import Testing

@testable import Splash

/// Each chat's unsent draft is kept on disk and comes back as the app opens again; sending it,
/// clearing it or deleting its chat removes it.
@MainActor
struct DraftTests {
    /// Writes the conversations `ids` into the store folder `folder`.
    static func write(_ ids: [UUID], in folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for id in ids {
            let conversation = Conversation(id: id, messages: [ChatMessage(role: .user, text: "Hi")])
            try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(id.uuidString).json"))
        }
    }

    static func folder() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "splash-drafts-\(UUID().uuidString)")
    }

    static let image: NSImage = {
        let image = NSImage(size: NSSize(width: 8, height: 8))
        image.lockFocus()
        NSColor.systemTeal.setFill()
        NSRect(x: 0, y: 0, width: 8, height: 8).fill()
        image.unlockFocus()
        return image
    }()

    @Test func draftsComeBackAsTheAppOpensAgain() async throws {
        let (a, b) = (UUID(), UUID())
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Self.write([a, b], in: folder)
        let store = ChatStore(folder: folder)
        var open: UUID? = a
        let draft = Draft(files: store.attachments, kept: store.drafts) { open }
        // Typed a character at a time, 10 ms apart, and an image: nothing is written until a pause.
        for character in "Half a thought about the parser" {
            draft.text += String(character)
            try await Task.sleep(for: .milliseconds(10))
        }
        draft.attach([Self.image])
        try await waitFor(draft.attaching == 0 && draft.attachments.count == 1, "the image attached")
        let attached = draft.attachments
        #expect(!FileManager.default.fileExists(atPath: store.drafts.path), "written while typing")
        open = nil
        draft.text = "A new chat's question"
        open = b
        draft.text = "  "
        try await waitFor(FileManager.default.fileExists(atPath: store.drafts.path), "the drafts written", seconds: 3)

        // The app opens again: A's text and image and the new chat's text; B's blank one is none.
        // Its launch keeps the image's file, which no conversation names.
        let next = ChatStore(folder: folder)
        open = a
        let reopened = Draft(files: next.attachments, kept: next.drafts) { open }
        next.adoptFiles(keeping: reopened.attachmentFiles)
        next.flush()
        #expect(reopened.text == "Half a thought about the parser")
        #expect(reopened.attachments == attached && reopened.thumbnails.count == 1)
        let file = next.attachments.appending(path: try #require(attached.first).file)
        #expect(FileManager.default.fileExists(atPath: file.path), "the draft's image went at launch")
        open = nil
        #expect(reopened.text == "A new chat's question")
        #expect(reopened.chats == [a])

        // Sent, and deleted with its chat: gone once the drafts are written, and the next opening
        // has neither.
        reopened.take(from: nil)
        reopened.forget([a])
        reopened.save()
        let last = Draft(files: next.attachments, kept: next.drafts) { open }
        #expect(last.chats.isEmpty && last.text.isEmpty)
    }

    /// A new install's first draft is kept before any chat is, its save making the chats folder,
    /// and goes once cleared.
    @Test func aNewInstallsDraftIsKeptUntilCleared() async throws {
        let folder = Self.folder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let store = ChatStore(folder: folder)
        let draft = Draft(files: store.attachments, kept: store.drafts) { nil }
        draft.text = "Something"
        draft.save()
        #expect(Draft(files: store.attachments, kept: store.drafts) { nil }.text == "Something")
        draft.text = ""
        draft.save()
        #expect(Draft(files: store.attachments, kept: store.drafts) { nil }.text.isEmpty)
    }

    /// The app's launch keeps the drafts of the chats it has, and forgets one whose chat another
    /// copy of the app deleted.
    @Test func theLaunchForgetsADraftWhoseChatIsGone() async throws {
        let fake = try FakeRuntime("splash-drafts-launch", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let (a, gone) = (UUID(), UUID())
        try Self.write([a], in: fake.chats)
        let store = ChatStore(folder: fake.chats)
        var open: UUID? = a
        let draft = Draft(files: store.attachments, kept: store.drafts) { open }
        draft.text = "Kept"
        open = gone
        draft.text = "Of a chat deleted elsewhere"
        draft.save()
        let state = fake.state()
        defer { state.quit() }
        await state.launch()
        #expect(state.draft.chats == [a])
    }

    /// The app's launch keeps the draft of a chat whose file it can't read, as one a newer app
    /// wrote: the chat may be there, and a draft the launch dropped would be gone at the next save.
    @Test func theLaunchKeepsADraftWhoseChatCannotBeRead() async throws {
        let fake = try FakeRuntime("splash-drafts-unread", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let unread = UUID()
        try FileManager.default.createDirectory(at: fake.chats, withIntermediateDirectories: true)
        try Data("{\"newer\": true}".utf8).write(to: fake.chats.appending(path: "\(unread.uuidString).json"))
        let store = ChatStore(folder: fake.chats)
        let draft = Draft(files: store.attachments, kept: store.drafts) { unread }
        draft.text = "Of a chat this app can't read"
        draft.save()
        let state = fake.state()
        defer { state.quit() }
        await state.launch()
        #expect(state.draft.chats == [unread])
    }
}
