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
            try await waitFor(!state.draft.attachments.isEmpty)
            let text = state.draft.text, attachments = state.draft.attachments
            let usage = try await count(state)
            #expect(usage.level == .full && state.contextUsage == usage)
            #expect(!state.canSendDraft && state.overflow == usage)
            state.sendDraft()
            #expect(state.draft.text == text && state.draft.attachments == attachments)
            #expect(state.chats.current?.messages == fixture.history.messages)
            #expect(try fixture.sent().isEmpty)
            await picture("chat-context-full", state: state)

            // A new chat, as the note says, takes the draft along and leaves the long one saved,
            // with no draft: the draft went, not a copy of it.
            state.newChat()
            #expect(state.draft.text == text && state.draft.attachments == attachments)
            #expect(state.overflow == nil)
            #expect(try await count(state).level == .normal)
            #expect(state.canSendDraft)
            await picture("chat-context-room", state: state)
            let counts = try fixture.counts()
            state.sendDraft()
            try await waitFor(state.chats.streaming == nil)
            #expect(state.draft.isEmpty)
            #expect(state.chats.current?.messages.last?.text == "Answer")
            let sent = try fixture.sent()
            #expect(sent.count == 1 && sent[0].count == 1)
            #expect(sent[0][0]["role"] as? String == "user")
            // Sending doesn't count the prompt again.
            #expect(try fixture.counts() == counts)
            #expect(state.chats.conversations.first { $0.id == fixture.history.id } == fixture.history)
            state.chats.selection = fixture.history.id
            #expect(state.draft.isEmpty)
        }
    }

    /// A draft its chat's context can hold stays in its chat as a new chat opens, which shows its
    /// own draft. A new chat that holds a draft of its own keeps it as a draft the context cannot
    /// hold asks for a new chat: that draft stays in its chat, and neither is lost or joined.
    @Test func aDraftItsChatCanHoldStaysAsANewChatOpens() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "Short?"
            let short = try await count(state)
            #expect(short.fits)
            state.newChat()
            #expect(state.draft.isEmpty)
            state.draft.text = "Typed in the new chat"
            state.chats.selection = fixture.history.id
            #expect(state.draft.text == "Short?")
            let long = String(repeating: "x", count: 300)
            state.draft.text = long
            let full = try await count(state)
            #expect(full.level == .full && state.overflow == full)
            state.newChat()
            #expect(state.draft.text == "Typed in the new chat")
            state.chats.selection = fixture.history.id
            #expect(state.draft.text == long)
        }
    }

    /// A draft the context cannot hold goes with a new chat while an image of it is still being
    /// encoded, and the image joins it there once kept: the new chat's draft holds the text and
    /// the image and can go, and the chat it left can send again. Joining the chat it was attached
    /// in, the image made a draft of its own there, which counted an image too few, and the new
    /// chat's draft waited for it for good: neither chat could send.
    @Test func aDraftGoesWithANewChatWithAnImageStillBeingEncoded() async throws {
        try await withServer { fixture in
            let state = fixture.state
            let long = String(repeating: "x", count: 300)
            state.draft.text = long
            let full = try await count(state)
            #expect(full.level == .full && state.overflow == full)
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            #expect(state.draft.attaching == 1 && state.overflow == full)
            state.newChat()
            #expect(state.draft.text == long && state.draft.attaching == 1)
            try await waitFor(state.draft.attaching == 0, "the image")
            #expect(state.draft.text == long && state.draft.attachments.count == 1 && state.canSendDraft)
            state.chats.selection = fixture.history.id
            #expect(state.draft.isEmpty && state.draft.attaching == 0)
            state.draft.text = "Short?"
            #expect(state.canSendDraft)
        }
    }

    /// Deleting a chat forgets its draft, and an image still being encoded for it joins no draft
    /// once kept. Kept, the deleted chat's draft held its text until the app quit, and the late
    /// image made it a draft again.
    @Test func deletingAChatForgetsItsDraft() async throws {
        try await withServer { fixture in
            let state = fixture.state
            let id = fixture.history.id
            state.draft.text = "Not sent"
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            state.delete([id])
            #expect(state.chats.conversations.isEmpty && state.chats.selection == nil && state.draft.isEmpty)
            // Attachments join in the order they were attached: once a later one is kept, the
            // deleted chat's has gone where it goes.
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            try await waitFor(state.draft.attachments.count == 1, "the later image")
            // The deleted chat's id, which nothing lists, as its draft was kept under it.
            state.chats.selection = id
            #expect(state.draft.isEmpty && state.draft.attaching == 0 && state.draft.notice == nil)
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

    /// Return pressed right after an image is pasted, while it is still being encoded, doesn't
    /// send the text without it and leave the image in the next draft: Send waits for the image,
    /// and the message goes with it.
    @Test func sendWaitsForAnImageStillBeingAttached() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.chats.newChat()
            state.draft.text = "What is in this picture?"
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            state.sendDraft()
            // Return again once the image shows, if the first one waited for it.
            if !state.draft.isEmpty {
                #expect(!state.canSendDraft)
                try await waitFor(!state.draft.attachments.isEmpty)
                state.sendDraft()
            }
            try await waitFor(state.chats.streaming == nil && !((try? fixture.sent()) ?? []).isEmpty)
            let sent = try fixture.sent()
            let parts = try #require(sent.last?.last?["content"] as? [[String: Any]])
            #expect(parts.map { $0["type"] as? String } == ["image_url", "text"])
            #expect(state.draft.isEmpty)
        }
    }

    /// A PDF attaches as its file, with its name and pages, and goes to the model as a file part;
    /// one over the model's pages stays in the draft, which says why.
    @Test func aPDFGoesAsAFileWithinTheModelsLimits() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.chats.newChat()
            #expect(state.server.limits?.pdf_pages == 3)
            let folder = fixture.root.appending(path: "documents")
            let long = try Showcase.pdf(pages: 4, at: folder.appending(path: "Long.pdf"))
            let plan = try Showcase.pdf(pages: 2, at: folder.appending(path: "Plan.pdf"))
            state.draft.attach(pdfs: [long])
            try await waitFor(!state.draft.attachments.isEmpty)
            #expect(state.draft.attachments.map(\.pages) == [4])
            #expect(state.attachmentProblem == "Long.pdf has more than the model’s 3 pages" && !state.canSendDraft)
            state.sendDraft()
            #expect(state.draft.attachments.count == 1 && state.chats.streaming == nil)
            state.draft.remove(at: 0)
            state.draft.attach(pdfs: [plan])
            try await waitFor(!state.draft.attachments.isEmpty)
            let attachment = state.draft.attachments[0]
            #expect(attachment.kind == .pdf && attachment.name == "Plan.pdf" && attachment.pages == 2)
            #expect(try Data(contentsOf: state.chats.attachments.appending(path: attachment.file)) == Data(contentsOf: plan))
            #expect(state.canSendDraft)
            state.sendDraft()
            try await waitFor(state.chats.streaming == nil && !((try? fixture.sent()) ?? []).isEmpty)
            let parts = try #require(try fixture.sent().last?.last?["content"] as? [[String: Any]])
            #expect(parts.map { $0["type"] as? String } == ["file", "text"])
            #expect(parts[0]["file"] as? [String: String]
                == ["filename": "Plan.pdf", "file_data": "data:application/pdf;base64,\(try Data(contentsOf: plan).base64EncodedString())"])
            #expect(state.chats.current?.messages.first?.attachments == [attachment])
        }
    }

    /// A message the server refuses goes back to the composer, its attachments with it, and says
    /// why until it is sent; the chat stays as it was, and a new one stays new.
    @Test func aRefusedMessageGoesBackToTheComposer() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.chats.newChat()
            state.draft.text = "Hello"
            state.sendDraft()
            // Once the model has titled it, which it does after the first exchange.
            try await waitFor(state.chats.streaming == nil && state.chats.current?.title == "A Title")
            let chat = try #require(state.chats.current)
            state.draft.text = "refuse this"
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            try await waitFor(!state.draft.attachments.isEmpty)
            let attachments = state.draft.attachments
            state.sendDraft()
            #expect(state.draft.isEmpty)
            try await waitFor(state.chats.streaming == nil && state.draft.notice != nil)
            #expect(state.draft.text == "refuse this" && state.draft.attachments == attachments)
            #expect(state.draft.thumbnails[attachments[0].id] != nil)
            #expect(state.draft.notice == "Request body too large")
            #expect(state.chats.current == chat)
            #expect(attachments.allSatisfy { FileManager.default.fileExists(atPath: state.chats.attachments.appending(path: $0.file).path) })
            // In a new chat, which stays new: nothing kept of it, in the list or on disk. Its draft
            // is its own: the other chat's message and why stay in that chat.
            state.chats.newChat()
            #expect(state.draft.isEmpty && state.draft.notice == nil)
            state.draft.text = "refuse this"
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            try await waitFor(!state.draft.attachments.isEmpty)
            let attached = state.draft.attachments
            let count = state.chats.conversations.count
            state.sendDraft()
            try await waitFor(state.chats.streaming == nil && state.draft.notice != nil)
            #expect(state.draft.notice == "Request body too large")
            #expect(state.chats.conversations.count == count && state.chats.selection == nil)
            #expect(state.draft.attachments == attached)
            state.chats.flush()
            let written = try FileManager.default.contentsOfDirectory(at: fixture.root.appending(path: "chats"), includingPropertiesForKeys: nil)
            #expect(written.filter { $0.pathExtension == "json" }.count == count)
            // Sent once it can go, the note goes with it.
            state.draft.text = "Then this"
            state.sendDraft()
            #expect(state.draft.notice == nil)
            try await waitFor(state.chats.streaming == nil && state.chats.current?.messages.count == 2)
            #expect(state.chats.current?.messages.first?.attachments == attached)
            state.chats.selection = chat.id
            #expect(state.draft.text == "refuse this" && state.draft.attachments == attachments)
            #expect(state.draft.notice == "Request body too large")
        }
    }

    /// The first message of a new chat stays in the composer, as it was sent, until the server
    /// takes it: while the server holds it and once it has refused it, the composer holds the text
    /// and the attachments it held, with their pictures, Send being Stop meanwhile, and no chat is
    /// listed, open or written; the refusal adds only why. Sent again and taken, the message is
    /// the chat's, which opens, listed once, and the composer is empty.
    @Test func aNewChatsFirstMessageStaysInTheComposerUntilTheServerTakesIt() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.chats.newChat()
            state.draft.text = "refuse this"
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            state.draft.attach(pdfs: [try Showcase.pdf(pages: 2, at: fixture.root.appending(path: "documents/Plan.pdf"))])
            try await waitFor(state.draft.attachments.count == 2, "the attachments")
            let before = Composed(state.draft)
            #expect(before.pictured.count == 1 && state.canSendDraft)
            let composed = Seen { Composed(state.draft) }
            let chats = Seen { Chats(state.chats) }
            let listed = Chats(state.chats)

            try fixture.hold()
            state.sendDraft()
            #expect(Composed(state.draft) == before && Chats(state.chats) == listed)
            #expect(state.chats.sendingFirstMessage && state.chats.answering && !state.canSendDraft && state.showsContext)
            try await fixture.held()
            #expect(Composed(state.draft) == before && Chats(state.chats) == listed)
            #expect(try fixture.written() == [fixture.history.id])
            try fixture.release()
            try await waitFor(state.chats.streaming == nil, "the refusal")
            var refused = before
            refused.notice = "Request body too large"
            #expect(composed.values == [before, refused] && chats.values == [listed], "through the refusal")
            #expect(!state.chats.sendingFirstMessage && !state.chats.answering && state.canSendDraft)
            #expect(try fixture.written() == [fixture.history.id])
            #expect(before.attachments.allSatisfy { FileManager.default.fileExists(atPath: state.chats.attachments.appending(path: $0.file).path) })

            // What the server takes: the same attachments under other words.
            state.draft.text = "Summarize the plan and say what the picture adds."
            let again = Composed(state.draft)
            #expect(again.notice == nil && again.attachments == before.attachments && again.pictured == before.pictured)
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            #expect(Composed(state.draft) == again && Chats(state.chats) == listed)
            #expect(try fixture.written() == [fixture.history.id])
            try fixture.release()
            try await waitFor(state.chats.streaming == nil, "the reply")
            let chat = try #require(state.chats.current)
            #expect(chat.id != fixture.history.id && chat.messages.first?.attachments == before.attachments)
            #expect(chat.messages.map(\.text) == ["Summarize the plan and say what the picture adds.", "Answer"])
            #expect(composed.values == [before, refused, again, Composed()], "the composer, through the flow")
            #expect(chats.values == [listed, Chats(listed: [chat.id, fixture.history.id], open: chat.id)], "listed once, as the server takes it")
            #expect(Set(try fixture.written()) == [chat.id, fixture.history.id])
        }
    }

    /// Stop while the server holds a new chat's first message gives the message back to edit: the
    /// composer holds it as it did, with nothing to say, and no chat was kept. Another chat opened
    /// while the server holds the message shows its own draft, to write in while Send waits for
    /// the reply, and the new chat opened again holds the message, Stop in Send's place. Another
    /// chat open as the server takes the message stays open, its draft as it was written, the
    /// message being the new chat's. With one draft for every chat, the other chat's composer held
    /// the new chat's message, which took no edit, under a Send that did nothing, with no Stop.
    @Test func aFirstMessageHeldIsStoppedOrTakenWhereverTheWindowIs() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.chats.newChat()
            state.draft.text = "Plan a trip to Kyoto in spring"
            let before = Composed(state.draft), listed = Chats(state.chats)
            let composed = Seen { Composed(state.draft) }
            let chats = Seen { Chats(state.chats) }
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            state.chats.stop()
            try await waitFor(state.chats.streaming == nil, "the stop")
            #expect(composed.values == [before] && chats.values == [listed] && state.canSendDraft)
            #expect(try fixture.sent().isEmpty && fixture.written() == [fixture.history.id])

            state.sendDraft()
            try await waitFor(state.chats.streaming != nil, "the message sent again")
            state.chats.selection = fixture.history.id
            #expect(state.draft.isEmpty && !state.chats.sendingFirstMessage && !state.chats.answering && !state.canSendDraft)
            state.draft.text = "Typed in the saved chat"
            state.chats.newChat()
            #expect(Composed(state.draft) == before && state.chats.sendingFirstMessage && state.chats.answering)
            state.chats.selection = fixture.history.id
            try fixture.release()
            try await waitFor(state.chats.streaming == nil, "the reply")
            #expect(state.draft.text == "Typed in the saved chat" && state.chats.selection == fixture.history.id)
            let new = try #require(state.chats.conversations.first)
            #expect(new.id != fixture.history.id && new.messages.map(\.text) == ["Plan a trip to Kyoto in spring", "Answer"])
            #expect(state.chats.conversations.count == 2)
            state.chats.newChat()
            #expect(state.draft.isEmpty)
        }
    }

    /// A quit while the server holds a new chat's first message writes no chat: what the app
    /// leaves on disk is what it had.
    @Test func aQuitWhileTheServerHoldsAFirstMessageWritesNoChat() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.chats.newChat()
            state.draft.text = "Plan a trip to Kyoto in spring"
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            #expect(state.quitCost.requests == 1)
            state.quit()
            AppDelegate(state: state).applicationWillTerminate(Notification(name: NSApplication.willTerminateNotification))
            #expect(try fixture.written() == [fixture.history.id])
            try await waitFor(state.chats.streaming == nil, "the stop")
            #expect(state.chats.conversations == [fixture.history] && state.draft.text == "Plan a trip to Kyoto in spring")
            try fixture.release()
        }
    }

    /// A quit while the server holds a later message keeps the message in its chat, which shows
    /// it, with its attachments and its answer stopped, and the next launch reads it there and
    /// keeps the attachments' files, which the chat names. Given back to the composer, which goes
    /// with the app, the message was lost, and the next launch took its attachments for those of
    /// a draft and removed them.
    @Test func aQuitWhileTheServerHoldsALaterMessageKeepsItInItsChat() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "Short?"
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            try await waitFor(!state.draft.attachments.isEmpty, "the picture")
            let attachments = state.draft.attachments
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            #expect(state.draft.isEmpty && state.quitCost.requests == 1)
            state.quit()
            AppDelegate(state: state).applicationWillTerminate(Notification(name: NSApplication.willTerminateNotification))
            try await waitFor(state.chats.streaming == nil, "the stop")
            #expect(state.draft.isEmpty)
            let kept = try #require(state.chats.conversations.first)
            #expect(kept.id == fixture.history.id && kept.messages.map(\.text) == fixture.history.messages.map(\.text) + ["Short?", ""])
            #expect(kept.messages.dropLast().last?.attachments == attachments && kept.messages.last?.stopped == true)
            state.chats.flush()
            let next = ChatStore(folder: fixture.root.appending(path: "chats"))
            #expect(next.conversations == [kept])
            next.adoptFiles(keeping: [])
            next.flush()
            #expect(attachments.allSatisfy { FileManager.default.fileExists(atPath: next.attachments.appending(path: $0.file).path) })
            try fixture.release()
        }
    }

    /// A later message stopped while the server holds it goes back to the composer, with nothing
    /// to say, and its chat is as it was, in the list and on disk.
    @Test func aLaterMessageStoppedBeforeTheServerTakesItGoesBack() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "Short?"
            try fixture.hold()
            state.sendDraft()
            #expect(state.draft.isEmpty && state.chats.current?.messages.count == 4 && state.chats.answering)
            try await fixture.held()
            state.chats.stop()
            try await waitFor(state.chats.streaming == nil, "the stop")
            #expect(state.draft.text == "Short?" && state.draft.notice == nil)
            #expect(state.chats.current == fixture.history && state.chats.conversations == [fixture.history])
            #expect(try fixture.sent().isEmpty)
            try fixture.release()
        }
    }

    /// A message that comes back once another chat is open comes back to its chat's draft, and
    /// its chat opens again, where the composer holds it, says why and Send sends it: a later
    /// message the server refuses, alone, what was typed in the other chat meanwhile staying in
    /// that chat's draft; a new chat's first message it refuses, which the new chat's composer
    /// still holds; and a later message stopped from elsewhere, as a switch of models stops it,
    /// back with nothing to say. Left under the chat opened meanwhile, the composer said nothing
    /// of why there, its reason being about another chat, and Send sent the message there; put
    /// back before what was typed in the other chat, a blank line apart, Send sent both to its
    /// chat.
    @Test func aMessageThatComesBackOnceAnotherChatIsOpenOpensItsChat() async throws {
        try await withServer { fixture in
            let state = fixture.state
            state.draft.text = "refuse this"
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            state.chats.newChat()
            state.draft.text = "Typed in the new chat"
            try fixture.release()
            try await waitFor(state.chats.streaming == nil, "the refusal")
            #expect(state.chats.selection == fixture.history.id && state.chats.current == fixture.history)
            #expect(state.draft.text == "refuse this" && state.draft.notice == "Request body too large")
            state.chats.newChat()
            #expect(state.draft.text == "Typed in the new chat" && state.draft.notice == nil)

            state.draft.text = "refuse this"
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            state.chats.selection = fixture.history.id
            try fixture.release()
            try await waitFor(state.chats.streaming == nil, "the refusal")
            #expect(state.chats.selection == nil && state.chats.conversations == [fixture.history])
            #expect(state.draft.text == "refuse this")
            #expect(state.draft.notice == "Request body too large")

            state.chats.selection = fixture.history.id
            state.draft.text = "Short?"
            try fixture.hold()
            state.sendDraft()
            try await fixture.held()
            state.chats.newChat()
            state.chats.stop()
            try await waitFor(state.chats.streaming == nil, "the stop")
            #expect(state.chats.selection == fixture.history.id && state.chats.current == fixture.history)
            #expect(state.draft.text == "Short?" && state.draft.notice == nil)
            #expect(try fixture.sent().isEmpty && fixture.written() == [fixture.history.id])
            try fixture.release()
        }
    }

    /// A chat opened shows its last count at once, while its prompt is counted again, and one not
    /// counted yet an empty ring: the composer's ring never goes as chats open. With the last
    /// count of all alone, it went as each chat opened and came back, fading in, once counted.
    @Test func theRingStaysAsChatsOpen() async throws {
        try await withServer { fixture in
            let state = fixture.state
            let saved = try await count(state)
            state.chats.newChat()
            state.draft.text = "Hello"
            state.sendDraft()
            try await waitFor(state.chats.streaming == nil && state.chats.current?.messages.count == 2)
            let answered = try #require(state.chats.selection)
            let usage = try await count(state)
            #expect(usage != saved)
            state.chats.selection = fixture.history.id
            #expect(state.showsContext && state.contextUsage == saved)
            state.chats.selection = answered
            #expect(state.showsContext && state.contextUsage == usage)
            // A chat not counted yet: its draft, in a new chat.
            state.chats.newChat()
            state.draft.text = "Plan a trip"
            #expect(state.showsContext && state.contextUsage == nil)
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

/// What the composer holds: its text, what is attached, which attachments show a picture, and what
/// it says over them.
private struct Composed: Equatable {
    var text = ""
    var attachments: [Splash.Attachment] = []
    var pictured: Set<UUID> = []
    var notice: String?
}

extension Composed {
    @MainActor init(_ draft: Draft) {
        self.init(text: draft.text, attachments: draft.attachments, pictured: Set(draft.thumbnails.keys), notice: draft.notice)
    }
}

/// What the chats show: those listed, in the list's order, and the open one.
private struct Chats: Equatable {
    var listed: [UUID]
    var open: UUID?
}

extension Chats {
    @MainActor init(_ store: ChatStore) {
        self.init(listed: store.conversations.map(\.id), open: store.selection)
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
        let runtime = try FakeRuntime("splash-context-flow", launcher: "context_server")
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

    /// Has the server hold the next message before it takes or refuses it.
    func hold() throws {
        try? FileManager.default.removeItem(at: root.appending(path: "held"))
        try Data().write(to: root.appending(path: "hold"))
    }

    /// Waits until the server holds a message.
    func held() async throws {
        try await waitFor(FileManager.default.fileExists(atPath: root.appending(path: "held").path), "the server to hold the message")
    }

    /// Lets the server go on with the message it holds.
    func release() throws {
        try FileManager.default.removeItem(at: root.appending(path: "hold"))
    }

    /// The chats on disk, by their files' names, once the app has written what it changed so far.
    func written() throws -> [UUID] {
        state.chats.flush()
        return try FileManager.default.contentsOfDirectory(at: root.appending(path: "chats"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.compactMap { UUID(uuidString: $0.deletingPathExtension().lastPathComponent) }
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
}
