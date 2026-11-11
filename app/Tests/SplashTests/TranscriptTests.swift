import AppKit
import Foundation
import SwiftUI
import Testing
import WebKit

@testable import Splash

@MainActor
@Suite(.serialized)
struct TranscriptTests {
    let asked = Date(timeIntervalSince1970: 1_000)

    @Test func sendsMessagesAsThePageReadsThem() throws {
        let conversation = Conversation(messages: [
            ChatMessage(role: .user, text: "What is this?",
                        attachments: [Showcase.pictureAttachment,
                                      Splash.Attachment(kind: .pdf, type: "application/pdf", bytes: 900, name: "Plan.pdf", pages: 1)]),
            ChatMessage(role: .assistant, text: "A cat.", reasoning: "Whiskers.", model: "a/Big-GGUF:Q4", tokensPerSecond: 38.4,
                        thinkingTime: 3.2, reasoningTokens: 1_200, stopped: true),
            ChatMessage(role: .user, text: "Sure?"),
            ChatMessage(role: .assistant, text: ""),
        ])
        let live = ChatStore.Live(asked: asked, read: 0.45)
        let name = { (model: String) in "Name of \(model)" }
        let page = TranscriptPage(conversation, askAgain: .init(enabled: false, help: "Ask Big Again"), name: name)
            .streaming(TranscriptMessage(conversation.messages[3], live: live, asleep: false, name: name))
        let json = try #require(JSONSerialization.jsonObject(with: WebPage.encoder.encode(page)) as? [String: Any])
        #expect(json["id"] as? String == conversation.id.uuidString)
        #expect(json["askAgain"] as? [String: AnyHashable] == ["enabled": false, "help": "Ask Big Again"])
        let messages = try #require(json["messages"] as? [[String: Any]])
        #expect(messages.map { $0["role"] as? String } == ["user", "assistant", "user", "assistant"])
        // Every key there, null for none.
        #expect(messages.allSatisfy { Set($0.keys) == ["id", "role", "text", "reasoning", "images", "documents", "error", "line", "live", "details"] })
        // Images as the addresses the page loads them from, which the app serves; PDFs as their
        // chips say them.
        #expect(messages[0]["images"] as? [String] == ["chat-image://\(conversation.messages[0].id.uuidString)/0"])
        #expect(messages[0]["documents"] as? [[String: String]] == [["name": "Plan.pdf", "details": "1 page · \(Format.size(900))"]])
        #expect(messages[2]["documents"] as? [[String: String]] == [])
        #expect(messages[0]["line"] is NSNull && messages[0]["error"] is NSNull && messages[0]["details"] as? String == "")
        #expect(messages[1]["line"] as? [String: AnyHashable] == ["verb": "Thinking", "facts": "1.2K tokens · 3.2s", "since": NSNull()])
        #expect(messages[1]["details"] as? String == "Name of a/Big-GGUF:Q4 · 38 tok/s · Stopped")
        #expect(messages[1]["live"] as? Bool == false)
        // The reply that streams, the last.
        #expect(messages[3]["live"] as? Bool == true)
        #expect(messages[3]["line"] as? [String: AnyHashable] == ["verb": "Reading", "facts": "45%", "since": NSNull()])
        // Dates in milliseconds, as JavaScript counts them.
        let waiting = TranscriptMessage(conversation.messages[3], live: ChatStore.Live(asked: asked),
                                        asleep: false, name: { $0 })
        let line = try #require(JSONSerialization.jsonObject(with: WebPage.encoder.encode(waiting.line)) as? [String: Any])
        #expect(line["since"] as? Double == 1_000_000)
    }

    @Test func theReplyLineSaysWhatComesBeforeTheAnswer() {
        let waiting = ChatMessage(role: .assistant, text: "")
        let started = asked + 2
        func line(_ message: ChatMessage, _ live: ChatStore.Live?, asleep: Bool = false) -> ReplyLine? {
            ReplyLine(message, live: live, asleep: asleep)
        }
        // Reading: how much of the prompt, or the seconds since it was asked until the server says.
        #expect(line(waiting, .init(asked: asked)) == ReplyLine(verb: "Reading", facts: nil, since: asked))
        #expect(line(waiting, .init(asked: asked, read: 0.456))
            == ReplyLine(verb: "Reading", facts: "45%", since: nil))
        // A model the idle release put to sleep loads again first.
        #expect(line(waiting, .init(asked: asked, read: 0.5), asleep: true)
            == ReplyLine(verb: "Loading into memory", facts: nil, since: asked))
        // Thinking, the seconds since it began.
        let thinking = ChatMessage(role: .assistant, text: "", reasoning: "Hm.")
        #expect(line(thinking, .init(asked: asked, thinking: started))
            == ReplyLine(verb: "Thinking", facts: nil, since: started))
        // Once it answers, what the thinking took, though the reply still streams.
        var answered = thinking
        answered.text = "Yes."
        answered.thinkingTime = 9.96
        #expect(line(answered, .init(asked: asked, thinking: started))
            == ReplyLine(verb: "Thinking", facts: "10s", since: nil))
        answered.thinkingTime = nil
        #expect(line(answered, nil) == ReplyLine(verb: "Thinking", facts: nil, since: nil))
        // No line for a reply that answers without thinking, or once it has ended.
        #expect(line(ChatMessage(role: .assistant, text: "Yes."), .init(asked: asked)) == nil)
        #expect(line(waiting, nil) == nil)
    }

    @Test func onlyTheStreamingReplyGoesAgainWhileItStreams() {
        let user = ChatMessage(role: .user, text: "Hi")
        var reply = ChatMessage(role: .assistant, text: "He")
        let conversation = Conversation(messages: [user, reply])
        let askAgain = TranscriptPage.AskAgain(enabled: false, help: "Ask Again")
        func page(_ messages: [ChatMessage], id: UUID = conversation.id, askAgain: TranscriptPage.AskAgain = askAgain) -> TranscriptPage {
            TranscriptPage(Conversation(id: id, messages: messages), askAgain: askAgain, name: { $0 })
        }
        let shown = page([user, reply])
        #expect(Transcript.change(from: nil, to: shown) == .page(shown))
        #expect(Transcript.change(from: shown, to: page([user, reply])) == .none)
        reply.text = "Hello"
        let grown = page([user, reply])
        #expect(Transcript.change(from: shown, to: grown) == .message(grown.messages[1]))
        // Anything else, the whole conversation: another one, a message sent, an earlier one
        // changed, Ask Again allowed.
        for other in [page([user, reply], id: UUID()), page([user, reply, user]), page([ChatMessage(role: .user, text: "Hey"), reply]),
                      page([user, reply], askAgain: .init(enabled: true, help: "Ask Again"))] {
            #expect(Transcript.change(from: shown, to: other) == .page(other))
        }
    }

    @Test func thePageDrawsRepliesSafelyAndAsksAgain() async throws {
        try Showcase.requireScripts()
        let transcript = Transcript(runtime: Showcase.runtime, attachments: Showcase.attachments)
        var askedAgain = 0
        transcript.askAgain = { askedAgain += 1 }
        transcript.page.load()
        let webView = try #require(transcript.page.webView)
        try await waitFor(transcript.page.loaded)
        let conversation = Conversation(messages: [
            ChatMessage(role: .user, text: "<b>Hi</b>",
                        attachments: [Showcase.pictureAttachment,
                                      Splash.Attachment(kind: .pdf, type: "application/pdf", bytes: 2_048, name: "<i>Plan</i>.pdf", pages: 2)]),
            ChatMessage(role: .assistant, text: """
                [web](https://a.b) [mail](mailto:a@b.c) [file](file:///Applications) [script](javascript:alert(1)) \
                ![cat](https://a.b/cat.png) <b>bold</b>

                3. three
                4. four

                | a | b |
                |--:|---|
                | 1<br>2 | x |

                ```swift
                let a = 1
                ```
                """),
        ])
        let askAgain = TranscriptPage.AskAgain(enabled: true, help: "Ask Big Again")
        transcript.show(TranscriptPage(conversation, askAgain: askAgain, name: { $0 }), reply: nil)
        func run(_ script: String) async throws -> String {
            try await webView.callAsyncJavaScript(script, contentWorld: .page) as? String ?? ""
        }
        // Raw HTML stays text.
        #expect(try await run("return document.querySelector('.bubble').textContent") == "<b>Hi</b>")
        // The user's image, which the app serves.
        let loaded = try await settled({ try await run("const image = document.querySelector('.attachments img'); return image.complete ? String(image.naturalWidth) : ''") }) {
            !$0.isEmpty
        }
        #expect(loaded == "240")
        // The PDF's chip after the images: its glyph, name and details, as text.
        #expect(try await run("return document.querySelector('.attachments :last-child').className") == "pdf")
        #expect(try await run("return document.querySelector('.pdf-name').textContent") == "<i>Plan</i>.pdf")
        #expect(try await run("return document.querySelector('.pdf-details').textContent") == "2 pages · \(Format.size(2_048))")
        #expect(try await run("return getComputedStyle(document.querySelector('.pdf .glyph')).width") != "0px")
        #expect(try await run("return document.querySelector('.markdown p').innerHTML").contains("&lt;b&gt;bold&lt;/b&gt;"))
        // Web and mail links stay links; others keep their text; an image is a link to it.
        #expect(try await run("return [...document.querySelectorAll('.markdown a')].map(a => a.getAttribute('href')).join(' ')")
            == "https://a.b mailto:a@b.c https://a.b/cat.png")
        #expect(try await run("return document.querySelector('.markdown p').textContent") == "web mail file script cat <b>bold</b>")
        #expect(try await run("return document.querySelector('.markdown img') === null ? 'none' : 'image'") == "none")
        // Numbered from the list's start; a table cell's <br> a line break, its alignment the grid's.
        #expect(try await run("return [...document.querySelectorAll('.marker')].map(m => m.textContent).join(' ')") == "3. 4.")
        #expect(try await run("return document.querySelector('td div').innerHTML") == "1<br>\n2")
        #expect(try await run("return String(document.querySelector('th').getAttribute('style'))") == "null")
        // Code under its language, highlighted.
        #expect(try await run("return document.querySelector('figcaption span').textContent") == "swift")
        #expect(try await run("return document.querySelector('pre code').textContent") == "let a = 1")
        #expect(try await run("return String(document.querySelectorAll('pre .hljs-keyword').length)") == "1")
        // Ask Again on the last reply, which asks the app.
        #expect(try await run("return document.querySelector('.last .again').title") == "Ask Big Again")
        _ = try await run("document.querySelector('.last .again').click(); return ''")
        try await waitFor(askedAgain == 1)
    }

    @Test func theComposerTakesTheKeyboardBackFromThePage() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-transcript-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Conversation(messages: [ChatMessage(role: .user, text: "Hi"), ChatMessage(role: .assistant, text: "Hello.")])
        try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        let state = AppState.forTests(in: folder)
        state.chats.selection = conversation.id
        _ = NSApplication.shared
        let window = NSWindow(contentViewController: NSHostingController(rootView: ChatScreen().environment(state)))
        window.setContentSize(CGSize(width: 900, height: 600))
        window.setFrameOrigin(NSPoint(x: -30000, y: -30000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        try await waitFor(state.transcript.page.loaded)
        let webView = try #require(state.transcript.page.webView)
        // The reader clicked in the page, then in the composer.
        #expect(window.makeFirstResponder(webView))
        let composer = try #require(Self.textViews(in: window.contentView!).first { !$0.isFieldEditor })
        #expect(window.makeFirstResponder(composer))
        func key(_ characters: String, _ code: UInt16) throws -> NSEvent {
            try #require(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0,
                                          windowNumber: window.windowNumber, context: nil, characters: characters,
                                          charactersIgnoringModifiers: characters, isARepeat: false, keyCode: code))
        }
        window.sendEvent(try key("a", 0))
        #expect(state.draft.text == "a")
        // The page leaves Return to the composer, which sends with it.
        #expect(!webView.performKeyEquivalent(with: try key("\r", 36)))
    }

    @Test func followsTheLatestUntilTheReaderScrollsUp() async throws {
        let transcript = Transcript(runtime: Showcase.runtime, attachments: Showcase.attachments)
        transcript.page.load()
        let webView = try #require(transcript.page.webView)
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: -30000, y: -30000, width: 800, height: 500), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.contentView?.addSubview(webView)
        webView.frame = window.contentView?.bounds ?? .zero
        // Drawn as on screen, as WebKit holds back the animation frames of a page it can't see.
        Snapshot.drawsOffScreen(webView)
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        try await waitFor(transcript.page.loaded)
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 12)
        var messages = (0..<12).flatMap { _ in [ChatMessage(role: .user, text: "Tell me more."), ChatMessage(role: .assistant, text: paragraph)] }
        let id = UUID()
        func show(_ messages: [ChatMessage]) {
            transcript.show(TranscriptPage(Conversation(id: id, messages: messages), askAgain: .init(enabled: false, help: ""), name: { $0 }), reply: nil)
        }
        // Where the page is two frames on, however long WebKit takes to draw them: by then the
        // scroll and resize observers have seen what came before, as the follow of a reply that grew.
        // A page that draws no frame is read after 5 s, so that the expectation fails where it is.
        func scrolled() async throws -> (top: Double, bottom: Double, latest: Bool) {
            let values = try await webView.callAsyncJavaScript(
                """
                await Promise.race([new Promise(drawn => requestAnimationFrame(() => requestAnimationFrame(drawn))),
                                    new Promise(late => setTimeout(late, 5000))])
                return [scrollY, document.scrollingElement.scrollHeight - innerHeight, document.getElementById('latest').classList.contains('shown') ? 1 : 0]
                """, contentWorld: .page) as? [Double] ?? []
            return (values[0], values[1], values[2] == 1)
        }
        // A conversation opens at its latest message, and follows a reply that grows.
        show(messages)
        var position = try await scrolled()
        #expect(position.top == position.bottom && position.bottom > 0 && !position.latest)
        messages[messages.count - 1].text += "\n\n" + paragraph
        show(messages)
        position = try await scrolled()
        #expect(position.top == position.bottom && !position.latest)
        // Scrolled up, it stays where the reader left it, and offers the way back: at the top, so
        // that no message across the screen's edge, laid out at its own height in place of the room
        // kept for it, moves the scroller to keep what the reader reads in place.
        _ = try await webView.callAsyncJavaScript("scrollTo(0, 0)", contentWorld: .page)
        _ = try await scrolled()
        messages[messages.count - 1].text += "\n\n" + paragraph
        show(messages)
        position = try await scrolled()
        #expect(position.top == 0 && position.bottom > 0 && position.latest)
        // Which glides there over 0.4 s, and the conversation follows again.
        _ = try await webView.callAsyncJavaScript("document.getElementById('latest').click()", contentWorld: .page)
        position = try await settled(scrolled, seconds: 2) { $0.top == $0.bottom }
        #expect(position.top == position.bottom && !position.latest)
        // A message sent scrolls to it, wherever the reader was.
        _ = try await webView.callAsyncJavaScript("scrollTo(0, 0)", contentWorld: .page)
        messages += [ChatMessage(role: .user, text: "And then?"), ChatMessage(role: .assistant, text: "")]
        show(messages)
        position = try await scrolled()
        #expect(position.top == position.bottom && !position.latest)
    }

    @Test func aReplyStreamsAtAnEvenPaceDrawingOnlyItsEnd() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let question = ChatMessage(role: .user, text: "Show me.")
        var reply = ChatMessage(role: .assistant, text: "")
        let page = TranscriptPage(Conversation(messages: [question, reply]), askAgain: .init(enabled: false, help: ""), name: { $0 })
        let live = ChatStore.Live(asked: .now)
        transcript.show(page, reply: nil)
        /// Sends `reply`, streaming or not, as the app does, and returns what the page shows of it
        /// at once, before any frame: its text's length.
        func send(live streaming: Bool) async throws -> Int {
            let message = TranscriptMessage(reply, live: streaming ? live : nil, asleep: false, name: { $0 })
            let json = String(decoding: try WebPage.encoder.encode(message), as: UTF8.self)
            return try await webView.callAsyncJavaScript(
                "transcript.update(JSON.parse(json)); return document.querySelector('.reply .markdown').textContent.length",
                arguments: ["json": json], contentWorld: .page) as? Int ?? -1
        }
        func run(_ script: String) async throws -> String {
            try await webView.callAsyncJavaScript(script, contentWorld: .page) as? String ?? ""
        }
        func shows(_ text: String) async throws {
            let shown = try await settled({ try await run("return document.querySelector('.reply .markdown').textContent") }) { $0 == text }
            #expect(shown == text)
        }
        // What comes shows over the frames that follow, not at once, and then all of it; a fence
        // still open shows its code as it is, its last line cut short too.
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 20).trimmingCharacters(in: .whitespaces)
        reply.text = "\(paragraph)\n\n```swift\nlet a = 1"
        #expect(try await send(live: true) < paragraph.count)
        // The paragraph, then the code under its language.
        try await shows("\(paragraph)swiftlet a = 1")
        #expect(try await run("return String(document.querySelectorAll('pre .hljs-keyword').length)") == "0")
        // The paragraph, which is done, and the code's button, which a redraw would replace.
        _ = try await run("window.kept = [document.querySelector('.reply .markdown > p'), document.querySelector('.reply .copy-code')]; return ''")
        // The fence closes: its code is highlighted, once, and the rest stays as drawn.
        reply.text += "\nlet b = 2\n```\n\nDone."
        _ = try await send(live: true)
        try await shows("\(paragraph)swiftlet a = 1\nlet b = 2Done.")
        #expect(try await run("return String(document.querySelectorAll('pre .hljs-keyword').length)") == "2")
        #expect(try await run("""
            return [document.querySelector('.reply .markdown > p') === kept[0], document.querySelector('.reply .copy-code') === kept[1]].join(' ')
            """) == "true true")
        // A reply that ends shows all of it at once, whatever is left.
        reply.text += String(repeating: " And more.", count: 200)
        #expect(try await send(live: false) == "\(paragraph)swiftlet a = 1\nlet b = 2Done.".count + 10 * 200)
        // Copy gives its Markdown.
        _ = try await run("document.querySelector('.copy-reply').click(); return ''")
        #expect(try await settled({ NSPasteboard.general.string(forType: .string) ?? "" }) { $0 == reply.text } == reply.text)
    }

    @Test func aLongThoughtShowsItsLatestLinesWhileItStreamsAndAllOnceOpened() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let thought = (1...300).map { "Line \($0) of the model’s thinking." }.joined(separator: "\n")
        let reply = ChatMessage(role: .assistant, text: "", reasoning: thought)
        let page = TranscriptPage(Conversation(messages: [ChatMessage(role: .user, text: "Think."), reply]),
                                  askAgain: .init(enabled: false, help: ""), name: { $0 })
        let live = ChatStore.Live(asked: .now, thinking: .now)
        transcript.show(page, reply: TranscriptMessage(reply, live: live, asleep: false, name: { $0 }))
        func text() async throws -> String {
            try await webView.callAsyncJavaScript("return document.querySelector('.reply .text > div').textContent",
                                                  contentWorld: .page) as? String ?? ""
        }
        // Its latest lines, from a line's start, which the line fades at the top.
        let latest = try await settled(text) { $0.hasSuffix("Line 300 of the model’s thinking.") }
        #expect(latest.hasPrefix("Line ") && latest.count <= 2000 && thought.hasSuffix(latest))
        #expect(try await settled({
            try await webView.callAsyncJavaScript("return document.querySelector('.reply .line').classList.contains('tall')",
                                                  contentWorld: .page) as? Bool ?? false
        }) { $0 })
        // The reader closes it and opens it again: all of it.
        _ = try await webView.callAsyncJavaScript("document.querySelector('.reply .toggle').click(); document.querySelector('.reply .toggle').click()",
                                                  contentWorld: .page)
        #expect(try await text() == thought)
    }

    @Test func aThoughtFoldsFromItsLatestLinesAsTheAnswerStarts() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let thought = (1...40).map { "Line \($0) of the model’s thinking." }.joined(separator: "\n")
        var reply = ChatMessage(role: .assistant, text: "", reasoning: thought)
        let page = TranscriptPage(Conversation(messages: [ChatMessage(role: .user, text: "Think."), reply]),
                                  askAgain: .init(enabled: false, help: ""), name: { $0 })
        let live = ChatStore.Live(asked: .now, thinking: .now)
        transcript.show(page, reply: TranscriptMessage(reply, live: live, asleep: false, name: { $0 }))
        func height() async throws -> Double {
            try await webView.callAsyncJavaScript("return document.querySelector('.reply .thought').getBoundingClientRect().height",
                                                  contentWorld: .page) as? Double ?? -1
        }
        // Open on its latest lines: 96 px of them under 8.
        let open = try await settled(height) { $0 == 104 }
        #expect(open == 104)
        // The answer starts: the thought folds, from no taller than it was, a frame at a time.
        reply.text = "Yes."
        let json = String(decoding: try WebPage.encoder.encode(TranscriptMessage(reply, live: live, asleep: false, name: { $0 })),
                          as: UTF8.self)
        let heights = try await webView.callAsyncJavaScript("""
            transcript.update(JSON.parse(json))
            const thought = document.querySelector('.reply .thought'), heights = []
            do {
              await new Promise(drawn => requestAnimationFrame(drawn))
              heights.push(thought.getBoundingClientRect().height)
            } while (heights.at(-1) > 0 && heights.length < 120)
            return heights
            """, arguments: ["json": json], contentWorld: .page) as? [Double] ?? []
        #expect(heights.last == 0 && heights.allSatisfy { $0 <= open }, "\(heights.map { Int($0.rounded()) })")
    }

    @Test func aConversationOpensAtItsLatestScreenAndDrawsTheRestAfter() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 30)
        func conversation() -> TranscriptPage {
            let messages = (0..<20).flatMap { index in
                [ChatMessage(role: .user, text: "Tell me more."), ChatMessage(role: .assistant, text: "**\(index)** \(paragraph)")]
            }
            return TranscriptPage(Conversation(messages: messages), askAgain: .init(enabled: false, help: ""), name: { $0 })
        }
        let (first, second) = (conversation(), conversation())
        // Shown by the page itself, so that what it draws at once is read before it draws more.
        func open(_ page: TranscriptPage) async throws -> [String] {
            let json = String(decoding: try WebPage.encoder.encode(page), as: UTF8.self)
            return try await webView.callAsyncJavaScript("""
                transcript.show(JSON.parse(json))
                return [...document.querySelectorAll('.reply .markdown')].map(body => body.textContent === '' ? '' : 'drawn')
                """, arguments: ["json": json], contentWorld: .page) as? [String] ?? []
        }
        // The latest replies at once, the earliest after.
        var drawn = try await open(first)
        #expect(drawn.first == "" && drawn.last == "drawn")
        drawn = try await settled({
            try await webView.callAsyncJavaScript(
                "return [...document.querySelectorAll('.reply .markdown')].map(body => body.textContent === '' ? '' : 'drawn')",
                contentWorld: .page) as? [String] ?? []
        }) { !$0.contains("") }
        #expect(drawn == Array(repeating: "drawn", count: 20))
        // Another conversation, then the first again, whose replies were drawn before: all at once.
        _ = try await open(second)
        #expect(try await open(first) == Array(repeating: "drawn", count: 20))
        _ = transcript
    }

    @Test func aReplyThatStreamsShowsWhatItHasAtOnceWhereItOpens() async throws {
        let (_, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 20).trimmingCharacters(in: .whitespaces)
        let reply = ChatMessage(role: .assistant, text: paragraph)
        let streaming = TranscriptPage(Conversation(messages: [ChatMessage(role: .user, text: "Go on."), reply]),
                                       askAgain: .init(enabled: false, help: ""), name: { $0 })
            .streaming(TranscriptMessage(reply, live: ChatStore.Live(asked: .now), asleep: false, name: { $0 }))
        let other = TranscriptPage(Conversation(messages: [ChatMessage(role: .user, text: "Hi.")]),
                                   askAgain: .init(enabled: false, help: ""), name: { $0 })
        /// What the page shows of the reply as it opens `page`, before any frame.
        func open(_ page: TranscriptPage) async throws -> String {
            let json = String(decoding: try WebPage.encoder.encode(page), as: UTF8.self)
            return try await webView.callAsyncJavaScript("""
                transcript.show(JSON.parse(json))
                return document.querySelector('.reply .markdown')?.textContent ?? ''
                """, arguments: ["json": json], contentWorld: .page) as? String ?? ""
        }
        // Opened while it streams, as once the page has loaded again, and again after another.
        #expect(try await open(streaming) == paragraph)
        _ = try await open(other)
        #expect(try await open(streaming) == paragraph)
    }

    @Test func aReplyThatStreamsIsCutOnlyWhereWhatComesBeforeDrawsAsInTheWhole() async throws {
        let (_, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        // Each text streamed a character at a time, and each step drawn as it streams and whole,
        // which once past `settled` are the same: a list goes on past a blank line, its next
        // marker, indented or not, coming a character at a time; a fence, of backticks or tildes,
        // closes only on its own character, as long or longer, with nothing after it, and less
        // than four columns in from its container; a backtick fence's info has no backtick; a
        // fence opens where its list item does, and a line indented four columns is code, marker
        // or not; lines end in any of the parser's line ends.
        let texts = [
            ["", "Intro.\n\n1. a\n\n2. b\n\n   More of b.\n\n- c\n\n- d\n\nDone."],
            ["", "Intro.\n\n~~~\nt\n\nu\n~~~\n\n~~~\n```\nx\n\ny\n~~~\n\n````\n```\nz\n\nw\n````\n\n```\n```js\nu\n\nv\n```\n\nDone."],
            ["```swift\nlet a = 1\n```", "\n```a`b\n\nDone."],
            ["", "Intro.\n\n```\n# Tool\n\n1. Install:\n\n    ```bash\n    pip\n    ```\n\n## Usage\n\nRun.\n```\n\nDone."],
            ["", "```\ndef a():\n    ```\n\ndef b():\n    pass\n```\n\nDone."],
            ["", "  ```\nx\n     ```\n\ny\n  ```\n\nDone."],
            ["", "- ```\n  code\n  ```\n\nDone.\n\n```\nx\n\ny\n```\n\nEnd."],
            ["", "    ```\n\nDone.\n\n```\nx\n\ny\n```\n\nEnd."],
            ["", "Intro.\r\n\r\n```\r\nx\r\n\r\ny\r\n```\r\n\r\nDone."],
            ["", "a\r\r\rb\n```\n\nd\n```\n\nDone."],
            ["", "1. a\n\n 2. b\n\nDone."],
        ]
        let differs = try await webView.callAsyncJavaScript("""
            const differs = []
            for (const [settled, rest] of JSON.parse(json)) {
              const text = settled + rest
              const live = document.createElement('div'), whole = document.createElement('div')
              for (let end = 1; end <= text.length; end++) {
                markdown.draw(live, text.slice(0, end), true)
                markdown.draw(whole, text.slice(0, end))
                if (end >= settled.length && live.innerHTML !== whole.innerHTML) {
                  differs.push(text.slice(0, end))
                  break
                }
              }
            }
            return differs
            """, arguments: ["json": String(decoding: try JSONEncoder().encode(texts), as: UTF8.self)], contentWorld: .page) as? [String]
        #expect(differs == [])
    }

    /// Code that streams shows as it is until its fence closes, whatever looks like closing it.
    @Test func aFenceStaysOpenUntilTheParserClosesIt() async throws {
        let (_, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let cases: [(String, Int)] = [
            ("```swift\nlet a = 1\n", 0),
            ("```swift\n", 0),
            ("```swift\nlet a = 1", 0),
            ("```swift\nfunc f() {\n    ", 0),
            ("```swift\nfunc f() {\n\t", 0),
            ("```swift\rlet a = 1\r", 0),
            ("```swift\rfunc f() {\r    ", 0),
            ("- ```swift\n  let a = 1\n  ", 0),
            ("- ```swift\n  let a = 1", 0),
            ("```swift\nlet a = 1\n    ```\nlet b = 2\n", 0),
            ("- ```swift\n  let a = 1\n", 0),
            ("> ```swift\n> let a = 1\n", 0),
            ("```swift\nlet a = 1\n```", 1),
            ("```swift\nlet a = 1\n```\n", 1),
            ("```swift\rlet a = 1\r```\r", 1),
            ("```swift\nlet a = 1\n  ```\n", 1),
            ("- ```swift\n  let a = 1\nDone.", 1),
            ("> ```swift\n> let a = 1\nDone.", 1),
        ]
        for (text, keywords) in cases {
            let drawn = try await webView.callAsyncJavaScript("""
                const live = document.createElement('div')
                markdown.draw(live, text, true)
                return live.querySelectorAll('.hljs-keyword').length
                """, arguments: ["text": text], contentWorld: .page) as? Int
            #expect(drawn == keywords, "\(text)")
        }
    }

    /// Without Markdown a reply shows as it came, as it streams too.
    @Test func withoutMarkdownAReplyStreamsAsItCame() async throws {
        let transcript = Transcript(runtime: Runtime(root: URL(filePath: "/nonexistent")), attachments: Showcase.attachments)
        transcript.page.load()
        let webView = try #require(transcript.page.webView)
        try await waitFor(transcript.page.loaded)
        let drawn = try await webView.callAsyncJavaScript("""
            const live = document.createElement('div')
            markdown.draw(live, 'a **b**\\n\\nc', true)
            return live.innerHTML
            """, contentWorld: .page) as? String
        #expect(drawn == "<p>a **b**\n\nc</p>")
    }

    /// Text let out over the frames never ends in half of a character of two UTF-16 units. Two
    /// texts of a length, the characters of each a unit further on, which each frame lets out as
    /// much of: wherever it cuts them, one has half a character there.
    @Test func textLetOutNeverShowsHalfACharacter() async throws {
        let (_, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let cut = try await webView.callAsyncJavaScript("""
            const faces = '\\u{1F600}'.repeat(300)
            const whole = {first: faces + '.', last: '.' + faces}
            const shown = []
            const paced = page.paced(texts => shown.push(texts))
            paced.to({first: '', last: ''}, true)
            paced.to(whole, true)
            const done = () => shown.at(-1).first === whole.first && shown.at(-1).last === whole.last
            const deadline = performance.now() + 10000
            while (!done() && performance.now() < deadline) await new Promise(requestAnimationFrame)
            return done() ? shown.flatMap(Object.values).filter(text => /[\\uD800-\\uDBFF]$/.test(text)).length : -1
            """, contentWorld: .page) as? Int
        #expect(cut == 0)
    }

    /// With Reduce Motion a reply shows all that came at once, and the page follows it at once.
    @Test func withReduceMotionAReplyShowsAtOnceAndThePageFollowsAtOnce() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        // The system's setting as the page reads it, which it reads each time.
        _ = try await webView.callAsyncJavaScript("""
            Object.defineProperty(MediaQueryList.prototype, 'matches', {configurable: true, get() { return this.media.includes('reduced-motion') }})
            """, contentWorld: .page)
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 12).trimmingCharacters(in: .whitespaces)
        let earlier = (0..<8).flatMap { _ in [ChatMessage(role: .user, text: "Tell me more."), ChatMessage(role: .assistant, text: paragraph)] }
        var reply = ChatMessage(role: .assistant, text: paragraph)
        let page = TranscriptPage(Conversation(messages: earlier + [ChatMessage(role: .user, text: "And?"), reply]),
                                  askAgain: .init(enabled: false, help: ""), name: { $0 })
        let live = ChatStore.Live(asked: .now)
        transcript.show(page, reply: nil)
        // Every reply drawn, which the page does after it shows, and it at the bottom.
        let ready = try await settled({
            try await webView.callAsyncJavaScript("""
                await new Promise(drawn => requestAnimationFrame(() => requestAnimationFrame(drawn)))
                return [...document.querySelectorAll('.reply .markdown')].every(body => body.textContent !== '')
                    && document.scrollingElement.scrollHeight - innerHeight - scrollY === 0
                """, contentWorld: .page) as? Bool ?? false
        }) { $0 }
        #expect(ready)
        for _ in 0..<3 {
            reply.text += " " + paragraph
            let message = TranscriptMessage(reply, live: live, asleep: false, name: { $0 })
            let shown = try await webView.callAsyncJavaScript("""
                transcript.update(JSON.parse(json))
                return document.querySelector('#messages > article:last-child .markdown').textContent
                """, arguments: ["json": String(decoding: try WebPage.encoder.encode(message), as: UTF8.self)], contentWorld: .page) as? String
            #expect(shown == reply.text)
            let gap = try await webView.callAsyncJavaScript("""
                await new Promise(drawn => requestAnimationFrame(() => requestAnimationFrame(drawn)))
                return document.scrollingElement.scrollHeight - innerHeight - scrollY
                """, contentWorld: .page) as? Double
            #expect(gap == 0)
        }
    }

    @Test func thePageFollowsAReplyThatStreamsUntilTheReaderScrollsUp() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 12)
        let earlier = (0..<8).flatMap { _ in [ChatMessage(role: .user, text: "Tell me more."), ChatMessage(role: .assistant, text: paragraph)] }
        var reply = ChatMessage(role: .assistant, text: "")
        let page = TranscriptPage(Conversation(messages: earlier + [ChatMessage(role: .user, text: "And?"), reply]),
                                  askAgain: .init(enabled: false, help: ""), name: { $0 })
        let live = ChatStore.Live(asked: .now)
        transcript.show(page, reply: nil)
        func position() async throws -> (gap: Double, latest: Bool) {
            let values = try await webView.callAsyncJavaScript(
                "return [document.scrollingElement.scrollHeight - innerHeight - scrollY, document.getElementById('latest').classList.contains('shown') ? 1 : 0]",
                contentWorld: .page) as? [Double] ?? []
            return (values[0], values[1] == 1)
        }
        func grow() {
            reply.text += paragraph + "\n\n"
            transcript.show(page, reply: TranscriptMessage(reply, live: live, asleep: false, name: { $0 }))
        }
        // It eases to the bottom as the reply grows, and rests there.
        for _ in 0..<4 {
            grow()
            try await Task.sleep(for: .milliseconds(60))
        }
        var resting = try await settled(position) { $0.gap == 0 }
        #expect(resting.gap == 0 && !resting.latest)
        // Scrolled up, at the top, it stays where the reader left it, and offers the way back.
        _ = try await webView.callAsyncJavaScript("scrollTo(0, 0)", contentWorld: .page)
        try await Task.sleep(for: .milliseconds(100))
        grow()
        try await Task.sleep(for: .milliseconds(400))
        let top = try await webView.callAsyncJavaScript("return scrollY", contentWorld: .page) as? Double
        resting = try await position()
        #expect(top == 0 && resting.latest)
    }

    @Test func messagesOffTheScreenAreLaidOutAsTheyComeAndKeepWhatTheReaderReadsInPlace() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 30)
        let messages = (0..<20).flatMap { index in
            [ChatMessage(role: .user, text: "Question \(index)"), ChatMessage(role: .assistant, text: "Answer \(index). \(paragraph)")]
        }
        transcript.show(TranscriptPage(Conversation(messages: messages), askAgain: .init(enabled: false, help: ""), name: { $0 }), reply: nil)
        func run(_ script: String) async throws -> String {
            try await webView.callAsyncJavaScript("""
                await new Promise(drawn => requestAnimationFrame(() => requestAnimationFrame(drawn)))
                \(script)
                """, contentWorld: .page) as? String ?? ""
        }
        // Scrolled up to where the reader has not been, a message drawn above the screen as it
        // comes near, which takes its own height in place of the room kept for it, leaves the one
        // the reader reads where it was.
        _ = try await run("document.querySelectorAll('article')[12].style.setProperty('--estimate', '3000px'); return ''")
        let read = try await run("""
            const articles = document.querySelectorAll('article')
            dispatchEvent(new WheelEvent('wheel'))
            scrollTo(0, articles[13].offsetTop - 100)
            window.read = articles[13]
            return String(read.getBoundingClientRect().top) + ' ' + articles[12].offsetHeight
            """)
        let after = try await settled({ try await run("return String(read.getBoundingClientRect().top) + ' ' + read.previousElementSibling.offsetHeight") }) {
            $0 == "100 39"
        }
        #expect(read == "100 3000" && after == "100 39", "where it is, and how tall the message above it")
        // All of them in a copy of the page.
        let copied = try await settled({ try await run("""
            getSelection().selectAllChildren(document.getElementById('messages'))
            return getComputedStyle(document.querySelector('article')).contentVisibility + ' ' + getSelection().toString()
            """) }) { $0.contains("Answer 19.") }
        #expect(copied.hasPrefix("auto ") && copied.contains("Question 0") && copied.contains("Answer 0.") && copied.contains("Answer 19."))
    }

    /// At the column's width, which the window sets: the chat at the smallest window with the
    /// sidebar open, 520 px across, and again once the window is wider.
    @Test func messagesNotDrawnYetKeepAboutTheRoomTheyTake() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        // Once every reply is drawn, at `width`, each message laid out, against the room kept for it.
        func measure(at width: Double) async throws -> (count: Int, total: Double, each: Double) {
            let measured = try await settled({
                try await webView.callAsyncJavaScript("""
                    if (innerWidth !== \(width)) return []
                    await new Promise(drawn => requestAnimationFrame(() => requestAnimationFrame(drawn)))
                    const articles = [...document.querySelectorAll('#messages > article')]
                    if (articles.some(article => article.querySelector('.markdown')?.textContent === '')) return []
                    for (const article of articles) article.style.contentVisibility = 'visible'
                    const measured = articles.map(article => [parseFloat(article.style.getPropertyValue('--estimate')),
                                                              article.getBoundingClientRect().height])
                    for (const article of articles) article.style.contentVisibility = ''
                    return measured.flat()
                    """, contentWorld: .page) as? [Double] ?? []
            }) { !$0.isEmpty }
            let kept = stride(from: 0, to: measured.count, by: 2).map { measured[$0] }
            let heights = stride(from: 1, to: measured.count, by: 2).map { measured[$0] }
            return (heights.count, kept.reduce(0, +) / heights.reduce(0, +) - 1,
                    zip(kept, heights).map { abs($0 - $1) / $1 }.reduce(0, +) / Double(heights.count))
        }
        webView.frame.size.width = 520
        transcript.show(TranscriptPage(Conversation(messages: Self.longChat), askAgain: .init(enabled: false, help: ""), name: { $0 }),
                        reply: nil)
        for width in [520.0, 800] {
            webView.frame.size.width = width
            let (count, total, each) = try await measure(at: width)
            #expect(count == Self.longChat.count && abs(total) < 0.1 && each < 0.2, "at \(width): total \(total), each \(each)")
        }
    }

    /// A long chat, as people have them: short questions, and replies of a few paragraphs, some
    /// with a heading, a list or code, some after thinking.
    private static let longChat: [ChatMessage] = {
        var seed: UInt64 = 7
        func next(_ count: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((seed >> 33) % UInt64(count))
        }
        let words = """
            the model reads each token of the prompt once and keeps what it learned in its cache so that a reply \
            can go on without reading it again which makes long chats fast on a Mac with enough memory for them
            """.split(separator: " ")
        func sentence() -> String {
            let text = (0..<(6 + next(18))).map { _ in String(words[next(words.count)]) }.joined(separator: " ")
            return text.prefix(1).uppercased() + text.dropFirst() + "."
        }
        func paragraph() -> String { (0..<(1 + next(5))).map { _ in sentence() }.joined(separator: " ") }
        return (0..<150).flatMap { _ in
            var blocks = next(5) == 0 ? ["## " + sentence().dropLast()] : []
            for _ in 0..<(1 + next(5)) {
                switch next(8) {
                case 0: blocks.append((0..<(2 + next(4))).map { _ in "- " + sentence() }.joined(separator: "\n"))
                case 1: blocks.append((1...(2 + next(4))).map { "\($0). " + sentence() }.joined(separator: "\n"))
                case 2: blocks.append("```swift\n" + (0..<(3 + next(10))).map { "let value\($0) = compute(\($0))" }.joined(separator: "\n") + "\n```")
                default: blocks.append(paragraph())
                }
            }
            let question = (0..<(1 + next(2))).map { _ in sentence() }.joined(separator: "\n")
            return [ChatMessage(role: .user, text: question),
                    ChatMessage(role: .assistant, text: blocks.joined(separator: "\n\n"), reasoning: next(3) == 0 ? paragraph() : "",
                                model: "a/Big-GGUF:Q4", tokensPerSecond: 40, thinkingTime: 2)]
        }
    }()

    @Test func theChatShowsAgainWithoutLoadingOrDrawingItsPageAgain() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-transcript-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Conversation(messages: [ChatMessage(role: .user, text: "Hi"), ChatMessage(role: .assistant, text: "Hello.")])
        let other = Conversation(messages: [ChatMessage(role: .user, text: "Bye"), ChatMessage(role: .assistant, text: "Goodbye.")])
        for conversation in [conversation, other] {
            try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        }
        let state = AppState.forTests(in: folder)
        state.chats.selection = conversation.id
        state.section = .chat
        /// The window's content as MainView switches it, which builds the Chat anew each time.
        struct Sections: View {
            @Environment(AppState.self) private var state
            var body: some View {
                if state.section == .chat { ChatScreen() } else { Text("Server") }
            }
        }
        _ = NSApplication.shared
        let window = NSWindow(contentViewController: NSHostingController(rootView: Sections().environment(state)))
        window.setContentSize(CGSize(width: 900, height: 600))
        window.setFrameOrigin(NSPoint(x: -30000, y: -30000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        try await waitFor(state.transcript.page.loaded)
        let webView = try #require(state.transcript.page.webView)
        func run(_ script: String) async throws -> String {
            try await webView.callAsyncJavaScript(script, contentWorld: .page) as? String ?? ""
        }
        try await waitFor(webView.window === window)
        // Marks on the page and on the reply drawn, which a load or a redraw would take away.
        _ = try await run("window.mark = 'kept'; document.querySelector('.reply').dataset.mark = 'kept'; return ''")
        state.section = .server
        try await waitFor(webView.window == nil)
        state.section = .chat
        try await waitFor(webView.window === window)
        #expect(state.transcript.page.webView === webView)
        #expect(try await run("return window.mark + ' ' + document.querySelector('.reply').dataset.mark") == "kept kept")
        // Another conversation is sent to the page, which stays.
        state.chats.selection = other.id
        let shown = try await settled({ try await run("return window.mark + ' ' + document.querySelector('.reply .markdown').textContent") }) {
            $0 == "kept Goodbye."
        }
        #expect(shown == "kept Goodbye.")
    }

    /// The transcript's page, loaded in a window off the screen, drawn as on screen, as WebKit
    /// holds back the animation frames of a page it can't see.
    private func shownPage() async throws -> (Transcript, WKWebView, NSWindow) {
        try Showcase.requireScripts()
        let transcript = Transcript(runtime: Showcase.runtime, attachments: Showcase.attachments)
        transcript.page.load()
        let webView = try #require(transcript.page.webView)
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: -30000, y: -30000, width: 800, height: 500), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.contentView?.addSubview(webView)
        webView.frame = window.contentView?.bounds ?? .zero
        Snapshot.drawsOffScreen(webView)
        window.orderFrontRegardless()
        try await waitFor(transcript.page.loaded)
        return (transcript, webView, window)
    }

    private static func textViews(in view: NSView) -> [NSTextView] {
        view.subviews.flatMap { ($0 as? NSTextView).map { [$0] } ?? textViews(in: $0) }
    }

    /// `value` read until `done` holds of it, or until `seconds` pass: what the page shows once it
    /// has caught up, however long that takes on a busy Mac.
    private func settled<Value>(_ value: () async throws -> Value, seconds: Double = 10,
                                until done: (Value) -> Bool) async throws -> Value {
        let deadline = ContinuousClock.now + .seconds(seconds)
        var read = try await value()
        while !done(read), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
            read = try await value()
        }
        return read
    }
}

/// The page of the Chat's transcript, apart from what it shows. A suite of its own, as its test
/// waits a minute, while the others run.
@MainActor
struct WebPageTests {
    @Test func aPageWhoseProcessKeepsEndingLoadsAgainThreeTimesAMinute() async throws {
        let transcript = Transcript(runtime: Showcase.runtime, attachments: Showcase.attachments)
        let page = transcript.page
        page.load()
        let webView = try #require(page.webView)
        try await waitFor(page.loaded, "the load")
        let first = ContinuousClock.now
        for _ in 0..<3 {
            page.webViewWebContentProcessDidTerminate(webView)
            #expect(!page.loaded)
            try await waitFor(page.loaded, "the load")
        }
        // A fourth time within the minute, once the minute since the first has passed.
        page.webViewWebContentProcessDidTerminate(webView)
        try await Task.sleep(for: .seconds(2))
        #expect(!page.loaded)
        try await waitFor(page.loaded, "the load", seconds: 70)
        #expect(ContinuousClock.now - first >= .seconds(60))
    }
}

/// A conversation with every kind of content the Chat shows, drawn to `transcript-*.png` in
/// `SPLASH_APP_SNAPSHOTS`, light and dark, for people to review the transcript's look: the whole
/// window at the conversation's start, middle and end, then the reply line's states while a reply
/// streams.
@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct TranscriptPictures {
    @Test func showcase() async throws {
        try Showcase.requireScripts()
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-transcript-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Showcase.conversation
        try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        let state = AppState.forTests(in: folder, runtime: Showcase.runtime)
        state.section = .chat
        state.chats.selection = conversation.id
        state.transcript.page.load()
        Snapshot.drawsOffScreen(try #require(state.transcript.page.webView))
        await Snapshot.save("transcript", ChatScreen().environment(state))
        for (name, top) in [("transcript-top", 0), ("transcript-middle", 700)] {
            await Snapshot.save(name, ChatScreen().environment(state)) { _ in
                _ = try? await state.transcript.page.webView?.evaluateJavaScript("scrollTo(0, \(top))")
                try? await Task.sleep(for: .seconds(0.5))
            }
        }
        // The first reply's thinking, opened, set apart by its rule.
        await Snapshot.save("transcript-thought", ChatScreen().environment(state)) { _ in
            _ = try? await state.transcript.page.webView?.evaluateJavaScript("document.querySelector('.line:not(.open) .toggle')?.click(); scrollTo(0, 0)")
            try? await Task.sleep(for: .seconds(0.5))
        }
    }

    @Test func whileAReplyStreams() async throws {
        try Showcase.requireScripts()
        let transcript = Transcript(runtime: Showcase.runtime, attachments: Showcase.attachments)
        // Loaded first, as the app loads it with its window, so that each picture shows its state,
        // and drawn as on screen from the first, whose Reading shows half a second in.
        transcript.page.load()
        Snapshot.drawsOffScreen(try #require(transcript.page.webView))
        while !transcript.page.loaded { try await Task.sleep(for: .milliseconds(20)) }
        let question = ChatMessage(role: .user, text: "Why is the sea salty?")
        let thought = (1...14).map { "Line \($0) of the model’s thinking, which goes on for a while to wrap." }.joined(separator: "\n")
        let states: [(String, ChatMessage, () -> ChatStore.Live, Bool)] = [
            ("reading", ChatMessage(role: .assistant, text: ""), { .init(asked: .now - 3, read: 0.45) }, false),
            ("loading", ChatMessage(role: .assistant, text: ""), { .init(asked: .now - 3.2) }, true),
            ("thinking", ChatMessage(role: .assistant, text: "", reasoning: thought),
             { .init(asked: .now - 9, thinking: .now - 5.4) }, false),
            ("streaming", ChatMessage(role: .assistant, text: "Rivers carry salts from rocks into the sea.\n\n```python\nsalt = 35  # grams per litre\n",
                                      reasoning: thought, model: "a/Big-GGUF:Q4", thinkingTime: 5.4, reasoningTokens: 320),
             { .init(asked: .now - 9, thinking: .now - 7) }, false),
        ]
        for (name, reply, live, asleep) in states {
            let conversation = Conversation(messages: [question, reply])
            let page = TranscriptPage(conversation, askAgain: .init(enabled: false, help: "Ask Big Again"), name: { _ in "Big" })
            let streaming = TranscriptMessage(reply, live: live(), asleep: asleep, name: { _ in "Big" })
            await Snapshot.save("transcript-\(name)", size: CGSize(width: 860, height: 520),
                                TranscriptView(transcript: transcript, page: page, reply: streaming).background(.background))
        }
    }
}
