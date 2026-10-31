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
            ChatMessage(role: .user, text: "What is this?", images: [Showcase.picture]),
            ChatMessage(role: .assistant, text: "A cat.", reasoning: "Whiskers.", model: "a/Big-GGUF:Q4", tokensPerSecond: 38.4,
                        thinkingTime: 3.2, reasoningTokens: 1_200, stopped: true),
            ChatMessage(role: .user, text: "Sure?"),
            ChatMessage(role: .assistant, text: ""),
        ])
        let live = ChatStore.Live(conversation: conversation.id, asked: asked, read: 0.45)
        let name = { (model: String) in "Name of \(model)" }
        let page = TranscriptPage(conversation, askAgain: .init(enabled: false, help: "Ask Big Again"), name: name)
            .streaming(TranscriptMessage(conversation.messages[3], live: live, asleep: false, name: name))
        let json = try #require(JSONSerialization.jsonObject(with: WebPage.encoder.encode(page)) as? [String: Any])
        #expect(json["id"] as? String == conversation.id.uuidString)
        #expect(json["askAgain"] as? [String: AnyHashable] == ["enabled": false, "help": "Ask Big Again"])
        let messages = try #require(json["messages"] as? [[String: Any]])
        #expect(messages.map { $0["role"] as? String } == ["user", "assistant", "user", "assistant"])
        // Every key there, null for none.
        #expect(messages.allSatisfy { Set($0.keys) == ["id", "role", "text", "reasoning", "images", "error", "line", "live", "details"] })
        // Images as the addresses the page loads them from, which the app serves.
        #expect(messages[0]["images"] as? [String] == ["chat-image://\(conversation.messages[0].id.uuidString)/0"])
        #expect(messages[0]["line"] is NSNull && messages[0]["error"] is NSNull && messages[0]["details"] as? String == "")
        #expect(messages[1]["line"] as? [String: AnyHashable] == ["verb": "Thinking", "facts": "1.2K tokens · 3.2s", "since": NSNull()])
        #expect(messages[1]["details"] as? String == "Name of a/Big-GGUF:Q4 · 38 tok/s · Stopped")
        #expect(messages[1]["live"] as? Bool == false)
        // The reply that streams, the last.
        #expect(messages[3]["live"] as? Bool == true)
        #expect(messages[3]["line"] as? [String: AnyHashable] == ["verb": "Reading", "facts": "45%", "since": NSNull()])
        // Dates in milliseconds, as JavaScript counts them.
        let waiting = TranscriptMessage(conversation.messages[3], live: ChatStore.Live(conversation: conversation.id, asked: asked),
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
        let id = UUID()
        // Reading: how much of the prompt, or the seconds since it was asked until the server says.
        #expect(line(waiting, .init(conversation: id, asked: asked)) == ReplyLine(verb: "Reading", facts: nil, since: asked))
        #expect(line(waiting, .init(conversation: id, asked: asked, read: 0.456))
            == ReplyLine(verb: "Reading", facts: "45%", since: nil))
        // A model the idle release put to sleep loads again first.
        #expect(line(waiting, .init(conversation: id, asked: asked, read: 0.5), asleep: true)
            == ReplyLine(verb: "Loading into memory", facts: nil, since: asked))
        // Thinking, the seconds since it began.
        let thinking = ChatMessage(role: .assistant, text: "", reasoning: "Hm.")
        #expect(line(thinking, .init(conversation: id, asked: asked, thinking: started))
            == ReplyLine(verb: "Thinking", facts: nil, since: started))
        // Once it answers, what the thinking took, though the reply still streams.
        var answered = thinking
        answered.text = "Yes."
        answered.thinkingTime = 9.96
        #expect(line(answered, .init(conversation: id, asked: asked, thinking: started))
            == ReplyLine(verb: "Thinking", facts: "10s", since: nil))
        answered.thinkingTime = nil
        #expect(line(answered, nil) == ReplyLine(verb: "Thinking", facts: nil, since: nil))
        // No line for a reply that answers without thinking, or once it has ended.
        #expect(line(ChatMessage(role: .assistant, text: "Yes."), .init(conversation: id, asked: asked)) == nil)
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
        let transcript = Transcript(runtime: Showcase.runtime)
        var askedAgain = 0
        transcript.askAgain = { askedAgain += 1 }
        transcript.page.load()
        let webView = try #require(transcript.page.webView)
        try await waitFor(transcript.page.loaded)
        let conversation = Conversation(messages: [
            ChatMessage(role: .user, text: "<b>Hi</b>", images: [Showcase.picture]),
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
        transcript.show(TranscriptPage(conversation, askAgain: askAgain, name: { $0 }))
        func run(_ script: String) async throws -> String {
            try await webView.callAsyncJavaScript(script, contentWorld: .page) as? String ?? ""
        }
        // Raw HTML stays text.
        #expect(try await run("return document.querySelector('.bubble').textContent") == "<b>Hi</b>")
        // The user's image, which the app serves.
        let loaded = try await settled({ try await run("const image = document.querySelector('.images img'); return image.complete ? String(image.naturalWidth) : ''") }) {
            !$0.isEmpty
        }
        #expect(loaded == "240")
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
        let state = AppState(runtime: Showcase.runtime, chats: ChatStore(folder: folder), keychain: Keychain(service: "Splash Tests"))
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
        let transcript = Transcript(runtime: Showcase.runtime)
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
            transcript.show(TranscriptPage(Conversation(id: id, messages: messages), askAgain: .init(enabled: false, help: ""), name: { $0 }))
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
        // Scrolled up, it stays where the reader left it, and offers the way back.
        _ = try await webView.callAsyncJavaScript("scrollTo(0, 100)", contentWorld: .page)
        _ = try await scrolled()
        messages[messages.count - 1].text += "\n\n" + paragraph
        show(messages)
        position = try await scrolled()
        #expect(position.top == 100 && position.bottom > 100 && position.latest)
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
        let live = ChatStore.Live(conversation: page.id, asked: .now)
        transcript.show(page)
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
        // still open shows its code as it is.
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 20).trimmingCharacters(in: .whitespaces)
        reply.text = "\(paragraph)\n\n```swift\nlet a = 1\n"
        #expect(try await send(live: true) < paragraph.count)
        // The paragraph, then the code under its language.
        try await shows("\(paragraph)swiftlet a = 1")
        #expect(try await run("return String(document.querySelectorAll('pre .hljs-keyword').length)") == "0")
        // The paragraph, which is done, and the code's button, which a redraw would replace.
        _ = try await run("window.kept = [document.querySelector('.reply .markdown > p'), document.querySelector('.reply .copy-code')]; return ''")
        // The fence closes: its code is highlighted, once, and the rest stays as drawn.
        reply.text += "let b = 2\n```\n\nDone."
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
        let live = ChatStore.Live(conversation: page.id, asked: .now, thinking: .now)
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

    @Test func thePageFollowsAReplyThatStreamsUntilTheReaderScrollsUp() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 12)
        let earlier = (0..<8).flatMap { _ in [ChatMessage(role: .user, text: "Tell me more."), ChatMessage(role: .assistant, text: paragraph)] }
        var reply = ChatMessage(role: .assistant, text: "")
        let page = TranscriptPage(Conversation(messages: earlier + [ChatMessage(role: .user, text: "And?"), reply]),
                                  askAgain: .init(enabled: false, help: ""), name: { $0 })
        let live = ChatStore.Live(conversation: page.id, asked: .now)
        transcript.show(page)
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
        // Scrolled up, it stays where the reader left it, and offers the way back.
        _ = try await webView.callAsyncJavaScript("scrollTo(0, 100)", contentWorld: .page)
        try await Task.sleep(for: .milliseconds(100))
        grow()
        try await Task.sleep(for: .milliseconds(400))
        let top = try await webView.callAsyncJavaScript("return scrollY", contentWorld: .page) as? Double
        resting = try await position()
        #expect(top == 100 && resting.latest)
    }

    @Test func messagesOffTheScreenAreLaidOutAsTheyComeAndKeepWhatTheReaderReadsInPlace() async throws {
        let (transcript, webView, window) = try await shownPage()
        defer { window.orderOut(nil) }
        let paragraph = String(repeating: "A sentence that goes on for a while. ", count: 30)
        let messages = (0..<20).flatMap { index in
            [ChatMessage(role: .user, text: "Question \(index)"), ChatMessage(role: .assistant, text: "Answer \(index). \(paragraph)")]
        }
        transcript.show(TranscriptPage(Conversation(messages: messages), askAgain: .init(enabled: false, help: ""), name: { $0 }))
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

    @Test func theChatShowsAgainWithoutLoadingOrDrawingItsPageAgain() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-transcript-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Conversation(messages: [ChatMessage(role: .user, text: "Hi"), ChatMessage(role: .assistant, text: "Hello.")])
        let other = Conversation(messages: [ChatMessage(role: .user, text: "Bye"), ChatMessage(role: .assistant, text: "Goodbye.")])
        for conversation in [conversation, other] {
            try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        }
        let state = AppState(runtime: Showcase.runtime, chats: ChatStore(folder: folder), keychain: Keychain(service: "Splash Tests"))
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
        let transcript = Transcript(runtime: Showcase.runtime)
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

    private func waitFor(_ condition: @autoclosure () -> Bool, seconds: Double = 10) async throws {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !condition() {
            try #require(ContinuousClock.now < deadline, "timed out")
            try await Task.sleep(for: .milliseconds(20))
        }
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
        let state = AppState(runtime: Showcase.runtime, chats: ChatStore(folder: folder), keychain: Keychain(service: "Splash Tests"))
        state.section = .chat
        state.chats.selection = conversation.id
        state.transcript.page.load()
        Snapshot.drawsOffScreen(try #require(state.transcript.page.webView))
        await Snapshot.save("transcript", ChatScreen().environment(state))
        for (name, top) in [("transcript-top", 0), ("transcript-middle", 700)] {
            await Snapshot.save(name, ChatScreen().environment(state)) {
                _ = try? await state.transcript.page.webView?.evaluateJavaScript("scrollTo(0, \(top))")
                try? await Task.sleep(for: .seconds(0.5))
            }
        }
        // The first reply's thinking, opened, set apart by its rule.
        await Snapshot.save("transcript-thought", ChatScreen().environment(state)) {
            _ = try? await state.transcript.page.webView?.evaluateJavaScript("document.querySelector('.line:not(.open) .toggle')?.click(); scrollTo(0, 0)")
            try? await Task.sleep(for: .seconds(0.5))
        }
    }

    @Test func whileAReplyStreams() async throws {
        try Showcase.requireScripts()
        let transcript = Transcript(runtime: Showcase.runtime)
        // Loaded first, as the app loads it with its window, so that each picture shows its state,
        // and drawn as on screen from the first, whose Reading shows half a second in.
        transcript.page.load()
        Snapshot.drawsOffScreen(try #require(transcript.page.webView))
        while !transcript.page.loaded { try await Task.sleep(for: .milliseconds(20)) }
        let question = ChatMessage(role: .user, text: "Why is the sea salty?")
        let thought = (1...14).map { "Line \($0) of the model’s thinking, which goes on for a while to wrap." }.joined(separator: "\n")
        let states: [(String, ChatMessage, (UUID) -> ChatStore.Live, Bool)] = [
            ("reading", ChatMessage(role: .assistant, text: ""), { .init(conversation: $0, asked: .now - 3, read: 0.45) }, false),
            ("loading", ChatMessage(role: .assistant, text: ""), { .init(conversation: $0, asked: .now - 3.2) }, true),
            ("thinking", ChatMessage(role: .assistant, text: "", reasoning: thought),
             { .init(conversation: $0, asked: .now - 9, thinking: .now - 5.4) }, false),
            ("streaming", ChatMessage(role: .assistant, text: "Rivers carry salts from rocks into the sea.\n\n```python\nsalt = 35  # grams per litre\n",
                                      reasoning: thought, model: "a/Big-GGUF:Q4", thinkingTime: 5.4, reasoningTokens: 320),
             { .init(conversation: $0, asked: .now - 9, thinking: .now - 7) }, false),
        ]
        for (name, reply, live, asleep) in states {
            let conversation = Conversation(messages: [question, reply])
            let page = TranscriptPage(conversation, askAgain: .init(enabled: false, help: "Ask Big Again"), name: { _ in "Big" })
            let streaming = TranscriptMessage(reply, live: live(conversation.id), asleep: asleep, name: { _ in "Big" })
            await Snapshot.save("transcript-\(name)", size: CGSize(width: 860, height: 520),
                                TranscriptView(transcript: transcript, page: page, reply: streaming).background(.background))
        }
    }
}

enum Showcase {
    /// The source checkout these tests run from, whose runtime files the page's scripts come from.
    static let runtime = Runtime(root: URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../.."))

    /// Fails a test that draws Markdown without the checkout's markdown-it and highlight.js, where
    /// the page would show the replies as plain text.
    static func requireScripts() throws {
        for script in runtime.markdownScripts {
            try #require(FileManager.default.fileExists(atPath: script.path(percentEncoded: false)), "\(script.path(percentEncoded: false)) is missing")
        }
    }

    static let conversation = Conversation(title: "Getting started", messages: [
        ChatMessage(role: .user, text: "How do I get started with Splash? Show me everything a reply can hold."),
        ChatMessage(role: .assistant, text: reply,
                    reasoning: "The user wants an overview.\nI should cover installing, picking a model and chatting.\nA table compares the sizes.",
                    model: "mlx-community/Qwen3-30B-A3B-4bit", tokensPerSecond: 38.4, thinkingTime: 3.2, reasoningTokens: 120),
        ChatMessage(role: .user, text: "What’s in this picture?", images: [picture]),
        ChatMessage(role: .assistant, text: "A gradient from teal to blue, square, with nothing else in it.",
                    model: "mlx-community/Qwen3-30B-A3B-4bit", tokensPerSecond: 41,
                    error: "The reply reached the end of the model’s context."),
        ChatMessage(role: .user, text: "Thanks!\nOne more line, to see a message of two."),
        ChatMessage(role: .assistant, text: "You’re welcome. Ask again whenever you like.",
                    model: "mlx-community/Qwen3-30B-A3B-4bit", tokensPerSecond: 39.6, stopped: true),
    ])

    /// A reply with every kind of Markdown a reply holds.
    static let reply = """
        # Getting started with Splash

        Splash runs models **privately** on this Mac. Replies can use *emphasis*, `inline code`, \
        ~~strikethrough~~, and [links](https://example.com); an image shows as a link: ![diagram](https://example.com/d.png).

        ## Install

        1. Download the app and open it.
        2. Pick a model:
           - **Small** models start fast
           - **Large** models answer better
             - but need more memory
        3. Start chatting.

        ```bash
        curl -fsSL https://example.com/install.sh | sh
        splash serve --model mlx-community/Qwen3-30B-A3B-4bit --port 8000 --max-context 32K --kv-format bf16
        ```

        ### Comparison

        | Model | Memory | Speed | Notes |
        |:------|-------:|------:|-------|
        | Small | 16 GB | 95 tok/s | Good for quick questions |
        | Large | 64 GB | 38 tok/s | Better at code and long documents<br>Slower to load |
        | Largest | 128 GB | 21 tok/s | The best answers of the three, for the longest and hardest work, if the Mac has the memory for it |

        > Models never leave this Mac. Nothing you type is sent anywhere.
        >
        > — The Splash team

        ---

        ```
        plain code without a language
        ```

        - Done
        """

    /// A picture the user attached: a square gradient, as PNG.
    static let picture: Data = {
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 240, pixelsHigh: 240, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        NSGradient(starting: .systemTeal, ending: .systemBlue)!.draw(in: NSRect(x: 0, y: 0, width: 240, height: 240), angle: 45)
        NSGraphicsContext.restoreGraphicsState()
        return bitmap.representation(using: .png, properties: [:])!
    }()
}
