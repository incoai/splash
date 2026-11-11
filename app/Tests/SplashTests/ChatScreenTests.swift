import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// The Chat's motion, frame by frame: what its window draws as its composer, its transcript and its
/// list change.
@MainActor
@Suite(.serialized)
struct ChatScreenTests {
    /// The composer's words for the server change in place as a stop changes them, as a control's
    /// title does: no frame drawn through the stop has them fading, as a cross-fade of the words
    /// before and after drew them over each other.
    @Test func theComposersWordsChangeInPlaceAsTheServerStops() async throws {
        let fake = try FakeRuntime("splash-chat-words", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await ServerScreenTests.launchedWithModels(fake)
        defer { state.quit() }
        state.options["--port"] = String(fake.port)
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.start("unsloth/flint-GGUF:UD-Q4_K_XL")
        try await waitFor(state.server.phase == .ready, "ready")
        state.section = .chat
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        let field = try Self.field(in: window)
        // The row under the text past Attach Images: the mark, the words and the menu's chevron.
        let slot = CGRect(x: field.minX + 22, y: window.frame.height - field.minY + 8, width: 300, height: 30)
        let stop = try await Self.frames(of: window) { state.stop() }
        let limit = 1.5 * max(Self.fading(stop.before, slot, window), Self.fading(stop.after, slot, window))
        let drawn = stop.drawn.map { Self.fading($0, slot, window) }
        #expect(drawn.allSatisfy { $0 <= limit }, "faint ink against full, per frame: \(drawn), at most \(limit)")
    }
}

extension ChatScreenTests {
    /// The composer's mark moves with its words as a switch of models moves them, its pose alone
    /// changing on an animation of its own: in every frame the slot's ink, from the mark's start to
    /// the chevron's end, spans what it does once settled. With its look's animation moving it
    /// too, the mark trailed the words and drew over their first letter.
    @Test func theComposersMarkMovesWithItsWords() async throws {
        let fake = try FakeRuntime("splash-chat-mark", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await ServerScreenTests.launchedWithModels(fake)
        defer { state.quit() }
        state.options["--port"] = String(fake.port)
        let steps = ["loaded", "warm"].map { fake.root.appending(path: $0) }
        for step in steps { FileManager.default.createFile(atPath: step.path, contents: nil) }
        state.start("mlx-community/Qwen3.8-27B-4bit")
        try await waitFor(state.server.phase == .ready, "ready")
        // The next model stays starting.
        for step in steps { try FileManager.default.removeItem(at: step) }
        state.section = .chat
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        let field = try Self.field(in: window)
        // The row under the text, from Attach Images to well short of the controls at its end.
        let row = CGRect(x: field.minX - 10, y: window.frame.height - field.minY + 8, width: 330, height: 30)
        /// How wide the ink in the row is, in points: whatever is not the composer's white.
        func span(_ picture: NSBitmapImageRep) -> CGFloat {
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height, color = Self.colors(of: picture)
            let inked = (Int(row.minX * scale)..<Int(row.maxX * scale)).filter { x in
                (Int(row.minY * scale)..<Int(row.maxY * scale)).contains { y in
                    let (r, g, b) = color(x, y)
                    return (r + g + b) / 3 < 0.8 || max(r, g, b) - min(r, g, b) > 0.15
                }
            }
            return CGFloat((inked.last ?? 0) - (inked.first ?? 0)) / scale
        }
        let frames = try await Self.frames(of: window) { state.choose("unsloth/flint-GGUF:UD-Q4_K_XL") }
        let settled = span(frames.after)
        let drawn = frames.drawn.map(span)
        #expect(drawn.allSatisfy { abs($0 - settled) <= 2 }, "the slot's span per frame \(drawn), settled \(settled)")
    }
}

extension ChatScreenTests {
    /// The first message of a new chat moves the composer to the foot with its Send button, which
    /// turns into Stop as it goes: in every frame the button keeps its place in the composer's row
    /// of controls, level with the mark at its start, as the change of its glyph moved it on a
    /// shorter animation of its own, ahead of the composer.
    @Test func sendMovesWithTheComposer() async throws {
        let fake = try FakeRuntime("splash-chat-send", launcher: "chat_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        state.server.start("example/Model", settings: ServeSettings(values: ["--port": String(fake.port)]))
        try await waitFor(state.server.phase == .ready, "ready")
        state.section = .chat
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        state.draft.text = "Plan a trip to Kyoto in spring"
        pump(0.5)
        /// How far below the mark at the start of the button's row its middle is, in the window's
        /// points: both in the mark's color, the button the composer's one solid element; nil where
        /// either is not drawn.
        func button() async throws -> CGFloat? {
            let field = try Self.field(in: window)
            let picture = try #require(await Snapshot.picture(of: window))
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height
            let color = Self.colors(of: picture)
            /// The middle row of what is drawn in the mark's color between `minX` and `maxX`, under
            /// the toolbar.
            func middle(_ minX: CGFloat, _ maxX: CGFloat) -> CGFloat? {
                var rows: [Int] = []
                for y in Int(60 * scale)..<picture.pixelsHigh {
                    for x in Int(minX * scale)..<Int(maxX * scale) {
                        let (r, g, b) = color(x, y)
                        if g - r > 0.3 && b > 0.4 { rows.append(y) }
                    }
                }
                return rows.isEmpty ? nil : CGFloat(rows.reduce(0, +)) / CGFloat(rows.count) / scale
            }
            guard let mark = middle(field.minX - 10, field.minX + 60), let button = middle(field.maxX - 40, field.maxX + 20)
            else { return nil }
            return button - mark
        }
        let start = try Self.field(in: window).minY
        // The reply waits while the frames are drawn: on a loaded Mac they took long enough for
        // its first words to change the mark, which then matched no color.
        let hold = fake.root.appending(path: "hold")
        FileManager.default.createFile(atPath: hold.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: hold) }
        state.sendDraft()
        var places: [CGFloat?] = []
        for _ in 0..<10 {
            pump(0.025)
            places.append(try await button())
        }
        try FileManager.default.removeItem(at: hold)
        #expect(try Self.field(in: window).minY < start - 100, "the composer moved to the foot")
        let end = try #require(places.last ?? nil)
        #expect(places.allSatisfy { $0.map { abs($0 - end) <= 2 } ?? false }, "the button against the mark: \(places)")
    }
}

extension ChatScreenTests {
    /// A new chat opened from a conversation takes its transcript away at once, as the composer
    /// moves to the middle, rather than fading it out under the composer moving over it; a
    /// conversation opened from a new chat fades it in once the composer has reached the foot,
    /// rather than under the composer moving away.
    @Test func aNewChatTakesTheTranscriptAwayAtOnce() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-swap-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let conversation = Conversation(title: "Kyoto in spring", messages: [
            ChatMessage(role: .user, text: "Tell me about Kyoto in spring."),
            ChatMessage(role: .assistant, text: String(repeating: "Cherry blossoms line the river in early April. ", count: 40)),
        ])
        try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        let state = AppState.forTests(in: folder, runtime: Showcase.runtime)
        state.chats.selection = conversation.id
        state.section = .chat
        state.transcript.page.load()
        let webView = try #require(state.transcript.page.webView)
        let window = await Snapshot.window(ChatScreen().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        try await waitFor(webView.window === window)
        func shown() -> Double {
            guard webView.window != nil else { return 0 }
            var opacity = 1.0, view: NSView? = webView
            while let next = view {
                opacity *= Double(next.alphaValue) * Double(next.layer?.presentation()?.opacity ?? next.layer?.opacity ?? 1)
                view = next.superview
            }
            return opacity
        }
        // Each frame: how much of the transcript shows, and where the composer is.
        func frames() throws -> [(shown: Double, composer: CGFloat)] {
            try (0..<16).map { _ in
                pump(0.03)
                return (shown(), try Self.field(in: window).minY)
            }
        }
        state.chats.newChat()
        let opening = try frames()
        #expect(opening.allSatisfy { $0.shown == 0 }, "the transcript as a new chat opens: \(opening)")
        pump(0.5)
        state.chats.selection = conversation.id
        let returning = try frames()
        pump(0.5)
        let foot = try Self.field(in: window).minY
        #expect(returning.contains { $0.composer > foot + 50 }, "the composer moves to the foot: \(returning)")
        // Within the few points the end of its animation takes.
        #expect(returning.allSatisfy { $0.shown == 0 || abs($0.composer - foot) < 5 },
                "the transcript before the composer is at the foot: \(returning)")
        #expect(shown() == 1)
    }
}

extension ChatScreenTests {
    /// The list takes each change at once, its rows never drawn over each other: a chat pinned or unpinned shows
    /// at once in its new place, one deleted goes at once, and the rows around them take their places. Pinned, a
    /// chat slid to its new place over the rows it passed; deleted, it faded out under the row moving into its
    /// place; with the list scrolled, a row put aside above the screen flew from the list's foot to its top over
    /// every row; and with the rows snapping, one that came left its new place empty for a quarter of a second
    /// before it faded in. Every frame drawn after a change shows the list as it ends, each line of ink one title
    /// tall.
    @Test func theListsRowsNeverDrawOverEachOther() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-chat-list-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let titles = ["Weekly groceries", "Release notes", "Why is the sea salty?", "Kyoto in spring", "Bread recipe",
                      "Explain KV cache paging", "SwiftUI transactions", "Birthday ideas", "Tax questions", "Garden planning",
                      "Rust lifetimes", "Marathon training", "Moving checklist", "Night sky settings", "Old project ideas",
                      "Learning Japanese", "Laptop comparison", "Wedding speech", "Cover letter", "Sourdough starter"]
        let days = [9, 3, 0, 0, 1, 3, 4, 5, 6, 10, 12, 15, 20, 25, 40, 60, 90, 120, 200, 300]
        for (index, title) in titles.enumerated() {
            let conversation = Conversation(title: title, updated: .now - Double(days[index] * 86_400 + index * 60),
                                            messages: [ChatMessage(role: .user, text: title), ChatMessage(role: .assistant, text: "Yes.")],
                                            pinned: index < 2 ? .now - Double(index * 60) : nil)
            try JSONEncoder().encode(conversation).write(to: folder.appending(path: "\(conversation.id.uuidString).json"))
        }
        let state = AppState.forTests(in: folder)
        func id(_ title: String) throws -> UUID { try #require(state.chats.conversations.first { $0.title == title }).id }
        state.chats.selection = try id("Why is the sea salty?")
        state.section = .chat
        let window = await Snapshot.window(ChatScreen().environment(state), size: CGSize(width: 900, height: 600), appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        // The list under the search field, across the rows' titles.
        let list = CGRect(x: 18, y: 140, width: 240, height: 440)
        /// The tallest line of ink in the list in `picture`, in points: dark gray pixels, titles
        /// and the groups' names, which the open row's highlight is lighter than.
        func tallest(_ picture: NSBitmapImageRep) -> CGFloat {
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height, color = Self.colors(of: picture)
            var run = 0, longest = 0
            for y in Int(list.minY * scale)..<Int(list.maxY * scale) {
                let inked = (Int(list.minX * scale)..<Int(list.maxX * scale)).contains { x in
                    let (r, g, b) = color(x, y)
                    return max(r, g, b) - min(r, g, b) < 0.08 && (r + g + b) / 3 < 0.75
                }
                run = inked ? run + 1 : 0
                longest = max(longest, run)
            }
            return CGFloat(longest) / scale
        }
        /// The list in `picture`, a row of pixels at a time.
        func rows(_ picture: NSBitmapImageRep) -> [Data] {
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height, data = picture.bitmapData!
            let start = Int(list.minX * scale) * picture.bitsPerPixel / 8, length = Int(list.width * scale) * picture.bitsPerPixel / 8
            return (Int(list.minY * scale)..<Int(list.maxY * scale)).map { Data(bytes: data + $0 * picture.bytesPerRow + start, count: length) }
        }
        func check(_ what: String, _ change: () throws -> Void) async throws {
            var thrown: (any Error)?
            let frames = try await Self.frames(of: window) { do { try change() } catch { thrown = error } }
            if let thrown { throw thrown }
            let line = max(tallest(frames.before), tallest(frames.after)) + 2
            let drawn = frames.drawn.map(tallest)
            #expect(drawn.allSatisfy { $0 <= line }, "\(what): the tallest line of ink per frame \(drawn), a title's \(line)")
            let ending = rows(frames.after)
            let unlike = frames.drawn.map { zip(rows($0), ending).count { $0 != $1 } }
            #expect(unlike.allSatisfy { $0 == 0 }, "\(what): rows of pixels unlike the list as it ends, per frame: \(unlike)")
        }
        try await check("pinning a chat of yesterday") { state.chats.pin(try id("Bread recipe"), true) }
        try await check("unpinning it") { state.chats.pin(try id("Bread recipe"), false) }
        try await check("deleting a chat") { state.chats.delete([try id("Birthday ideas")]) }
        // Scrolled so that the pinned and today's chats are above the screen.
        func scrollViews(in view: NSView) -> [NSScrollView] {
            view.subviews.flatMap { ($0 as? NSScrollView).map { [$0] + scrollViews(in: $0) } ?? scrollViews(in: $0) }
        }
        let content = try #require(window.contentView)
        let sidebar = try #require(scrollViews(in: content).first { $0.convert($0.bounds, to: nil).maxX <= 282 })
        sidebar.scroll(by: 200)
        pump(0.3)
        try await check("pinning a chat above the screen") { state.chats.pin(try id("Kyoto in spring"), true) }
    }
}

extension ChatScreenTests {
    /// The open chat's title shows in the toolbar, after the sidebar's divider, as the Chat shows in a window
    /// open on the Server, as it does in a window opened on the Chat: it showed faint, 36 points to the left,
    /// over the sidebar.
    @Test func theTitleShowsAsTheChatShowsInAnOpenWindow() async throws {
        let fake = try FakeRuntime("splash-chat-title", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let conversation = Conversation(title: "Kyoto in spring", messages: [
            ChatMessage(role: .user, text: "Tell me about Kyoto in spring."), ChatMessage(role: .assistant, text: "Blossoms."),
        ])
        try JSONEncoder().encode(conversation).write(to: fake.chats.appending(path: "\(conversation.id.uuidString).json"))
        let state = try await ServerScreenTests.launchedWithModels(fake)
        defer { state.quit() }
        state.chats.selection = conversation.id
        /// The title's ink in the toolbar past the Chat's switch, its dark pixels, and where the first of them
        /// is, in points from the window's leading edge, once the window has laid out what changed, as one on
        /// the screen does as it draws.
        func title(in window: NSWindow) async throws -> (ink: Int, start: CGFloat) {
            // Waiting, so that the main actor runs what the probe queues for it, as an app's does between events.
            for _ in 0..<10 {
                try await Task.sleep(for: .milliseconds(100))
                window.layoutIfNeeded()
                window.displayIfNeeded()
                pump(0.02)
            }
            let picture = try #require(await Snapshot.picture(of: window))
            let scale = CGFloat(picture.pixelsHigh) / window.frame.height, color = Self.colors(of: picture)
            var ink = 0, start = Int.max
            for y in Int(10 * scale)..<Int(44 * scale) {
                for x in Int(210 * scale)..<Int((sidebarWidth + 300) * scale) {
                    let (r, g, b) = color(x, y)
                    if (r + g + b) / 3 < 0.4 {
                        ink += 1
                        start = min(start, x)
                    }
                }
            }
            return (ink, CGFloat(start) / scale)
        }
        state.section = .chat
        let opened = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        let shown = try await title(in: opened)
        opened.orderOut(nil)
        state.section = .server
        let window = await Snapshot.window(MainView().environment(state), size: ServerPictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        pump(0.5)
        state.section = .chat
        let swapped = try await title(in: window)
        #expect(shown.ink > 50 && shown.start > sidebarWidth, "in a window opened on the Chat: \(shown)")
        #expect(swapped.ink > shown.ink / 2 && swapped.start == shown.start, "in a window open on the Server: \(swapped), opened on the Chat: \(shown)")
    }
}

extension ChatScreenTests {
    /// At the window's least width, a long chat title truncates to the room the toolbar has before its buttons at
    /// the end at their widest, the Server's naming a model of a long name, sidebar shown or hidden: the title took
    /// up to 360 points past the sidebar and pushed the Server's button into the toolbar's overflow menu.
    @Test func aLongTitleLeavesTheToolbarsButtons() async throws {
        let fake = try FakeRuntime("splash-chat-long-title", launcher: "stepped_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let conversation = Conversation(title: String(repeating: "A long title that goes on ", count: 4), messages: [
            ChatMessage(role: .user, text: "Hello"), ChatMessage(role: .assistant, text: "Hi."),
        ])
        try JSONEncoder().encode(conversation).write(to: fake.chats.appending(path: "\(conversation.id.uuidString).json"))
        let model = "example/A-Model-Whose-Name-Is-Long-Enough-To-Fill-The-Server-Button-GGUF:Q4"
        try fake.install(model)
        let state = fake.state(keychain: Keychain(service: "Splash Tests"))
        defer { state.quit() }
        await state.launch()
        for step in ["loaded", "warm"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.server.start(model, settings: ServeSettings(values: ["--port": String(fake.port)]))
        try await waitFor(state.server.phase == .ready, "ready")
        state.chats.selection = conversation.id
        state.section = .chat
        let window = await Snapshot.window(MainView().environment(state), size: CGSize(width: 800, height: 600), appearance: .aqua)
        defer { window.orderOut(nil) }
        /// The toolbar's items with a view, each in the toolbar and inside the window, or not, once the window has laid
        /// out what changed, waiting so that the main actor runs what the title's probe queues for it.
        func shown() async throws -> [Bool] {
            for _ in 0..<10 {
                try await Task.sleep(for: .milliseconds(100))
                window.layoutIfNeeded()
                window.displayIfNeeded()
                pump(0.02)
            }
            return (window.toolbar?.items ?? []).compactMap { item in
                item.view.map { view in
                    item.isVisible && view.window === window && view.convert(view.bounds, to: nil).maxX <= window.frame.width
                }
            }
        }
        let withSidebar = try await shown()
        state.toggleSidebar()
        let withoutSidebar = try await shown()
        #expect(withSidebar.count == 5 && withSidebar.allSatisfy { $0 }, "with the sidebar: \(withSidebar)")
        #expect(withoutSidebar.count == 5 && withoutSidebar.allSatisfy { $0 }, "without the sidebar: \(withoutSidebar)")
    }
}

extension ChatScreenTests {
    /// The text view of the composer in `window`, where it lies in the window.
    static func field(in window: NSWindow) throws -> CGRect {
        func textViews(in view: NSView) -> [NSTextView] {
            view.subviews.flatMap { ($0 as? NSTextView).map { [$0] } ?? textViews(in: $0) }
        }
        let text = try #require(window.contentView.flatMap { textViews(in: $0).first { !$0.isFieldEditor } }?.enclosingScrollView)
        return text.convert(text.bounds, to: nil)
    }

    /// The frames `window` draws through `change`: the one before it, `count` drawn some 30 ms
    /// apart after it, and the one once it has stopped changing, which a slow Mac, as a virtual
    /// one is, reaches long after the animations' length.
    static func frames(of window: NSWindow, count: Int = 8, _ change: () -> Void) async throws
        -> (before: NSBitmapImageRep, drawn: [NSBitmapImageRep], after: NSBitmapImageRep) {
        func picture() async throws -> NSBitmapImageRep { try #require(await Snapshot.picture(of: window)) }
        func bytes(_ picture: NSBitmapImageRep) -> Data { Data(bytes: picture.bitmapData!, count: picture.bytesPerRow * picture.pixelsHigh) }
        let before = try await picture()
        change()
        var drawn: [NSBitmapImageRep] = []
        for _ in 0..<count {
            pump(0.03)
            drawn.append(try await picture())
        }
        var after = try await picture()
        for _ in 0..<40 {
            pump(0.25)
            let next = try await picture()
            if bytes(next) == bytes(after) { break }
            after = next
        }
        return (before, drawn, after)
    }

    /// How faint the gray ink of `picture` in `rect`, in `window`'s points from its top left, is
    /// against the ink at full strength: the ratio of the pixels of text fading, or at a glyph's
    /// edge, to those of text drawn whole, on a light ground. Text cross-fading or fading out
    /// raises it well past what the same text drawn whole has.
    static func fading(_ picture: NSBitmapImageRep, _ rect: CGRect, _ window: NSWindow) -> Double {
        let scale = CGFloat(picture.pixelsHigh) / window.frame.height
        let color = colors(of: picture)
        var full = 0, faint = 0
        for y in Int(rect.minY * scale)..<Int(rect.maxY * scale) {
            for x in Int(rect.minX * scale)..<Int(rect.maxX * scale) {
                let (r, g, b) = color(x, y)
                guard max(r, g, b) - min(r, g, b) < 0.08 else { continue }
                let lightness = (r + g + b) / 3
                if lightness < 0.6 { full += 1 } else if lightness < 0.9 { faint += 1 }
            }
        }
        return Double(faint) / Double(max(full, 1))
    }

    /// The color of each pixel of `picture`, by column and row from its top left, from 0 to 1 a
    /// component: read from its bytes, as the tests read thousands of pixels a frame.
    static func colors(of picture: NSBitmapImageRep) -> (Int, Int) -> (Double, Double, Double) {
        let data = picture.bitmapData!, row = picture.bytesPerRow, size = picture.bitsPerPixel / 8
        let first = picture.bitmapFormat.contains(.alphaFirst) ? 1 : 0
        return { x, y in
            let at = y * row + x * size + first
            return (Double(data[at]) / 255, Double(data[at + 1]) / 255, Double(data[at + 2]) / 255)
        }
    }
}
