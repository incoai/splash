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
    /// The first message of a new chat, as the server takes it, moves the composer to the foot
    /// with its button, Stop since Send: in every frame the button keeps its place in the composer's
    /// row of controls, level with the mark at its start, as the change of its glyph moved it on a
    /// shorter animation of its own, ahead of the composer.
    @Test func sendMovesWithTheComposer() async throws {
        let fake = try FakeRuntime("splash-chat-send", launcher: "chat_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = fake.state()
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
        let wait = fake.root.appending(path: "wait")
        FileManager.default.createFile(atPath: wait.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: wait) }
        state.sendDraft()
        // The server takes the message in a turn of the app's own on the main actor, which a
        // test's frames leave no room for; the window draws nothing of it until the frames.
        try await waitFor(state.chats.selection != nil, "the server to take the message")
        var places: [CGFloat?] = []
        for _ in 0..<10 {
            pump(0.025)
            places.append(try await button())
        }
        try FileManager.default.removeItem(at: wait)
        #expect(try Self.field(in: window).minY < start - 100, "the composer moved to the foot")
        let end = try #require(places.last ?? nil)
        #expect(places.allSatisfy { $0.map { abs($0 - end) <= 2 } ?? false }, "the button against the mark: \(places)")
    }
}

extension ChatScreenTests {
    /// A new chat's first message leaves the window as it was until the server takes it. While the
    /// server holds the message, every frame is the window before Send but for the button, Stop
    /// since; and as the server refuses it, the list and the toolbar stay as they were and the
    /// composer stays in the middle, saying why over its text. With the chat opened and listed at
    /// Send, the composer went to the foot, a row came into the list and a title into the toolbar,
    /// and a refusal took them all back.
    @Test func aFirstMessageTheServerRefusesLeavesTheWindowAsItWas() async throws {
        let fake = try FakeRuntime("splash-chat-refused", launcher: "chat_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.chatting(fake)
        defer { state.quit() }
        let window = await Snapshot.window(MainView().environment(state), size: FirstMessagePictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        state.draft.text = FirstMessagePictures.draft
        state.draft.attach([try #require(NSImage(data: Showcase.picture))])
        try await waitFor(state.draft.attachments.count == 1, "the attachment")
        pump(0.5)
        let field = try Self.field(in: window)
        let size = window.frame.size
        // The button's square, at the end of the row of controls under the text.
        let button = CGRect(x: field.maxX - 40, y: size.height - field.minY, width: 48, height: 50)
        let sidebar = CGRect(x: 0, y: 60, width: sidebarWidth, height: size.height - 60)
        let toolbar = CGRect(x: 0, y: 0, width: size.width, height: 50)
        func picture() async throws -> NSBitmapImageRep { try #require(await Snapshot.picture(of: window)) }
        let before = try await picture()
        for step in ["hold", "refuse"] { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
        state.sendDraft()
        pump(0.03)
        // Held as it was sent: what is typed meanwhile, or composed by an input method, would be
        // lost as the chat takes the message.
        let text = try #require(window.contentView?.firstTextView)
        text.insertText(" And more.", replacementRange: NSRange(location: NSNotFound, length: 0))
        text.setMarkedText("zai", selectedRange: NSRange(location: 3, length: 0), replacementRange: NSRange(location: NSNotFound, length: 0))
        #expect(text.string == FirstMessagePictures.draft && !text.hasMarkedText() && state.draft.text == FirstMessagePictures.draft)
        // Each frame after a turn of the app's own on the main actor, where it takes in what the
        // server says.
        var held: [Int] = []
        for _ in 0..<8 {
            pump(0.03)
            await Task.yield()
            held.append(Self.unlike(try await picture(), before, in: CGRect(origin: .zero, size: size), but: button, window))
        }
        #expect(state.chats.streaming != nil && held.allSatisfy { $0 == 0 },
                "pixels unlike the window before Send, the button apart, per frame while the server holds the message: \(held)")
        #expect(Self.unlike(try await picture(), before, in: button, window) > 0, "Stop, since Send")

        try FileManager.default.removeItem(at: fake.root.appending(path: "hold"))
        // The frames through the refusal, and 8 from when the app has it, each with how many of
        // its pixels have the yellow of the mark of the line that says why, at the text's start.
        var refused: [(list: Int, toolbar: Int, composer: CGFloat, mark: Int?)] = []
        let deadline = ContinuousClock.now + .seconds(10)
        var since = 0
        while since < 8 {
            try #require(ContinuousClock.now < deadline, "timed out waiting for the refusal")
            pump(0.03)
            await Task.yield()
            if state.chats.streaming == nil { since += 1 }
            let drawn = try await picture()
            let scale = CGFloat(drawn.pixelsHigh) / size.height, color = Self.colors(of: drawn)
            let mark = (Int(60 * scale)..<drawn.pixelsHigh).reduce(0) { count, y in
                count + (Int((field.minX - 2) * scale)..<Int((field.minX + 22) * scale)).count { x in
                    color(x, y).0 - color(x, y).2 > 0.08
                }
            }
            refused.append((Self.unlike(drawn, before, in: sidebar, window), Self.unlike(drawn, before, in: toolbar, window),
                            try Self.field(in: window).minY - field.minY, since > 0 ? mark : nil))
        }
        #expect(state.draft.notice == "This message is longer than the model’s context." && state.draft.text == FirstMessagePictures.draft)
        #expect(!state.chats.answering && state.canSendDraft)
        // The message is the user's to edit again.
        text.insertText(" And more.", replacementRange: NSRange(location: NSNotFound, length: 0))
        #expect(state.draft.text == FirstMessagePictures.draft + " And more." && state.draft.notice == nil)
        #expect(refused.allSatisfy { $0.list == 0 && $0.toolbar == 0 }, "the list and the toolbar through the refusal: \(refused)")
        // The line that says why comes over the text, which stays.
        #expect(refused.allSatisfy { $0.composer == 0 }, "the composer's text against where it was, per frame: \(refused.map(\.composer))")
        // There from the first frame the refusal draws, whole: nothing moves that it would wait for.
        let marks = refused.compactMap(\.mark)
        #expect(marks.count == 8 && marks.allSatisfy { $0 > 20 && $0 == marks.last }, "its mark's pixels, per frame from the refusal: \(marks)")
    }
}

extension ChatScreenTests {
    /// A new chat's composer keeps its text in its place as what is over its text comes and goes:
    /// a picture and a PDF attached, the line that says why the server refused the message, gone
    /// as the text changes, back, a picture attached under it, and gone with an attachment taken
    /// off, the last ones taken off. In every frame drawn through each, the text starts where it
    /// did; what comes grows the composer up. Split a third over and two thirds under with the
    /// room the window leaves, the line that says why moved the text 17 points down and the mark 8
    /// up. A click on the text reaches it, in the middle and at a conversation's foot. A draft as
    /// tall as the composer takes, with all of that over it, still fits the least window, the
    /// mark under the toolbar and the button over the window's foot.
    @Test func aNewChatsTextKeepsItsPlaceAsWhatIsOverItComesAndGoes() async throws {
        let fake = try FakeRuntime("splash-chat-over-text", launcher: "chat_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.chatting(fake)
        defer { state.quit() }
        let window = await Snapshot.window(MainView().environment(state), size: FirstMessagePictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        state.draft.text = FirstMessagePictures.draft
        pump(0.5)
        let start = try Self.field(in: window)
        let refuse = fake.root.appending(path: "refuse")
        /// The text's top in every frame drawn as `change` is made, until `done` and 0.3 s more.
        func tops(_ change: () throws -> Void, until done: () -> Bool) async throws -> [CGFloat] {
            try change()
            var tops: [CGFloat] = []
            let deadline = ContinuousClock.now + .seconds(10)
            var after = 0
            while after < 15 {
                try #require(ContinuousClock.now < deadline, "timed out")
                pump(0.02)
                await Task.yield()
                tops.append(try Self.field(in: window).maxY)
                if done() { after += 1 }
            }
            return tops
        }
        let steps: [(String, () throws -> Void, () -> Bool)] = [
            ("a picture attached", { state.draft.attach([try #require(NSImage(data: Showcase.picture))]) },
             { state.draft.attachments.count == 1 }),
            ("a PDF attached", { state.draft.attach(pdfs: [try Showcase.pdf(pages: 12, at: fake.root.appending(path: "documents/Kyoto itinerary.pdf"))]) },
             { state.draft.attachments.count == 2 }),
            ("refused", {
                FileManager.default.createFile(atPath: refuse.path, contents: nil)
                state.sendDraft()
            }, { state.chats.streaming == nil && state.draft.notice != nil }),
            ("its reason gone as the text changes", { state.draft.text += " " }, { state.draft.notice == nil }),
            ("refused again", { state.sendDraft() }, { state.chats.streaming == nil && state.draft.notice != nil }),
            ("a picture attached under the reason", { state.draft.attach([try #require(NSImage(data: Showcase.picture))]) },
             { state.draft.attachments.count == 3 && state.draft.notice != nil }),
            ("the PDF taken off, and the reason", { state.draft.remove(at: 1) }, { state.draft.notice == nil && state.draft.attachments.count == 2 }),
            ("the pictures taken off", {
                state.draft.remove(at: 1)
                state.draft.remove(at: 0)
            }, { state.draft.attachments.isEmpty }),
        ]
        for (what, change, done) in steps {
            let tops = try await tops(change, until: done)
            #expect(tops.allSatisfy { $0 == start.maxY }, "\(what): the text's top per frame \(tops), before \(start.maxY)")
        }
        // A click on the text reaches it, the composer in the middle and at a conversation's foot.
        #expect(try Self.clicksReachTheText(in: window), "in the middle")
        state.chats.selection = try #require(state.chats.conversations.first { $0.title == "Kyoto in spring" }).id
        pump(1)
        #expect(try Self.clicksReachTheText(in: window), "at the foot")
        state.chats.newChat()
        pump(1)

        // The least window, a draft taller than the composer takes, all of it over the text.
        window.setContentSize(CGSize(width: 800, height: 560))
        state.draft.text = Array(repeating: "Another line of the plan, to see what fits.", count: 20).joined(separator: "\n")
        state.draft.attach([try #require(NSImage(data: Showcase.picture))])
        state.draft.attach(pdfs: [try Showcase.pdf(pages: 12, at: fake.root.appending(path: "documents/Kyoto itinerary.pdf"))])
        try await waitFor(state.draft.attachments.count == 2, "the attachments")
        state.sendDraft()
        try await waitFor(state.chats.streaming == nil && state.draft.notice != nil, "the refusal")
        pump(0.5)
        let field = try Self.field(in: window)
        let picture = try #require(await Snapshot.picture(of: window))
        let scale = CGFloat(picture.pixelsHigh) / window.frame.height, color = Self.colors(of: picture)
        /// The rows of the mark's color down the window at `x`, in the window's points from its top.
        func teal(at x: CGFloat) -> [CGFloat] {
            (0..<picture.pixelsHigh).filter { y in
                let (r, g, b) = color(Int(x * scale), y)
                return g - r > 0.3 && b > 0.4
            }.map { CGFloat($0) / scale }
        }
        // The mark's middle, over the text's middle; the button's, 14 points in from the text's end.
        let mark = teal(at: field.midX).filter { $0 < window.frame.height - field.maxY }, button = teal(at: field.maxX - 14)
        #expect(mark.count > 5 && mark.allSatisfy { $0 >= 52 }, "the mark, under the toolbar: \(mark.first ?? -1)")
        // Over the room the composer keeps under it at the foot: its inset and margin, 36 points.
        #expect(button.count > 5 && (button.last ?? .infinity) <= window.frame.height - 36, "the button, over the foot: \(button.last ?? -1)")
    }

    /// What the composer held goes at once as its chat opens: from the first frame the window
    /// draws once the server has taken a new chat's first message, nothing of the picture
    /// attached, nor of the line that said why the server had refused the message before, is
    /// left where the composer was, while the composer travels to the foot. On the travel's
    /// animation they faded there, over the transcript, behind the composer moving away.
    @Test func whatTheComposerHeldGoesAtOnceAsItsChatOpens() async throws {
        let opening = try await Self.opening("splash-chat-held")
        // The picture, by its blues, and the mark of the line that says why, by its yellow, however
        // faint: nothing else there, the composer's ground, its edge or the window's, has a tint.
        func left(_ picture: NSBitmapImageRep) -> (picture: Int, mark: Int) {
            let color = Self.colors(of: picture)
            var blue = 0, yellow = 0
            for y in 0..<picture.pixelsHigh {
                for x in 0..<picture.pixelsWide {
                    let (r, _, b) = color(x, y)
                    if b - r > 0.08 { blue += 1 }
                    if r - b > 0.08 { yellow += 1 }
                }
            }
            return (blue, yellow)
        }
        let before = left(opening.before.held)
        #expect(before.picture > 1000 && before.mark > 20, "the picture and the line's mark, before: \(before)")
        let frames = opening.drawn.map { (ms: $0.ms, left: left($0.held)) }
        #expect(frames.allSatisfy { $0.left == (0, 0) }, "pixels of them left where the composer was, per frame: \(frames)")
        #expect(opening.foot < opening.field.minY - 100, "the composer travelled to the foot")
    }

    /// The line that says why the server refused a new chat's first message comes back with the
    /// composer as the new chat opens again from another chat: in every frame that shows its
    /// mark, the mark is over the composer's text, as far from it as once still, the composer in
    /// the middle. Coming on the travel's animation, it faded in at its place in the middle, over
    /// nothing, ahead of the composer coming up under it; once the composer had moved, the
    /// composer travelled with an empty row for it.
    @Test func theReasonComesBackWithTheComposer() async throws {
        let fake = try FakeRuntime("splash-chat-reason", launcher: "chat_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await Self.chatting(fake)
        defer { state.quit() }
        let window = await Snapshot.window(MainView().environment(state), size: FirstMessagePictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        state.draft.text = FirstMessagePictures.draft
        let refuse = fake.root.appending(path: "refuse")
        FileManager.default.createFile(atPath: refuse.path, contents: nil)
        state.sendDraft()
        try await waitFor(state.chats.streaming == nil && state.draft.notice != nil, "the refusal")
        pump(0.5)
        let middle = try Self.field(in: window), size = window.frame.size
        // Down the window under the toolbar, across the line's mark, at the text's start.
        let strip = CGRect(x: middle.minX - 2, y: 60, width: 24, height: size.height - 60)
        /// How many of the strip's pixels have the mark's yellow, however faint, as the window draws now, and how
        /// far over the composer's text their middle row is, in points.
        func mark() throws -> (pixels: Int, over: CGFloat) {
            let frame = try #require(window.contentView?.superview)
            let area = frame.convert(CGRect(x: strip.minX, y: size.height - strip.maxY, width: strip.width, height: strip.height), from: nil)
            let bitmap = try #require(frame.bitmapImageRepForCachingDisplay(in: area))
            frame.cacheDisplay(in: area, to: bitmap)
            let color = Self.colors(of: bitmap)
            let rows = (0..<bitmap.pixelsHigh).map { y in
                (0..<bitmap.pixelsWide).count { x in
                    let (r, _, b) = color(x, y)
                    return r - b > 0.08
                }
            }
            let inked = rows.indices.filter { rows[$0] > 0 }
            let scale = CGFloat(bitmap.pixelsHigh) / strip.height
            let middle = inked.isEmpty ? 0 : strip.minY + CGFloat(inked.first! + inked.last!) / 2 / scale
            return (rows.reduce(0, +), size.height - (try Self.field(in: window).maxY) - middle)
        }
        let resting = try mark()
        #expect(resting.pixels > 20, "the line's mark, before")
        /// Waits until the composer's text has stopped moving, however long a slow Mac takes to
        /// draw its travel.
        func still() async throws -> CGFloat {
            var last = try Self.field(in: window).minY
            for _ in 0..<100 {
                pump(0.3)
                let now = try Self.field(in: window).minY
                if now == last { break }
                last = now
            }
            return last
        }
        state.chats.selection = try #require(state.chats.conversations.first { $0.title == "Kyoto in spring" }).id
        pump(0.3)
        #expect(try await still() < middle.minY - 100 && mark().pixels == 0, "another chat open, the composer at its foot")
        state.chats.newChat()
        var frames: [(ms: Int, mark: (pixels: Int, over: CGFloat), short: CGFloat)] = []
        let start = ContinuousClock.now
        for ms in [0, 25, 50, 75, 100, 150, 200, 240, 300, 400, 600] {
            while ContinuousClock.now < start + .milliseconds(ms) { pump(0.005) }
            frames.append((ms, try mark(), middle.minY - (try Self.field(in: window).minY)))
        }
        pump(0.3)
        #expect(try await still() == middle.minY && mark().pixels > 20, "the line again, the composer in the middle")
        #expect(frames.contains { $0.short > 50 && $0.mark.pixels > 20 }, "the composer travels with the line: \(frames)")
        #expect(frames.allSatisfy { $0.mark.pixels == 0 || abs($0.mark.over - resting.over) <= 1 },
                "the mark's pixels, how far over the text, \(resting.over) once still, and how far the composer is short of the middle, per frame: \(frames)")
    }

    /// The chat's title fades in at its place as its chat opens: every frame the window draws
    /// once the server has taken a new chat's first message has ink in the toolbar only where the
    /// title, its separator and its chevron are once still, after the sidebar's divider. Fading in
    /// on one animation with the room the toolbar gives it, it came some points from its place and
    /// cut short, its chevron over its words, and slid to its place as it showed.
    @Test func theTitleFadesInAtItsPlaceAsItsChatOpens() async throws {
        let opening = try await Self.opening("splash-chat-title-place")
        let before = Self.colors(of: opening.before.toolbar), still = Self.colors(of: opening.still.toolbar)
        let wide = opening.still.toolbar.pixelsWide, high = opening.still.toolbar.pixelsHigh
        func light(_ color: (Double, Double, Double)) -> Double { (color.0 + color.1 + color.2) / 3 }
        // Where the title is once still, and a pixel around, for its edges.
        var place = Set<Int>(), first = Int.max
        for y in 0..<high {
            for x in 0..<wide where abs(light(still(x, y)) - light(before(x, y))) > 0.02 {
                first = min(first, x)
                for (dx, dy) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] { place.insert((y + dy) * wide + x + dx) }
            }
        }
        // Its separator in line with the sidebar's divider, within the point the toolbar rounds to.
        let scale = CGFloat(wide) / Self.toolbarPart.width, from = CGFloat(first) / scale
        let divider = sidebarWidth - Self.toolbarPart.minX
        #expect(place.count > 500 && abs(from - divider) <= 1, "the title, once still, from \(from), the divider at \(divider)")
        /// The pixels `picture` inks that are none of the title's once still.
        func astray(_ picture: NSBitmapImageRep) -> Int {
            let color = Self.colors(of: picture)
            return (0..<high).reduce(0) { count, y in
                count + (0..<wide).count { x in abs(light(color(x, y)) - light(before(x, y))) > 0.06 && !place.contains(y * wide + x) }
            }
        }
        let frames = opening.drawn.map { (ms: $0.ms, astray: astray($0.toolbar)) }
        #expect(frames.allSatisfy { $0.astray == 0 }, "pixels inked where the title is not once still, per frame: \(frames)")
    }

    /// The composer's card is as tall as what it holds as its chat opens: in every frame the window
    /// draws once the server has taken a new chat's first message, the card is as tall as it is
    /// once still at the foot, empty, and its row of controls, its button, only ever moves down,
    /// from where it was. Moving and changing its height on one animation, the card drew empty and
    /// tall, 240 points shrinking to 110 over the first 60 ms of its travel.
    @Test func theComposerIsAsTallAsWhatItHoldsAsItsChatOpens() async throws {
        let opening = try await Self.opening("splash-chat-card")
        let before = try #require(Self.card(in: opening.before.card), "the card before")
        let still = try #require(Self.card(in: opening.still.card), "the card once still")
        #expect(before.height > still.height + 100, "the card before, \(before), and once still, \(still)")
        let frames = opening.drawn.map { (ms: $0.ms, card: Self.card(in: $0.card)) }
        // A frame with no card found is kept, to tell a card not drawn from one drawn otherwise.
        let kept = zip(opening.drawn, frames).filter { $1.card == nil }.map { drawn, _ in
            let file = FileManager.default.temporaryDirectory.appending(path: "composer-card-\(drawn.ms)ms-\(UUID().uuidString).png")
            try? drawn.card.representation(using: .png, properties: [:])?.write(to: file)
            return file.path(percentEncoded: false)
        }
        #expect(frames.allSatisfy { $0.card.map { abs($0.height - still.height) <= 2 } ?? false },
                "the card per frame, \(frames), once still \(still); the frames without one in \(kept)")
        let buttons = [before.button] + frames.compactMap(\.card?.button)
        #expect(zip(buttons, buttons.dropFirst()).allSatisfy { $1 >= $0 - 1 }, "its button per frame from before: \(buttons)")
    }

    /// Where the composer's card is in `strip`, a picture of the window's column at the card's
    /// end (`Parts.card`): its height, between the rows of its hairline over and under its button,
    /// in the room at its end, and its button's middle, the row of the mark's color, in the
    /// window's points from its top; nil where no card is drawn.
    static func card(in strip: NSBitmapImageRep) -> (height: CGFloat, button: CGFloat)? {
        let scale = CGFloat(strip.pixelsWide) / cardEnd.width, color = colors(of: strip)
        let rows = 0..<strip.pixelsHigh
        let teal = rows.filter { y in
            let (r, g, b) = color(Int((cardEnd.width - 30) * scale), y)
            return g - r > 0.3 && b > 0.4
        }
        guard let first = teal.first, let last = teal.last else { return nil }
        // The hairline, a gray between the card's ground and ink, past the end of its row of controls:
        // its ink in this row and the next, as a card moving between whole pixels draws it half in
        // each (a 1x display: 0.953 in two rows where it is 0.92 in one, on the Mac mini M6).
        func ink(_ y: Int) -> Double {
            guard rows.contains(y) else { return 0 }
            let (r, g, b) = color(Int((cardEnd.width - 10) * scale), y)
            return max(r, g, b) - min(r, g, b) < 0.04 && (r + g + b) / 3 > 0.5 ? 1 - (r + g + b) / 3 : 0
        }
        func hairline(_ y: Int) -> Bool { ink(y) + ink(y + 1) > 0.06 }
        guard let top = rows.prefix(first).last(where: hairline), let bottom = rows.suffix(from: last).first(where: hairline)
        else { return nil }
        return (CGFloat(bottom - top) / scale, cardEnd.minY + CGFloat(first + last) / 2 / scale)
    }

    /// The column at the end of the composer's card in a window of `FirstMessagePictures.size`, the
    /// sidebar shown, from under the toolbar down, in the window's points from its top left: its
    /// button, 30 points in from the card's end, and the room past it, 10 points in.
    static let cardEnd = CGRect(x: 1000, y: 52, width: 40, height: FirstMessagePictures.size.height - 52)

    /// What a window draws as the server takes a new chat's first message, where the change
    /// shows apart from the composer's travel: over the composer's text at its start, where the
    /// picture attached and the line that says why are, and in the toolbar from the Chat's switch
    /// to past the title at its widest, where the chat's title comes (`toolbarPart`); and the
    /// column at the composer's end (`cardEnd`).
    struct Parts {
        let held: NSBitmapImageRep
        let toolbar: NSBitmapImageRep
        let card: NSBitmapImageRep
    }

    /// The toolbar from the Chat's switch, where the toolbar first puts a title, to past the
    /// title at its widest, 360 points from the sidebar's divider, in the window's points from its
    /// top left.
    static let toolbarPart = CGRect(x: 150, y: 0, width: sidebarWidth - 150 + 420, height: 52)

    /// A window on a new chat whose first message, a text of several lines with a picture and a
    /// PDF, the server refused, so that the composer says why, and what it draws as the server
    /// takes the message sent again: its parts before, each frame of them from the first the
    /// window draws of the chat, some 0, 25, 50, 75, 100, 150, 240 and 480 ms after, with how long
    /// after it was, and its parts once still, the reply waiting.
    struct Opening {
        let before: Parts
        /// Where the composer's text was before.
        let field: CGRect
        var drawn: [(ms: Int, held: NSBitmapImageRep, toolbar: NSBitmapImageRep, card: NSBitmapImageRep)] = []
        var still: Parts
        /// How high the composer's text is once still.
        var foot: CGFloat
    }

    static func opening(_ name: String) async throws -> Opening {
        let fake = try FakeRuntime(name, launcher: "chat_server")
        defer { try? FileManager.default.removeItem(at: fake.root) }
        let state = try await chatting(fake)
        defer { state.quit() }
        let window = await Snapshot.window(MainView().environment(state), size: FirstMessagePictures.size, appearance: .aqua)
        defer { window.orderOut(nil) }
        state.draft.text = FirstMessagePictures.draft
        state.draft.attach([try #require(NSImage(data: Showcase.picture))])
        state.draft.attach(pdfs: [try Showcase.pdf(pages: 12, at: fake.root.appending(path: "documents/Kyoto itinerary.pdf"))])
        try await waitFor(state.draft.attachments.count == 2, "the attachments")
        let refuse = fake.root.appending(path: "refuse"), wait = fake.root.appending(path: "wait")
        FileManager.default.createFile(atPath: refuse.path, contents: nil)
        state.sendDraft()
        try await waitFor(state.chats.streaming == nil && state.draft.notice != nil, "the refusal")
        try FileManager.default.removeItem(at: refuse)
        // The reply waits, so that the frames are of the chat opening alone.
        FileManager.default.createFile(atPath: wait.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: wait) }
        pump(0.5)
        let field = try field(in: window), size = window.frame.size
        // In the window's points from its top left: 110 points over the text, 64 from its start,
        // short of the transcript's messages.
        let held = CGRect(x: field.minX - 2, y: size.height - field.maxY - 110, width: 64, height: 110)
        let toolbar = toolbarPart
        /// The window's `rect` as it draws now. Each part alone, as drawing the whole window
        /// takes some 70 ms, longer than the frames are apart.
        func picture(_ rect: CGRect) throws -> NSBitmapImageRep {
            let frame = try #require(window.contentView?.superview)
            let area = frame.convert(CGRect(x: rect.minX, y: size.height - rect.maxY, width: rect.width, height: rect.height), from: nil)
            let bitmap = try #require(frame.bitmapImageRepForCachingDisplay(in: area))
            frame.cacheDisplay(in: area, to: bitmap)
            return bitmap
        }
        func parts() throws -> Parts { Parts(held: try picture(held), toolbar: try picture(toolbar), card: try picture(cardEnd)) }
        var opening = Opening(before: try parts(), field: field, still: try parts(), foot: field.minY)
        state.sendDraft()
        // The server takes the message in a turn of the app's own on the main actor; the window
        // draws nothing of it until the frames.
        try await waitFor(state.chats.selection != nil, "the server to take the message")
        let start = ContinuousClock.now
        for ms in [0, 25, 50, 75, 100, 150, 240, 480] {
            while ContinuousClock.now < start + .milliseconds(ms) { pump(0.005) }
            // What the window's own views queue for the main actor, as the title's place.
            await Task.yield()
            let elapsed = (ContinuousClock.now - start).components
            let drawn = try parts()
            opening.drawn.append((Int(elapsed.seconds) * 1000 + Int(elapsed.attoseconds / 1_000_000_000_000_000), drawn.held, drawn.toolbar,
                                  drawn.card))
        }
        // Still once two pictures 0.3 s apart are alike, however long a slow Mac takes to draw the
        // travel and the title's fade.
        func bytes(_ picture: NSBitmapImageRep) -> Data { Data(bytes: picture.bitmapData!, count: picture.bytesPerRow * picture.pixelsHigh) }
        for _ in 0..<100 {
            pump(0.3)
            await Task.yield()
            let now = try parts(), foot = try Self.field(in: window).minY
            let alike = bytes(now.toolbar) == bytes(opening.still.toolbar) && foot == opening.foot
            (opening.still, opening.foot) = (now, foot)
            if alike { break }
        }
        return opening
    }
}

extension ChatScreenTests {
    /// An app on `fake`, a `chat_server`, with a model running and chats of several days in its
    /// list, one pinned, its window to open on a new chat.
    static func chatting(_ fake: FakeRuntime) async throws -> AppState {
        let chats = [("Kyoto in spring", 0), ("Why is the sea salty?", 0), ("Bread recipe", 1), ("Explain KV cache paging", 3),
                     ("Release notes", 9), ("Tax questions", 40)]
        for (index, (title, days)) in chats.enumerated() {
            let conversation = Conversation(title: title, updated: .now - Double(days * 86_400 + index * 60),
                                            messages: [ChatMessage(role: .user, text: title), ChatMessage(role: .assistant, text: "Yes.")],
                                            pinned: title == "Release notes" ? .now : nil)
            try JSONEncoder().encode(conversation).write(to: fake.chats.appending(path: "\(conversation.id.uuidString).json"))
        }
        // Installed, as a model that runs is, so that its name opens the models' menu.
        try fake.install("example/Model")
        let state = fake.state()
        await state.launch()
        state.server.start("example/Model", settings: ServeSettings(values: ["--port": String(fake.port)]))
        try await waitFor(state.server.phase == .ready, "ready")
        state.section = .chat
        return state
    }

    /// How many pixels of `picture` in `rect`, in `window`'s points from its top left, are unlike
    /// those of `other`, those in `hole` apart.
    static func unlike(_ picture: NSBitmapImageRep, _ other: NSBitmapImageRep, in rect: CGRect, but hole: CGRect = .null,
                       _ window: NSWindow) -> Int {
        let scale = CGFloat(picture.pixelsHigh) / window.frame.height, size = picture.bitsPerPixel / 8
        guard let one = picture.bitmapData, let two = other.bitmapData, picture.bytesPerRow == other.bytesPerRow,
              picture.pixelsHigh == other.pixelsHigh else { return .max }
        func pixels(_ rect: CGRect) -> (x: Range<Int>, y: Range<Int>) {
            (max(0, Int(rect.minX * scale))..<min(picture.pixelsWide, Int(rect.maxX * scale)),
             max(0, Int(rect.minY * scale))..<min(picture.pixelsHigh, Int(rect.maxY * scale)))
        }
        let area = pixels(rect), apart = hole.isNull ? nil : pixels(hole)
        var count = 0
        for y in area.y {
            let row = y * picture.bytesPerRow
            for x in area.x where !(apart.map { $0.x.contains(x) && $0.y.contains(y) } ?? false) {
                if memcmp(one + row + x * size, two + row + x * size, size) != 0 { count += 1 }
            }
        }
        return count
    }
}

/// A new chat's first message drawn to `first-message-*.png` in `SPLASH_APP_SNAPSHOTS`, light and
/// dark, for people to review its motion, with chats in the list and a draft of several lines, a
/// picture and a PDF attached to it: Send while the server holds the message, its refusal, another
/// chat opened and the new chat again, Send again and Stop, Send once more, the server taking it
/// and the reply, then a new chat with a picture attached and taken off and a draft as tall as the
/// composer takes, each at 0, 25, 50, 75 and 100% of `settle`,
/// at twice that, once what comes has faded in, and at five times that, 1.2 s, by when a window
/// drawn off screen shows the button's new glyph too.
@MainActor
@Suite(.serialized, .enabled(if: Snapshot.directory != nil))
struct FirstMessagePictures {
    /// The app's window as it first opens.
    static let size = CGSize(width: 1120, height: 780)
    static let draft = """
        Plan a week in Kyoto in early April for the two of us.
        We land at Kansai on the 3rd, like walking more than museums, and want one day trip out of the city.
        Start from the attached itinerary, and say what the picture adds.
        """

    @Test func aRefusalAStopAndATakeFrameByFrame() async throws {
        let directory = try #require(Snapshot.directory)
        for (suffix, appearance) in [("light", NSAppearance.Name.aqua), ("dark", .darkAqua)] {
            let fake = try FakeRuntime("splash-first-message-pictures", launcher: "chat_server")
            defer { try? FileManager.default.removeItem(at: fake.root) }
            let state = try await ChatScreenTests.chatting(fake)
            defer { state.quit() }
            state.transcript.page.load()
            Snapshot.drawsOffScreen(try #require(state.transcript.page.webView))
            let window = await Snapshot.window(MainView().environment(state), size: Self.size, appearance: appearance)
            defer { window.orderOut(nil) }
            state.draft.text = Self.draft
            pump(0.5)
            let pdf = try Showcase.pdf(pages: 12, at: fake.root.appending(path: "documents/Kyoto itinerary.pdf"))
            func signal(_ step: String) { FileManager.default.createFile(atPath: fake.root.appending(path: step).path, contents: nil) }
            func clear(_ step: String) throws { try FileManager.default.removeItem(at: fake.root.appending(path: step)) }
            func frames(_ name: String, at percents: [Int] = [0, 25, 50, 75, 100, 200, 500]) async {
                let start = ContinuousClock.now
                for percent in percents {
                    while ContinuousClock.now < start + .milliseconds(percent * 24 / 10) { pump(0.005) }
                    try? await Snapshot.picture(of: window)?.representation(using: .png, properties: [:])?
                        .write(to: directory.appending(path: "first-message-\(name)-\(percent)-\(suffix).png"))
                }
            }
            /// Waits for `condition`, letting the app's own work on the main actor run meanwhile.
            func until(_ condition: () -> Bool, _ what: String) async throws {
                let deadline = ContinuousClock.now + .seconds(30)
                while !condition() {
                    try #require(ContinuousClock.now < deadline, "timed out waiting for \(what)")
                    pump(0.002)
                    await Task.yield()
                }
            }
            await frames("0-new", at: [0])
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            try await until({ state.draft.attachments.count == 1 }, "the picture")
            await frames("0a-picture")
            state.draft.attach(pdfs: [pdf])
            try await until({ state.draft.attachments.count == 2 }, "the PDF")
            await frames("0b-pdf")
            signal("hold")
            signal("refuse")
            state.sendDraft()
            await frames("1-send")
            try clear("hold")
            try await until({ state.chats.streaming == nil }, "the refusal")
            await frames("2-refused")
            // Another chat opened and the new chat again, as their rows in the list open them.
            state.chats.selection = state.chats.conversations.first { $0.title == "Kyoto in spring" }?.id
            await frames("2a-other-chat")
            state.chats.newChat()
            await frames("2b-new-chat-again")
            signal("hold")
            state.sendDraft()
            await frames("3-send-again")
            state.chats.stop()
            try await until({ state.chats.streaming == nil }, "the stop")
            await frames("4-stopped")
            try clear("refuse")
            signal("wait")
            state.sendDraft()
            await frames("5-send-once-more")
            try clear("hold")
            try await until({ state.chats.selection != nil }, "the server to take the message")
            await frames("6-taken")
            try clear("wait")
            try await until({ state.chats.streaming == nil && state.chats.current?.title == "A Title" }, "the reply and its title")
            await frames("7-answered", at: [500])
            // A new chat again, a picture attached and taken off, and a draft as tall as the
            // composer takes with a picture and a PDF, in the window and in the least one.
            state.chats.newChat()
            await frames("8-new-chat")
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            try await until({ state.draft.attachments.count == 1 }, "the picture")
            await frames("9-picture")
            state.draft.remove(at: 0)
            await frames("9a-taken-off")
            state.draft.text = Array(repeating: "Another line of the plan, to see what fits.", count: 20).joined(separator: "\n")
            state.draft.attach([try #require(NSImage(data: Showcase.picture))])
            state.draft.attach(pdfs: [pdf])
            try await until({ state.draft.attachments.count == 2 }, "the attachments")
            await frames("10-tall", at: [500])
            window.setContentSize(CGSize(width: 800, height: 560))
            await frames("10a-tall-least", at: [500])
        }
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
        let state = fake.state()
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
        let text = try #require(window.contentView?.firstTextView?.enclosingScrollView)
        return text.convert(text.bounds, to: nil)
    }

    /// Whether a click in the middle of the composer's text in `window` goes to its text view.
    static func clicksReachTheText(in window: NSWindow) throws -> Bool {
        let content = try #require(window.contentView), field = try field(in: window)
        let point = try #require(content.superview).convert(NSPoint(x: field.midX, y: field.midY), from: nil)
        return content.hitTest(point) === content.firstTextView
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
