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
        state.sendDraft()
        var places: [CGFloat?] = []
        for _ in 0..<10 {
            pump(0.025)
            places.append(try await button())
        }
        #expect(try Self.field(in: window).minY < start - 100, "the composer moved to the foot")
        let end = try #require(places.last ?? nil)
        #expect(places.allSatisfy { $0.map { abs($0 - end) <= 2 } ?? false }, "the button against the mark: \(places)")
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
