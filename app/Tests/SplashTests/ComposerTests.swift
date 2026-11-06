import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Splash

/// The composer as the input method and the chat around it see it.
@MainActor
@Suite(.serialized)
struct ComposerTests {
    /// SwiftUI's updates of the composer, as each piece of a streamed reply makes, leave what the
    /// input method composes alone; the draft gets the text once it is committed.
    @Test func theComposerKeepsWhatTheInputMethodComposes() async throws {
        let (state, window, folder) = try chat()
        defer { window.orderOut(nil); try? FileManager.default.removeItem(at: folder) }
        let composer = try await focusedComposer(in: window)
        composer.insertText("你好", replacementRange: NSRange(location: NSNotFound, length: 0))
        Self.update(window)
        composer.setMarkedText("zai", selectedRange: NSRange(location: 3, length: 0),
                               replacementRange: NSRange(location: NSNotFound, length: 0))
        // Something the composer shows changes meanwhile.
        state.draft.focus()
        Self.update(window)
        #expect(composer.hasMarkedText())
        #expect(composer.string == "你好zai")
        #expect(state.draft.text == "你好")
        composer.insertText("再次", replacementRange: NSRange(location: NSNotFound, length: 0))
        #expect(state.draft.text == "你好再次")
    }

    /// A new chat, and a chat's first message, keep the composer's text view, and the keyboard in it.
    @Test func aNewChatKeepsTheComposer() async throws {
        let (state, window, folder) = try chat()
        defer { window.orderOut(nil); try? FileManager.default.removeItem(at: folder) }
        let composer = try await focusedComposer(in: window)
        let conversation = try #require(state.chats.selection)
        state.chats.newChat()
        state.draft.focus()
        Self.update(window)
        #expect(Self.composer(in: window) === composer)
        try await waitFor(window.firstResponder === composer)
        state.chats.selection = conversation
        Self.update(window)
        #expect(Self.composer(in: window) === composer)
        try await waitFor(window.firstResponder === composer)
    }

    /// A new chat moves the composer from the conversation's foot to the middle in one animation:
    /// its text view travels with its controls, rather than being there at once; with Reduce
    /// Motion it is there at once.
    @Test func aNewChatMovesTheWholeComposer() async throws {
        let (state, window, folder) = try chat()
        defer { window.orderOut(nil); try? FileManager.default.removeItem(at: folder) }
        let field = try #require(try await focusedComposer(in: window).enclosingScrollView)
        func height() -> CGFloat { field.convert(field.bounds, to: nil).minY }
        Self.update(window)
        let foot = height()
        state.chats.newChat()
        Self.update(window)
        let start = height()
        try await Task.sleep(for: .seconds(1))
        let middle = height()
        #expect(middle > foot + 100)
        if Motion.reduced {
            #expect(start == middle)
        } else {
            #expect(start < middle - 50)
        }
    }

    /// A photo attaches without holding the main thread while it is scaled and encoded; images
    /// join the draft in the order they were attached, a photo before a small image attached after
    /// it, which is encoded sooner.
    @Test func imagesAttachAwayFromTheMainThreadInTheirOrder() async throws {
        let draft = Draft()
        let photo = try #require(NSImage(data: Self.photo(width: 4032, height: 3024)))
        let picture = try #require(NSImage(data: Showcase.picture))
        draft.attach([photo])
        draft.attach([picture])
        #expect(draft.images.isEmpty)
        try await waitFor(draft.images.count == 2)
        let widths = draft.images.map { NSBitmapImageRep(data: $0)?.pixelsWide }
        #expect(widths == [2048, 240])
    }

    /// A new line grows the field in the layout that follows it, before the window shows it: its
    /// text doesn't scroll up a line for a frame and back. Text cleared elsewhere, as Send does,
    /// takes it back to one line, a narrower field wraps and grows, and it grows to 220 points at
    /// most, past which the text scrolls.
    @Test func theFieldIsAsTallAsItsTextWhenItShows() async throws {
        let (state, window, folder) = try chat()
        defer { window.orderOut(nil); try? FileManager.default.removeItem(at: folder) }
        let composer = try await focusedComposer(in: window)
        let scroll = try #require(composer.enclosingScrollView)
        let nowhere = NSRange(location: NSNotFound, length: 0)
        Self.update(window)
        let line = scroll.frame.height
        composer.insertText("one", replacementRange: nowhere)
        // Shift-Return.
        composer.insertNewlineIgnoringFieldEditor(nil)
        composer.insertText("two", replacementRange: nowhere)
        Self.update(window)
        #expect(scroll.frame.height > line + 10)
        #expect(scroll.contentView.bounds.minY == 0)

        state.draft.text = ""
        Self.update(window)
        #expect(scroll.frame.height == line)

        composer.insertText(String(repeating: "word ", count: 30), replacementRange: nowhere)
        Self.update(window)
        let wide = scroll.frame.height
        window.setContentSize(CGSize(width: 600, height: 600))
        Self.update(window)
        #expect(scroll.frame.height > wide)

        composer.insertText(String(repeating: "\nline", count: 20), replacementRange: nowhere)
        Self.update(window)
        #expect(scroll.frame.height == 220)
        // The caret's line in view, at the text's end.
        #expect(composer.frame.maxY - scroll.contentView.bounds.maxY < 5)
    }

    /// The ring of the composer's context and the Server button's download: its fraction filled in
    /// its tint from the top clockwise, the rest a faint track, as wide as asked.
    @Test func aRingFillsItsFractionClockwiseFromTheTop() throws {
        let size: CGFloat = 20, scale: CGFloat = 4
        let renderer = ImageRenderer(content: Ring(fraction: 0.25, tint: Color.red, size: size))
        renderer.scale = scale
        let bitmap = NSBitmapImageRep(cgImage: try #require(renderer.cgImage))
        #expect(bitmap.pixelsWide == Int(size * scale) && bitmap.pixelsHigh == Int(size * scale))
        // Whether the ring is in its tint `degrees` clockwise from the top, within its stroke.
        func tinted(_ degrees: Double) -> Bool {
            let center = size * scale / 2, radius = center - scale / 2, angle = degrees * .pi / 180
            let x = Int(center + radius * sin(angle)), y = Int(center - radius * cos(angle))
            guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { return false }
            return color.alphaComponent > 0.5 && color.redComponent > 0.8 && color.greenComponent < 0.5
        }
        #expect(tinted(20) && tinted(45) && tinted(70))
        #expect(!tinted(135) && !tinted(225) && !tinted(315))
    }

    /// A window showing the Chat of a conversation with one exchange.
    private func chat() throws -> (AppState, NSWindow, URL) {
        let folder = FileManager.default.temporaryDirectory.appending(path: "splash-composer-\(UUID().uuidString)")
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
        return (state, window, folder)
    }

    private func focusedComposer(in window: NSWindow) async throws -> NSTextView {
        try await waitFor(Self.composer(in: window) != nil)
        let composer = try #require(Self.composer(in: window))
        try await waitFor(window.firstResponder === composer)
        return composer
    }

    private static func composer(in window: NSWindow) -> NSTextView? {
        func textViews(in view: NSView) -> [NSTextView] {
            view.subviews.flatMap { ($0 as? NSTextView).map { [$0] } ?? textViews(in: $0) }
        }
        return window.contentView.flatMap { textViews(in: $0).first { !$0.isFieldEditor } }
    }

    /// A photo as a camera takes one, `width` by `height` pixels, as JPEG.
    private static func photo(width: Int, height: Int) -> Data {
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
                                      samplesPerPixel: 3, hasAlpha: false, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        let pixels = bitmap.bitmapData!
        var noise: UInt32 = 1
        for y in 0..<height {
            for x in 0..<width {
                noise = noise &* 1_664_525 &+ 1_013_904_223
                let grain = Int(noise >> 28), at = y * bitmap.bytesPerRow + x * 3
                pixels[at] = UInt8((x * 255 / width + grain) & 255)
                pixels[at + 1] = UInt8((y * 255 / height + grain) & 255)
                pixels[at + 2] = UInt8(((x + y) * 255 / (width + height) + grain) & 255)
            }
        }
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])!
    }

    /// What SwiftUI owes the window, done now: its views updated for every change made so far.
    private static func update(_ window: NSWindow) {
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
    }
}
