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
        let state = AppState(runtime: Showcase.runtime, chats: ChatStore(folder: folder), keychain: Keychain(service: "Splash Tests"))
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

    /// What SwiftUI owes the window, done now: its views updated for every change made so far.
    private static func update(_ window: NSWindow) {
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
    }

    private func waitFor(_ condition: @autoclosure () -> Bool, seconds: Double = 10) async throws {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !condition() {
            try #require(ContinuousClock.now < deadline, "timed out")
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
