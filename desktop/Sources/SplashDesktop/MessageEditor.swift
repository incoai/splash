import AppKit
import SwiftUI

/// NSTextView gives chat input native selection, composition and newline behavior.
struct MessageEditor: NSViewRepresentable {
    @Binding var text: String
    @Binding var height: CGFloat
    let focusRequest: Int
    let send: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        let editor = Editor()
        editor.delegate = context.coordinator
        editor.isRichText = false
        editor.importsGraphics = false
        editor.drawsBackground = false
        editor.font = .systemFont(ofSize: 14)
        editor.textColor = .labelColor
        editor.insertionPointColor = .labelColor
        editor.isAutomaticQuoteSubstitutionEnabled = false
        editor.isAutomaticDashSubstitutionEnabled = false
        editor.textContainerInset = .zero
        editor.textContainer?.lineFragmentPadding = 0
        editor.isHorizontallyResizable = false
        editor.isVerticallyResizable = true
        editor.autoresizingMask = [.width]
        editor.textContainer?.widthTracksTextView = true
        editor.minSize = NSSize(width: 0, height: 40)
        editor.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        editor.setAccessibilityLabel("Message Splash")
        scroll.documentView = editor
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let editor = scroll.documentView as? Editor else { return }
        context.coordinator.parent = self
        editor.send = send
        if editor.string != text {
            editor.string = text
            editor.setSelectedRange(NSRange(location: text.utf16.count, length: 0))
            editor.needsDisplay = true
        }
        context.coordinator.resize(editor)
        if context.coordinator.lastFocusRequest != focusRequest {
            context.coordinator.lastFocusRequest = focusRequest
            DispatchQueue.main.async { editor.window?.makeFirstResponder(editor) }
        }
    }

    @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: MessageEditor
        var lastFocusRequest = -1

        init(_ parent: MessageEditor) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard let editor = notification.object as? Editor else { return }
            parent.text = editor.string
            editor.needsDisplay = true
            resize(editor)
        }

        func resize(_ editor: NSTextView) {
            guard let layout = editor.layoutManager, let container = editor.textContainer else { return }
            layout.ensureLayout(for: container)
            let value = min(150, max(40, ceil(layout.usedRect(for: container).height) + 4))
            if abs(value - parent.height) >= 1 {
                DispatchQueue.main.async { [self] in parent.height = value }
            }
        }
    }

    final class Editor: NSTextView {
        var send: (() -> Void)?

        override func keyDown(with event: NSEvent) {
            if [36, 76].contains(event.keyCode), !hasMarkedText(), !event.modifierFlags.contains(.shift) {
                send?()
            } else {
                super.keyDown(with: event)
            }
        }

        override func draw(_ dirtyRect: NSRect) {
            super.draw(dirtyRect)
            if string.isEmpty {
                ("Message Splash…" as NSString).draw(at: NSPoint(x: 0, y: 0), withAttributes: [
                    .font: NSFont.systemFont(ofSize: 14), .foregroundColor: NSColor.placeholderTextColor,
                ])
            }
        }
    }
}
