import AppKit
import SwiftUI

/// SwiftUI content in a scroll view of AppKit's, which moves it on a thread of its own as it
/// scrolls (responsive scrolling), as a browser moves a page: a scroll keeps to the display's rate
/// whatever the main thread does meanwhile. SwiftUI lays the content out whole, once. A scroll view
/// of SwiftUI's moves its content in SwiftUI's update of each frame, on the main thread, with the
/// views of AppKit's in it (text fields, switches, menus); on the Server screen that update ran past
/// a frame at 120 Hz often enough that the screen scrolled at 60.
///
/// The content starts at the top and scrolls under a bar at the bottom, the safe area there, but not
/// under the toolbar, the safe area at the top: the toolbar has no band of its own (`MainView`), and
/// what scrolled under it would read through its controls. It is a SwiftUI hierarchy of its own: it
/// takes no environment and no `animation` modifier from around it.
struct AppKitScroll<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        GeometryReader { proxy in
            Scroller(bottom: proxy.safeAreaInsets.bottom, content: content)
                .ignoresSafeArea(edges: .bottom)
        }
    }
}

/// The scroll view of an `AppKitScroll`, its document the content's hosting view.
private struct Scroller<Content: View>: NSViewRepresentable {
    /// The safe area's height at the bottom, which the content ends above.
    let bottom: CGFloat
    let content: Content

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = FocusScroll()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.horizontalScrollElasticity = .none
        scroll.automaticallyAdjustsContentInsets = false
        let document = Document(rootView: Measured(content: content, height: context.coordinator.measured))
        document.sizingOptions = []
        document.autoresizingMask = [.width]
        document.frame = NSRect(x: 0, y: 0, width: scroll.contentSize.width, height: 1)
        scroll.documentView = document
        context.coordinator.scroll = scroll
        context.coordinator.inset(bottom)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        (scroll.documentView as? Document)?.rootView = Measured(content: content, height: context.coordinator.measured)
        context.coordinator.inset(bottom)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    /// The scroll view, which keeps what has the keyboard in view, as a form does: a text field
    /// tabbed to below what shows scrolls up into it. SwiftUI's focus scrolls only its own scroll
    /// views.
    final class FocusScroll: NSScrollView {
        private var watch: NSKeyValueObservation?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            watch = window?.observe(\.firstResponder, options: [.new]) { [weak self] window, _ in
                MainActor.assumeIsolated { self?.reveal(window.firstResponder) }
            }
        }

        private func reveal(_ responder: NSResponder?) {
            guard let document = documentView, var view = responder as? NSView, view !== document,
                  view.isDescendant(of: document) else { return }
            // A text field's editor stands in for the field.
            if let editor = view as? NSTextView, editor.isFieldEditor, let field = editor.delegate as? NSView { view = field }
            document.scrollToVisible(view.convert(view.bounds, to: document).insetBy(dx: 0, dy: -Grouped.rowPadding))
        }
    }

    /// The content's hosting view, which AppKit moves on its scrolling thread. A hosting view takes
    /// scroll-wheel events, which leaves it out of responsive scrolling unless it says it works with
    /// it: nothing in the content follows a scroll's events.
    final class Document: NSHostingView<Measured> {
        override class var isCompatibleWithResponsiveScrolling: Bool { true }
    }

    /// The content at the scroll view's width and as tall as it is, which it says as it changes.
    struct Measured: View {
        let content: Content
        let height: (CGFloat) -> Void

        var body: some View {
            content
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .top)
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height($0) }
        }
    }

    @MainActor
    final class Coordinator {
        weak var scroll: NSScrollView?

        /// Makes the document as tall as its content.
        func measured(_ height: CGFloat) {
            guard let scroll, let document = scroll.documentView else { return }
            let size = NSSize(width: scroll.contentView.bounds.width, height: max(1, height.rounded(.up)))
            if document.frame.size != size { document.setFrameSize(size) }
        }

        /// Ends the content `bottom` above the scroll view's bottom edge.
        func inset(_ bottom: CGFloat) {
            guard let scroll, scroll.contentInsets.bottom != bottom else { return }
            scroll.contentInsets = NSEdgeInsets(top: 0, left: 0, bottom: bottom, right: 0)
        }
    }
}
