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

    /// The content at the scroll view's width and as tall as it is, which it says as it changes,
    /// held to the document's top. The document takes a new height of the content's only after the
    /// update that changed it, and a hosting view centers a root view of another height than its
    /// own: a change of the content's height moved the whole page by half of it and slid it back as
    /// the document caught up, as picking a model whose family uses other launch options did. So the
    /// root view is the document's height whatever the content's (a least height of 0, as one of
    /// the content's own would take a taller content's), the content at its top, past its bottom
    /// until the document grows.
    struct Measured: View {
        let content: Content
        let height: (CGFloat) -> Void

        var body: some View {
            content
                .fixedSize(horizontal: false, vertical: true)
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height($0) }
                .frame(maxWidth: .infinity, minHeight: 0, maxHeight: .infinity, alignment: .top)
        }
    }

    @MainActor
    final class Coordinator: NSObject {
        weak var scroll: NSScrollView?
        /// The scroll to the content's new end under way (`measured`): from where, to where, the
        /// content's height, when it began, and where it put the clip last, to tell the reader's
        /// scrolling.
        private var glide: (from: CGFloat, to: CGFloat, height: CGFloat, began: CFTimeInterval, placed: CGFloat?)?
        private var link: CADisplayLink?

        /// Makes the document as tall as its content. A content that gets shorter than where the
        /// page is scrolled to, as a notice going above while the page shows its end, moves up
        /// within `settle`, and the page scrolls to its new end with it, in step: the scroll view
        /// clamped its scroll at once, and the page jumped down by what went and slid back up.
        /// Its width changing, as the window's does, the content changes at once, and so does the
        /// scroll.
        func measured(_ height: CGFloat) {
            guard let scroll, let document = scroll.documentView else { return }
            let size = NSSize(width: scroll.contentView.bounds.width, height: max(1, height.rounded(.up)))
            guard document.frame.size != size else { return }
            let clip = scroll.contentView
            let end = max(size.height - clip.bounds.height + scroll.contentInsets.bottom, -scroll.contentInsets.top)
            if !Motion.reduced, size.width == document.frame.width, clip.bounds.minY > end + 0.5 {
                glide = (clip.bounds.minY, end, size.height, CACurrentMediaTime(), nil)
                if link == nil {
                    link = scroll.displayLink(target: self, selector: #selector(step))
                    link?.add(to: .main, forMode: .common)
                }
                return
            }
            stop()
            document.setFrameSize(size)
        }

        @objc private func step(_ link: CADisplayLink) {
            guard let scroll, let document = scroll.documentView, let glide else { return stop() }
            let clip = scroll.contentView
            // Scrolled by the reader since: the content takes its height, as far as it goes.
            if let placed = glide.placed, abs(clip.bounds.minY - placed) > 0.5 {
                document.setFrameSize(NSSize(width: document.frame.width, height: glide.height))
                return stop()
            }
            let time = link.targetTimestamp - glide.began
            let done = time >= Motion.settleSeconds * 2
            let y = done ? glide.to : glide.from + (glide.to - glide.from) * Motion.settleSpring.value(target: 1.0, time: time)
            // The document keeps room for the clip until the scroll ends.
            let visible = clip.bounds.height - scroll.contentInsets.bottom
            document.setFrameSize(NSSize(width: document.frame.width, height: done ? glide.height : max(glide.height, (y + visible).rounded(.up))))
            clip.setBoundsOrigin(NSPoint(x: clip.bounds.minX, y: y))
            scroll.reflectScrolledClipView(clip)
            self.glide?.placed = clip.bounds.minY
            if done { stop() }
        }

        private func stop() {
            link?.invalidate()
            link = nil
            glide = nil
        }

        /// Ends the content `bottom` above the scroll view's bottom edge.
        func inset(_ bottom: CGFloat) {
            guard let scroll, scroll.contentInsets.bottom != bottom else { return }
            scroll.contentInsets = NSEdgeInsets(top: 0, left: 0, bottom: bottom, right: 0)
        }
    }
}
