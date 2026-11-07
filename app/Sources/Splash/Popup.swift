import AppKit
import SwiftUI

/// A choice of a `PopupMenu`.
struct PopupItem: Identifiable, Hashable {
    let id: String
    let title: String
    /// A word beside the title, such as Default.
    var badge: String?
    /// Set apart from the items before it by a line.
    var separated = false
}

/// A menu in Splash's own style: a rounded panel under the control that opens it, or over it,
/// with any header and its items, each a title and any badge, the chosen one checked. It opens at
/// once, its control held while it is open. The arrow keys move through it, Return or Space picks;
/// a choice, a click or a scroll elsewhere, Escape or any other key, or its control going away
/// closes it, fading out, a choice still highlighted.
struct PopupMenu<Label: View>: View {
    var header: String?
    let items: [PopupItem]
    let chosen: String?
    var above = false
    let choose: (String) -> Void
    @ViewBuilder let label: () -> Label
    @State private var popup = Popup()

    var body: some View {
        Button {
            popup.toggle(header: header, items: items, chosen: chosen, above: above, choose: choose)
        } label: {
            label()
        }
        .buttonStyle(HoverStyle(padding: EdgeInsets(), held: popup.isOpen))
        .background(ViewAnchor { popup.anchor = $0 })
        .onDisappear { popup.close() }
    }
}

/// The panel of a `PopupMenu`, apart from the control, the item under the pointer or the arrow
/// keys highlighted.
struct PopupList: View {
    let header: String?
    let items: [PopupItem]
    let chosen: String?
    @Binding var highlighted: String?
    let choose: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            if let header {
                Text(header)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.top, 4)
                    .padding(.bottom, 3)
            }
            ForEach(items) { item in
                if item.separated {
                    Divider().padding(.horizontal, 8).padding(.vertical, 4)
                }
                PopupRow(item: item, chosen: item.id == chosen, highlighted: $highlighted) { choose(item.id) }
            }
        }
        .padding(5)
        .frame(minWidth: 200, alignment: .leading)
        .fixedSize()
        .background(Self.ground, in: .rect(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(.separator) }
    }

    /// What the panel stands on: white, or near black in dark mode, solid.
    private static let ground = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? NSColor(white: 0.17, alpha: 1) : .white
    })
}

/// An item, highlighted as `HoverStyle` highlights, but one at a time, for the pointer and the
/// arrow keys alike.
private struct PopupRow: View {
    let item: PopupItem
    let chosen: Bool
    @Binding var highlighted: String?
    let choose: () -> Void

    var body: some View {
        Button(action: choose) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                HStack(spacing: 8) {
                    Text(item.title)
                    if let badge = item.badge {
                        Text(badge)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(.quaternary, in: .rect(cornerRadius: 5))
                    }
                }
                Spacer(minLength: 16)
                Image(systemName: "checkmark")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.tint)
                    .opacity(chosen ? 1 : 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            // Concentric with the panel's corners.
            .background(.quaternary.opacity(highlighted == item.id ? 0.75 : 0), in: .rect(cornerRadius: 7))
            .contentShape(.rect(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .onHover { inside in
            if inside { highlighted = item.id } else if highlighted == item.id { highlighted = nil }
        }
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

/// The open panel of a `PopupMenu`: a borderless child window of the control's, at its trailing
/// edge, under it or over it as asked where the screen has room, else the other way. One is open at
/// a time.
@MainActor @Observable
final class Popup {
    @ObservationIgnored weak var anchor: NSView?
    /// The item the pointer or the arrow keys are on.
    var highlighted: String?
    private(set) var isOpen = false
    @ObservationIgnored private var panel: NSPanel?
    @ObservationIgnored private var items: [PopupItem] = []
    @ObservationIgnored private var chosen: String?
    @ObservationIgnored private var choose: (String) -> Void = { _ in }
    @ObservationIgnored private var monitors: [Any] = []
    @ObservationIgnored private var resigned: NSObjectProtocol?
    /// The popup open in the app.
    private static weak var open: Popup?

    func toggle(header: String?, items: [PopupItem], chosen: String?, above: Bool, choose: @escaping (String) -> Void) {
        if panel != nil { return close() }
        Self.open?.close()
        guard let anchor, let window = anchor.window else { return }
        (self.items, self.chosen, self.choose, highlighted) = (items, chosen, choose, nil)
        let list = PopupList(header: header, items: items, chosen: chosen,
                             highlighted: Binding { self.highlighted } set: { self.highlighted = $0 }) { [weak self] in self?.pick($0) }
        let host = NSHostingView(rootView: list)
        let size = host.fittingSize
        let control = window.convertToScreen(anchor.convert(anchor.bounds, to: nil))
        let screen = window.screen?.visibleFrame ?? .infinite
        let under = control.minY - 6 - size.height, over = control.maxY + 6
        let roomUnder = under >= screen.minY + 8, roomOver = over + size.height <= screen.maxY - 8
        let y = above ? (roomOver || !roomUnder ? over : under) : (roomUnder || !roomOver ? under : over)
        let origin = NSPoint(x: min(max(control.maxX - size.width, screen.minX + 8), screen.maxX - size.width - 8),
                             y: min(max(y, screen.minY + 8), screen.maxY - size.height - 8))
        let panel = NSPanel(contentRect: NSRect(origin: origin, size: size), styleMask: [.borderless, .nonactivatingPanel],
                            backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        // A hairline sets it off, no shadow.
        panel.hasShadow = false
        panel.contentView = host
        window.addChildWindow(panel, ordered: .above)
        self.panel = panel
        isOpen = true
        Self.open = self
        monitors = [
            NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown, .scrollWheel]) { [weak self] event in
                // A click on the control is its own: it closes the panel itself.
                if let self, event.window !== panel, event.type == .scrollWheel || !onAnchor(event) { close() }
                return event
            },
            NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self else { return event }
                switch event.keyCode {
                case 125: move(1)
                case 126: move(-1)
                case 36, 76, 49:  // Return, Enter, Space
                    if let highlighted { pick(highlighted) } else { close() }
                case 53: close()  // Escape
                default:
                    close()
                    return event
                }
                return nil
            },
        ].compactMap { $0 }
        resigned = NotificationCenter.default.addObserver(forName: NSApplication.didResignActiveNotification, object: nil,
                                                          queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.close() }
        }
    }

    /// Closes the panel, which fades out as a menu does, taking no more clicks meanwhile.
    func close() {
        monitors.forEach(NSEvent.removeMonitor)
        monitors = []
        resigned.map(NotificationCenter.default.removeObserver)
        resigned = nil
        if Self.open === self { Self.open = nil }
        guard let panel else { return }
        self.panel = nil
        isOpen = false
        panel.ignoresMouseEvents = true
        Task {
            await NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.12
                context.timingFunction = CAMediaTimingFunction(name: .easeIn)
                panel.animator().alphaValue = 0
            }
            panel.parent?.removeChildWindow(panel)
            panel.orderOut(nil)
        }
    }

    /// Takes the choice at once, under the panel fading out with it highlighted.
    private func pick(_ id: String) {
        highlighted = id
        let choose = choose
        close()
        choose(id)
    }

    /// Moves the highlight `step` items on, starting at the chosen item when none is highlighted.
    private func move(_ step: Int) {
        guard !items.isEmpty else { return }
        let index = if let current = items.firstIndex(where: { $0.id == highlighted }) {
            current + step
        } else {
            items.firstIndex { $0.id == chosen } ?? (step > 0 ? 0 : items.count - 1)
        }
        highlighted = items[min(max(index, 0), items.count - 1)].id
    }

    private func onAnchor(_ event: NSEvent) -> Bool {
        guard let anchor, event.window === anchor.window else { return false }
        return anchor.bounds.contains(anchor.convert(event.locationInWindow, from: nil))
    }
}

/// The AppKit view behind a SwiftUI view, handed to `found` as it is made and updated.
struct ViewAnchor: NSViewRepresentable {
    let found: (NSView) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        found(view)
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        found(view)
    }
}

/// A view to scroll the scroll view it lies in to, brought to the top while the content under it
/// unfolds: each frame it eases toward the top, as far as the content reaches by then, so that it
/// rises as the content grows rather than after. The reader's own scrolling takes over.
@MainActor
final class ScrollTarget: NSObject {
    weak var view: NSView?
    private var link: CADisplayLink?
    /// Where it is to end up, and where it put the content last, to tell the reader's scrolling.
    private var top: CGFloat = 0
    private var placed: CGFloat?
    private var began: CFTimeInterval = 0

    /// Puts the view at the top of its scroll view, as far as the content goes.
    func scrollToTop(margin: CGFloat = 12) {
        guard let view, let scrollView = view.enclosingScrollView, let document = scrollView.documentView else { return }
        // Below the toolbar the content scrolls under, as far as its inset goes.
        let frame = view.convert(view.bounds, to: document), inset = scrollView.contentInsets.top
        top = document.isFlipped ? frame.minY - margin - inset : frame.maxY + margin - scrollView.contentView.bounds.height + inset
        placed = nil
        began = CACurrentMediaTime()
        link?.invalidate()
        link = scrollView.displayLink(target: self, selector: #selector(step))
        link?.add(to: .main, forMode: .common)
    }

    @objc private func step(_ link: CADisplayLink) {
        guard let scrollView = view?.enclosingScrollView, let document = scrollView.documentView else { return stop() }
        let clip = scrollView.contentView
        let y = clip.bounds.minY
        // Moved by someone else since: the reader scrolls.
        if let placed, abs(y - placed) > 0.5 { return stop() }
        let lowest = -scrollView.contentInsets.top
        let reach = min(max(top, lowest), max(document.frame.height - clip.bounds.height + scrollView.contentInsets.bottom, lowest))
        // Motion.glide's travel: most of the way in 0.4 s; at once with Reduce Motion.
        let next = Motion.reduced ? reach : y + (reach - y) * (1 - exp(-(link.targetTimestamp - link.timestamp) / 0.08))
        clip.setBoundsOrigin(NSPoint(x: clip.bounds.minX, y: abs(reach - next) < 0.5 ? reach : next))
        scrollView.reflectScrolledClipView(clip)
        let now = clip.bounds.minY
        placed = now
        // There, or as far as the content goes once it has stopped growing.
        if abs(top - now) < 0.5 || CACurrentMediaTime() - began > 1 { stop() }
    }

    private func stop() {
        link?.invalidate()
        link = nil
    }
}
