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
    /// Shown faded, as a menu's item that can't be chosen now: the pointer and the arrow keys pass
    /// it by.
    var disabled = false
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
    /// The height its items scroll in, where they take more room than the panel has; nil where they
    /// all show.
    var height: CGFloat?
    let choose: (String) -> Void

    var body: some View {
        Group {
            if let height {
                // The item the arrow keys reach scrolls into view.
                ScrollViewReader { proxy in
                    ScrollView { rows }
                        .frame(height: height)
                        .onChange(of: highlighted) { _, item in if let item { proxy.scrollTo(item) } }
                }
            } else {
                rows
            }
        }
        .padding(5)
        .frame(minWidth: 200, alignment: .leading)
        .fixedSize()
        .background(Self.ground, in: .rect(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(.separator) }
    }

    private var rows: some View {
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
                    .id(item.id)
            }
        }
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
        .disabled(item.disabled)
        // As `HoverStyle` fades a button disabled.
        .opacity(item.disabled ? 0.4 : 1)
        .onHover { inside in
            if inside, !item.disabled { highlighted = item.id } else if highlighted == item.id { highlighted = nil }
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
        var list = PopupList(header: header, items: items, chosen: chosen,
                             highlighted: Binding { self.highlighted } set: { self.highlighted = $0 }) { [weak self] in self?.pick($0) }
        var host = NSHostingView(rootView: list)
        var size = host.fittingSize
        let control = window.convertToScreen(anchor.convert(anchor.bounds, to: nil))
        let screen = window.screen?.visibleFrame ?? .infinite
        // Taller than the room on the control's roomier side, or than some twelve items, its items
        // scroll in that height, less the panel's padding.
        let room = min(max(control.minY - screen.minY, screen.maxY - control.maxY) - 14, 400)
        if size.height > room {
            list.height = room - 10
            host = NSHostingView(rootView: list)
            size = host.fittingSize
        }
        let under = control.minY - 6 - size.height, over = control.maxY + 6
        let roomUnder = under >= screen.minY + 8, roomOver = over + size.height <= screen.maxY - 8
        let y = above ? (roomOver || !roomUnder ? over : under) : (roomUnder || !roomOver ? under : over)
        // Toward the window's middle: from the control's leading edge in the window's left half,
        // from its trailing edge in its right half.
        let x = control.midX < window.frame.midX ? control.minX : control.maxX - size.width
        let origin = NSPoint(x: min(max(x, screen.minX + 8), screen.maxX - size.width - 8),
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

    /// Moves the highlight `step` items on, of those that can be chosen, starting at the chosen item
    /// when none is highlighted.
    private func move(_ step: Int) {
        let items = self.items.filter { !$0.disabled }
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

/// A view to scroll the scroll view it lies in to, brought to the top while what is under it
/// unfolds, as far as SwiftUI has drawn the unfolding. `pushed` is a view SwiftUI moves down with
/// the unfolding's end, frame by frame; `top`, a view that stays, and `height` under it are where
/// the layout puts that end. At each display frame the scroll has gone the share of its way that
/// `pushed` has gone of its own. What is under the unfolded rows moves down with `pushed`, by their
/// whole height, more than the way to the top from the page's end, so it is never drawn above
/// where it was, whenever a frame comes.
///
/// The display link ends as `pushed` reaches the layout's place for it; as `pushed` has stood still
/// for as long as an unfolding takes (`Motion.settleSpring`), wherever it stopped; as the reader
/// scrolls; as the views go; and as what unfolds folds again (`stop`).
@MainActor
final class ScrollTarget: NSObject {
    weak var view: NSView?
    weak var pushed: NSView?
    weak var top: NSView?
    /// How tall the layout makes what unfolds, from `top` to `pushed`, which SwiftUI's geometry gives
    /// at once while the views move toward it frame by frame. The document takes its new height at
    /// once too, before the change's `onChange` runs: its growth can't tell how far it has got.
    var height: CGFloat = 0
    private(set) var link: CADisplayLink?
    /// Where the content and `pushed` were as the unfolding began.
    private var start: (y: CGFloat, pushed: CGFloat)?
    /// Where it put the content last, to tell the reader's scrolling, and where `pushed` was then.
    private var placed: (y: CGFloat, pushed: CGFloat)?
    /// How long `pushed` has stood still, in the frames the link has called `step` for: a main
    /// thread too busy to draw holds SwiftUI's frames and these alike, and its wait counts as one.
    private var still: TimeInterval = 0

    /// Puts the view at the top of its scroll view, as far as the content goes, as what is under it
    /// unfolds.
    func scrollToTop() {
        guard let pushed, let scrollView = pushed.enclosingScrollView, let document = scrollView.documentView else { return }
        let at = pushed.convert(pushed.bounds, to: document).maxY
        start = (scrollView.contentView.bounds.minY, at)
        // As if placed there: the reader's scrolling from the click on, as momentum still running,
        // takes over before the scroll's first frame too.
        placed = start
        still = 0
        link?.invalidate()
        link = pushed.displayLink(target: self, selector: #selector(step))
        link?.add(to: .main, forMode: .common)
    }

    /// Leaves the content where it is.
    func stop() {
        link?.invalidate()
        link = nil
    }

    @objc private func step(_ link: CADisplayLink) {
        guard let view, let pushed, let top, let start, let scrollView = view.enclosingScrollView, let document = scrollView.documentView
        else { return stop() }
        let clip = scrollView.contentView
        let y = clip.bounds.minY
        // Moved by someone else since: the reader scrolls.
        if let placed, abs(y - placed.y) > 0.5 { return stop() }
        let at = pushed.convert(pushed.bounds, to: document).maxY, foot = top.convert(top.bounds, to: document).minY + height
        // Not moved yet, or not since the last frame: SwiftUI draws the unfolding's frames on its
        // own, the first some frames after the click on a busy main thread.
        guard at != start.pushed, at != placed?.pushed, foot > start.pushed else {
            still += link.targetTimestamp - link.timestamp
            if still >= Motion.settleSpring.settlingDuration { stop() }
            return
        }
        still = 0
        // 12 points below the toolbar the content scrolls under, as far as its inset goes, in the
        // document's flipped coordinates (SwiftUI's), as `at` and `foot` are.
        let frame = view.convert(view.bounds, to: document), inset = scrollView.contentInsets.top
        let goal = frame.minY - 12 - inset
        let lowest = -inset
        let reach = min(max(goal, lowest), max(document.frame.height - clip.bounds.height + scrollView.contentInsets.bottom, lowest))
        let share = min(max((at - start.pushed) / (foot - start.pushed), 0), 1)
        clip.setBoundsOrigin(NSPoint(x: clip.bounds.minX, y: start.y + (reach - start.y) * share))
        scrollView.reflectScrolledClipView(clip)
        placed = (clip.bounds.minY, at)
        if at >= foot { stop() }
    }
}
