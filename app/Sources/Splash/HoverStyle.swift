import SwiftUI

/// The look of the app's own buttons: plain, with a rounded highlight under the pointer, a little
/// deeper while pressed or `held` (a control whose menu is open, the conversation open); faded
/// while disabled.
struct HoverStyle: ButtonStyle {
    /// Room between the label and the highlight's edge.
    var padding = EdgeInsets(top: 3, leading: 6, bottom: 3, trailing: 6)
    var shape = AnyShape(.rect(cornerRadius: 7))
    var held = false

    func makeBody(configuration: Configuration) -> some View {
        Highlight(configuration: configuration, padding: padding, shape: shape, held: held)
    }

    private struct Highlight: View {
        let configuration: ButtonStyleConfiguration
        let padding: EdgeInsets
        let shape: AnyShape
        let held: Bool
        @State private var hovering = false
        @Environment(\.isEnabled) private var enabled

        var body: some View {
            configuration.label
                .opacity(enabled ? 1 : 0.4)
                .padding(padding)
                .background(.quaternary.opacity(!enabled ? 0 : configuration.isPressed || held ? 1 : hovering ? 0.75 : 0), in: shape)
                .contentShape(shape)
                // At once, as AppKit's own controls light up: a fade would redraw the window in
                // each of its frames as the pointer crosses one row after another.
                .onHover { hovering = $0 }
        }
    }
}

extension ButtonStyle where Self == HoverStyle {
    static var hover: HoverStyle { HoverStyle() }
    /// The highlight on the label's own bounds, for a label that pads itself.
    static var hoverFlush: HoverStyle { HoverStyle(padding: EdgeInsets()) }
    /// The highlight across a row of a form, which keeps its text where the row's other text is.
    static var hoverRow: HoverStyle { HoverStyle(padding: EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8)) }
}

extension View {
    /// Undoes `hoverRow`'s room, so that the row's text stays in line with the form's.
    func hoverRowAligned() -> some View {
        padding(.horizontal, -8).padding(.vertical, -4)
    }
}

/// A button that copies, and says so: its label shows a check for 1.5 s, which another copy holds
/// longer, as the transcript page's copy buttons do.
struct CopyButton<Label: View>: View {
    let copy: () -> Void
    @ViewBuilder let label: (_ copied: Bool) -> Label
    @State private var copies = 0

    var body: some View {
        Button {
            copy()
            withAnimation(Motion.quick) { copies += 1 }
        } label: {
            label(copies > 0)
        }
        .task(id: copies) {
            guard copies > 0 else { return }
            try? await Task.sleep(for: .seconds(1.5))
            if !Task.isCancelled { withAnimation(Motion.quick) { copies = 0 } }
        }
    }
}

extension CopyButton where Label == CopyGlyph {
    /// With the copy symbol, which turns into a check.
    init(copy: @escaping () -> Void) {
        self.init(copy: copy) { CopyGlyph(copied: $0) }
    }
}

/// The mark of a control that opens a popup menu.
struct MenuChevron: View {
    var body: some View {
        Image(systemName: "chevron.up.chevron.down").font(.system(size: 9, weight: .semibold)).foregroundStyle(.secondary)
    }
}

extension NSPasteboard {
    /// Puts `string` on the general pasteboard, in place of what it held.
    static func put(_ string: String) {
        general.clearContents()
        general.setString(string, forType: .string)
    }
}

struct CopyGlyph: View {
    let copied: Bool

    var body: some View {
        Image(systemName: copied ? "checkmark" : "doc.on.doc").contentTransition(.symbolEffect(.replace))
    }
}

/// A row of a form that folds and unfolds the rows after it, its chevron turned while they show,
/// any detail after its title.
struct DisclosureRow: View {
    let title: String
    @Binding var open: Bool
    var detail: String?

    var body: some View {
        Button {
            withAnimation(Motion.settle) { open.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .rotationEffect(.degrees(open ? 90 : 0))
                    .accessibilityHidden(true)
                Text(title)
                if let detail {
                    Text(detail).font(.callout).foregroundStyle(.secondary).monospacedDigit().lineLimit(1)
                }
                Spacer()
            }
        }
        .buttonStyle(.hoverRow)
        .hoverRowAligned()
        .accessibilityValue(open ? "Expanded" : "Collapsed")
    }
}
