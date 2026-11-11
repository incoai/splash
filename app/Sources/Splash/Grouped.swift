import SwiftUI

/// The measures of a page of grouped cards, as a grouped form keeps them.
enum Grouped {
    /// Around the cards, in the column.
    static let margin: CGFloat = 20
    /// Between one section's end and the next one's header.
    static let spacing: CGFloat = 20
    /// From a card's edges to its rows' content and separators, and its header's and footer's.
    static let inset: CGFloat = 10
    /// Above and below a row's content.
    static let rowPadding: CGFloat = 10
    /// The line of a row's labeled control: a control taller than its label reaches into the row's
    /// padding.
    static let line: CGFloat = 20
    static let corner: CGFloat = 11
}

/// A page of sections of rows in cards, as a grouped form draws them, laid out whole, once, in a
/// scroll view of AppKit's (`AppKitScroll`). A form builds the rows that come into view as it
/// scrolls, and estimates the height of those it hasn't built: a scroll then works through much of
/// each frame, and the content moves as an estimate comes true.
///
/// Its labeled controls and toggles lay out as a grouped form lays them out. The sections are a
/// SwiftUI hierarchy of their own, with the app's state: what they show and how it animates is up to
/// them, as a view of their own. The page ends at the toolbar, under its line.
struct GroupedPage<Content: View>: View {
    /// The column's width, the cards inset in it.
    let column: CGFloat
    @ViewBuilder let content: Content

    @Environment(AppState.self) private var state

    var body: some View {
        AppKitScroll {
            VStack(alignment: .leading, spacing: Grouped.spacing) { content }
                .padding(Grouped.margin)
                .frame(maxWidth: column)
                .labeledContentStyle(GroupedLabeledContent())
                .toggleStyle(GroupedSwitch())
                .environment(state)
        }
        .toolbarLine()
    }
}

/// A section of a `GroupedPage`: its header, its rows in a card, a line between each two, and its
/// footer; nothing where all of them are empty.
struct GroupedSection<Content: View, Header: View, Footer: View>: View {
    @ViewBuilder let content: Content
    @ViewBuilder let header: Header
    @ViewBuilder let footer: Footer

    init(@ViewBuilder content: () -> Content, @ViewBuilder header: () -> Header, @ViewBuilder footer: () -> Footer) {
        self.content = content()
        self.header = header()
        self.footer = footer()
    }

    var body: some View {
        Group(subviews: content) { rows in
            Group(subviews: header) { headers in
                Group(subviews: footer) { footers in
                    // Its parts come and go as its rows do (`Motion.comingAndGoing`): a card that
                    // came with its first row, as a download's under the first run's welcome, faded
                    // in with it over the rows still moving down to make room for it, and a footer
                    // that went faded out over the rows moving up into its place.
                    if !rows.isEmpty || !headers.isEmpty || !footers.isEmpty {
                        VStack(alignment: .leading, spacing: 0) {
                            if !headers.isEmpty {
                                VStack(alignment: .leading, spacing: 0) { ForEach(headers) { $0 } }
                                    .padding(.horizontal, Grouped.inset)
                                    .padding(.bottom, Grouped.inset)
                                    .transition(Motion.comingAndGoing)
                            }
                            if !rows.isEmpty {
                                card(rows)
                                    .transition(Motion.comingAndGoing)
                            }
                            if !footers.isEmpty {
                                VStack(alignment: .leading, spacing: 0) { ForEach(footers) { $0 } }
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal, Grouped.inset)
                                    .padding(.vertical, Grouped.inset)
                                    .transition(Motion.comingAndGoing)
                            }
                        }
                        .transition(Motion.comingAndGoing)
                    }
                }
            }
        }
    }

    private func card(_ rows: SubviewsCollection) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(rows) { row in
                Group {
                    row
                        .padding(.horizontal, Grouped.inset)
                        .padding(.vertical, Grouped.rowPadding)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if row.id != rows.last?.id {
                        Rectangle()
                            .fill(.quaternary.opacity(0.5))
                            .frame(height: 1)
                            .padding(.horizontal, Grouped.inset)
                    }
                }
                .transition(Motion.comingAndGoing)
            }
        }
        // A grouped form's card, a shade off the window in either appearance.
        .background(Color(nsColor: .quaternarySystemFill), in: .rect(cornerRadius: Grouped.corner))
    }
}

extension GroupedSection where Footer == EmptyView {
    init(@ViewBuilder content: () -> Content, @ViewBuilder header: () -> Header) {
        self.init(content: content, header: header) { EmptyView() }
    }
}

extension GroupedSection where Header == EmptyView, Footer == EmptyView {
    init(@ViewBuilder content: () -> Content) {
        self.init(content: content, header: { EmptyView() }) { EmptyView() }
    }
}

/// A labeled control as a grouped form lays one out: the label at the row's start, the control at its
/// end in a control's line (`Grouped.line`), their first lines of text on one line.
struct GroupedLabeledContent: LabeledContentStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .firstTextBaseline) {
            configuration.label
            Spacer(minLength: 16)
            // Moving with its row as one (`geometryGroup`): a button's title, redrawn as it was
            // enabled, as the suggestions' Download buttons are as a first download ends, left its
            // bezel for where the row ends up as the row moved down.
            configuration.content.frame(height: Grouped.line).geometryGroup()
        }
    }
}

/// A toggle as a grouped form shows one: its label at the row's start, a mini switch at its end.
struct GroupedSwitch: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack {
            configuration.label
            Spacer(minLength: 16)
            Toggle("", isOn: configuration.$isOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.mini)
                // In a control's line, as a labeled control's is, not at the height the switch of
                // AppKit's gives: switches coming in with Advanced, asked theirs as they were
                // made, laid the card out a row short for the frame after the click, the
                // sections under it jumping up by as much.
                .frame(height: Grouped.line)
        }
        .accessibilityRepresentation { Toggle(isOn: configuration.$isOn) { configuration.label } }
    }
}
