import SwiftUI

/// The main window: the Server or the Chat, as the switch at the toolbar's start picks. Its toolbar
/// holds, besides, a download under way.
struct MainView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        // The window works out its least size from its content: the frame's minimum alone, without
        // laying either screen out for it on every change.
        Offered { content }
            .frame(minWidth: 800, minHeight: 560)
            .navigationTitle("Splash")
            .toolbar(removing: .title)
            .toolbar {
                if state.available {
                    ToolbarItem(placement: .navigation) { SectionSwitch() }
                        .sharedBackgroundVisibility(.hidden)
                }
                // In both sections: an item that came and went with the switch left its glass
                // behind, sliding under the switch.
                if state.installer.running {
                    ToolbarItem { DownloadButton() }
                }
                if state.available && state.section == .chat {
                    // At the toolbar's end, clear of the sidebar.
                    ToolbarSpacer(.flexible)
                    ToolbarItem {
                        Button {
                            state.chats.newChat()
                            state.draft.focus()
                        } label: {
                            Label("New Chat", systemImage: "square.and.pencil")
                        }
                        .help("New Chat (⌘N)")
                    }
                }
            }
            .onChange(of: state.status) { _, status in
                AccessibilityNotification.Announcement(status).post()
            }
            // The Chat's page loads with the window, so that the Chat shows at once.
            .onAppear { state.transcript.page.load() }
    }

    @ViewBuilder private var content: some View {
        if let blocker = state.blocker {
            ContentUnavailableView(blocker.title, systemImage: blocker.symbol, description: Text(blocker.message))
        } else if !state.launched {
            ProgressView().controlSize(.small).frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            switch state.section {
            case .server: ServerView()
            case .chat: ChatScreen()
            }
        }
    }
}

/// Takes the space it's offered without asking its content, as a shape does.
struct Offered: Layout {
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for subview in subviews {
            subview.place(at: bounds.origin, proposal: ProposedViewSize(bounds.size))
        }
    }

    // No alignment guides of its own either, which would otherwise be looked for in its content.
    func explicitAlignment(of guide: HorizontalAlignment, in bounds: CGRect, proposal: ProposedViewSize,
                           subviews: Subviews, cache: inout ()) -> CGFloat? { nil }

    func explicitAlignment(of guide: VerticalAlignment, in bounds: CGRect, proposal: ProposedViewSize,
                           subviews: Subviews, cache: inout ()) -> CGFloat? { nil }
}

/// The switch between the Server and the Chat.
struct SectionSwitch: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var colorScheme
    @Namespace private var pill

    var body: some View {
        HStack(spacing: 2) {
            segment("Server", .server, help: "Models and how they run (⌘1)")
            segment("Chat", .chat, help: "Chat with the model that runs (⌘2)")
        }
        .padding(2)
        .background(.primary.opacity(colorScheme == .dark ? 0.08 : 0.06), in: .capsule)
        // The pill slides; the window's content swaps at once.
        .animation(Motion.settle, value: state.section)
    }

    private func segment(_ title: String, _ section: AppState.Section, help: String) -> some View {
        let selected = state.section == section
        return Button { state.section = section } label: {
            Text(title)
                .font(.callout.weight(.medium))
                .foregroundStyle(selected ? .primary : .secondary)
                .padding(.horizontal, 14)
                .frame(height: 28)
                .background {
                    if selected {
                        Capsule()
                            .fill(colorScheme == .dark ? AnyShapeStyle(.white.opacity(0.16)) : AnyShapeStyle(.white))
                            .shadow(color: .black.opacity(0.12), radius: 1.5, y: 0.5)
                            .matchedGeometryEffect(id: "pill", in: pill)
                    }
                }
                .contentShape(.capsule)
        }
        .buttonStyle(HoverStyle(padding: EdgeInsets(), shape: AnyShape(.capsule)))
        .help(help)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// A download under way; opens the Server, which shows it.
struct DownloadButton: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Button { state.section = .server } label: {
            // It spins until the download has told its size, rather than show an empty ring.
            Group {
                if let progress = state.installer.progress, progress.total > 0 {
                    ProgressView(value: progress.fraction)
                } else {
                    ProgressView()
                }
            }
            .progressViewStyle(.circular)
            .controlSize(.small)
        }
        .help(state.downloadStatus ?? "")
        .accessibilityLabel("Download")
        .accessibilityValue(state.downloadStatus ?? "")
    }
}

/// A figure of the engine's: its value large, its unit beside it, what it is under it; a dash
/// before there is one. The value changes in place.
struct Figure: View {
    let value: String?
    var unit: String?
    let caption: String
    var compact = false
    var help = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value ?? "–")
                    .font(.system(size: compact ? 15 : 22, weight: .medium))
                    .foregroundStyle(value == nil ? .tertiary : .primary)
                    .monospacedDigit()
                    // In place: a rolling or fading figure is redrawn on the CPU in every frame
                    // of its animation, some 35 frames a poll.
                    .contentTransition(.identity)
                if let unit, value != nil {
                    Text(unit).font(compact ? .subheadline : .callout).foregroundStyle(.secondary)
                }
            }
            Text(caption).font(.subheadline).foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .help(help)
    }
}

enum Format {
    /// Decimal units, as Finder counts: 120 MB, 6.7 GB, 21 GB. Whole GB from 10 GB up, so a
    /// model's size reads the same everywhere.
    static func size(_ bytes: Int64) -> String {
        let gigabytes = Double(bytes) / 1e9
        guard gigabytes >= 1 else { return bytes.formatted(.byteCount(style: .file)) }
        return "\(gigabytes.formatted(.number.precision(.fractionLength(gigabytes < 10 ? 1 : 0)))) GB"
    }

    static func gigabytes(_ count: Int) -> Int64 { Int64(count) * 1_000_000_000 }

    /// A count in a few figures: 950, 12K, 3.4M.
    static func count(_ value: Int) -> String {
        value.formatted(.number.notation(.compactName))
    }

    /// A figure with no fraction: 37, 1,204.
    static func whole(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(0)))
    }

    /// A time to first token, and its unit: 136 ms, or 1.4 s.
    static func latency(_ duration: Duration) -> (String, String) {
        // Rounded first, so that 999.7 ms reads 1.0 s.
        let milliseconds = (Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15).rounded()
        return milliseconds < 1000 ? (whole(milliseconds), "ms")
            : ((milliseconds / 1000).formatted(.number.precision(.fractionLength(1))), "s")
    }

    /// A figure and its unit, as `size` says it: 4.3 and GB.
    static func parts(_ text: String) -> (String, String?) {
        guard let space = text.lastIndex(of: " ") else { return (text, nil) }
        return (String(text[..<space]), String(text[text.index(after: space)...]))
    }

    /// A while, as the chat says it: 9.8s, 24s, 1m 24s.
    static func seconds(_ seconds: Double) -> String {
        // Rounded first, so that 9.96 s reads 10s.
        let tenths = (seconds * 10).rounded() / 10
        if tenths < 10 { return "\(tenths.formatted(.number.precision(.fractionLength(1))))s" }
        let whole = Int(seconds.rounded())
        return whole < 60 ? "\(whole)s" : "\(whole / 60)m \(whole % 60)s"
    }
}
