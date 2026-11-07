import SwiftUI

/// The main window: the Chat, as the switch at the toolbar's start picks, or the Server, which the
/// button at its end opens and the window opens on, as the app starts models.
struct MainView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        // The window works out its least size from its content: the frame's minimum alone, without
        // laying either screen out for it on every change.
        Offered {
            WithDrawer {
                // A change of what the server is at changes both screens as one: what comes and
                // goes with it fades, what is beside it moves along, in the Server's rows and bar
                // and the composer's controls alike. The toolbar and the drawer stay as they are.
                content.animation(Motion.settle, value: state.server.stage)
            }
        }
            .frame(minWidth: 800, minHeight: 560)
            .navigationTitle("Splash")
            .toolbar(removing: .title)
            // No band of its own behind the toolbar, which the window puts there once a transcript
            // scrolls under it, over the sidebar too: the sidebar runs up to the window's top,
            // and a transcript draws its own line under the toolbar.
            .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
            .toolbar {
                if state.available {
                    ToolbarItem(placement: .navigation) { SidebarButton() }
                        .sharedBackgroundVisibility(.hidden)
                    ToolbarItem(placement: .navigation) { SectionSwitch() }
                        .sharedBackgroundVisibility(.hidden)
                }
                // At the toolbar's end, clear of the sidebar: what the Chat makes, then the Server,
                // last in every part, where an item that came and went with the part would leave it
                // in place.
                ToolbarSpacer(.flexible)
                if state.available && state.section == .chat {
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
                if state.available {
                    ToolbarItem {
                        Button { state.toggleDrawer() } label: {
                            Label("Terminal and Logs", systemImage: "terminal")
                        }
                        .help(state.showsDrawer ? "Hide Terminal and Logs (⌘J)" : "Show Terminal and Logs (⌘J)")
                    }
                    ToolbarItem { ServerButton() }
                        .sharedBackgroundVisibility(.hidden)
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

/// The switch of the window's parts but the Server, which the button at the toolbar's end opens:
/// the Chat, not picked while the Server shows.
struct SectionSwitch: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Segments(selected: state.section, height: 28, pick: { state.section = $0 }, segments: [
            .init(title: "Chat", value: .chat, help: "Chat with the model that runs (⌘2)"),
        ])
    }
}

/// Segments in a well, the picked one on a pill that slides to the next picked: the window's
/// switch, and the drawer's. None is picked while `selected` is none of theirs.
struct Segments<Value: Hashable>: View {
    struct Segment {
        let title: String
        let value: Value
        var help = ""
    }

    let selected: Value
    var height: CGFloat = 24
    let pick: (Value) -> Void
    let segments: [Segment]
    @Environment(\.colorScheme) private var colorScheme
    @Namespace private var pill

    var body: some View {
        HStack(spacing: 2) {
            ForEach(segments, id: \.value) { segment in
                let picked = segment.value == selected
                Button { pick(segment.value) } label: {
                    Text(segment.title)
                        .font(height < 28 ? .subheadline.weight(.medium) : .callout.weight(.medium))
                        .foregroundStyle(picked ? .primary : .secondary)
                        .padding(.horizontal, height < 28 ? 11 : 14)
                        .frame(height: height)
                        .background {
                            if picked {
                                Capsule.pill(colorScheme)
                                    .matchedGeometryEffect(id: "pill", in: pill)
                            }
                        }
                        .contentShape(.capsule)
                }
                .buttonStyle(HoverStyle(padding: EdgeInsets(), shape: AnyShape(.capsule)))
                .help(segment.help)
                .accessibilityAddTraits(picked ? .isSelected : [])
            }
        }
        .padding(2)
        .background(Capsule.track(colorScheme))
        // The pill slides; what it picks changes at once.
        .animation(Motion.settle, value: selected)
    }
}

/// The Chat's sidebar beside its content, a column of the window's height, up through the toolbar,
/// that slides away as the toolbar's button and ⌘B hide it. Hidden, it keeps its place in the
/// window at no width, with what was typed in it and how far it scrolled; the keyboard leaves it.
struct WithSidebar<Side: View, Content: View>: View {
    static var width: CGFloat { 280 }

    @Environment(AppState.self) private var state
    @FocusState private var focused: Bool
    @ViewBuilder let side: Side
    @ViewBuilder let content: Content

    var body: some View {
        let shown = state.showsSidebar
        HStack(spacing: 0) {
            HStack(spacing: 0) {
                side.frame(width: Self.width)
                // Up through the toolbar, as far as the sidebar's background goes.
                Divider().ignoresSafeArea(edges: .top)
            }
            .focused($focused)
            // Slides out at its leading edge as its column narrows.
            .frame(width: shown ? nil : 0, alignment: .trailing)
            .clipped()
            .allowsHitTesting(shown)
            .accessibilityHidden(!shown)
            content
        }
        .onChange(of: shown) { if !shown { focused = false } }
    }
}

extension AppState {
    /// A terminal in the home folder.
    func newTerminal() {
        drawer.add(TerminalSession(shell: drawer.shell(), environment: runtime.userEnvironment))
    }

    /// Opens or closes the drawer, sliding; opened on the terminals, it starts one where there is
    /// none.
    func toggleDrawer() {
        withAnimation(Motion.settle) { showsDrawer.toggle() }
        if showsDrawer && drawer.part == .terminal && drawer.terminals.isEmpty { newTerminal() }
    }

    /// Opens the drawer on `part`.
    func showDrawer(_ part: Drawer.Part) {
        drawer.part = part
        if part == .terminal && drawer.terminals.isEmpty { newTerminal() }
        withAnimation(Motion.settle) { showsDrawer = true }
    }

    /// Opens the drawer on `log`.
    func showLog(_ log: LogSource) {
        drawer.log = log
        showDrawer(.logs)
    }

    /// Hides the sidebar shown, or shows the one hidden, sliding.
    func toggleSidebar() {
        withAnimation(Motion.settle) { showsSidebar.toggle() }
    }
}

/// Hides and shows the Chat's sidebar, as ⌘B does; the Server has none.
struct SidebarButton: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Button { state.toggleSidebar() } label: {
            Image(systemName: "sidebar.left")
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .frame(width: 32, height: 32)
                .contentShape(.circle)
        }
        .buttonStyle(HoverStyle(padding: EdgeInsets(), shape: AnyShape(.circle)))
        .accessibilityLabel(state.showsSidebar ? "Hide Sidebar" : "Show Sidebar")
        .help(state.showsSidebar ? "Hide Sidebar (⌘B)" : "Show Sidebar (⌘B)")
        .disabled(state.section == .server)
    }
}

extension AppState {
    /// The Server button's title: the model's name while one starts, runs or stops, the word
    /// Server while none does, Downloading while a download comes. It names the model without the
    /// variant that tells installed variants apart, which the button's help and the Server give,
    /// so the toolbar stays short. It holds no figure that changes
    /// while it shows, as a step, a speed or a percentage would: the toolbar lays its items out
    /// again at each change of its width, and moves the buttons beside it. The indicator shows a
    /// start and a download's progress; the button's help says the step and the percentage, the
    /// Server and the menu bar's panel the speed.
    var serverTitle: String {
        if installer.running { return "Downloading…" }
        guard let name = server.model.map(catalog.name(of:)) else { return "Server" }
        switch server.phase {
        case .starting, .ready, .stopping: return name
        case .stopped, .failed: return "Server"
        }
    }
}

/// The Server, at the toolbar's end: a light and the model that runs, or the word Server while
/// none does; a spinner while one starts, the download's ring while one comes. It opens the
/// Server, and shows picked while the Server shows, as the switch's segments do.
struct ServerButton: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        let selected = state.section == .server
        Button { state.section = .server } label: {
            HStack(spacing: 7) {
                indicator
                    .frame(width: 12, height: 12)
                Text(state.serverTitle)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 220)
            }
            .font(.callout.weight(.medium))
            .foregroundStyle(selected ? .primary : .secondary)
            .padding(.horizontal, 12)
            .frame(height: 28)
            .background {
                if selected { Capsule.pill(colorScheme) }
            }
            .padding(2)
            .background(Capsule.track(colorScheme))
            .contentShape(.capsule)
        }
        .buttonStyle(HoverStyle(padding: EdgeInsets(), shape: AnyShape(.capsule)))
        .animation(Motion.settle, value: selected)
        .help("\(state.downloadStatus ?? state.status) (⌘1)")
        .accessibilityLabel("Server")
        .accessibilityValue(state.downloadStatus ?? state.status)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    @ViewBuilder private var indicator: some View {
        if state.installer.running {
            if let progress = state.installer.progress, progress.total > 0 {
                Ring(fraction: progress.fraction, tint: Color.splash, size: 11)
            } else {
                ProgressView().controlSize(.mini)
            }
        } else {
            switch state.serverStatus.look {
            case .loading: ProgressView().controlSize(.mini)
            case .on: Light()
            case .asleep: Light(color: .splash.opacity(0.4))
            case .failed: Light(color: .red)
            case .off: Light(color: .secondary.opacity(0.5))
            }
        }
    }
}

extension Capsule {
    /// The well the switch's segments and the Server's button sit in.
    static func track(_ scheme: ColorScheme) -> some View {
        Capsule().fill(.primary.opacity(scheme == .dark ? 0.08 : 0.06))
    }

    /// The pill under what is picked.
    static func pill(_ scheme: ColorScheme) -> some View {
        Capsule()
            .fill(scheme == .dark ? AnyShapeStyle(.white.opacity(0.16)) : AnyShapeStyle(.white))
            .shadow(color: .black.opacity(0.12), radius: 1.5, y: 0.5)
    }
}

/// The Server button's light: the mark's color while a model runs, fainter while it sleeps, red
/// where it failed, gray while none runs.
struct Light: View {
    var color = Color.splash

    var body: some View {
        Circle().fill(color).frame(width: 6, height: 6)
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

    /// Memory in binary units, as Apple sells a Mac's and the memory limit counts it: 512 MB,
    /// 16 GB for 16 GiB, 17.2 GB.
    static func memory(_ bytes: Int64) -> String {
        let gigabytes = Double(bytes) / Double(1 << 30)
        guard gigabytes >= 1 else { return bytes.formatted(.byteCount(style: .memory)) }
        return "\(gigabytes.formatted(.number.precision(.fractionLength(0...1)))) GB"
    }

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

    /// A figure and its unit, as `size` and `memory` say it: 4.3 and GB.
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
