import SwiftUI

/// The width of the composer: the transcript's column (`main`'s max-width in page.css), so
/// that the two line up.
private let column: CGFloat = 680

/// The conversations at the side of the open one, in a sidebar that hides (`WithSidebar`), and the
/// open one's title in the toolbar, which opens its menu. The window's toolbar is the Server's too,
/// so that the switch and the Server's button stay where they are. The list and the title ask the
/// same questions, to rename and to delete chats, which show here, whether the sidebar shows or not.
struct ChatScreen: View {
    @Environment(AppState.self) private var state
    /// The chat asked a new name for, and the name typed.
    @State private var renaming: Conversation?
    @State private var name = ""
    /// The question to delete chats, while it asks, as it was asked.
    @State private var deletion: ChatDeletion?
    /// The chats picked to delete while the list picks them, from a chat's Select; nil otherwise.
    @State private var picked: Set<UUID>?
    /// Where the toolbar puts the open chat's title in the window, and how wide the window is (`ChatTitle`), kept
    /// while a new chat has none, so that the next title starts in its place.
    @State private var titlePlace: TitlePlace?

    var body: some View {
        WithSidebar {
            ChatList(picked: $picked, deletion: $deletion, actions: actions)
        } content: {
            ConversationView()
        }
        // The composer takes the keyboard as the Chat shows, unless a terminal of the drawer shows
        // with it, which keeps it, as in a window that ⌘J opens.
        .onAppear { if !state.terminalShows { state.draft.focus() } }
        // The open chat's title, in the window's top row; a new chat, with nothing in it, has none.
        .toolbar {
            if let open = state.chats.current, !open.messages.isEmpty {
                ToolbarItem(placement: .navigation) {
                    ChatTitle(conversation: open, place: $titlePlace) { actions(for: open) }
                }
                // Plain text, as a window's title is, not a control on a band of glass.
                .sharedBackgroundVisibility(.hidden)
            }
        }
        .alert("Rename Chat", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } }
        ), presenting: renaming) { conversation in
            TextField("Name", text: $name)
            Button("Rename") { state.chats.rename(conversation.id, to: name) }
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            Button("Cancel", role: .cancel) {}
        }
        .confirmationDialog(deletion?.title ?? "", isPresented: Binding(
            get: { deletion != nil }, set: { if !$0 { deletion = nil } }
        ), presenting: deletion) { deletion in
            Button("Delete", role: .destructive) {
                state.chats.delete(deletion.ids)
                // Those picked, deleted, end the picking; another chat deleted leaves the rest.
                if picked == deletion.ids { picked = nil } else { picked?.subtract(deletion.ids) }
            }
        } message: { deletion in
            Text(deletion.message)
        }
    }

    /// What can be done with a chat, as its title in the window's top row, its row's menu and a right
    /// click on its row give it: pin it to the list's top, or take it off there; rename it; delete it.
    @ViewBuilder private func actions(for conversation: Conversation) -> some View {
        Button(conversation.pinned == nil ? "Pin" : "Unpin") { state.chats.pin(conversation.id, conversation.pinned == nil) }
        Button("Rename…") {
            name = conversation.title
            renaming = conversation
        }
        Button("Delete…", role: .destructive) { deletion = ChatDeletion([conversation]) }
    }
}

/// The open chat's title, in the window's top row after a separator, which lines up with the sidebar's divider while
/// the sidebar shows and follows the switch while it hides. Plain text, highlighted under the pointer, which opens the
/// chat's menu.
private struct ChatTitle<Actions: View>: View {
    @Environment(AppState.self) private var state
    let conversation: Conversation
    /// Where the toolbar puts the title in the window, from which its way to the sidebar's divider is measured, and
    /// how wide the window is; until that is known, the title doesn't show.
    @Binding var place: TitlePlace?
    @ViewBuilder let actions: () -> Actions

    var body: some View {
        // The way from where the toolbar puts the item to the sidebar's divider: room before the separator while
        // the sidebar shows, after the title while it hides, so that the item keeps its size as the sidebar
        // slides, and the toolbar leaves it where it is.
        let room = max(0, sidebarWidth - (place?.x ?? sidebarWidth))
        let before = state.showsSidebar ? room : 0
        // Either way the item ends that far past the sidebar's divider, the separator and the title: the title
        // truncates to what the window leaves before the toolbar's buttons at its end at their widest, as in the
        // window's least width before its place is known. Its 360 points in every window pushed the Server's button
        // into the toolbar's overflow menu in the narrowest, a model running.
        let window = place?.window ?? leastWindowWidth
        WholePoints {
            HStack(spacing: 8) {
                Divider().frame(height: 16).accessibilityHidden(true)
                menu.frame(maxWidth: min(360, window - sidebarWidth - 9 - toolbarEnd), alignment: .leading)
            }
            .padding(.leading, before)
            .padding(.trailing, room - before)
        }
        // At its own size, which the toolbar sizes the item to as it changes: a frame that took any width offered
        // kept the item at the width it first had, the title's alone, and the room added once its place was known
        // drew past the item's edge, where clicks don't reach.
        .fixedSize()
        // Fading in once it has its place, as the toolbar sizes the item to it.
        .opacity(place == nil ? 0 : 1)
        .animation(Motion.settle, value: place == nil)
        .background(ToolbarProbe { place = $0 })
        // Along with the sidebar as it slides.
        .animation(Motion.settle, value: state.showsSidebar)
    }

    private var menu: some View {
        Menu(content: actions) {
            HStack(spacing: 5) {
                Text(conversation.title.isEmpty ? "New Chat" : conversation.title)
                    .font(.body.weight(.medium))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
        }
        .menuStyle(.button)
        .buttonStyle(HoverStyle(padding: EdgeInsets()))
        .menuIndicator(.hidden)
        .help("Pin, Rename or Delete This Chat")
        .accessibilityLabel(conversation.title.isEmpty ? "New Chat" : conversation.title)
        .accessibilityHint("Pins, renames or deletes this chat")
    }
}

/// Where the toolbar puts the open chat's title in its window, and how wide the window is, as the title's probe tells
/// them (`ToolbarProbe`).
struct TitlePlace: Equatable {
    let x: CGFloat
    let window: CGFloat
}

/// The toolbar's buttons at its end at their widest, with its gaps: the terminal's, 36 points, and the Server's, 267
/// with its title at its widest, 220 (`ServerButton`), 8 points apart and from the window's edge, after a flexible
/// space of at least 8 with a gap of 8 on either side.
private let toolbarEnd: CGFloat = 8 + 267 + 8 + 36 + 8 + 8 + 8

/// The window's least width (`MainView`).
private let leastWindowWidth: CGFloat = 800

/// Lays its content out from its leading edge, in whole points. The toolbar centers each item in a slot it rounds up
/// to whole points, so that an item a fraction of a point wide sits half a point in from where one a point wider
/// would: a title placed by where it sits would move it, and chase itself.
private struct WholePoints: Layout {
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let size = subviews.first?.sizeThatFits(proposal) ?? .zero
        return CGSize(width: size.width.rounded(.up), height: size.height.rounded(.up))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        subviews.first?.place(at: bounds.origin, proposal: proposal)
    }
}

/// Behind a toolbar item's content, at the content's own size. It tells where the item lies along x in its window,
/// for the content to line up with the window's content under it, which the toolbar lays out apart, and how wide the
/// window is: as it is first laid out in a window, as a window ends going into full screen or out of it, which moves
/// the toolbar's items, and as the window is resized.
/// Nothing else moves an item for good: as the toolbar sizes one, it centers it for a moment in its old place first,
/// which the probe doesn't follow. As the content's size changes, it lays the toolbar out, which sizes the item to it:
/// the toolbar lays itself out only as the window's content changes.
private struct ToolbarProbe: NSViewRepresentable {
    let report: @MainActor (TitlePlace) -> Void

    func makeNSView(context: Context) -> Probe { Probe(report: report) }

    func updateNSView(_ probe: Probe, context: Context) { probe.report = report }

    final class Probe: NSView {
        var report: @MainActor (TitlePlace) -> Void
        private var told: TitlePlace?

        init(report: @escaping @MainActor (TitlePlace) -> Void) {
            self.report = report
            super.init(frame: .zero)
            // Observers by selector go as the probe does.
            for name in [NSWindow.didEnterFullScreenNotification, NSWindow.didExitFullScreenNotification, NSWindow.didResizeNotification] {
                NotificationCenter.default.addObserver(self, selector: #selector(moved), name: name, object: nil)
            }
        }

        @available(*, unavailable)
        required init?(coder: NSCoder) { nil }

        override func setFrameSize(_ size: NSSize) {
            let resized = size != frame.size
            super.setFrameSize(size)
            guard resized, window != nil else { return }
            // Up to the toolbar, which measures its items as it lays them out.
            var view = superview
            while let ancestor = view {
                ancestor.needsLayout = true
                view = ancestor.superview
            }
        }

        override func layout() {
            super.layout()
            if told == nil { tell() }
        }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            told = nil
            needsLayout = true
        }

        @objc private func moved(_ notification: Notification) { tell() }

        private func tell() {
            guard let window else { return }
            let place = TitlePlace(x: convert(bounds, to: nil).minX, window: window.frame.width)
            guard place != told else { return }
            told = place
            Task { @MainActor [report] in report(place) }
        }
    }
}

/// A question to delete chats: which, and what it says of them, as it was asked.
struct ChatDeletion {
    let ids: Set<UUID>
    let title: String
    let message: String

    init(_ chats: [Conversation]) {
        ids = Set(chats.map(\.id))
        if chats.count == 1 {
            title = "Delete this chat?"
            message = "“\(chats[0].title)” will be deleted from this Mac."
        } else {
            title = "Delete \(chats.count) chats?"
            message = "They will be deleted from this Mac."
        }
    }
}

/// A group of the chat list: the pinned chats, then the others by how long ago they changed.
struct ChatSection: Identifiable {
    enum Group: CaseIterable {
        case pinned, today, yesterday, week, month, older

        var title: String {
            switch self {
            case .pinned: "Pinned"
            case .today: "Today"
            case .yesterday: "Yesterday"
            case .week: "Previous 7 Days"
            case .month: "Previous 30 Days"
            case .older: "Older"
            }
        }
    }

    let group: Group
    let chats: [Conversation]
    var id: Group { group }

    /// The groups of `conversations`: those pinned, the last pinned first, so that they keep their
    /// places as they change; then the others in their order by the day each changed last, in
    /// `calendar`, as of `now`; a group that holds no chat left out.
    static func of(_ conversations: [Conversation], now: Date, calendar: Calendar = .current) -> [ChatSection] {
        let today = calendar.startOfDay(for: now)
        let grouped = Dictionary(grouping: conversations) { conversation -> Group in
            guard conversation.pinned == nil else { return .pinned }
            let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: conversation.updated), to: today).day ?? 0
            return switch days {
            case ...0: .today
            case 1: .yesterday
            case 2...7: .week
            case 8...30: .month
            default: .older
            }
        }
        return Group.allCases.compactMap { group in
            grouped[group].map { chats in
                ChatSection(group: group, chats: group == .pinned ? chats.sorted { $0.pinned ?? .distantPast > $1.pinned ?? .distantPast } : chats)
            }
        }
    }
}

/// The conversations, the pinned first, then the others by how recent they are, under a new
/// chat and the search; each has its menu, whose Select starts picking chats to delete at once.
private struct ChatList<Actions: View>: View {
    @Environment(AppState.self) private var state
    @Binding var picked: Set<UUID>?
    @Binding var deletion: ChatDeletion?
    @ViewBuilder let actions: (Conversation) -> Actions
    @State private var query = ""
    /// When the day groups were made; a new day makes them again, the app open past midnight.
    @State private var today = Date.now

    var body: some View {
        let chats = state.chats
        let shown = chats.matching(query)
        VStack(spacing: 0) {
            // A new chat, at the top of the chats it joins, its icon in line with the search's.
            Button {
                chats.newChat()
                state.draft.focus()
            } label: {
                Label("New Chat", systemImage: "plus")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(.rect)
            }
            .buttonStyle(HoverStyle(padding: EdgeInsets(top: 6, leading: 8, bottom: 6, trailing: 8)))
            .help("New Chat (⌘N)")
            .padding(.horizontal, 10)
            .padding(.top, 10)
            .padding(.bottom, 4)
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search", text: $query, prompt: Text("Search"))
                    .textFieldStyle(.plain)
                    // Esc clears it, as it does a search field.
                    .onExitCommand { query = "" }
                if !query.isEmpty {
                    Button { query = "" } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear Search")
                    .accessibilityLabel("Clear Search")
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 7))
            .padding(.horizontal, 10)
            .padding(.bottom, 4)
            if let picked {
                picking(picked, shown: shown)
            }
            if shown.isEmpty {
                empty
            } else {
                list(shown)
            }
        }
        .background(.background.secondary)
        .onReceive(NotificationCenter.default.publisher(for: .NSCalendarDayChanged)) { _ in today = .now }
    }

    /// While the list picks chats to delete, under the search field: picking every chat shown, or
    /// none again; deleting those picked, which asks first; and Done, which ends the picking.
    private func picking(_ picked: Set<UUID>, shown: [Conversation]) -> some View {
        let all = !shown.isEmpty && shown.allSatisfy { picked.contains($0.id) }
        return HStack(spacing: 2) {
            Button(all ? "Deselect All" : "Select All") { self.picked = all ? [] : Set(shown.map(\.id)) }
                .disabled(shown.isEmpty)
            Spacer()
            // The chats picked that the search shows.
            Button("Delete") { deletion = ChatDeletion(shown.filter { picked.contains($0.id) }) }
                .disabled(!shown.contains { picked.contains($0.id) })
            Button("Done") { self.picked = nil }
        }
        .buttonStyle(.hover)
        .padding(.horizontal, 12)
        .padding(.bottom, 4)
    }

    /// What the list says where it has no chat to show: that there is none yet, or that none holds
    /// what is searched for, which the search field's button clears.
    private var empty: some View {
        VStack(spacing: 4) {
            Text(query.isEmpty ? "No Chats Yet" : "No Results for “\(query)”")
                .font(.callout.weight(.medium))
            Text(query.isEmpty ? "Chats show here once they have a message." : "Check the spelling or try a new search.")
                .font(.caption)
        }
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .padding(.horizontal, 20)
        .padding(.top, 28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    /// The conversations in their groups, the latest first in each, the open one held highlighted.
    /// The list takes each change at once, as a lazy stack can: a chat pinned, unpinned, sent to or
    /// new shows at once in its new place, one deleted goes at once, and the rows around them take
    /// their places. Sliding to its new place, a chat drew over the rows it passed; one coming or
    /// going faded over the rows moving; and with nothing moving, one that came after the others
    /// had moved left its place empty for a quarter of a second first.
    private func list(_ shown: [Conversation]) -> some View {
        let chats = state.chats
        let sections = ChatSection.of(shown, now: today)
        // A lazy stack, not a sidebar list, whose source-list rows AppKit insets further: with
        // every row laid out, showing the Chat took 160 ms with 500 chats and 550 with 2,000,
        // against 30 to 50 lazily. Its rows don't slide: scrolled, it moved a row it had put aside
        // from where it last had it, which flew from the list's foot to its top over every row as
        // a chat above the screen was pinned.
        return ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                ForEach(sections) { section in
                    Text(section.group.title)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.top, section.id == sections.first?.id ? 6 : 14)
                        .padding(.bottom, 2)
                        .accessibilityAddTraits(.isHeader)
                    ForEach(section.chats) { conversation in
                        let id = conversation.id
                        ChatRow(conversation: conversation, open: chats.selection == id, picked: picked.map { $0.contains(id) },
                                answering: chats.streaming == id) {
                            if picked == nil {
                                chats.selection = id
                                state.draft.focus()
                            } else {
                                picked?.formSymmetricDifference([id])
                            }
                        } actions: {
                            actions(conversation)
                            if picked == nil {
                                Divider()
                                Button("Select") { picked = [id] }
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
        }
        // At once, whatever else changes with it, as the server's stage animates the window (`MainView`).
        .animation(nil, value: sections.map { $0.chats.map(\.id) })
    }
}

/// A conversation in the list: its title, highlighted as the app's buttons are, held while open;
/// the highlight in line with the search field. Under the pointer, a button at its end opens its
/// menu, as a right click does; while the list picks chats to delete, a check before the title
/// says whether it is picked, and a click picks it or not.
private struct ChatRow<Actions: View>: View {
    let conversation: Conversation
    let open: Bool
    /// Whether it is picked, while the list picks chats to delete; nil otherwise.
    let picked: Bool?
    /// Whether its reply streams.
    let answering: Bool
    let choose: () -> Void
    @ViewBuilder let actions: () -> Actions
    @State private var hovering = false

    var body: some View {
        let more = picked == nil && hovering
        Button(action: choose) {
            HStack(spacing: 6) {
                if let picked {
                    Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(picked ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                }
                Text(conversation.title.isEmpty ? "New Chat" : conversation.title)
                    .lineLimit(1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if answering && !more {
                    ProgressView().controlSize(.mini).accessibilityLabel("Answering")
                }
            }
            .contentShape(.rect)
        }
        // Room at the end for the menu's button while it shows, and the highlight kept while the
        // pointer is on that button.
        .buttonStyle(HoverStyle(padding: EdgeInsets(top: 6, leading: 8, bottom: 6, trailing: more ? 30 : 8), held: open && picked == nil,
                                hovered: hovering))
        .overlay(alignment: .trailing) {
            if more { RowMenuButton(trailing: 3, actions: actions) }
        }
        .onHover { hovering = $0 }
        .contextMenu(menuItems: actions)
        .accessibilityAddTraits(open || picked == true ? .isSelected : [])
    }
}

/// The open conversation, its transcript a web page (`Transcript`), over the composer.
private struct ConversationView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let shown = state.chats.current.flatMap { $0.messages.isEmpty ? nil : $0 }
        VStack(spacing: 0) {
            // What the composer leaves goes at once, and what it makes room for fades in once it
            // has moved (`Motion.comingAndGoing`): the transcript faded out under the composer
            // moving over it as a new chat opened, and in under the composer moving away as a
            // chat got its first message, and so did the mark.
            if let conversation = shown {
                LiveTranscript(page: page(conversation))
                    .toolbarLine()
                    .transition(Motion.comingAndGoing)
            } else {
                // A new chat: the mark, showing what the Server does, over the composer, and while a
                // start warms up why it can take long, in the room under the mark, so that the
                // composer stays in its place.
                Spacer()
                Group {
                    StatusMark(look: state.serverStatus.look, since: state.server.since, size: 44)
                    Text(state.serverStatus.note ?? "")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .padding(.horizontal, 32)
                        .frame(height: 28)
                }
                .transition(Motion.comingAndGoing)
            }
            // One composer either way, so that its text view, and what the input method composes
            // in it, stays when a chat gets its first message or a new chat opens.
            footer
            if shown == nil {
                Spacer()
                Spacer()
            }
        }
        // The composer moves, whole, between a new chat's middle and a conversation's foot.
        .animation(Motion.settle, value: shown == nil)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }

    private var footer: some View {
        Composer(draft: state.draft)
            .frame(maxWidth: column)
            .padding(.horizontal, 32)
            .padding(.bottom, 24)
    }

    /// `conversation` as the page shows it, Ask Again as the Server allows.
    private func page(_ conversation: Conversation) -> TranscriptPage {
        let server = state.server
        let askAgain = TranscriptPage.AskAgain(enabled: server.answers && state.chats.streaming == nil,
                                               help: "Ask \(server.model.map(state.displayName(of:)) ?? "the model") Again")
        return TranscriptPage(conversation, askAgain: askAgain, name: state.displayName(of:))
    }
}

/// The transcript of the open conversation, with the reply that streams in it: a view of its own,
/// the one that reads the reply, so that each flush of it draws the transcript alone, which sends
/// the page that reply, and the conversation, the same, is neither built nor compared again.
private struct LiveTranscript: View {
    @Environment(AppState.self) private var state
    let page: TranscriptPage

    var body: some View {
        let reply = state.chats.reply.flatMap { $0.conversation == page.id ? $0 : nil }
        TranscriptView(transcript: state.transcript, page: page, reply: reply.map {
            TranscriptMessage($0.message, live: $0.live, asleep: state.server.asleep, name: state.displayName(of:))
        })
    }
}
