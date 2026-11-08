import SwiftUI

/// The width of the composer: the transcript's column (`main`'s max-width in page.css), so
/// that the two line up.
private let column: CGFloat = 680

/// The conversations at the side of the open one, in a sidebar that hides (`WithSidebar`). The
/// window's toolbar is the Server's too, so that the switch and the Server's button stay where
/// they are. The questions the list asks, to rename and to delete chats, show here.
struct ChatScreen: View {
    @Environment(AppState.self) private var state
    /// The chat asked a new name for, and the name typed.
    @State private var renaming: Conversation?
    @State private var name = ""
    /// The question to delete chats, while it asks, as it was asked.
    @State private var deletion: ChatDeletion?
    /// The chats picked to delete while the list picks them, from a chat's Select; nil otherwise.
    @State private var picked: Set<UUID>?

    var body: some View {
        WithSidebar {
            ChatList(picked: $picked, deletion: $deletion, actions: actions)
        } content: {
            ConversationView()
        }
        .onAppear { state.draft.focus() }
        .alert("Rename Chat", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } }
        ), presenting: renaming) { conversation in
            TextField("Name", text: $name)
            Button("Rename") { state.chats.rename(conversation.id, to: name) }
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

    /// What can be done with a chat, as its row's menu and a right click on its row give it: pin it
    /// to the list's top, or take it off there; rename it; delete it.
    @ViewBuilder private func actions(for conversation: Conversation) -> some View {
        Button(conversation.pinned == nil ? "Pin" : "Unpin") { state.chats.pin(conversation.id, conversation.pinned == nil) }
        Button("Rename…") {
            name = conversation.title
            renaming = conversation
        }
        Button("Delete…", role: .destructive) { deletion = ChatDeletion([conversation]) }
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
        let shown = query.isEmpty ? chats.conversations : chats.conversations.filter { conversation in
            conversation.title.localizedStandardContains(query)
                || conversation.messages.contains { $0.text.localizedStandardContains(query) }
        }
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
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 7))
            .padding(.horizontal, 10)
            .padding(.bottom, 4)
            if let picked {
                picking(picked, shown: shown)
            }
            list(shown)
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
            Spacer()
            Button("Delete") { deletion = ChatDeletion(state.chats.conversations.filter { picked.contains($0.id) }) }
                .disabled(picked.isEmpty)
            Button("Done") { self.picked = nil }
        }
        .buttonStyle(.hover)
        .padding(.horizontal, 12)
        .padding(.bottom, 4)
    }

    /// The conversations in their groups, the latest first in each, the open one held highlighted;
    /// a conversation that gets a message moves to the top of its group, and one deleted closes its
    /// gap.
    private func list(_ shown: [Conversation]) -> some View {
        let chats = state.chats
        let sections = ChatSection.of(shown, now: today)
        // A stack, not a sidebar list, whose source-list rows AppKit insets further.
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
                        ChatRow(conversation: conversation, open: chats.selection == id, picked: picked.map { $0.contains(id) }) {
                            if picked == nil { chats.selection = id } else { picked?.formSymmetricDifference([id]) }
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
        .animation(Motion.settle, value: sections.map { $0.chats.map(\.id) })
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
            }
            .contentShape(.rect)
        }
        // Room at the end for the menu's button while it shows, and the highlight kept while the
        // pointer is on that button.
        .buttonStyle(HoverStyle(padding: EdgeInsets(top: 6, leading: 8, bottom: 6, trailing: more ? 30 : 8), held: open && picked == nil,
                                hovered: hovering))
        .overlay(alignment: .trailing) {
            if more {
                // A square button, as the menu buttons of a sidebar's rows are.
                Menu(content: actions) {
                    Image(systemName: "ellipsis").foregroundStyle(.secondary).frame(width: 24, height: 24)
                }
                .menuStyle(.button)
                .buttonStyle(HoverStyle(padding: EdgeInsets(), shape: AnyShape(.rect(cornerRadius: 6))))
                .menuIndicator(.hidden)
                .fixedSize()
                .padding(.trailing, 3)
                .help("More")
                .accessibilityLabel("More")
            }
        }
        .onHover { hovering = $0 }
        .contextMenu(menuItems: actions)
        .accessibilityAddTraits(open || picked == true ? .isSelected : [])
    }
}

/// The open conversation, its transcript a web page (`Transcript`), over the composer.
private struct ConversationView: View {
    @Environment(AppState.self) private var state
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        let shown = state.chats.current.flatMap { $0.messages.isEmpty ? nil : $0 }
        VStack(spacing: 0) {
            if let conversation = shown {
                LiveTranscript(page: page(conversation))
                    // The line the window draws under its toolbar over a scroll view there, which
                    // a web view isn't to it.
                    .overlay(alignment: .top) { Rectangle().fill(.separator).frame(height: 1 / displayScale) }
            } else {
                // A new chat: the mark, showing what the Server does, over the composer.
                Spacer()
                StatusMark(look: state.serverStatus.look, since: state.server.since, size: 44)
                    .padding(.bottom, 28)
            }
            // One composer either way, so that its text view, and what the input method composes
            // in it, stays when a chat gets its first message or a new chat opens.
            footer
            if shown == nil {
                Spacer()
                Spacer()
            }
        }
        // The composer moves, whole, between a new chat's middle and a conversation's foot, as the
        // mark and the transcript fade.
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
