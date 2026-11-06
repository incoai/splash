import AppKit
import SwiftUI

/// The width of the composer: the transcript's column (`main`'s max-width in page.css), so
/// that the two line up.
private let column: CGFloat = 680

/// The conversations at the side of the open one, in a sidebar that hides (`WithSidebar`). The
/// window's toolbar is the Server's too, so that the switch and the Server's button stay where
/// they are.
struct ChatScreen: View {
    @Environment(AppState.self) private var state

    var body: some View {
        WithSidebar {
            ChatList()
        } content: {
            ConversationView()
        }
        .onAppear { state.draft.focus() }
    }
}

/// The chats picked in the list: the open one at first, then those a click with Command or Shift
/// picks, and the row a later Shift-click extends from.
struct ChatListSelection: Equatable {
    private(set) var ids: Set<UUID> = []
    private(set) var anchor: UUID?

    /// Takes a click: a plain one picks and opens `id`, a Command-click picks or unpicks it alone,
    /// and a Shift-click picks everything between the row clicked before and `id`. Returns whether
    /// the click opens the chat.
    @discardableResult
    mutating func click(_ id: UUID, command: Bool, shift: Bool, in order: [UUID]) -> Bool {
        if shift, let anchor, let start = order.firstIndex(of: anchor), let end = order.firstIndex(of: id) {
            ids = Set(order[start <= end ? start...end : end...start])
            return false
        }
        if command {
            if ids.contains(id) { ids.remove(id) } else { ids.insert(id) }
            anchor = id
            return false
        }
        ids = [id]
        anchor = id
        return true
    }

    /// Follows the conversation the app opens elsewhere, as when a new chat starts or a chat
    /// another part deletes closes.
    mutating func follow(_ id: UUID?) {
        guard let id else {
            ids = []
            anchor = nil
            return
        }
        ids = [id]
        anchor = id
    }

    /// Drops the chats no longer in the list; a Shift-click then has no row to extend from.
    mutating func prune(to present: Set<UUID>) {
        ids.formIntersection(present)
        if let anchor, !present.contains(anchor) { self.anchor = nil }
    }

    /// The picked chats among the rows `shown`: a search can hide a picked chat, which a delete
    /// of the rows in view leaves alone.
    func picked(in shown: [UUID]) -> Set<UUID> {
        ids.intersection(shown)
    }

    mutating func clear() {
        ids = []
        anchor = nil
    }
}

private struct ChatList: View {
    @Environment(AppState.self) private var state
    @State private var query = ""
    @State private var selection = ChatListSelection()
    @State private var deleting: Conversation?
    @State private var deletingSelected: Set<UUID>?
    @State private var deletingAll = false
    @State private var renaming: Conversation?
    @State private var title = ""

    var body: some View {
        @Bindable var chats = state.chats
        let shown = query.isEmpty ? chats.conversations : chats.conversations.filter { conversation in
            conversation.title.localizedStandardContains(query)
                || conversation.messages.contains { $0.text.localizedStandardContains(query) }
        }
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search", text: $query, prompt: Text("Search"))
                    .textFieldStyle(.plain)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 7))
            .padding(.horizontal, 10)
            .padding(.top, 10)
            .padding(.bottom, 4)
            list(shown)
            Divider()
            HStack {
                Button("Delete All Chats") { deletingAll = true }
                    .buttonStyle(.hover)
                    .foregroundStyle(.tint)
                    .disabled(chats.conversations.isEmpty)
                    .help("Delete All Chats")
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
        .background(.background.secondary)
        .onAppear { follow() }
        .onChange(of: chats.selection) { follow() }
        .onChange(of: chats.conversations.map(\.id)) { _, ids in selection.prune(to: Set(ids)) }
        .confirmationDialog("Delete this chat?", isPresented: Binding(
            get: { deleting != nil }, set: { if !$0 { deleting = nil } }
        ), presenting: deleting) { conversation in
            Button("Delete", role: .destructive) { chats.delete(conversation.id) }
        } message: { conversation in
            Text("“\(conversation.title)” will be deleted from this Mac.")
        }
        .confirmationDialog("Delete selected chats?", isPresented: Binding(
            get: { deletingSelected != nil }, set: { if !$0 { deletingSelected = nil } }
        ), presenting: deletingSelected) { ids in
            Button("Delete", role: .destructive) {
                chats.delete(ids)
                follow()
            }
        } message: { ids in
            Text("\(ids.count) chats will be deleted from this Mac. This cannot be undone.")
        }
        .confirmationDialog("Delete all chats?", isPresented: $deletingAll) {
            Button("Delete All Chats", role: .destructive) {
                chats.deleteAll()
                selection.clear()
            }
        } message: {
            Text("This cannot be undone.")
        }
        .alert("Rename Chat", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } }
        ), presenting: renaming) { conversation in
            TextField("Name", text: $title)
            Button("Rename") { chats.rename(conversation.id, to: title) }
            Button("Cancel", role: .cancel) {}
        }
    }

    /// Picks the conversation the app opened elsewhere, or none.
    private func follow() {
        if let id = state.chats.selection, state.chats.conversations.contains(where: { $0.id == id }) {
            selection.follow(id)
        } else {
            selection.follow(nil)
        }
    }

    /// The conversations, the latest first, each picked as the click says; a conversation that
    /// gets a message moves to the top, and one deleted closes its gap.
    private func list(_ shown: [Conversation]) -> some View {
        let picked = selection.picked(in: shown.map(\.id))
        // A stack, not a sidebar list, whose source-list rows AppKit insets further.
        return ScrollView {
            LazyVStack(spacing: 2) {
                ForEach(shown) { conversation in
                    ChatRow(conversation: conversation, selected: selection.ids.contains(conversation.id),
                            open: state.chats.selection == conversation.id) {
                        click(conversation, in: shown)
                    }
                    .contextMenu {
                        if picked.count > 1, picked.contains(conversation.id) {
                            Button("Delete Selected Chats…", role: .destructive) {
                                deletingSelected = picked
                            }
                        } else {
                            Button("Rename…") {
                                title = conversation.title
                                renaming = conversation
                            }
                            Button("Delete…", role: .destructive) { deleting = conversation }
                        }
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
        }
        .animation(Motion.settle, value: shown.map(\.id))
    }

    /// Takes a click with the keys Command and Shift, as a sidebar picks its rows.
    private func click(_ conversation: Conversation, in shown: [Conversation]) {
        let flags = NSEvent.modifierFlags
        if selection.click(conversation.id, command: flags.contains(.command), shift: flags.contains(.shift),
                           in: shown.map(\.id)) {
            state.chats.selection = conversation.id
        }
    }
}

/// A conversation in the list: its title, highlighted as the app's buttons are, held while picked;
/// the highlight in line with the search field. The open chat, when it is not picked, takes an
/// outline, so that the list always shows which chat is open.
private struct ChatRow: View {
    let conversation: Conversation
    let selected: Bool
    let open: Bool
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            Text(conversation.title.isEmpty ? "New Chat" : conversation.title)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(.rect)
        }
        .buttonStyle(HoverStyle(padding: EdgeInsets(top: 6, leading: 8, bottom: 6, trailing: 8), held: selected))
        .overlay {
            if open && !selected {
                RoundedRectangle(cornerRadius: 7).strokeBorder(.separator).allowsHitTesting(false)
            }
        }
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityValue(open ? "Open" : "")
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
