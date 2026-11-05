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

private struct ChatList: View {
    @Environment(AppState.self) private var state
    @State private var query = ""
    @State private var deleting: Conversation?
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
        }
        .background(.background.secondary)
        .confirmationDialog("Delete this chat?", isPresented: Binding(
            get: { deleting != nil }, set: { if !$0 { deleting = nil } }
        ), presenting: deleting) { conversation in
            Button("Delete", role: .destructive) { chats.delete(conversation.id) }
        } message: { conversation in
            Text("“\(conversation.title)” will be deleted from this Mac.")
        }
        .alert("Rename Chat", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } }
        ), presenting: renaming) { conversation in
            TextField("Name", text: $title)
            Button("Rename") { chats.rename(conversation.id, to: title) }
            Button("Cancel", role: .cancel) {}
        }
    }

    /// The conversations, the latest first, the open one held highlighted; a conversation that
    /// gets a message moves to the top, and one deleted closes its gap.
    private func list(_ shown: [Conversation]) -> some View {
        let chats = state.chats
        // A stack, not a sidebar list, whose source-list rows AppKit insets further.
        return ScrollView {
            LazyVStack(spacing: 2) {
                ForEach(shown) { conversation in
                    ChatRow(conversation: conversation, open: chats.selection == conversation.id) {
                        chats.selection = conversation.id
                    }
                    .contextMenu {
                        Button("Rename…") {
                            title = conversation.title
                            renaming = conversation
                        }
                        Button("Delete…", role: .destructive) { deleting = conversation }
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
        }
        .animation(Motion.settle, value: shown.map(\.id))
    }
}

/// A conversation in the list: its title, highlighted as the app's buttons are, held while open;
/// the highlight in line with the search field.
private struct ChatRow: View {
    let conversation: Conversation
    let open: Bool
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            Text(conversation.title.isEmpty ? "New Chat" : conversation.title)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(.rect)
        }
        .buttonStyle(HoverStyle(padding: EdgeInsets(top: 6, leading: 8, bottom: 6, trailing: 8), held: open))
        .accessibilityAddTraits(open ? .isSelected : [])
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
