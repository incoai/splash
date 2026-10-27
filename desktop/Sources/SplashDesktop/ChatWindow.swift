import AppKit
import SplashChat
import SwiftUI

private enum Palette {
    static let background = Color(light: 0xFAFAF8, dark: 0x1C1D1F)
    static let surface = Color(light: 0xFFFFFF, dark: 0x252629)
    static let bubble = Color(light: 0xEFEFEC, dark: 0x303134)
    static let accent = Color(light: 0xBC5B32, dark: 0xF2A17B)
    static let accentWash = Color(light: 0xF5E7DF, dark: 0x3A2D27)
}

private extension Color {
    init(light: UInt32, dark: UInt32) {
        func native(_ hex: UInt32) -> NSColor {
            NSColor(srgbRed: Double((hex >> 16) & 255) / 255,
                    green: Double((hex >> 8) & 255) / 255,
                    blue: Double(hex & 255) / 255, alpha: 1)
        }
        self.init(nsColor: NSColor(name: nil) { appearance in
            native(appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light)
        })
    }
}

struct ChatWindow: View {
    @Bindable var chat: ChatStore
    @State private var showingConnection = false
    @State private var confirmingClear = false
    @State private var composerFocusRequest = 0
    @State private var composerHeight: CGFloat = 40

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().opacity(0.5)
            if chat.messages.isEmpty {
                welcome.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                conversation
            }
            VStack(spacing: 12) {
                if let error = chat.replyError {
                    notice(error, retry: { chat.retry() })
                }
                if let error = chat.storageError { notice(error, retry: nil) }
                if !chat.connected { connectionNotice }
                composer
            }
            .frame(maxWidth: 700)
            .padding(.horizontal, 28)
            .padding(.bottom, 18)
            .frame(maxWidth: .infinity)
        }
        .background(Palette.background)
        .tint(Palette.accent)
        .sheet(isPresented: $showingConnection) {
            ConnectionSettings(chat: chat)
        }
        .alert("Clear this chat?", isPresented: $confirmingClear) {
            Button("Cancel", role: .cancel) {}
            Button("Clear chat", role: .destructive) { chat.clear(); composerFocusRequest += 1 }
        } message: {
            Text("This removes the conversation saved on this Mac.")
        }
        .task {
            composerFocusRequest += 1
            chat.connect()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(10)) } catch { break }
                if !chat.connecting && !chat.generating { chat.connect() }
            }
        }
        .onChange(of: chat.generating) { _, running in
            if !running { composerFocusRequest += 1 }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Palette.accent)
            VStack(alignment: .leading, spacing: 3) {
                Text("Splash").font(.system(size: 14, weight: .semibold))
                HStack(spacing: 5) {
                    Circle().fill(chat.connected ? Color.green : Color.secondary.opacity(0.5)).frame(width: 5, height: 5)
                    Text(chat.connected ? shortModel : (chat.connecting ? "Connecting…" : "Waiting for Splash"))
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .help(chat.connected ? chat.model : chat.address)
                }
            }
            Spacer()
            Button { confirmingClear = true } label: {
                Image(systemName: "square.and.pencil").frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .help("Clear chat (⌘N)")
            .accessibilityLabel("Clear chat")
            .keyboardShortcut("n", modifiers: .command)
            .disabled(chat.messages.isEmpty || chat.generating)
            Button { showingConnection.toggle() } label: {
                Image(systemName: "slider.horizontal.3").frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .help("Connection settings (⌘,)")
            .accessibilityLabel("Connection settings")
            .keyboardShortcut(",", modifiers: .command)
            .disabled(chat.generating)
        }
        .padding(.leading, 80)
        .padding(.trailing, 20)
        .frame(height: 64)
    }

    private var shortModel: String {
        chat.model.split(separator: "/").last.map(String.init) ?? chat.model
    }

    private var welcome: some View {
        VStack(spacing: 16) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(Palette.accent)
                .frame(width: 64, height: 64)
                .background(Palette.accentWash, in: RoundedRectangle(cornerRadius: 20))
            Text("What’s on your mind?")
                .font(.system(size: 26, weight: .medium, design: .rounded))
            Text("A quiet space to chat with your local model.")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                suggestion("Explain something", prompt: "Explain speculative decoding in simple terms.")
                suggestion("Write some code", prompt: "Write a small Python function that removes duplicates from a list, preserving order.")
                suggestion("Find an idea", prompt: "Help me brainstorm a small project I could build this weekend.")
            }
            .padding(.top, 10)
        }
        .padding(24)
    }

    private func suggestion(_ title: String, prompt: String) -> some View {
        Button { chat.draft = prompt; composerFocusRequest += 1 } label: {
            Text(title).font(.system(size: 11)).padding(.horizontal, 12).padding(.vertical, 9)
                .background(Palette.surface, in: Capsule())
                .overlay(Capsule().strokeBorder(.primary.opacity(0.08)))
        }
        .buttonStyle(.plain)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    ForEach(chat.messages) { message in
                        MessageView(message: message,
                                    streaming: chat.generating && message.id == chat.messages.last?.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: 644)
                .padding(.horizontal, 28)
                .padding(.vertical, 28)
                .frame(maxWidth: .infinity)
            }
            .defaultScrollAnchor(.bottom)
            .onChange(of: chat.messages.last?.content) { _, _ in
                DispatchQueue.main.async { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: chat.messages.count) { _, _ in
                DispatchQueue.main.async { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private var connectionNotice: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "network").padding(.top, 2)
            VStack(alignment: .leading, spacing: 5) {
                Text(chat.connectionError ?? "Connecting to your local Splash server…")
                    .font(.system(size: 12, weight: .medium))
                Text("Start Splash in Terminal with your model, then reconnect.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Button("Reconnect") { chat.connect() }
                .disabled(chat.connecting)
        }
        .font(.system(size: 12))
        .padding(14)
        .background(Palette.bubble.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
    }

    private var composer: some View {
        VStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 14) {
                MessageEditor(text: $chat.draft, height: $composerHeight,
                              focusRequest: composerFocusRequest, send: { chat.send() })
                    .frame(height: composerHeight)
                HStack {
                    Toggle(isOn: $chat.thinking) {
                        Label("Thinking", systemImage: "sparkle")
                            .font(.system(size: 11))
                    }
                    .toggleStyle(.checkbox)
                    .disabled(chat.generating)
                    .foregroundStyle(.secondary)
                    .help("Use the model’s default reasoning. Turn off for a direct answer.")
                    Spacer()
                    if chat.generating {
                        Button { chat.stop() } label: {
                            Image(systemName: "stop.fill").font(.system(size: 11))
                                .frame(width: 30, height: 30)
                        }
                        .buttonStyle(.plain)
                        .background(Palette.accent, in: Circle())
                        .foregroundStyle(Palette.background)
                        .accessibilityLabel("Stop response")
                        .help("Stop response (Esc)")
                        .keyboardShortcut(.escape, modifiers: [])
                    } else {
                        Button { chat.send() } label: {
                            Image(systemName: "arrow.up").font(.system(size: 14, weight: .semibold))
                                .frame(width: 30, height: 30)
                        }
                        .buttonStyle(.plain)
                        .background(chat.canSend ? Palette.accent : Color.secondary.opacity(0.13), in: Circle())
                        .foregroundStyle(chat.canSend ? Palette.background : Color.secondary)
                        .disabled(!chat.canSend)
                        .accessibilityLabel("Send message")
                        .help("Send message (Return)")
                    }
                }
            }
            .padding(16)
            .background(Palette.surface, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(.primary.opacity(0.1)))
            .shadow(color: .black.opacity(0.025), radius: 10, y: 3)
            HStack {
                Text("On your Mac. In your flow.")
                Spacer()
                if let speed = chat.tokensPerSecond {
                    Text("\(speed, specifier: "%.1f") tokens/s")
                } else {
                    Text("Return to send · Shift Return for a new line")
                }
            }
            .font(.system(size: 10))
            .foregroundStyle(.tertiary)
            .padding(.horizontal, 4)
        }
    }

    private func notice(_ message: String, retry: (() -> Void)?) -> some View {
        HStack(alignment: .top) {
            Image(systemName: "exclamationmark.circle")
            Text(message).textSelection(.enabled)
            Spacer(minLength: 0)
            if let retry { Button("Try again", action: retry).disabled(chat.generating || !chat.connected) }
        }
        .font(.system(size: 12))
        .foregroundStyle(Palette.accent)
    }
}

private struct MessageView: View {
    let message: ChatMessage
    let streaming: Bool
    @State private var showingThinking = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.role == "user" { Spacer(minLength: 70) }
            VStack(alignment: .leading, spacing: 12) {
                if message.role == "assistant" {
                    Label("Splash", systemImage: "bolt.fill")
                        .font(.system(size: 11, weight: .medium)).foregroundStyle(Palette.accent)
                    if !message.reasoning.isEmpty {
                        DisclosureGroup(streaming && message.content.isEmpty ? "Thinking…" : "Thinking", isExpanded: $showingThinking) {
                            Text(message.reasoning)
                                .font(.system(size: 12)).foregroundStyle(.secondary)
                                .textSelection(.enabled).padding(.top, 8)
                        }
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    }
                    if message.content.isEmpty && message.reasoning.isEmpty && streaming {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("Working on it…").font(.system(size: 12)).foregroundStyle(.secondary)
                        }
                    }
                    if !message.content.isEmpty { ReplyText(content: message.content) }
                    if !streaming && !message.content.isEmpty {
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(message.content, forType: .string)
                        } label: { Label("Copy", systemImage: "doc.on.doc") }
                        .buttonStyle(.plain)
                        .font(.system(size: 10)).foregroundStyle(.tertiary)
                        .help("Copy reply")
                    }
                } else {
                    Text(message.content).font(.system(size: 14)).textSelection(.enabled)
                }
            }
            .padding(message.role == "user" ? 14 : 0)
            .background(message.role == "user" ? Palette.bubble : .clear, in: RoundedRectangle(cornerRadius: 16))
            if message.role == "assistant" { Spacer(minLength: 0) }
        }
    }
}

private struct ReplyText: View {
    let content: String

    var body: some View {
        let sections = content.components(separatedBy: "```")
        VStack(alignment: .leading, spacing: 12) {
            ForEach(sections.indices, id: \.self) { index in
                if index.isMultiple(of: 2) {
                    let text = sections[index].trimmingCharacters(in: .newlines)
                    if !text.isEmpty {
                        Text((try? AttributedString(markdown: text,
                            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text))
                            .font(.system(size: 14)).lineSpacing(5).textSelection(.enabled)
                    }
                } else {
                    let parts = sections[index].split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false)
                    let code = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .newlines) : ""
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text(parts.first.map(String.init) ?? "Code")
                            Spacer()
                            Button("Copy code") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(code, forType: .string)
                            }.buttonStyle(.plain)
                        }
                        .font(.system(size: 10)).foregroundStyle(.secondary)
                        ScrollView(.horizontal) {
                            Text(code).font(.system(size: 12, design: .monospaced))
                                .fixedSize(horizontal: true, vertical: true)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(14)
                    .background(Palette.bubble, in: RoundedRectangle(cornerRadius: 12))
                }
            }
        }
    }
}

private struct ConnectionSettings: View {
    @Bindable var chat: ChatStore
    @Environment(\.dismiss) private var dismiss
    @State private var address = ""
    @State private var apiKey = ""
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Connection").font(.system(size: 16, weight: .semibold))
            VStack(alignment: .leading, spacing: 6) {
                Text("Splash server").font(.system(size: 12, weight: .medium))
                TextField("http://127.0.0.1:8000", text: $address).textFieldStyle(.roundedBorder)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("API key · optional").font(.system(size: 12, weight: .medium))
                SecureField("Only if your server requires one", text: $apiKey).textFieldStyle(.roundedBorder)
                Text("Kept in memory until you quit Splash.").font(.system(size: 10)).foregroundStyle(.secondary)
            }
            if let error { Text(error).font(.system(size: 11)).foregroundStyle(.red) }
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Connect") {
                    do {
                        try chat.configure(address: address, apiKey: apiKey)
                        dismiss()
                    } catch { self.error = error.localizedDescription }
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            }
        }
        .padding(22)
        .frame(width: 340)
        .onAppear { address = chat.address; apiKey = chat.apiKey }
    }
}
