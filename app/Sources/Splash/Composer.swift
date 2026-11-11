import AppKit
import PDFKit
import SwiftUI
import UniformTypeIdentifiers

/// The message being written: its text and what is attached to it, kept as files in the store's
/// attachments folder as it attaches (`ChatStore.attachments`), which become the message's once it
/// is sent; one taken off goes.
@MainActor @Observable
final class Draft {
    var text = "" {
        didSet { if text != oldValue { notice = nil } }
    }
    private(set) var attachments: [Attachment] = []
    /// The images attached, as the composer shows them, by attachment.
    private(set) var thumbnails: [UUID: NSImage] = [:]
    /// How many attached images are still being encoded and kept, which join `attachments` once
    /// they are.
    private(set) var attaching = 0
    /// The attachment that joins `attachments` last, which the next one waits for.
    private var lastAttachment: Task<Void, Never>?
    /// Asks the text field for the keyboard; counts the requests.
    private(set) var focusRequests = 0
    /// What the composer says over the message, in the chat it is about, until the draft changes
    /// or is sent: why the server refused what the draft holds again or a reply asked again
    /// (`restore`), or a file that couldn't be attached.
    private(set) var notice: Notice?

    struct Notice: Equatable {
        let text: String
        /// The conversation it is about; nil for a new one.
        let conversation: UUID?
    }
    /// Where attachments are kept.
    let files: URL

    init(files: URL) {
        self.files = files
    }

    var isEmpty: Bool { text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && attachments.isEmpty }

    func focus() { focusRequests += 1 }

    /// Attaches `images` in a form the server takes: at most 2048 pixels on the long side (its
    /// limit is about four million pixels), PNG when they have transparency and JPEG otherwise.
    /// They are encoded away from the main thread, which a photo would hold for a tenth of a second
    /// or more, and join the draft in the order they were attached.
    func attach(_ images: [NSImage], unreadable: Int = 0) {
        let sources = images.compactMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
        let missed = images.count - sources.count + unreadable
        guard !sources.isEmpty else {
            if missed > 0 { noticeUnattached(missed) }
            return
        }
        attaching += sources.count
        let files = files
        let encoding = Task.detached(priority: .userInitiated) {
            sources.compactMap { source in Self.encoded(source).flatMap { Self.kept($0, in: files) } }
        }
        let previous = lastAttachment
        lastAttachment = Task {
            await previous?.value
            let kept = await encoding.value
            for (attachment, data) in kept {
                attachments.append(attachment)
                thumbnails[attachment.id] = NSImage(data: data)
            }
            attaching -= sources.count
            let failed = missed + sources.count - kept.count
            if failed > 0 { noticeUnattached(failed) }
        }
    }

    private func noticeUnattached(_ failed: Int) {
        notice = Notice(text: failed == 1 ? "An image couldn’t be attached" : "\(failed) images couldn’t be attached",
                        conversation: conversation)
    }

    /// Attaches the PDFs at `urls` as they are, with their names and pages, read and kept away from
    /// the main thread, in the order they were attached; one PDFKit can't open still attaches, for
    /// the server to say why.
    func attach(pdfs urls: [URL]) {
        guard !urls.isEmpty else { return }
        attaching += urls.count
        let files = files
        let reading = Task.detached(priority: .userInitiated) { urls.map { ($0, Self.keptPDF($0, in: files)) } }
        let previous = lastAttachment
        lastAttachment = Task {
            await previous?.value
            let read = await reading.value
            attachments += read.compactMap(\.1)
            attaching -= urls.count
            let failed = read.filter { $0.1 == nil }.map { $0.0.lastPathComponent }
            if let first = failed.first {
                notice = Notice(text: failed.count == 1 ? "“\(first)” couldn’t be read" : "\(failed.count) PDFs couldn’t be read",
                                conversation: conversation)
            }
        }
    }

    /// The conversation the composer writes in, for a notice about it (`ComposerField` sets it).
    @ObservationIgnored var conversation: UUID?

    /// The record of the PDF at `url`, kept as a file in `files`; nil where it can't be read or
    /// kept.
    nonisolated private static func keptPDF(_ url: URL, in files: URL) -> Attachment? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        let attachment = Attachment(kind: .pdf, type: "application/pdf", bytes: data.count, name: url.lastPathComponent,
                                    pages: PDFDocument(data: data)?.pageCount)
        return attachment.write(data, in: files) ? attachment : nil
    }

    /// Takes attachment `index` off, and its file.
    func remove(at index: Int) {
        notice = nil
        let attachment = attachments.remove(at: index)
        thumbnails[attachment.id] = nil
        let file = files.appending(path: attachment.file)
        Task.detached { try? FileManager.default.removeItem(at: file) }
    }

    /// Takes the message, leaving the draft empty; its attachments' files are the message's.
    func take() -> (text: String, attachments: [Attachment]) {
        defer {
            text = ""
            attachments = []
            thumbnails = [:]
            notice = nil
        }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), attachments)
    }

    /// Takes back `refused`, what the server didn't take, with why, in its chat: its text before
    /// what was typed since, a blank line apart, and its attachments before those attached since.
    func restore(_ refused: ChatStore.Refused) {
        let typed = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : text
        text = [refused.text, typed].filter { !$0.isEmpty }.joined(separator: "\n\n")
        attachments.insert(contentsOf: refused.attachments, at: 0)
        for attachment in refused.attachments where attachment.kind == .image {
            thumbnails[attachment.id] = NSImage(contentsOf: files.appending(path: attachment.file))
        }
        notice = Notice(text: refused.reason, conversation: refused.conversation)
        focus()
    }

    /// The record of image `data`, kept as a file in `files`, with the data; nil where it can't be
    /// written.
    nonisolated private static func kept(_ data: Data, in files: URL) -> (Attachment, Data)? {
        let attachment = Attachment.image(data)
        return attachment.write(data, in: files) ? (attachment, data) : nil
    }

    nonisolated private static func encoded(_ source: CGImage) -> Data? {
        let scale = min(1, 2048 / Double(max(source.width, source.height)))
        let width = Int(Double(source.width) * scale), height = Int(Double(source.height) * scale)
        guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
                                            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let context = NSGraphicsContext(bitmapImageRep: bitmap)
        else { return nil }
        context.cgContext.interpolationQuality = .high
        context.cgContext.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        let opaque = source.alphaInfo == .none || source.alphaInfo == .noneSkipFirst || source.alphaInfo == .noneSkipLast
        return opaque
            ? bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
            : bitmap.representation(using: .png, properties: [:])
    }
}

/// Where a message is written: the text and the images attached; under them what the Server runs,
/// how much of the model's context the next prompt takes, how much the model thinks, and Send, or
/// Stop while a reply streams.
struct Composer: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var colorScheme
    @Bindable var draft: Draft
    @State private var picking = false

    var body: some View {
        let images = state.server.acceptsImages
        VStack(alignment: .leading, spacing: 10) {
            if let notice = draft.notice, notice.conversation == state.chats.selection {
                // As a reply says why it stopped short.
                Label {
                    Text(notice.text).foregroundStyle(.secondary)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor)
                }
                .font(.system(size: 12))
            }
            if !draft.attachments.isEmpty {
                Attachments(draft: draft)
            }
            ComposerField(text: $draft.text, focusRequests: draft.focusRequests,
                          yields: state.terminalShows,
                          placeholder: "Message Splash", submit: state.sendDraft, cancel: stop, acceptsImages: images,
                          attach: { draft.attach($0, unreadable: $1) }, attachPDFs: { draft.attach(pdfs: $0) })
            // As the server's stage changes them (`MainView`), a control that goes goes at once, and
            // one that comes fades in once those beside it have moved, as a row of a page does: as
            // a switch began, the model's words slid over Attach Images fading out.
            HStack(spacing: 6) {
                // Only for a model that sees images and reads PDFs.
                if images {
                    Button { picking = true } label: { Image(systemName: "plus") }
                        .foregroundStyle(.secondary)
                        .help("Attach Images or PDFs")
                        .accessibilityLabel("Attach Images or PDFs")
                        .transition(Motion.comingAndGoing)
                }
                EngineSlot()
                Spacer(minLength: 12)
                if state.showsContext {
                    ContextRing(usage: state.contextUsage)
                        .transition(Motion.comingAndGoing)
                }
                EffortMenu()
                // Send turns into Stop at once, as a control's title changes, or as the composer moves
                // with a new chat's first message: on an animation of its own, the button moved
                // apart from the composer, ahead of it.
                Button { if streaming { state.chats.stop() } else { state.sendDraft() } } label: {
                    Image(systemName: streaming ? "stop.fill" : "arrow.up")
                        .font(.system(size: streaming ? 10 : 13, weight: streaming ? .bold : .semibold))
                        .contentTransition(.symbolEffect(.replace))
                }
                .buttonStyle(InkCircle())
                .disabled(!streaming && !state.canSendDraft)
                .keyboardShortcut(streaming ? KeyboardShortcut(".", modifiers: .command) : nil)
                .help(streaming ? "Stop" : help)
                .accessibilityLabel(streaming ? "Stop" : "Send")
            }
            .buttonStyle(.hover)
            // The first control's glyph on the text's edge.
            .padding(.leading, -6)
            .animation(Motion.settle, value: state.showsContext)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(colorScheme == .dark ? AnyShapeStyle(.fill.tertiary) : AnyShapeStyle(.background),
                    in: .rect(cornerRadius: 22))
        .overlay { RoundedRectangle(cornerRadius: 22).strokeBorder(.separator) }
        .shadow(color: .black.opacity(0.06), radius: 14, y: 4)
        .fileImporter(isPresented: $picking, allowedContentTypes: [.image, .pdf], allowsMultipleSelection: true) { result in
            let urls = (try? result.get()) ?? []
            let pdfs = urls.filter { UTType(filenameExtension: $0.pathExtension)?.conforms(to: .pdf) == true }
            let files = urls.filter { !pdfs.contains($0) }
            let images = files.compactMap(NSImage.init(contentsOf:))
            draft.attach(images, unreadable: files.count - images.count)
            draft.attach(pdfs: pdfs)
        }
        // The ring counts the next prompt as it is written.
        .onChange(of: state.contextRequest, initial: true) { _, request in
            state.chatContext.refresh(request)
        }
        // A file that can't be attached is told in the chat it was attached in.
        .onChange(of: state.chats.selection, initial: true) { _, open in draft.conversation = open }
    }

    /// Whether the open conversation's reply streams.
    private var streaming: Bool {
        state.chats.streaming != nil && state.chats.streaming == state.chats.current?.id
    }

    private var help: String {
        if !state.server.answers { return "Available once a model is running" }
        if draft.attaching > 0 { return "Reading the attachments…" }
        if let streaming = state.chats.streaming, streaming != state.chats.selection {
            let title = state.chats.conversations.first { $0.id == streaming }?.title ?? ""
            return title.isEmpty ? "Wait for the reply in another chat" : "Wait for the reply in “\(title)”"
        }
        if let problem = state.attachmentProblem { return problem }
        if let note = state.overflow?.note { return note }
        return "Send"
    }

    /// Stops the open conversation's reply, the way Escape cancels; false when none streams.
    private func stop() -> Bool {
        guard streaming else { return false }
        state.chats.stop()
        return true
    }
}

/// How much of the model's context the next prompt takes, as a ring filling clockwise: in the
/// secondary color while it leaves room for a reply, orange once it leaves less than the reply
/// allowance, red once the context cannot hold it; empty, with no help, until the chat's prompt is
/// first counted. Its help gives the exact figures.
private struct ContextRing: View {
    let usage: ContextUsage?

    var body: some View {
        Ring(fraction: usage?.fraction ?? 0, tint: tint, size: 14)
            // Room around the ring for the pointer to find its help.
            .padding(5)
            .contentShape(.rect)
            .animation(Motion.settle, value: usage)
            .help(usage?.explanation ?? "")
            .accessibilityElement()
            .accessibilityLabel("Context")
            .accessibilityValue(usage?.explanation ?? "")
    }

    private var tint: AnyShapeStyle {
        switch usage?.level ?? .normal {
        case .normal: AnyShapeStyle(.secondary)
        case .tight: AnyShapeStyle(.orange)
        case .full: AnyShapeStyle(.red)
        }
    }
}

/// How full something is, as a ring that `fraction`, from 0 to 1, fills in `tint` from the top
/// clockwise, over a faint track, `size` points across: the context in the Chat's composer
/// (`ContextRing`), and a download in the Server's button (`ServerButton`).
struct Ring<Tint: ShapeStyle>: View {
    let fraction: Double
    let tint: Tint
    let size: CGFloat

    var body: some View {
        ZStack {
            Circle().stroke(.quaternary, lineWidth: 2)
            Circle()
                .trim(from: 0, to: fraction)
                .stroke(tint, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: size, height: size)
    }
}

/// The send and stop button: a circle, the composer's one solid element, in the mark's color, the
/// app's one accent, while it can send or stop; gray while it can't.
private struct InkCircle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.background)
            .frame(width: 28, height: 28)
            .background(isEnabled ? AnyShapeStyle(Color.splash) : AnyShapeStyle(.tertiary), in: .circle)
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

/// What the Server runs, at the composer's start: the model's name and the mark showing what it
/// does, or why nothing answers. It opens the models on this Mac, the one that runs checked, to
/// start one or switch to it, as the menu bar's panel does (`AppState.choose`), each where it can
/// start now (`AppState.canStart`), and Manage Models…, which opens the Server; with no model on
/// this Mac, or none it can start now, as while the Keychain refuses the API key, it opens the
/// Server itself.
private struct EngineSlot: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let server = state.server
        let name = server.model.map(state.displayName(of:)) ?? ""
        let words = switch server.phase {
        case .ready where !server.recovering: name
        case .stopped: state.installed.isEmpty ? "No model on this Mac" : "No model running"
        default: state.status
        }
        let mark = StatusMark(look: state.serverStatus.look, since: server.since, size: 12)
        if !state.installed.contains(where: { state.canStart($0.id) }) {
            Button { state.section = .server } label: {
                HStack(spacing: 6) {
                    mark
                    Text(words).foregroundStyle(.secondary).contentTransition(.identity)
                    Text("Open Server").foregroundStyle(.tint)
                }
                .font(.callout)
                .lineLimit(1)
            }
            .help("Open Server (⌘1)")
        } else {
            // The models by their ids, and Manage Models… by an empty one, which no model has.
            let models = state.installed.map { model in
                PopupItem(id: model.id, title: state.displayName(of: model.id), disabled: !state.canStart(model.id))
            }
            PopupMenu(header: "Model", items: models + [PopupItem(id: "", title: "Manage Models…", separated: true)],
                      chosen: state.checkedModel, above: true) { choice in
                if choice.isEmpty { state.section = .server } else { state.choose(choice) }
            } label: {
                HStack(spacing: 6) {
                    mark
                    // In place, as a control's title changes: cross-faded as the server's stage
                    // changed, the words of a switch drew over those of the model before.
                    Text(words).foregroundStyle(.secondary).contentTransition(.identity)
                    MenuChevron()
                }
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                // The mark, the words and the chevron change places within the slot at once as
                // the words change, while the slot moves whole with the controls beside it: on the
                // server's stage's animation, the chevron trailed words that grew and drew over
                // them, and the mark trailed words that moved and drew over their first letter.
                // Only then, so that the mark's pose, which follows in an update of its own, keeps
                // its animation.
                .transaction(value: words) { $0.animation = nil }
                .geometryGroup()
            }
            .help("Start or Switch Models")
            .accessibilityLabel("Model")
            .accessibilityValue(words)
        }
    }
}

/// How much the model thinks, which a click changes, as its template tells efforts apart (its
/// levels, or thinking on or off), its default marked. A model that decides itself shows nothing.
private struct EffortMenu: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if state.server.phase == .ready, let reasoning = state.server.reasoning, let thinks = reasoning.thinks {
            let levels = !reasoning.efforts.isEmpty
            // The choice that leaves the effort to the model: its default, by name where it has one.
            let initial = !thinks ? "none" : levels ? reasoning.default ?? "" : "on"
            let choices = (levels ? reasoning.efforts.reversed() : ["on"])
                + (levels && reasoning.default == nil && thinks ? [""] : []) + ["none"]
            let chosen = state.effort.isEmpty ? initial : !levels && state.effort != "none" ? "on" : state.effort
            PopupMenu(header: "Thinking", items: choices.map { choice in
                // The default's own item is named Default already.
                PopupItem(id: choice, title: Self.title(choice), badge: choice == initial && !choice.isEmpty ? "Default" : nil)
            }, chosen: chosen, above: true) { choice in
                // On, for a model whose thinking is only on or off, is any effort but none.
                state.effort = choice == initial ? "" : choice == "on" ? "medium" : choice
            } label: {
                HStack(spacing: 4) {
                    Text(Self.title(chosen))
                    MenuChevron()
                }
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
            }
            .fixedSize()
            .transition(Motion.comingAndGoing)
            .help("How much the model thinks before it answers")
            .accessibilityLabel("Thinking")
            .accessibilityValue(Self.title(chosen))
        }
    }

    /// An effort's name.
    private static func title(_ effort: String) -> String {
        switch effort {
        case "": "Default"
        case "on": "Thinking"
        case "none": "No Thinking"
        case "xhigh": "Extra High"
        default: effort.capitalized
        }
    }
}

/// What is attached to a draft, each with a button that takes it off: an image as itself, a PDF as
/// its name and pages, in red where the model's limits leave it out. Each has a hairline, so that
/// a white one keeps its edge on the composer, and its button a solid disc, which stands out over
/// any image in either appearance.
private struct Attachments: View {
    @Environment(AppState.self) private var state
    let draft: Draft

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(draft.attachments.enumerated()), id: \.element.id) { index, attachment in
                    ZStack(alignment: .topTrailing) {
                        Group {
                            if attachment.kind == .pdf {
                                document(attachment)
                            } else if let image = draft.thumbnails[attachment.id] {
                                Image(nsImage: image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 56, height: 56)
                            }
                        }
                        .clipShape(.rect(cornerRadius: 10))
                        .overlay { RoundedRectangle(cornerRadius: 10).strokeBorder(.separator) }
                        Button { draft.remove(at: index) } label: {
                            Image(systemName: "xmark.circle.fill")
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.6))
                        }
                        .buttonStyle(.borderless)
                        .offset(x: 6, y: -6)
                        .accessibilityLabel(attachment.kind == .pdf ? "Remove PDF" : "Remove Image")
                    }
                }
            }
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
    }
}

extension Attachments {
    /// A PDF's chip: its glyph, name and pages, which show in red where the model's limits leave
    /// it out by itself.
    private func document(_ attachment: Attachment) -> some View {
        let over = AppState.attachmentProblem([attachment], vision: true, limits: state.server.limits) != nil
        return HStack(spacing: 8) {
            Image(systemName: "doc.text").font(.system(size: 22, weight: .light)).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(attachment.name ?? "PDF").font(.callout).lineLimit(1).truncationMode(.middle)
                Text(attachment.details)
                    .font(.caption)
                    .foregroundStyle(over ? AnyShapeStyle(.red) : AnyShapeStyle(.secondary))
            }
        }
        .padding(.horizontal, 10)
        .frame(maxWidth: 200, minHeight: 56, maxHeight: 56, alignment: .leading)
        .background(.quaternary.opacity(0.5))
    }
}

/// The message's text view: Return sends and Shift-Return starts a new line, and Escape stops the
/// reply, after the input method has finished what it composes; pasted and dropped images and PDF
/// files attach when the model takes them. It is as tall as its text, from one line to 220 points, past which
/// the text scrolls.
private struct ComposerField: NSViewRepresentable {
    @Binding var text: String
    let focusRequests: Int
    /// Whether a terminal of the drawer shows, which keeps the keyboard as the field shows
    /// (`AppState.terminalShows`).
    let yields: Bool
    let placeholder: String
    let submit: () -> Void
    let cancel: () -> Bool
    /// Whether the model takes images and PDFs.
    let acceptsImages: Bool
    /// Takes images, and how many of those chosen could not be read.
    let attach: ([NSImage], Int) -> Void
    let attachPDFs: ([URL]) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    /// The text's height as the view lays it out, at its width; a change of its height asks again
    /// (`MessageTextView.setFrameSize`).
    func sizeThatFits(_ proposal: ProposedViewSize, nsView scroll: NSScrollView, context: Context) -> CGSize? {
        guard let view = scroll.documentView as? NSTextView, let container = view.textContainer,
              let layout = view.layoutManager else { return nil }
        layout.ensureLayout(for: container)
        let height = layout.usedRect(for: container).height + view.textContainerInset.height * 2
        let width = proposal.width.flatMap { $0.isFinite ? $0 : nil } ?? scroll.frame.width
        return CGSize(width: width, height: min(max(height, 22), 220))
    }

    func makeNSView(context: Context) -> NSScrollView {
        // TextKit 1: the height is measured with its layout manager.
        let view = MessageTextView(usingTextLayoutManager: false)
        view.delegate = context.coordinator
        view.coordinator = context.coordinator
        view.placeholder = placeholder
        view.font = .systemFont(ofSize: 15)
        view.textColor = .labelColor
        view.drawsBackground = false
        view.isRichText = false
        view.allowsUndo = true
        view.isAutomaticQuoteSubstitutionEnabled = false
        view.isAutomaticDashSubstitutionEnabled = false
        view.isAutomaticTextReplacementEnabled = false
        view.isContinuousSpellCheckingEnabled = true
        view.textContainerInset = NSSize(width: 0, height: 2)
        // As a message reads once sent.
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineSpacing = 3
        view.defaultParagraphStyle = paragraph
        view.typingAttributes[.paragraphStyle] = paragraph
        view.textContainer?.lineFragmentPadding = 0
        view.isVerticallyResizable = true
        view.autoresizingMask = [.width]
        view.registerForDraggedTypes([.fileURL, .png, .tiff])
        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.documentView = view
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.field = self
        guard let view = scroll.documentView as? MessageTextView else { return }
        // Only a change made elsewhere, as sending the message clears it, goes to the view: the
        // view's own edits are in `text` already, and what the input method composes is in the
        // view alone until it is committed.
        if text != context.coordinator.published {
            context.coordinator.published = text
            // A change made here is no edit to undo.
            view.string = text
            context.coordinator.undo.removeAllActions()
        }
        if context.coordinator.focusRequests != focusRequests {
            // A request made before the field showed, as for the Chat a window opens on, gives way
            // to a terminal shown with it; one made since takes the keyboard.
            let pending = !context.coordinator.shown
            context.coordinator.focusRequests = focusRequests
            if !(pending && yields) {
                DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
            }
        }
        context.coordinator.shown = true
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var field: ComposerField
        var focusRequests = 0
        /// Whether the field has shown, past its first update.
        var shown = false
        /// The text the view and the draft last agreed on.
        var published = ""
        /// The field's own edits, which a message sent clears.
        let undo = UndoManager()

        init(_ field: ComposerField) {
            self.field = field
        }

        func undoManager(for view: NSTextView) -> UndoManager? { undo }

        func textDidChange(_ notification: Notification) {
            guard let view = notification.object as? NSTextView else { return }
            published = view.string
            field.text = view.string
        }

        func textView(_ view: NSTextView, doCommandBy selector: Selector) -> Bool {
            switch selector {
            case #selector(NSResponder.insertNewline(_:)):
                if NSApp.currentEvent?.modifierFlags.contains(.shift) == true {
                    view.insertNewlineIgnoringFieldEditor(nil)
                } else {
                    field.submit()
                }
                return true
            case #selector(NSResponder.cancelOperation(_:)):
                // Stops a reply; else does nothing, rather than open the text view's completions.
                _ = field.cancel()
                return true
            default:
                return false
            }
        }
    }
}

private final class MessageTextView: NSTextView {
    weak var coordinator: ComposerField.Coordinator?
    var placeholder = ""

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty else { return }
        let attributes: [NSAttributedString.Key: Any] = [.font: font ?? .systemFont(ofSize: 15),
                                                         .foregroundColor: NSColor.placeholderTextColor]
        placeholder.draw(at: NSPoint(x: textContainerInset.width, y: textContainerInset.height), withAttributes: attributes)
    }

    override func didChangeText() {
        super.didChangeText()
        needsDisplay = true
    }

    // The view is as tall as its text laid out: an edit, what the input method composes, or a new
    // width that wraps it differently. The field takes the new height in the layout that follows,
    // before the window shows it, rather than scrolling its text a line meanwhile.
    override func setFrameSize(_ size: NSSize) {
        let height = frame.height
        super.setFrameSize(size)
        if frame.height != height { enclosingScrollView?.invalidateIntrinsicContentSize() }
    }

    // A plain-text view turns Paste off for a clipboard of image data alone.
    override func validateUserInterfaceItem(_ item: any NSValidatedUserInterfaceItem) -> Bool {
        if item.action == #selector(paste(_:)), holdsAttachments(.general) { return true }
        return super.validateUserInterfaceItem(item)
    }

    override func paste(_ sender: Any?) {
        if attach(from: .general) { return }
        pasteAsPlainText(sender)
    }

    // A text view takes only text it can read; images and PDFs it takes here, when the model
    // takes them.
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        holdsAttachments(sender.draggingPasteboard) ? .copy : super.draggingEntered(sender)
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        holdsAttachments(sender.draggingPasteboard) ? .copy : super.draggingUpdated(sender)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        attach(from: sender.draggingPasteboard) || super.performDragOperation(sender)
    }

    /// Attaches what `pasteboard` holds to attach (`holdsAttachments`): its image and PDF files,
    /// or, with none, its images; false when it holds none.
    private func attach(from pasteboard: NSPasteboard) -> Bool {
        guard holdsAttachments(pasteboard) else { return false }
        let pdfs = (pasteboard.readObjects(forClasses: [NSURL.self], options: Self.pdfFiles) as? [URL]) ?? []
        // A file comes with a picture of itself, which is no image to attach.
        let files = pasteboard.canReadObject(forClasses: [NSURL.self], options: Self.files)
            ? (pasteboard.readObjects(forClasses: [NSURL.self], options: Self.imageFiles) as? [URL]) ?? []
            : nil
        let images = files?.compactMap(NSImage.init(contentsOf:)) ?? (pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage]) ?? []
        let unreadable = (files?.count ?? images.count) - images.count
        guard !images.isEmpty || !pdfs.isEmpty || unreadable > 0 else { return false }
        if !images.isEmpty || unreadable > 0 { coordinator?.field.attach(images, unreadable) }
        if !pdfs.isEmpty { coordinator?.field.attachPDFs(pdfs) }
        return true
    }

    /// Whether `pasteboard` holds something to attach, when the model takes images and PDFs: image
    /// or PDF files, or images without text, as copied rich text comes with a picture of itself.
    /// Told by their types alone: Paste's menu item and every move of a drag ask, so nothing is
    /// read until they are attached.
    private func holdsAttachments(_ pasteboard: NSPasteboard) -> Bool {
        guard coordinator?.field.acceptsImages == true else { return false }
        if pasteboard.canReadObject(forClasses: [NSURL.self], options: Self.files) { return true }
        return pasteboard.string(forType: .string) == nil && pasteboard.canReadObject(forClasses: [NSImage.self])
    }

    private static let imageFiles: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingContentsConformToTypes: [UTType.image.identifier]]
    private static let pdfFiles: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingContentsConformToTypes: [UTType.pdf.identifier]]
    private static let files: [NSPasteboard.ReadingOptionKey: Any] =
        [.urlReadingContentsConformToTypes: [UTType.image.identifier, UTType.pdf.identifier]]
}
