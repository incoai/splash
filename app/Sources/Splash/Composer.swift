import AppKit
import PDFKit
import SwiftUI
import UniformTypeIdentifiers

/// The message being written: its text, the images and the PDFs attached to it.
@MainActor @Observable
final class Draft {
    var text = ""
    private(set) var images: [Data] = []
    private(set) var documents: [ChatDocument] = []
    /// Why the last file picked did not attach.
    private(set) var notice: String?
    /// How many attached images are still being encoded, which join `images` once they are.
    private(set) var attaching = 0
    /// The attachment that joins `images` last, which the next one waits for.
    private var lastAttachment: Task<Void, Never>?
    /// Asks the text field for the keyboard; counts the requests.
    private(set) var focusRequests = 0

    var isEmpty: Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && images.isEmpty && documents.isEmpty
    }

    func focus() { focusRequests += 1 }

    /// Attaches `images` in a form the server takes: at most 2048 pixels on the long side (its
    /// limit is about four million pixels), PNG when they have transparency and JPEG otherwise.
    /// They are encoded away from the main thread, which a photo would hold for a tenth of a second
    /// or more, and join the draft in the order they were attached.
    func attach(_ images: [NSImage]) {
        notice = nil
        let sources = images.compactMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
        guard !sources.isEmpty else { return }
        attaching += sources.count
        let encoding = Task.detached(priority: .userInitiated) { sources.compactMap(Self.encoded) }
        let previous = lastAttachment
        lastAttachment = Task {
            await previous?.value
            self.images += await encoding.value
            attaching -= sources.count
        }
    }

    /// Attaches the images and PDFs among `urls`, reading a PDF whole as the server takes it. A
    /// file the server would refuse stays out, and `notice` says why.
    func attach(files urls: [URL]) {
        notice = nil
        var pictures: [NSImage] = []
        for url in urls {
            let type = try? url.resourceValues(forKeys: [.contentTypeKey]).contentType
            if type?.conforms(to: .pdf) == true || url.pathExtension.caseInsensitiveCompare("pdf") == .orderedSame {
                if let document = document(at: url) { documents.append(document) }
            } else if let image = NSImage(contentsOf: url) {
                pictures.append(image)
            } else {
                notice = "\(url.lastPathComponent) can’t be read."
            }
        }
        // The images keep the notice, if a file before them left one.
        let kept = notice
        attach(pictures)
        notice = kept
    }

    private func document(at url: URL) -> ChatDocument? {
        let name = url.lastPathComponent
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size >= ChatDocument.maxBytes {
            notice = "\(name) must be under \(ChatDocument.maxBytes / (1024 * 1024)) MB."
            return nil
        }
        guard let data = try? Data(contentsOf: url), let pdf = PDFDocument(data: data) else {
            notice = "\(name) can’t be read."
            return nil
        }
        if pdf.isLocked {
            notice = "\(name) needs a password."
            return nil
        }
        if pdf.pageCount > ChatDocument.maxPages {
            notice = "\(name) has more than \(ChatDocument.maxPages) pages."
            return nil
        }
        return ChatDocument(name: name, data: data)
    }

    func remove(at index: Int) { images.remove(at: index) }
    func removeDocument(at index: Int) { documents.remove(at: index) }

    /// Takes the message, leaving the draft empty.
    func take() -> (text: String, images: [Data], documents: [ChatDocument]) {
        defer { text = ""; images = []; documents = []; notice = nil }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), images, documents)
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

/// Where a message is written: the text and the attachments; under them what the Server runs,
/// how much of the model's context the next prompt takes, how much the model thinks, and Send, or
/// Stop while a reply streams.
struct Composer: View {
    @Environment(AppState.self) private var state
    @Environment(\.colorScheme) private var colorScheme
    @Bindable var draft: Draft
    @State private var picking = false

    var body: some View {
        let images = state.server.acceptsImages
        let pdfs = state.server.acceptsPDFs
        VStack(alignment: .leading, spacing: 10) {
            if !draft.images.isEmpty || !draft.documents.isEmpty {
                Attachments(draft: draft)
            }
            if let notice = draft.notice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            ComposerField(text: $draft.text, focusRequests: draft.focusRequests,
                          placeholder: "Message Splash", submit: state.sendDraft, cancel: stop,
                          acceptsImages: images, acceptsPDFs: pdfs,
                          attach: draft.attach, attachFiles: draft.attach(files:))
            HStack(spacing: 6) {
                // Only for a model that takes images or PDFs.
                if images || pdfs {
                    Button { picking = true } label: { Image(systemName: "plus") }
                        .foregroundStyle(.secondary)
                        .help(attachHelp)
                        .accessibilityLabel(attachHelp)
                }
                EngineSlot()
                Spacer(minLength: 12)
                if let usage = state.contextUsage {
                    ContextRing(usage: usage)
                        .transition(.opacity)
                }
                EffortMenu()
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
                .animation(Motion.quick, value: streaming)
            }
            .buttonStyle(.hover)
            // The first control's glyph on the text's edge.
            .padding(.leading, -6)
            .animation(Motion.settle, value: state.contextUsage == nil)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(colorScheme == .dark ? AnyShapeStyle(.fill.tertiary) : AnyShapeStyle(.background),
                    in: .rect(cornerRadius: 22))
        .overlay { RoundedRectangle(cornerRadius: 22).strokeBorder(.separator) }
        .shadow(color: .black.opacity(0.06), radius: 14, y: 4)
        .fileImporter(isPresented: $picking, allowedContentTypes: allowedTypes, allowsMultipleSelection: true) { result in
            draft.attach(files: (try? result.get()) ?? [])
        }
        // The ring counts the next prompt as it is written.
        .onChange(of: state.contextRequest, initial: true) { _, request in
            state.chatContext.refresh(request)
        }
    }

    /// Whether the open conversation's reply streams.
    private var streaming: Bool {
        state.chats.streaming != nil && state.chats.streaming == state.chats.current?.id
    }

    /// What the file picker takes: the kinds of attachment the model accepts.
    private var allowedTypes: [UTType] {
        var types: [UTType] = []
        if state.server.acceptsImages { types.append(.image) }
        if state.server.acceptsPDFs { types.append(.pdf) }
        return types.isEmpty ? [.data] : types
    }

    private var attachHelp: String {
        switch (state.server.acceptsImages, state.server.acceptsPDFs) {
        case (true, true): "Attach Images or PDFs"
        case (true, false): "Attach Images"
        case (false, true): "Attach PDFs"
        case (false, false): "Attach Files"
        }
    }

    private var help: String {
        if !state.server.answers { return "Available once a model is running" }
        if !draft.images.isEmpty && !state.server.acceptsImages { return "This model can’t see images" }
        if !draft.documents.isEmpty && !state.server.acceptsPDFs { return "This model can’t read PDFs" }
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
/// allowance, red once the context cannot hold it. Its help gives the exact figures.
private struct ContextRing: View {
    let usage: ContextUsage

    var body: some View {
        Ring(fraction: usage.fraction, tint: tint, size: 14)
            // Room around the ring for the pointer to find its help.
            .padding(5)
            .contentShape(.rect)
            .animation(Motion.settle, value: usage)
            .help(usage.explanation)
            .accessibilityElement()
            .accessibilityLabel("Context")
            .accessibilityValue(usage.explanation)
    }

    private var tint: AnyShapeStyle {
        switch usage.level {
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

/// What the Server runs, at the composer's start, which opens the Server: the model's name, the
/// mark showing what it does; or why nothing answers, with Open Server. Models change on the
/// Server only.
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
        let blocked = if case .failed = server.phase { true } else { server.phase == .stopped }
        Button { state.section = .server } label: {
            HStack(spacing: 6) {
                StatusMark(look: state.serverStatus.look, since: server.since, size: 12)
                Text(words).foregroundStyle(.secondary)
                if blocked { Text("Open Server").foregroundStyle(.tint) }
            }
            .font(.callout)
            .lineLimit(1)
            .truncationMode(.middle)
        }
        .help("Open Server (⌘1)")
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

/// The images and PDFs attached to a draft, each with a button that takes it off. Each image has a
/// hairline, so that a white one keeps its edge on the composer, and its button a solid disc, which
/// stands out over any image in either appearance.
private struct Attachments: View {
    let draft: Draft

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(draft.images.enumerated()), id: \.offset) { index, data in
                    ZStack(alignment: .topTrailing) {
                        if let image = NSImage(data: data) {
                            Image(nsImage: image)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 56, height: 56)
                                .clipShape(.rect(cornerRadius: 10))
                                .overlay { RoundedRectangle(cornerRadius: 10).strokeBorder(.separator) }
                        }
                        Button { draft.remove(at: index) } label: {
                            Image(systemName: "xmark.circle.fill")
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.6))
                        }
                        .buttonStyle(.borderless)
                        .offset(x: 6, y: -6)
                        .accessibilityLabel("Remove Image")
                    }
                }
                ForEach(Array(draft.documents.enumerated()), id: \.offset) { index, document in
                    ZStack(alignment: .topTrailing) {
                        VStack(spacing: 3) {
                            Image(systemName: "doc.text")
                            Text(document.name)
                                .font(.caption2)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        .frame(width: 56, height: 56)
                        .background(.quaternary, in: .rect(cornerRadius: 10))
                        .help(document.name)
                        Button { draft.removeDocument(at: index) } label: {
                            Image(systemName: "xmark.circle.fill")
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.6))
                        }
                        .buttonStyle(.borderless)
                        .offset(x: 6, y: -6)
                        .accessibilityLabel("Remove PDF")
                    }
                }
            }
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
    }
}

/// The message's text view: Return sends and Shift-Return starts a new line, and Escape stops the
/// reply, after the input method has finished what it composes; pasted and dropped images and PDFs
/// attach when the model takes them. It is as tall as its text, from one line to 220 points, past which
/// the text scrolls.
private struct ComposerField: NSViewRepresentable {
    @Binding var text: String
    let focusRequests: Int
    let placeholder: String
    let submit: () -> Void
    let cancel: () -> Bool
    let acceptsImages: Bool
    let acceptsPDFs: Bool
    let attach: ([NSImage]) -> Void
    let attachFiles: ([URL]) -> Void

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
            context.coordinator.focusRequests = focusRequests
            DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
        }
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var field: ComposerField
        var focusRequests = 0
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
                return field.cancel()
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

    // A text view takes only text it can read; images and PDFs it takes here, when the model does.
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        holdsAttachments(sender.draggingPasteboard) ? .copy : super.draggingEntered(sender)
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        holdsAttachments(sender.draggingPasteboard) ? .copy : super.draggingUpdated(sender)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        attach(from: sender.draggingPasteboard) || super.performDragOperation(sender)
    }

    /// Attaches what `pasteboard` holds (`holdsAttachments`), files read; false when it holds
    /// nothing to attach.
    private func attach(from pasteboard: NSPasteboard) -> Bool {
        guard holdsAttachments(pasteboard), let field = coordinator?.field else { return false }
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: fileOptions) as? [URL]) ?? []
        if !files.isEmpty {
            field.attachFiles(files)
            return true
        }
        let images = (pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage]) ?? []
        guard !images.isEmpty else { return false }
        field.attach(images)
        return true
    }

    /// Whether `pasteboard` holds what to attach, when the model takes it: image and PDF files, or
    /// images without text, as copied rich text comes with a picture of itself. Told by their
    /// types alone: Paste's menu item and every move of a drag ask, so nothing is read until they
    /// are attached.
    private func holdsAttachments(_ pasteboard: NSPasteboard) -> Bool {
        guard let field = coordinator?.field, field.acceptsImages || field.acceptsPDFs else { return false }
        if pasteboard.canReadObject(forClasses: [NSURL.self], options: fileOptions) { return true }
        return field.acceptsImages && pasteboard.string(forType: .string) == nil
            && pasteboard.canReadObject(forClasses: [NSImage.self])
    }

    /// The files to attach: images and PDFs, those the model takes.
    private var fileOptions: [NSPasteboard.ReadingOptionKey: Any] {
        var types: [String] = []
        if coordinator?.field.acceptsImages == true { types.append(UTType.image.identifier) }
        if coordinator?.field.acceptsPDFs == true { types.append(UTType.pdf.identifier) }
        return [.urlReadingContentsConformToTypes: types]
    }
}
