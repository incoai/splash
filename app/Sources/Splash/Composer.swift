import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The message being written: its text and what is attached to it, kept as files in the store's
/// attachments folder as it attaches (`ChatStore.attachments`), which become the message's once it
/// is sent; one taken off goes.
@MainActor @Observable
final class Draft {
    var text = ""
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
    func attach(_ images: [NSImage]) {
        let sources = images.compactMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
        guard !sources.isEmpty else { return }
        attaching += sources.count
        let files = files
        let encoding = Task.detached(priority: .userInitiated) {
            sources.compactMap { source in Self.encoded(source).flatMap { Self.kept($0, in: files) } }
        }
        let previous = lastAttachment
        lastAttachment = Task {
            await previous?.value
            for (attachment, data) in await encoding.value {
                attachments.append(attachment)
                thumbnails[attachment.id] = NSImage(data: data)
            }
            attaching -= sources.count
        }
    }

    /// Takes attachment `index` off, and its file.
    func remove(at index: Int) {
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
        }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), attachments)
    }

    /// The record of image `data`, kept as a file in `files`, with the data; nil where it can't be
    /// written.
    nonisolated private static func kept(_ data: Data, in files: URL) -> (Attachment, Data)? {
        let attachment = Attachment.image(data)
        try? FileManager.default.createDirectory(at: files, withIntermediateDirectories: true)
        guard (try? data.write(to: files.appending(path: attachment.file), options: .atomic)) != nil else { return nil }
        return (attachment, data)
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
            if !draft.attachments.isEmpty {
                Attachments(draft: draft)
            }
            ComposerField(text: $draft.text, focusRequests: draft.focusRequests,
                          placeholder: "Message Splash", submit: state.sendDraft, cancel: stop, acceptsImages: images,
                          attach: draft.attach)
            HStack(spacing: 6) {
                // Only for a model that sees images.
                if images {
                    Button { picking = true } label: { Image(systemName: "plus") }
                        .foregroundStyle(.secondary)
                        .help("Attach Images")
                        .accessibilityLabel("Attach Images")
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
        .fileImporter(isPresented: $picking, allowedContentTypes: [.image], allowsMultipleSelection: true) { result in
            draft.attach(((try? result.get()) ?? []).compactMap(NSImage.init(contentsOf:)))
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

    private var help: String {
        if !state.server.answers { return "Available once a model is running" }
        if !draft.attachments.isEmpty && !state.server.acceptsImages { return "This model can’t see images" }
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

/// What the Server runs, at the composer's start: the model's name and the mark showing what it
/// does, or why nothing answers. It opens the models on this Mac, the one that runs checked, to
/// start one or switch to it, as the menu bar's panel does (`AppState.choose`), and Manage
/// Models…, which opens the Server; with no model on this Mac, or none it can start now
/// (`AppState.canStart`, the Keychain refusing the API key), it opens the Server itself.
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
        if state.installed.isEmpty || !state.canStart {
            Button { state.section = .server } label: {
                HStack(spacing: 6) {
                    mark
                    Text(words).foregroundStyle(.secondary)
                    Text("Open Server").foregroundStyle(.tint)
                }
                .font(.callout)
                .lineLimit(1)
            }
            .help("Open Server (⌘1)")
        } else {
            // The models by their ids, and Manage Models… by an empty one, which no model has.
            PopupMenu(header: "Model", items: state.installed.map { PopupItem(id: $0.id, title: state.displayName(of: $0.id)) }
                + [PopupItem(id: "", title: "Manage Models…", separated: true)], chosen: state.checkedModel, above: true) { choice in
                if choice.isEmpty { state.section = .server } else { state.choose(choice) }
            } label: {
                HStack(spacing: 6) {
                    mark
                    Text(words).foregroundStyle(.secondary)
                    MenuChevron()
                }
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
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

/// The images attached to a draft, each with a button that takes it off. Each has a hairline, so
/// that a white one keeps its edge on the composer, and its button a solid disc, which stands out
/// over any image in either appearance.
private struct Attachments: View {
    let draft: Draft

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(draft.attachments.enumerated()), id: \.element.id) { index, attachment in
                    ZStack(alignment: .topTrailing) {
                        if let image = draft.thumbnails[attachment.id] {
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
            }
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
    }
}

/// The message's text view: Return sends and Shift-Return starts a new line, and Escape stops the
/// reply, after the input method has finished what it composes; pasted and dropped images attach
/// when the model sees images. It is as tall as its text, from one line to 220 points, past which
/// the text scrolls.
private struct ComposerField: NSViewRepresentable {
    @Binding var text: String
    let focusRequests: Int
    let placeholder: String
    let submit: () -> Void
    let cancel: () -> Bool
    let acceptsImages: Bool
    let attach: ([NSImage]) -> Void

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
        if item.action == #selector(paste(_:)), holdsImages(.general) { return true }
        return super.validateUserInterfaceItem(item)
    }

    override func paste(_ sender: Any?) {
        if attach(from: .general) { return }
        pasteAsPlainText(sender)
    }

    // A text view takes only text it can read; images it takes here, when the model sees them.
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        holdsImages(sender.draggingPasteboard) ? .copy : super.draggingEntered(sender)
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        holdsImages(sender.draggingPasteboard) ? .copy : super.draggingUpdated(sender)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        attach(from: sender.draggingPasteboard) || super.performDragOperation(sender)
    }

    /// Attaches the images `pasteboard` holds; false when it holds none to attach.
    private func attach(from pasteboard: NSPasteboard) -> Bool {
        let images = images(in: pasteboard)
        guard !images.isEmpty else { return false }
        coordinator?.field.attach(images)
        return true
    }

    /// Whether `pasteboard` holds images to attach, when the model sees images: image files, or
    /// images without text, as copied rich text comes with a picture of itself. Told by their
    /// types alone: Paste's menu item and every move of a drag ask, so nothing is read until they
    /// are attached.
    private func holdsImages(_ pasteboard: NSPasteboard) -> Bool {
        guard coordinator?.field.acceptsImages == true else { return false }
        if pasteboard.canReadObject(forClasses: [NSURL.self], options: Self.imageFiles) { return true }
        return pasteboard.string(forType: .string) == nil && pasteboard.canReadObject(forClasses: [NSImage.self])
    }

    /// The images `pasteboard` holds to attach (`holdsImages`), image files read.
    private func images(in pasteboard: NSPasteboard) -> [NSImage] {
        guard holdsImages(pasteboard) else { return [] }
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: Self.imageFiles) as? [URL]) ?? []
        if !files.isEmpty { return files.compactMap(NSImage.init(contentsOf:)) }
        return (pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage]) ?? []
    }

    private static let imageFiles: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingContentsConformToTypes: [UTType.image.identifier]]
}
