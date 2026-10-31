import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The message being written: its text and the images attached to it.
@MainActor @Observable
final class Draft {
    var text = ""
    private(set) var images: [Data] = []
    /// Asks the text field for the keyboard; counts the requests.
    private(set) var focusRequests = 0

    var isEmpty: Bool { text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && images.isEmpty }

    func focus() { focusRequests += 1 }

    /// Attaches `images` in a form the server takes: at most 2048 pixels on the long side (its
    /// limit is about four million pixels), PNG when they have transparency and JPEG otherwise.
    func attach(_ images: [NSImage]) {
        self.images += images.compactMap(Self.encoded)
    }

    func remove(at index: Int) { images.remove(at: index) }

    /// Takes the message, leaving the draft empty.
    func take() -> (text: String, images: [Data]) {
        defer { text = ""; images = [] }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), images)
    }

    private static func encoded(_ image: NSImage) -> Data? {
        guard let source = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
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
    @State private var height: CGFloat = 22

    var body: some View {
        let images = state.server.acceptsImages
        VStack(alignment: .leading, spacing: 10) {
            if !draft.images.isEmpty {
                Attachments(draft: draft)
            }
            ComposerField(text: $draft.text, height: $height, focusRequests: draft.focusRequests,
                          placeholder: "Message Splash", submit: state.sendDraft, cancel: stop, acceptsImages: images,
                          attach: draft.attach)
                .frame(height: min(max(height, 22), 220))
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
        if !draft.images.isEmpty && !state.server.acceptsImages { return "This model can’t see images" }
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

/// The send and stop button: a circle in the label color, the composer's one solid element, faded
/// as the app's controls fade while it can't be used.
private struct InkCircle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.background)
            .frame(width: 28, height: 28)
            // The mark's color while it sends or stops, the app's one accent; gray while it can't.
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
        case .ready where server.recovering: state.status
        case .ready: name
        case .starting, .stopping: state.status
        case .failed: "\(name) ran into a problem"
        case .stopped: state.installed.isEmpty ? "No model on this Mac" : "No model running"
        }
        let blocked = if case .failed = server.phase { true } else { server.phase == .stopped }
        Button { state.section = .server } label: {
            HStack(spacing: 6) {
                StatusMark(look: state.markLook, since: server.since, size: 12)
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

/// The images attached to a draft, each with a button that takes it off. Each has a hairline, so
/// that a white one keeps its edge on the composer, and its button a solid disc, which stands out
/// over any image in either appearance.
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
            }
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
    }
}

/// The message's text view: Return sends and Shift-Return starts a new line, and Escape stops the
/// reply, after the input method has finished what it composes; pasted and dropped images attach
/// when the model sees images.
private struct ComposerField: NSViewRepresentable {
    @Binding var text: String
    @Binding var height: CGFloat
    let focusRequests: Int
    let placeholder: String
    let submit: () -> Void
    let cancel: () -> Bool
    let acceptsImages: Bool
    let attach: ([NSImage]) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

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
            context.coordinator.measure(view)
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
            measure(view)
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

        func measure(_ view: NSTextView) {
            guard let container = view.textContainer, let layout = view.layoutManager else { return }
            layout.ensureLayout(for: container)
            let height = layout.usedRect(for: container).height + view.textContainerInset.height * 2
            // Not while SwiftUI updates the view, which measures too.
            DispatchQueue.main.async { [self] in if field.height != height { field.height = height } }
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

    // A plain-text view turns Paste off for a clipboard of image data alone.
    override func validateUserInterfaceItem(_ item: any NSValidatedUserInterfaceItem) -> Bool {
        if item.action == #selector(paste(_:)), !images(in: .general).isEmpty { return true }
        return super.validateUserInterfaceItem(item)
    }

    override func paste(_ sender: Any?) {
        if attach(from: .general) { return }
        pasteAsPlainText(sender)
    }

    // A text view takes only text it can read; images it takes here, when the model sees them.
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        images(in: sender.draggingPasteboard).isEmpty ? super.draggingEntered(sender) : .copy
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        images(in: sender.draggingPasteboard).isEmpty ? super.draggingUpdated(sender) : .copy
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

    /// The images `pasteboard` holds, image files too, when the model sees images; none for text
    /// that comes with a picture of itself, as copied rich text does.
    private func images(in pasteboard: NSPasteboard) -> [NSImage] {
        guard coordinator?.field.acceptsImages == true else { return [] }
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingContentsConformToTypes: [UTType.image.identifier]]) as? [URL]) ?? []
        if !files.isEmpty { return files.compactMap(NSImage.init(contentsOf:)) }
        guard pasteboard.string(forType: .string) == nil else { return [] }
        return (pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage]) ?? []
    }
}
