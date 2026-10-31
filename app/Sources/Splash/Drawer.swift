import AppKit
import SwiftTerm
import SwiftUI

/// What the drawer under the window's content holds (`WithDrawer`): terminals, each a login shell
/// in the home folder, and the logs of what the app runs. Whether it shows,
/// and its height, are the window's (`AppState.showsDrawer`), as the sidebar's are.
@MainActor @Observable
final class Drawer {
    enum Part: Hashable { case terminal, logs }

    var part = Part.terminal
    private(set) var terminals: [TerminalSession] = []
    var terminal: TerminalSession.ID?
    var log = LogSource.server

    func add(_ session: TerminalSession) {
        terminals.append(session)
        terminal = session.id
        part = .terminal
    }

    /// Ends terminal `id`'s shell and takes its tab away.
    func close(_ id: TerminalSession.ID) {
        guard let index = terminals.firstIndex(where: { $0.id == id }) else { return }
        terminals.remove(at: index).end()
        if terminal == id { terminal = terminals[safe: min(index, terminals.count - 1)]?.id }
    }

    /// Ends every shell, as the app quits.
    func endAll() {
        terminals.forEach { $0.end() }
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

/// A login shell in a pseudo-terminal, as Terminal opens one, in the home folder, with the app's
/// own environment (`Runtime.userEnvironment`), which the login shell completes as in Terminal. Its
/// view outlives the drawer's, so that a terminal keeps what it shows while the drawer is closed or
/// the window shows another part.
@MainActor @Observable
final class TerminalSession: Identifiable {
    let id = UUID()
    /// What the shell calls itself, its title escape, or the home folder's name until it does.
    private(set) var title: String
    private(set) var ended = false
    @ObservationIgnored let view: LocalProcessTerminalView
    @ObservationIgnored private var delegate: Delegate?

    init(environment: [String: String]) {
        let folder = URL.homeDirectory
        title = folder.lastPathComponent
        view = LocalProcessTerminalView(frame: NSRect(x: 0, y: 0, width: 640, height: 280))
        view.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        view.optionAsMetaKey = true
        let delegate = Delegate(self)
        self.delegate = delegate
        view.processDelegate = delegate
        let shell = String(cString: getpwuid(getuid()).pointee.pw_shell)
        var environment = environment
        environment["TERM"] = "xterm-256color"
        environment["COLORTERM"] = "truecolor"
        environment["TERM_PROGRAM"] = "Splash"
        if environment["LANG"] == nil { environment["LANG"] = "en_US.UTF-8" }
        view.startProcess(executable: shell, environment: environment.map { "\($0.key)=\($0.value)" },
                          execName: "-" + (shell as NSString).lastPathComponent, currentDirectory: folder.path)
    }

    /// Paints it in the window's appearance: the text and background of a text view.
    func paint(_ scheme: ColorScheme) {
        let appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua) ?? NSAppearance.currentDrawing()
        appearance.performAsCurrentDrawingAppearance {
            view.nativeForegroundColor = NSColor.textColor.usingColorSpace(.sRGB) ?? .textColor
            view.nativeBackgroundColor = NSColor.textBackgroundColor.usingColorSpace(.sRGB) ?? .textBackgroundColor
            view.caretColor = NSColor.controlAccentColor
        }
    }

    func end() {
        guard !ended else { return }
        ended = true
        view.terminate()
    }

    /// SwiftTerm's delegate, which calls on the main thread.
    @MainActor
    private final class Delegate: NSObject, @preconcurrency LocalProcessTerminalViewDelegate {
        weak var session: TerminalSession?

        init(_ session: TerminalSession) {
            self.session = session
        }

        func sizeChanged(source: LocalProcessTerminalView, newCols: Int, newRows: Int) {}

        func setTerminalTitle(source: LocalProcessTerminalView, title: String) {
            if !title.isEmpty { session?.title = title }
        }

        func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {}

        func processTerminated(source: TerminalView, exitCode: Int32?) {
            session?.ended = true
        }
    }
}

/// A log the drawer shows: the server's, which each start begins again, and the downloads'.
enum LogSource: String, CaseIterable, Identifiable {
    case server = "Server", downloads = "Downloads"

    var id: Self { self }
    @MainActor var url: URL { self == .server ? Server.log : Installer.log }
}

/// The content with the drawer under it, its edge dragged to resize it.
struct WithDrawer<Content: View>: View {
    @Environment(AppState.self) private var state
    @ViewBuilder let content: Content

    var body: some View {
        let shown = state.showsDrawer
        VStack(spacing: 0) {
            content
            // Closed, it keeps its place at no height, its terminals and their views with it.
            DrawerView()
                .frame(height: shown ? state.drawerHeight : 0, alignment: .top)
                .clipped()
                .allowsHitTesting(shown)
                .accessibilityHidden(!shown)
        }
    }
}

private struct DrawerView: View {
    @Environment(AppState.self) private var state
    @State private var dragged: CGFloat?

    var body: some View {
        let drawer = state.drawer
        VStack(spacing: 0) {
            Divider()
                .overlay(alignment: .center) {
                    // The edge to drag: a few points tall, over the divider.
                    Color.clear
                        .frame(height: 8)
                        .contentShape(.rect)
                        .pointerStyle(.frameResize(position: .top))
                        .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                            .onChanged { value in
                                let start = dragged ?? state.drawerHeight
                                if dragged == nil { dragged = start }
                                state.drawerHeight = min(max(start - value.translation.height, 120), 640)
                            }
                            .onEnded { _ in dragged = nil })
                }
            HStack(spacing: 10) {
                Segments(selected: drawer.part, pick: { state.showDrawer($0) }, segments: [
                    .init(title: "Terminal", value: .terminal),
                    .init(title: "Logs", value: .logs),
                ])
                switch drawer.part {
                case .terminal: TerminalTabs()
                case .logs:
                    Segments(selected: drawer.log, pick: { drawer.log = $0 },
                             segments: LogSource.allCases.map { .init(title: $0.rawValue, value: $0) })
                    Spacer(minLength: 0)
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([drawer.log.url])
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.hover)
                    .help("Show in Finder")
                }
                Button {
                    state.toggleDrawer()
                } label: {
                    Image(systemName: "chevron.down")
                }
                .buttonStyle(.hover)
                .help("Hide (⌘J)")
            }
            .padding(.horizontal, 12)
            .frame(height: 36)
            Divider()
            ZStack {
                // Every terminal stays in the window, the one picked on top, so that each keeps
                // its view and what it shows.
                ForEach(drawer.terminals) { session in
                    let shown = drawer.part == .terminal && drawer.terminal == session.id
                    TerminalPane(session: session)
                        .opacity(shown ? 1 : 0)
                        .allowsHitTesting(shown)
                        .accessibilityHidden(!shown)
                }
                if drawer.part == .logs {
                    LogText(url: drawer.log.url)
                        .id(drawer.log)
                }
            }
        }
        .background(.background)
    }
}

/// The terminals' tabs, each with its title and a close button on hover, then New Terminal.
private struct TerminalTabs: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let drawer = state.drawer
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(drawer.terminals) { session in
                    let picked = drawer.terminal == session.id
                    HStack(spacing: 6) {
                        Button { drawer.terminal = session.id } label: {
                            Text(session.title)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(maxWidth: 160)
                                .foregroundStyle(session.ended ? .tertiary : picked ? .primary : .secondary)
                        }
                        .buttonStyle(.plain)
                        Button { drawer.close(session.id) } label: {
                            Image(systemName: "xmark").font(.system(size: 9, weight: .semibold))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Close Terminal")
                    }
                    .font(.callout)
                    .padding(.horizontal, 10)
                    .frame(height: 26)
                    .background(picked ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.clear), in: .capsule)
                }
                Button { state.newTerminal() } label: {
                    Image(systemName: "plus")
                }
                .buttonStyle(.hover)
                .help("New Terminal")
            }
        }
        Spacer(minLength: 0)
    }
}

/// A terminal's view, moved into the drawer each time it is laid out there, in the window's
/// appearance, with a margin of its background around it.
private struct TerminalPane: View {
    let session: TerminalSession

    var body: some View {
        Hosted(session: session)
            .padding(EdgeInsets(top: 6, leading: 10, bottom: 4, trailing: 4))
            .background(Color(nsColor: .textBackgroundColor))
    }

    private struct Hosted: NSViewRepresentable {
        let session: TerminalSession

        func makeNSView(context: Context) -> NSView {
            let container = NSView()
            session.view.removeFromSuperview()
            session.view.frame = container.bounds
            session.view.autoresizingMask = [.width, .height]
            container.addSubview(session.view)
            return container
        }

        func updateNSView(_ container: NSView, context: Context) {
            session.paint(context.environment.colorScheme)
        }
    }
}

/// A log file, read only, as it grows: its last 512 KB at first, then each line as it is written,
/// kept in view while the end is; ⌘F finds in it. A file written again from its start, as the
/// server's log is at each start, is read again from there.
private struct LogText: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.drawsBackground = false
        let text = scroll.documentView as! NSTextView
        text.isEditable = false
        text.isSelectable = true
        text.drawsBackground = false
        text.usesFindBar = true
        text.isIncrementalSearchingEnabled = true
        text.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        text.textColor = .labelColor
        text.textContainerInset = NSSize(width: 10, height: 8)
        context.coordinator.follow(url, in: text)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {}

    static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) {
        coordinator.stop()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor
    final class Coordinator {
        private var source: DispatchSourceFileSystemObject?
        private var handle: FileHandle?
        private var offset: UInt64 = 0
        private weak var text: NSTextView?
        private var url: URL?

        func follow(_ url: URL, in text: NSTextView) {
            self.url = url
            self.text = text
            reopen()
        }

        func stop() {
            source?.cancel()
            source = nil
        }

        /// Opens the file again, as when it was replaced, and reads its end.
        private func reopen() {
            stop()
            text?.string = ""
            guard let url, let handle = try? FileHandle(forReadingFrom: url) else { return }
            self.handle = handle
            let size = (try? handle.seekToEnd()) ?? 0
            offset = size > 512 * 1024 ? size - 512 * 1024 : 0
            read(scrolling: true)
            let source = DispatchSource.makeFileSystemObjectSource(
                fileDescriptor: handle.fileDescriptor, eventMask: [.extend, .write, .delete, .rename], queue: .main)
            source.setEventHandler { [weak self] in
                MainActor.assumeIsolated {
                    guard let self, let source = self.source else { return }
                    if !source.data.isDisjoint(with: [.delete, .rename]) { self.reopen() } else { self.read() }
                }
            }
            source.setCancelHandler { try? handle.close() }
            self.source = source
            source.resume()
        }

        /// Reads what the file has gained, keeping the end in view where it was, or `scrolling` to it.
        private func read(scrolling: Bool = false) {
            guard let handle, let text, let storage = text.textStorage else { return }
            let size = (try? handle.seekToEnd()) ?? 0
            if size < offset {
                // Written again from its start.
                offset = 0
                storage.setAttributedString(NSAttributedString())
            }
            try? handle.seek(toOffset: offset)
            guard let data = try? handle.readToEnd(), !data.isEmpty else { return }
            offset += UInt64(data.count)
            let atEnd = (text.enclosingScrollView?.verticalScroller?.floatValue ?? 1) > 0.98
                || text.visibleRect.maxY >= text.bounds.maxY - 4
            storage.append(NSAttributedString(string: String(decoding: data, as: UTF8.self), attributes: [
                .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                .foregroundColor: NSColor.labelColor,
            ]))
            if atEnd { text.scrollToEndOfDocument(nil) }
            // At first the view has no size yet: to its end once it has been laid out.
            if scrolling { DispatchQueue.main.async { [weak text] in text?.scrollToEndOfDocument(nil) } }
        }
    }
}
