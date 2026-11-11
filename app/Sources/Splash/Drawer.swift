import AppKit
import SwiftTerm
import SwiftUI

/// What the drawer under the window's content holds (`WithDrawer`): terminals, each running the
/// program `shell` names as it starts, in the home folder or a coding agent's project folder, and
/// the logs of what the app runs. Whether it shows, and its height, are the window's
/// (`AppState.showsDrawer`), as the sidebar's are.
@MainActor @Observable
final class Drawer {
    enum Part: Hashable { case terminal, logs }

    /// Asked for each terminal, so that a login shell the user changes (chsh) runs in the next one.
    let shell: () -> Shell
    var part = Part.terminal
    private(set) var terminals: [TerminalSession] = []
    var terminal: TerminalSession.ID?
    var log = LogSource.server
    /// The terminal whose close waits for an answer: it runs a program, which closing ends.
    var closing: TerminalSession.ID?
    /// Called as the last terminal goes, for the window to close the drawer (`AppState`).
    @ObservationIgnored var emptied: () -> Void = {}
    /// The log's text while the drawer shows it, for Find… (`find`).
    @ObservationIgnored weak var logText: NSTextView?

    init(shell: @escaping () -> Shell) {
        self.shell = shell
    }

    func add(_ session: TerminalSession) {
        terminals.append(session)
        terminal = session.id
        part = .terminal
        // A shell that exits cleanly, as `exit` ends it, takes its tab with it, as in Terminal; one
        // that fails stays, dimmed, with what it said.
        session.exitedCleanly = { [weak self, id = session.id] in self?.close(id) }
    }

    /// Closes terminal `id`, or asks first where it runs a program, which closing ends (`closing`).
    func askToClose(_ id: TerminalSession.ID) {
        if terminals.first(where: { $0.id == id })?.runsProgram == true { closing = id } else { close(id) }
    }

    /// Ends terminal `id`'s shell and takes its tab away.
    func close(_ id: TerminalSession.ID) {
        guard let index = terminals.firstIndex(where: { $0.id == id }) else { return }
        terminals.remove(at: index).end()
        if terminal == id { terminal = terminals[safe: min(index, terminals.count - 1)]?.id }
        if terminals.isEmpty { emptied() }
    }

    /// Shows the find bar of what the drawer shows, its terminal or its log, which takes the
    /// keyboard.
    func find() {
        let view: NSView? = part == .terminal ? terminals.first { $0.id == terminal }?.view : logText
        guard let view, let window = view.window, window.makeFirstResponder(view) else { return NSSound.beep() }
        view.showFindBar()
    }

    /// Whether the keyboard is in the drawer, as `window` has it: with a terminal, each of which is
    /// the drawer's, or its log, or the find bar of either; or with nothing, as when the terminal
    /// that had it has closed. The drawer gives the composer the keyboard as it hides only then
    /// (`WithDrawer`).
    func hasKeyboard(in window: NSWindow?) -> Bool {
        guard let responder = window?.firstResponder as? NSView else { return true }
        let log = logText.map { $0.enclosingScrollView ?? $0 }
        return sequence(first: responder, next: \.superview).contains { $0 is TypingTerminalView || $0 === log }
    }

    /// Ends every shell at once, for a quit that goes on to wait for the model or a download to
    /// stop. Any other end of the app hangs them up (`TerminalSession`).
    func endAll() {
        terminals.forEach { $0.end() }
    }

    /// How many terminals run a program besides their shell, which a quit ends (`AppState.QuitCost`).
    var runningPrograms: Int {
        terminals.count { $0.runsProgram }
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

/// The program a terminal runs.
struct Shell {
    let path: String
    let arguments: [String]
    /// The name it runs under, which starts with "-" for a login shell.
    let name: String

    /// The user's login shell, as Terminal starts it, which completes the app's environment from
    /// the user's startup files.
    static var login: Shell {
        let named = getpwuid(getuid())?.pointee.pw_shell.map { String(cString: $0) }
        return login(named, environment: ProcessInfo.processInfo.environment)
    }

    /// The login shell at `path`, the one the user database names; where it names none, as when it
    /// is damaged, the one `environment`'s SHELL names, else zsh, the system's default.
    static func login(_ path: String?, environment: [String: String]) -> Shell {
        let path = path ?? environment["SHELL"] ?? "/bin/zsh"
        return Shell(path: path, arguments: [], name: "-" + (path as NSString).lastPathComponent)
    }
}

/// A shell in a pseudo-terminal, as Terminal opens one, in the home folder or a coding agent's
/// project folder (`AppState.open`), with the app's own environment (`Runtime.userEnvironment`),
/// and Option typing what the keyboard layout gives, as there. Its view outlives the drawer's, so
/// that a terminal keeps what it shows while the drawer is closed or the window shows another part.
@MainActor @Observable
final class TerminalSession: Identifiable {
    let id = UUID()
    /// What the shell calls itself, its title escape, or its folder's name until it does.
    private(set) var title: String
    private(set) var ended = false
    @ObservationIgnored let view: TypingTerminalView
    @ObservationIgnored private var delegate: Delegate?
    /// Called as the shell exits with no error (`Drawer.add`).
    @ObservationIgnored var exitedCleanly: () -> Void = {}

    /// Starts `shell` in `folder`, typing `typing` into it once, as a command a user typed at its
    /// first prompt.
    init(shell: Shell, environment: [String: String], folder: URL = .homeDirectory, typing: String? = nil) {
        title = folder.lastPathComponent
        view = TypingTerminalView(frame: NSRect(x: 0, y: 0, width: 640, height: 280))
        view.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        view.optionAsMetaKey = false
        let delegate = Delegate(self)
        self.delegate = delegate
        view.processDelegate = delegate
        var environment = environment
        environment["TERM"] = "xterm-256color"
        environment["COLORTERM"] = "truecolor"
        environment["TERM_PROGRAM"] = "Splash"
        if environment["LANG"] == nil { environment["LANG"] = "en_US.UTF-8" }
        view.startProcess(executable: shell.path, args: shell.arguments, environment: environment.map { "\($0.key)=\($0.value)" },
                          execName: shell.name, currentDirectory: folder.path)
        // Only the app holds the terminal's master, not the shells started after this one, nor what
        // they leave running: the app's end hangs this shell up, however it ends.
        _ = fcntl(view.process.childfd, F_SETFD, FD_CLOEXEC)
        if let typing { view.type(typing) }
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

    /// Whether a program other than the shell has the terminal, as a coding agent has it while it
    /// runs: the terminal's foreground process group is not the shell's, which a shell at its
    /// prompt holds itself.
    var runsProgram: Bool {
        guard !ended, view.process.shellPid != 0 else { return false }
        let group = tcgetpgrp(view.process.childfd)
        return group > 0 && group != view.process.shellPid
    }

    /// Hangs the shell up, as closing a terminal window does, which ends it and its jobs, and
    /// waits for it off the main thread, as SwiftTerm stops waiting once told to stop.
    func end() {
        guard !ended else { return }
        ended = true
        let shell = view.process.shellPid
        view.terminate()
        // None where it failed to start.
        guard shell != 0 else { return }
        kill(shell, SIGHUP)
        DispatchQueue.global(qos: .utility).async { waitpid(shell, nil, 0) }
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

        /// `exitCode` is the shell's status as `waitpid` gives it: 0 where it exited with no error.
        func processTerminated(source: TerminalView, exitCode: Int32?) {
            session?.ended = true
            if exitCode == 0 { session?.exitedCleanly() }
        }
    }
}

/// A terminal view that types a line into its shell as the shell first reads one: once its line
/// editor takes the terminal (no canonical input), as zsh, bash and fish do to show their prompt,
/// so that the line shows once, at the prompt, rather than echoed ahead of what the shell prints
/// as it starts too; or after 5 s, for a shell without one, where a shell whose start outlasts
/// them only has the line echoed ahead of what it prints.
final class TypingTerminalView: LocalProcessTerminalView {
    private var typing: String?
    /// Whether it takes the keyboard as it moves into a window, once.
    var takesKeyboard = false

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard takesKeyboard, window != nil else { return }
        takesKeyboard = false
        // Once the window has set itself up, which gives the keyboard to its content as it opens.
        DispatchQueue.main.async { [weak self] in
            guard let self, let window else { return }
            window.makeFirstResponder(self)
        }
    }

    func type(_ line: String) {
        typing = line + "\n"
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in self?.typeNow() }
    }

    override func dataReceived(slice: ArraySlice<UInt8>) {
        super.dataReceived(slice: slice)
        guard typing != nil else { return }
        var settings = termios()
        if tcgetattr(process.childfd, &settings) == 0, settings.c_lflag & tcflag_t(ICANON) == 0 { typeNow() }
    }

    private func typeNow() {
        guard let typing else { return }
        self.typing = nil
        send(txt: typing)
    }
}

/// A log the drawer shows: the server's and the downloads', which each download begins again.
enum LogSource: String, CaseIterable {
    case server = "Server", downloads = "Downloads"

    @MainActor func url(in state: AppState) -> URL { self == .server ? state.server.log : state.installer.log }
}

/// The content with the drawer under it, its edge dragged to resize it.
struct WithDrawer<Content: View>: View {
    @Environment(AppState.self) private var state
    @FocusState private var focused: Bool
    @ViewBuilder let content: Content
    /// The room the content and the drawer share.
    @State private var room: CGFloat = 0
    /// Its AppKit view, for the window whose keyboard it gives back as it hides.
    @State private var anchor = Anchor()

    /// The tallest the drawer is: 640 points, leaving the content 240 at least, once the room is
    /// measured.
    private var tallest: CGFloat { room == 0 ? 640 : max(120, min(640, room - 240)) }

    var body: some View {
        let shown = state.showsDrawer
        let height = min(state.drawerHeight, tallest)
        VStack(spacing: 0) {
            // Nothing in the content or the drawer animates with the slide (`Slide`).
            content
                .transaction(value: shown) { $0.animation = nil }
            // Closed, it keeps its place and its height, clipped to none, its terminals and their
            // views with it; the keyboard leaves it. Its tabs keep to its top edge, with no
            // terminal under them too.
            DrawerView(tallest: tallest)
                .frame(height: height, alignment: .top)
                .focused($focused)
                .transaction(value: shown) { $0.animation = nil }
                .modifier(Slide(shown: shown ? 1 : 0, height: height))
                .allowsHitTesting(shown)
                .accessibilityHidden(!shown)
        }
        // It slides as it shows and hides, the content giving it room. The change is animated here
        // rather than where it is made: there it came in one turn after changes made without an
        // animation, as a closing terminal's, which SwiftUI then updated the window for at once,
        // reading the drawer as it was, and missed its closing.
        .animation(Motion.settle, value: shown)
        // The keyboard leaves it only where it is in it, for the composer: a terminal that exits
        // while chat search or a rename field has it leaves it there.
        .onChange(of: shown) {
            if !shown && state.drawer.hasKeyboard(in: anchor.view?.window) {
                focused = false
                state.returnKeyboard()
            }
        }
        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { room = $0 }
        .background(ViewAnchor { anchor.view = $0 })
    }

    /// A view's AppKit view, held by reference: taking it changes no state the view reads.
    private final class Anchor {
        weak var view: NSView?
    }
}

/// The drawer's height as it slides, `shown` of `height`: one value SwiftUI animates, laying
/// the window out again at each frame, so that what is above the drawer and in it moves with
/// its edge whatever animates in it. Animating the views moved by the change instead moved each
/// as it would animate: the Server's control bar, whose words change at once, jumped to where
/// the drawer was going, over the screen, ahead of its own background and the drawer, and the
/// tab of the terminal the drawer opened with faded in under the New Terminal button sliding
/// past it.
private struct Slide: ViewModifier, @preconcurrency Animatable {
    var shown: CGFloat
    let height: CGFloat

    var animatableData: CGFloat {
        get { shown }
        set { shown = newValue }
    }

    func body(content: Content) -> some View {
        content
            // In whole points, so that the content's bottom and the drawer's edge, each rounded to
            // the screen's pixels on its own, keep together through the slide.
            .frame(height: (height * shown).rounded(), alignment: .top)
            .clipped()
    }
}

private struct DrawerView: View {
    @Environment(AppState.self) private var state
    /// The tallest the drawer is in the window it is in.
    let tallest: CGFloat
    @State private var dragged: CGFloat?
    /// The terminal with the keyboard: the one shown, picked in the drawer shown; none otherwise.
    /// One value for them all moves the keyboard from the terminal hidden to the one shown in one
    /// step; each giving it up on its own would leave it to the window on macOS 26.5.
    @FocusState private var focused: TerminalSession.ID?

    var body: some View {
        let drawer = state.drawer
        let shown = state.showsDrawer && drawer.part == .terminal ? drawer.terminal : nil
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
                                let start = dragged ?? min(state.drawerHeight, tallest)
                                if dragged == nil { dragged = start }
                                state.drawerHeight = min(max(start - value.translation.height, 120), tallest)
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
                        NSWorkspace.shared.activateFileViewerSelecting([drawer.log.url(in: state)])
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.hover)
                    .help("Show in Finder")
                    .accessibilityLabel("Show in Finder")
                }
                Button {
                    state.toggleDrawer()
                } label: {
                    Image(systemName: "chevron.down")
                }
                .buttonStyle(.hover)
                .help("Hide (⌘J)")
                .accessibilityLabel("Hide Terminal and Logs")
            }
            .padding(.horizontal, 12)
            .frame(height: 36)
            Divider()
            ZStack {
                // Every terminal stays in the window, the one picked on top, so that each keeps
                // its view and what it shows.
                ForEach(drawer.terminals) { session in
                    let picked = drawer.part == .terminal && drawer.terminal == session.id
                    TerminalPane(session: session, focused: $focused, shown: shown == session.id)
                        .opacity(picked ? 1 : 0)
                        .allowsHitTesting(picked)
                        .accessibilityHidden(!picked)
                }
                // Only while it shows: hidden, it would follow its file and lay its text out
                // again at each change of the window's width. Shown, it has the keyboard.
                if state.showsDrawer && drawer.part == .logs {
                    LogText(url: drawer.log.url(in: state)) { drawer.logText = $0 }
                        .id(drawer.log)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(.background)
        .onChange(of: shown, initial: true) { focused = shown }
    }
}

/// The terminals' tabs, each with its title and a close button on hover, then New Terminal.
private struct TerminalTabs: View {
    @Environment(AppState.self) private var state

    /// The New Terminal button's place in the strip.
    private static let end = "New Terminal"

    var body: some View {
        let drawer = state.drawer
        ScrollViewReader { proxy in
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
                            Button { drawer.askToClose(session.id) } label: {
                                Image(systemName: "xmark").font(.system(size: 9, weight: .semibold))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .help("Close Terminal")
                            .accessibilityLabel("Close Terminal")
                        }
                        .font(.callout)
                        .padding(.horizontal, 10)
                        .frame(height: 26)
                        .background(picked ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.clear), in: .capsule)
                        .id(session.id)
                    }
                    Button { state.newTerminal() } label: {
                        Image(systemName: "plus")
                    }
                    .buttonStyle(.hover)
                    .help("New Terminal")
                    .accessibilityLabel("New Terminal")
                    .id(Self.end)
                }
            }
            // The tab picked shows, past the tabs that fill the drawer's width, as one added does,
            // the last, with New Terminal after it: they stayed out of view at the strip's end.
            .onChange(of: drawer.terminal, initial: true) { _, id in
                guard let id else { return }
                proxy.scrollTo(id == drawer.terminals.last?.id ? AnyHashable(Self.end) : AnyHashable(id))
            }
        }
        .confirmationDialog("Close this terminal?", isPresented: Binding(
            get: { drawer.closing != nil }, set: { if !$0 { drawer.closing = nil } })) {
            Button("Close", role: .destructive) {
                if let id = drawer.closing { drawer.close(id) }
            }
        } message: {
            Text("A program is running in it. Closing the terminal ends the program.")
        }
        Spacer(minLength: 0)
    }
}

/// A terminal's view, moved into the drawer each time it is laid out there, in the window's
/// appearance, with a margin of its background around it. It has the keyboard while `focused` is
/// its terminal.
private struct TerminalPane: View {
    let session: TerminalSession
    let focused: FocusState<TerminalSession.ID?>.Binding
    /// Whether it is the terminal the drawer shows.
    let shown: Bool

    var body: some View {
        Hosted(session: session, shown: shown)
            .focused(focused, equals: session.id)
            .padding(EdgeInsets(top: 6, leading: 10, bottom: 4, trailing: 4))
            .background(Color(nsColor: .textBackgroundColor))
    }

    private struct Hosted: NSViewRepresentable {
        let session: TerminalSession
        let shown: Bool

        func makeNSView(context: Context) -> NSView {
            let container = NSView()
            session.view.removeFromSuperview()
            session.view.frame = container.bounds
            session.view.autoresizingMask = [.width, .height]
            // Shown as its window opens, as ⌘J opens one, it takes the keyboard once in it: the
            // drawer's focus can't give it before.
            session.view.takesKeyboard = shown
            container.addSubview(session.view)
            return container
        }

        func updateNSView(_ container: NSView, context: Context) {
            session.paint(context.environment.colorScheme)
        }
    }
}

/// A log file, read only, as it grows: the lines that start in its last `limit` bytes at first,
/// then what is written to it, the line being written as far as it is, the oldest lines going an
/// eighth of `limit` at a time past it; ⌘F finds in it. It keeps its place as it changes (`LogScrollView`). A file written
/// again from its start, as the downloads' log is at each download, is read again from there.
struct LogText: NSViewRepresentable {
    let url: URL
    /// Given its text view as it shows, which then takes the keyboard.
    var shows: (NSTextView) -> Void = { _ in }

    /// How much of a log it holds: 512 KiB of its end at first, then as many UTF-16 units, which
    /// are never more than the bytes they decode from.
    static let limit = 512 * 1024

    func makeNSView(context: Context) -> LogScrollView {
        let scroll = LogScrollView()
        context.coordinator.follow(url, in: scroll)
        shows(scroll.text)
        DispatchQueue.main.async { scroll.window?.makeFirstResponder(scroll.text) }
        return scroll
    }

    func updateNSView(_ scroll: LogScrollView, context: Context) {}

    static func dismantleNSView(_ scroll: LogScrollView, coordinator: Coordinator) {
        coordinator.stop()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor
    final class Coordinator {
        private var source: DispatchSourceFileSystemObject?
        private var handle: FileHandle?
        private var offset: UInt64 = 0
        private var lines = LogLines()
        /// The line being written, as the view shows it at its end.
        private var unfinished = Data()
        private weak var scroll: LogScrollView?
        private var url: URL?

        func follow(_ url: URL, in scroll: LogScrollView) {
            self.url = url
            self.scroll = scroll
            reopen()
        }

        func stop() {
            source?.cancel()
            source = nil
        }

        /// Opens the file again, as when it was replaced, and reads its end. One not written yet is
        /// made, empty, with its folder, and followed from its start.
        private func reopen() {
            stop()
            clear()
            guard let url else { return }
            try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let descriptor = open(url.path, O_RDONLY | O_CREAT | O_CLOEXEC, 0o644)
            guard descriptor >= 0 else { return }
            let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            self.handle = handle
            let size = (try? handle.seekToEnd()) ?? 0
            // The lines that start in its last `limit` bytes: read from the byte before those, a
            // newline where one starts with them.
            let tail = UInt64(LogText.limit)
            offset = size > tail ? size - tail - 1 : 0
            lines = LogLines(inside: size > tail)
            read()
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

        private func clear() {
            lines = LogLines()
            unfinished = Data()
            scroll?.text.string = ""
        }

        /// Shows what the file has gained: the lines it ends in place of the line shown as being
        /// written, then the line now being written.
        private func read() {
            guard let handle, let scroll, let storage = scroll.text.textStorage else { return }
            let size = (try? handle.seekToEnd()) ?? 0
            if size < offset {
                // Written again from its start.
                offset = 0
                clear()
            }
            try? handle.seek(toOffset: offset)
            guard let data = try? handle.readToEnd(), !data.isEmpty else { return }
            offset += UInt64(data.count)
            let ended = lines.take(data)
            let shown = unfinished
            unfinished = lines.unfinished
            guard !ended.isEmpty || unfinished != shown else { return }
            let length = String(decoding: shown, as: UTF8.self).utf16.count
            let added = String(decoding: ended + unfinished, as: UTF8.self)
            let unfinishedLength = String(decoding: unfinished, as: UTF8.self).utf16.count
            scroll.keepingPlace {
                storage.replaceCharacters(in: NSRange(location: storage.length - length, length: length), with: NSAttributedString(
                    string: added, attributes: [.font: LogScrollView.font, .foregroundColor: NSColor.labelColor]))
                // Past the limit, the lines that start in its last seven eighths stay: each removal
                // lays the whole text out again, so the oldest lines go many at a time. All the
                // whole lines go where the line being written is longer.
                guard storage.length > LogText.limit else { return 0 }
                let excess = storage.length - LogText.limit / 8 * 7
                let newline = storage.mutableString.range(
                    of: "\n", options: [], range: NSRange(location: excess - 1, length: storage.length - excess + 1))
                let removed = newline.location == NSNotFound ? storage.length - unfinishedLength : NSMaxRange(newline)
                storage.deleteCharacters(in: NSRange(location: 0, length: removed))
                return removed
            }
        }
    }
}

/// A log's text, which keeps its place as the text and its own size change: on the end while
/// the end is in view, else on the line at the top of the view, as far above it as it was. Its
/// layout is exact, so that each place is where the text is.
final class LogScrollView: NSScrollView {
    static let font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)

    /// Laid out by a layout manager, which places every line where it is, rather than by
    /// estimates of where the lines not yet shown are, which move the view as they come true.
    let text = NSTextView(usingTextLayoutManager: false)

    init() {
        super.init(frame: .zero)
        hasVerticalScroller = true
        drawsBackground = false
        text.minSize = .zero
        text.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
        text.isVerticallyResizable = true
        text.autoresizingMask = .width
        text.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        text.textContainer?.widthTracksTextView = true
        text.isEditable = false
        text.isSelectable = true
        text.drawsBackground = false
        text.usesFindBar = true
        text.isIncrementalSearchingEnabled = true
        text.font = Self.font
        text.textColor = .labelColor
        text.textContainerInset = NSSize(width: 10, height: 8)
        documentView = text
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    /// A new size, as when the view is first shown, the drawer is resized or the find bar shows,
    /// keeps the view's place too.
    override func tile() {
        keepingPlace {
            super.tile()
            return 0
        }
    }

    /// Makes `change`, which gives how many characters it removed from the start of the text,
    /// keeping the view's place.
    func keepingPlace(_ change: () -> Int) {
        guard let layout = text.layoutManager, let container = text.textContainer else { _ = change(); return }
        let place = place(in: layout, container)
        let removed = change()
        layout.ensureLayout(for: container)
        text.sizeToFit()
        let y: CGFloat
        switch place {
        case .end:
            y = text.bounds.maxY - contentView.bounds.height
        case let .line(character, offset):
            let character = character - removed
            // The line at the top went with the oldest lines: the text's start is there now.
            guard character >= 0, layout.numberOfGlyphs > 0 else { y = 0; break }
            let line = layout.lineFragmentRect(
                forGlyphAt: min(layout.glyphIndexForCharacter(at: character), layout.numberOfGlyphs - 1), effectiveRange: nil)
            y = line.minY + text.textContainerOrigin.y + offset
        }
        let bounds = contentView.constrainBoundsRect(NSRect(origin: NSPoint(x: contentView.bounds.minX, y: y), size: contentView.bounds.size))
        contentView.scroll(to: bounds.origin)
        reflectScrolledClipView(contentView)
    }

    private enum Place {
        case end
        /// The line starting at `character`, `offset` points above the top of the view.
        case line(Int, CGFloat)
    }

    /// Where the view is: on the end while the end is in view, as before it was ever shown.
    private func place(in layout: NSLayoutManager, _ container: NSTextContainer) -> Place {
        let visible = text.visibleRect
        guard !visible.isEmpty, visible.maxY < text.bounds.maxY - 1, layout.numberOfGlyphs > 0 else { return .end }
        var range = NSRange()
        let glyph = layout.glyphIndex(for: NSPoint(x: 0, y: visible.minY - text.textContainerOrigin.y), in: container)
        let line = layout.lineFragmentRect(forGlyphAt: glyph, effectiveRange: &range)
        return .line(layout.characterIndexForGlyph(at: range.location), visible.minY - text.textContainerOrigin.y - line.minY)
    }
}
